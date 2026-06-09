# Faz 3 — Ön İşleme & Temizleme (Fold-Safe Preprocessing)

## Amaç
Ham `train.csv` / `test_x.csv` kolonlarını, **tüm `fit` işlemleri yalnızca dış fold'un train parçasında yapılan**, deterministik ve sızıntısız bir `fit/transform` boru hattına dönüştürmek; eksik değerleri MNAR semantiğine uygun (kanıtlanmış count==0 çakışmasıyla) doldurmak, tipleri/kategorikleri modele hazır hâle getirmek ve tahmin sonrası `clip[0,100]` sözleşmesini kurumsallaştırmak.

## "0 Overfit" Rolü
Bu faz, "0 Overfit" north-star'ının **mekanik kalbidir**: tek başına model seçmez ama her modelin CV'sinin güvenilirliğini belirler. Eğer bir tek istatistik (medyan, encoding ortalaması, scaler μ/σ) tüm train üzerinde hesaplanırsa, OOF tahmini valid satırını dolaylı olarak "görmüş" olur — CV iyimser çıkar, private %40 bölmesinde çöker. Bu fazın tek işi, faz 02'de kilitlenen `data/folds.parquet` şemasıyla **satır-hizalı**, fold-içi fit edilen bir transformer üretip CV ile private leaderboard arasındaki sapmayı sıfıra yaklaştırmaktır. Ayrıca yıl kolonlarının (`application_year`, `graduation_year`) ham/türetilmiş hâlde matrise sızmasını burada fiziksel olarak engelleyerek tek bilinen dağılım kaymasını öldürür.

## Girdiler / Çıktı Artefaktları
**Girdiler (önceki fazlardan):**
- `data/train.csv` (10.000 × 47), `data/test_x.csv` (10.000 × 46) — ham veri (Faz 01 EDA portresi ile teyitli).
- `data/folds.parquet` (`student_id`, `repeat`, `fold`) — Faz 02'de üretilen master fold tablosu (5-fold × 3-repeat, stratify: `==100` ayrı bin + qcut9). **Bu fazda asla yeniden üretilmez, sadece okunur.**
- Faz 01 EDA raporu: NA oranları, dtype'lar, kategorik kardinaliteler, adversarial AUC bulguları.

**Çıktılar (sonraki fazlara):**
- `src/preprocessing.py` — `build_preprocessor()` fonksiyonu: fold-içi `fit_transform` / `transform` sözleşmesini sağlayan sklearn `Pipeline` + `ColumnTransformer` (veya eşdeğer fonksiyonel transformer). Faz 04 (FE) ve Faz 06 (modelleme) bunu doğrudan import eder.
- `src/cleaning.py` — `clean_raw(df)`: tip düzeltmeleri, kolon drop listesi (`student_id`, yıl kolonları), kategorik dtype atama. **İstatistik içermez** (fold-bağımsız, sızıntısız saf dönüşüm).
- `src/postprocess.py` — `clip_predictions(pred)`: `np.clip(pred, 0, 100)` + clip-dışı değer görürse `assert` ile hata fırlatan submission koruyucu.
- Kolon manifesti: `data/column_spec.json` — hangi kolon sayısal / kategorik / drop / NA-flag, ve her NA kolonu için doldurma stratejisi (`zero_flag` vs `median_flag`); deterministik liste (hardcode değil, türetilmiş ama versiyonlanmış).
- Sözleşme: her base model bu transformer'ın çıktısını alır; `fit` çağrıları **yalnızca** dış-fold train indeksinde.

## Detaylı Adımlar

### 1. Yükleme & encoding (sızıntısız, fold-bağımsız)
- `pd.read_csv(..., encoding='utf-8')`. Dosya temiz UTF-8 (Faz 01 ham byte teyidi: `ö`=`\xc3\xb6`). **ftfy / latin1 mojibake fix YAPILMAZ** — veriyi bozar (leakageRules: TURKCE LOWERCASE/ENCODING tuzağı).
- BOM'a dikkat: ilk kolon header'ı `﻿` (BOM) ile gelebilir (ham byte'ta teyitli) → `df.columns = df.columns.str.replace('﻿','')` ile temizle, aksi halde `student_id` adı eşleşmez.

### 2. Kolon drop & tip düzeltmeleri (`clean_raw`, istatistiksiz)
- **DROP — `student_id`**: STU_xxxxxx sentetik anahtar, non-predictive, sıra-ezberi riski (leakageRules: ID/SATIR-SIRASI). Submission için ayrı saklanır, matrise girmez.
- **DROP — `application_year`, `graduation_year`**: ham yıl YASAK (adversarial AUC yıllarla 0.664, yıllarsız 0.491). lockedDecisions: yıl kolonları → tek dağılım kaymasını öldürmek için fiziksel drop. (`years_since_graduation` gibi shift-invariant türev YALNIZCA Faz 04'te denenebilir ve adversarial AUC ~0.5 teyidi gerektirir; bu fazda üretilmez.)
- **Kategorik dtype**: `department` (7 seviye), `university_tier` (4), `target_role` (11), `hobby` (8), `preferred_social_media_platform` (6) → `astype('category')`. `university_tier` ordinal görünse de doğal sıra belirsiz; nominal kategorik bırak (LGBM/CatBoost/HistGBR split'leri keşfetsin).
- **Sayısal**: kalan ~38 sayısal kolon `float64` (sayımlar int → float, NaN tutabilmek için). `mentor_feedback_text` → Faz 05'e devredilir, bu fazda matrise girmez (yalnızca pas-through, ayrı saklanır).

### 3. Eksik değer doldurma — MNAR-duyarlı (kanıta dayalı), fold-içi fit
NA'li 7 kolon (Faz 01 + bu fazda gerçek veriden teyitli sayımlar ve doldurma kararları):

| Kolon | NA sayısı | NA % | Strateji | Gerekçe (gerçek veriden ölçülü) |
|---|---|---|---|---|
| `internship_duration_months` | 1657 | 16.6 | **0 ile doldur + `_missing` flag** (MNAR `zero_flag`) | NA'lerin **%82.14**'ü `internship_count==0` ile çakışır → "staj yok = süre yok" gerçek MNAR sıfırı |
| `english_exam_score` | 953 | 9.5 | fold-içi **medyan** + `_missing` flag | sürekli sınav skoru; eksiklik veri-toplama boşluğu |
| `github_avg_stars` | 910 | 9.1 | fold-içi **medyan** + `_missing` flag (log1p Faz 04'te) | aşağıdaki çift kolonun yarısı; eksiklik "0 yıldız" değil |
| `open_source_contribution_count` | 910 | 9.1 | fold-içi **medyan** + `_missing` flag (DÜZELTİLDİ) | `github_avg_stars` ile **tam aynı 910 satırda** NA; bu satırların yalnızca **%3.19**'unda `github_repo_count==0`, NA-satır repo medyanı **4** → aktif GitHub kullanıcısı, "katkı yok" DEĞİL |
| `hr_interview_score` | 780 | 7.8 | fold-içi **medyan** + `_missing` flag | sürekli mülakat skoru |
| `linkedin_profile_score` | 668 | 6.7 | fold-içi **medyan** + `_missing` flag | sürekli profil skoru |
| `portfolio_score` | 364 | 3.6 | fold-içi **medyan** + `_missing` flag | sürekli profil skoru |

- **MNAR `0+flag` YALNIZCA `internship_duration_months` için geçerli**: NA'lerin **%82.14**'ü `internship_count==0` ile çakışır (bu fazda gerçek train.csv'den ölçüldü). Staj yoksa süre yoktur → `0 + flag` doğru semantik; medyan impute "sahte orta staj" uydurur ve sinyali bozar.
- **`open_source_contribution_count` MNAR DEĞİL → medyan + flag (kritik düzeltme)**: Bu kolon `github_avg_stars` ile **bayt-bayt aynı 910 satırda** NA'dır (üç kez teyit: `(osc.isna()==gas.isna()).all()==True`, çakışma 910/910). Bu satırların yalnızca **%3.19**'unda `github_repo_count==0`'dır; NA-satırlarının `github_repo_count` medyanı **4** (genel medyan 6). Yani bu öğrenciler **aktif GitHub'a sahip** — eksiklik "katkı yok" değil, `github_avg_stars` ile birlikte çekilmemiş bir **veri-toplama boşluğudur**. Bu satırlara `0` enjekte etmek **sahte "0 katkı" sinyali** üretir ve aynı boşluğa sahip `github_avg_stars`'ı medyan-impute ederken `open_source_contribution_count`'u `0`'a çekmek **iç tutarsızlıktır**. Bu nedenle `open_source_contribution_count`, `internship_duration_months` gibi değil, `github_avg_stars` gibi ele alınır: **fold-içi medyan + `_missing` flag** (`median_flag`).
- **Skor + count-but-not-MNAR kolonları (medyan)**: süreklilik gösteren sınav/mülakat/profil skorları ve veri-toplama boşluğu olan `github_avg_stars` / `open_source_contribution_count` için fold-içi **medyan** (ortalama değil — çarpık olabilir, medyan dayanıklı). Her biri için `_missing` boolean flag eklenir (eksikliğin kendisi sinyal taşıyabilir; özellikle bu çift kolonun ortak eksikliği "GitHub verisi çekilmemiş" sinyalidir).
- **KRİTİK fold-içi kural**: tüm medyan değerleri `sklearn.impute.SimpleImputer(strategy='median')` ile **yalnızca dış-fold train'inden** fit edilir, valid/test'e `transform` edilir. `zero_flag` kolonu (`internship_duration_months`) `SimpleImputer(strategy='constant', fill_value=0)` ile fold-bağımsız doldurulur (sabit 0, istatistik yok). Flag'ler fold-bağımsız (sadece `isna()` — hedefe bakmaz), ama medyan doldurma değeri fold-içi (leakageRules: IMPUTATION LEAK). Test missingness train ile neredeyse özdeş ama OOF saflık kuralı yine de geçerli.

### 4. Outlier ele alma (MSE-duyarlı, muhafazakâr)
- **Felsefe**: MSE büyük hataları karesel cezalar → uç **tahminlerden** kaçınmak hayati; ama **girdi** outlier'larını agresif kırpmak sinyal öldürür. GBDT'ler split-tabanlı olduğu için girdi outlier'larına büyük ölçüde dayanıklı → **ham girdileri kırpma YAPMA**.
- **Tek istisna — ağır kuyruklu count'lar**: `github_avg_stars`, `open_source_contribution_count`, `github_repo_count`, `hackathon_awards`, `freelance_project_count` gibi log-normal dağılımlı kolonlar Faz 04'te `log1p` ile sıkıştırılır (bu fazda yalnızca flag/impute; dönüşüm FE'de). Winsorize/clip ham girdide **uygulanmaz** (CV ile kanıtlanmadıkça).
- **Asıl outlier savunması çıktıda**: tahminler `np.clip(0,100)` ile sınırlanır (Adım 6). Girdi tarafında savunma agresif değil, çünkü ağaç modelleri zaten dayanıklı ve kırpma genelleme riski taşır (0.25*cv_std kabul kapısından geçmedikçe eklenmez).

### 5. Kategorik encoding stratejisi (ablation ile seçilir, fold-safe)
Üç model ailesi farklı yol kullanır:
- **CatBoost**: kategorikler **native** verilir (`cat_features`), ordered target statistics CatBoost'un kendi sızıntısız mekanizmasıyla → bizim encode etmemize gerek yok. Bağımsız bias kaynağı (ensemble çeşitliliği).
- **LightGBM**: iki aday, **Faz 04/06'da ablation** ile seçilir:
  - (a) **One-Hot Encoding** — kardinalite düşük (4–11), toplam ~36 kolon, sızıntısız ve basit. Varsayılan kabul edilir (Occam). `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - (b) **Fold-safe OOF target encoding** — `category_encoders` veya el-yapımı, **Bayesian smoothing** (m≈20–50), **yalnızca dış-fold train'inden** hesaplanır. GLOBAL encoding **KESİNLİKLE YASAK** (leakageRules: TARGET-ENCODING LEAK). Sadece OHE'yi 0.25*cv_std kabul kapısından geçerse benimsenir.
  - Not: LGBM'in dahili `categorical_feature` desteği de bir seçenek; ancak OHE/TE ablation'ı kontrollü kıyas için tercih edilir.
- **HistGradientBoostingRegressor**: `categorical_features` parametresi ile native kategorik desteği (sklearn ≥1.0). Kategorik kolonlar pandas `category` dtype ile verilir, encode edilmez.
- Tüm encode/transform işlemleri `ColumnTransformer` içinde fold-içi fit edilir.

### 6. Tahmin sonrası işleme — `clip[0,100]` sözleşmesi (`postprocess.py`)
- `clip_predictions(pred) = np.clip(pred, 0.0, 100.0)`. Gerçek hedef kesin `[0,100]` (min=0.00, max=100.00). Log/logit dönüşümü **YOK** (skew ~-0.45, neredeyse normal; çift-sınırlı yığınla logit kötü çalışır).
- **Submission koruyucu**: yazıcı clip uygulamadan önce ham tahminde `[0,100]` dışı değer görürse log'lar; clip sonrası `assert pred.min()>=0 and pred.max()<=100`. `sample_submission`'daki 123.94 yalnızca format örneğidir, hedef sınırı değil.
- İki-aşamalı (P(y=100) classifier + regressor) **varsayılan DEĞİL** — sadece OOF-MSE'yi net iyileştirirse; varsayılan saf clip (clever-but-fragile yaklaşımlar CV-LB gap açar).

### 7. Pipeline montajı (deterministik, reproducible)
- `build_preprocessor(model_family)` → `ColumnTransformer`:
  - **medyan-impute branch**: `english_exam_score`, `github_avg_stars`, **`open_source_contribution_count`** (DÜZELTİLDİ — artık burada), `hr_interview_score`, `linkedin_profile_score`, `portfolio_score` → `SimpleImputer(strategy='median')` (fold-içi fit). FE kompozitleri Faz 04'te eklenir.
  - **MNAR (`zero_flag`) branch**: yalnızca `internship_duration_months` → `SimpleImputer(strategy='constant', fill_value=0)`. (`open_source_contribution_count` BU BRANCH'TEN ÇIKARILDI; %82 count==0 kanıtı yalnızca internship için var.)
  - **missing-flag branch**: 7 NA kolonu için `MissingIndicator` (veya manuel `isna()`), fold-bağımsız (hedefe bakmaz). 7 flag: her NA kolonu için tam olarak bir tane.
  - **kategorik branch**: `model_family`'ye göre passthrough (CatBoost/HistGBR native) **veya** OHE/TE (LGBM).
- `SEED=42` her yerde; transformer deterministik (kolon sıralaması sabit, `category` kategori sırası sabit). Çıktı kolon adları `get_feature_names_out()` ile sabitlenir, `data/column_spec.json`'a yazılır → satır VE kolon hizası garanti.

## Kararlar & Gerekçeler
- **Yıl kolonlarını drop > shift-invariant türev**: Türev (`years_since_graduation`) cazip ama bu fazda matristen tamamen çıkarmak en güvenli zemin; türev denemesi Faz 04'e ait ve adversarial kapısı (AUC ~0.5) gerektirir. Erken sızıntı riskini sıfırlar.
- **MNAR `0+flag` SADECE `internship_duration_months` için (kanıta dayalı ayrım)**: Doldurma stratejisi kolonun *neden* eksik olduğuna göre seçilir, kolonun "count" olmasına göre değil. `internship_duration_months` NA'lerinin **%82.14**'ü `internship_count==0` ile çakışır → gerçek yapısal sıfır. `open_source_contribution_count` ise `github_avg_stars` ile **aynı 910 satırda** eksiktir ve bu satırların **%96.81**'i aktif repo'ya sahiptir (NA-repo medyanı 4) → veri-toplama boşluğu, yapısal sıfır değil. Bir count kolonunu körü körüne `0`'a doldurmak yanlış MNAR varsayımıdır; her kolon için count==0 çakışması ayrı ölçülür.
- **`open_source_contribution_count`: medyan+flag > 0+flag (düzeltme)**: `github_avg_stars` ile bayt-bayt aynı eksiklik maskesine sahip iki kolonu farklı doldurmak (`0` vs medyan) tutarsızlık ve sahte sinyal yaratır. İkisi de aynı veri-toplama boşluğunu paylaştığı için ikisi de fold-içi medyan + `_missing` flag alır; `_missing` flag zaten "GitHub verisi çekilmemiş" bilgisini taşır.
- **Skor kolonlarında medyan > ortalama > model-tabanlı (KNN/IterativeImputer)**: medyan çarpıklığa dayanıklı ve deterministik; KNN/Iterative fold-içi fit'i ağırlaştırır, reproducibility ve overfit riski getirir, marjinal kazanç (0.25*cv_std kapısını geçmesi şüpheli) → elenir.
- **Girdi outlier kırpma YOK > winsorize**: GBDT split-tabanlı, outlier'a dayanıklı; kırpma genelleme riski + CV-LB gap açma riski. Savunma çıktı clip'inde.
- **OHE varsayılan > TE (LGBM)**: düşük kardinalite (≤11) OHE'yi basit ve sızıntısız kılar; TE doğru yapılsa bile fold-içi OOF + smoothing karmaşıklığı taşır, global yapılırsa hedef sızar. Occam: eşitlikte basit kazanır.
- **Saf clip > log/logit > iki-aşama**: hedef çift-sınırlı ve neredeyse normal; clip bedava/nötr kazanç, diğerleri fragile ve CV-LB gap açar.
- **`category` dtype > label-encode int**: HistGBR/LGBM native kategorik desteği için pandas `category` doğru taşıma; label-encode sahte ordinal sıra yaratır.

## Leakage / Overfit Guardrail'ları
1. **FOLD-İÇİ FİT MUTLAK KURAL**: `SimpleImputer(median)`, OHE kategori öğrenme, target encoding — **hepsi yalnızca dış-fold train'inden `fit`**, valid/test'e `transform`. Hiçbir istatistik tüm train veya train+test birleşiminden hesaplanmaz. (`internship_duration_months` constant=0 istatistik içermez, fold-bağımsız.) (leakageRules madde 1.)
2. **TARGET-ENCODING LEAK**: global mean encoding YASAK; yalnızca fold-içi OOF + Bayesian smoothing. (leakageRules madde 2.)
3. **IMPUTATION LEAK**: medyan doldurma değeri fold-içi fit; `_missing` flag'ler fold-bağımsız (sadece `isna()`, hedefe bakmaz). MNAR varsayımı (count==0 çakışması) **hedefe değil, başka bir feature'a** bakarak doğrulandığından target leakage değildir. (leakageRules madde 4.)
4. **YIL KOLONU LEAK**: `application_year`/`graduation_year` ham veya yıl-bazlı agregasyon YASAK; bu fazda fiziksel drop. (leakageRules madde 5.)
5. **ID/SATIR-SIRASI LEAK**: `student_id` matrise asla girmez; sentetik üretim sırası ezberi engellenir. (leakageRules madde 7.)
6. **ADVERSARIAL SİGORTA**: bu fazın çıktı matrisinde (yıl drop sonrası) train(0)/test(1) sınıflandırıcı AUC ~0.5 olmalı; AUC>0.6 ise suçlu feature (özellikle gizli yıl türevi) incelenir. (cvProtocol: ADVERSARIAL SİGORTA.)
7. **YANLIŞ-IMPUTE SİNYAL LEAK (yeni)**: bir kolona dağılımına uymayan sabit değer (`open_source_contribution_count`'a `0`) enjekte etmek, OOF'ta görünmeyen ama private'ta farklılaşabilecek sahte bir sinyal yaratır. Doldurma kararı her NA kolonu için count==0 / eşlik-eden-kolon kanıtıyla ayrı verilir; "count → otomatik 0" sezgisi yasak.
8. **DETERMİNİZM**: `SEED=42`, kolon sıralaması sabit, `category` kategori sırası sabit → aynı OOF tekrar üretilir (reproducibility şartı).

## Teslimler (Deliverables)
- `src/cleaning.py` (`clean_raw`), `src/preprocessing.py` (`build_preprocessor`), `src/postprocess.py` (`clip_predictions` + assert).
- `data/column_spec.json` (sayısal/kategorik/drop/flag kolon manifesti + her NA kolonu için doldurma stratejisi: `internship_duration_months → zero_flag`, diğer 6 → `median_flag`).
- Birim test / sanity script: aynı input → aynı output; fold-içi fit'in valid satırını görmediğini doğrulayan test (örn. valid medyanının train medyanından bağımsız üretildiğini kanıtla). **Ek test**: `open_source_contribution_count`'un `0` ile değil medyanla dolduğunu ve `github_avg_stars` ile aynı `_missing` maskesini taşıdığını assert et.
- Faz 02 `data/folds.parquet` ile satır-hizalı `fit/transform` örnek koşusu (1 dış fold'da end-to-end).

## Definition of Done
- [ ] `build_preprocessor().fit_transform(train_fold)` + `.transform(valid_fold/test)` çalışıyor, NaN bırakmıyor (clip/impute sonrası `X.isna().sum().sum()==0` — metin kolonu hariç).
- [ ] `student_id`, `application_year`, `graduation_year` çıktı matrisinde **yok**; 7 NA kolonu için 7 `_missing` flag **var**.
- [ ] Yalnızca `internship_duration_months` `0` ile dolduruluyor (`zero_flag`); diğer 6 NA kolonu (`open_source_contribution_count` dâhil) fold-içi medyanla dolduruluyor (`median_flag`).
- [ ] `open_source_contribution_count`'a doldurulan değer fold'a göre değişir (medyan) ve sabit `0` DEĞİL; `_missing` maskesi `github_avg_stars`'ınkiyle birebir aynı.
- [ ] Çıktı matrisinde adversarial train/test AUC ≤ 0.55 (yıl drop teyidi).
- [ ] `clip_predictions` clip-dışı değerde `assert` fırlatıyor; clip sonrası tüm değerler `[0,100]`.
- [ ] Determinizm: iki ardışık koşu bit-aynı çıktı verir (`SEED=42`).
- [ ] Sızıntı testi geçer: valid-fold istatistiği train-fold'dan bağımsız üretiliyor.
- [ ] Çıktı, Faz 06 anchor LGBM-num'a beslendiğinde ~91.6 CV MSE mertebesinde sonuç üretir (boru hattının sağlık kontrolü).

## Riskler & Azaltım
- **Risk: sessiz global fit (en sık leak)** → Azaltım: tüm fit'ler `ColumnTransformer` içinde, CV döngüsünde fold-train slice'ında çağrılır; kod review'da "tüm train üzerinde `.fit`" araması.
- **Risk: yanlış MNAR varsayımı (count → körü körüne 0)** → Azaltım: her NA count kolonu için count==0 çakışması gerçek veriden ölçüldü. `internship_duration_months` %82.14 → `0+flag`; `open_source_contribution_count` yalnızca %3.19 → medyan+flag (DÜZELTİLDİ). "Count olması 0-impute gerektirir" sezgisi reddedildi; karar kanıta bağlı.
- **Risk: eşlik-eden eksiklik tutarsızlığı** → Azaltım: `open_source_contribution_count` ve `github_avg_stars` aynı 910 satırda NA (bayt-bayt teyitli) → ikisi de aynı strateji (medyan+flag) ile işlenir, ortak `_missing` sinyali korunur.
- **Risk: kategorik kardinalite test'te yeni seviye** → Azaltım: OHE `handle_unknown='ignore'`; CatBoost/HistGBR native zaten dayanıklı. (Sentetik veri, ID ortüşmesi 0 ama kategori değerleri ortak — yine de güvene al.)
- **Risk: clip'in bias yaratması (uç-100 yığını)** → Azaltım: clip bedava/nötr; bias kontrolü Faz 07'de OOF residual analizi ile izlenir.
- **Risk: BOM yüzünden `student_id` kolonunun bulunamaması** → Azaltım: yükleme adımında `﻿` strip + kolon listesi assert.

## Süre / Zaman Kutusu
- **Gün 1 (9 Haz) sonu — Gün 2 (10 Haz) başı**: bu faz, Faz 02 (`folds.parquet`) hemen ardından gelir. Sızıntı-güvenli `ColumnTransformer` iskeleti **Gün 1**'de anchor LGBM-num (~91.6 CV) için zorunlu; MNAR/impute/flag ve encoding ablation altyapısı **Gün 2** FE çalışmasıyla iç içe tamamlanır. Bu faz ayrı bir tam gün almaz — Gün 1–2 boyunca tüm modelleme altyapısının temelidir.

## Çapraz Referanslar
- **Faz 02 (Validation)**: `data/folds.parquet` şeması ve fold-içi fit sözleşmesinin kaynağı; bu faz onu tüketir.
- **Faz 04 (Feature Engineering)**: kompozitler (tech_mean/soft_mean/interview_mean/profile_mean), `log1p`, `years_since_graduation` türevi ve encoding ablation bu faza dayanır; `build_preprocessor` FE adımlarıyla genişletilir. **§5.4 (MNAR/impute gruplaması) bu fazla hizalı tutulmalı**: `internship_duration_months` `zero_flag` grubunda, `open_source_contribution_count` ise `github_avg_stars` ile birlikte `median_flag` (log1p sonrası) grubunda işlenir — Faz 04'teki gruplama bu düzeltmeyi yansıtmalı (eski "internship + open_source ortak 0+flag" gruplaması iptal).
- **Faz 05 (NLP)**: `mentor_feedback_text` bu fazda passthrough; TF-IDF/Ridge OOF aynı fold-içi fit kuralına tabi.
- **Faz 06 (Modeling)**: CatBoost native / LGBM OHE-TE / HistGBR native encoding kararları burada kurulur, orada ablation ile sonlandırılır.
- **Faz 07 (Submission)**: `clip_predictions` + assert koruyucu final yazıcıda kullanılır.
- **Kanonik strateji**: `cvProtocol` (fold-içi fit, adversarial sigorta), `leakageRules` (madde 1,2,4,5,7), `lockedDecisions` (yıl kolonları, hedef işleme & post-processing, MNAR FE) ile birebir tutarlı. **Düzeltme notu**: lockedDecisions'taki "MNAR: internship_duration NA→0 + bayrak + diğer 6 NA-kolonu için medyan" ifadesi `open_source_contribution_count`'u açıkça "diğer 6" içinde sayar; bu faz o ifadeyle birebir hizalıdır (yalnızca internship 0, kalan 6 medyan).
```
