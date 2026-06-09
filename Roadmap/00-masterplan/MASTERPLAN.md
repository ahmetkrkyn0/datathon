# Datathon 2026 — Kazanma Master Planı

> **BTK Akademi / Google / Girişimcilik Vakfı "Datathon 2026" (Kaggle)**
> Görev: Gözetimli **regresyon** — her öğrenci için `career_success_score` (sürekli, 0-100) tahmini.
> Metrik: **MSE** (düşük = iyi). Büyük hatalar karesel cezalanır → uç tahminlerden kaçın, tüm tahminleri `clip[0,100]`.
> Deadline: **14 Haziran 23:59** (bugün 9 Haziran, ~5 gün). Günde max 5 submission, finalde 2 seçilir. Skor %60 public / %40 private (rastgele bölme).
>
> Bu dosya tüm fazların (01-07) **orkestrasyon otoritesidir.** Faz SPEC'leri *nasıl*'ı tanımlar; bu plan *ne zaman, hangi sırayla, hangi karar kapısından geçerek* yapılacağını sabitler. Kanonik strateji (`northStar`, `lockedDecisions`, `cvProtocol`, `leakageRules`, `modelStack`, `nlpPlan`, `submissionPolicy`) ile birebir tutarlıdır.

---

## Kazanma Tezi (north star)

**TEK HEDEF: SIFIR OVERFIT.** Yerel CV (Repeated Stratified 5-fold OOF MSE), private leaderboard'un **sapmasız tahmincisi** olacak; tüm karar otoritesi CV'dedir, public LB **SADECE sağlık sensörüdür.**

Bu tez bir hipoteze değil, **ölçülmüş veriye** dayanır:
- **Adversarial validation kanıtı:** Yıl kolonları (`application_year`, `graduation_year`) çıkarılınca train/test ayırt edilemez (**AUC 0.4995**); yıl kolonlarıyla ayırt edilebilir (**AUC 0.6654**). Demek ki **tek dağılım kayması yıl kolonlarındadır** → yılları ham feature olarak KULLANMA, random stratified KFold private MSE'yi sadık temsil eder.
- **Hedef portresi:** mean 76.94, std 15.19, skew -0.451 (neredeyse normal), `==100` %7.73, `<=50` %4.97, max=100 kesin (>100 yok), var(y) 230.63 → çift-sınırlı yığın, `clip[0,100]` bedava/nötr kazanç.
- **NLP değer kanıtı:** NUM-only CV MSE ~89.86 → +Türkçe metin (Ridge-OOF + lexicon) **~83** (~%7 mutlak iyileşme, düşük varyans). Metinde rakam yok → hazır-cevap sızıntısı yok, gerçek semantik.

**Kazanma mantığı (5 kaldıraç):**
1. **Sızıntı-güvenli fold-içi pipeline** ile CV'yi güvenilir kıl (her `fit` yalnız dış-fold train'inden).
2. **MSE'yi doğrudan optimize eden GBDT'ler** (L2 objective) + `clip[0,100]` ile bedava kazanç.
3. **Türkçe mentor metnini** Ridge-on-TFIDF nested-OOF meta-feature (`txt_ridge_pred`) olarak modele kanıtlanmış ~%7 iyileştirmeyle şok et.
4. **Seed-averaging + basit Ridge/NNLS blend** ile varyansı düşür (sıfır overfit riskli adım önce).
5. **`0.25*cv_std` kabul kapısıyla** marjinal/overfit eden her değişikliği reddet; eşitlikte basit model kazanır (Occam).

**Anchor referanslar:** sadece-sayısal GBM 5-fold CV MSE ~91.6; metin eklenince ~83. Final 2 submission **yapısal olarak farklı** (sade çapa tek-model + en iyi CV ensemble) seçilerek %40 private bölmesine karşı risk dağıtılır. **Public LB ASLA optimizasyon hedefi değildir.**

---

## Genel Mimari & Faz Akışı

7 faz, bir **sızıntı-güvenli OOF boru hattı** etrafında zincirlenir. Faz 02 (Validation) tüm fazların **üst otoritesidir**; her model/feature/HP yalnızca onun protokolünden geçerse kabul edilir.

```
                         ┌─────────────────────────────────────────────┐
                         │  FAZ 02 — VALIDATION (ÜST OTORİTE)           │
                         │  data/folds.parquet (5-fold × 3-repeat)      │
                         │  0.25*cv_std kabul kapısı · OOF sözleşmesi   │
                         │  adversarial sigorta · clip[0,100]           │
                         └───────────────┬─────────────────────────────┘
                                         │ (tüm fazlar bu protokolü kullanır)
                                         ▼
  ham veri      ┌──────────┐      ┌──────────────┐      ┌──────────────────┐
  train.csv ───►│ FAZ 01   │─────►│ FAZ 03       │─────►│ FAZ 04           │
  test_x.csv    │ EDA      │ portre│ Preprocessing│ fold-│ Feature Eng.     │
                │ adversar.│ ───── │ impute+flag  │ içi  │ kompozit/çarpım  │
                │ AUC      │ stratify│ clip kuralı│ pipe │ tech_mean,pq×tech│
                └──────────┘       └──────┬───────┘      └────────┬─────────┘
                                          │                       │
                                          ▼                       ▼
                                   ┌──────────────┐      ┌──────────────────┐
                                   │ FAZ 05 — NLP │      │  feature matrisi │
                                   │ TF-IDF→Ridge │─────►│  + txt_ridge_pred│
                                   │ nested-OOF   │concat│  + lexicon kol.  │
                                   │ lexicon (TR) │      │  (yıl kolonu YOK)│
                                   └──────────────┘      └────────┬─────────┘
                                                                  │
                                                                  ▼
                                              ┌───────────────────────────────┐
                                              │ FAZ 06 — MODELING & ENSEMBLE   │
                                              │ LGBM-L2 / CatBoost / HistGBR   │
                                              │ seed-avg → OOF Ridge/NNLS blend│
                                              │ oof_{M}.npy · test_{M}.npy     │
                                              └───────────────┬───────────────┘
                                                              │
                                                              ▼
                                              ┌───────────────────────────────┐
                                              │ FAZ 07 — EVALUATION & SUBMIT   │
                                              │ CV-only seçim · gap takibi     │
                                              │ SUB-1 (çapa) + SUB-2 (ensemble)│
                                              │ reproducibility · jüri sunumu  │
                                              └───────────────────────────────┘
```

**Bağlantı sözleşmesi:** Faz 01 ölçer (karar vermez), Faz 02 protokolü kurar (tüm fazlar tüketir), Faz 03-05 sızıntı-güvenli feature matrisini üretir, Faz 06 OOF tahminlerini üretir, Faz 07 CV-only seçimle teslim eder. Hiçbir istatistik tüm-train veya train+test üzerinden hesaplanmaz; her `fit` `data/folds.parquet`'in dış-fold train slice'ında.

---

## 5 Günlük Zaman Çizelgesi

### Gün 1 (9 Haz) — Temel & Anchor
- **Faz 01 (EDA) + Faz 02 (Validation) ana günü.** EDA bulgularını (hedef portresi, missingness, kategorik kardinalite, adversarial AUC) `reports/eda/*` artefaktlarına yaz.
- Adversarial validation tekrar teyit (yıllı ~0.66 / yılsız ~0.50).
- `data/folds.parquet` üret (5-fold × 3-repeat, stratify: `==100` ayrı bin + qcut9). Assert: her (repeat, fold)'da `mean(y==100)` ve `mean(y<=50)` global orandan ±%1 içinde; her satır her repeat'te tam 1 kez validation.
- `src/cv.py` (`make_strat_bins`, `get_folds`, `run_oof` fold-bagging test-üretimiyle, `compute_cv_mse`).
- Sızıntı-güvenli fold-içi `ColumnTransformer` iskeleti (Faz 03 başlangıcı): impute + `_missing` bayrak.
- **Anchor: sadece-sayısal LGBM-num (~91.6 CV MSE, std ~4.68)** → `oof_lgbm_num.npy`, `test_lgbm_num.npy`, `cv_scores.csv`.
- `requirements.txt` pinle, `SEED=42` + deterministik bayraklar.
- **1 submit (anchor)** → ilk CV-LB gap ölçümü.

### Gün 2 (10 Haz) — Feature Engineering
- **Faz 04 ana günü.** Kompozitler: `tech_mean` / `soft_mean` / `interview_mean` / `profile_mean` + grup-içi std; **`project_quality_score × tech_mean`** (ölçülen korr ~0.606, en değerli feature); github `log1p`.
- **MNAR işleme (audit-düzeltilmiş):** `internship_duration_months` NA→0 + bayrak (NA'lerin %82.14'ü `internship_count==0`); diğer 6 NA-kolonu için `_missing` bayrak + fold-içi medyan. **`open_source_contribution_count` MNAR DEĞİL** → `github_avg_stars` ile aynı 910 satırda eksik (veri-toplama boşluğu, satırların %96.8'inde repo VAR) → fold-içi medyan + bayrak, **0 enjekte ETME.**
- `conv_rate` ÜRETİLMEZ (korr 0.014, gürültü).
- Kategorik: OOF target-encoding (m~20-50) vs one-hot ablation; global encoding YASAK.
- Her feature grubu için CV-MSE delta + `0.25*std` kabul kapısı (ablation tablosu, anchor 91.6 referans).
- Nihai matriste adversarial AUC ~0.5 teyit.
- **1 submit (FE'li LGBM)** gap teyidi.

### Gün 3 (11 Haz) — Türkçe NLP
- **Faz 05 ana günü.** Türkçe-duyarlı TF-IDF (word 1-2gram, min_df=3, sublinear_tf, max_features~20k) + Ridge(alpha~2) → **nested inner-KFold OOF `txt_ridge_pred`** (tek sürekli kolon).
- 8-12 elle lexicon/yapı özelliği: `n_pos`, `n_neg`, `pos_minus_neg`, `has_ancak`, `len_word`, `n_sentence`, skill-mention. Lexicon ALAN-BİLGİSİYLE sabit (hedefe bakarak SEÇME).
- **Char n-gram ABLATION** (CV kötüleşti 89.38→90.71 → çıkar, belgele).
- Hedef: CV MSE ~89.86 → ~83. Ablation tablosu (NLP'siz vs NLP'li OOF) hazırla.
- **1-2 submit (NLP'li model)** gap ölç; sızıntı yoksa devam.

### Gün 4 (12 Haz) — Modelleme & Ensemble
- **Faz 06 ana günü.** CatBoost-full (native kategorik, ordered boosting) + HistGBR-full (saf sklearn, bağımlılıksız 3. çeşit) base'leri ekle, her base 3-5 **seed-averaging** (ilk varyans düşürücü).
- Optuna ≤50 trial (CV-mean objective, cv_std cezalı) yalnız LGBM için dar arama; seçilen HP ayrı seed/repeat ile doğrula.
- `oof_*` üzerinde Ridge(alpha CV)/NNLS blend (aynı dış fold şeması); GBM-stacker SADECE >1 cv_std geçerse.
- `best_iteration` OOF ortalamasıyla tüm-train refit (refit'te early stopping için test/valid YASAK).
- **1-2 submit (ensemble)** gap teyidi (≥3 hak rezerv).

### Gün 5 (13-14 Haz) — Dondurma & Final Seçim
- **Faz 07 ana günü. YENİ RİSK YOK.**
- Tek temiz notebook 'Save & Run All' internet-kapalı **reproducibility testi** (aynı OOF-MSE ve aynı `test_{M}.npy` üretmeli).
- **FINAL 2 SUB seçimi CV ile:** SUB-1 sade çapa tek-model (en düşük cv_mean'li en basit güçlü GBDT + clip), SUB-2 en iyi CV ensemble (Ridge-stack/NNLS). İkisi yapısal farklı.
- `submissions_log` + ablation + CV-LB gap grafiğiyle jüri sunumu (6-7 slayt) + Q&A kartları.
- BTK başvuru/takım-adı uygunluk checklist.
- **14 Haz 23:59 deadline öncesi final 2 seçimi onayla.**

---

## Faz Bağımlılıkları

| Faz | Bağımlı olduğu | Üretir | Tüketen |
|---|---|---|---|
| **01 EDA** | ham veri | hedef/missing/kategorik portresi, adversarial AUC baz | 02, 03, 04, 05 |
| **02 Validation** | 01 (portre, adversarial) | `data/folds.parquet`, `src/cv.py`, OOF sözleşmesi, kabul kapısı | 03, 04, 05, 06, 07 (**hepsi**) |
| **03 Preprocessing** | 01 (missingness), 02 (folds, fold-içi fit) | `build_preprocessor`, impute+flag, `clip_predictions` | 04, 06 |
| **04 Feature Eng.** | 01 (korr), 02 (kabul kapısı), 03 (pipeline) | feature matrisi (kompozit/çarpım), encoding kararı | 05 (concat slot), 06 |
| **05 NLP** | 02 (nested-OOF sözleşmesi), 03 (TR lowercase), 04 (concat) | `oof_txt_ridge.npy`, lexicon kolonları, NLP ablation | 06 |
| **06 Modeling** | 02 (folds, OOF), 03 (pipeline), 04 (matris), 05 (txt_ridge) | `oof_{M}.npy`, `test_{M}.npy`, blend, refit | 07 |
| **07 Submission** | 02 (artefakt isim/şema, gap eşik), 06 (SUB-1/SUB-2 adayları) | final 2 submission, jüri sunumu | — (teslim) |

**Kritik yol:** 01→02→03→(04∥05)→06→07. Faz 04 ve 05 paralel ilerleyebilir (NLP, FE matrisine concat slotu olarak girer). Faz 02 her şeyin önkoşuludur; `data/folds.parquet` üretilmeden hiçbir CV ölçümü güvenilir değildir.

---

## Kilitli Kararlar (kanonik stratejiden)

### CV Protokolü (en kritik karar)
- **Repeated Stratified 5-fold × 3-repeat** (seeds 42/2026/7, toplam 15 fit). Global `SEED=42`.
- **Stratify:** `==100` AYRI bin (%7.73) + kalan için `qcut(q=9, duplicates='drop')` → ~10 bin.
- **Master fold:** `data/folds.parquet` (`student_id`, `repeat`, `fold`) bir kez üretilir, TÜM modeller aynı dosyayı kullanır → OOF satır-hizalı.
- **OOF sözleşmesi:** her base model `M` için `oof_M` (her satır görülmediği fold'dan) + `test_M` (15 fold modelinin test ortalaması = **fold-bagging KANONİK YOL**). CV-MSE = 15 fold-MSE'nin mean ve std'si.
- **Test-üretim yolu:** **fold-bagging varsayılan/zorunlu** (`test_M` ile `oof_M` aynı 15 modelden → CV submission'ı sapmasız temsil eder). Tüm-train refit yalnızca opsiyonel, OOF-doğrulamalı + `submissions_log.csv`'de "refit" işaretli; iki yol karıştırılmaz.
- **Kabul kapısı:** `yeni_cv_mse_mean < eski_cv_mse_mean - 0.25*cv_mse_std`. Marjinal "0.1 MSE" kazanımları REDDEDİLİR. Eşitlikte daha basit model kazanır (Occam).
- 10-fold tek tur DEĞİL (5×3 aynı 15-fit bütçesiyle daha düşük varyanslı: fold başına 2000 vs 1000 satır).

### Model Stack
- **Seviye-0 base learners** (hepsi `data/folds.parquet`, fold-içi fit, clip[0,100]):
  - **LGBM-num** (anchor, ~91.6 CV): sadece sayısal+FE+kategorik, `objective='regression_l2'`.
  - **LGBM-full** (~83 hedef): + `txt_ridge_pred` + lexicon özellikleri.
  - **CatBoost-full**: kategorikler native, ordered boosting (bağımsız bias).
  - **HistGBR-full**: saf sklearn, bağımlılıksız reproducible 3. çeşit.
  - (Opsiyonel) **XGBoost-full**: yalnız CV'ye çeşitlilik katarsa.
  - Her base 3-5 **seed-averaging** (ilk varyans düşürücü).
- **Muhafazakar HP:** num_leaves 31-63, max_depth 5-7, lr 0.02-0.05 + fold-içi early stopping, min_child_samples 50-100, feature_fraction 0.7, bagging_fraction 0.8, reg_lambda≥1. Optuna ≤50 trial, objective=repeated-CV mean, cv_std cezalı.
- **Seviye-1 blend:** `oof_*` üzerinde Ridge(alpha CV) veya NNLS (aynı dış fold). GBM-stacker varsayılan DEĞİL.
- **Post:** tüm tahminler `np.clip(0,100)`; log/logit YOK; `best_iteration` OOF ortalamasıyla sabit tüm-train refit.

### NLP Planı (Türkçe)
- **Katman A (ana, zorunlu):** word TF-IDF (1-2gram, min_df=3, sublinear_tf, max_features~20k) → Ridge(alpha~2) → **nested inner-KFold OOF** → tek sürekli kolon `txt_ridge_pred`. Ölçülen: NUM+ridge-OOF = 83.21 (std 4.05) > SVD30 (84.40) > ham TF-IDF ağaca.
- **Katman B (ana yanında):** 8-12 elle Türkçe sentiment/yapı özelliği (`n_pos`, `n_neg`, `pos_minus_neg`, `has_ancak`, `len_word`, `n_sentence`). Lexicon ALAN-BİLGİSİYLE sabit, substring/kök eşleştirme.
- **Çıkarılan:** char n-gram (CV kötüleşti 89.38→90.71, ablation tablosuyla belgelenir).
- **Katman C (opsiyonel, sadece SUB-2):** Turkish BERT (dbmdz/bert-base-turkish-cased) frozen mean-pooled → SVD 16-32; SADECE CV'de >0.5 MSE net geçerse. Fine-tune YOK.
- Encoding: temiz UTF-8, mojibake fix YASAK. Türkçe-duyarlı lowercase (I/ı, İ/i); metin ve lexicon AYNI normalizasyon.

### Leakage Kuralları
1. **FOLD-İÇİ FİT MUTLAK:** impute, target/mean encoding, TF-IDF, scaler, SVD, Ridge-on-TFIDF — HER fit yalnız dış-fold train'inden; valid/test'e transform. Hiçbir istatistik tüm-train üzerinde hesaplanmaz.
2. **TARGET-ENCODING:** global mean/target encoding KESİNLİKLE YASAK; yalnız OOF target-encoding + Bayesian smoothing (m~20-50) veya one-hot.
3. **TF-IDF/SVD/EMBEDDING:** asla train+test birleşimine veya tüm-train'e fit edilmez; `txt_ridge_pred` nested inner-KFold OOF ile üretilir.
4. **IMPUTATION:** 7 NA'lı sayısal kolonun impute değeri fold-içi fit; `_missing` bayrakları fold-bağımsız (sadece `isna()`, hedefe bakmaz).
5. **YIL KOLONU:** `application_year`/`graduation_year` ham veya yıl-bazlı agregasyon/target-encode YASAK (adversarial AUC 0.664); sadece shift-invariant türev + AUC ~0.5 teyidi.
6. **LEXICON:** hedefe bakarak (kelime-hedef korelasyonuyla) seçim de-facto target leakage; alan-bilgisiyle sabitlenir.
7. **ID/SIRA:** `student_id` ASLA feature değil (sentetik, train/test örtüşme 0).
8. **STACK/BLEND:** meta-model SADECE out-of-fold level-1 tahminleri üzerinde; in-fold stacking aşırı-iyimser OOF üretir.
9. **FINAL-FIT:** tüm-train refit'te `best_iteration` OOF ortalamasıyla SABİT; refit'te test/valid ile early stopping YASAK.
10. **OPTUNA META-LEAK:** ≤50 trial, objective=repeated-CV mean, cv_std cezalı; nihai HP ayrı seed/repeat ile doğrula.
11. **TÜRKÇE LOWERCASE:** metin ve lexicon AYNI Türkçe-duyarlı normalizasyon; karışık normalizasyon sessizce eşleşme kaçırır. Mojibake fix YAPMA.

---

## Submission Politikası

**ALTIN KURAL:** Public LB SADECE sağlık-kontrolü sensörü; **optimizasyon hedefi DEĞİL.** Tüm model/HP/feature kararları CV-MSE(mean, std) ile. Public %60 ağırlıklı olsa da rastgele bölündüğü için public'e overfit private %40'ta çöker.

**Bütçe (günde max 5 hak):** Her gün 5 harcanmaz; günlük hakkın **≥3'ü rezerv.** Submission yalnızca: (a) yeni model ailesi ilk kez LB'ye çıkarken gap ölçmek, (b) final adaylarını teyit etmek. "Biraz daha deneyeyim" tuzağına düşme.

**Defter (`reports/submissions_log.csv`):** `tarih, model_aciklama, notebook_commit_hash, cv_mse_mean, cv_mse_std, public_lb_mse, gap(=public-cv), esik_durumu, test_uretim_yolu, secildi(bool)`.

**CV-LB gap eşikleri:**
- 🟢 **Sağlıklı:** `|gap| ≤ 1.5*cv_std`.
- 🟡 **Sarı:** `1.5-3*cv_std` → sızıntı incele, public'e göre SEÇME.
- 🔴 **Kırmızı:** `>3*cv_std` VE `public < CV` (sızıntı şüphesi) → DUR.

**FINAL 2 SUBMISSION** (her ikisi de CV ile seçilir, public-en-yüksek ASLA seçilmez):
- **SUB-1 (ÇAPA/safe):** en düşük cv_mean'li EN BASİT/az-feature tek güçlü GBDT + clip, gap sağlıklı. "Ne olursa olsun makul" adayı.
- **SUB-2 (EN İYİ CV):** en düşük cv_mean'li Ridge-stack/NNLS ensemble, gap sağlıklı.
- İkisi **YAPISAL farklı** (sade tek-model vs ensemble) → private'ta aynı anda çökme olasılığı düşük = risk dağıtımı. cv farkı <0.25*std ise birini bilerek daha basit/farklı tut.

**Submission format:** `student_id` + `career_success_score`, test ID sırası (STU_010001..020000), index'siz, UTF-8. `sample_submission` ile satır sayısı (10000) ve ID kümesi birebir eşitliği assert. Tüm tahminler `clip[0,100]` (yazıcı clip-dışı görürse hata fırlatır; sample'daki 123.94 yalnız format örneği).

---

## Risk Register

| Risk | Etki | Azaltım |
|---|---|---|
| Fold-dışı fit (sessiz global fit) | CV optimistik, private çöker | TÜM transform `run_oof` döngüsü İÇİNDE `Pipeline.fit`; kod review'da "tüm train üzerinde `.fit`" araması; sızıntı testi (valid medyanı train medyanından bağımsız) |
| CV ile submission farklı modelden gelir (fold-bagging vs refit karışır) | CV-LB gap açılır, private sürpriz | Kanonik yol **fold-bagging**; refit yalnız OOF-doğrulamalı + log-işaretli; bir model için iki yol karıştırılmaz |
| Yıl kolonu/türevi geri sızar | Random CV geçersiz, CV iyi private kötü | Yıl kolonları fiziksel drop; her feature-matris değişiminde adversarial AUC ~0.5 tekrar ölç; AUC>0.6 → suçlu feature reddet |
| `open_source_contribution_count`'a sahte 0 enjekte (yanlış MNAR) | Aktif öğrencilere "0 katkı" sahte sinyali, private'ta farklılaşır | **AUDIT DÜZELTMESİ:** medyan+flag (0 DEĞİL); `github_avg_stars` ile aynı 910 satır maskesi; her NA kolonu için count==0 çakışması ayrı ölçülür |
| Ham TF-IDF'i ağaca verme | 20k seyrek kolonla overfit | Ridge-OOF tek kolona sıkıştır (83.21 < ham TF-IDF); char n-gram ablation ile eler |
| Stacking overfit (10k OOF'ta) | Blend OOF base'lerden çok düşük, gap kırmızı | GBM-stacker yasak; Ridge/NNLS varsayılan; >1 cv_std kapısı; meta yalnız OOF üzerinde |
| Optuna/ablation CV'ye overfit (çoklu karşılaştırma) | Best trial CV düşük, public sapar | ≤50 trial, cv_std cezalı objective; seçilen HP **ayrı seed/repeat ile teyit**; her FE grubu 0.25*std kapısı |
| Nested-OOF unutulur (NLP) | NLP CV sahte iyimser (83 → private yukarı) | `txt_ridge_pred` yalnız nested kosucu üzerinden; inner/dış fold kesişmemesi assert; OOF kapsamı %100 |
| Public LB'ye karar kayması (insan zafiyeti) | Private'ta çökme | Altın kural: public = sağlık sensörü; karar sadece CV; günlük hakkın ≥3'ü rezerv |
| 3 repeat yetersiz çözünürlük | Gürültü kovalama | 0.25*std kapısı gürültü bandını keser; gerekirse 5. repeat (zaman izin verirse) |
| Reproducibility kırılır (farklı OOF tekrar koşuda) | Jüri notebook'u çalışmaz, eleme | SEED=42 her yerde, deterministik bayraklar, sabit thread, pinli requirements; 'Save & Run All' internet-kapalı test (Gün 5 DoD) |
| BTK başvuru/takım-adı uyumsuz | Teknik dışı eleme | Gün 5 uygunluk checklist; takım adı BTK başvurusuyla birebir aynı |

---

## Definition of Done (yarışma)

- [ ] `data/folds.parquet` üretildi; stratify assert geçti (her fold'da `==100`/`<=50` oranı ±%1; her satır her repeat'te 1 kez validation).
- [ ] Anchor LGBM-num reproduce: `cv_mse_mean ~91.6`, `std ~4.68` → `cv_scores.csv`.
- [ ] Nihai feature matrisinde **adversarial AUC ~0.5** (yıl kolonu/türevi yok teyidi).
- [ ] LGBM-full + NLP ile `cv_mse_mean ~83` ve kabul kapısı (`< anchor - 0.25*std`) geçti.
- [ ] `open_source_contribution_count` medyan+flag ile dolduruldu (0 DEĞİL), `github_avg_stars` ile aynı `_missing` maskesi.
- [ ] CatBoost-full + HistGBR-full base'leri 3-5 seed-averaged, OOF/test `.npy` diske yazıldı.
- [ ] Seviye-1 blend kabul kapısını geçti VEYA geçemediyse en iyi tekil model SUB-2 olarak belgelendi (Occam).
- [ ] Tüm tahminler `clip[0,100]`; clip-dışı değer yok (assert).
- [ ] SUB-1 (sade çapa) ve SUB-2 (ensemble) yapısal FARKLI, ikisi de CV ile seçildi; `submissions_log` gap'leri sağlıklı (`|gap| ≤ 1.5*std`).
- [ ] Tek temiz notebook 'Save & Run All' internet-kapalı aynı OOF-MSE ve `test_{M}.npy` üretti.
- [ ] `submissions_log.csv` dolu; final 2 submission 14 Haz 23:59 öncesi onaylandı.
- [ ] BTK başvuru/takım-adı uygunluk checklist tamam.

## Jüri Sunum Hazırlığı (5+3 dk)

İlk 10 takım jüriye **5 dk sunum + 3 dk Q&A** yapar ve **temiz, reproducible notebook** paylaşmak ZORUNDA. Anlatı ekseni: **"validation felsefemiz farkımız."**

**6-7 slayt:**
1. **Problem & metrik** — career_success regresyonu, MSE, clip[0,100] mantığı.
2. **Veri portresi** — hedef dağılımı (==100 yığını, sol kuyruk), missingness, **adversarial AUC bulgusu** (0.66 vs 0.50).
3. **Validation felsefesi (FARKIMIZ)** — neden random stratified KFold private'ı temsil eder (yıl-drop kanıtı); 5×3 repeat, kabul kapısı, public LB'ye koşmama.
4. **Sızıntı-güvenli pipeline** — fold-içi fit, OOF sözleşmesi, MNAR düzeltmesi (internship vs open_source ayrımı).
5. **Türkçe NLP ablation** — NUM-only 89.86 → +Ridge-OOF+lexicon 83.21; char n-gram "denedik, kötüleşti, çıkardık" (en güçlü 30 saniye).
6. **Ensemble & final-2 gerekçesi** — seed-avg + blend; SUB-1/SUB-2 yapısal çeşitlilikle private risk dağıtımı.
7. **Reproduce adımları** — SEED=42, pinli requirements, 'Save & Run All' internet-kapalı.

**Q&A kartları:** "Neden public LB'yi takip etmediniz?", "Yılları neden attınız?", "NLP gerçekten katkı yaptı mı?", "Overfit'ten nasıl emin oldunuz?" — hepsi ölçülmüş sayılarla cevaplanır.

---

## Faz Index

- [Faz 01 — EDA & Veri Anlama](../01-eda-data-understanding/SPEC.md)
- [Faz 02 — Doğrulama Stratejisi (CV) · ÜST OTORİTE](../02-validation-strategy/SPEC.md)
- [Faz 03 — Ön İşleme & Temizleme](../03-preprocessing-cleaning/SPEC.md)
- [Faz 04 — Feature Engineering](../04-feature-engineering/SPEC.md)
- [Faz 05 — NLP: Türkçe Metin Özellikleri](../05-nlp-text-features/SPEC.md)
- [Faz 06 — Modelleme & Ensembling](../06-modeling-ensembling/SPEC.md)
- [Faz 07 — Değerlendirme & Submission](../07-evaluation-submission/SPEC.md)
