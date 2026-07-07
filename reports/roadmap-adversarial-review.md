# Datathon 2026 — Roadmap Adversarial (Red-Team) Denetimi

> **Kapsam:** `CLAUDE.md`, [Roadmap/00-masterplan/MASTERPLAN.md](Roadmap/00-masterplan/MASTERPLAN.md) ve [01](Roadmap/01-eda-data-understanding/SPEC.md)–[07](Roadmap/07-evaluation-submission/SPEC.md) SPEC'leri + [DatathonNotes/](DatathonNotes/).
> **Yöntem:** Salt-okunur plan denetimi + `data/train.csv` & `data/test_x.csv` üzerinde tek seferlik salt-okunur Python doğrulaması (hedef/eksik/MNAR/korelasyon/yıl/metin istatistikleri + hızlı LGBM 5-fold CV + yıl-kayması ağırlıklandırma + adversarial AUC) + bulguların bağımsız çoklu-ajan adversarial doğrulaması (8 refute/confirm + 1 completeness ajanı).
> **Bu dosya tek çıktıdır; hiçbir mevcut dosya değiştirilmedi, commit yapılmadı.**

---

## 1. Yönetici Özeti

Plan, **sızıntı (leakage) mekaniği ve EDA dürüstlüğü açısından olağanüstü disiplinli.** Doğruladığım her betimleyici "ölçülmüş" iddia veriyle neredeyse birebir tuttu (hedef portresi, MNAR %82.14, github maskesi bayt-bayt aynı, korelasyonlar, lexicon frekansları, metinde 0 rakam). MNAR audit-düzeltmesi (`open_source_contribution_count` → medyan+bayrak, 0 DEĞİL) veriyle kanıtlı ve doğru; NLP nested inner-KFold OOF tasarımı sızıntısız; fold-içi fit + tek `folds.parquet` + OOF sözleşmesi mimarisi prensipte tam isabetli; CV-only final seçim disiplini sağlam. **Bunlar gerçek güçlü yanlar.**

**Ama planın 1 numaralı tezi — "yerel CV, private leaderboard'un SAPMASIZ tahmincisidir" — bu veri üzerinde EMPİRİK OLARAK YANLIŞ, ve bu tek bir öz-kusurdan kaynaklanıyor.** Plan, tek dağılım kaymasını (yıl kolonları) doğru tespit ediyor; sonra **yanlış hamleyi** yapıyor: kaymayı taşıyan ama aynı zamanda **en değerli, test'te mevcut sinyal grubunu (yıllar) ampute ediyor**, oysa doğru yanıt validation'ı test dağılımına göre ağırlıklandırmaktı. Ölçtüğüm sonuçlar (Bölüm 4):

- **Yılları eklemek CV-MSE'yi 89.97 → 83.17 düşürüyor (~6.8 MSE) — bu, planın etrafında kurulduğu TÜM NLP hattının kazancı (89.86→83.21) kadar.** Yıllar drop'lanıyor.
- **Recency-ağırlıklı (private'ı temsil eden) tahmin yıllı modelde 94.2, yılsız modelde 103.3 — yani yılları atmak gerçek private MSE'yi ~9 MSE KÖTÜLEŞTİRİYOR**, sadece CV sayısını "daha dürüst" gösteriyor.
- **Unweighted random CV, recency-yoğun private'ı her iki modelde de ~10-15 MSE iyimser tahmin ediyor** → "sapmasız tahminci" iddiası tutmuyor.
- Üstelik planın kırmızı-alarm kuralı **yanlış işarete bağlı**: yalnızca `public < CV` (sızıntı yönü) için DUR diyor; ama veri `public > CV` (kayma yönü) öngörüyor → gerçek başarısızlık modu alarmı asla tetikleyemez.

**En kritik 3-5 risk:**
1. **(Critical) Yıl kolonlarını drop kararı muhtemelen YANLIŞ** — verilen iki kolonu atmak, hem CV'de hem private-tahmininde ~7–11 MSE bırakıyor; gerekçe (adversarial AUC) yanlış uygulanmış.
2. **(High) "CV = sapmasız private tahmincisi" iddiası yanlış + gap-alarmı ters işarette** — recency kayması CV'yi ~10-15 iyimser yapıyor; planın güvenlik kuralı bunu yakalayamaz.
3. **(High) 5 günlük TEK-kişilik takvim + hayali "04∥05 paralel" + maliyetlenmemiş Gün-4 compute + MVP geri-çekilme merdiveni yok** → tek-nokta-arıza, deadline riski.
4. **(High) Reproducibility/determinizm konfigürasyonu, jüri-teslim kapısı olan "bit-aynı / ±1e-6 yeniden koşu" DoD'unu karşılayamaz** (LGBM `force_row_wise`, OMP/MKL thread, HistGBR OpenMP eksik).
5. **(Medium) Anchor referansı çelişkili (91.6 vs 89.86 vs 89.38)** — `91.6` yeniden üretilemiyor; kabul kapısının taban çizgisi belirsiz.

**Genel verdict:** Mühendislik hijyeni güçlü; merkezi modelleme kararı (yıl-drop) ve validation kalibrasyonu kusurlu. **Plan bu haliyle DONDURULMAYA HAZIR DEĞİL** — en az "yıl kararı" ve "private-tahmin kalibrasyonu" finalize öncesi revize edilmeli. İyi haber: önerilen düzeltmeler ucuz ve planın kendi altyapısıyla (folds.parquet + kabul kapısı) trivially uygulanabilir.

---

## 2. Bulgular (severity sıralı)

### 🔴 C1 — Yıl kolonlarını "fiziksel drop" kararı muhtemelen YANLIŞ ve en pahalı tek karar
- **Hedef:** `lockedDecisions` (yıl kolonları); [MASTERPLAN](Roadmap/00-masterplan/MASTERPLAN.md) north-star & risk register; [Faz 02 §4/§5](Roadmap/02-validation-strategy/SPEC.md), [Faz 03 §2](Roadmap/03-preprocessing-cleaning/SPEC.md), [Faz 04 §5.6](Roadmap/04-feature-engineering/SPEC.md) | **Severity: Critical**
- **Saldırı:** Plan `application_year` ve `graduation_year`'i **yalnızca adversarial AUC (0.665) yüksek diye** atıyor. Adversarial AUC bir **kovaryat-kayma dedektörüdür, bir "zarar" dedektörü DEĞİL.** Yıllar (a) test'te mevcut (legit feature), (b) tüm test yıl değerleri train'de de var → **ekstrapolasyon yok, sadece yeniden-karışım**, (c) public ve private **aynı test setinin rastgele bölmeleri** olduğundan bir feature tek başına public/private ayrışması YARATAMAZ. Planın risk-register'daki "yıl sızar → CV iyi, private çöker" maddesi, ortak (her iki bölmede de aynı) bir kayma sorununu yanlışlıkla "private çöküşü" gibi çerçeveliyor.
- **Kanıt (kendi ölçümüm, LGBM 5-fold, `regression_l2`, clip[0,100], fold-içi early stopping):**
  - num+kategorik (yılsız): **CV-MSE 89.97**  → num+kategorik+**YILLAR**: **CV-MSE 83.17** (Δ ≈ **−6.8**, tüm NLP hattı kadar).
  - **Recency-ağırlıklı (private proxy)**: yılsız **103.3**, yıllı **94.2** → yılları atmak private'ı **~9 MSE kötüleştiriyor.**
  - Yılsız modelin `graduation_year` boyunca residual'ı (pred−true) **monoton −2.50 (2018) → +2.92 (2026)**; test yıl-karışımında beklenen sistematik sapma **+0.73** (recent öğrencileri fazla tahmin).
  - Bağımsız "savunma" ajanı (yılları tutmanın geri tepip tepmeyeceğini araştırmakla görevli) dahi **yılları tutmanın incelemeden geçtiği** sonucuna vardı.
- **Önerilen düzeltme:** Yılları **a priori atma**; planın kendi `folds.parquet` + 0.25·std kapısıyla **A/B et**. Sızıntı-paranoyası için orta yol: yılları tut **AMA** karar metriğini **recency-ağırlıklı (test yıl-dağılımına importance-weighted) OOF-MSE** veya **temporal-holdout** ile raporla. Tek dürüst çekince: importance-weighting `P(y|yıl)` kararlılığı varsayar ve `graduation_year`'da gerçek hedef-drift var → "yılları tut" daha iyi bahis, risksiz değil; bu yüzden asıl çözüm **feature değil, validation tasarımı.**

### 🟠 H1 — "CV sapmasız private tahmincisidir" tezi yanlış + gap-alarmı TERS işarette
- **Hedef:** [MASTERPLAN](Roadmap/00-masterplan/MASTERPLAN.md) north-star + Submission Politikası gap eşikleri; [Faz 02 §2.3, §7](Roadmap/02-validation-strategy/SPEC.md); [Faz 07 §5.2](Roadmap/07-evaluation-submission/SPEC.md) | **Severity: High**
- **Saldırı:** İki parça. (1) **Tez yanlış:** recency kayması nedeniyle test-yıl-ağırlıklı OOF-MSE, unweighted CV'den her iki modelde de **~10-15 yüksek**. Plan tekrar tekrar (ve jüri slaytı 3'te) sağlayamayacağı bir sapmasızlık iddia ediyor. (2) **Alarm ters:** kırmızı kural `gap>3·std VE public<CV` ister (yani "public şüpheli derecede İYİ → sızıntı"). Ama veri **public'in CV'den KÖTÜ (public>CV)** çıkacağını öngörüyor (recency kayması). En olası gerçek senaryo (public ≈ 95 vs CV ≈ 83, ~+12) sarı banda düşer; sarıya tepki "sızıntı incele, public'e göre SEÇME" → ekip "sızıntı yok (public daha kötü)" deyip **iyimser CV'ye güvenmeye devam eder.** Alarm, gerçek başarısızlık modunu (dağılım kayması) yakalayacak şekilde kablolanmamış.
- **Kanıt:** Reweighted OOF: yılsız 89.97→103.3, yıllı 83.17→94.2. Adversarial AUC numeric-only 0.494 (planın 0.4995 iddiasını DOĞRULUYOR) ama num+kategorik 0.535 → ikincil (küçük) kategorik kayma da var. Not (kalibrasyon lehine): kayma yaklaşık ortak ofset olduğundan **model SIRALAMASI büyük ölçüde korunur** → yanlış model seçimi riski sınırlı; asıl zarar mutlak-MSE beklentisi + jüri iddiası + politika deliğidir.
- **Önerilen düzeltme:** Jüri anlatısında "unbiased" yerine "düşük-varyanslı, recency-düzeltmeli tahminci" de. Headline metriği recency-ağırlıklı/year-stratified OOF-MSE yap. Gap politikasına **simetrik dağılım-kayması dalı** ekle: `public − CV > 3·std` (public çok kötü) ve birden çok model ailesinde kalıcıysa → CV'yi düşük-yanlı kabul et, reweighted-OOF'a geç (bunu "sızıntı değil" diye geçiştirme).

### 🟠 H2 — 5 günlük TEK-kişilik takvim: hayali paralellik, maliyetlenmemiş compute, geri-çekilme merdiveni yok
- **Hedef:** [MASTERPLAN](Roadmap/00-masterplan/MASTERPLAN.md) §5-Günlük Çizelge & Faz Bağımlılıkları ("04∥05 paralel"); [Faz 06 §8/§Süre](Roadmap/06-modeling-ensembling/SPEC.md) | **Severity: High**
- **Saldırı:** Repo tek geliştirici işaret ediyor (`tuna` branch, tek git user). "Faz 04 ve 05 paralel ilerleyebilir" tek kişi için kurgusal. Compute maliyetlenmemiş: yalnız Gün-4'te CatBoost-full + HistGBR-full + LGBM (her biri 3-5 seed × 15 fit ≈ **~225 fit**) + Optuna ≤50 trial × 15 fit (**~750 fit**) + nested-OOF txt_ridge (5 inner × 15 outer = **75** TF-IDF+Ridge fit, ~20k feature). Tek Windows dizüstüde saatlerce–güne yakın saf compute, tampon yok. Herhangi bir tek arıza (CatBoost determinizm, BERT-dataset yükleme, Gün-5 repro uyumsuzluğu) deadline'a yayılır.
- **Kanıt:** Plan metni; hiçbir SPEC wall-clock dakikası vermiyor, "geride kalırsan X'i kes" merdiveni yok.
- **Önerilen düzeltme:** Açık MVP merdiveni ekle ("Gün-3 sonu geride → LGBM-full tek modeli her iki SUB tabanı yap; CatBoost/HistGBR/Optuna/BERT'i kes"). Gün-4 adımlarını gerçek makinede dakika cinsinden maliyetle. Reproducibility testini Gün-4 akşamına çek (Gün-5'e kurtarma payı kalsın).

### 🟠 H3 — Determinizm konfigürasyonu "bit-aynı / ±1e-6 yeniden koşu" DoD'unu (= jüri kapısı) karşılayamaz
- **Hedef:** [Faz 06 §8 + Determinizm guardrail](Roadmap/06-modeling-ensembling/SPEC.md); [Faz 07 §7](Roadmap/07-evaluation-submission/SPEC.md); [Faz 03 DoD "iki ardışık koşu bit-aynı"](Roadmap/03-preprocessing-cleaning/SPEC.md) | **Severity: High**
- **Saldırı:** Plan `deterministic=True`, `seed=42`, "sabit thread" + `feature_fraction=0.7`, `bagging_fraction=0.8` pinliyor. Ama LGBM `deterministic=True` **gerçek reproducibility için `force_row_wise=True` (veya col-wise) ister** — yoksa uyarı verip non-deterministik histogram yoluna düşebilir; belirtilmemiş. HistGradientBoosting OpenMP ile paralelleşir; `random_state` tek başına bit-aynılık vermez (`OMP_NUM_THREADS`/`n_jobs` kısıtlanmalı). DoD "bit-aynı"/"±1e-6 yeniden koşu" diyor ama bunu sağlayacak konfig (force_row_wise, OMP/MKL env, BLAS thread, scipy/NNLS determinizmi) yok. Bu test **ilk-10 jüri-teslim şartı** — Gün-5'te sessizce patlarsa kurtarma yok.
- **Kanıt:** SPEC'lerde force_row_wise / OMP env / thread-pinning yok; "sabit thread" hand-wave. (`oof_*.npy`'den MSE'yi yeniden hesaplayıp `cv_scores.csv` ile ±1e-6 eşleme DoD'u sorun değil — o aynı diziden; sorun **tam yeniden-koşunun** bit-aynılığı.)
- **Önerilen düzeltme:** LGBM `force_row_wise=True`+`deterministic=True`+`num_threads=N`; `OMP_NUM_THREADS`/`MKL_NUM_THREADS` env set et; HistGBR `n_jobs=1` ya da DoD'u "bit-aynı"dan **belgelenmiş toleransa** çevir; iki-koşu deltasını **Faz 06'da** doğrula (Faz 07'ye bırakma).

### 🟡 M1 — Anchor taban çizgisi çelişkili: 91.6 vs 89.86 vs 89.38; `91.6` yeniden üretilemiyor
- **Hedef:** [MASTERPLAN L19/L28](Roadmap/00-masterplan/MASTERPLAN.md), [Faz 02 §Adım7](Roadmap/02-validation-strategy/SPEC.md), [Faz 04 L212/L224](Roadmap/04-feature-engineering/SPEC.md), [Faz 05 §Amaç/DoD](Roadmap/05-nlp-text-features/SPEC.md) | **Severity: Medium**
- **Saldırı:** Aynı "metinsiz" konfig iki farklı sabite çapalanmış: Faz 04 FE kapısını **`< 91.6 − 0.25·std`** ve sütunu `delta_vs_anchor(91.6)` yazıyor; Faz 05 NLP kapısını **`anchor 89.86`**'ya, north-star da 89.86→83'e çapalıyor. Üçüncü sabit 89.38 (char-ablation tabanı). Metin-%7 anlatısı (89.86→83) ile FE kapısı (91.6→FE) **farklı taban kullanıyor** → tek monoton ablation merdiveni (anchor→+FE→+metin) kurulamaz. "Ölçülmüş kanıt, hipotez değil" iddiasını ve DoD'daki "anchor ~91.6 reproduce" maddesini zayıflatır.
- **Kanıt:** num+kategorik ölçümüm **89.97** (≈89.86); saf numeric-only **94.83**. **`91.6` hiçbiriyle eşleşmiyor** ve FE+kategorik eklemek MSE'yi 89.97'nin ÜSTÜNE çıkaramaz → `91.6` yeniden-üretilemez. Bonus: "NUM-only=89.86" etiketi de yanlış; gerçek numeric-only 94.83, 89.86 aslında num+kategorik.
- **Önerilen düzeltme:** Tek taban sabitle: gerçek num+kategorik anchor'ı runtime'da ölç, tüm fazlara aynı sayıyı yaz; FE ve NLP kapılarını **aynı** "eski" değere göre uygula; tilde-bantları "hedef" diye etiketle, DoD'daki sabit reproduce iddiasını gerçek ölçümle değiştir.

### 🟡 M2 — Ensemble kabul kapısı, zorunlu "yapısal çeşitliliği" öldürebilir (SUB-1/SUB-2 ikisi de tek-model olur)
- **Hedef:** [Faz 06 §9 + DoD#4/#8](Roadmap/06-modeling-ensembling/SPEC.md), [Faz 07 §2.3 + DoD#5](Roadmap/07-evaluation-submission/SPEC.md), [MASTERPLAN L203](Roadmap/00-masterplan/MASTERPLAN.md) | **Severity: Medium**
- **Saldırı:** "SUB-1 tek-model vs SUB-2 ensemble yapısal FARKLI ZORUNLU" sert DoD; ama **koşulsuz** kapı: `ensemble_mse < min(base_mse) − 0.25·cv_std` (~1.17) sağlanmazsa ensemble REDDEDİLİR, "en iyi tek base SUB-2 olur (Occam)". Korelasyonlu GBDT blend'leri çoğu zaman <1 MSE kazandırır → muhtemel sonuç: SUB-2 tek-modele çöker, **iki final de tek-model** → private-risk-dağıtımı gerekçesi geçersizleşir. Plan kapı-mı-çeşitlilik-mi önceliğini çözmüyor (iç çelişki).
- **Kanıt:** cv_std ≈ 4.68 → kapı ≈ 1.17; blend kazancı tipik <1. CV geçerliliğini/sızıntıyı bozmaz; bu bir seçim-robustluğu boşluğu.
- **Önerilen düzeltme:** Önceliği netleştir: SUB-2 için kapı, çeşitlilik mandasına tabi olsun; ya da **farklı-aile tek model** (CatBoost ordered-boosting + native kategorik) yapısal-farklı SUB-2 olarak kabul edilsin (blend kapıyı geçemese bile).

### 🟡 M3 — Hedef-stratifyli fold'lar CV'yi iyimser/düşük-varyanslı yapar ("sapmasız" overclaim'in ikinci kaynağı)
- **Hedef:** [Faz 02 §Adım1-2 `make_strat_bins`](Roadmap/02-validation-strategy/SPEC.md); MASTERPLAN CV protokolü | **Severity: Medium**
- **Saldırı:** Fold'lar **hedefin kendi binleri** üzerinde stratify (`==100` ayrı bin + `qcut(rest,9)`). Bu, her valid fold'un y-dağılımını global'e neredeyse eşitler → fold-MSE varyansı yapay küçülür, CV-mean **rastgele (hedef-kör) bir bölmeye göre hafif iyimser/düşük-varyanslı.** Private test seti train hedef-dağılımına göre çekilmedi (recency-çarpık) → hedef-stratify CV/private ayrışmasını iyimser yönde besler ve yıl-kayması sorunuyla bileşir.
- **Kanıt:** Tasarım gereği fold y-dağılımı global'e kilitli; reweighted private proxy (~94) zaten unweighted'tan (~83) çok yüksek.
- **Önerilen düzeltme:** Stratify'i fold-dengesi için tut **AMA** "unbiased" deme; ikinci bir **stratifysiz KFold CV-mean'i** muhafazakâr private tahmini olarak, ve/veya recency-ağırlıklı OOF-MSE'yi headline olarak raporla.

### 🟡 M4 — `years_since_graduation` "shift-invariant" varsayılıyor ama değil; 3 SPEC 3 farklı duruş veriyor
- **Hedef:** [Faz 02 §6 leakage kural 5](Roadmap/02-validation-strategy/SPEC.md) ("onaylı türev") vs [Faz 03 §2](Roadmap/03-preprocessing-cleaning/SPEC.md) ("Faz 04'te denenebilir") vs [Faz 04 §5.6](Roadmap/04-feature-engineering/SPEC.md) ("varsayılan YOK") | **Severity: Medium**
- **Saldırı:** `years_since_graduation = application_year − graduation_year`. İki kayan kolonun farkı **ancak ortak additif kayma varsa** invariant olur; oysa iki kolon **farklı miktarlarda** kayıyor (application_year test'te 2025'e çok daha yığılı) → fark da kaymış olur. Plan invariance'ı test etmeden "özellik" diye varsayıyor; tek koruma adversarial AUC ama numeric-only kontrol kategorik 0.535 kaymasını zaten kaçırıyor → zayıf bir yıl-türevi 0.55 eşiğinin altından temporal sızıntıyı geri sokabilir.
- **Kanıt:** Üç SPEC'te üç stance; numeric-only AUC 0.494 vs num+kat 0.535.
- **Önerilen düzeltme:** Tek stance'a indirge (öneri: tamamen bırak, Faz 04 varsayılanıyla). Denersen invariance'ı **türevin TEK BAŞINA adversarial AUC'si ~0.50** ile kanıtla (sadece nihai matris ≤0.55 yetmez) + paired kabul kapısı.

### 🟡 M5 — Blend ağırlıkları aynı 10k OOF satırında fit ediliyor; ensemble kapısı kısmen in-sample
- **Hedef:** [Faz 06 §9](Roadmap/06-modeling-ensembling/SPEC.md), [Faz 07 §2 "base OOF zaten fold-dışı → nested gerekmez"](Roadmap/07-evaluation-submission/SPEC.md) | **Severity: Medium**
- **Saldırı:** Base OOF'lar satır-bazında fold-dışı olsa da **blend AĞIRLIKLARI (NNLS/Ridge katsayıları) tüm 10k OOF'ta fit ediliyor** ve sonra AYNI 10k satırda `oof_ensemble` MSE'si hesaplanıp kabul kapısına sokuluyor. Base'ler yüksek korelasyonlu (hepsi aynı feature + txt_ridge) → NNLS ağırlığı yoğunlaştırır, küçük DoF fold-gürültüsünü cımbızlayabilir. "Nesting gerekmez" yarı-doğru; ensemble-vs-en-iyi-base kararı (yapısal final seçimi) hafif iyimser.
- **Kanıt:** Plan metni; etki küçük (düşük DoF) ama sıfır değil.
- **Önerilen düzeltme:** Blend ağırlıklarını dış-fold döngüsü içinde fit et (her fold'da diğer repeat'lerin OOF'unda NNLS, held-out fold'u tahmin) → gerçekten held-out `oof_ensemble`; ya da hem in-sample hem nested ensemble-MSE raporla, kapıyı **nested** olanla geçir.

### 🟡 M6 — 0.25·std kapısı paired test değil; "std" tanımı tutarsız (folds bağımsız değil)
- **Hedef:** [Faz 02 §2.1 + §6 `compute_cv_mse`](Roadmap/02-validation-strategy/SPEC.md); tüm fazların kabul kapısı | **Severity: Medium**
- **Saldırı:** `compute_cv_mse`, 15 fold-MSE'nin `np.std`'ını döndürür — bu **mean'in standart hatası DEĞİL**, fold-arası MSE yayılımı. 3 repeat aynı 10k satırı tekrar kullandığından 15 fold bağımsız değil → §2.1'deki `SE = std/sqrt(15) ≈ 1.21` argümanı bağımsızlık varsayar (gerçek SE daha büyük). Kapı `0.25·std (≈1.17)` ise fold-arası std'a dayanıyor; iki sayı sqrt(15)≈4 numerolojisiyle tesadüfen örtüşüyor. **15 paired fold-MSE varken paired per-fold delta testi (çok daha güçlü ve doğru) kullanılmıyor.** Çoklu karşılaştırma (FE grupları, Optuna ≤50, model aileleri) için düzeltme yok.
- **Kanıt:** §6 `np.std(per_fold)`; §2.1 SE=1.21; kapı 0.25·std. **Yön notu:** std-of-15 ≈ √15× daha büyük olduğundan kapı aslında **DAHA SIKI/muhafazakâr** → "küçük gerçek kazançları kaçırır", overfit-davetiyesi değil.
- **Önerilen düzeltme:** Kapıyı **paired per-fold delta** üzerinde tanımla (Δ_fold = mse_old − mse_new; ortalama ve std'ı 15 fark üzerinden) + paired t/Wilcoxon. SE=std/sqrt(15)'i bağımsızmış gibi alıntılama. Çokluk için ayrı-seed/repeat re-validasyonu koru.

### 🟡 M7 — `==100` tavan kütlesi (%7.73) yapısal düşük-tahmin; 0.25·std kapısı düzeltmesini oto-reddedebilir
- **Hedef:** [Faz 07 §3.4](Roadmap/07-evaluation-submission/SPEC.md), [Faz 03 §6](Roadmap/03-preprocessing-cleaning/SPEC.md), MASTERPLAN clip gerekçesi | **Severity: Medium**
- **Saldırı:** 773 satır tam 100'de (sansürlü kütle). L2 GBDT bunları koşullu ortalamaya çeker → neredeyse hepsini <100 tahmin eder; **clip yalnızca >100 çıktıyı düzeltir, düşük-tahmini KURTARMAZ.** Bu, kaçınılmaz büyük-MSE residual kaynağı. Plan varsayılanı "saf clip; kalibrasyon yalnız 0.25·std geçerse" — ama tavan-bias katkısı bu mertebede olabilir → kapı, en büyük yapısal hata kaynağının düzeltmesini reddedebilir (iki kilitli kararın kendi-kendini-baltalayan etkileşimi). "clip bedava/nötr" iddiası >100 için doğru, tavan-altı-tahmin için sıfır yardım.
- **Kanıt:** ==100 %7.73 doğrulandı (773); clip asimetrisi.
- **Önerilen düzeltme:** Tavan-kalibrasyonunu birinci-sınıf aday yap (gated-sonrası değil): yüksek tahminlere isotonic/affine post-kalibrasyon veya iki-parçalı P(y=100)+truncated regressor'ı OOF'ta **paired** testle değerlendir; OOF-MSE delta'sını kapıdan bağımsız RAPORLA ki karar bilgili olsun (oto-red değil).

### 🟢 L1 — Fazlar-arası artefakt isim/yol/sütun uyuşmazlıkları
- **Hedef:** `CLAUDE.md` klasör yapısı; [Faz 02](Roadmap/02-validation-strategy/SPEC.md)/[04](Roadmap/04-feature-engineering/SPEC.md)/[05](Roadmap/05-nlp-text-features/SPEC.md)/[06](Roadmap/06-modeling-ensembling/SPEC.md)/[07](Roadmap/07-evaluation-submission/SPEC.md) | **Severity: Low**
- **Saldırı / Kanıt (hepsi doğrulandı):**
  - **OOF/test dizileri 3 yerde 3 farklı:** `CLAUDE.md` → `data/oof_*.npy`; Faz 02/06/07 → `artifacts/oof_{M}.npy`; Faz 05 → `data/oof_txt_ridge.npy`.
  - **Feature matrisi:** Faz 04 üretir `data/features_train.parquet`/`features_test.parquet`; **tüketicisi** Faz 06 okur `data/train_fe.parquet`/`test_fe.parquet` → üretici↔tüketici sınırında farklı isim.
  - **submissions_log konumu:** `CLAUDE.md` `submissions/` vs MASTERPLAN & Faz 02 `reports/`. **Sütun adları da farklı:** `CLAUDE.md` (cv_mean, cv_std, public_lb...) vs Faz 02 (cv_mse_mean, cv_mse_std, public_lb_mse, esik_durumu, **test_uretim_yolu**...); Faz 07 sütun listesi `test_uretim_yolu`'nu düşürüyor.
  - **Düzeltme/Not:** Faz 07 bütünlük denetimi **içerik-tabanlı** (uzunluk, satır-hizası, MSE ±1e-6) → isim/sütun farkında oto-patlamaz; gerçek etki **el-ile entegrasyon footgun'ı.**
- **Önerilen düzeltme:** Tek kanon: tüm OOF/test `artifacts/` altında (Faz 05'i de taşı), tek feature-matris adı, submissions_log `reports/` altında + Faz 02 sütun şeması her yerde (`test_uretim_yolu` dahil).

### 🟢 L2 — HistGBR fold-içi early stopping talimatı API olarak uygulanamaz (yazıldığı haliyle)
- **Hedef:** [Faz 06 §6](Roadmap/06-modeling-ensembling/SPEC.md) "HistGradientBoostingRegressor(... early_stopping=True, validation_fraction yerine fold-içi valid kullan)" | **Severity: Low**
- **Saldırı:** sklearn `HistGradientBoostingRegressor` **harici eval set kabul etmez**; early stopping iç `validation_fraction` ile train'den parça ayırır. "Fold-içi valid kullan" yazıldığı gibi infeasible (ama iç split fold-train'den karıldığı için **sızıntısız**).
- **Kanıt:** sklearn API.
- **Önerilen düzeltme:** İç `validation_fraction`'ı dış-fold train slice'ından kullan (sızıntısız) **veya** `max_iter`'i OOF `best_iteration` ile sabitle. Tek satırlık düzeltme.

### 🟢 L3 — Adversarial validation metodoloji tutarsızlığı + numeric-only kontrol kategorik kaymayı kaçırıyor
- **Hedef:** [Faz 01 §9](Roadmap/01-eda-data-understanding/SPEC.md) (HistGBM sınıflandırıcı) vs [Faz 02 §8](Roadmap/02-validation-strategy/SPEC.md) (LGBM ile yeniden-kontrol) | **Severity: Low–Medium**
- **Saldırı:** Baz çizgi (0.50) HistGBM ile, nihai-matris yeniden-kontrol LGBM ile → farklı sınıflandırıcılar AUC'yi doğrudan kıyaslanamaz kılar. Ayrıca planın "yılsız AUC≈0.50" iddiası **yalnız numeric-only için** geçerli (ben 0.494 buldum); **num+kategorik 0.535** → küçük bir kategorik dağılım kayması (muhtemelen `target_role`) numeric-only kontrolün gözünden kaçar.
- **Kanıt:** with-years 0.666 (≈0.665 ✓), numeric-only 0.494 (≈0.4995 ✓), num+kategorik no-years **0.535**.
- **Önerilen düzeltme:** Tek sınıflandırıcı sabitle; nihai-matris adversarial kontrolü **kodlanmış kategorikler dahil** yapılsın; "yıllar TEK kayma" ifadesini "ana kayma yıllar; kategoriklerde küçük ikincil kayma var" diye düzelt.

---

## 3. Fazlar Arası Tutarsızlıklar (özet)

| # | Tutarsızlık | Nerede | Etki |
|---|---|---|---|
| 1 | Anchor taban: **91.6 vs 89.86 vs 89.38** (91.6 reproduce edilemiyor) | MP, Faz 02/04/05 | Kabul kapısı tabanı belirsiz; "ölçülmüş" iddiası + DoD reproduce zayıflar (M1) |
| 2 | OOF/test yolu: `data/` (CLAUDE.md, Faz05) vs `artifacts/` (Faz 02/06/07) | CLAUDE.md, Faz 02/05/06/07 | Entegrasyon footgun (L1) |
| 3 | Feature matrisi: `features_train.parquet` (Faz04) vs `train_fe.parquet` (Faz06) | Faz 04 ↔ 06 | Üretici/tüketici isim uyuşmazlığı (L1) |
| 4 | submissions_log: konum `submissions/` vs `reports/`; sütunlar `cv_mean...` vs `cv_mse_mean...`; `test_uretim_yolu` Faz07'de düşmüş | CLAUDE.md, MP, Faz02/07 | Defter şeması tutarsız (L1) |
| 5 | `years_since_graduation`: "onaylı türev" (Faz02) vs "denenebilir" (Faz03) vs "varsayılan YOK" (Faz04) | Faz 02/03/04 | Karar belirsiz + invariance test edilmemiş (M4) |
| 6 | Adversarial sınıflandırıcı: HistGBM (Faz01) vs LGBM (Faz02); numeric-only vs +kategorik AUC | Faz 01 ↔ 02 | Baz çizgi kıyaslanamaz; ikincil kayma kaçar (L3) |
| 7 | Ensemble kapısı (red ederse tek-model) vs "yapısal farklı ZORUNLU" | Faz 06/07 ↔ MP | Çelişki: çeşitlilik çökebilir (M2) |
| 8 | Test-üretim yolu Faz02'de fold-bagging'e sabitlenmiş; Faz07 guardrail hâlâ "fold-bagging YA DA refit" der | Faz 02 ↔ 07 | Faz02 düzeltmiş ama Faz07 metni güncellenmemiş (düşük) |

---

## 4. Doğrulanan vs Doğrulanamayan İddialar (veriyle)

**✅ DOĞRULANDI (veriyle neredeyse birebir):**

| İddia | Plan | Ölçtüğüm |
|---|---|---|
| Hedef mean / std / median / skew / var | 76.94 / 15.19 / 77.81 / −0.451 / 230.63 | **76.94 / 15.19 / 77.81 / −0.451 / 230.63** ✓ |
| ==100 / <=50 / ==0 oranı | %7.73 (773) / %4.97 / 1 satır | **%7.73 (773) / %4.97 (497) / 1 satır**; >100 ve <0 yok ✓ |
| Eksik oranları (7 kolon) | 16.57/9.53/9.10/9.10/7.80/6.68/3.64 | **birebir aynı**; test ≈ train; metin 0 NA ✓ |
| internship MNAR | NA'lerin %82.14'ü count==0 | **%82.14** (genel %30.7) ✓ |
| github maskesi | osc & github_avg_stars **aynı 910 satır**; %96.8 repo VAR | **bayt-bayt aynı 910/910**; repo==0 yalnız %3.19; NA-medyan repo 4 vs genel 6 ✓ |
| Korelasyonlar | project_quality 0.541, tech_mean 0.338, pq×tech **0.606**, conv_rate ~0.014 | **0.541 / 0.338 / 0.606 / 0.011** ✓ |
| Kategorik kardinalite | 7/4/11/8/6, test-only yok, Tier1-4 | **birebir**, test-only seviye yok ✓ |
| Yıl kayması | train uniform, test recent; grad-year hedef 77.6→74.0 | train ~uniform; test 2024-26 yığılı; **77.61→73.98** ✓ |
| student_id | train 000001-010000, test 010001-020000, örtüşme 0 | **birebir**, örtüşme 0 ✓ |
| Metin | 33 kelime, 273 char, 10000 unique, 0 rakam | **33.17 / 273.5 / 10000 / 0** ✓ |
| Lexicon frekansları | ancak 5831, geliştir 6302, güçlü 3097, başarı 2526, mükemmel 468, olağanüstü 184, üstün 74 | **birebir aynı** ✓ |
| Adversarial AUC (yıllı) | 0.6654 | **0.666** ✓ |
| Adversarial AUC (yılsız, numeric-only) | 0.4995 | **0.494** ✓ |
| Anchor "NUM-only" ~89.86 | 89.86 | num+kategorik **89.97** ✓ (etiket karışık; bkz aşağı) |

**❌ DOĞRULANAMADI / ÇELİŞTİ:**

| İddia | Sorun |
|---|---|
| **"91.6 anchor" (num + temel FE + kategorik)** | num+kategorik = **89.97**; FE eklemek MSE'yi 89.97'nin ÜSTÜNE çıkaramaz → **91.6 yeniden üretilemiyor** (M1) |
| **"yılsız train/test ayırt edilemez (AUC~0.50)"** → "tek kayma yıllar" | Yalnız numeric-only doğru (0.494); **num+kategorik = 0.535** → küçük ikincil kategorik kayma var (L3, H1) |
| **"random stratified KFold private MSE'nin sapmasız tahmincisidir"** | Recency-ağırlıklı OOF: yılsız **103.3**, yıllı **94.2** vs unweighted 89.97/83.17 → CV ~10-15 **iyimser**; sapmasız DEĞİL (H1, M3) |
| **"NUM-only = 89.86" etiketi** | Gerçek numeric-only (kategoriksiz) = **94.83**; 89.86 aslında num+kategorik → etiket yanlış (M1) |
| **"yılları atmak ~bedava (3.6 puan drift)"** | Atmak unweighted CV'de −6.8, private-proxy'de −9 MSE **maliyetli** (C1) |

**⚪ BAĞIMSIZ DOĞRULANMADI (NLP hattı kurulmadı; kanıt planın iç loglarına bağlı):** char n-gram 89.38→90.71; SVD30=84.40; Ridge-OOF=83.21; Katman B tek başına 86.52; `pos_minus_neg` etkisi −7.4; anchor std ~4.68 (benim tek-5-fold std'larım 4.6–5.3 bandında, tutarlı). Bunlar makul ama planın final reproduce koşusunda teyit edilmeli.

> **Reddedilen aday bulgu (kalibrasyon için):** "FE fazı tiyatro; plan korelasyon≠CV-kazancı ayrımını yapmıyor" iddiam **YANLIŞ çıktı** — [Faz 04 §7 guardrail 6](Roadmap/04-feature-engineering/SPEC.md) ("Korelasyon-güdümlü feature seçimi tuzağı") ve `conv_rate`/`interview_rate` ön-reddi tam da bu ayrımı yapıyor. Geriye yalnızca küçük bir not kalıyor: Faz 04 DoD#2'nin "tek başına FE ile gözle görülür düşüş" beklentisi hafif iyimser, çünkü gerçek kazanç **yıllar (atılan) + NLP**'de.

---

## 5. Eksikler (planda hiç ele alınmamış)

1. **Recency-ağırlıklı / temporal-holdout private tahmini.** Plan, test'in recency-yoğun olduğunu ölçüyor ama CV'yi buna göre ağırlıklandırmıyor → en büyük tek kör nokta (H1/C1).
2. **Yıl-kararı için A/B kanıtı.** Yıllar a priori kilitleniyor; "tut vs at"ı kabul kapısından geçiren bir ablation YOK — oysa altyapı buna hazır.
3. **Gap politikasında dağılım-kayması (public≫CV) dalı yok** — yalnız sızıntı (public≪CV) yönü tanımlı (H1).
4. **Tavan (==100) modellemesi** birinci-sınıf değil; en büyük yapısal MSE kaynağı gated-afterthought (M7).
5. **Compute bütçesi / wall-clock maliyeti ve MVP geri-çekilme merdiveni yok** (H2).
6. **Bit-aynılık için somut determinizm konfigürasyonu yok** (force_row_wise, OMP/MKL env) — ama bu jüri kapısı (H3).
7. **Paired CV karşılaştırması** mevcut paired fold-MSE'lere rağmen kullanılmıyor (M6).
8. **Nested-blend** (ensemble ağırlıklarının held-out değerlendirmesi) yok (M5).
9. **Sentetik-veri farkındalığı:** `mentor_feedback_text` muhtemelen hedeften üretilmiş (yüksek açıklayıcılık, 0 rakam) — bu **sızıntı değil** (test'te de metin var) ama plan bu olasılığı ve "metin neden bu kadar güçlü" sorusunu adreslemiyor (jüri sorabilir).
10. **CV submission bütçesi vs ölçüm ihtiyacı gerilimi:** günde 5 hak, ≥3 rezerv → yeni-aile probe'ları için günde ~2 → 5 modeli + finalleri LB'de görmek için takvim sıkışık (zaman planıyla çapraz-kontrol edilmemiş).

---

## 6. Alt Satır — Önceliklendirilmiş Düzeltme Listesi

**Önce (donmadan ÖNCE şart):**
1. **[C1] Yıl kararını yeniden aç.** Yılları tutan modeli, recency-ağırlıklı/temporal-holdout OOF ile A/B et. Beklenti: private-proxy'de ~9 MSE iyileşme. Kilit kararı "fiziksel drop"tan "kanıtla seç"e çevir.
2. **[H1] Validation'ı recency'ye kalibre et + gap-alarmını simetrikleştir.** Headline metrik = recency-ağırlıklı OOF-MSE; "unbiased" iddiasını düzelt; `public−CV > 3·std` (public kötü) için DUR/yeniden-kalibre dalı ekle.
3. **[H3] Determinizm konfigürasyonunu somutla** (force_row_wise, OMP/MKL thread, HistGBR n_jobs) ve iki-koşu deltasını Faz 06'da doğrula; DoD'u "bit-aynı"dan belgelenmiş toleransa çek (jüri repro kapısı).
4. **[H2] MVP geri-çekilme merdiveni + Gün-4 compute maliyeti ekle**; reproducibility testini Gün-4'e çek.

**Sonra (donmadan önce iyi olur):**
5. **[M1]** Tek anchor sabiti; FE & NLP kapıları aynı tabana; `91.6`'yı kaldır/yenile.
6. **[M2]** Ensemble-kapı vs yapısal-çeşitlilik önceliğini çöz (farklı-aile tek model SUB-2 kabul et).
7. **[M6]** Kabul kapısını **paired per-fold delta** testine çevir.
8. **[M3]** "unbiased" overclaim'ini kaldır; stratifysiz CV'yi muhafazakâr referans ekle.
9. **[M7]** Tavan-kalibrasyonunu kapıdan bağımsız OOF'ta değerlendir/raporla.
10. **[M4]** `years_since_graduation` duruşunu tek SPEC'e indirge + invariance'ı tek-başına AUC ile kanıtla.
11. **[M5]** Nested-blend ile ensemble kapısını geçir.

**Kozmetik (footgun temizliği):**
12. **[L1]** Artefakt isim/yol/sütun kanonunu tek standarda çek.
13. **[L2]** HistGBR early-stopping talimatını API-uyumlu yaz.
14. **[L3]** Adversarial kontrolde tek sınıflandırıcı + kategorikleri dahil et.

---

## 7. Kalibrasyon — Gerçek Güçlü Yanlar (abartısız)

1. **EDA dürüstlüğü olağanüstü:** "ölçülmüş" denen onlarca sayı veriyle neredeyse birebir tuttu. MNAR audit-düzeltmesi (`open_source` → medyan+bayrak, `internship` → 0+bayrak ayrımı) **veriyle kanıtlı ve doğru** — gerçek bir özen işareti. Çoğu yarışmacının kaba "count→0" sezgisine düşeceği yerde plan doğru kararı veriyor.
2. **NLP nested inner-KFold OOF tasarımı sızıntısız ve doğru:** dış-train feature'ı (inner-OOF) ile dış-valid/test feature'ı (inner-model ortalaması) **aynı ~6400-satır alt-modellerden** üretiliyor → klasik stacking sızıntısı engellenmiş; %100 kapsam + inner/dış kesişmeme assert'i belirtilmiş.
3. **Submission disiplini + sızıntı-karantinası mimarisi prensipte tam isabetli:** CV-only seçim, public-en-yüksek ASLA, ≥3 rezerv, 2 yapısal-farklı final, tek `folds.parquet` + fold-içi fit + OOF sözleşmesi, global target-encoding yasağı. Plan **yanlış şeyleri değil, doğru şeyleri** ciddiye alıyor — sorun uygulama hijyeninde değil, bir merkezi modelleme kararında (yıl-drop) ve onun validation'a yansımasında.

> **Tek cümlelik özet:** Plan sızıntıya karşı kusursuza yakın disiplinli; ama tespit ettiği tek kaymayı (yıllar) yanlış yönetiyor — en değerli sinyali atıp, "CV=private" iddiasını sürdürüp, güvenlik alarmını yanlış işarete bağlıyor. Donmadan önce **yıl kararı + recency-kalibrasyonu** mutlaka revize edilmeli.
