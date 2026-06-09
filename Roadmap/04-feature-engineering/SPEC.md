# Faz 4 — Feature Engineering (Tablosal)

> **Manda (Faz 02'den miras):** SIFIR OVERFIT. Bu fazda uretilen HER feature, `02-validation-strategy`
> protokoluyle (Repeated Stratified 5-fold x 3, `data/folds.parquet`) olculur ve ancak
> **kabul kapisindan** (`yeni_cv_mse_mean < eski_cv_mse_mean - 0.25*cv_mse_std`) gecerse repoda kalir.
> Public LB'ye BAKILMAZ. "Kitchen sink" yasak; her feature grubu ablation tablosuyla hak eder.

---

## 1. Faz 4 — Feature Engineering (Tablosal)

Sayisal skorlardan turetilmis kompozitler, sinirli sayida etkilesim/carpim, sayim-bazli ozellikler,
MNAR-bilincli eksik-deger bayraklari ve `target_role`/`department`/`university_tier`/`hobby`/
`preferred_social_media_platform` icin **fold-safe** kategorik kodlama (OOF target-encoding vs one-hot).
Metin (`mentor_feedback_text`) bu fazda DEGIL — o Faz 05'in isi; burada sadece Faz 05'in urettigi
`txt_ridge_pred` kolonu icin yer ayrilir.

---

## 2. Amac

Anchor'i (sadece-sayisal LGBM CV-MSE ~91.6) sizinti-guvenli, az sayida yuksek-sinyalli turetilmis
feature ile **olculebilir ve kalici** sekilde dusurmek; tum donusumleri tek bir fold-ici
`fit/transform` transformer'inda toplayarak CV'nin private MSE'nin sapmasiz tahmincisi kalmasini saglamak.

---

## 3. "0 Overfit" Rolu — bu faz genellemeye nasil hizmet ediyor

- **Az ama guclu feature** felsefesi: olculmus en guclu sinyal `project_quality_score * tech_mean`
  (korr ~0.606) iken hicbir ham kolonun korr'u ~0.54'u gecmiyor. Demek ki sinyal **etkilesimde** ve
  kompozitte; rastgele 100+ feature uretmek agaca gurultu ekler ve CV-ye overfit yuzeyi acar.
- **Her feature kabul kapisindan gecer:** `0.25*cv_std` esigi (Faz 02 §3.4) marjinal/sansa-bagli
  feature'lari mekanik olarak reddeder. Esitlikte daha az feature kazanir (Occam).
- **Fold-ici fit zorunlulugu** (impute, target-encode) CV'nin iyimser kacmasini engeller; CV-LB gap'i
  kapali tutar.
- **Yil kolonlari (application_year, graduation_year) ham/yil-bazli HIC kullanilmaz** (adversarial AUC
  0.664 -> dagilim kaymasi). Bu, CV'nin private'i temsil etme garantisinin tek somut temelidir.
- Cikti `data/features_*.parquet` tek bir kaynaktir; tum base modeller (Faz 06) ayni matrisi yer,
  reproducibility ve OOF satir-hizasi korunur.

---

## 4. Girdiler / Cikti Artefaktlari

### Girdiler (onceki fazlardan)
| Artefakt | Kaynak | Icerik |
|---|---|---|
| `data/train.csv`, `data/test_x.csv` | ham veri | 47 kolon (46 feature + metin + hedef) / hedefsiz |
| `data/folds.parquet` | Faz 02 | `(student_id, repeat, fold)` — TEK fold otoritesi |
| Faz 03 temizleme ciktisi | Faz 03 | dtype duzeltme, kategorik normalize, NA tespiti (NA'lar BU fazda fold-ici doldurulur) |
| `txt_ridge_pred` (yer ayrildi) | Faz 05 | metin Ridge-OOF meta kolonu — Faz 06'da matrise eklenir |
| `SEED=42` | Faz 02 | global tohum |

### Ciktilar (sonraki fazlara)
| Artefakt | Tuketici | Aciklama |
|---|---|---|
| `src/features.py` | Faz 06 | `build_feature_pipeline()` -> sklearn `Pipeline`+`ColumnTransformer`; fold-ici `fit/transform` |
| `data/features_train.parquet`, `data/features_test.parquet` | Faz 06 | DETERMINISTIK (fold-bagimsiz) feature'lar onbellek; fold-ici olanlar pipeline ile uretilir |
| `reports/fe_ablation.csv` | Faz 06/07 sunum | her feature grubu icin `cv_mse_mean, cv_mse_std, delta_vs_anchor, kabul(bool)` |
| `reports/adversarial_auc.txt` | Faz 02/07 | nihai feature matrisinde train/test AUC (~0.5 teyidi) |
| `config/feature_groups.yaml` | Faz 06 | feature grup listesi + secim bayraklari (ablation sonucu) |
| Kategorik kodlama karari (TE vs OHE) | Faz 06 | ablation ile sabitlenir; LGBM/HistGBR icin uygulanir, CatBoost native |

---

## 5. Detayli Adimlar

> Tum sayisal degerler train uzerinden olculmustur; kolon adlari `data/train.csv` basligindan birebir alindi.

### 5.0 Kolon sozlugu (rol ayrimi)
- **Hedef:** `career_success_score` (0–100).
- **ID (ASLA feature degil):** `student_id`.
- **YASAK ham (temporal kayma):** `application_year`, `graduation_year`. `age` SUPHELI — once
  adversarial AUC katkisi olculur; yil-turevi davranis gosterirse cikar (bkz. §5.6).
- **Skor kolonlari (22):** `cgpa`, `english_exam_score`, `attendance_rate`, `coding_score`,
  `problem_solving_score`, `data_structures_score`, `sql_score`, `machine_learning_score`,
  `backend_score`, `frontend_score`, `cloud_score`, `devops_score`, `project_quality_score`,
  `portfolio_score`, `linkedin_profile_score`, `cv_quality_score`, `technical_interview_score`,
  `hr_interview_score`, `communication_score`, `teamwork_score`, `leadership_score`,
  `presentation_score`.
- **Sayim kolonlari (14):** `failed_courses_count`, `real_client_project_count`, `internship_count`,
  `internship_duration_months`, `freelance_project_count`, `hackathon_count`, `hackathon_awards`,
  `github_repo_count`, `github_avg_stars`, `open_source_contribution_count`, `certification_count`,
  `bootcamp_count`, `applications_sent`, `interviews_attended`.
- **Kategorik (5):** `department`, `university_tier`, `target_role`, `hobby`,
  `preferred_social_media_platform`.

### 5.1 Kompozit ortalamalar (grup-mean) + grup-ici std
Pandas vektorize; satir-bazli `mean`/`std` (NA'lar fold-ici impute SONRASI veya `skipna=True` ile).
- `tech_mean` = mean(`coding_score`, `problem_solving_score`, `data_structures_score`, `sql_score`,
  `machine_learning_score`, `backend_score`, `frontend_score`, `cloud_score`, `devops_score`).
  **Olculen korr ~0.338 — her tekil teknik skordan guclu.**
- `soft_mean` = mean(`communication_score`, `teamwork_score`, `leadership_score`, `presentation_score`).
- `interview_mean` = mean(`technical_interview_score`, `hr_interview_score`).
- `profile_mean` = mean(`portfolio_score`, `linkedin_profile_score`, `cv_quality_score`).
- Grup-ici tutarsizlik: `tech_std`, `soft_std` (std). Gerekce: dengeli profil mi yoksa tek-yon mu —
  agac bu degiskenle dengesizligi yakalar. **Once tech_std/soft_std denenir; gecmezse cikarilir.**

### 5.2 Anahtar etkilesim/carpimlar (SINIRLI: toplam 4–6)
- **`pq_x_tech` = `project_quality_score` * `tech_mean`** — **olculen korr ~0.606, tum ham/kompozit
  ustunde. Bu fazin en degerli tek feature'i.**
- `interview_x_tech` = `interview_mean` * `tech_mean` (mulakat performansi teknik yetkinlikle birlikte).
- `pq_x_soft` = `project_quality_score` * `soft_mean` (ekip projeleri sinyali).
- `profile_x_interview` = `profile_mean` * `interview_mean` (vitrin + mulakat).
- **Kombinatoryal patlama YASAK:** her yeni carpim kabul kapisindan tek tek gecer; gecmeyen silinir.
  4'u baz, 5–6'ya ancak ablation onaylarsa cikilir.

### 5.3 Sayim-bazli ozellikler ve oranlar
- **Dogrulanmis tek oran:** YOK — `conv_rate = interviews_attended / applications_sent` denendi,
  hedefle korr ~0.014 (gurultu) -> **URETILMEZ.** (Bu kararin ablation kaniti `fe_ablation.csv`'de.)
  Yine de tek bir aday olarak `interview_rate` (= `interviews_attended/(applications_sent+1)`)
  kabul kapisina sokulur; gecmezse atilir (varsayilan: atilir).
- **Skew duzeltme (log1p):** `github_avg_stars`, `open_source_contribution_count`, `github_repo_count`
  (uzun-kuyruklu sayimlar). `np.log1p`; agac monoton donusumden etkilenmese de uc degerlerin
  yaprak-bolme uzerindeki etkisini yumusatir ve lineer/Ridge tarafinda (Faz 05 ile uyum) yardimci olur.
- **Toplam aktivite kabasi:** `total_projects` = `real_client_project_count` + `freelance_project_count`
  + `open_source_contribution_count`; `total_credentials` = `certification_count` + `bootcamp_count`.
  Yalnizca kapi gecerse.

### 5.4 MNAR (Missing Not At Random) eksik-deger isleme
Faz 02 §0: test ve train missingness orani neredeyse ESIT (dagilim kaymasi yok) ama OOF saflik
kurali yine de gecerli. 7 NA-li sayisal kolon:
- **`internship_duration_months` (~%16.6 NA):** NA'lerin buyuk cogunlugu `internship_count == 0` ile
  ortak (semantik: staj yoksa sure yok). **NA -> 0 + `internship_duration_missing` bayrak (0/1).**
  Medyan impute YASAK (sahte "orta sure" uydurur).
- **Diger 6 kolon** (`github_avg_stars` ~%9.1, `open_source_contribution_count` ~%9.1,
  `english_exam_score` ~%9.5, `hr_interview_score` ~%7.8, `linkedin_profile_score` ~%6.7,
  `portfolio_score` ~%3.6): her biri icin `<kolon>_missing` bayrak + **fold-ici medyan impute**
  (`SimpleImputer(strategy='median')`, ColumnTransformer icinde, sadece train-fold'a `fit`).
- Bayraklar agaca "eksiklik sinyalini" verir; impute sadece sayisal slotu doldurur.

### 5.5 Kategorik kodlama (fold-safe) — TE vs OHE ablation
Kardinaliteler dusuk-orta (4–11). Iki yol ABLATION ile karsilastirilir, kazanan sabitlenir:
- **One-Hot (OHE):** `OneHotEncoder(handle_unknown='ignore')` — basit, sizintisiz, dusuk kardinalitede
  ideal. LGBM/HistGBR icin varsayilan aday.
- **OOF Target-Encoding:** `category_encoders` veya elle; **Bayesian smoothing (m ~ 20–50)**,
  **fold-ici hesap** (sadece o dis-fold'un train parcasinda kategori-hedef ortalamasi). GLOBAL TE YASAK
  (Faz 02 leakageWarnings) — hedef sizar. Gorulmemis kategori -> global prior'a duser.
- **CatBoost:** 5 kategoriyi `cat_features` ile native verir (kendi ordered TE'si) — bu base icin
  ek kodlama URETILMEZ; cesitlilik kaynagi.
- Karar: OHE vs OOF-TE, kabul kapisi (0.25*std) ile; fark anlamsizsa OHE secilir (daha basit, sizintisiz).

### 5.6 `age` ve yil-turevi guvenlik kontrolu
- `age`: oncelikle ham birakilir ama adversarial AUC'ye katkisi olculur (§5.7). Yil-turevi gibi
  davranirsa (test'te kaymis) cikarilir.
- `years_since_graduation` gibi shift-invariant bir turev **denenebilir** ANCAK: (1) sabit capaya gore
  (or. `application_year - graduation_year` farki — yil farki shift-invariant olabilir), (2) nihai
  matriste adversarial AUC ~0.5 KALDIGI dogrulanirsa. Ayrismayi yeniden acan turev derhal reddedilir.
  **Varsayilan: yil-turevi YOK** (ekstra is, marjinal kazanc, yuksek kayma riski).

### 5.7 Pipeline insasi ve olcum dongusu
1. `build_feature_pipeline()`: deterministik FE (kompozit/carpim/log1p/bayraklar) ham pandas; fold-ici
   olanlar (impute, TE, OHE, opsiyonel scaler) `ColumnTransformer` icinde.
2. Faz 02'nin `data/folds.parquet`'i ile dis-fold dongusu; her dis-fold train'inde `fit`, valid/test'te
   `transform`. Anchor LGBM (`objective='regression_l2'`, muhafazakar HP) ile OOF-MSE(mean,std) olc.
3. Feature gruplari **incremental** eklenir (anchor -> +5.1 -> +5.2 -> +5.3 -> +5.4 -> kategorik karar),
   her adimda `fe_ablation.csv`'ye satir. Kabul kapisi gecmeyen grup geri alinir.
4. Nihai matriste **adversarial validation** (train=0/test=1, ayni 5-fold) -> AUC raporu; >0.6 ise suclu
   feature (ozellikle yil-turevi/`age`) cikarilir.
5. Secilen feature grup bayraklari `config/feature_groups.yaml`'a yazilir; deterministik feature'lar
   `features_*.parquet`'e onbelleklenir.

---

## 6. Kararlar & Gerekceler

| Karar | Gerekce | Elenen alternatif |
|---|---|---|
| Kompozit `tech_mean`/`soft_mean`/... uret | Olculen korr (tech_mean 0.338) tekil skorlari asiyor; sinyal toplulukta | Tum skorlari ham birakip agaca sec birakmak — etkilesim sinyalini gec yakalar |
| `pq_x_tech` carpimi mutlaka | Korr 0.606, en guclu tek feature | Sadece toplam/ortalama — carpimsal etkilesimi kaciriyor |
| Carpimlari 4–6 ile sinirla | Kombinatoryal patlama overfit kaynagi; her biri kapidan gecmeli | Tum ikili carpimlar (kitchen sink) — CV-ye overfit |
| `conv_rate` URETME | Olculen korr 0.014, gurultu | Sezgisel "donusum orani" — kanitsiz, sinyalsiz |
| `internship_duration` NA->0 + bayrak | MNAR: NA'lar count==0 ile; medyan sahte sure uydurur | Medyan impute — yanlis semantik |
| Diger 6 kolon: bayrak + fold-ici medyan | Missingness bilgi tasiyabilir; impute fold-ici sizintisiz | Global impute — OOF saflik ihlali |
| Yil kolonlari ham YASAK | Adversarial AUC 0.664 (yilli) vs 0.491 (yilsiz) | Ham yil/yil-target-encode — CV iyi, private kotu |
| OHE vs OOF-TE ablation ile | Kardinalite dusuk; OHE cogu zaman yeter ve sizintisiz | Global TE — hedef sizar (kesin yasak) |
| log1p sadece uzun-kuyruk sayimlar | Uc degerleri yumusatir, Faz 05 lineer tarafiyla uyum | Tum sayisallari donusturmek — gereksiz, skor kolonlari zaten sinirli araliklı |
| Tek `features.py` + ColumnTransformer | Reproducibility + fold-ici fit garantisi + OOF satir-hizasi | Script disina dagilmis ad-hoc FE — sizinti ve tekrar-uretilemezlik riski |

---

## 7. Leakage / Overfit Guardrail'lari

Faz 02 §2.2 ve kanonik `leakageRules` ile birebir tutarli:

1. **FOLD-ICI FIT MUTLAK:** `SimpleImputer` (medyan), target-encoding istatistigi, `OneHotEncoder`
   kategori kumesi, varsa `StandardScaler` — HEPSI sadece dis-fold train'inden `fit`. Hicbir istatistik
   tum-train uzerinde hesaplanmaz. (Nihai test tahmininde tum-train'e fit dogru; CV OLCUMUNDE yasak.)
2. **GLOBAL TARGET-ENCODING KESIN YASAK:** `department`/`target_role`/`university_tier`/`hobby`/
   `preferred_social_media_platform` mean-encode yalnizca OOF + Bayesian smoothing (m~20–50).
3. **YIL KOLONU LEAK:** `application_year`/`graduation_year` ham/agregasyon/target-encode YASAK. Yil-turevi
   ancak adversarial AUC ~0.5 kaldigi DOGRULANIRSA.
4. **ID / SIRA LEAK:** `student_id` ASLA feature degil; sira-bazli (index, satir no) feature URETILMEZ.
5. **IMPUTATION LEAK:** impute degeri fold-ici; `_missing` bayraklar deterministik (NA->1) ama deger
   fold-ici fit.
6. **KORRELASYON-GUDUMLU FEATURE SECIMI TUZAGI:** hangi feature'in tutulacagina **tum-train hedef
   korrelasyonuna** bakarak DEGIL, **CV-MSE kabul kapisina** gore karar verilir (korr sadece hipotez
   uretir; secim CV yapar). Aksi de-facto target leakage.
7. **HEDEF SIZINTISI:** hicbir feature `career_success_score`'dan turetilmez; metin meta-kolonu
   (`txt_ridge_pred`) Faz 05'te nested-OOF ile uretilir, bu fazda yalnizca slot olarak gecer.
8. **Adversarial sigorta:** nihai matriste AUC ~0.5 zorunlu teyit (>0.6 -> feature incele/cikar).

---

## 8. Teslimler (Deliverables)

- `src/features.py` — `build_feature_pipeline()` (sklearn Pipeline + ColumnTransformer), deterministik
  FE fonksiyonlari, fold-ici transformer'lar.
- `data/features_train.parquet`, `data/features_test.parquet` — deterministik feature onbellegi.
- `config/feature_groups.yaml` — feature grup listesi + ablation sonucu secim bayraklari.
- `reports/fe_ablation.csv` — `grup, cv_mse_mean, cv_mse_std, delta_vs_anchor(91.6), kabul`.
- `reports/adversarial_auc.txt` — nihai feature matrisi train/test AUC + suclu-feature notu.
- Kisa karar notu (markdown degil; `fe_ablation.csv` + `feature_groups.yaml` yorumlari): TE vs OHE karari,
  `conv_rate` ret kaniti, yil-turevi karari.

---

## 9. Definition of Done (olculebilir bitti kriterleri)

1. `src/features.py` calisiyor; pipeline `data/folds.parquet` ile fold-ici `fit/transform` ediyor,
   train+test ayni `transform` yolundan geciyor (NaN ciktisi yok, dtype tutarli).
2. Anchor (sadece-sayisal) CV-MSE ~91.6 (mean,std) yeniden uretildi; FE'li matris kabul kapisini
   (`< 91.6 - 0.25*std`) **anlamli** gecti — hedef bant: tek-basina FE ile gozle gorulur dusus (metin
   sonrasi nihai hedef ~83 Faz 05 ile birlikte).
3. `fe_ablation.csv` her feature grubu icin delta + kabul(bool) iceriyor; reddedilen gruplar matriste YOK.
4. Nihai feature matrisinde adversarial AUC <= ~0.55 (ideal ~0.5); >0.6 cikaran feature cikarildi.
5. `student_id`, `application_year`, `graduation_year` matriste YOK; `_missing` bayraklari mevcut ve
   `internship_duration_months` NA->0 mantigi dogrulandi.
6. TE secildiyse global-TE olmadigi (fold-ici uretildigi) kod incelemesiyle teyitli.
7. Ciktilar (`features_*.parquet`, `feature_groups.yaml`) Faz 06'nin tukettigi sema ile uyumlu.

---

## 10. Riskler & Azaltim

| Risk | Etki | Azaltim |
|---|---|---|
| Feature-bombasi -> CV'ye overfit | private MSE patlar | 0.25*std kabul kapisi; carpim sayisi <=6; reddedileni geri al |
| Gizli yil sizintisi (`age`/turev) | CV iyi, private kotu | adversarial AUC zorunlu teyit; >0.6 -> cikar |
| Global TE kazara | hedef sizar, CV-LB gap acilir | TE yalniz ColumnTransformer fold-ici; kod incelemesi DoD'de |
| `internship_duration` yanlis impute | sahte sinyal | NA->0 + bayrak kuralina sadik kal, medyan kullanma |
| Korr-gudumlu secim (de-facto leak) | iyimser CV | secim SADECE CV-MSE ile; korr sadece hipotez |
| ColumnTransformer kolon sirasi degisimi | OOF satir/kolon hizasi bozulur | sabit kolon listesi + parquet sema testi; SEED=42 |
| Optuna/HP'nin FE ablation ile karismasi | cifte overfit | FE ablation SABIT HP (anchor) ile; tuning Faz 06'da ayri |

---

## 11. Sure / Zaman Kutusu

**Gun 2 (10 Haziran) — Feature Engineering.** Tum gun bu faza ayrildi:
- Sabah: kompozitler (§5.1) + carpimlar (§5.2) + log1p/MNAR (§5.3–5.4), pipeline iskeleti.
- Ogle: TE vs OHE ablation (§5.5), `fe_ablation.csv` doldur.
- Aksam: adversarial teyit (§5.7), `feature_groups.yaml` sabitle, **1 submission** (FE'li LGBM) ile
  CV-LB gap teyidi (Faz 07 submission politikasina gore; gunluk 5 hakkin >=3'u rezerv).
- Cikti Faz 03 (temizleme, eszamanli/once) ve Faz 05 (NLP, Gun 3) tarafindan tuketilmeye hazir.

---

## 12. Capraz Referanslar

- **Faz 02 — Validation Strategy:** UST OTORITE. `data/folds.parquet`, OOF sozlesmesi (§2.1), kabul
  kapisi `0.25*std` (§3.4), adversarial validation (§5), leakage kurallari (§2.2). Bu fazin TUM olcumleri
  oradaki protokolle yapilir.
- **Faz 01 — EDA:** korrelasyon/missingness/dagilim bulgularinin kaynagi (tech_mean 0.338, pq_x_tech
  0.606, conv_rate 0.014, NA oranlari). FE hipotezleri EDA'dan dogar.
- **Faz 03 — Preprocessing/Cleaning:** dtype/kategorik normalize, NA tespiti; bu faz NA'lari fold-ici
  doldurur. Iki faz arasinda gorev siniri: 03 = temizleme/tip; 04 = turetme/kodlama.
- **Faz 05 — NLP:** `txt_ridge_pred` meta-kolonu + lexicon ozellikleri; bu faz onlar icin matriste slot
  birakir. Metin feature'lari Faz 05'in sorumlulugunda, ayni fold semasiyla uretilir.
- **Faz 06 — Modeling/Ensembling:** `features_*.parquet` + `feature_groups.yaml` tuketicisi; CatBoost
  native kategorik, LGBM/HistGBR icin bu fazin TE/OHE karari. HP tuning orada (FE ablation'dan ayri).
- **Faz 07 — Evaluation/Submission:** clip[0,100], `submissions_log.csv` gap takibi, final-2 secimi.
