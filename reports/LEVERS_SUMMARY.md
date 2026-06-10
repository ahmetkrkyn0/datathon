# Faz 06 — 3 Lever Ozeti (HistGBR / ==100 iki-asama / TF-IDF->SVD->GBDT)

> KARAR METRIGI = recency-weighted nested OOF-MSE. Public LB'ye BAKILMADI. Her lever ancak
> CLAUDE.md kabul kapisini (`yeni_rw < blend_rw - 0.25*std`) gecerse blend'e alindi.
> Blend std = 3.0238 -> esik = **85.4945 - 0.7560 = 84.7385** (lever bu degerin ALTINA inmeli).

## Sonuc: UC LEVER DE ELENDI. Blend rw-OOF degismedi (85.4945). 2 final ayni.

| # | Lever | Standalone rw-OOF | Blend'e etki (nested rw-OOF) | Kapi (≤84.7385?) | Karar | Neden |
|---|-------|-------------------|------------------------------|------------------|-------|-------|
| 1 | **histgbr_full** (HistGradientBoosting, 3. GBDT ailesi, FULL matris) | 88.54 | 85.4945 → **85.5560** (+0.06) | HAYIR | **ELENDI** | ridge_pos blend agirlik = **0.0000**; iyilesme yok, hafif kotulesti (gurultu). LGBM/CatBoost zaten bu sinyali kapsiyor. |
| 2 | **blend_p100** (==100 iki-asama, P(y≥100) LGBM-binary OOF, pred'=blend+a·p·(100−blend)) | — (post-process) | 85.4945 → **85.2959** (−0.20) | HAYIR | **ELENDI** | Sinifsayici guclu (AUC 0.958) ve gercek iyilesme var (−0.20) AMA oracle tavani (−3.57) mukemmel siniflandirma; gercekte false-positive'ler 100'e itiliyor -> kazancin ~%95'i kayboluyor. nested α≈0.65 (naif 1.0 degil). Kapidan uzak. |
| 3 | **txt_svd_gbdt** (TF-IDF → fold-ici TruncatedSVD(80) → LGBM, nonlineer metin) | 182.66 | 85.4945 → **85.5123** (+0.02) | HAYIR | **ELENDI** | SVD %19 varyans (kayipli) -> txt_ridge'den zayif; blend'de txt_ridge agirligini yiyor (corr 0.886, fazla redundant), net fayda yok. |

### Detaylar / fold-safe kanit
- **Hepsi fold-safe:** histgbr/svd-gbdt -> `cv.run_oof` + tek `folds.parquet`; vectorizer+SVD ve
  P(y≥100) siniflandirici HER dis-fold train'inde fit (dis-valid/test'e fit YOK), `random_state=42`.
  iki-asama α'si **nested** secildi (her (repeat,fold) hucresi DISINDAN) -> meta-overfit yok.
- **Tum OOF clip[0,100]** (`cv.clip_predictions`); DoD-4 ic-tutarlilik (oof.npy yeniden-hesap ±1e-6) gecti.
- Artefaktlar + ledger satirlari (reports/model_scores.csv) **dokumantasyon icin tutuldu**; nihai
  blend havuzu (`ensemble.py CANDIDATE_POOL`) degismedi — uc lever de yorum-blok olarak belgelendi.
- `finalize_submissions.py` SUB-1 secimine **prefix-disleme guvencesi** eklendi (blend*/txt_ridge*/*_w
  turetilmis/elenmis satirlar tek-model havuzunda olmasin; blend_p100 yanlislikla SUB-1 olmaz).

## Guncel 2 final (DEGISMEDI — uc lever de kapiyi gecemedi)

| Final | Model | rw-OOF | Yapi | test_uretim |
|-------|-------|--------|------|-------------|
| **SUB-1 (CAPA/safe)** | `catboost_full` | **86.4149** | sade tek GBDT (en dusuk rw-OOF tek-model) | fold-bagging (15 model) |
| **SUB-2 (EN IYI CV)** | `blend` (ridge_pos: lgbm_full+lgbm_num+lgbm_full_w+catboost_full+catboost_full_w+txt_ridge) | **85.4945** | recency-weighted ensemble | OOF-stack (recency-weighted) |

- Yapisal cesitlilik: SUB-1 vs SUB-2 ortalama |fark| = 0.78 (tek-model vs ensemble -> private %40 risk dagitimi).
- Guncel en iyi rw-OOF = **85.4945** (blend). Bu turdaki overnight calismadan **degismedi**; uc lever
  de gurultu-bandinda/zararli cikti ve Occam + sifir-overfit ilkesiyle disarida birakildi.

## BERT
Bilincli sonraya birakildi (bu turda denenmedi).

---

# TIER-2 — Dusuk-risk rotuslar (IWCV cross-check / fractional-logit / per-year isotonic)

> Forensik 85.4945'i NOISE FLOOR olarak kanitladi (iki bagimsiz GBDT %96.9 hemfikir;
> rw ≈ uw × 265.7/230.6). Beklenti: cogu kapiyi gecmez — kapi korur. Amac metrik-rigor (juri),
> ucuz gated rotuslar, due-diligence. KARAR = nested rw-OOF, kapi = 85.4945 − 0.25·3.0238 = **84.7385**.
> Public LB'ye BAKILMADI.

## Sonuc: hicbiri blend'i dusurmedi; 85.4945 ve 2 final DEGISMEDI. (1 cross-check teyit, 2 elendi/atlandi.)

| # | Lever | Tur | Sonuc | Karar | Neden |
|---|-------|-----|-------|-------|-------|
| 1 | **IWCV + RuLSIF** (`src/iwcv_rulsif.py`) | metrik-rigor cross-check (juri), PRIMER metrik DEGIL | Spearman(PRIMER rw, IWCV)=**1.0000**; en-iyi-tek (catboost_full) + blend secimi degismiyor | **TEYIT** | Cok-degiskenli kovaryat-kayma agirligi (39-dim RuLSIF, HEDEF YOK; ESS 78–90%) mevcut gy-marjinal recency-weight ile AYNI sirayi veriyor. Forensik teyit: test/train hedef-varyans orani **1.152** (forensik 265.7/230.6 ile birebir) -> rw>uw farki dagilim ozelligi, sizinti degil. PRIMER metrik (compute_recency_weighted_mse) korundu. |
| 2 | **Fractional logit base** (statsmodels GLM Binomial/logit) | base model adayi | DENENMEDI (kasitli atlandi) | **ATLANDI** | (a) statsmodels yuklu DEGIL + requirements.txt'te pinli degil -> kurmak pinned-repro yuzeyini bozar (CLAUDE.md "pinli bagimlilik / internet-kapali ayni sonuc"); dusuk-risk rotus felsefesine ters. (b) Forensik lineer-latent yaklasimlarin zayifligini zaten gosterdi (Tobit in-sample MSE 103 > GBDT 76; latent ≈ agirlikli toplam, Spearman≈Pearson -> guclu nonlineer donusum YOK). Logit base'in kapiyi gecme olasiligi cok dusuk; token-ekonomik karar: atla + defterle. |
| 3 | **Per-year isotonic** (`src/isotonic_peryear.py`) | post-kalibrasyon (forensik 3.5) | blend 85.4945 → nested **86.7539** (delta **+1.2595**) | **ELENDI** | graduation_year-grupli isotonic (OOF→hedef), nested fold-safe + thin-cell GLOBAL fallback (2018: 265<300). Her yil-hucresi OOF gurultusune overfit; nested protokol bunu yakaliyor (forensik tutarli: GLOBAL isotonic nested +0.42 idi, yil-grupli daha kotu). Kapidan (84.7385) cok uzak. Artefakt YAZILMADI (Occam). |

### Fold-safe kanit (TIER-2)
- **IWCV:** RuLSIF agirligi HEDEF GORMEZ (yalniz kovaryatlar); mevcut nested/fold-ici OOF'lar yeni
  bir agirlikla yeniden-PUANLANDI (yeni fit yok). alpha-relative ratio UST-sinirli -> ESS cokusu yok.
- **Per-year isotonic:** isotonic map HEDEF GORUR -> NESTED (her hucre DISINDAN fit). Bu, GLOBAL
  isotonic'in dustugu in-sample-overfit tuzagindan kacinir; nested rw-OOF DURUST karar skorudur.
- Tum OOF clip[0,100]; pure numpy/scipy/sklearn (pinned) — yeni bagimlilik EKLENMEDI.

## 2 final (TIER-2 sonrasi DEGISMEDI)
SUB-1 = `catboost_full` (rw 86.4149), SUB-2 = `blend` (rw 85.4945). Duvar ve final adaylar sabit.
