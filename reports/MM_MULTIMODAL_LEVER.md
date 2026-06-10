# MM MULTIMODAL LEVER — XLM-R-large + tabular NN (TIER-3, FARKLI fonksiyon sinifi) → **KABUL**

> **Amac:** GBDT/linear ailenin forensik "noise-floor" tezini ORTOGONAL bir fonksiyon sinifiyla
> (neural multimodal) kirmak. `mentor_feedback_text` (XLM-R-large fine-tune, mean+CLS) + fold-safe
> tabular NN dali fusion. **Karar metrigi = nested recency-weighted OOF-MSE (rw-OOF) + PAIRED-TEST.**
> Public LB'ye BAKILMADI.
>
> **SONUC (TL;DR):** mm **KABUL edildi** ve **SUB-2 blend'e kalici eklendi**. Standalone unw-OOF
> **83.30** / rw-OOF **94.82** (tek basina catboost_full 86.41'den zayif AMA ortogonal: corr e5=0.727,
> txt=0.690). Blend'e katkisi **−0.6071** (84.8464 → 84.2393). Literal kapiyi (−0.756) gecmedi AMA e5
> ile AYNI gerekce: paired model-vs-model delta ANLAMLI (15/15 hucre, t=−5.19, p=1.36e-4, bootstrap
> %95 CI [−1.02,−0.21]). Paired-dogrulama + kullanici onayi ile KABUL.

---

## A. URETIM — fold-safe multimodal NN (bizim folds.parquet repeat-0)

| Asama | Detay |
|---|---|
| **Notebook** | `colab_mm_multimodal.ipynb` (GPU/Colab, A100). Self-contained: `data/folds.parquet` **repeat-0** (5-fold) okur; tabular matrisi inline kurar (`src/tabular_mm.py` ile **byte-ozdes** dogrulandi, NaN-aware). |
| **Text dali** | `xlm-roberta-large` (1024-dim), `last_hidden_state` mean-pool + CLS. Fold-ici fine-tune: MAX_LEN=192, BATCH=16, EPOCHS=3, LR_bert=1e-5, LR_head=1e-3, AdamW, bf16, %10 warmup. |
| **Tabular dali** | Fold-safe 82-feat (37 num + 2 yil ham + 7 missing-flag + 36 one-hot). **PER-FOLD** median-impute + StandardScaler (yalniz fold-train'e fit). MLP(256→128). **HEDEF-ENCODING YOK.** |
| **Fusion/head** | `[mean ⊕ cls ⊕ tab(128)]` → Linear(→256)→ReLU→Drop→Linear(→1). Hedef z-score (fold-train ymean/ystd), tahmin ×ystd+ymean, clip[0,100]. |
| **Fold-safety** | NN her dis-fold'un SADECE fold-train'inde egitildi; impute/scale/z-score fold-train'e fit. `oof_mm[i]` = i'nin val oldugu fold tahmini (her satir 1x val). `test_mm` = 5 fold modelinin test ort. |

**Reproducibility:** seed=42 (torch/numpy/cuda manual_seed). Neural/GPU **bit-deterministik DEGIL**
(cuDNN/atomik/bf16). `oof_mm.npy`/`test_mm.npy` KANONIK artefakt — `mm_blend.py`/`ensemble.py` bunlari
yukler, torch'a IHTIYAC DUYMAZ. SUB-2'ye girdigi icin repro **"belgelenmis tolerans"** (bit-ayni degil).
SUB-1 (catboost_full) bit-reproducible kalir (jiri DoD-9 temiz fallback).

## B. STANDALONE — zayif ama ORTOGONAL

| model | unw-OOF | rw-OOF | corr(mm) |
|---|---|---|---|
| **mm** | **83.30** | **94.82** | — |
| catboost_full | 76.30 | 86.41 | 0.948 |
| lgbm_full | 77.03 | 87.27 | 0.945 |
| e5_ridge | 140.16 | 158.46 | **0.727** |
| txt_ridge | 147.49 | 168.02 | **0.690** |

mm tek basina GBDT'lerden zayif, AMA metin kanallarina (e5/txt) korelasyonu DUSUK → blend'e NET-YENI
sinyal. (Not: mm hem metin hem tabular gordugu icin GBDT'lere 0.95 yakin; deger marjinal ortogonalitede.)

## C. BLEND KATKISI — −0.6071, paired-test ile ANLAMLI

`ridge_pos` blend, mm havuza eklenince yeniden secildi:

| Konfig | blend nested rw-OOF | delta |
|---|---|---|
| base (e5'li, mm-siz) | 84.8464 | — |
| **+mm** | **84.2393** | **−0.6071** |

Final ridge_pos agirliklari: `lgbm_full=0.117, lgbm_num=0.289, catboost_full=0.261, catboost_full_w=0.129,
e5_ridge=0.112, mm=0.212, lgbm_full_w=0.000, txt_ridge=0.000, intercept=−9.22`. mm **0.212 agirlik** aldi
(txt_ridge'i 0.000'a itti — mm metin sinyalini de tasiyor; substitution DEGIL, e5+mm birlikte kaldi).

## D. KARAR — literal kapi vs paired-test olcutu

**Literal kabul kapisi (CLAUDE.md):** `yeni < eski − 0.25·std`, std=2.8389 → band 0.7528.
`84.2393` vs `84.8464 − 0.7528 = 84.0936` → literal olarak **GECMEZDI** (−0.607 < −0.753 degil).

**Itiraz (e5 ile AYNI, bagimsiz dogrulandi):** Kapinin std'si blend'in **MUTLAK-MSE seviye-varyansidir**
(zorluk-varyansi); biz **eslesmis (paired) model-vs-model deltasini** yargiliyoruz. Paired delta'nin
KENDI belirsizligi cok daha dar (`src/mm_gate.py`):

| Olcut | deger |
|---|---|
| 15 CV hucresinde iyilesen | **15/15** (hepsi negatif) |
| paired delta mean ± std | −0.6079 ± 0.4534 |
| paired t (p) | **−5.192** (1.36e-4) |
| row-bootstrap %95 CI (B=5000) | **[−1.0154, −0.2070]** tamamen sifir-alti, P(Δ≥0)=0.0020 |

→ Kazanc **gurultu DEGIL, robust sinyal**. e5 kabulu (15/15, t=−8.11, CI[−1.01,−0.29]) ile **AYNI kalite
kanit**. FARK: e5 frozen-embedding+Ridge metin kanali; mm **ORTOGONAL neural multimodal sinifi** →
forensics "GBDT noise-floor" tezini FARKLI fonksiyon sinifiyla kirdi.

**KABUL gerekcesi:** Kapinin AMACI gurultu-ici iyilesmeleri elemek; mm kazanci paired testte gurultu
DEGIL. Kullanici onayi ile **KABUL** → mm kalici blend uyesi.

**Durustluk caveat'lari (kayit icin):** (1) mutlak kazanc kucuk (~0.61, base'in ~%0.72'si); (2) standalone
mm GBDT'lerden zayif (deger sadece blend ortogonalitesinde); (3) mm bit-deterministik DEGIL → SUB-2 repro
belgelenmis tolerans (bit-ayni degil). (4) ilk test repeat-0 (5-fit); full 15-fit'e cikilabilir ama paired
kanit repeat-0'da zaten robust. (5) Karar literal kapiyi DEGIL, daha-dogru paired-yorumunu kullandi.

## E. FINAL ETKISI

| Final | Onceki (e5'li) | Yeni (e5+mm) |
|---|---|---|
| **SUB-1** (safe tek-GBDT) | catboost_full (86.41) | **catboost_full (86.41)** — DEGISMEDI (sade, neural-siz, bit-reproducible) |
| **SUB-2** (ensemble blend) | blend 84.8464 | **blend 84.2393** (e5+mm) |

İki final hala yapisal farkli (tek-model vs neural-iceren ensemble; |fark|≈1.10) → private %40 bolmesine
karsi risk dagitimi korundu. Rakip public ~83.18; biz rw-OOF 84.24'e indik (durust private tahmin).

---

### Artefaktlar / kanit
- `colab_mm_multimodal.ipynb` (GPU notebook, folds.parquet repeat-0), `src/tabular_mm.py` (fold-safe 82-feat matris).
- `artifacts/oof_mm.npy`, `test_mm.npy` (KANONIK; clip[0,100], satir-hizali). `artifacts/tabular_{train,test}.npy` + `tabular_cols.json`.
- `src/mm_blend.py` (artefakt denetim + standalone rapor), `src/mm_gate.py` (paired-test gate).
- `src/ensemble.py` CANDIDATE_POOL'da mm (kabul notu inline). `reports/model_scores.csv` (mm + blend satirlari).
- `submissions/sub2_blend.csv` (e5+mm blend), `submissions/sub1_catboost_full.csv` (degismedi).
