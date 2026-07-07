# OVERNIGHT STATUS — Faz 06 (ensemble + recency-optimize) + Faz 07-hafif

**Tarih:** 2026-06-10 (gece otonom kosu) · **Branch:** `tuna` · **SEED=42, deterministik**

## TL;DR
- **HEDEF TUTTU:** karar metrigi **recency-weighted OOF 87.27 → 85.49** (-1.77). 2 final hazir.
- **KARAR METRIGI = recency-weighted OOF-MSE** (`cv.compute_recency_weighted_mse`). Public LB'ye gore
  HICBIR secim yapilmadi (public-overfit tuzagi; final private'da). Unweighted CV yalniz bilgi.
- **2 FINAL (ikisi de rw-OOF ile secildi, yapisal farkli — risk dagitimi):**
  - **SUB-1 (safe, tek-model):** `submissions/sub1_catboost_full.csv` — catboost_full, **rw-OOF 86.41**
  - **SUB-2 (ensemble):** `submissions/sub2_blend.csv` — recency-weighted ridge_pos blend, **rw-OOF 85.49**
  - SUB-1 vs SUB-2 ortalama |fark| = 0.78 (tek guclu GBDT vs 6-model lineer stack).

## Tum modeller / lever'lar (karar defteri = `reports/model_scores.csv`)
| model | unweighted CV | **rw-OOF (KARAR)** | not |
|---|---|---|---|
| **blend (SUB-2)** | 75.75 | **85.49** | ridge_pos, recency-weighted, durust nested meta-CV |
| **catboost_full (SUB-1)** | 76.30 | **86.41** | CatBoost native-kategorik FULL, tek-seed, thread=6 |
| lgbm_full (eski taban) | 77.03 | 87.27 | Faz05 FULL base (num+txt_ridge+lexicon) |
| catboost_full_w | 77.78 | 87.30 | recency-weighted egitim → rw-OOF DUSMEDI |
| lgbm_full_w | 78.31 | 88.02 | recency-weighted egitim → rw-OOF DUSMEDI |
| lgbm_num | 81.70 | 92.82 | yapisal anchor (metinsiz) |
| txt_ridge | 147.49 | 168.02 | metin meta-feature (tek-basina zayif) |
| txt_ridge_wc | 144.10 | 163.89 | STEP2 word+char; blend'e -0.11 (gurultu) → REDDEDILDI |

## Ne denendi, ne ogrenildi
1. **catboost_full (KAZANAN tek-model):** LGBM'den yapisal farkli 2. GBDT. rw-OOF 86.41 < lgbm_full 87.27.
   FULL matris (num+YIL+native-kategorik+missing-flag+txt_ridge_pred(nested-OOF)+lexicon). Tek-seed
   (zaman: thread=1 58s/fit cok yavas; thread=6 reproducible olcuwldu, 2 fit max|diff|=0 → kullanildi).
2. **Yil-agirlikli egitim (recency sample_weight) — HER IKI GBDT'de de ZARARLI:** lgbm_full_w 88.02,
   catboost_full_w 87.30 (ikisi de unweighted'tan KOTU). "Modeli LB dagilimina optimize et" sezgisi
   burada islemedi; eski yillari downweight etmek sinyal atiyor (iliski yillar arasi stabil). Bu
   modeller tek-model SUB DEGIL; blend cesitliligi icin tutuldu (ridge_pos catboost_full_w'ye 0.19,
   lgbm_full_w'ye ~0 agirlik verdi).
3. **ENSEMBLE (SUB-2):** OOF uzerinde recency-weighted NNLS / Ridge(positive) / greedy-forward.
   **DURUST degerlendirme:** her blend NESTED meta-CV ile puanlandi (agirliklar her hucre DISINDA fit →
   meta-overfit yok). Secilen = ridge_pos (tum 6 base, recency sample_weight). nested rw-OOF 85.49.
   Agirliklar: catboost_full 0.39, lgbm_num 0.29, catboost_full_w 0.19, lgbm_full 0.17, txt_ridge 0.04,
   lgbm_full_w 0.00, intercept -6.22. (greedy-NNLS 85.94; ridge_pos kazandi.)
4. **STEP2 spekulatif metin (word+char birlesik, `text_strong.py`):** standalone rw 163.89 (eski 168.02),
   ama corr(txt_ridge)=0.974 → blend'e yalniz -0.11 (85.49→85.38) = **0.25*std gurultu bandi icinde**.
   CLAUDE.md kabul kapisi marjinal ~0.1 kazanci REDDEDER (Occam/sifir-overfit) → nihai blend DISI.
   Artefakt+kod dokumantasyon icin repo'da; yeniden eklemek icin ensemble.py CANDIDATE_POOL'a ekle.

## Fold-safe / sizinti durumu
- Tum base'ler `cv.run_oof` + `data/folds.parquet` ile (15 fit, fold-ici). txt_ridge nested-OOF artefakti.
- recency_weights yalniz `graduation_year` (kovaryat) marjinalinden — HEDEFE dokunmaz → target sizintisi yok.
- Ensemble meta-CV ayni folds.parquet'ten; durust nested estimate.
- CatBoost determinizmi: random_seed + thread_count SABIT (6) — ayni script ayni sonuc (olculdu).
- Tum oof/test clip[0,100]; submission FORMAT assert'leri (10000 satir, ID birebir, NaN yok, [0,100]) GECTI.

## Uretilen dosyalar
- Kod: `src/lgbm_full_w.py`, `src/catboost_full.py`, `src/ensemble.py`, `src/finalize_submissions.py`,
  `src/text_strong.py`; `src/artifacts_io.py` (model_scores ledger).
- Artefakt: `artifacts/{oof,test}_{lgbm_full_w,catboost_full,catboost_full_w,blend,txt_ridge_wc}.npy`.
- Rapor: `reports/model_scores.csv` (karar defteri), `reports/ensemble_report.csv`, `reports/submissions_log.csv`.
- Submission: `submissions/sub1_catboost_full.csv`, `submissions/sub2_blend.csv`.

## Sabah icin oneriler (kullanici yapacak)
1. **Repro/consistency audit:** `python src/repro_test.py` (taze subprocess, internet kapali) — bilerek
   gece YAPILMADI. CatBoost thread_count=6'ya bagli; ayni makine/thread'de reproducible olmali.
2. **Submission butcesi:** istenirse SUB-1 + SUB-2 LB'ye cikip gap olc (sadece sensor; karar yine rw-OOF).
   Beklenti: lgbm_full rw 87.27 → gercek public 87.61 (gap ~+0.34); blend ~85.5 → public ~85.8-86 civari
   beklenir (ayni iliski tutarsa). public<<rw ise sizinti incele (beklenmiyor).
3. **Opsiyonel iyilestirmeler (dusuk oncelik, hepsi marjinal beklenir):**
   - CatBoost 3-seed averaging (gece tek-seed'e dusuruldu, zaman) — varyansi biraz azaltabilir.
   - 3. final guvenlik adayi olarak lgbm_full (en basit, kanitli 87.27) tutulabilir.
4. **DEGISTIRME:** karar metrigi DAIMA rw-OOF; public'in pesinden KOSMA. 2 final yapisal farkli, korunsun.
