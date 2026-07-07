# E5 EMBEDDING LEVER — multilingual-e5-large meta-feature (TIER-3) → **KABUL**

> **Amac:** `mentor_feedback_text`'ten transformer-tabanli yogun gomme (`intfloat/multilingual-e5-large`,
> 1024-dim) cikararak, mevcut TF-IDF→Ridge metin kanalindan (`txt_ridge`) daha zengin bir metin
> sinyali yakalamak ve blend'i iyilestirmek. **Karar metrigi = nested recency-weighted OOF-MSE (rw-OOF).**
> Public LB'ye BAKILMADI.
>
> **SONUC (TL;DR):** e5_ridge **KABUL edildi** ve **SUB-2 blend'e kalici eklendi**. Standalone rw-OOF
> **158.46** (tum onceki metin kanallarindan guclu). Blend'e katkisi **−0.648** (85.4945 → 84.8464).
> Literal kabul kapisini (−0.756) sayisal olarak gecmedi AMA kapinin std referansi bu paired
> karsilastirma icin yanlis olcut; **paired-test ile kazanc KESIN ANLAMLI** (15/15 hucre, t=−8.11,
> p=1.2e-6). 3 high-confidence skeptik + bagimsiz paired-dogrulama + kullanici onayi ile KABUL.

---

## A. URETIM — frozen embedding + fold-safe nested-OOF Ridge

| Asama | Detay |
|---|---|
| **Embedding** | `intfloat/multilingual-e5-large` (1024-dim), her metne `"query: "` prefix, `normalize_embeddings=True`. GPU/Colab'da uretildi (CPU'da OOM; bu makinede 15.8GB/~4GB bos). Cikti: `artifacts/emb_train.npy`, `emb_test.npy` (10000×1024 float32, **birim-norm** dogrulandi, finite, satir-distinct). |
| **Fold-safety** | Embedding **FROZEN + GLOBAL** → satir-bagimsiz, fold-leakage YAPISAL OLARAK IMKANSIZ. |
| **Meta-feature** | `e5_ridge` = `src/e5_ridge.py`. `text_utils.build_tfidf_ridge_oof`'un BIREBIR analogu: ayni `folds.parquet` dis-dongusu + nested inner-KFold(`random_state=SEED+r`), Ridge yalniz ic-train'de fit, dis-valid+test o fold'un inner modellerinin ortalamasi. clip[0,100]. |
| **Alpha** | repeat-0 fold-safe OOF taramasi (gate-kor, y'ye bakar). Grid embedding-uygun: `(0.1,0.3,1,3,10)`. Secilen **alpha=0.1** (repeat-0 OOF min: a=0.1→140.75). |

**Reproducibility:** `.npy` cache KANONIK artefakt — CV pipeline (`e5_ridge.py`, `ensemble.py`) bunlari yukler,
torch'a IHTIYAC DUYMAZ. torch yalniz `requirements-embed.txt` (ana `requirements.txt` KIRLENMEDI).
Embedding tekrar uretmek: `notebooks/colab_e5_embed.py` (GPU) veya `src/bert_embed.py` (CPU, chunk'li).

## B. STANDALONE — tum onceki metin kanallarindan guclu

| Metin kanali | standalone rw-OOF | not |
|---|---|---|
| **e5_ridge** (alpha=0.1) | **158.46** | frozen e5-large → nested-OOF Ridge |
| txt_rich (word1-3+char2-6) | 162.27 | onceki en iyi metin (FORENSICS, elendi) |
| txt_ridge (word1-2 TF-IDF) | 168.02 | mevcut blend metin kanali |
| txt_ridge_wc / txt_svd_gbdt | 163.89 / 182.66 | elenmis metin lever'leri |

corr(e5_ridge, txt_ridge) = **0.859** → tam redundant degil.

## C. BLEND KATKISI — −0.648, paired-test ile ANLAMLI

`ridge_pos` blend, e5_ridge havuza eklenince yeniden secildi:

| Konfig | blend nested rw-OOF | delta | e5 agirlik |
|---|---|---|---|
| base (e5'siz) | 85.4945 | — | — |
| **+e5_ridge (alpha=0.1)** | **84.8464** | **−0.648** | 0.1527 |
| +e5_ridge (alpha=1.0) | 84.8055 | −0.689 | 0.1724 |

e5_ridge, txt_ridge'in agirligini **0.0444 → 0.0000** ETTI ama bu **substitution DEGIL**:
- GBDT-only blend = 85.493; **+txt yalniz −0.001** (txt zaten redundant); **+e5 (txt'siz) = 84.862**
  → e5 GBDT-only uzerine **NET-YENI +0.631**.

**Secilen konfig: alpha=0.1** (veri-odakli repeat-0 OOF secimi; alpha=1.0 blend'de 0.04 daha iyi olsa da
onu secmek gate-chasing olurdu → reddedildi, Occam/sifir-overfit).

## D. KARAR — literal kapi vs paired-test olcutu

**Literal kabul kapisi (CLAUDE.md):** `yeni < eski − 0.25·std`, std = `compute_cv_mse` std = **3.0238**
→ band 0.756. `84.8464 > 84.7385` → literal olarak **GECMEZDI**.

**Itiraz (signal-vs-noise skeptik, bagimsiz dogrulandi):** Kapinin std'si (3.0238) blend'in **MUTLAK-MSE
seviye-varyansidir** (zorluk-varyansi). Biz ise **eslesmis (paired) model-vs-model deltasini** (−0.648)
yargiliyoruz. Paired delta'nin KENDI belirsizligi cok daha dar:

| Olcut | deger |
|---|---|
| 15 CV hucresinde iyilesen | **15/15** (hepsi negatif) |
| paired delta mean ± std | −0.6465 ± **0.3089** (4× daha siki band) |
| paired t (p) | **−8.11** (1.2e-6) |
| row-bootstrap %95 CI (B=5000) | **[−1.01, −0.29]** tamamen sifir-alti, P(Δ≥0)=0.0006 |

→ Kazanc **gurultu DEGIL, robust sinyal**. FORENSICS gurultu-tabani tezi BURADA GECERSIZ (o tez
GBDT-vs-GBDT artik uyumuyla ilgili; e5 **ortogonal frozen-embedding metin modalitesi**).

**Adversarial dogrulama (4 skeptik):** gate-aritmetigi (eleme literal dogru), fold-safety (−0.648
guvenilir, leakage YOK), alpha-overfit (alpha gate-kor, ceviremez) — 3'u high-conf "literal eleme
dogru"; signal-vs-noise (medium-conf) "ama band yanlis olcut, kazanc anlamli". Paired hesap **bagimsiz
yeniden uretildi** (yukaridaki t/p/CI birebir tutti).

**KABUL gerekcesi:** Kapinin AMACI gurultu-ici iyilesmeleri elemek; e5 kazanci paired testte gurultu
DEGIL. Kullanici onayi ile **KABUL** → e5_ridge kalici blend uyesi.

**Durustluk caveat'lari (kayit icin):** (1) mutlak kazanc kucuk (~0.65, base'in ~%0.76'si); (2) alpha
post-hoc 1.0→0.1 secimi sonucu ~0.04 oynatir (minor researcher freedom) — ikisi de paired-anlamliligi
bozmaz. (3) Karar literal kapiyi DEGIL, kapinin daha-dogru paired-yorumunu kullandi; bu tek seferlik
ve belgelidir.

## E. FINAL ETKISI

| Final | Onceki | Yeni |
|---|---|---|
| **SUB-1** (safe tek-GBDT) | catboost_full (86.41) | **catboost_full (86.41)** — DEGISMEDI (sade, e5'siz; risk dagitimi) |
| **SUB-2** (ensemble blend) | blend 85.4945 | **blend 84.8464** (e5'li) |

İki final hala yapisal farkli (tek-model vs ensemble; |fark|≈1.03) → private %40 bolmesine karsi risk
dagitimi korundu.

---

### Artefaktlar / kanit
- `src/bert_embed.py` (CPU chunk'li), `notebooks/colab_e5_embed.py` (GPU), `requirements-embed.txt` (izole).
- `artifacts/emb_{train,test}.npy` (kanonik, .gitignore degil — kucuk 40MB; model `models/` gitignore).
- `src/e5_ridge.py` → `artifacts/{oof,test}_e5_ridge.npy`; `reports/model_scores.csv` (e5_ridge + blend satirlari).
- `src/ensemble.py` CANDIDATE_POOL'da e5_ridge (kabul notu inline).
- `submissions/sub2_blend.csv` (e5'li blend), `submissions/sub1_catboost_full.csv`.
