"""
Faz 6 — ENSEMBLE: base OOF uzerinde recency-weighted NNLS / Ridge(positive) blend.
==================================================================================

    python src/ensemble.py

KARAR METRIGI = recency_weighted_oof_mse (review H1). META AGIRLIKLARI recency sample_weight ile
fit edilir (sample_weight=recency); blend KARARI recency-weighted OOF'a gore.

DURUST (optimism-yok) DEGERLENDIRME — KRITIK:
  Blend agirliklarini TUM OOF'a fit edip AYNI OOF'ta puanlamak iyimser (meta-overfit). Bunun
  yerine her blend NESTED meta-CV ile puanlanir: her (repeat,fold) hucresi icin agirliklar
  o hucre DISINDAKI OOF satirlarindan fit edilir, hucre tahmin edilir -> meta_oof (3-repeat avg).
  rw-OOF(meta_oof) = blend'in DURUST karar skoru. Final TEST tahmini icin agirliklar tum OOF'a
  fit edilir (held-out yok) -> standart stacking. Saklanan oof_blend.npy = NESTED meta_oof
  (ledger rw-OOF ile birebir tutarli).

SECIM: {NNLS-full, Ridge(pos)-full, greedy-forward(NNLS)} arasindan EN DUSUK nested rw-OOF.
  Bir model blend'e ancak nested rw-OOF'u dusururse girer (greedy bunu zorlar; NNLS faydasiza
  ~0 agirlik verir). Esitlikte daha az model (Occam).

FOLD-SAFE: base OOF'lar zaten nested/fold-ici uretildi; meta sadece bu OOF kolonlarini birlestirir,
  meta-CV split'i ayni folds.parquet'ten (repeat-fold). Hedef sizintisi yok.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

import artifacts_io as aio
import cv

# Aday havuzu (mevcut olanlar otomatik secilir). txt_ridge zayif ama NNLS ~0 agirlik verebilir.
CANDIDATE_POOL = [
    "lgbm_full", "lgbm_num", "lgbm_full_w",
    "catboost_full", "catboost_full_w", "txt_ridge",
    "e5_ridge",  # TIER-3 KABUL EDILDI (asagidaki nota bak) -> kalici blend uyesi.
    "mm",        # TIER-3 KABUL EDILDI (asagidaki MM nota bak) -> kalici blend uyesi.
    "lgbm_full_h",  # TIER-3 KABUL EDILDI (asagidaki HUBER nota bak) -> kalici blend uyesi.
    "lgbm_full_ht",  # TIER-3 KABUL EDILDI (asagidaki HT nota bak) -> kalici blend uyesi.
    "xlmr",         # BIRLESME-T2 KABUL EDILDI (asagidaki XLMR nota bak) -> kalici blend uyesi.
    "ourteam_tf",   # BIRLESME-T3 KABUL EDILDI (asagidaki OURTEAM_TF nota bak) -> kalici blend uyesi.
]
# BIRLESME-T3 KABUL NOT (ourteam_tf): Ahmet'in TAM cozumu (lgbm+xgb+cat+mlp+nn NNLS blend), BIZIM
# folds.parquet (RepeatedStratified 5x3) ile YENIDEN EGITILDI (ahmettengelenler/ourteam_oof_tunafolds.npy;
# eski KFold-10 ourteam_oof DEGIL -> fold-hizasizlik COZULDU). HIZALAMA DOGRULANDI: y satir-bazli
# OZDES + Ahmet'in application_year'i bizimkiyle %100 esit (satir sirasi tam hizali; src/
# probe_ahmet_tunafolds.py). Eski KFold-10 GERCEKTEN iyimser-yanliymis: standalone rw 83.5577 ->
# fold-hizali 84.2320 (+0.67 sisme; tunadeniz itirazi DOGRULANDI). AMA kazanc fold-artefakti DEGIL:
# fold-hizali DURUST OOF'la BILE 11-model blend 83.6286 -> +ourteam_tf 82.2398 (delta -1.3888).
# PAIRED-ANLAMLI (kendi gate'imiz, mm/e5/xlmr ile AYNI olcut): 15/15 hucre, t=-9.885, p=1.08e-7,
# 5000-bootstrap %95 CI [-1.9358,-0.8391] (tamamen sifir-alti). Onceki RED (eski OOF, 11/15 p=0.137)
# ARTIK GECERSIZ -> fold-hizalama farki. corr(ourteam_tf, xlmr)=0.823 / blend11=0.982 ama ORTOGONAL
# katki net (Ahmet'in farkli model ailesi: lgbm/xgb/cat/mlp/nn + segment-yil TE + kohort-z FE).
# team_blend_v2 public 82.3678 (eski sisik OOF'la bile gap +0.064 YESIL) bu kalibrasyonu ON-TEYIT etti.
# SUB-1 (catboost_full) dokunulmaz. REPRO: ourteam_tf .npy KANONIK artefakt (Ahmet GPU/NN; belgelenmis
# tolerans). Detay: BIRLESME_YOL_HARITASI.md §10.
# BIRLESME-T2 KABUL NOT (xlmr): Ahmet'in en iyi metin modeli XLM-R-large TEXT-ONLY, BIZIM fold-safe
# altyapida (folds.parquet repeat-0, 5-fit) sifirdan nested-OOF uretildi (colab_xlmr.ipynb; ham OOF
# karistirma DEGIL -> fold-hizasizlik/winner's-curse YOK). standalone rw 140.72 (metin-tek; unw-CV
# 126.07 = Ahmet gunlugu 126.1 ile birebir). ORTOGONAL: corr(mm)=0.780 / corr(e5)=0.883 /
# corr(txt_ridge)=0.843 -> mm (kendisi de XLM-R) ile ayrisik cunku text-only (mm = text+tabular).
# BLEND: 10-model 84.0212 -> +xlmr 83.6286 (delta -0.3925). PAIRED-ANLAMLI (src/xlmr_gate.py,
# mm/e5 ile AYNI olcut): 13/15 hucre, t=-3.005, p=9.45e-3, 5000-bootstrap %95 CI [-0.7106,-0.0845]
# (tamamen sifir-alti, P(delta>=0)=0.0066). lgbm_num_h emsali (11/15, CI sifiri kapsar) ile NET
# AYRISIR -> KABUL. Literal 0.25*std (0.707) gecmez ama paired olcut esas (e5/mm ayni gerekce).
# Birlesme yol haritasi Tier-1 (cohort-z/segment-TE/xgb) corr ~0.99 REDDEDILMISTI; XLM-R ilk gercek
# kazanc (farkli fonksiyon sinifi + ortogonal metin). SUB-1 (catboost_full) dokunulmaz. REPRO: mm gibi
# belgelenmis tolerans (neural/GPU). Detay: BIRLESME_YOL_HARITASI.md §8.
# TIER-3 KABUL NOT (lgbm_full_ht): lgbm_full_h + SIKI regularizasyon (num_leaves=15, min_child=80;
# src/lgbm_full_ht.py). ONCEDEN-KAYITLI 12-konfig HP gridinin (gate-kor, tek-yon degisimler,
# repeat-0 fold-safe) tek anlamli kazanani; post-hoc kombinasyon YAPILMADI. Full-15'te curume yok
# (repeat-0 -0.53 -> full -0.54) -> etki gercek. Standalone rw 85.7810 = EN DUSUK tek-model
# (lgbm_full_h 86.32, catboost_full 86.41). Blend EKLE 84.0991 -> 84.0212 (-0.078); paired-anlamli:
# 13/15 hucre, t=-4.211, p=8.7e-4, bootstrap %95 CI [-0.1552,-0.0019] (ust sinir INCE ama sifir-alti;
# kiyas: ayni buyuklukteki lgbm_num_h -0.074 tutarsizliktan dustu, 11/15 p=0.012 CI sifiri kapsar).
# IKAME (h->ht, -0.051) olculup elendi; EKLE secildi (ridge redundansi yonetir). SUB-1 DEGISMEZ
# (catboost_full; sigorta=maksimum-bagimsiz aile, finalize _h/_ht dislamasi). Gece vardiyasinin
# diger 7 mekanizma sondasi (tweedie/GLS/cens-obj/dart/seedbag/lgbm_num_h/tabpfn-lokal) ELENDI
# (reports/CEILING_AUDIT.md gece bolumu). Detay: reports/ROBUST_LOSS_LEVER.md.
# TIER-3 KABUL NOT (lgbm_full_h): lgbm_full'un Huber(alpha=5) robust-loss varyanti (src/lgbm_full_h.py).
# MEKANIZMA: alt-kuyruk surprizleri (ex-ante ayirt edilemez dusuk-y satirlar; iki-asama Bayes kaniti
# reports/LOW_TAIL_LEVER.md) L2 egitiminde karesel gradyanla kutlenin fit'ini zehirliyor; Huber bunu
# gradyan-kapatmayla engeller. Standalone rw 86.3222 (L2 ikizi 87.2663, -0.944; en dusuk tek-model rw).
# Blend katkisi 84.2393 -> 84.0991 (-0.140); kucuk AMA PAIRED-ANLAMLI (13/15 hucre, t=-3.825,
# p=1.9e-3, 5000-ornek bootstrap %95 CI [-0.264,-0.018] tamamen sifir-alti) — yil-etkilesimi gibi
# gurultu-bandi adaylarin (p=0.094, CI sifiri kapsar) reddedildigi AYNI testten net ayrisir.
# Ekleme-vs-ikame olculdu: ekle 84.0991 / lgbm_full yerine ikame 84.1099 (es; additive e5/mm emsali).
# CATBOOST-HUBER REDDEDILDI: Huber:delta=5 rw 104.18 / delta=8 rw 107.98 (L2 86.41'e karsi felaket;
# CatBoost Huber implementasyonu bu butcede underfit) -> kurtarma denemesi HP-balikciligi olur, yok.
# SUB-1 NOT: lgbm_full_h (86.32) vs catboost_full (86.41) farki paired'de GURULTU (7/15, t=-0.44,
# p=0.67) -> esitlikte yapisal-farkli incumbent kazanir (CLAUDE.md); SUB-1 catboost_full KALIR
# (finalize_submissions _h dislamasi). Detay: reports/ROBUST_LOSS_LEVER.md.
# TIER-3 KABUL NOT (mm): XLM-R-large + tabular NN MULTIMODAL (FARKLI fonksiyon sinifi; GPU/Colab
# colab_mm_multimodal.ipynb, bizim folds.parquet repeat-0 5-fit). Standalone unw-OOF 83.30 / rw-OOF
# 94.82 (tek basina catboost_full 86.41'den zayif AMA ORTOGONAL: corr e5=0.727, txt=0.690; GBDT'lere
# 0.945 ama ozdes degil -> blend'e NET-YENI sinyal). Havuza eklenince ridge_pos blend nested rw-OOF
# 84.8464 -> 84.2393 (delta -0.6071). LITERAL kapi (0.25*std=0.7528) GECMEZDI; ANCAK e5 ile AYNI
# gerekce: o std blend'in MUTLAK-MSE seviye-varyansi (yanlis olcut), PAIRED model-vs-model delta'nin
# kendi belirsizligi cok daha dar. Paired testte mm kazanci KESIN ANLAMLI (src/mm_gate.py):
# 15/15 CV hucresi iyilesti, paired t=-5.192 (p=1.36e-4), 5000-ornek row-bootstrap %95 CI
# [-1.0154,-0.2070] tamamen sifir-alti (P(delta>=0)=0.0020). e5 kabulu (15/15, t=-8.11) ile AYNI
# kalite kanit. FARK: e5 frozen-embedding+Ridge metin kanali; mm ORTOGONAL neural multimodal sinifi
# -> forensics "GBDT noise-floor" tezini FARKLI fonksiyon sinifiyla kirdi. Karar: paired-dogrulama +
# kullanici onayi -> KABUL. SUB-2 blend artik e5_ridge + mm ICERIR. SUB-1 (catboost_full) dokunulmaz.
# REPRO: mm bit-deterministik DEGIL (neural/GPU/cuDNN/bf16) -> SUB-2 'belgelenmis tolerans' (bit-ayni
# degil); oof_mm/test_mm .npy KANONIK artefakt. Detay: reports/MM_MULTIMODAL_LEVER.md.
# TIER-3 KABUL NOT (e5_ridge): FROZEN multilingual-e5-large (1024-dim) -> nested-OOF Ridge(alpha=0.1,
# src/e5_ridge.py; emb GPU/Colab artifacts/emb_*.npy). Standalone rw-OOF 158.46 (tum onceki metin
# kanallarindan GUCLU: txt_ridge 168.02, txt_rich 162.27). Havuza eklenince ridge_pos blend nested
# rw-OOF 85.4945 -> 84.8464 (delta -0.648; e5 agirlik 0.1527, txt_ridge'i 0.0444->0.0000 ETTI).
# LITERAL KABUL KAPISI (0.25*std, std=3.0238 -> band 0.756) ile -0.648 GECMEZDI. ANCAK kapinin std'si
# blend'in MUTLAK-MSE seviye-varyansidir (yanlis olcut); PAIRED model-vs-model delta'nin kendi std'si
# 0.309 (4x daha siki). Paired testte e5 kazanci KESIN ANLAMLI: 15/15 CV hucresi iyilesti, paired
# t=-8.11 (p=1.2e-6), 5000-ornek row-bootstrap %95 CI [-1.01,-0.29] tamamen sifir-alti. Substitution
# DEGIL: GBDT-only blend 85.493 -> +e5 84.862 (NET-YENI +0.631), +txt yalniz -0.001 (txt zaten
# redundant). FORENSICS gurultu-tabani tezi BURADA GECERSIZ (o GBDT-vs-GBDT artigi; e5 ortogonal
# frozen-embedding metin modalitesi). Karar: 3 high-conf skeptik + bagimsiz paired-dogrulama + kullanici
# onayi -> KABUL. SUB-2 blend artik e5_ridge ICERIR. Detay: reports/E5_EMBEDDING_LEVER.md.
# FORENSICS NOT: txt_rich (zengin TF-IDF word(1-3)+char(2-6) nested-OOF metin, src/text_rich.py)
# DENENDI (C3 reversal: adversarial-verify "metin redundant" tezini curuttu -> metin etkilesimli +
# mevcut txt_ridge metni doyurmamis, zengin TF-IDF num tabaninda -0.87 uw marjinal verdi). Standalone
# rw-OOF 162.27 (txt_ridge 168.02'den GUCLU). AMA havuza eklenince ridge_pos blend nested rw-OOF
# 85.4945->85.4116 (-0.083; txt_ridge'i 0.099 agirlikla degistirir, net ~0) = 0.25*std (~0.76) GURULTU
# BANDI ICINDE -> REDDEDILDI (Occam/kabul-kapisi, nihai blend DISI). GBDT etkilesimleri metin sinyalini
# zaten yakaliyor. Artefakt+ledger dokuman icin tutuldu. Yeniden denemek: asagiya "txt_rich" ekle.
# LEVER3 NOT: txt_svd_gbdt (TF-IDF -> fold-ici TruncatedSVD(80) -> LGBM, src/txt_svd_gbdt.py)
# DENENDI. Standalone rw-OOF 182.66 (txt_ridge 168.02'den ZAYIF; SVD %19 varyans -> kayipli).
# Havuza eklenince blend nested rw-OOF 85.4945->85.5123 (+0.02, IYILESME YOK; txt_ridge agirligini
# yiyor, corr 0.886 fazla redundant) -> REDDEDILDI (Occam/kabul-kapisi, nihai blend DISI). Artefakt
# + ledger dokuman icin tutuldu. Yeniden denemek: yukaridaki listeye "txt_svd_gbdt" ekle.
# LEVER1 NOT: histgbr_full (HistGradientBoosting, UCUNCU GBDT ailesi, src/histgbr_full.py) DENENDI.
# Standalone rw-OOF 88.54 (lgbm_full 87.27 / catboost_full 86.41'den ZAYIF). Havuza eklenince
# ridge_pos blend ona 0.0000 AGIRLIK verdi ve nested rw-OOF 85.4945->85.5560 (+0.06, GURULTU,
# IYILESME YOK). 0.25*std (~0.76) kapisindan geri donmek soyle dursun blend'i KOTULESTIRDI ->
# REDDEDILDI (Occam + sifir-overfit). Artefakt (oof/test_histgbr_full.npy) + ledger satiri
# dokumantasyon icin tutuldu. Yeniden denemek: yukaridaki listeye "histgbr_full" ekle.
# STEP2 NOT: txt_ridge_wc (word+char birlesik metin, text_strong.py) DENENDI ve havuza eklenince
# blend nested rw-OOF yalniz 85.49->85.38 (-0.11) dustu = 0.25*std (~0.75) GURULTU BANDI ICINDE
# (corr(txt_ridge)=0.974; sadece txt_ridge'i degistirip 0.09 agirlik aldi). CLAUDE.md kabul kapisi
# marjinal ~0.1 MSE kazancini REDDEDER (Occam + sifir-overfit) -> nihai blend'e ALINMADI. Artefakt
# + ledger satiri dokumantasyon icin tutuldu. Yeniden eklemek: yukaridaki listeye "txt_ridge_wc" ekle.
ENSEMBLE_REPORT_PATH = cv.REPORTS_DIR / "ensemble_report.csv"
GREEDY_EPS = 1e-6  # iyilesme bu kadar bile yoksa modeli ekleme (Occam)


# --------------------------------------------------------------------------- #
# Meta-combiner: recency-weighted NNLS / Ridge(positive)
# --------------------------------------------------------------------------- #
def fit_weights(P, y, w, method: str):
    """Doner: predict(Q)-fonksiyonu. method 'nnls' (intercept yok) | 'ridge_pos' (intercept var)."""
    sw = np.sqrt(np.asarray(w, dtype=float))
    if method == "nnls":
        A = P * sw[:, None]
        b = np.asarray(y, dtype=float) * sw
        coef, _ = nnls(A, b)
        return lambda Q: Q @ coef, coef, 0.0
    if method == "ridge_pos":
        r = Ridge(alpha=1.0, positive=True, fit_intercept=True)
        r.fit(P, y, sample_weight=w)
        coef, intc = r.coef_.copy(), float(r.intercept_)
        return lambda Q: Q @ coef + intc, coef, intc
    raise ValueError(method)


def nested_rw_oof(P, y, w, folds, sid, method: str) -> tuple[float, np.ndarray]:
    """3-repeat nested meta-CV -> (rw-OOF, meta_oof). Agirliklar her hucre DISINDA fit (durust)."""
    n = len(y)
    s = np.zeros(n)
    c = np.zeros(n)
    for r in range(cv.N_REPEATS):
        fold_of = cv.fold_of_rows(folds, sid, r)
        for g in range(cv.N_SPLITS):
            va = np.where(fold_of == g)[0]
            tr = np.where(fold_of != g)[0]
            pred_fn, _, _ = fit_weights(P[tr], y[tr], w[tr], method)
            s[va] += pred_fn(P[va])
            c[va] += 1.0
    assert np.all(c == cv.N_REPEATS)
    meta_oof = cv.clip_predictions(s / c)
    return cv.compute_recency_weighted_mse(meta_oof, y, w), meta_oof


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    # --- mevcut adaylari topla + (eksikse) ledger'a backfill ---
    cands = []
    for m in CANDIDATE_POOL:
        p_oof = cv.ARTIFACTS_DIR / f"oof_{m}.npy"
        p_te = cv.ARTIFACTS_DIR / f"test_{m}.npy"
        if p_oof.exists() and p_te.exists():
            cands.append(m)
    assert "lgbm_full" in cands, "lgbm_full base bulunamadi."

    oof_mat = {m: np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy") for m in cands}
    test_mat = {m: np.load(cv.ARTIFACTS_DIR / f"test_{m}.npy") for m in cands}

    print("[ensemble] adaylar (tek-model rw-OOF):")
    single_rw = {}
    for m in cands:
        rw_m = cv.compute_recency_weighted_mse(oof_mat[m], y, w)
        cm, cs, _ = cv.compute_cv_mse(oof_mat[m], y, folds, sid)
        single_rw[m] = rw_m
        # ledger backfill (weighted_training=_w son ekli; en iyi caba)
        aio.log_model_score(m, cm, cs, rw_m, weighted_training=m.endswith("_w"),
                            note="base (ensemble backfill)")
        print(f"  {m:16s} rw-OOF={rw_m:8.4f}  unweighted_cv={cm:8.4f}")

    best_single = min(single_rw, key=single_rw.get)
    best_single_rw = single_rw[best_single]
    print(f"[ensemble] EN IYI TEK MODEL: {best_single} (rw-OOF {best_single_rw:.4f})")

    P_all = np.column_stack([oof_mat[m] for m in cands])
    T_all = np.column_stack([test_mat[m] for m in cands])

    report_rows = []

    def eval_blend(name, cols, method):
        idx = [cands.index(c) for c in cols]
        P = P_all[:, idx]
        rw, meta_oof = nested_rw_oof(P, y, w, folds, sid, method)
        # final agirliklar (tum OOF) -> test
        pred_fn, coef, intc = fit_weights(P, y, w, method)
        report_rows.append(dict(
            blend=name, method=method, models="+".join(cols),
            weights=";".join(f"{c}={wt:.4f}" for c, wt in zip(cols, np.atleast_1d(coef))),
            intercept=round(float(intc), 4), nested_rw_oof=round(float(rw), 6),
        ))
        return rw, meta_oof, (idx, coef, intc, method)

    # 1) NNLS-full , 2) Ridge(pos)-full
    results = []
    rw1, oof1, cfg1 = eval_blend("nnls_full", cands, "nnls")
    results.append(("nnls_full", rw1, oof1, cfg1))
    rw2, oof2, cfg2 = eval_blend("ridge_pos_full", cands, "ridge_pos")
    results.append(("ridge_pos_full", rw2, oof2, cfg2))

    # 3) greedy forward selection (NNLS) — model ancak nested rw-OOF'u dusururse girer
    selected = [best_single]
    cur_rw, cur_oof, cur_cfg = eval_blend("greedy_step", selected, "nnls")
    improved = True
    while improved:
        improved = False
        best_add, best_add_rw, best_pack = None, cur_rw - GREEDY_EPS, None
        for m in cands:
            if m in selected:
                continue
            trial = selected + [m]
            rw_t, oof_t, cfg_t = eval_blend(f"greedy_try_{m}", trial, "nnls")
            if rw_t < best_add_rw:
                best_add, best_add_rw, best_pack = m, rw_t, (oof_t, cfg_t, trial)
        if best_add is not None:
            selected = best_pack[2]
            cur_rw, cur_oof, cur_cfg = best_add_rw, best_pack[0], best_pack[1]
            improved = True
            print(f"[ensemble] greedy + {best_add} -> nested rw-OOF {cur_rw:.4f}")
    results.append(("greedy_nnls", cur_rw, cur_oof, cur_cfg))
    print(f"[ensemble] greedy secilen: {'+'.join(selected)} (rw-OOF {cur_rw:.4f})")

    # --- en dusuk nested rw-OOF blend'i sec ---
    best_name, best_rw, best_oof, best_cfg = min(results, key=lambda t: t[1])
    idx, coef, intc, method = best_cfg
    print(f"[ensemble] >>> SECILEN BLEND: {best_name} (method={method}) nested rw-OOF={best_rw:.4f}  "
          f"(en iyi tek model {best_single} {best_single_rw:.4f}; delta {best_rw - best_single_rw:+.4f})")

    # final test tahmini = tum-OOF agirliklari * test
    P_sel = P_all[:, idx]
    T_sel = T_all[:, idx]
    pred_fn, coef_f, intc_f = fit_weights(P_sel, y, w, method)
    blend_test = cv.clip_predictions(pred_fn(T_sel))
    blend_oof = best_oof  # NESTED meta_oof (durust; ledger ile tutarli)

    sel_models = [cands[i] for i in idx]
    print(f"[ensemble] final agirliklar ({method}): "
          + ", ".join(f"{m}={wt:.4f}" for m, wt in zip(sel_models, np.atleast_1d(coef_f)))
          + (f", intercept={intc_f:.4f}" if method == "ridge_pos" else ""))

    # --- artefaktlar ---
    aio.save_oof_test("blend", blend_oof, blend_test)
    blend_cv_mean, blend_cv_std, _ = cv.compute_cv_mse(blend_oof, y, folds, sid)
    aio.write_cv_score("blend", blend_cv_mean, blend_cv_std, 0.0)  # cv_scores.csv blend satiri (finalize okur)
    note = (f"blend={best_name} method={method} models={'+'.join(sel_models)} "
            f"nested_rw_oof={best_rw:.4f} (durust). weights="
            + ";".join(f"{m}={wt:.4f}" for m, wt in zip(sel_models, np.atleast_1d(coef_f))))
    aio.log_model_score("blend", blend_cv_mean, blend_cv_std, best_rw,
                        weighted_training=False, note=note)
    cv.assert_in_range(blend_oof, "oof_blend")
    cv.assert_in_range(blend_test, "test_blend")

    pd.DataFrame(report_rows).to_csv(ENSEMBLE_REPORT_PATH, index=False)
    print(f"[ensemble] yazildi: artifacts/oof_blend.npy, test_blend.npy, {ENSEMBLE_REPORT_PATH.name}")
    print(f"[ensemble] blend unweighted_cv={blend_cv_mean:.4f}  nested_rw_oof={best_rw:.4f}")


if __name__ == "__main__":
    main()
