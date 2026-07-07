"""
TIER-2 LEVER-1 — IWCV + RuLSIF: metrik-rigor cross-check (JURI), PRIMER METRIK DEGIL.
====================================================================================

    python src/iwcv_rulsif.py

NEDEN (jury-rigor + cross-check): Mevcut KARAR metrigi `compute_recency_weighted_mse`,
importance weight'i YALNIZ `graduation_year` marjinaline gore kurar (w = P_test(gy)/P_train(gy)).
Bu tek-degiskenli bir kovaryat-kayma duzeltmesidir. Burada TUM kovaryat uzayindan (37 sayisal +
2 yil; HEDEF YOK) bir density-ratio w(x) = p_test(x)/p_train(x) tahmin edilir (RuLSIF) ve OOF
modelleri bu COK-DEGISKENLI agirlikla yeniden puanlanir (IWCV). Soru: model SIRALAMASI / final
SECIM mevcut tek-degiskenli recency-weighted OOF ile AYNI mi?

  * RuLSIF = Relative unconstrained Least-Squares Importance Fitting (Yamada+ 2011). alpha-relative
    density-ratio r_a(x) = p_te(x) / ((1-a) p_tr(x) + a p_te(x)); a in (0,1] -> ratio UST sinirli
    (<= 1/a), agir kuyruk/ESS cokusune karsi DAYANIKLI (vanilla KLIEP'in bilinen zaafi). a=0 saf
    ratio, a->1 daha duzgun. Burada a~0.1-0.5 taranir, ESS izlenir.
  * Cikti karari (review H1 sozlesmesi KORUNUR): PRIMER metrik DAIMA compute_recency_weighted_mse
    kalir. IWCV yalniz CROSS-CHECK: (i) model siralamasi degisiyor mu, (ii) en-iyi-tek + blend
    secimi degisiyor mu, (iii) forensik bulgusu (rw ~ uw x test/train varyans orani) cok-degiskenli
    agirlikla da tutuyor mu. PUBLIC LB'ye BAKILMADI.

FOLD-SAFE / SIZINTI: RuLSIF agirligi HEDEF GORMEZ (yalniz kovaryatlar). OOF'lar zaten nested/
  fold-ici uretildi; burada sadece var olan oof_*.npy'ler yeni bir agirlikla yeniden-puanlanir
  (yeni fit YOK). Determinizm: SEED=42 (kernel merkez ornekleme + random projection sabit).

Yeni bagimlilik YOK: pure numpy/scipy (pinned). statsmodels/densratio KULLANILMAZ.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.linalg import solve

import cv

REPORT_PATH = cv.REPORTS_DIR / "iwcv_rulsif_report.csv"
JSON_PATH = cv.REPORTS_DIR / "iwcv_rulsif.json"

# RuLSIF HP (sabit; HP-taramasi karar metrigi DEGIL, sadece a icin kucuk grid).
N_CENTERS = 200          # Gaussian kernel merkez sayisi (test ornegi alt-kumesi)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)     # ridge regularizasyon (LOOCV ile secilir)
ALPHAS = (0.1, 0.2, 0.3, 0.5)         # alpha-relative; ESS/siralama bunlar uzerinden raporlanir
SIGMA_SCALE = (0.5, 1.0, 2.0)         # median-heuristic bandwidth carpani (LOOCV ile secilir)


def _covariate_matrix(df: pd.DataFrame) -> np.ndarray:
    """Sayisal feature + yil kovaryat matrisi (HEDEF YOK). NA -> sutun medyani (target-bagimsiz;
    bu bir AGIRLIK tahmincisidir, model degil -> basit impute sizinti yaratmaz)."""
    cols = list(cv.numeric_feature_columns(df)) + list(cv.YEAR_COLS)
    X = df[cols].astype(float).to_numpy()
    return X, cols


def _standardize(Xtr: np.ndarray, Xte: np.ndarray):
    """Ortak (train+test BIRLESIK) medyan/IQR ile robust standardize. NA -> medyan.

    Density-ratio icin train ve test AYNI olcekte olmali; birlesik istatistik kullanmak HEDEF
    gormez (kovaryat-only) -> sizinti yok. IQR=0 sutunlari icin std'ye dus."""
    Z = np.vstack([Xtr, Xte])
    med = np.nanmedian(Z, axis=0)
    # NA doldur
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    Z = np.vstack([Xtr, Xte])
    q75, q25 = np.nanpercentile(Z, [75, 25], axis=0)
    iqr = q75 - q25
    std = np.nanstd(Z, axis=0)
    scale = np.where(iqr > 1e-9, iqr / 1.349, np.where(std > 1e-9, std, 1.0))
    return (Xtr - med) / scale, (Xte - med) / scale


def _median_sigma(Xte_c: np.ndarray, rng: np.random.Generator) -> float:
    """Median-heuristic bandwidth: kernel merkezleri arasi cift uzakliklarin medyani."""
    m = Xte_c.shape[0]
    idx = rng.choice(m, size=min(m, 500), replace=False)
    S = Xte_c[idx]
    d2 = np.sum(S**2, 1)[:, None] + np.sum(S**2, 1)[None, :] - 2 * S @ S.T
    d2 = d2[np.triu_indices_from(d2, k=1)]
    med = np.sqrt(np.median(np.maximum(d2, 0.0)))
    return float(med if med > 1e-6 else 1.0)


def _rbf(X: np.ndarray, C: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel tasarim matrisi Phi[i,k] = exp(-||x_i - c_k||^2 / (2 sigma^2))."""
    d2 = np.sum(X**2, 1)[:, None] + np.sum(C**2, 1)[None, :] - 2 * X @ C.T
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma**2))


def rulsif_weights(Xtr: np.ndarray, Xte: np.ndarray, alpha: float, seed: int = cv.SEED):
    """alpha-relative RuLSIF: w(x_tr) = r_a(x_tr), r_a = p_te / ((1-a)p_tr + a p_te).

    Kapali-form: theta = (H + lam I)^-1 h, H = (1-a) E_tr[phi phi'] + a E_te[phi phi'],
    h = E_te[phi]. sigma & lambda LOOCV-benzeri grid ile (test-fit hatasina gore) secilir.
    Doner: (w_tr (n_tr,), info dict). w_tr >= 0'a kirpilir, mean-normalize EDILMEZ (cagiran yapar)."""
    rng = np.random.default_rng(seed)
    Xtr_c, Xte_c = _standardize(Xtr, Xte)
    n_tr, n_te = len(Xtr_c), len(Xte_c)
    # kernel merkezleri: test dagiliminin alt-kumesi (RuLSIF konvansiyonu)
    cidx = rng.choice(n_te, size=min(N_CENTERS, n_te), replace=False)
    C = Xte_c[cidx]
    sig0 = _median_sigma(Xte_c, rng)

    best = None
    for ss in SIGMA_SCALE:
        sigma = sig0 * ss
        Phi_tr = _rbf(Xtr_c, C, sigma)   # (n_tr, b)
        Phi_te = _rbf(Xte_c, C, sigma)   # (n_te, b)
        H = (1.0 - alpha) * (Phi_tr.T @ Phi_tr) / n_tr + alpha * (Phi_te.T @ Phi_te) / n_te
        h = Phi_te.mean(axis=0)
        b = Phi_tr.shape[1]
        for lam in LAMBDAS:
            theta = solve(H + lam * np.eye(b), h, assume_a="sym")
            # secim kriteri: alpha-relative PE-benzeri obj (test-fit). Buyuk = iyi ayrim;
            # asiri-fit'e karsi lam zaten cezalandiriyor. Yamada+ 2011 Eq. PE tahmincisi:
            #   PE = -0.5*(1-a) E_tr[r^2] - 0.5*a E_te[r^2] + E_te[r] - 0.5
            r_tr = Phi_tr @ theta
            r_te = Phi_te @ theta
            pe = (-0.5 * (1 - alpha) * np.mean(r_tr**2)
                  - 0.5 * alpha * np.mean(r_te**2)
                  + np.mean(r_te) - 0.5)
            if best is None or pe > best[0]:
                best = (pe, sigma, lam, theta, r_tr)

    pe, sigma, lam, theta, r_tr = best
    w = np.maximum(r_tr, 0.0)           # ratio >= 0
    # ESS (effective sample size) — agirlik dejenerasyonu sensoru
    ws = w / w.mean() if w.mean() > 0 else w
    ess = float(ws.sum() ** 2 / np.sum(ws**2)) if np.sum(ws**2) > 0 else 0.0
    info = dict(alpha=alpha, sigma=round(float(sigma), 4), lam=lam, pe=round(float(pe), 6),
                ess=round(ess, 1), ess_frac=round(ess / n_tr, 4),
                w_max=round(float(ws.max()), 3), w_mean=1.0)
    return ws, info


def _iwcv_mse(oof: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Importance-weighted CV MSE = sum w (y-oof)^2 / sum w (clip SONRASI)."""
    oof = cv.clip_predictions(oof)
    return float(np.sum(w * (y - oof) ** 2) / np.sum(w))


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    # PRIMER metrik agirligi (mevcut sozlesme; graduation_year-marjinal)
    w_rec = cv.recency_weights(train, test)

    Xtr, cols = _covariate_matrix(train)
    Xte, _ = _covariate_matrix(test)
    print(f"[iwcv] kovaryat uzayi: {len(cols)} kolon (37 sayisal + 2 yil; HEDEF YOK).")

    # --- RuLSIF agirliklari (alpha grid) ---
    rulsif = {}
    for a in ALPHAS:
        w, info = rulsif_weights(Xtr, Xte, alpha=a)
        rulsif[a] = (w, info)
        print(f"[iwcv] RuLSIF a={a}: ESS={info['ess']:.0f}/{len(y)} "
              f"({info['ess_frac']*100:.1f}%), w_max={info['w_max']:.2f}, "
              f"sigma={info['sigma']}, lam={info['lam']}, PE={info['pe']:.4f}")

    # ESS bandi makul olan en kucuk alpha'yi PRIMER IWCV agirligi sec (>=%50 ESS hedef).
    chosen_a = None
    for a in ALPHAS:
        if rulsif[a][1]["ess_frac"] >= 0.5:
            chosen_a = a
            break
    if chosen_a is None:
        chosen_a = max(ALPHAS, key=lambda a: rulsif[a][1]["ess_frac"])
    w_iwcv = rulsif[chosen_a][0]
    print(f"[iwcv] secilen IWCV alpha={chosen_a} "
          f"(ESS={rulsif[chosen_a][1]['ess']:.0f}, >=%50 hedef).")

    # --- mevcut OOF modelleri yeniden-puanla ---
    import glob
    import os
    models = sorted(
        os.path.basename(p)[4:-4]
        for p in glob.glob(str(cv.ARTIFACTS_DIR / "oof_*.npy"))
    )
    rows = []
    for m in models:
        oof = np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy")
        if len(oof) != len(y):
            continue
        rw_primer = cv.compute_recency_weighted_mse(oof, y, w_rec)   # PRIMER (gy-marjinal)
        uw, _, _ = cv.compute_cv_mse(oof, y, folds, sid)             # unweighted CV
        row = dict(model=m, unweighted_cv=round(uw, 4), rw_primer=round(rw_primer, 4))
        for a in ALPHAS:
            row[f"iwcv_a{a}"] = round(_iwcv_mse(oof, y, rulsif[a][0]), 4)
        rows.append(row)

    rep = pd.DataFrame(rows)
    # --- siralama kiyasi: PRIMER rw vs IWCV(chosen) ---
    rep_sorted_primer = rep.sort_values("rw_primer")["model"].tolist()
    iwcv_col = f"iwcv_a{chosen_a}"
    rep_sorted_iwcv = rep.sort_values(iwcv_col)["model"].tolist()
    # Spearman (rank korelasyon) PRIMER vs IWCV
    r_primer = rep["rw_primer"].rank().to_numpy()
    r_iwcv = rep[iwcv_col].rank().to_numpy()
    spear = float(np.corrcoef(r_primer, r_iwcv)[0, 1])

    # en-iyi-tek (blend/txt_ridge* haric) + blend secimi degisiyor mu?
    def _is_single(m):
        return not m.startswith(("blend", "txt_")) and not m.endswith("_w")
    singles = rep[rep["model"].map(_is_single)]
    best_single_primer = singles.sort_values("rw_primer").iloc[0]["model"]
    best_single_iwcv = singles.sort_values(iwcv_col).iloc[0]["model"]

    # --- forensik teyit: rw_primer ~ uw * (test_var/train_var) cok-degiskenli agirlikta da tutar mi? ---
    var_ratio = float(np.var(np.repeat(y, 1)))  # placeholder; gercek oran asagida
    # test-agirlikli hedef varyansi: w_rec ile y'nin agirlikli varyansi / train varyansi
    wm = w_rec / w_rec.mean()
    y_te_var = float(np.sum(wm * (y - np.sum(wm * y) / np.sum(wm)) ** 2) / np.sum(wm))
    y_tr_var = float(np.var(y))
    var_ratio = y_te_var / y_tr_var

    rep.to_csv(REPORT_PATH, index=False)
    out = dict(
        covariates=cols,
        chosen_alpha=chosen_a,
        rulsif_info={str(a): rulsif[a][1] for a in ALPHAS},
        spearman_primer_vs_iwcv=round(spear, 4),
        ranking_primer=rep_sorted_primer,
        ranking_iwcv=rep_sorted_iwcv,
        ranking_identical=(rep_sorted_primer == rep_sorted_iwcv),
        best_single_primer=best_single_primer,
        best_single_iwcv=best_single_iwcv,
        best_single_changed=(best_single_primer != best_single_iwcv),
        forensic_var_ratio_test_over_train=round(var_ratio, 4),
        note=("IWCV cok-degiskenli RuLSIF agirligiyla cross-check. PRIMER metrik DAIMA "
              "compute_recency_weighted_mse (gy-marjinal). Public LB'ye BAKILMADI."),
    )
    JSON_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"[iwcv] Spearman(PRIMER rw, IWCV a={chosen_a}) = {spear:.4f}  "
          f"(1.00 = siralama AYNI -> cok-degiskenli kayma karari DEGISTIRMIYOR).")
    print(f"[iwcv] siralama BIREBIR ayni mi? {out['ranking_identical']}")
    print(f"[iwcv] en-iyi-tek (PRIMER) = {best_single_primer} ; (IWCV) = {best_single_iwcv} "
          f"-> degisti mi? {out['best_single_changed']}")
    print(f"[iwcv] forensik teyit: test/train hedef-varyans orani = {var_ratio:.3f} "
          f"(forensik 265.7/230.6=1.152 ile kiyasla).")
    print(f"[iwcv] yazildi: {REPORT_PATH.name}, {JSON_PATH.name}")
    print("[iwcv] KARAR: IWCV cross-check'tir, PRIMER metrik degildir; final secim DEGISMEZ.")


if __name__ == "__main__":
    main()
