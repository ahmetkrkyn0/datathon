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
]
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
