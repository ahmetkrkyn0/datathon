"""
submission_v7 — 5-model blend + yil-bazli kalibrasyon (final paket).

v4b'den (LB 86.58) farklar:
  1. +2 divers model: ExtraTrees + MLP (NNLS blend cesitliligi)
  2. txt_bert feature (BERT OOF tahmini)
  3. Yil-bazli affine kalibrasyon (nested-CV ile durust dogrulanir,
     kazandiriyorsa uygulanir; OOF eğimler 2025-26'da >1 — hafif genisletme)
  4. 10-fold x 2 seed

Calistir: python -u src/train_model_v7.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v7.csv"

SEED = 42
N_FOLDS = 10
SEEDS = [42, 7]
NAMES = ["lgbm", "xgb", "cat", "et", "mlp"]


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def target_encode(tr_col, apply_col, y_tr, gmean, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y_tr}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + gmean * smoothing) / (stats["count"] + smoothing)
    return apply_col.map(enc).fillna(gmean).values


def main():
    print("Feature'lar yukleniyor...")
    train, test, y, w_fit, num_cols = F.build_features()
    train["txt_bert"] = np.load(CACHE / "bert_oof.npy")
    test["txt_bert"] = np.load(CACHE / "bert_test.npy")
    num_cols = num_cols + ["txt_bert"]
    cat_input_cols = num_cols + F.CAT_COLS
    gmean = y.mean()
    yr_tr = train["application_year"].values
    yr_te = test["application_year"].values
    print(f"  sayisal {len(num_cols)} | modeller: {NAMES} | {N_FOLDS}-fold x {SEEDS}")

    best = json.loads((CACHE / "best_params.json").read_text())
    cat_p, lgbm_p = best["cat"]["params"], best["lgbm"]["params"]

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def make(name, seed):
        if name == "lgbm":
            return LGBMRegressor(**lgbm_p, subsample_freq=1, random_state=seed,
                                 n_jobs=8, verbose=-1)
        if name == "xgb":
            return XGBRegressor(
                n_estimators=1500, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                min_child_weight=5, random_state=seed, n_jobs=8,
                tree_method="hist", device="cuda")
        if name == "cat":
            return CatBoostRegressor(**cat_p, task_type="GPU", devices="0",
                                     one_hot_max_size=16, verbose=0,
                                     allow_writing_files=False, random_seed=seed)
        if name == "et":
            return ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=5, max_features=0.6,
                random_state=seed, n_jobs=8)
        # mlp: impute+scale pipeline gerekir
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                         learning_rate_init=1e-3, batch_size=256,
                         max_iter=120, early_stopping=True,
                         n_iter_no_change=8, random_state=seed))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(train)) for m in NAMES}
    test_pred = {m: np.zeros(len(test)) for m in NAMES}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        Xtr = train.iloc[tr_idx][num_cols].copy()
        Xva = train.iloc[va_idx][num_cols].copy()
        Xte = test[num_cols].copy()
        for c in F.CAT_COLS:
            Xtr[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[tr_idx][c], y[tr_idx], gmean)
            Xva[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[va_idx][c], y[tr_idx], gmean)
            Xte[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           test[c], y[tr_idx], gmean)
        Ctr = train.iloc[tr_idx][cat_input_cols]
        Cva = train.iloc[va_idx][cat_input_cols]
        Cte = test[cat_input_cols]
        wf = w_fit[tr_idx]
        ns = len(SEEDS)

        for seed in SEEDS:
            for n in NAMES:
                m = make(n, seed)
                if n == "cat":
                    m.fit(Ctr, y[tr_idx], cat_features=F.CAT_COLS, sample_weight=wf)
                    va_p, te_p = m.predict(Cva), m.predict(Cte)
                elif n == "mlp":
                    # pipeline: sample_weight MLP'ye iletilmez -> agirliksiz egit
                    m.fit(Xtr, y[tr_idx])
                    va_p, te_p = m.predict(Xva), m.predict(Xte)
                else:
                    m.fit(Xtr, y[tr_idx], sample_weight=wf)
                    va_p, te_p = m.predict(Xva), m.predict(Xte)
                oof[n][va_idx] += va_p / ns
                test_pred[n] += te_p / (ns * N_FOLDS)
        print(f"  fold {fold}/{N_FOLDS} bitti")

    print("\n=== TEKIL OOF (duz | agirlikli ~ LB proxy) ===")
    for n in NAMES:
        p = np.clip(oof[n], 0, 100)
        print(f"  {n:5s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    # --- NNLS blend ---
    M = np.column_stack([oof[m] for m in NAMES])
    Mte = np.column_stack([test_pred[m] for m in NAMES])
    sw = np.sqrt(w_fit)
    wb, _ = nnls(M * sw[:, None], y * sw)
    wb /= wb.sum()
    ens = np.clip(M @ wb, 0, 100)
    base_score = wmse(y, ens, w_fit)
    print("\n=== ENSEMBLE ===")
    for n, wi in zip(NAMES, wb):
        print(f"  {n:5s} agirlik = {wi:.3f}")
    print(f"  agirlikli (LB proxy) = {base_score:.4f}")

    # --- Yil-bazli affine kalibrasyon (nested-CV dogrulama) ---
    cal_oof = ens.copy()
    for tr_i, va_i in kf.split(ens):
        for yil in np.unique(yr_tr):
            m_tr = tr_i[yr_tr[tr_i] == yil]
            m_va = va_i[yr_tr[va_i] == yil]
            if len(m_tr) > 50 and len(m_va) > 0:
                b, a = np.polyfit(ens[m_tr], y[m_tr], 1)
                cal_oof[m_va] = a + b * ens[m_va]
    cal_oof = np.clip(cal_oof, 0, 100)
    cal_score = wmse(y, cal_oof, w_fit)
    use_cal = cal_score < base_score
    print(f"Yil-kalibrasyon: {base_score:.4f} -> {cal_score:.4f} "
          f"({'UYGULANIYOR' if use_cal else 'atlandi'})")

    final = np.clip(Mte @ wb, 0, 100)
    if use_cal:
        for yil in np.unique(yr_te):
            m_tr = yr_tr == yil
            m_te = yr_te == yil
            if m_tr.sum() > 50:
                b, a = np.polyfit(ens[m_tr], y[m_tr], 1)
                final[m_te] = a + b * final[m_te]
        final = np.clip(final, 0, 100)
    final = final.round(3)

    np.savez(CACHE / "preds_v7.npz", y=y, w_fit=w_fit, years=yr_tr,
             **{f"oof_{m}": oof[m] for m in NAMES},
             **{f"test_{m}": test_pred[m] for m in NAMES})

    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    score = cal_score if use_cal else base_score
    print(f"\nYAZILDI -> {OUT}")
    print(f"LB tahmini ~{score:.2f}")
    print(f"satir: {len(sub)} | aralik: {final.min()}-{final.max()} | ort: {final.mean():.2f}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
