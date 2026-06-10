"""
submission_v5 — v4 + BERT metin tahmini feature'i.

Fark: bert_text_oof.py'nin urettigi txt_bert (OOF, sizintisiz) feature
olarak eklenir. Geri kalani v4 ile ayni (tuned parametreler, 10-fold,
2 seed, NNLS + Ridge-meta blend kiyasi).

Calistir: python -u src/train_model_v5.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v5.csv"
PARAMS_JSON = CACHE / "best_params.json"

SEED = 42
N_FOLDS = 10
SEEDS = [42, 7]


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def target_encode(tr_col, apply_col, y_tr, gmean, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y_tr}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + gmean * smoothing) / (stats["count"] + smoothing)
    return apply_col.map(enc).fillna(gmean).values


def main():
    print("Feature'lar yukleniyor...")
    train, test, y, w_fit, num_cols = F.build_features()
    # --- BERT feature ---
    train["txt_bert"] = np.load(CACHE / "bert_oof.npy")
    test["txt_bert"] = np.load(CACHE / "bert_test.npy")
    num_cols = num_cols + ["txt_bert"]
    cat_input_cols = num_cols + F.CAT_COLS
    gmean = y.mean()
    print(f"  train {train.shape} | sayisal {len(num_cols)} (txt_bert dahil)")

    best = json.loads(PARAMS_JSON.read_text())
    cat_p = best["cat"]["params"]
    lgbm_p = best["lgbm"]["params"]

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def make_cat(seed):
        return CatBoostRegressor(**cat_p, task_type="GPU", devices="0",
                                 one_hot_max_size=16, verbose=0,
                                 allow_writing_files=False, random_seed=seed)

    def make_lgbm(seed):
        return LGBMRegressor(**lgbm_p, subsample_freq=1, random_state=seed,
                             n_jobs=8, verbose=-1)

    def make_xgb(seed):
        return XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            min_child_weight=5, random_state=seed, n_jobs=8,
            tree_method="hist", device="cuda")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    names = ["lgbm", "xgb", "cat"]
    oof = {m: np.zeros(len(train)) for m in names}
    test_pred = {m: np.zeros(len(test)) for m in names}

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
            m = make_lgbm(seed); m.fit(Xtr, y[tr_idx], sample_weight=wf)
            oof["lgbm"][va_idx] += m.predict(Xva) / ns
            test_pred["lgbm"] += m.predict(Xte) / (ns * N_FOLDS)
            m = make_xgb(seed); m.fit(Xtr, y[tr_idx], sample_weight=wf)
            oof["xgb"][va_idx] += m.predict(Xva) / ns
            test_pred["xgb"] += m.predict(Xte) / (ns * N_FOLDS)
            m = make_cat(seed)
            m.fit(Ctr, y[tr_idx], cat_features=F.CAT_COLS, sample_weight=wf)
            oof["cat"][va_idx] += m.predict(Cva) / ns
            test_pred["cat"] += m.predict(Cte) / (ns * N_FOLDS)
        print(f"  fold {fold}/{N_FOLDS} bitti")

    print("\n=== TEKIL OOF (duz | agirlikli ~ LB proxy) ===")
    for n in names:
        p = np.clip(oof[n], 0, 100)
        print(f"  {n:5s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    # --- NNLS blend ---
    M = np.column_stack([oof[m] for m in names])
    Mte = np.column_stack([test_pred[m] for m in names])
    sw = np.sqrt(w_fit)
    w_blend, _ = nnls(M * sw[:, None], y * sw)
    w_blend = w_blend / w_blend.sum()
    nnls_oof = np.clip(M @ w_blend, 0, 100)
    nnls_score = wmse(y, nnls_oof, w_fit)

    # --- Ridge meta (v4b'de kazanmisti) ---
    extra_tr = train[["application_year", "project_quality_score"]].fillna(0).values
    extra_te = test[["application_year", "project_quality_score"]].fillna(0).values
    Xmeta = np.column_stack([M, extra_tr])
    ridge_oof_p = np.zeros(len(y))
    for tr_i, va_i in kf.split(Xmeta):
        r = Ridge(alpha=1.0)
        r.fit(Xmeta[tr_i], y[tr_i], sample_weight=w_fit[tr_i])
        ridge_oof_p[va_i] = r.predict(Xmeta[va_i])
    ridge_oof_p = np.clip(ridge_oof_p, 0, 100)
    ridge_score = wmse(y, ridge_oof_p, w_fit)

    print("\n=== BLEND ===")
    print(f"  NNLS      : {nnls_score:.4f} (agirliklar {np.round(w_blend,3)})")
    print(f"  Ridge meta: {ridge_score:.4f}")

    if ridge_score < nnls_score:
        mdl = Ridge(alpha=1.0).fit(Xmeta, y, sample_weight=w_fit)
        final = np.clip(mdl.predict(np.column_stack([Mte, extra_te])), 0, 100)
        chosen, score = "ridge-meta", ridge_score
    else:
        final = np.clip(Mte @ w_blend, 0, 100)
        chosen, score = "nnls", nnls_score
    final = final.round(3)

    np.savez(CACHE / "preds_v5.npz", y=y, w_fit=w_fit,
             **{f"oof_{m}": oof[m] for m in names},
             **{f"test_{m}": test_pred[m] for m in names})

    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"\nYAZILDI -> {OUT} (blend: {chosen}, LB tahmini ~{score:.2f})")
    print(f"satir: {len(sub)} | aralik: {final.min()}-{final.max()} | ort: {final.mean():.2f}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
