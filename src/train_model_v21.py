"""
submission_v21 — FOLD-ESLI embedding'li final model.

v19 tarifi (yn + uniform + segment-yil TE + bert skalarlar) +
fold-esli BERT embedding'leri (bert_foldemb.npz):
  GBM fold k'da: PCA64, model_k'nin train-fold embedding'lerinde fit edilir;
  val ve test ayni modelin uzayindan donusturulur. Boylece v17'deki
  uzay-karisikligi yok — her fold tek tutarli uzayda.

Cikti: submission_v21.csv + submission_v22.csv (0.5*v21 + 0.5*v7)

Calistir: python -u src/train_model_v21.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v21.csv"
OUT_MIX = ROOT / "submissions" / "submission_v22.csv"

SEED = 42
N_FOLDS = 10  # bert_foldmatched ile ayni
SEEDS = [42, 7]
N_EMB = 48
REG_NAMES = ["lgbm", "xgb", "cat", "mlp"]
ALL_NAMES = REG_NAMES + ["nn"]
SEG_COLS = ["target_role", "university_tier", "hobby",
            "preferred_social_media_platform"]
SM = 20


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def target_encode(tr_col, apply_col, y_tr, gmean, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y_tr}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + gmean * smoothing) / (stats["count"] + smoothing)
    return apply_col.map(enc).fillna(gmean).values


def cell_te(seg_tr, yil_tr, y_tr, seg_ap, yil_ap, sm=SM):
    df = pd.DataFrame({"s": seg_tr, "yil": yil_tr, "y": y_tr})
    cell = df.groupby(["s", "yil"])["y"].agg(["mean", "count"])
    yil_mean = df.groupby("yil")["y"].mean()
    cell["te"] = (cell["mean"] * cell["count"]
                  + cell.index.get_level_values("yil").map(yil_mean) * sm) / (cell["count"] + sm)
    s = pd.Series(list(zip(seg_ap, yil_ap))).map(cell["te"])
    return s.fillna(pd.Series(yil_ap).map(yil_mean)).values


def main():
    print("Feature'lar yukleniyor...")
    train, test, y, w_fit, num_cols = F.build_features()
    train["txt_bert"] = np.load(CACHE / "bert_oof.npy")
    test["txt_bert"] = np.load(CACHE / "bert_test.npy")
    train["txt_bert2"] = np.load(CACHE / "bert2_oof.npy")
    test["txt_bert2"] = np.load(CACHE / "bert2_test.npy")
    train["txt_bert3"] = np.load(CACHE / "bert3_oof.npy")
    test["txt_bert3"] = np.load(CACHE / "bert3_test.npy")
    num_cols = num_cols + ["txt_bert", "txt_bert2", "txt_bert3"]
    foldemb = np.load(CACHE / "bert_foldemb.npz")
    gmean = y.mean()
    yr_tr = train["application_year"].values
    yr_te = test["application_year"].values
    segs = {c: (train[c].astype(str).values, test[c].astype(str).values)
            for c in SEG_COLS}

    bp = json.loads((CACHE / "best_params.json").read_text())
    lgbm_p, cat_p = bp["lgbm"]["params"], bp["cat"]["params"]
    print(f"  {N_FOLDS}-fold x {SEEDS} | fold-esli emb PCA{N_EMB}")

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
            return CatBoostRegressor(**cat_p, one_hot_max_size=16,
                                     task_type="GPU", devices="0", verbose=0,
                                     allow_writing_files=False, random_seed=seed)
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                         learning_rate_init=1e-3, batch_size=256,
                         max_iter=120, early_stopping=True,
                         n_iter_no_change=8, random_state=seed))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(train)) for m in REG_NAMES}
    test_pred = {m: np.zeros(len(test)) for m in REG_NAMES}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        st = pd.DataFrame({"yil": yr_tr[tr_idx], "y": y[tr_idx]}).groupby("yil")["y"].agg(["mean", "std"])
        mu_tr = pd.Series(yr_tr[tr_idx]).map(st["mean"]).values
        sd_tr = pd.Series(yr_tr[tr_idx]).map(st["std"]).values
        yn = (y[tr_idx] - mu_tr) / sd_tr
        mu_va = pd.Series(yr_tr[va_idx]).map(st["mean"]).values
        sd_va = pd.Series(yr_tr[va_idx]).map(st["std"]).values
        mu_te = pd.Series(yr_te).map(st["mean"]).values
        sd_te = pd.Series(yr_te).map(st["std"]).values

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
        for c, (seg_tr_full, seg_te_full) in segs.items():
            Xtr[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx],
                                         seg_tr_full[tr_idx], yr_tr[tr_idx])
            Xva[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx],
                                         seg_tr_full[va_idx], yr_tr[va_idx])
            Xte[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx],
                                         seg_te_full, yr_te)

        # --- FOLD-ESLI embedding: model_k uzayinda PCA ---
        e_tr_full = foldemb[f"emb_tr_f{fold}"]
        e_te_full = foldemb[f"emb_te_f{fold}"]
        svd = TruncatedSVD(n_components=N_EMB, random_state=SEED)
        E_tr = svd.fit_transform(e_tr_full[tr_idx])
        E_va = svd.transform(e_tr_full[va_idx])
        E_te = svd.transform(e_te_full)
        for i in range(N_EMB):
            Xtr[f"femb_{i}"] = E_tr[:, i]
            Xva[f"femb_{i}"] = E_va[:, i]
            Xte[f"femb_{i}"] = E_te[:, i]

        ns = len(SEEDS)
        for seed in SEEDS:
            for n in REG_NAMES:
                m = make(n, seed)
                m.fit(Xtr, yn)
                va_p, te_p = m.predict(Xva), m.predict(Xte)
                oof[n][va_idx] += (va_p * sd_va + mu_va) / ns
                test_pred[n] += (te_p * sd_te + mu_te) / (ns * N_FOLDS)
        print(f"  fold {fold+1}/{N_FOLDS} bitti")

    oof_all = dict(oof)
    test_all = dict(test_pred)
    oof_all["nn"] = np.load(CACHE / "nn_oof.npy")
    test_all["nn"] = np.load(CACHE / "nn_test.npy")

    print("\n=== TEKIL OOF (duz | agirlikli ~ LB proxy) ===")
    for n in ALL_NAMES:
        p = np.clip(oof_all[n], 0, 100)
        print(f"  {n:5s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    M = np.column_stack([oof_all[m] for m in ALL_NAMES])
    Mte = np.column_stack([test_all[m] for m in ALL_NAMES])
    sw = np.sqrt(w_fit)
    wb, _ = nnls(M * sw[:, None], y * sw)
    wb /= wb.sum()
    s_nnls = wmse(y, np.clip(M @ wb, 0, 100), w_fit)

    extra_tr = train[["application_year", "project_quality_score"]].fillna(0).values
    extra_te = test[["application_year", "project_quality_score"]].fillna(0).values
    Xmeta = np.column_stack([M, extra_tr])
    r_oof = np.zeros(len(y))
    for tr_i, va_i in kf.split(Xmeta):
        r = Ridge(alpha=1.0)
        r.fit(Xmeta[tr_i], y[tr_i], sample_weight=w_fit[tr_i])
        r_oof[va_i] = r.predict(Xmeta[va_i])
    s_ridge = wmse(y, np.clip(r_oof, 0, 100), w_fit)

    print("\n=== BLEND ===")
    print(f"  NNLS      : {s_nnls:.4f}")
    print(f"  Ridge meta: {s_ridge:.4f}")

    if s_ridge < s_nnls:
        mdl = Ridge(alpha=1.0).fit(Xmeta, y, sample_weight=w_fit)
        final = np.clip(mdl.predict(np.column_stack([Mte, extra_te])), 0, 100)
        chosen, score = "ridge-meta", s_ridge
    else:
        final = np.clip(Mte @ wb, 0, 100)
        chosen, score = "nnls", s_nnls

    np.savez(CACHE / "preds_v21.npz", y=y, w_fit=w_fit, years=yr_tr,
             **{f"oof_{m}": oof_all[m] for m in ALL_NAMES},
             **{f"test_{m}": test_all[m] for m in ALL_NAMES})

    final = final.round(3)
    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"\nYAZILDI -> {OUT} (blend: {chosen})")
    print(f"v21 proxy ~{score:.2f}")

    v7 = pd.read_csv(ROOT / "submissions" / "submission_v7.csv")
    mix = (0.5 * final + 0.5 * v7[F.TARGET].values).round(3)
    pd.DataFrame({F.ID: test[F.ID], F.TARGET: mix}).to_csv(OUT_MIX, index=False)
    print(f"YAZILDI -> {OUT_MIX} (0.5*v21 + 0.5*v7)")


if __name__ == "__main__":
    main()
