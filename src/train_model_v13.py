"""
submission_v13 — yeni rejim (yil-norm + uniform) re-tuned final montaj.

- Modeller (hepsi yn + uniform): lgbm, xgb, cat (best_params_v13.json'dan)
  + huber-lgbm (ayni lgbm parametreleri, objective=huber) + mlp
- + torch NN (cache, y-uzayi)
- 12-fold x 3 seed
- Blendler: NNLS / Ridge-meta / basit-ortalama(top4) — hepsi raporlanir
- Cikti 1: submission_v13.csv  (en iyi blend + gerekirse yil-calib)
- Cikti 2: submission_v13b.csv (0.5*v13 + 0.5*v7 — rejim cesitliligi,
  agirlik-fit'siz sabit ortalama)

Calistir: python -u src/train_model_v13.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v13.csv"
OUT_B = ROOT / "submissions" / "submission_v13b.csv"

SEED = 42
N_FOLDS = 12
SEEDS = [42, 7, 2024]
REG_NAMES = ["lgbm", "xgb", "cat", "huber", "mlp"]
ALL_NAMES = REG_NAMES + ["nn"]


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

    bp = json.loads((CACHE / "best_params_v13.json").read_text())
    lgbm_p, cat_p, xgb_p = bp["lgbm"]["params"], bp["cat"]["params"], bp["xgb"]["params"]
    print(f"  {N_FOLDS}-fold x {len(SEEDS)} seed | yeni rejim (yn + uniform)")

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def make(name, seed):
        if name == "lgbm":
            return LGBMRegressor(**lgbm_p, subsample_freq=1, random_state=seed,
                                 n_jobs=8, verbose=-1)
        if name == "huber":
            return LGBMRegressor(**lgbm_p, objective="huber", alpha=5.0,
                                 subsample_freq=1, random_state=seed,
                                 n_jobs=8, verbose=-1)
        if name == "xgb":
            return XGBRegressor(**xgb_p, random_state=seed, n_jobs=8,
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

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
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
        Ctr = train.iloc[tr_idx][cat_input_cols]
        Cva = train.iloc[va_idx][cat_input_cols]
        Cte = test[cat_input_cols]
        ns = len(SEEDS)

        for seed in SEEDS:
            for n in REG_NAMES:
                m = make(n, seed)
                if n == "cat":
                    m.fit(Ctr, yn, cat_features=F.CAT_COLS)
                    va_p, te_p = m.predict(Cva), m.predict(Cte)
                else:
                    m.fit(Xtr, yn)
                    va_p, te_p = m.predict(Xva), m.predict(Xte)
                oof[n][va_idx] += (va_p * sd_va + mu_va) / ns
                test_pred[n] += (te_p * sd_te + mu_te) / (ns * N_FOLDS)
        print(f"  fold {fold}/{N_FOLDS} bitti")

    oof_all = dict(oof)
    test_all = dict(test_pred)
    oof_all["nn"] = np.load(CACHE / "nn_oof.npy")
    test_all["nn"] = np.load(CACHE / "nn_test.npy")

    print("\n=== TEKIL OOF (duz | agirlikli ~ LB proxy) ===")
    for n in ALL_NAMES:
        p = np.clip(oof_all[n], 0, 100)
        print(f"  {n:6s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    M = np.column_stack([oof_all[m] for m in ALL_NAMES])
    Mte = np.column_stack([test_all[m] for m in ALL_NAMES])
    sw = np.sqrt(w_fit)

    # 1) NNLS
    wb, _ = nnls(M * sw[:, None], y * sw)
    wb /= wb.sum()
    s_nnls = wmse(y, np.clip(M @ wb, 0, 100), w_fit)

    # 2) Ridge meta
    extra_tr = train[["application_year", "project_quality_score"]].fillna(0).values
    extra_te = test[["application_year", "project_quality_score"]].fillna(0).values
    Xmeta = np.column_stack([M, extra_tr])
    r_oof = np.zeros(len(y))
    for tr_i, va_i in kf.split(Xmeta):
        r = Ridge(alpha=1.0)
        r.fit(Xmeta[tr_i], y[tr_i], sample_weight=w_fit[tr_i])
        r_oof[va_i] = r.predict(Xmeta[va_i])
    s_ridge = wmse(y, np.clip(r_oof, 0, 100), w_fit)

    # 3) basit ortalama (en iyi 4 agirlikli-OOF uyesi, fit'siz)
    top4 = sorted(ALL_NAMES, key=lambda n: wmse(y, np.clip(oof_all[n], 0, 100), w_fit))[:4]
    avg_oof = np.clip(np.mean([oof_all[n] for n in top4], axis=0), 0, 100)
    s_avg = wmse(y, avg_oof, w_fit)

    print("\n=== BLENDLER ===")
    print(f"  NNLS        : {s_nnls:.4f}  ({dict(zip(ALL_NAMES, np.round(wb,3)))})")
    print(f"  Ridge meta  : {s_ridge:.4f}")
    print(f"  Ortalama(4) : {s_avg:.4f}  ({top4})")

    scores = {"nnls": s_nnls, "ridge": s_ridge, "avg": s_avg}
    chosen = min(scores, key=scores.get)
    if chosen == "nnls":
        ens = np.clip(M @ wb, 0, 100)
        final = np.clip(Mte @ wb, 0, 100)
    elif chosen == "ridge":
        ens = np.clip(r_oof, 0, 100)
        mdl = Ridge(alpha=1.0).fit(Xmeta, y, sample_weight=w_fit)
        final = np.clip(mdl.predict(np.column_stack([Mte, extra_te])), 0, 100)
    else:
        ens = avg_oof
        final = np.clip(np.mean([test_all[n] for n in top4], axis=0), 0, 100)
    blend_score = scores[chosen]

    # yil-kalibrasyon kontrolu
    cal = ens.copy()
    for tr_i, va_i in kf.split(ens):
        for yil in np.unique(yr_tr):
            mt = tr_i[yr_tr[tr_i] == yil]
            mv = va_i[yr_tr[va_i] == yil]
            if len(mt) > 50 and len(mv) > 0:
                b, a = np.polyfit(ens[mt], y[mt], 1)
                cal[mv] = a + b * ens[mv]
    cal = np.clip(cal, 0, 100)
    s_cal = wmse(y, cal, w_fit)
    use_cal = s_cal < blend_score
    print(f"Yil-kalibrasyon: {blend_score:.4f} -> {s_cal:.4f} "
          f"({'UYGULANIYOR' if use_cal else 'atlandi'})")
    if use_cal:
        for yil in np.unique(yr_te):
            mt = yr_tr == yil
            me = yr_te == yil
            if mt.sum() > 50:
                b, a = np.polyfit(ens[mt], y[mt], 1)
                final[me] = a + b * final[me]
        final = np.clip(final, 0, 100)

    np.savez(CACHE / "preds_v13.npz", y=y, w_fit=w_fit, years=yr_tr,
             **{f"oof_{m}": oof_all[m] for m in ALL_NAMES},
             **{f"test_{m}": test_all[m] for m in ALL_NAMES})

    final = final.round(3)
    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"\nYAZILDI -> {OUT} (blend: {chosen})")
    print(f"v13 proxy ~{min(blend_score, s_cal):.2f}")

    # --- v13b: 0.5*v13 + 0.5*v7 (rejim cesitliligi, fit'siz) ---
    v7 = pd.read_csv(ROOT / "submissions" / "submission_v7.csv")
    mix = (0.5 * final + 0.5 * v7[F.TARGET].values).round(3)
    pd.DataFrame({F.ID: test[F.ID], F.TARGET: mix}).to_csv(OUT_B, index=False)
    print(f"YAZILDI -> {OUT_B} (0.5*v13 + 0.5*v7)")
    print(f"\nv13 ilk 5:\n{sub.head().to_string(index=False)}")


if __name__ == "__main__":
    main()
