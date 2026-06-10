"""
Tarif #3 — farkli loss fonksiyonlari (super-blend cesitlilik uyeleri).

Son-yil gurultusune farkli tepki veren kayiplar:
  - huber : LGBM objective='huber' (aykiri-dirençli)
  - quant : LGBM quantile q={0.35,0.5,0.65} ortalamasi (robust merkez)
  - catmae: CatBoost MAE (medyan tahmincisi, GPU)

10-fold x 2 seed, y-uzayi, w_fit agirlikli. Cikti: preds_r3.npz
Sonra super_blend.py ile v7+v9+r3 birlestirilir.

Calistir: python -u src/train_recipe3.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
SEED = 42
N_FOLDS = 10
SEEDS = [42, 7]
NAMES = ["huber", "quant", "catmae"]


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

    lgbm_p = json.loads((CACHE / "best_params.json").read_text())["lgbm"]["params"]

    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(train)) for m in NAMES}
    test_pred = {m: np.zeros(len(test)) for m in NAMES}
    QS = [0.35, 0.5, 0.65]

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
            # huber LGBM
            m = LGBMRegressor(**lgbm_p, objective="huber", alpha=5.0,
                              subsample_freq=1, random_state=seed,
                              n_jobs=8, verbose=-1)
            m.fit(Xtr, y[tr_idx], sample_weight=wf)
            oof["huber"][va_idx] += m.predict(Xva) / ns
            test_pred["huber"] += m.predict(Xte) / (ns * N_FOLDS)

            # quantile trio ortalamasi
            qva = np.zeros(len(va_idx))
            qte = np.zeros(len(test))
            for q in QS:
                mq = LGBMRegressor(**lgbm_p, objective="quantile", alpha=q,
                                   subsample_freq=1, random_state=seed,
                                   n_jobs=8, verbose=-1)
                mq.fit(Xtr, y[tr_idx], sample_weight=wf)
                qva += mq.predict(Xva) / len(QS)
                qte += mq.predict(Xte) / len(QS)
            oof["quant"][va_idx] += qva / ns
            test_pred["quant"] += qte / (ns * N_FOLDS)

            # CatBoost MAE (GPU)
            mc = CatBoostRegressor(
                iterations=2000, learning_rate=0.03, depth=6,
                loss_function="MAE", l2_leaf_reg=5.0,
                task_type="GPU", devices="0", one_hot_max_size=16,
                verbose=0, allow_writing_files=False, random_seed=seed)
            mc.fit(Ctr, y[tr_idx], cat_features=F.CAT_COLS, sample_weight=wf)
            oof["catmae"][va_idx] += mc.predict(Cva) / ns
            test_pred["catmae"] += mc.predict(Cte) / (ns * N_FOLDS)
        print(f"  fold {fold}/{N_FOLDS} bitti")

    print("\n=== TEKIL OOF (duz | agirlikli) ===")
    for n in NAMES:
        p = np.clip(oof[n], 0, 100)
        print(f"  {n:7s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    np.savez(CACHE / "preds_r3.npz", y=y, w_fit=w_fit,
             years=train["application_year"].values,
             **{f"oof_{m}": oof[m] for m in NAMES},
             **{f"test_{m}": test_pred[m] for m in NAMES})
    print(f"KAYDEDILDI -> {CACHE}/preds_r3.npz")


if __name__ == "__main__":
    main()
