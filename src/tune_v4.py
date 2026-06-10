"""
Optuna hiperparametre aramasi (v4 icin).

- CatBoost: GPU (RTX 4070), native kategorik, 30 trial
- LightGBM: CPU, fold-ici target encoding, 15 trial
- Hedef metrik: yil-agirlikli OOF MSE (LB proxy'si) — 3-fold, tek seed
- MedianPruner: umutsuz trial'lari fold sonunda keser
- En iyi parametreler data/cache/best_params.json'a yazilir

Calistir: python -u src/tune_v4.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold

import features as F

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "cache" / "best_params.json"
SEED = 42
TUNE_FOLDS = 3


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


print("Feature'lar yukleniyor/uretiliyor...")
train, test, y, w_fit, num_cols = F.build_features()
cat_input_cols = num_cols + F.CAT_COLS
print(f"  train {train.shape} | sayisal {len(num_cols)}")

kf = KFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(kf.split(train))
gmean = y.mean()


# ---------------- CatBoost (GPU) ----------------
def cat_objective(trial):
    from catboost import CatBoostRegressor
    params = dict(
        iterations=trial.suggest_int("iterations", 800, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        depth=trial.suggest_int("depth", 5, 9),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        task_type="GPU", devices="0", verbose=0,
        allow_writing_files=False, random_seed=SEED,
    )
    oof = np.zeros(len(train))
    for i, (tr_idx, va_idx) in enumerate(FOLDS):
        m = CatBoostRegressor(**params)
        m.fit(train.iloc[tr_idx][cat_input_cols], y[tr_idx],
              cat_features=F.CAT_COLS, sample_weight=w_fit[tr_idx])
        oof[va_idx] = m.predict(train.iloc[va_idx][cat_input_cols])
        # pruning: fold bazinda ara skor bildir
        partial = wmse(y[va_idx], oof[va_idx], w_fit[va_idx])
        trial.report(partial, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return wmse(y, oof, w_fit)


# ---------------- LightGBM (CPU) ----------------
def target_encode(tr_col, va_col, y_tr, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y_tr}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + gmean * smoothing) / (stats["count"] + smoothing)
    return va_col.map(enc).fillna(gmean).values


def lgbm_objective(trial):
    from lightgbm import LGBMRegressor
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 800, 2200),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.07, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1,
    )
    oof = np.zeros(len(train))
    for i, (tr_idx, va_idx) in enumerate(FOLDS):
        Xtr = train.iloc[tr_idx][num_cols].copy()
        Xva = train.iloc[va_idx][num_cols].copy()
        for c in F.CAT_COLS:
            Xtr[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[tr_idx][c], y[tr_idx])
            Xva[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[va_idx][c], y[tr_idx])
        m = LGBMRegressor(**params)
        m.fit(Xtr, y[tr_idx], sample_weight=w_fit[tr_idx])
        oof[va_idx] = m.predict(Xva)
        partial = wmse(y[va_idx], oof[va_idx], w_fit[va_idx])
        trial.report(partial, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return wmse(y, oof, w_fit)


def run_study(name, objective, n_trials):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    def cb(study, trial):
        s = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        print(f"  [{name}] trial {trial.number:>2}: {s}  "
              f"(best={study.best_value:.4f})")
    study.optimize(objective, n_trials=n_trials, callbacks=[cb])
    print(f"[{name}] EN IYI: {study.best_value:.4f}")
    print(f"[{name}] params: {study.best_params}")
    return study.best_params, study.best_value


if __name__ == "__main__":
    results = {}
    print("\n=== CatBoost (GPU) tuning — 30 trial ===")
    cat_params, cat_score = run_study("cat", cat_objective, 30)
    results["cat"] = {"params": cat_params, "wmse_3fold": cat_score}

    print("\n=== LightGBM (CPU) tuning — 15 trial ===")
    lgbm_params, lgbm_score = run_study("lgbm", lgbm_objective, 15)
    results["lgbm"] = {"params": lgbm_params, "wmse_3fold": lgbm_score}

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nKAYDEDILDI -> {OUT_JSON}")
