"""
YENI REJIM tuning: yil-normalize hedef + UNIFORM egitim agirligi.

Eski tune'lar (best_params.json) raw-y + w_fit rejimi icindi. Uniform+yn
rejiminde optimum farkli olabilir (etkin temiz veri artti -> daha buyuk
kapasite kaldirabilir). Eval: y-uzayina cevrilip yil-agirlikli MSE.

LGBM 25 + CatBoost(GPU) 25 + XGB(GPU) 15 trial, 3-fold, MedianPruner.
Cikti: data/cache/best_params_v13.json

Calistir: python -u src/tune_v13.py
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
CACHE = ROOT / "data" / "cache"
OUT_JSON = CACHE / "best_params_v13.json"
SEED = 42

print("Feature'lar yukleniyor...")
train, test, y, w_fit, num_cols = F.build_features()
train["txt_bert"] = np.load(CACHE / "bert_oof.npy")
num_cols = num_cols + ["txt_bert"]
cat_input_cols = num_cols + F.CAT_COLS
years = train["application_year"].values

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
FOLDS = list(kf.split(train))

# fold-ici yil istatistikleri onceden hazirla
FOLD_STATS = []
for tr_idx, va_idx in FOLDS:
    st = pd.DataFrame({"yil": years[tr_idx], "y": y[tr_idx]}).groupby("yil")["y"].agg(["mean", "std"])
    yn = (y[tr_idx] - pd.Series(years[tr_idx]).map(st["mean"]).values) / \
         pd.Series(years[tr_idx]).map(st["std"]).values
    mu_v = pd.Series(years[va_idx]).map(st["mean"]).values
    sd_v = pd.Series(years[va_idx]).map(st["std"]).values
    FOLD_STATS.append((yn, mu_v, sd_v))


def wmse(yy, p, w):
    return float(np.average((yy - np.clip(p, 0, 100)) ** 2, weights=w))


def run_cv(make_model, use_cat_cols=False):
    oof = np.zeros(len(train))
    for i, ((tr_idx, va_idx), (yn, mu_v, sd_v)) in enumerate(zip(FOLDS, FOLD_STATS)):
        m = make_model()
        if use_cat_cols:
            m.fit(train.iloc[tr_idx][cat_input_cols], yn, cat_features=F.CAT_COLS)
            pn = m.predict(train.iloc[va_idx][cat_input_cols])
        else:
            m.fit(train.iloc[tr_idx][num_cols], yn)
            pn = m.predict(train.iloc[va_idx][num_cols])
        oof[va_idx] = pn * sd_v + mu_v
        yield i, wmse(y[va_idx], oof[va_idx], w_fit[va_idx])
    yield -1, wmse(y, oof, w_fit)


def make_objective(builder, use_cat_cols=False):
    def obj(trial):
        final = None
        for i, score in run_cv(lambda: builder(trial), use_cat_cols):
            if i >= 0:
                trial.report(score, i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            else:
                final = score
        return final
    return obj


def lgbm_builder(trial):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 800, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.05, 20.0, log=True),
        subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1)


def cat_builder(trial):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=trial.suggest_int("iterations", 800, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        depth=trial.suggest_int("depth", 5, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 15.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        one_hot_max_size=16, task_type="GPU", devices="0", verbose=0,
        allow_writing_files=False, random_seed=SEED)


def xgb_builder(trial):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 800, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
        random_state=SEED, n_jobs=8, tree_method="hist", device="cuda")


def run_study(name, objective, n_trials):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    def cb(study, trial):
        s = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        print(f"  [{name}] trial {trial.number:>2}: {s}  (best={study.best_value:.4f})")
    study.optimize(objective, n_trials=n_trials, callbacks=[cb])
    print(f"[{name}] EN IYI: {study.best_value:.4f} | {study.best_params}")
    return study.best_params, study.best_value


if __name__ == "__main__":
    results = {}
    print("\n=== CatBoost (GPU, yeni rejim) 25 trial ===")
    p, s = run_study("cat", make_objective(cat_builder, use_cat_cols=True), 25)
    results["cat"] = {"params": p, "wmse_3fold": s}

    print("\n=== LGBM (yeni rejim) 25 trial ===")
    p, s = run_study("lgbm", make_objective(lgbm_builder), 25)
    results["lgbm"] = {"params": p, "wmse_3fold": s}

    print("\n=== XGB (GPU, yeni rejim) 15 trial ===")
    p, s = run_study("xgb", make_objective(xgb_builder), 15)
    results["xgb"] = {"params": p, "wmse_3fold": s}

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nKAYDEDILDI -> {OUT_JSON}")
