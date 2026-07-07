"""XGBoost tuning (GPU) — best_params.json'a 'xgb' anahtari ekler.

Calistir: python -u src/tune_xgb.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import optuna
from sklearn.model_selection import KFold

import features as F

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "cache" / "best_params.json"
SEED = 42

print("Feature'lar yukleniyor...")
train, test, y, w_fit, num_cols = F.build_features()
train["txt_bert"] = np.load(ROOT / "data" / "cache" / "bert_oof.npy")
num_cols = num_cols + ["txt_bert"]

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
FOLDS = list(kf.split(train))
X = train[num_cols]


def wmse(yy, p, w):
    return float(np.average((yy - np.clip(p, 0, 100)) ** 2, weights=w))


def objective(trial):
    from xgboost import XGBRegressor
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 800, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 9),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        random_state=SEED, n_jobs=8, tree_method="hist", device="cuda",
    )
    oof = np.zeros(len(train))
    for i, (tr_idx, va_idx) in enumerate(FOLDS):
        m = XGBRegressor(**params)
        m.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=w_fit[tr_idx])
        oof[va_idx] = m.predict(X.iloc[va_idx])
        trial.report(wmse(y[va_idx], oof[va_idx], w_fit[va_idx]), i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return wmse(y, oof, w_fit)


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", pruner=pruner,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    def cb(study, trial):
        s = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        print(f"  [xgb] trial {trial.number:>2}: {s}  (best={study.best_value:.4f})")
    study.optimize(objective, n_trials=20, callbacks=[cb])
    print(f"[xgb] EN IYI: {study.best_value:.4f}")
    print(f"[xgb] params: {study.best_params}")

    results = json.loads(OUT_JSON.read_text())
    results["xgb"] = {"params": study.best_params, "wmse_3fold": study.best_value}
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"KAYDEDILDI -> {OUT_JSON}")
