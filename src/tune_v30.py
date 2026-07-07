"""
Re-tune (LGBM + CatBoost) — TAM guncel feature pipeline'iyla.

Harness: yn + uniform + segment-yil TE (fold-ici) + kohort-z (A seti)
+ 3 metin skalari. 3-fold, tek seed, agirlikli OOF hedefi.
LGBM 25 trial (CPU), CatBoost 20 trial (GPU). MedianPruner.
Cikti: data/cache/best_params_v30.json

Calistir: python -u src/tune_v30.py
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
OUT_JSON = CACHE / "best_params_v30.json"
SEED = 42
SEGS = ["target_role", "university_tier", "hobby",
        "preferred_social_media_platform"]
KEY5 = ["project_quality_score", "technical_interview_score",
        "technical_avg", "portfolio_avg", "coding_score"]

print("Feature'lar hazirlaniyor...")
train, test, y, w_fit, num_cols = F.build_features()
for s, f in [("txt_bert2", "bert2_oof.npy"), ("txt_bert3", "bert3_oof.npy"),
             ("txt_mdeb", "mdeb_oof.npy")]:
    train[s] = np.load(CACHE / f)
num_cols = num_cols + ["txt_bert2", "txt_bert3", "txt_mdeb"]
years = train["application_year"].values
roles = train["target_role"].astype(str).values

# kohort-z (A seti)
zcols = []
for c in KEY5:
    for by, name in [(years, "yil"), (roles, "rol"),
                     (list(zip(roles, years)), "rolyil")]:
        g = pd.DataFrame({"b": [str(b) for b in by], "v": train[c]})
        st = g.groupby("b")["v"].agg(["mean", "std"])
        col = f"{c}_z_{name}"
        train[col] = ((g["v"] - g["b"].map(st["mean"])) /
                      g["b"].map(st["std"]).replace(0, 1)).values
        zcols.append(col)
g = pd.DataFrame({"b": [f"{r}|{yy}" for r, yy in zip(roles, years)],
                  "v": train["role_skill"]})
st = g.groupby("b")["v"].agg(["mean", "std"])
train["role_skill_z_rolyil"] = ((g["v"] - g["b"].map(st["mean"])) /
                                g["b"].map(st["std"]).replace(0, 1)).values
zcols.append("role_skill_z_rolyil")
ALL_COLS = num_cols + zcols
print(f"  feature: {len(ALL_COLS)} + 4 seg-yil TE (fold-ici)")

kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
FOLDS = list(kf.split(train))


def cell_te(seg_tr, yil_tr, y_tr, seg_ap, yil_ap, sm=20):
    df = pd.DataFrame({"s": seg_tr, "yil": yil_tr, "y": y_tr})
    cell = df.groupby(["s", "yil"])["y"].agg(["mean", "count"])
    yil_mean = df.groupby("yil")["y"].mean()
    cell["te"] = (cell["mean"] * cell["count"]
                  + cell.index.get_level_values("yil").map(yil_mean) * sm) / (cell["count"] + sm)
    s = pd.Series(list(zip(seg_ap, yil_ap))).map(cell["te"])
    return s.fillna(pd.Series(yil_ap).map(yil_mean)).values


# fold basina sabit hazirliklar (her trial'da tekrar hesaplamamak icin)
FOLD_DATA = []
for tr_idx, va_idx in FOLDS:
    st = pd.DataFrame({"yil": years[tr_idx], "y": y[tr_idx]}).groupby("yil")["y"].agg(["mean", "std"])
    yn = (y[tr_idx] - pd.Series(years[tr_idx]).map(st["mean"]).values) / \
         pd.Series(years[tr_idx]).map(st["std"]).values
    mu_v = pd.Series(years[va_idx]).map(st["mean"]).values
    sd_v = pd.Series(years[va_idx]).map(st["std"]).values
    Xtr = train.iloc[tr_idx][ALL_COLS].copy()
    Xva = train.iloc[va_idx][ALL_COLS].copy()
    for scn in SEGS:
        seg = train[scn].astype(str).values
        Xtr[f"{scn}_yil_te"] = cell_te(seg[tr_idx], years[tr_idx], y[tr_idx],
                                       seg[tr_idx], years[tr_idx])
        Xva[f"{scn}_yil_te"] = cell_te(seg[tr_idx], years[tr_idx], y[tr_idx],
                                       seg[va_idx], years[va_idx])
    FOLD_DATA.append((Xtr, Xva, yn, va_idx, mu_v, sd_v))


def wmse(yy, p, w):
    return float(np.average((yy - np.clip(p, 0, 100)) ** 2, weights=w))


def make_objective(builder):
    def obj(trial):
        oof = np.zeros(len(train))
        for i, (Xtr, Xva, yn, va_idx, mu_v, sd_v) in enumerate(FOLD_DATA):
            m = builder(trial)
            m.fit(Xtr, yn)
            oof[va_idx] = m.predict(Xva) * sd_v + mu_v
            trial.report(wmse(y[va_idx], oof[va_idx], w_fit[va_idx]), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return wmse(y, oof, w_fit)
    return obj


def lgbm_builder(trial):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 600, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 24, 256, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        subsample_freq=1, random_state=SEED, n_jobs=8, verbose=-1)


def cat_builder(trial):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=trial.suggest_int("iterations", 800, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        depth=trial.suggest_int("depth", 5, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 15.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        task_type="GPU", devices="0", verbose=0,
        allow_writing_files=False, random_seed=SEED)


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
    print("\n=== LGBM (tam pipeline) 25 trial ===")
    p, s = run_study("lgbm", make_objective(lgbm_builder), 25)
    results["lgbm"] = {"params": p, "wmse_3fold": s}

    print("\n=== CatBoost GPU (tam pipeline) 20 trial ===")
    p, s = run_study("cat", make_objective(cat_builder), 20)
    results["cat"] = {"params": p, "wmse_3fold": s}

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nKAYDEDILDI -> {OUT_JSON}")
