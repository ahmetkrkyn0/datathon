"""
FORENSICS PART 3 — noise structure + text-residual recoverability + heteroskedasticity.
=========================================================================================
Sorular:
1. Metin, sayisal-SONRASI residual'i ne kadar aciklar? (latent = f(num)+g(text)+noise mi?)
   residual = y - num_oof ; bunu metinle (TF-IDF->Ridge OOF) tahmin et -> R^2.
   Eger metin residual'in BUYUK kismini aciyorsa -> daha guclu metin modeli (BERT?) lever.
2. Gurultu HETEROSKEDASTIK mi? residual-var feature'lara/tahmine bagli mi? (MSE altinda
   optimal tahmin yine kosullu ortalama -> heteroskedastisite exploit DEGIL ama floor'u anlatir.)
3. NOISE FLOOR alt-sinir: num+text+cat+all FE ile en guclu tek model OOF MSE (unweighted) ->
   gercek floor tahmini. 85.49'a (rw) ne kadar yakiniz?
4. Latent additive mi? num_oof + text_resid_pred toplami, joint modelden iyi mi? (additive yapı izi)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; REPORTS = ROOT / "reports"
SEED = 42
TARGET, ID, TEXT = "career_success_score", "student_id", "mentor_feedback_text"
CAT = ["department", "university_tier", "target_role", "hobby", "preferred_social_media_platform"]


def numeric_cols(df):
    drop = {ID, TARGET, TEXT, *CAT}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def tr_lower(s):
    return s.str.replace("I", "ı").str.replace("İ", "i").str.lower()


def main():
    np.random.seed(SEED)
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    y = tr[TARGET].values.astype(float)
    n = len(y)
    num = numeric_cols(tr)
    dump = {}

    from sklearn.model_selection import KFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack
    import lightgbm as lgb
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    num_oof = np.load(REPORTS / "_forensics_numoof.npy")       # num-only LGBM OOF (clip)
    txt_oof = np.load(REPORTS / "_forensics_txtoof.npy")       # text-only Ridge OOF (clip)
    full_oof = np.load(REPORTS / "_forensics_fulloof.npy")     # num+text(meta) OOF (clip)

    # ---------------------------------------------------------------- #
    print("== 1. METIN, SAYISAL-SONRASI RESIDUAL'I ACIKLIYOR MU? ==")
    # residual'i HAM num_oof'tan (clip oncesi yerine clip'li yeterli teshis icin)
    resid = y - num_oof   # num-sonrasi aciklanmamis
    print(f"  num-sonrasi residual std={resid.std():.4f}")
    # metinle residual'i tahmin et (fold-safe)
    txt = tr_lower(tr[TEXT].astype(str))
    resid_pred = np.zeros(n)
    for tri, vai in kf.split(txt):
        wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
        cvz = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=40000, sublinear_tf=True)
        Xtr = hstack([wv.fit_transform(txt.iloc[tri]), cvz.fit_transform(txt.iloc[tri])]).tocsr()
        Xva = hstack([wv.transform(txt.iloc[vai]), cvz.transform(txt.iloc[vai])]).tocsr()
        rg = Ridge(alpha=2.0, random_state=SEED).fit(Xtr, resid[tri])
        resid_pred[vai] = rg.predict(Xva)
    ss_res = np.sum((resid - resid_pred) ** 2); ss_tot = np.sum((resid - resid.mean()) ** 2)
    r2_resid = 1 - ss_res / ss_tot
    print(f"  metin -> num-residual R^2 = {r2_resid:.4f}  (metin residual'in bu kadarini aciyor)")
    # additive reconstruction: num_oof + resid_pred
    add = np.clip(num_oof + resid_pred, 0, 100)
    mse_add = float(np.mean((y - add) ** 2))
    print(f"  ADDITIVE (num_oof + text-on-residual) MSE = {mse_add:.4f}  vs joint num+text {np.mean((y-full_oof)**2):.4f}")
    dump["text_on_residual_r2"] = float(r2_resid)
    dump["additive_mse"] = mse_add

    # ---------------------------------------------------------------- #
    print("\n== 2. HETEROSKEDASTISITE: gurultu tahmine/feature'a bagli mi? ==")
    res_full = y - full_oof
    # tahmin bin'lerine gore residual std
    q = pd.qcut(full_oof, 10, duplicates="drop")
    het = pd.DataFrame({"pred": full_oof, "res": res_full, "abs": np.abs(res_full)}).groupby(q, observed=True).agg(
        res_std=("res", "std"), n=("res", "size"))
    print(het.round(3).to_string())
    # alt-aralikta (dusuk pred) gurultu daha mi buyuk?
    dump["hetero_pred_bins"] = {str(k): float(v) for k, v in het["res_std"].items()}

    # ---------------------------------------------------------------- #
    print("\n== 3. EN GUCLU TEK MODEL NOISE FLOOR (num+text+cat, all-in LGBM) ==")
    txt_oof_raw = txt_oof  # clip'li yeterli
    Xall = tr[num].copy()
    for c in CAT:
        Xall[c] = tr[c].astype("category")
    Xall["__txt"] = txt_oof_raw
    floor_oof = np.zeros(n)
    params = dict(n_estimators=3000, learning_rate=0.015, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=3.0,
                  random_state=SEED, n_jobs=4, verbosity=-1)
    for tri, vai in kf.split(Xall):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xall.iloc[tri], y[tri], eval_set=[(Xall.iloc[vai], y[vai])], eval_metric="l2",
              categorical_feature=CAT, callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
        floor_oof[vai] = m.predict(Xall.iloc[vai], num_iteration=m.best_iteration_)
    floor_oof = np.clip(floor_oof, 0, 100)
    mse_floor = float(np.mean((y - floor_oof) ** 2))
    print(f"  num+text+cat LGBM OOF MSE (unweighted) = {mse_floor:.4f}  residual std={np.std(y-floor_oof):.4f}")
    dump["best_single_floor_mse"] = mse_floor

    # ---------------------------------------------------------------- #
    print("\n== 4. CENSORING ETKISI floor uzerinde (interior-only MSE) ==")
    interior = (y > 0.5) & (y < 99.5)
    mse_interior = float(np.mean((y[interior] - floor_oof[interior]) ** 2))
    mse_100 = float(np.mean((y[np.isclose(y, 100)] - floor_oof[np.isclose(y, 100)]) ** 2))
    print(f"  interior MSE = {mse_interior:.4f} (n={int(interior.sum())})")
    print(f"  ==100 grubu MSE = {mse_100:.4f} (n={int(np.isclose(y,100).sum())})  -> bu grup floor'u yukseltiyor")
    print(f"  ==100 grubu MSE'nin toplam MSE'ye katkisi: {mse_100*np.isclose(y,100).sum()/n:.4f} / {mse_floor:.4f}")
    dump["interior_mse"] = mse_interior; dump["mse_at_100"] = mse_100

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "forensics_dump3.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print("\n[forensics3] dump -> reports/forensics_dump3.json")


if __name__ == "__main__":
    main()
