"""
TIER-1 SENTETIK-VERI FORENSIGI — hedef (career_success_score) URETIM formulunu geri-muhendislik.
=================================================================================================

Amac: 85.4945 rw-OOF duvarini kiracak YAPISAL icgoru. Bu DOSYA SADECE ANALIZ (feature uretmez,
hicbir cikti modele girmez). Analiz tum-train'de calisir (anlama amaci) — leakage YASAK
(feature olarak hicbir hedef-tureviolusturulmaz; burasi yalniz teshis).

Calistir:  python src/forensics.py    (sonuclari stdout + reports/forensics_dump.json)

Bolumler:
  A. CENSORING: hedef = clip(latent, 0, 100) mi? ==100 kutle + ==0 + residual asimetrisi.
  B. NOISE FLOOR: guclu modelin (LGBM full) residual-var'i -> indirgenemez gurultu tahmini.
  C. LATENT FORMUL: lineer/agirlikli feature kombinasyonu near-exact mi? OLS R^2, segment-ici.
  D. METIN ROLU: sayisal-sonrasi residual'i metin ne kadar aciklar (txt_ridge OOF korelasyonu).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SEED = 42

TARGET = "career_success_score"
ID = "student_id"
TEXT = "mentor_feedback_text"
CAT = ["department", "university_tier", "target_role", "hobby", "preferred_social_media_platform"]
YEARS = ["application_year", "graduation_year"]


def load():
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    te = pd.read_csv(DATA / "test_x.csv", encoding="utf-8-sig")
    return tr, te


def numeric_cols(df):
    drop = {ID, TARGET, TEXT, *CAT}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    np.random.seed(SEED)
    tr, te = load()
    y = tr[TARGET].values.astype(float)
    n = len(y)
    dump = {}

    # ------------------------------------------------------------------ #
    section("A. CENSORING / TARGET DISTRIBUTION FINGERPRINT")
    # ------------------------------------------------------------------ #
    eq100 = np.isclose(y, 100.0)
    eq0 = np.isclose(y, 0.0)
    n100, n0 = int(eq100.sum()), int(eq0.sum())
    print(f"n={n}  ==100: {n100} ({100*n100/n:.2f}%)   ==0: {n0} ({100*n0/n:.3f}%)")
    print(f"max={y.max():.6f}  min={y.min():.6f}  mean={y.mean():.4f}  median={np.median(y):.4f}")
    # rounding fingerprint: kac ondalik basamak?
    frac = np.round(y - np.floor(y), 6)
    n_int = int(np.isclose(frac, 0).sum())
    # En sik kullanilan ondalik adimi (0.01 grid mi?)
    times100 = np.round(y * 100)
    on_001_grid = int(np.isclose(y * 100, times100, atol=1e-6).sum())
    times10 = np.round(y * 10)
    on_01_grid = int(np.isclose(y * 10, times10, atol=1e-6).sum())
    print(f"tam-sayi deger: {n_int} ({100*n_int/n:.2f}%)  | 0.1-grid: {on_01_grid} | 0.01-grid: {on_001_grid} ({100*on_001_grid/n:.2f}%)")
    # 99-100 araligi yogunlugu (censoring imzasi: 100'e dogru ASIRI yigilma, 99-100 arasi 'cukur' degil)
    for lo, hi in [(98, 99), (99, 99.99), (99.99, 100)]:
        m = (y >= lo) & (y < hi)
        print(f"  [{lo},{hi}): {int(m.sum())}")
    # 100 oncesi son bin'ler: censoring ise 100-epsilon araligi BOS olur (kutle 100'e atlamis)
    near = y[(y >= 95) & (y < 100)]
    print(f"  [95,100): {len(near)} satir, [99.5,100): {int(((y>=99.5)&(y<100)).sum())}")
    dump["censoring"] = dict(n=n, n_eq_100=n100, n_eq_0=n0, max=float(y.max()), min=float(y.min()),
                             on_001_grid=on_001_grid, n_int=n_int,
                             gap_99_5_to_100=int(((y >= 99.5) & (y < 100)).sum()))

    # ------------------------------------------------------------------ #
    section("C. LATENT FORMULA — LINEAR / WEIGHTED RECONSTRUCTION (OLS)")
    # ------------------------------------------------------------------ #
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    num = numeric_cols(tr)
    # yil-disi sayisal + yillar
    Xnum = tr[num].copy()
    # impute median (sadece OLS teshisi icin; fold-safe degil ama burada SIZINTI YOK cunku feature uretmiyoruz)
    Xnum = Xnum.fillna(Xnum.median())
    sc = StandardScaler()
    Xs = sc.fit_transform(Xnum)

    # OLS: latent ~ tum sayisal. R^2 ve residual var.
    ols = LinearRegression().fit(Xs, y)
    pred_ols = ols.predict(Xs)
    r2_ols = ols.score(Xs, y)
    res_ols = y - pred_ols
    print(f"OLS (tum {len(num)} sayisal, in-sample) R^2={r2_ols:.5f}  resid_std={res_ols.std():.4f}  resid_var={res_ols.var():.4f}")

    # Censored-aware: 100'e clip edilmis satirlari cikar, latent>100 olabilir; sadece ic-aralik (0<y<100)
    interior = (y > 0.5) & (y < 99.5)
    ols_in = LinearRegression().fit(Xs[interior], y[interior])
    r2_in = ols_in.score(Xs[interior], y[interior])
    res_in = y[interior] - ols_in.predict(Xs[interior])
    print(f"OLS (interior 0.5<y<99.5, n={int(interior.sum())}) R^2={r2_in:.5f}  resid_std={res_in.std():.4f}")

    # Standardize edilmis katsayilar (en buyuk |coef| = en guclu lineer katki) — formul izi
    coefs = pd.Series(ols_in.coef_, index=num).sort_values(key=np.abs, ascending=False)
    print("\nEn guclu 20 lineer katki (interior OLS, standardize coef):")
    for name, c in coefs.head(20).items():
        print(f"  {c:+8.4f}  {name}")
    dump["ols"] = dict(r2_all=float(r2_ols), resid_std_all=float(res_ols.std()),
                       r2_interior=float(r2_in), resid_std_interior=float(res_in.std()),
                       top_coefs={k: float(v) for k, v in coefs.head(25).items()})

    # ------------------------------------------------------------------ #
    section("B. NOISE FLOOR — STRONG NONLINEAR MODEL (LGBM) RESIDUAL VARIANCE")
    # ------------------------------------------------------------------ #
    # in-sample LGBM ile near-deterministik mi? (in-sample R^2 ~1 ise gurultu az; <1 ise gercek gurultu floor)
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    Xlg = tr[num].copy()  # NaN korunur (LGBM native)
    # 5-fold OOF (in-sample degil) ile DURUST noise floor — guclu HP
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(n)
    params = dict(n_estimators=2000, learning_rate=0.02, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                  random_state=SEED, n_jobs=4, verbosity=-1)
    for tri, vai in kf.split(Xlg):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xlg.iloc[tri], y[tri], eval_set=[(Xlg.iloc[vai], y[vai])], eval_metric="l2",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[vai] = m.predict(Xlg.iloc[vai], num_iteration=m.best_iteration_)
    oof_clip = np.clip(oof, 0, 100)
    mse_oof = float(np.mean((y - oof_clip) ** 2))
    print(f"LGBM num-only 5-fold OOF MSE (clip)={mse_oof:.4f}  (unweighted, vanilla KFold)")
    res = y - oof_clip
    print(f"OOF residual std={res.std():.4f}  var={res.var():.4f}")
    # residual asimetrisi: 100-grubunda residual nasil? (censoring -> orada residual<=0 baskin beklenir)
    print(f"  residual mean @==100: {res[eq100].mean():+.4f} (censoring ise model 100'un altini tahmin -> res>0)")
    print(f"  residual mean @interior: {res[interior].mean():+.4f}")
    print(f"  residual skew: {pd.Series(res).skew():.4f}  kurtosis: {pd.Series(res).kurtosis():.4f}")
    dump["noise_floor"] = dict(lgbm_oof_mse=mse_oof, resid_std=float(res.std()),
                               resid_mean_at_100=float(res[eq100].mean()),
                               resid_mean_interior=float(res[interior].mean()),
                               resid_skew=float(pd.Series(res).skew()))

    # ------------------------------------------------------------------ #
    section("C2. SEGMENT-ICI DETERMINIZM (target_role / department)")
    # ------------------------------------------------------------------ #
    for seg in ["target_role", "department", "university_tier"]:
        g = tr.groupby(seg)[TARGET].agg(["mean", "std", "count"])
        print(f"\n{seg}:")
        print(g.round(3).to_string())

    # ------------------------------------------------------------------ #
    # NOTE: B-residual ve metin rolu Bolum D'de ayri scriptte (txt_ridge OOF gerekir).
    # ------------------------------------------------------------------ #
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "forensics_dump.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"\n[forensics] dump -> reports/forensics_dump.json")
    # OOF'u sonraki bolum (metin rolu) icin diske at
    np.save(REPORTS / "_forensics_numoof.npy", oof_clip)
    np.save(REPORTS / "_forensics_olspred.npy", pred_ols)


if __name__ == "__main__":
    main()
