"""
FORENSICS PART 4 — exact-weight recovery + noise law + interaction probe.
==========================================================================
1. EXACT WEIGHTS: latent = a + sum(w_k * feature_k) + noise mi? Ham (standardize-disi) OLS
   katsayilari TEMIZ/yuvarlak mi (sentetik formul imzasi)? Skor-feature'lari [0,100] olcekte mi?
   feature'lari 0-1'e normalize edip agirlik topla -> agirliklar ~yuvarlak/orantili mi?
2. NOISE LAW: interior residual'in dagilimi normal mi? std, skew, kurtosis, QQ tipi olcum.
   Heteroskedastik: noise std ~ a - b*latent (lineer mi?).
3. INTERACTION PROBE: en guclu carpimsal terim (pq*tech vb.) latent'i lineer'den iyi mi
   acikliyor? GBDT zaten yakaliyor; ama formul carpimsal mi additive mi anlamak icin.
4. RANK-BASED: latent monotonik tek-index mi (g(linear score))? Spearman vs Pearson.
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


def main():
    np.random.seed(SEED)
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    y = tr[TARGET].values.astype(float)
    n = len(y)
    num = numeric_cols(tr)
    dump = {}

    from sklearn.linear_model import LinearRegression
    from scipy import stats

    # skor-tipi feature'lar (0-100 olcek) ile sayim/diger ayrimi
    print("== FEATURE OLCEKLERI (min/max/mean) ==")
    desc = tr[num].describe().T[["min", "max", "mean"]]
    print(desc.round(2).to_string())
    # 0-100 skor feature'lari
    score_like = [c for c in num if tr[c].max() <= 100.5 and tr[c].min() >= -0.5 and tr[c].max() > 10]
    print(f"\n0-100 skor-benzeri feature ({len(score_like)}): {score_like}")

    # ---------------------------------------------------------------- #
    print("\n== 1. EXACT WEIGHT RECOVERY (interior, ham olcek OLS) ==")
    interior = (y > 0.5) & (y < 99.5)
    Xnum = tr[num].fillna(tr[num].median())
    ols = LinearRegression().fit(Xnum[interior], y[interior])
    coef = pd.Series(ols.coef_, index=num)
    print(f"  intercept = {ols.intercept_:.4f}")
    print("  ham katsayilar (en buyuk |.| 25):")
    for nm, c in coef.sort_values(key=np.abs, ascending=False).head(25).items():
        rng = tr[nm].max() - tr[nm].min()
        print(f"    {c:+9.5f}  (x range {rng:8.2f}, katki~{c*rng:+7.2f})  {nm}")

    # skor-feature'larin agirliklarini topla; ~esit mi? (ortalama-of-scores formulu mu?)
    sc_coef = coef[score_like]
    print(f"\n  skor-feature agirlik toplami = {sc_coef.sum():.4f}; ortalama = {sc_coef.mean():.5f}")
    print(f"  skor agirliklar std/mean = {sc_coef.std()/abs(sc_coef.mean()):.3f} (kucuk->esit-agirlik formulu izi)")
    dump["interior_intercept"] = float(ols.intercept_)
    dump["score_weight_sum"] = float(sc_coef.sum())
    dump["score_weight_cv"] = float(sc_coef.std() / abs(sc_coef.mean()))

    # ---------------------------------------------------------------- #
    print("\n== 2. NOISE LAW (interior residual) ==")
    res = y[interior] - ols.predict(Xnum[interior])
    print(f"  std={res.std():.4f}  skew={stats.skew(res):.4f}  kurtosis(excess)={stats.kurtosis(res):.4f}")
    print(f"  Shapiro benzeri: |skew|<0.2 ve |kurt|<0.5 -> ~normal gurultu")
    # heteroskedastik: |res| ~ latent? bin'le
    latent_hat = ols.predict(Xnum)
    q = pd.qcut(latent_hat[interior], 8, duplicates="drop")
    hb = pd.DataFrame({"lat": latent_hat[interior], "absres": np.abs(res)}).groupby(q, observed=True).agg(
        lat_mid=("lat", "mean"), res_std=("absres", lambda v: v.std()), n=("absres", "size"))
    print(hb.round(3).to_string())
    # lineer fit: res_std ~ a + b*lat
    sl = np.polyfit(hb["lat_mid"], hb["res_std"], 1)
    print(f"  res_std ~ {sl[1]:.3f} + ({sl[0]:.4f})*latent  (negatif egim -> yuksek latent az gurultu)")
    dump["noise"] = dict(std=float(res.std()), skew=float(stats.skew(res)),
                         kurt=float(stats.kurtosis(res)), het_slope=float(sl[0]))

    # ---------------------------------------------------------------- #
    print("\n== 3. ADDITIVE vs MULTIPLICATIVE (pq * tech etkisi lineer ustune) ==")
    TECH = ["coding_score","problem_solving_score","data_structures_score","sql_score",
            "machine_learning_score","backend_score","frontend_score","cloud_score","devops_score"]
    tech_mean = tr[TECH].mean(axis=1)
    pq = tr["project_quality_score"]
    base_feats = Xnum.copy()
    # lineer base R2
    r2_lin = LinearRegression().fit(base_feats[interior], y[interior]).score(base_feats[interior], y[interior])
    # + carpim
    base2 = base_feats.copy(); base2["pq_x_tech"] = pq * tech_mean
    r2_x = LinearRegression().fit(base2[interior], y[interior]).score(base2[interior], y[interior])
    print(f"  lineer R^2={r2_lin:.5f}  +pq*tech R^2={r2_x:.5f}  (delta {r2_x-r2_lin:+.5f})")
    # full carpim seti
    for nm, col in [("pq_x_tech", pq*tech_mean),
                    ("tech_sq", tech_mean**2),
                    ("pq_sq", pq**2)]:
        b = base_feats.copy(); b[nm] = col
        r2 = LinearRegression().fit(b[interior], y[interior]).score(b[interior], y[interior])
        print(f"    +{nm}: R^2={r2:.5f} ({r2-r2_lin:+.5f})")

    # ---------------------------------------------------------------- #
    print("\n== 4. TEK-INDEX / MONOTONIK? (Spearman latent_hat vs y) ==")
    sp = stats.spearmanr(latent_hat[interior], y[interior]).correlation
    pe = stats.pearsonr(latent_hat[interior], y[interior])[0]
    print(f"  Pearson={pe:.4f}  Spearman={sp:.4f}  (Spearman>>Pearson -> monoton nonlineer link)")
    dump["pearson"] = float(pe); dump["spearman"] = float(sp); dump["r2_lin"] = float(r2_lin); dump["r2_x"] = float(r2_x)

    (REPORTS / "forensics_dump4.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print("\n[forensics4] dump -> reports/forensics_dump4.json")


if __name__ == "__main__":
    main()
