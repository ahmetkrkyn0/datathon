"""
FORENSICS PART 2 — censoring exploit test + text-residual role + noise-floor refinement.
=========================================================================================
Calistir:  python src/forensics2.py

1. METIN ROLU: sayisal-OOF residual'i metin (TF-IDF->Ridge OOF) ne kadar aciklar?
   num-only OOF MSE vs num+text OOF MSE -> metnin marjinal MSE katkisi = noise-floor'un buyuk
   parcasi metinde mi?
2. CENSORING EXPLOIT: oof-residual @==100 = +6.23 -> model 100'leri 100'e itmiyor. Iki test:
   (a) ORACLE tavan: gercek ==100 satirlarini 100'e zorla -> MSE ne kadar duser? (ust sinir)
   (b) Gercekci: P(y==100) siniflandirici (zaten AUC 0.958) -> kalibre push.
   NOT: bunlar zaten LEVERS_SUMMARY'de denendi (blend_p100 -0.20, kapidan uzak). Burada
   num-only tabanda re-olc + Tobit yaklasimi degerlendir.
3. TOBIT / CENSORED: latent>100 olabilen censored-normal varsayimi. Basit yaklasim: y'yi
   100'de censored kabul edip, OLS yerine ust-kuyruk agirlikli fit. Pratik exploit var mi?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SEED = 42
TARGET, ID, TEXT = "career_success_score", "student_id", "mentor_feedback_text"
CAT = ["department", "university_tier", "target_role", "hobby", "preferred_social_media_platform"]


def numeric_cols(df):
    drop = {ID, TARGET, TEXT, *CAT}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def tr_lower(s: pd.Series) -> pd.Series:
    return s.str.replace("I", "ı").str.replace("İ", "i").str.lower()


def main():
    np.random.seed(SEED)
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    y = tr[TARGET].values.astype(float)
    n = len(y)
    eq100 = np.isclose(y, 100.0)
    num = numeric_cols(tr)
    dump = {}

    import lightgbm as lgb
    from sklearn.model_selection import KFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack, csr_matrix

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- text OOF (TF-IDF -> Ridge), fold-safe ----
    print("== TEXT OOF (TF-IDF char+word -> Ridge), fold-safe ==")
    txt = tr_lower(tr[TEXT].astype(str))
    txt_oof = np.zeros(n)
    for tri, vai in kf.split(txt):
        wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
        cvz = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=40000, sublinear_tf=True)
        Xw_tr = wv.fit_transform(txt.iloc[tri]); Xw_va = wv.transform(txt.iloc[vai])
        Xc_tr = cvz.fit_transform(txt.iloc[tri]); Xc_va = cvz.transform(txt.iloc[vai])
        Xtr = hstack([Xw_tr, Xc_tr]).tocsr(); Xva = hstack([Xw_va, Xc_va]).tocsr()
        rg = Ridge(alpha=2.0, random_state=SEED).fit(Xtr, y[tri])
        txt_oof[vai] = rg.predict(Xva)
    txt_oof_c = np.clip(txt_oof, 0, 100)
    mse_txt = float(np.mean((y - txt_oof_c) ** 2))
    print(f"  text-only OOF MSE (clip) = {mse_txt:.4f}")

    # ---- num OOF (reuse part1 if present) ----
    num_oof_path = REPORTS / "_forensics_numoof.npy"
    if num_oof_path.exists():
        num_oof = np.load(num_oof_path)
        print(f"  (num-only OOF loaded; MSE={np.mean((y-num_oof)**2):.4f})")
    else:
        num_oof = None

    # ---- num + text as meta (LGBM on num + txt_oof) ----
    print("\n== NUM + TEXT(meta) OOF ==")
    Xlg = tr[num].copy()
    params = dict(n_estimators=2000, learning_rate=0.02, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                  random_state=SEED, n_jobs=4, verbosity=-1)
    full_oof = np.zeros(n)
    # nested txt_oof: kullanmadan once her dis-fold icin AYRI txt_oof gerekir; basitlik icin
    # burada txt_oof (5-fold OOF) zaten leak-safe (her satir gorulmedigi fold'dan) -> meta olarak OK.
    Xmeta = Xlg.copy(); Xmeta["__txt"] = txt_oof  # ham txt_oof (clip'siz) meta-feature
    for tri, vai in kf.split(Xmeta):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xmeta.iloc[tri], y[tri], eval_set=[(Xmeta.iloc[vai], y[vai])], eval_metric="l2",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        full_oof[vai] = m.predict(Xmeta.iloc[vai], num_iteration=m.best_iteration_)
    full_oof_c = np.clip(full_oof, 0, 100)
    mse_full = float(np.mean((y - full_oof_c) ** 2))
    print(f"  num+text OOF MSE (clip) = {mse_full:.4f}")
    if num_oof is not None:
        print(f"  metnin marjinal MSE katkisi: {np.mean((y-num_oof)**2) - mse_full:+.4f}")
    res_full = y - full_oof_c
    print(f"  num+text residual std={res_full.std():.4f}  var={res_full.var():.4f}")

    dump["text_role"] = dict(text_only_mse=mse_txt, num_text_mse=mse_full,
                             residual_std_full=float(res_full.std()))

    # ---- CENSORING EXPLOIT TESTS (on num+text base) ----
    print("\n== CENSORING EXPLOIT (num+text base) ==")
    # (a) ORACLE tavan: gercek 100'leri 100'e zorla
    oracle = full_oof_c.copy(); oracle[eq100] = 100.0
    mse_oracle = float(np.mean((y - oracle) ** 2))
    print(f"  (a) ORACLE (gercek 100->100): MSE {mse_full:.4f} -> {mse_oracle:.4f}  (delta {mse_oracle-mse_full:+.4f}) [ULASILMAZ TAVAN]")

    # (b) residual @100 — model ne kadar dusuk tahmin ediyor?
    print(f"  (b) residual mean @==100 = {res_full[eq100].mean():+.4f}  (pozitif -> model 100'lerin altinda)")
    # (c) tum >=Q tahminleri 100'e push'lamanin etkisi (kaba): threshold tarama
    print(f"  (c) push-yuksek-tahmin tarama (pred>=thr -> 100):")
    best = (mse_full, None)
    for thr in [90, 92, 94, 95, 96, 97, 98]:
        pp = full_oof_c.copy()
        mask = full_oof_c >= thr
        pp[mask] = 100.0
        msep = float(np.mean((y - pp) ** 2))
        hit = float(eq100[mask].mean()) if mask.sum() else 0.0  # push'lananlarin gercek-100 orani
        print(f"     thr={thr}: push {int(mask.sum())} satir (precision={hit:.3f}) -> MSE {msep:.4f} ({msep-mse_full:+.4f})")
        if msep < best[0]:
            best = (msep, thr)
    print(f"  -> en iyi naif push: thr={best[1]} MSE={best[0]:.4f} ({best[0]-mse_full:+.4f})")

    # ---- TOBIT yaklasimi: latent reconstruction (censored OLS, statsmodels yoksa elle) ----
    print("\n== TOBIT / censored-aware OLS reconstruction ==")
    from sklearn.preprocessing import StandardScaler
    Xnum = tr[num].fillna(tr[num].median())
    Xs = StandardScaler().fit_transform(Xnum)
    # Tobit MLE (simple, scipy)
    from scipy import optimize, stats
    Xd = np.column_stack([np.ones(n), Xs])
    cens = eq100.astype(float)  # 1 if censored at 100

    def negll(params):
        beta = params[:-1]; logs = params[-1]; s = np.exp(logs)
        mu = Xd @ beta
        # uncensored: normal pdf; censored at 100: 1 - Phi((100-mu)/s)
        ll = np.empty(n)
        unc = cens == 0
        z = (y[unc] - mu[unc]) / s
        ll[unc] = -0.5 * np.log(2 * np.pi) - logs - 0.5 * z ** 2
        zc = (100.0 - mu[~unc.astype(bool) if False else (cens == 1)]) / s
        ll[cens == 1] = stats.norm.logsf(zc)
        return -ll.sum()

    # init from OLS
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression(fit_intercept=False).fit(Xd, y)
    p0 = np.concatenate([ols.coef_, [np.log(10.0)]])
    try:
        res = optimize.minimize(negll, p0, method="L-BFGS-B", options=dict(maxiter=300))
        beta = res.x[:-1]; s = np.exp(res.x[-1])
        mu_tobit = Xd @ beta
        # Tobit prediction = E[y|x] under censoring = mu*Phi + s*phi + 100*(1-Phi)
        z = (100 - mu_tobit) / s
        Phi = stats.norm.cdf(z); phi = stats.norm.pdf(z)
        ey = mu_tobit * Phi + s * phi + 100 * (1 - Phi)  # E[min(latent,100)]
        ey = np.clip(ey, 0, 100)
        mse_tobit_insample = float(np.mean((y - ey) ** 2))
        print(f"  Tobit MLE in-sample: sigma={s:.3f}, MSE(E[min(lat,100)])={mse_tobit_insample:.4f}")
        print(f"  (vs OLS in-sample interior R^2~0.52; Tobit modelle latent>100 acikca modellenir)")
        # Tobit'in 100-grubunu ne kadar yakaladigi
        print(f"  Tobit E[y] @==100 ortalama = {ey[eq100].mean():.4f} (100'e ne kadar yakin?)")
        dump["tobit"] = dict(sigma=float(s), insample_mse=mse_tobit_insample,
                             ey_at_100=float(ey[eq100].mean()))
    except Exception as e:
        print(f"  Tobit fit hata: {e}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "forensics_dump2.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    np.save(REPORTS / "_forensics_txtoof.npy", txt_oof_c)
    np.save(REPORTS / "_forensics_fulloof.npy", full_oof_c)
    print("\n[forensics2] dump -> reports/forensics_dump2.json")


if __name__ == "__main__":
    main()
