"""
Blend laboratuvari — preds_v4.npz uzerinde stacking stratejilerini kiyaslar.

train_model_v4.py'nin kaydettigi OOF/test tahminlerini yukler ve:
  1. NNLS blend (v4'un kullandigi — referans)
  2. Ridge meta-model (OOF tahminler + application_year + project_quality)
  3. LGBM meta-model (ayni girdiler, kucuk/regularize)
stratejilerini YIL-AGIRLIKLI MSE ile (nested CV, durust) kiyaslar.
En iyisi kazanirsa submission_v4b.csv yazar.

Calistir: python -u src/blend_lab.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v4b.csv"
SEED = 42
NAMES = ["lgbm", "xgb", "cat"]


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def main():
    d = np.load(CACHE / "preds_v4.npz")
    y, w = d["y"], d["w_fit"]
    M = np.column_stack([d[f"oof_{m}"] for m in NAMES])
    Mte = np.column_stack([d[f"test_{m}"] for m in NAMES])

    # meta-feature olarak yil + en guclu tabular sinyal
    train, test, _, _, _ = F.build_features()
    extra_tr = train[["application_year", "project_quality_score"]].fillna(0).values
    extra_te = test[["application_year", "project_quality_score"]].fillna(0).values

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- 1) NNLS (referans) ----
    sw = np.sqrt(w)
    wb, _ = nnls(M * sw[:, None], y * sw)
    wb /= wb.sum()
    nnls_oof = np.clip(M @ wb, 0, 100)
    print(f"NNLS blend             : agirlikli {wmse(y, nnls_oof, w):.4f} "
          f"(agirliklar {np.round(wb,3)})")

    # ---- 2) Ridge meta (nested CV) ----
    Xmeta = np.column_stack([M, extra_tr])
    ridge_oof = np.zeros(len(y))
    for tr, va in kf.split(Xmeta):
        r = Ridge(alpha=1.0)
        r.fit(Xmeta[tr], y[tr], sample_weight=w[tr])
        ridge_oof[va] = r.predict(Xmeta[va])
    ridge_oof = np.clip(ridge_oof, 0, 100)
    print(f"Ridge meta (+yil, +pq) : agirlikli {wmse(y, ridge_oof, w):.4f}")

    # ---- 3) LGBM meta (nested CV, kucuk) ----
    from lightgbm import LGBMRegressor
    lgbm_oof = np.zeros(len(y))
    for tr, va in kf.split(Xmeta):
        m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                          min_child_samples=50, reg_lambda=5.0,
                          random_state=SEED, verbose=-1, n_jobs=8)
        m.fit(Xmeta[tr], y[tr], sample_weight=w[tr])
        lgbm_oof[va] = m.predict(Xmeta[va])
    lgbm_oof = np.clip(lgbm_oof, 0, 100)
    print(f"LGBM meta              : agirlikli {wmse(y, lgbm_oof, w):.4f}")

    # ---- en iyi stratejiyle test tahmini ----
    scores = {
        "nnls": wmse(y, nnls_oof, w),
        "ridge": wmse(y, ridge_oof, w),
        "lgbm": wmse(y, lgbm_oof, w),
    }
    best = min(scores, key=scores.get)
    print(f"\nEN IYI: {best} ({scores[best]:.4f})")

    if best == "nnls":
        final = np.clip(Mte @ wb, 0, 100)
    else:
        Xmeta_te = np.column_stack([Mte, extra_te])
        if best == "ridge":
            mdl = Ridge(alpha=1.0).fit(Xmeta, y, sample_weight=w)
        else:
            mdl = LGBMRegressor(n_estimators=300, learning_rate=0.03,
                                num_leaves=15, min_child_samples=50,
                                reg_lambda=5.0, random_state=SEED,
                                verbose=-1, n_jobs=8)
            mdl.fit(Xmeta, y, sample_weight=w)
        final = np.clip(mdl.predict(Xmeta_te), 0, 100)

    final = final.round(3)
    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"YAZILDI -> {OUT} (strateji: {best})")
    print(f"aralik: {final.min()}-{final.max()} | ort: {final.mean():.2f}")


if __name__ == "__main__":
    main()
