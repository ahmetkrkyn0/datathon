"""
Super-blend v11 — tum tariflerin OOF/test tahminlerini birlestirir.

Uyeler: preds_v7 (5: lgbm,xgb,cat,et,mlp) + preds_v9 (5: yil-norm lgbm,
xgb,cat,mlp + nn) + preds_r3 (3: huber, quant, catmae) = 13 model vektoru.
NNLS (agirlikli) + nested yil-kalibrasyon -> submissions/submission_v11.csv

Calistir: python -u src/super_blend.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v11.csv"

RECIPES = {
    "v7": ["lgbm", "xgb", "cat", "et", "mlp"],
    "v9": ["lgbm", "xgb", "cat", "mlp", "nn"],
    "r3": ["huber", "quant", "catmae"],
}


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def main():
    data = {r: np.load(CACHE / f"preds_{r}.npz") for r in RECIPES}
    y = data["v7"]["y"]
    w = data["v7"]["w_fit"]
    yr = data["v7"]["years"]

    cols_oof, cols_te, labels = [], [], []
    for r, names in RECIPES.items():
        for n in names:
            cols_oof.append(data[r][f"oof_{n}"])
            cols_te.append(data[r][f"test_{n}"])
            labels.append(f"{r}_{n}")
    M = np.column_stack(cols_oof)
    Mte = np.column_stack(cols_te)
    print(f"{len(labels)} uye: {labels}")

    sw = np.sqrt(w)
    wb, _ = nnls(M * sw[:, None], y * sw)
    wb /= wb.sum()
    ens = np.clip(M @ wb, 0, 100)
    s0 = wmse(y, ens, w)
    print("\n=== NNLS AGIRLIKLARI (>0.01) ===")
    for l, wi in zip(labels, wb):
        if wi > 0.01:
            print(f"  {l:12s} {wi:.3f}")
    print(f"NNLS super-blend: {s0:.4f}")

    # nested yil-kalibrasyon
    kf = KFold(10, shuffle=True, random_state=42)
    cal = ens.copy()
    for tr_i, va_i in kf.split(ens):
        for yil in np.unique(yr):
            mt = tr_i[yr[tr_i] == yil]
            mv = va_i[yr[va_i] == yil]
            if len(mt) > 50 and len(mv) > 0:
                b, a = np.polyfit(ens[mt], y[mt], 1)
                cal[mv] = a + b * ens[mv]
    cal = np.clip(cal, 0, 100)
    s1 = wmse(y, cal, w)
    use_cal = s1 < s0
    print(f"+ yil-kalibrasyon: {s1:.4f} ({'UYGULANIYOR' if use_cal else 'atlandi'})")

    test = pd.read_csv(ROOT / "data" / "test_x.csv")
    final = np.clip(Mte @ wb, 0, 100)
    if use_cal:
        te_years = test["application_year"].values
        for yil in np.unique(yr):
            mt = yr == yil
            me = te_years == yil
            if mt.sum() > 50:
                b, a = np.polyfit(ens[mt], y[mt], 1)
                final[me] = a + b * final[me]
        final = np.clip(final, 0, 100)
    final = final.round(3)

    sub = pd.DataFrame({"student_id": test["student_id"],
                        "career_success_score": final})
    sub.to_csv(OUT, index=False)
    score = min(s0, s1)
    print(f"\nYAZILDI -> {OUT}")
    print(f"proxy ~{score:.2f} | LB beklentisi ~{score - 0.5:.1f}-{score:.1f}")
    print(f"satir: {len(sub)} | aralik: {final.min()}-{final.max()} | ort: {final.mean():.2f}")


if __name__ == "__main__":
    main()
