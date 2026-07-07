"""
FORENSICS PART 6 — near-deterministic subset karakteri + FOLD-SAFE nested isotonic testi.
==========================================================================================
1. |resid|<0.5 olan 838 satir KIM? (==100 grubu mu, yoksa ayri deterministik kume mi?)
   Eger cogu ==100 ise -> censoring artefakti, formul izi DEGIL. Degilse -> incele.
2. NESTED ISOTONIC: blend OOF'u isotonic ile recalibrate etmek fold-safe (nested) gercek mi?
   Her dis-hucre DISINDA fit edilen isotonic ile rw-OOF. in-sample -1.35'in ne kadari gercek?
   KARAR = nested rw-OOF; kapi = 85.4945 - 0.25*3.0238 = 84.7385.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; ART = ROOT / "artifacts"; REPORTS = ROOT / "reports"
TARGET = "career_success_score"


def main():
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    te = pd.read_csv(DATA / "test_x.csv", encoding="utf-8-sig")
    y = tr[TARGET].values.astype(float)
    n = len(y)
    sid = tr[cv.ID_COL].values
    w = cv.recency_weights(tr, te)
    folds = cv.load_folds()
    blend = np.clip(np.load(ART / "oof_blend.npy"), 0, 100)
    eq100 = np.isclose(y, 100.0)
    res = y - blend

    # ---------------------------------------------------------------- #
    print("== 1. NEAR-DETERMINISTIC SUBSET KARAKTERI (|resid|<0.5, 838 satir) ==")
    near0 = np.abs(res) < 0.5
    print(f"  near0 toplam: {int(near0.sum())}")
    print(f"  bunlarin ==100 olani: {int((near0 & eq100).sum())} ({100*(near0&eq100).sum()/near0.sum():.1f}%)")
    print(f"  near0 & interior (0.5<y<99.5): {int((near0 & (y>0.5) & (y<99.5)).sum())}")
    # interior near0'larin y dagilimi — belirli degerlerde mi yiginli?
    int_near0 = near0 & (y > 0.5) & (y < 99.5)
    print(f"  interior near0 y ornekleri (ilk 20): {np.round(np.sort(y[int_near0])[:20],2)}")
    # interior'da near0 orani sans-ustu mu? bin bazli kontrol (yuksek pred'de noise az -> beklenir)
    q = pd.qcut(blend, 10, duplicates="drop")
    nb = pd.DataFrame({"pred": blend, "near0": near0.astype(int), "is100": eq100.astype(int)}).groupby(q, observed=True).agg(
        pred_mid=("pred", "mean"), near0_rate=("near0", "mean"), is100_rate=("is100", "mean"), n=("near0", "size"))
    print(nb.round(3).to_string())
    print("  -> near0 yuksek-pred bin'lerinde yiginli VE is100 ile ortusuyorsa: censoring+heterosked artefakti")

    # ---------------------------------------------------------------- #
    print("\n== 2. NESTED ISOTONIC (fold-safe) — gercek iyilesme var mi? ==")
    from sklearn.isotonic import IsotonicRegression
    # nested: her (repeat,fold) hucresi icin, o hucre DISINDAki satirlarda isotonic fit -> hucreyi map'le
    # blend zaten OOF (her satir gorulmedigi modelden). isotonic recalib map'i de leak-safe olmali:
    # hucre i'nin map'i, i HARIC satirlardan fit.
    rw_wall = cv.compute_recency_weighted_mse(blend, y, w)
    print(f"  WALL rw = {rw_wall:.4f}  kapi = {rw_wall - 0.25*3.023803:.4f}")

    recals = []
    for r in sorted(folds["repeat"].unique()):
        fold_of = cv.fold_of_rows(folds, sid, r)
        cal_r = np.zeros(n)
        for f in sorted(folds["fold"].unique()):
            val = np.where(fold_of == f)[0]
            fit = np.where(fold_of != f)[0]
            iso = IsotonicRegression(out_of_bounds="clip").fit(blend[fit], y[fit])
            cal_r[val] = iso.predict(blend[val])
        recals.append(cal_r)
    cal = np.clip(np.mean(recals, axis=0), 0, 100)
    rw_iso = cv.compute_recency_weighted_mse(cal, y, w)
    uw_iso = float(np.mean((y - cal) ** 2))
    print(f"  NESTED isotonic rw = {rw_iso:.4f}  (delta {rw_iso - rw_wall:+.4f})   uw={uw_iso:.4f}")
    gate = rw_wall - 0.25 * 3.023803
    verdict = "GECTI" if rw_iso < gate else "GECMEDI"
    print(f"  kabul kapisi ({gate:.4f}): {verdict}")

    # Ayrica: spline/poly degil saf monoton; bir de '100-clip agresif' (ust uca isotonic dogal yapar)
    out = dict(near0_total=int(near0.sum()), near0_is100=int((near0 & eq100).sum()),
               near0_interior=int((near0 & (y > 0.5) & (y < 99.5)).sum()),
               wall_rw=rw_wall, nested_iso_rw=rw_iso, nested_iso_delta=rw_iso - rw_wall,
               gate=gate, iso_passes=bool(rw_iso < gate))
    (REPORTS / "forensics_dump6.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n[forensics6] dump -> reports/forensics_dump6.json")


if __name__ == "__main__":
    main()
