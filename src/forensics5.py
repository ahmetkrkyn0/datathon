"""
FORENSICS PART 5 — gercek proje artefaktlari uzerinde noise-floor + recalibration testleri.
=============================================================================================
Gercek oof_blend / oof_lgbm_full / oof_catboost_full (proje en iyileri) ile:
1. Noise-floor teyit: bu OOF'larin residual yapisi forensics1-4 ile tutarli mi?
2. NEAR-DETERMINISTIC: residual~0 satir kumesi var mi? (formul izi)
3. RECALIBRATION exploit'leri (fold-safe degerlendirme icin SADECE teshis; nested gerekirse
   ayrica): isotonic / variance-stabilizing. MSE altinda kosullu-ortalama optimal -> beklenti
   minimal ama OLC.
4. CENSORED-AWARE: blend OOF'a, P(y=100) sinyaliyle DEGIL, dogrudan ust-kuyruk bias-correction
   dene (residual @ high-pred negatif/pozitif mi?).
Bu dosya KARAR icin recency-weighted MSE'yi de raporlar (cv.py'den).
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
    w = cv.recency_weights(tr, te)  # recency importance weights (KARAR metrigi)

    blend = np.load(ART / "oof_blend.npy")
    lgbm = np.load(ART / "oof_lgbm_full.npy")
    cat = np.load(ART / "oof_catboost_full.npy")
    eq100 = np.isclose(y, 100.0)

    def rw(p):
        return cv.compute_recency_weighted_mse(p, y, w)
    def uw(p):
        return float(np.mean((y - np.clip(p, 0, 100)) ** 2))

    print("== GERCEK PROJE OOF — baseline ==")
    for nm, p in [("blend", blend), ("lgbm_full", lgbm), ("catboost_full", cat)]:
        print(f"  {nm:14s} uw_MSE={uw(p):.4f}  rw_MSE={rw(p):.4f}  resid_std={np.std(y-np.clip(p,0,100)):.4f}")
    print(f"  WALL (blend rw) = {rw(blend):.4f}")

    # ---------------------------------------------------------------- #
    print("\n== 2. NEAR-DETERMINISTIC satir var mi? (|resid|<0.5) ==")
    res = y - np.clip(blend, 0, 100)
    near0 = np.abs(res) < 0.5
    print(f"  |resid|<0.5: {int(near0.sum())} satir ({100*near0.mean():.2f}%)  (sans eseri beklenen ~{100*0.5*2/ (np.std(res)*np.sqrt(2*np.pi)):.1f}% gaussian)")
    print(f"  |resid|<0.1: {int((np.abs(res)<0.1).sum())}  |resid|<0.05: {int((np.abs(res)<0.05).sum())}")
    # exact reconstruction izi: 100-grubu disinda |resid|~0 yok ise -> gercek gurultu

    # ---------------------------------------------------------------- #
    print("\n== 3. ISOTONIC RECALIBRATION (fold-safe degil, in-sample tavan; rw ile) ==")
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(blend, y)
    blend_iso = np.clip(iso.predict(blend), 0, 100)
    print(f"  isotonic (in-sample, ust-tavan) rw_MSE={rw(blend_iso):.4f}  (vs {rw(blend):.4f}, delta {rw(blend_iso)-rw(blend):+.4f})")
    print("  (in-sample tavan; pozitif iyilesme yoksa fold-safe denemenin anlami yok)")

    # ---------------------------------------------------------------- #
    print("\n== 4. UST-KUYRUK BIAS (censoring) — blend residual by pred-bin ==")
    q = pd.qcut(blend, 12, duplicates="drop")
    bb = pd.DataFrame({"pred": blend, "res": res}).groupby(q, observed=True).agg(
        pred_mid=("pred", "mean"), res_mean=("res", "mean"), res_std=("res", "std"), n=("res", "size"))
    print(bb.round(3).to_string())
    print("  (en yuksek bin'de res_mean>0 -> model censoring tavanini yeterince itmiyor; <0 -> overshoot)")

    # ---------------------------------------------------------------- #
    print("\n== 5. AFFINE/SHIFT RECALIB (in-sample tavan; mean-shift + scale) ==")
    # y = a*blend + b en kucuk kareler (rw)
    from numpy.polynomial import polynomial as P
    A = np.vstack([blend, np.ones(n)]).T
    Wd = np.diag(w)
    coef = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    aff = np.clip(coef[0] * blend + coef[1], 0, 100)
    print(f"  affine a={coef[0]:.4f} b={coef[1]:.4f}  rw_MSE={rw(aff):.4f} (delta {rw(aff)-rw(blend):+.4f})")
    print("  (a~1,b~0 ve delta~0 -> blend zaten kalibre, affine exploit YOK)")

    out = dict(wall_rw=rw(blend), blend_uw=uw(blend),
               near0_count=int(near0.sum()), iso_rw=rw(blend_iso), aff_rw=rw(aff),
               affine_a=float(coef[0]), affine_b=float(coef[1]))
    (REPORTS / "forensics_dump5.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n[forensics5] dump -> reports/forensics_dump5.json")


if __name__ == "__main__":
    main()
