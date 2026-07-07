"""
TIER-2 LEVER-3 — Isotonic per-year post-kalibrasyon (forensik 3.5).
==================================================================

    python src/isotonic_peryear.py

NEDEN: Test recency-yogun (graduation_year 2024-26 agirlikli). Blend OOF'unda yil-grupli
monoton bir mis-kalibrasyon varsa (or. yeni mezunlarda sistematik over/under-prediction),
graduation_year-grupli isotonic (OOF->hedef) bunu duzeltebilir. Forensik tablosunda GLOBAL
isotonic recalib nested +0.42 (KOTU, overfit'ti) cikmisti; bu lever YIL-GRUPLU varyanti
fold-safe + thin-cell korumali dener (forensik 3.5 onerisi).

KARAR METRIGI = nested recency-weighted OOF-MSE (review H1). Kabul kapisi =
  85.4945 - 0.25*3.0238 = 84.7385. Public LB'ye BAKILMADI.

FOLD-SAFE (kritik — global isotonic'in DUSTUGU tuzaktan kacinmak icin):
  isotonic map HEDEF GORUR -> NESTED uygulanmali. Her (repeat,fold) hucresi icin per-year
  isotonic map o hucre DISINDAKI blend-OOF satirlarindan fit edilir, hucre transform edilir
  -> nested_cal_oof. rw-OOF(nested_cal_oof) = DURUST karar skoru. Tum-OOF'a fit edilmis map
  (FROZEN) yalniz kapidan GECERSE test'e uygulanir (SUB-2).

THIN-CELL KORUMASI: bir yil hucresinde fit-orneklemi < MIN_CELL ise o yil icin GLOBAL isotonic'e
  (tum yillar) dus -> kucuk-ornek isotonic overfit'i engellenir. clip[0,100] + monoton garanti.
Determinizm: SEED=42, IsotonicRegression deterministik (out_of_bounds='clip').
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import artifacts_io as aio
import cv

BLEND = "blend"
MODEL = "blend_isocal"   # kapidan gecerse SUB-2 adayi
REPORT_PATH = cv.REPORTS_DIR / "isotonic_peryear_report.json"

WALL_RW = 85.4945        # mevcut blend nested rw-OOF (karar duvari)
WALL_STD = 3.0238        # blend cv_mse_std
GATE = WALL_RW - 0.25 * WALL_STD   # 84.7385

MIN_CELL = 300           # bir yil hucresinde isotonic fit icin min ornek (altinda GLOBAL'e dus)


def _fit_peryear(cal_pred: np.ndarray, y: np.ndarray, gy: np.ndarray):
    """Per-year isotonic map'leri fit eder (+ GLOBAL fallback). cal_pred/y/gy fit-ornegi.

    Doner: dict(year -> IsotonicRegression) + 'global' anahtari. Transform sirasinda yil hucresi
    yetersizse veya yil haritada yoksa 'global' kullanilir."""
    maps = {}
    glob = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=100.0)
    glob.fit(cal_pred, y)
    maps["__global__"] = glob
    for yr in np.unique(gy):
        m = gy == yr
        if int(m.sum()) >= MIN_CELL:
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=100.0)
            ir.fit(cal_pred[m], y[m])
            maps[int(yr)] = ir
    return maps


def _apply_peryear(maps: dict, pred: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Per-year isotonic map'lerini uygula (yil yoksa/ince-hucreyse __global__)."""
    out = np.empty(len(pred), dtype=float)
    for i in range(len(pred)):
        ir = maps.get(int(gy[i]), maps["__global__"])
        out[i] = ir.predict([pred[i]])[0]
    return cv.clip_predictions(out)


def nested_cal_oof(blend_oof: np.ndarray, y: np.ndarray, gy: np.ndarray,
                   folds: pd.DataFrame, sid: np.ndarray) -> np.ndarray:
    """3-repeat nested per-year isotonic: her hucre DISINDAN fit, hucreyi transform (durust)."""
    n = len(y)
    s = np.zeros(n)
    c = np.zeros(n)
    for r in range(cv.N_REPEATS):
        fold_of = cv.fold_of_rows(folds, sid, r)
        for g in range(cv.N_SPLITS):
            va = np.where(fold_of == g)[0]
            tr = np.where(fold_of != g)[0]
            maps = _fit_peryear(blend_oof[tr], y[tr], gy[tr])
            s[va] += _apply_peryear(maps, blend_oof[va], gy[va])
            c[va] += 1.0
    assert np.all(c == cv.N_REPEATS), "nested kapsam bozuk."
    return cv.clip_predictions(s / c)


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    gy_tr = train[cv.RECENCY_COL].to_numpy()
    gy_te = test[cv.RECENCY_COL].to_numpy()
    w = cv.recency_weights(train, test)

    blend_oof = np.load(cv.ARTIFACTS_DIR / f"oof_{BLEND}.npy")
    blend_test = np.load(cv.ARTIFACTS_DIR / f"test_{BLEND}.npy")
    cv.assert_in_range(blend_oof, "oof_blend")
    cv.assert_in_range(blend_test, "test_blend")

    base_rw = cv.compute_recency_weighted_mse(blend_oof, y, w)
    print(f"[iso] mevcut blend nested rw-OOF = {base_rw:.4f}  (duvar {WALL_RW}); kapi = {GATE:.4f}")

    # --- thin-cell raporu (hangi yillar GLOBAL'e duser) ---
    cnt = pd.Series(gy_tr).value_counts().sort_index()
    thin = [int(k) for k, v in cnt.items() if v < MIN_CELL]
    print(f"[iso] yil hucre sayilari: {cnt.to_dict()}")
    print(f"[iso] MIN_CELL={MIN_CELL} altindaki yillar (GLOBAL fallback): {thin}")

    # --- NESTED per-year isotonic OOF (DURUST karar skoru) ---
    cal_oof = nested_cal_oof(blend_oof, y, gy_tr, folds, sid)
    cal_rw = cv.compute_recency_weighted_mse(cal_oof, y, w)
    delta = cal_rw - base_rw
    passed = cv.acceptance_gate(cal_rw, WALL_RW, WALL_STD)

    cal_cv, cal_std, _ = cv.compute_cv_mse(cal_oof, y, folds, sid)
    print(f"[iso] NESTED per-year isotonic rw-OOF = {cal_rw:.4f}  (delta {delta:+.4f})")
    print(f"[iso] unweighted CV = {cal_cv:.4f} +/- {cal_std:.4f}")
    print(f"[iso] kabul kapisi ({GATE:.4f}) GECTI mi? {passed}")

    # --- FROZEN map (tum-OOF) -> test (yalniz gecerse anlamli; her halukarda uret + raporla) ---
    frozen = _fit_peryear(blend_oof, y, gy_tr)
    cal_test = _apply_peryear(frozen, blend_test, gy_te)
    cv.assert_in_range(cal_test, "test_blend_isocal")

    report = dict(
        base_blend_rw_oof=round(float(base_rw), 6),
        wall=WALL_RW, gate=round(float(GATE), 4),
        nested_cal_rw_oof=round(float(cal_rw), 6),
        delta=round(float(delta), 6),
        unweighted_cv=round(float(cal_cv), 6),
        unweighted_cv_std=round(float(cal_std), 6),
        min_cell=MIN_CELL,
        year_counts={int(k): int(v) for k, v in cnt.items()},
        thin_years_global_fallback=thin,
        passed_gate=bool(passed),
        decision=("SUB-2'ye islenir" if passed else "ELENDI (gurultu bandi, Occam/sifir-overfit)"),
        note=("Per-year isotonic (OOF->hedef) nested fold-safe + thin-cell GLOBAL fallback. "
              "KARAR nested rw-OOF; public LB'ye BAKILMADI."),
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if passed:
        # SUB-2 adayi olarak artefaktla (finalize/ensemble buradan okuyabilir)
        aio.save_oof_test(MODEL, cal_oof, cal_test)
        aio.log_model_score(MODEL, cal_cv, cal_std, cal_rw, weighted_training=False,
                            note=("per-year isotonic post-kalibrasyon (forensik 3.5); nested "
                                  f"rw-OOF {cal_rw:.4f} blend {base_rw:.4f}'ten kapi GECTI -> SUB-2 adayi."))
        print(f"[iso] GECTI -> artifacts/oof_{MODEL}.npy, test_{MODEL}.npy yazildi (SUB-2 adayi).")
    else:
        # ledger'a ELENDI satiri (dokuman; artefakt YAZILMAZ -> Occam, nihai pipeline temiz kalir)
        aio.log_model_score(MODEL, cal_cv, cal_std, cal_rw, weighted_training=False,
                            note=("per-year isotonic post-kalibrasyon (forensik 3.5) DENENDI: nested "
                                  f"rw-OOF {cal_rw:.4f} (blend {base_rw:.4f}, delta {delta:+.4f}); "
                                  f"kabul kapisi {GATE:.4f} GECMEDI -> ELENDI (Occam/sifir-overfit). "
                                  "Artefakt YAZILMADI."))
        print(f"[iso] ELENDI (delta {delta:+.4f}, kapi {GATE:.4f}). Artefakt yazilmadi; ledger'a islendi.")

    print(f"[iso] yazildi: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
