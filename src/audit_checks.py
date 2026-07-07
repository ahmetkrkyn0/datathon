"""
Faz 7 — OOF/submission butunluk denetimi (SPEC 07 §1 + §4.2; READ-ONLY).
========================================================================

    python src/audit_checks.py

Hicbir artefakt YAZMAZ; mevcutlari denetler ve ihlalde assert ile durur:
  1. data/folds.parquet gecerli (kapsam + stratify dengesi, cv.validate_folds).
  2. Her oof_{M}/test_{M}.npy: 10000 uzunluk, NaN/Inf yok, clip[0,100] icinde.
  3. cv_scores.csv'deki cv_mse_mean/std, oof'tan compute_cv_mse ile +/-1e-6 yeniden uretiliyor.
  4. model_scores.csv'deki recency_weighted_oof_mse, oof'tan +/-1e-6 yeniden uretiliyor.
  5. Final submission CSV'leri (sub1/sub2): 10000 satir, kolon adlari, test_x ID kumesi+SIRASI
     birebir, [0,100], NaN yok; degerler kaynak test_{M}.npy ile birebir; sample_submission
     format stub'i uyumlu.
  6. test_M dagilimi oof_M ile makul ortusuyor (|mean farki| > 5 -> uyari; bilgi amacli).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cv

TOL = 1e-6

# (submission dosyasi, kaynak model) — finalize_submissions.py ciktilariyla birebir.
FINAL_SUBS = (
    ("sub1_catboost_full.csv", "catboost_full"),
    ("sub2_blend.csv", "blend"),
)


def check_folds(train: pd.DataFrame) -> pd.DataFrame:
    folds = cv.load_folds()
    cv.validate_folds(folds, train[cv.TARGET_COL].values, train[cv.ID_COL].values)
    print(f"[audit] folds.parquet OK: {len(folds)} satir "
          f"({cv.N_REPEATS} repeat x {len(train)} satir), stratify dengesi +/-1% icinde.")
    return folds


def check_artifacts(train: pd.DataFrame, test: pd.DataFrame, folds: pd.DataFrame) -> None:
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    rw = cv.recency_weights(train, test)

    cv_scores = pd.read_csv(cv.ARTIFACTS_DIR / "cv_scores.csv").set_index("model")
    model_scores = pd.read_csv(cv.REPORTS_DIR / "model_scores.csv").set_index("model")

    models = sorted({p.stem.removeprefix("oof_") for p in cv.ARTIFACTS_DIR.glob("oof_*.npy")})
    for m in models:
        oof = np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy")
        tst = np.load(cv.ARTIFACTS_DIR / f"test_{m}.npy")
        assert len(oof) == len(train) and len(tst) == len(test), f"{m}: uzunluk != 10000."
        cv.assert_in_range(oof, f"oof_{m}")
        cv.assert_in_range(tst, f"test_{m}")

        msgs = []
        if m in cv_scores.index:
            mean, std, _ = cv.compute_cv_mse(oof, y, folds, sid)
            row = cv_scores.loc[m]
            assert abs(mean - row["cv_mse_mean"]) < TOL and abs(std - row["cv_mse_std"]) < TOL, (
                f"{m}: cv_scores.csv ({row['cv_mse_mean']:.6f}) oof'tan yeniden uretilemedi "
                f"({mean:.6f}) — artefakt/defter uyumsuz (SPEC07 §1.2 DUR)."
            )
            msgs.append(f"cv={mean:.4f}+/-{std:.4f} (cv_scores OK)")
        if m in model_scores.index:
            rmse = cv.compute_recency_weighted_mse(oof, y, rw)
            ref = float(model_scores.loc[m, "recency_weighted_oof_mse"])
            assert abs(rmse - ref) < TOL, (
                f"{m}: model_scores rw-OOF {ref:.6f} != yeniden hesap {rmse:.6f}."
            )
            msgs.append(f"rw-OOF={rmse:.4f} (model_scores OK)")

        dmean = abs(float(tst.mean()) - float(oof.mean()))
        flag = "  [UYARI: test-oof mean farki > 5]" if dmean > 5.0 else ""
        print(f"[audit] {m:16s} OK: " + "; ".join(msgs)
              + f"; test mean={tst.mean():.2f} (oof {oof.mean():.2f}){flag}")


def check_final_submissions(test: pd.DataFrame) -> None:
    ref_ids = test[cv.ID_COL].to_numpy()
    sample = pd.read_csv(cv.DATA_DIR / "sample_submission.csv", encoding="utf-8-sig").dropna(how="all")
    assert list(sample.columns) == [cv.ID_COL, cv.TARGET_COL], "sample_submission kolon adlari uyumsuz."

    for fname, model in FINAL_SUBS:
        path = cv.ROOT / "submissions" / fname
        assert path.exists(), f"{fname} yok — once python src/finalize_submissions.py."
        sub = pd.read_csv(path, encoding="utf-8")

        assert len(sub) == 10000, f"{fname}: {len(sub)} satir != 10000."
        assert list(sub.columns) == [cv.ID_COL, cv.TARGET_COL], f"{fname}: kolonlar {list(sub.columns)}."
        assert sub[cv.ID_COL].is_unique, f"{fname}: student_id tekrarli."
        assert set(sub[cv.ID_COL]) == set(ref_ids), f"{fname}: ID kumesi test_x ile birebir degil."
        assert (sub[cv.ID_COL].to_numpy() == ref_ids).all(), f"{fname}: sira test_x kanonik sirasi degil."
        assert sub[cv.TARGET_COL].notna().all(), f"{fname}: NaN var."
        cv.assert_in_range(sub[cv.TARGET_COL].to_numpy(), fname)
        assert set(sample[cv.ID_COL]).issubset(set(sub[cv.ID_COL])), f"{fname}: sample stub ID'leri eksik."

        src = cv.clip_predictions(np.load(cv.ARTIFACTS_DIR / f"test_{model}.npy"))
        max_diff = float(np.max(np.abs(sub[cv.TARGET_COL].to_numpy() - src)))
        assert max_diff < 1e-9, f"{fname}: degerler test_{model}.npy ile uyusmuyor (max|diff|={max_diff})."

        p = sub[cv.TARGET_COL].to_numpy()
        print(f"[audit] {fname} OK: 10000 satir, ID kume+sira=test_x, [{p.min():.2f},{p.max():.2f}], "
              f"NaN yok, kaynak=test_{model}.npy (max|diff|={max_diff:.1e}) -> upload-ready.")


def main() -> None:
    train = cv.load_train()
    test = cv.load_test()
    folds = check_folds(train)
    check_artifacts(train, test, folds)
    check_final_submissions(test)
    print("[audit] TUM BUTUNLUK/FORMAT DENETIMLERI GECTI (SPEC07 §1 + §4.2).")


if __name__ == "__main__":
    main()
