"""Tek-model rw-OOF (agirlik=1.0, blend YOK) raporu. Sadece okur, hicbir sey yazmaz."""
from __future__ import annotations

import numpy as np

import cv

# Blend'e giren 10 model (ensemble.py CANDIDATE_POOL ile birebir).
MODELS = [
    "mm", "lgbm_num", "lgbm_full_ht", "lgbm_full_h", "catboost_full",
    "e5_ridge", "catboost_full_w", "lgbm_full", "lgbm_full_w", "txt_ridge",
]


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    rows = []
    for m in MODELS:
        oof = np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy")
        rw = cv.compute_recency_weighted_mse(oof, y, w)
        cm, cs, _ = cv.compute_cv_mse(oof, y, folds, sid)
        rows.append((m, rw, cm, cs))

    rows.sort(key=lambda r: r[1])  # rw-OOF artan
    print(f"{'model':16s} {'rw-OOF':>10s} {'unw_cv':>10s} {'cv_std':>8s}")
    print("-" * 48)
    for m, rw, cm, cs in rows:
        print(f"{m:16s} {rw:10.4f} {cm:10.4f} {cs:8.4f}")

    # referans: 10-model blend rw-OOF
    blend = np.load(cv.ARTIFACTS_DIR / "oof_blend.npy")
    rw_b = cv.compute_recency_weighted_mse(blend, y, w)
    print("-" * 48)
    print(f"{'blend (10-model)':16s} {rw_b:10.4f}  <-- referans (nested meta-OOF)")


if __name__ == "__main__":
    main()
