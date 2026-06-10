"""
TIER-3 ROBUST-LOSS — lgbm_num_h: anchor lgbm_num'un Huber(alpha=5) varyanti.
============================================================================

    python src/lgbm_num_h.py

NE: anchor_lgbm_num ile BIREBIR ayni yapisal matris (sayisal+YIL+native-kategorik+flag; METIN YOK)
  ve 15-fold protokol; TEK fark objective='huber' alpha=5. Cikti: oof/test_lgbm_num_h.npy + ledger.

NEDEN: lgbm_full_h ile ayni mekanizma (alt-kuyruk surprizlerinin L2 fit'ini zehirlemesini Huber
  engeller; reports/ROBUST_LOSS_LEVER.md). lgbm_num blend'in EN YUKSEK agirlikli uyesi (0.242,
  metinsiz cesitlilik ankrajı) -> robust ikizi olculmeye deger. Repeat-0 on-olcum: L2 94.08 ->
  huber(a=5) 93.17 (-0.91; lgbm_full'daki -1.31 ile tutarli). Alpha=5, lgbm_full tabanindaki
  inceltilmis taramada da optimum cikti (3:87.26 / 4:87.42 / 5:87.16 / 6:87.40 / 7:87.55 / 10:87.71).

KARAR: ensemble paired-test (e5/mm/lgbm_full_h olcutu) + kullanici onayi. SUB-1 adayligi YOK
  (finalize _h dislamasi). Public'e BAKILMAZ.
Determinizm: anchor ile ayni (SEED=42, deterministic=True, n_jobs=1).
"""

from __future__ import annotations

import numpy as np

import artifacts_io as aio
import cv
from lgbm_full_h import make_fit_fold_huber

MODEL = "lgbm_num_h"


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    cat_dtypes = cv.structured_cat_dtypes(train)
    X, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    X_test, cat_features_te = cv.build_structured_matrix(test, cat_dtypes)
    assert list(X.columns) == list(X_test.columns) and cat_features == cat_features_te
    print(f"[{MODEL}] {X.shape[1]} feature (anchor yapisal, metin yok); objective=huber alpha=5")

    out = cv.run_oof(make_fit_fold_huber(cat_features), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    w = cv.recency_weights(train, test)
    rw = cv.compute_recency_weighted_mse(oof, y, w)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))

    oof_l2 = np.load(cv.ARTIFACTS_DIR / "oof_lgbm_num.npy")
    rw_l2 = cv.compute_recency_weighted_mse(oof_l2, y, w)

    note = (
        f"{MODEL} = anchor lgbm_num'un Huber(alpha=5) varyanti (TIER-3 robust-loss; lgbm_full_h "
        f"mekanizmasi, METIN YOK). standalone rw-OOF={rw:.4f} (L2 ikizi lgbm_num {rw_l2:.4f}). "
        f"Blend karari ensemble paired-test + onay; SUB-1 adayligi yok (_h dislamasi)."
    )

    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, best_iter_mean)
    aio.write_cv_log(MODEL, cv_mean, cv_std, fold_mse, best_iters, best_iter_mean, note=note)
    aio.log_model_score(MODEL, cv_mean, cv_std, rw, weighted_training=False, note=note)

    cv.assert_in_range(oof, f"oof_{MODEL}")
    cv.assert_in_range(test_pred, f"test_{MODEL}")
    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, _, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6, "DoD-4 KIRIK."

    print(f"[{MODEL}] cv_mse_mean={cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"[{MODEL}] rw-OOF={rw:.4f}  (L2 ikizi lgbm_num {rw_l2:.4f}; fark {rw - rw_l2:+.4f})")
    print(f"[{MODEL}] YAZILDI: artifacts/oof_{MODEL}.npy, test_{MODEL}.npy + ledger satirlari")


if __name__ == "__main__":
    main()
