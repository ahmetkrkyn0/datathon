"""
TIER-3 ROBUST-LOSS+REG — lgbm_full_ht: lgbm_full_h'nin siki-regularize varyanti.
================================================================================

    python src/lgbm_full_ht.py

NE: lgbm_full_h (Huber alpha=5) ile BIREBIR ayni matris/protokol; TEK fark daha siki agac
  regularizasyonu: num_leaves=15 (31 yerine), min_child_samples=80 (50 yerine).

NEDEN: ONCEDEN-KAYITLI 12-konfig HP taramasi (gate-kor, repeat-0 fold-safe; tek-yon degisimler)
  icinde tek anlamli yon SIKILASTIRMA cikti: leaves15_mc80 rw 87.16->86.63 (-0.53). Muhafazakar
  anchor bile bu gurultu seviyesinde fazla kompleks. Full-15'te cürüme YOK (-0.54) -> etki gercek.
  Post-hoc konfig-kombinasyonu YAPILMADI (balikcilik); yalniz gridin kazanani uretildi.

SONUC (kabul kaniti): standalone rw 85.7810 (lgbm_full_h 86.3222'den -0.54; EN DUSUK tek-model).
  Blend EKLE 84.0991->84.0212 (-0.078); paired 13/15, t=-4.211, p=8.7e-4, bootstrap %95 CI
  [-0.1552,-0.0019] (ust sinir ince ama sifir-alti; 3 olcut de gecti). IKAME (-0.051) daha zayifti.

KARAR: KABUL -> kalici blend uyesi. SUB-1 adayligi YOK (_h(t) dislamasi; catboost_full kalir,
  yapisal cesitlilik). Public'e BAKILMAZ. Determinizm: lgbm_full_h ile ayni.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb

import artifacts_io as aio
import cv
import text_utils as tu
from anchor_lgbm_num import LGBM_PARAMS, EARLY_STOPPING_ROUNDS
from lgbm_full_h import HUBER_ALPHA

MODEL = "lgbm_full_ht"
NUM_LEAVES = 15
MIN_CHILD = 80


def make_fit_fold_ht(cat_features):
    params = dict(LGBM_PARAMS)
    params["objective"] = "huber"
    params["alpha"] = HUBER_ALPHA
    params["num_leaves"] = NUM_LEAVES
    params["min_child_samples"] = MIN_CHILD

    def fit_fold(X_tr, y_tr, X_val, y_val):
        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        best_it = model.best_iteration_ or params["n_estimators"]
        return (lambda X: model.predict(X, num_iteration=best_it)), int(best_it)

    return fit_fold


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    oof_txt = np.load(cv.ARTIFACTS_DIR / "oof_txt_ridge.npy")
    test_txt = np.load(cv.ARTIFACTS_DIR / "test_txt_ridge.npy")
    lex_tr = tu.extract_handcrafted_features(train[cv.TEXT_COL].values)
    lex_te = tu.extract_handcrafted_features(test[cv.TEXT_COL].values)
    cat_dtypes = cv.structured_cat_dtypes(train)
    X_struct, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    Xt_struct, _ = cv.build_structured_matrix(test, cat_dtypes)

    def _add_text(S, t, L):
        out = S.reset_index(drop=True).copy()
        out["txt_ridge_pred"] = np.asarray(t, dtype=float)
        for c in L.columns:
            out[c] = L[c].reset_index(drop=True).to_numpy()
        return out

    X = _add_text(X_struct, oof_txt, lex_tr)
    X_test = _add_text(Xt_struct, test_txt, lex_te)
    assert list(X.columns) == list(X_test.columns)
    print(f"[{MODEL}] {X.shape[1]} feature; huber a={HUBER_ALPHA:g}, leaves={NUM_LEAVES}, mc={MIN_CHILD}")

    out = cv.run_oof(make_fit_fold_ht(cat_features), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    w = cv.recency_weights(train, test)
    rw = cv.compute_recency_weighted_mse(oof, y, w)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))

    oof_h = np.load(cv.ARTIFACTS_DIR / "oof_lgbm_full_h.npy")
    rw_h = cv.compute_recency_weighted_mse(oof_h, y, w)

    note = (
        f"{MODEL} = lgbm_full_h + siki regularizasyon (leaves={NUM_LEAVES}, mc={MIN_CHILD}; "
        f"onceden-kayitli 12-konfig gridin kazanani, gate-kor repeat-0 secim). standalone "
        f"rw-OOF={rw:.4f} (lgbm_full_h {rw_h:.4f}). Blend kabul kaniti script docstring'inde."
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
    print(f"[{MODEL}] rw-OOF={rw:.4f}  (lgbm_full_h {rw_h:.4f}; fark {rw - rw_h:+.4f})")
    print(f"[{MODEL}] YAZILDI: artifacts/oof_{MODEL}.npy, test_{MODEL}.npy + ledger satirlari")


if __name__ == "__main__":
    main()
