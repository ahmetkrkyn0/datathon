"""
Faz 6 — histgbr_full: HistGradientBoosting (sklearn) FULL matris, deterministik.
================================================================================

    python src/histgbr_full.py

NEDEN: LGBM + CatBoost'a YAPISAL FARKLI UCUNCU GBDT (sklearn histogram-tabanli; farkli split/
  regularizasyon implementasyonu) -> blend cesitliligi (risk dagitimi). FULL matris lgbm_full /
  catboost_full ile AYNI parcalar: num+YIL + native-kategorik + missing-flag + txt_ridge_pred
  (nested-OOF) + lexicon(10). Kategorik NATIVE islenir (categorical_features=from_dtype: FULL
  matristeki CategoricalDtype kolonlari otomatik); NaN sayisal HistGBR tarafindan native islenir
  (impute YOK -> sizinti YOK).

KARAR METRIGI = recency_weighted_oof_mse (review H1). Tek-basina rw-OOF raporlanir; blend faydasi
  ensemble.py'de NESTED rw-OOF + 0.25*std kapisina gore karar verilir.

FOLD-SAFE: tum fit'ler cv.run_oof ile dis-fold train'inde; txt_ridge nested-OOF artefakti
  (fold-safe), lexicon hedef/fold-bagimsiz. Test = KANONIK fold-bagging (15 model).
Determinizm: random_state=42 SABIT + tek-thread (set_seed OMP/MKL=1; HistGBR OpenMP-paralel ama
  random_state ile reproducible — yine de tek-thread BLAS/OpenMP set_seed'de belgeli). DoD-4
  oof_histgbr_full.npy yeniden-hesap +/-1e-6.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import artifacts_io as aio
import cv
import text_utils as tu
from lgbm_full import OOF_TXT_PATH, TEST_TXT_PATH, _add_text

MODEL = "histgbr_full"

# Muhafazakar HistGBR (anchor felsefesi: overfit kapisi; HP taramasi YOK). lgbm/catboost ile
# ayni ruh: dusuk lr + early stopping + sade derinlik. max_iter ust sinir; early_stopping karar.
HGB_PARAMS = dict(
    loss="squared_error",
    learning_rate=0.03,
    max_iter=3000,             # ust sinir; early stopping karar verir
    max_leaf_nodes=31,         # lgbm num_leaves=31 ile esdeger karmasiklik
    min_samples_leaf=50,       # lgbm min_child_samples=50 ile esdeger
    l2_regularization=1.0,     # lgbm reg_lambda=1.0 ile esdeger
    max_features=0.8,          # lgbm colsample_bytree=0.8 ile esdeger (per-split feature subsample)
    early_stopping=True,
    validation_fraction=None,  # eval_set olarak X_val verecegiz (asagida custom yok -> ic split)
    n_iter_no_change=100,      # lgbm EARLY_STOPPING_ROUNDS=100 ile esdeger
    random_state=cv.SEED,
)


def make_fit_fold_hgb(cat_cols):
    """fit_fold -> (predict_fn, best_it). HistGBR ic-validation ile early stopping (validation_fraction).

    NOT: HistGBR eval_set kabul etmez; early stopping ic train'den ayirdigi validation_fraction
    parcasiyla yapilir (dis-valid'e ASLA dokunmaz -> fold-safe). categorical_features='from_dtype'
    -> FULL matristeki CategoricalDtype kolonlari native kategorik (one-hot/encode YOK)."""
    def fit_fold(X_tr, y_tr, X_val, y_val):
        params = dict(HGB_PARAMS)
        params["validation_fraction"] = 0.1  # ic-train'den early-stopping validation'i (fold-safe)
        m = HistGradientBoostingRegressor(categorical_features="from_dtype", **params)
        m.fit(X_tr, y_tr)
        best_it = int(getattr(m, "n_iter_", 0) or HGB_PARAMS["max_iter"])

        def predict(X):
            return m.predict(X)  # HAM; clip run_oof'ta

        return predict, best_it

    return fit_fold


def main() -> None:
    cv.set_seed()

    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    # --- txt_ridge fold-safe kolon (Faz 05 nested-OOF artefaktlari) ---
    oof_txt = np.load(OOF_TXT_PATH)
    test_txt = np.load(TEST_TXT_PATH)
    cv.assert_in_range(oof_txt, "oof_txt_ridge")
    cv.assert_in_range(test_txt, "test_txt_ridge")

    # --- lexicon (Katman B) — hedef/fold-bagimsiz ---
    lex_tr = tu.extract_handcrafted_features(train[cv.TEXT_COL].values)
    lex_te = tu.extract_handcrafted_features(test[cv.TEXT_COL].values)

    # --- yapisal anchor matris (CategoricalDtype kolonlar -> HistGBR native kategorik) ---
    cat_dtypes = cv.structured_cat_dtypes(train)
    X_struct, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    Xt_struct, cat_features_te = cv.build_structured_matrix(test, cat_dtypes)
    assert list(X_struct.columns) == list(Xt_struct.columns) and cat_features == cat_features_te

    # --- FULL matris = num + txt_ridge + lexicon (lgbm_full/catboost_full ile AYNI) ---
    X = _add_text(X_struct, oof_txt, lex_tr)
    X_test = _add_text(Xt_struct, test_txt, lex_te)
    assert list(X.columns) == list(X_test.columns), "FULL train/test kolonlari hizali degil."
    print(f"[histgbr] {MODEL}: {X.shape[1]} feature, {len(cat_features)} native-kategorik "
          f"(from_dtype). Determinizm random_state={cv.SEED}.")

    out = cv.run_oof(make_fit_fold_hgb(cat_features), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]
    genuine = out["genuine_fold_mse"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.recency_weights(train, test)
    recency_mse = cv.compute_recency_weighted_mse(oof, y, rw)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))
    single5fold_std = float(np.std(genuine[: cv.N_SPLITS]))

    note = (
        f"{MODEL} = HistGradientBoosting (sklearn) FULL matris (num+YIL+kategorik+flag+txt_ridge+"
        f"lexicon), native kategorik (from_dtype), random_state={cv.SEED}. UCUNCU GBDT ailesi "
        f"(blend cesitliligi). KARAR recency_weighted_oof_mse={recency_mse:.4f} "
        f"(catboost_full 86.4149, lgbm_full 87.2663). Unweighted CV={cv_mean:.4f}. Fold-safe."
    )

    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, best_iter_mean)
    aio.write_cv_log(
        MODEL, cv_mean, cv_std, fold_mse, best_iters, best_iter_mean,
        genuine_fold_mse=genuine, single5fold_std=single5fold_std, note=note,
    )
    aio.log_model_score(MODEL, cv_mean, cv_std, recency_mse, weighted_training=False, note=note)

    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, re_std, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6 and abs(re_std - cv_std) < 1e-6, "DoD-4 KIRIK."
    cv.assert_in_range(oof, f"oof_{MODEL}")
    cv.assert_in_range(test_pred, f"test_{MODEL}")

    print(f"[histgbr] unweighted cv_mse_mean = {cv_mean:.4f} +/- {cv_std:.4f}  (KARAR DEGIL)")
    print(f"[histgbr] recency_weighted_oof_mse = {recency_mse:.4f}   (KARAR METRIGI; "
          f"catboost_full 86.4149, lgbm_full 87.2663)")
    print(f"[histgbr] best_iteration_mean = {best_iter_mean:.1f}")
    print(f"[histgbr] test fold-bagging: mean={test_pred.mean():.3f} std={test_pred.std():.3f} "
          f"min={test_pred.min():.3f} max={test_pred.max():.3f}")
    print(f"[histgbr] DoD-4 ic tutarlilik GECTI (oof.npy -> {re_mean:.6f}).")
    print("[histgbr] blend faydasi: python src/ensemble.py (havuza ekle).")


if __name__ == "__main__":
    main()
