"""
Faz 6 — lgbm_full_w: lgbm_full ile BIREBIR ayni FULL matris + AYNI HP, tek fark RECENCY
SAMPLE-WEIGHT'li egitim. Modeli dogrudan test/private dagilimina (recency-yogun) optimize eder.
==============================================================================================

    python src/lgbm_full_w.py

NEDEN (review H1 / KARAR METRIGI = recency-weighted OOF):
  Test graduation_year dagilimi 2024-26'ya yigili (test ~%58 vs train ~%32). Unweighted egitim
  tum train satirlarina esit agirlik verir -> eski yillara fazla uyum -> recency-weighted OOF
  iyimser olmayan ama YUKSEK (lgbm_full 87.27). sample_weight = w_i = P_test(gy_i)/P_train(gy_i)
  (cv.recency_weights, mean-normalize, clip) ile egitim importance-weighted -> kayip dogrudan
  test dagilimini hedefler. eval_set early-stopping de eval_sample_weight=w_val ile AYNI agirlikta
  -> secilen iterasyon recency-weighted val L2'yi minimize eder (uctan uca tutarli).

FOLD-SAFE: w SADECE graduation_year (kovaryat) marjinallerinden gelir; HEDEFE (y) DOKUNMAZ ->
  target sizintisi yok (CLAUDE.md sizinti kurali hedef-bagimli istatistikler icindir). w tum-train
  + test marjinalinden hesaplanir (degerlendirme metrigiyle BIREBIR tutarli). Fold-ici: her fit
  yalniz dis-fold train satirlarinin agirligini kullanir (X_tr.index ile pozisyon-eslemeli).

txt_ridge_pred + lexicon FULL matris (Faz05) AYNEN; HP degismez (overfit riski yok, tek lever:
  sample_weight). Test = KANONIK fold-bagging. Determinizm: SEED=42, deterministic=True, n_jobs=1.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor

import artifacts_io as aio
import cv
import text_utils as tu
from anchor_lgbm_num import EARLY_STOPPING_ROUNDS, LGBM_PARAMS
from lgbm_full import OOF_TXT_PATH, TEST_TXT_PATH, _add_text

MODEL = "lgbm_full_w"

# Agirlik clip bandi: recency_weights dogal olarak [~0.27, ~2.02] (yil-orani sinirli). Clip
# genis tutulur (burada baglamaz) ama belge/guvenlik amacli: asiri agirlik -> egitim kararsizligi.
W_CLIP_LO, W_CLIP_HI = 0.10, 5.0

REF_LO, REF_HI = 84.0, 88.0  # rw-OOF beklenen bant (lgbm_full 87.27 ALTI hedef)


def make_fit_fold_weighted(cat_features, w_full: np.ndarray):
    """fit_fold(X_tr,y_tr,X_val,y_val) -> (predict_fn, best_it). X_tr.index (run_oof reset_index
    sonrasi orijinal pozisyon) ile w_full'dan dis-fold train agirliklarini ceker. SADECE fold-ici."""

    def fit_fold(X_tr, y_tr, X_val, y_val):
        # run_oof X'i reset_index(drop=True) yapar -> X_tr.index = orijinal satir pozisyonlari.
        pos_tr = np.asarray(X_tr.index, dtype=int)
        pos_val = np.asarray(X_val.index, dtype=int)
        w_tr = w_full[pos_tr]
        w_val = w_full[pos_val]

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],   # early stopping = recency-weighted val L2 (uctan uca tutarli)
            eval_metric="l2",
            categorical_feature=cat_features,
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_it = model.best_iteration_ or LGBM_PARAMS["n_estimators"]

        def predict(X):
            return model.predict(X, num_iteration=best_it)  # HAM; clip run_oof'ta

        return predict, int(best_it)

    return fit_fold


def main() -> None:
    cv.set_seed()

    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    # --- recency sample-weight (kovaryat-only, hedef-bagimsiz -> fold-safe) ---
    w_full = cv.recency_weights(train, test)
    w_full = np.clip(w_full, W_CLIP_LO, W_CLIP_HI)
    print(f"[lgbm_full_w] recency sample_weight: mean={w_full.mean():.4f} "
          f"min={w_full.min():.4f} max={w_full.max():.4f} (clip [{W_CLIP_LO},{W_CLIP_HI}])")

    # --- FULL matris = num + txt_ridge(nested-OOF) + lexicon (lgbm_full ile BIREBIR) ---
    oof_txt = np.load(OOF_TXT_PATH)
    test_txt = np.load(TEST_TXT_PATH)
    cv.assert_in_range(oof_txt, "oof_txt_ridge")
    cv.assert_in_range(test_txt, "test_txt_ridge")
    lex_tr = tu.extract_handcrafted_features(train[cv.TEXT_COL].values)
    lex_te = tu.extract_handcrafted_features(test[cv.TEXT_COL].values)

    cat_dtypes = cv.structured_cat_dtypes(train)
    X_struct, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    Xt_struct, cat_features_te = cv.build_structured_matrix(test, cat_dtypes)
    assert list(X_struct.columns) == list(Xt_struct.columns) and cat_features == cat_features_te

    X = _add_text(X_struct, oof_txt, lex_tr)
    X_test = _add_text(Xt_struct, test_txt, lex_te)
    assert list(X.columns) == list(X_test.columns), "FULL train/test kolonlari hizali degil."
    print(f"[lgbm_full_w] {MODEL}: {X.shape[1]} feature (lgbm_full ile ayni matris); "
          f"{len(cat_features)} native-kategorik. Tek lever: recency sample_weight.")

    # --- 15-fold sizintisiz OOF + kanonik fold-bagging test (agirlikli egitim) ---
    out = cv.run_oof(make_fit_fold_weighted(cat_features, w_full), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]
    genuine = out["genuine_fold_mse"]

    # Unweighted CV (siralama/denge; KARAR DEGIL) + KARAR METRIGI recency-weighted OOF.
    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.recency_weights(train, test)
    recency_mse = cv.compute_recency_weighted_mse(oof, y, rw)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))
    single5fold_std = float(np.std(genuine[: cv.N_SPLITS]))

    note = (
        "lgbm_full_w = lgbm_full FULL matris + AYNI HP, tek fark recency sample_weight (egitim ve "
        "eval_set early-stopping). KARAR METRIGI recency_weighted_oof_mse "
        f"={recency_mse:.4f} (lgbm_full 87.2663 ile karsilastir). Unweighted CV "
        f"={cv_mean:.4f} (agirlikli egitim unweighted'i bozabilir; KARAR DEGIL). Fold-safe: w "
        "kovaryat-only (graduation_year), hedefe dokunmaz."
    )

    # --- Artefaktlar + karar defteri ---
    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, best_iter_mean)
    aio.write_cv_log(
        MODEL, cv_mean, cv_std, fold_mse, best_iters, best_iter_mean,
        genuine_fold_mse=genuine, single5fold_std=single5fold_std, note=note,
    )
    aio.log_model_score(MODEL, cv_mean, cv_std, recency_mse, weighted_training=True, note=note)

    # --- DoD-4 ic tutarlilik ---
    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, re_std, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6 and abs(re_std - cv_std) < 1e-6, "DoD-4 KIRIK."

    cv.assert_in_range(oof, "oof_lgbm_full_w")
    cv.assert_in_range(test_pred, "test_lgbm_full_w")

    base_rw = 87.266319  # lgbm_full (karar defteri referansi)
    print(f"[lgbm_full_w] unweighted cv_mse_mean = {cv_mean:.4f} +/- {cv_std:.4f}  (KARAR DEGIL)")
    print(f"[lgbm_full_w] recency_weighted_oof_mse = {recency_mse:.4f}   (KARAR METRIGI; "
          f"lgbm_full {base_rw:.4f}, delta {recency_mse - base_rw:+.4f})")
    print(f"[lgbm_full_w] best_iteration_mean = {best_iter_mean:.1f}")
    print(f"[lgbm_full_w] test fold-bagging: mean={test_pred.mean():.3f} std={test_pred.std():.3f} "
          f"min={test_pred.min():.3f} max={test_pred.max():.3f}")
    verdict = "DUSURDU (degerli)" if recency_mse < base_rw else "dusurmedi"
    print(f"[lgbm_full_w] >>> recency-weighted OOF {verdict} (karar: rw-OOF).")


if __name__ == "__main__":
    main()
