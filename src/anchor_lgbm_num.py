"""
Faz 2 — ANCHOR: yapisal LightGBM (M=lgbm_num). CV altyapisinin DOGRU kuruldugunun
olculebilir kanitidir (SPEC §4 Adim 7, §8 DoD-2/3/4).

    python src/anchor_lgbm_num.py

FEATURE UZAYI (SPEC §4 Adim 7 + review C1 duzeltmesi):
  lgbm_num = sayisal (37) + YIL ham sayisal (2) + native-kategorik (5) + missing-flag (7)
  = 51 feature. METIN YOK. ("num" = yapisal/metinsiz; lgbm_full = lgbm_num + NLP, Faz 05.)
  NaN'lar LightGBM tarafindan native islenir; native kategorik (one-hot degil) hedef-bagimsiz
  -> hicbir istatistik tum-train'de hesaplanmaz -> sizinti yapisal olarak imkansiz (fold-safe).
  objective=regression_l2 + fold-ici early stopping. Test = KANONIK fold-bagging (15 model).

ANCHOR REFERANS NOTU (review C1/M1):
  * "~91.6" stale/yeniden-uretilemez (review M1). "87.91" ise ESKI YILSIZ matrisin degeriydi.
  * Review C1 duzeltmesi: yillar HAM SAYISAL dahil -> olculen yeni taban ~81.7 (dogrulama:
    yilsiz 87.913 -> +yillar 81.689, recency-proxy 101.09 -> 92.75). Bu kosuda OLCULEN deger
    downstream kabul kapisi (0.25*std) tabani olarak kilitlenir.
  * Co-headline: compute_recency_weighted_mse (private-durust tahmin) ayrica raporlanir.

Determinizm: deterministic=True, force_row_wise=True, n_jobs=1, seed=42.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor

import artifacts_io as aio
import cv

MODEL = "lgbm_num"

# Muhafazakar LGBM-L2 (MASTERPLAN model stack); yapisal anchor.
LGBM_PARAMS = dict(
    objective="regression_l2",
    n_estimators=3000,         # ust sinir; early stopping karar verir
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=cv.SEED,
    n_jobs=1,                  # determinizm
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 100

# Yapisal anchor (yillar DAHIL) icin makul referans bandi (olculen ~81.7; 87.91 eski/yilsiz).
REF_LO, REF_HI = 79.0, 84.0


def make_fit_fold(cat_features):
    """fit_fold(X_tr,y_tr,X_val,y_val) -> (predict_fn, best_iteration). SADECE fold-ici veri."""

    def fit_fold(X_tr, y_tr, X_val, y_val):
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
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

    # Sabit kategori evreni (train==test seviyeleri; test-only YOK) -> native-kategorik hizali.
    cat_dtypes = cv.structured_cat_dtypes(train)
    X, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    X_test, cat_features_te = cv.build_structured_matrix(test, cat_dtypes)
    assert list(X.columns) == list(X_test.columns) and cat_features == cat_features_te
    print(f"[anchor] {MODEL}: {X.shape[1]} feature "
          f"({len(cv.numeric_feature_columns(train))} sayisal + {len(cv.YEAR_COLS)} yil + "
          f"{len(cv.NA_COLS)} flag + {len(cat_features)} native-kategorik). Metin YOK.")

    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    out = cv.run_oof(make_fit_fold(cat_features), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]
    genuine = out["genuine_fold_mse"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    # Co-headline (review H1): recency-agirlikli OOF-MSE = private-durust tahmin.
    rw = cv.recency_weights(train, test)
    recency_mse = cv.compute_recency_weighted_mse(oof, y, rw)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))
    # SPEC §2 referansi: repeat0'in 5 fold-MSE std'si ("tek 5-fold fold-std ~4.68").
    single5fold_std = float(np.std(genuine[: cv.N_SPLITS]))

    note = (
        "lgbm_num = sayisal+YIL(ham)+native-kategorik+missing-flag (metin yok; review C1: "
        "yillar dahil, eski yilsiz taban 87.91). cv_mse_mean/std = compute_cv_mse(oof) "
        f"(avg-oof, DoD-4). recency_weighted_oof_mse={recency_mse:.4f} (private-durust "
        "co-headline, review H1). OLCULEN deger downstream 0.25*std tabani. "
        "std ~4.7 referansi single5fold_std (SPEC §2); compute_cv_mse std 3-repeat ile dusuktur."
    )

    # --- Artefaktlar (SPEC §7) ---
    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, best_iter_mean)
    aio.write_cv_log(
        MODEL, cv_mean, cv_std, fold_mse, best_iters, best_iter_mean,
        genuine_fold_mse=genuine, single5fold_std=single5fold_std, note=note,
    )

    # --- DoD-4 ic tutarlilik: kaydedilen oof'tan yeniden hesapla, +/-1e-6 esle ---
    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, re_std, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6 and abs(re_std - cv_std) < 1e-6, (
        f"DoD-4 KIRIK: oof_{MODEL}.npy'den yeniden hesap {re_mean:.6f} != {cv_mean:.6f}."
    )

    # --- clip teyidi (hem oof hem test [0,100]) ---
    cv.assert_in_range(oof, "oof_lgbm_num")
    cv.assert_in_range(test_pred, "test_lgbm_num")

    print(f"[anchor] cv_mse_mean = {cv_mean:.4f}   (compute_cv_mse / avg-oof; cv_scores.csv'ye)")
    print(f"[anchor] cv_mse_std  = {cv_std:.4f}")
    print(f"[anchor] recency_weighted_oof_mse = {recency_mse:.4f}   "
          f"(co-headline; private-durust tahmin, review H1)")
    print(f"[anchor] genuine-15  : mean={np.mean(genuine):.4f}  std={np.std(genuine):.4f}  "
          f"(standart repeated-CV fold skorlari)")
    print(f"[anchor] single5fold_std = {single5fold_std:.4f}  (SPEC §2 'tek 5-fold ~4.68' referansi)")
    print(f"[anchor] best_iteration_mean = {best_iter_mean:.1f}  (15 fold)")
    print(f"[anchor] test fold-bagging: mean={test_pred.mean():.3f} std={test_pred.std():.3f} "
          f"min={test_pred.min():.3f} max={test_pred.max():.3f}")
    print(f"[anchor] DoD-4 ic tutarlilik GECTI (oof.npy -> {re_mean:.6f}).")

    if REF_LO <= cv_mean <= REF_HI:
        print(f"[anchor] cv_mse_mean {cv_mean:.2f} yapisal baseline bandinda ({REF_LO}-{REF_HI}); "
              f"dogrulama olcumu 81.69 ile uyumlu -> ALTYAPI DOGRU. "
              f"(87.91 eski/yilsiz, 91.6 stale; bkz review C1/M1.)")
    else:
        print(f"[anchor][UYARI] cv_mse_mean {cv_mean:.2f} beklenen yapisal banttin ({REF_LO}-{REF_HI}) "
              f"DISINDA -> CV altyapisini incele.")


if __name__ == "__main__":
    main()
