"""
Faz 6 — LEVER3: TF-IDF -> fold-ici TruncatedSVD -> GBDT (nonlineer metin base modeli).
=======================================================================================

    python src/txt_svd_gbdt.py

NEDEN: Mevcut metin sinyali txt_ridge LINEER (TF-IDF->Ridge). SVD ile yogun (dense) latent metin
  boyutlari cikarip GBDT'ye vermek NONLINEER metin etkilesimleri yakalayabilir -> txt_ridge'den
  FARKLI bir metin gorunumu -> blend cesitliligi. CEKIRDEGI RISKE ATMAZ: yeni oof_txt_svd_gbdt
  artefakti; ensemble havuzuna girer, NESTED rw-OOF DUSERSE tutulur (yoksa ~0 agirlik/elenir).

FOLD-SAFE (PAZARLIKSIZ): TfidfVectorizer + TruncatedSVD HER dis-fold train'inde fit_fold ICINDE
  fit edilir (dis-valid/test'e ASLA fit YOK), random_state=42. cv.run_oof X olarak HAM normalize
  metni alir (tek-kolon DataFrame); fit_fold X_tr metnini vektörize+SVD eder, X_val/X_test'i AYNI
  fit'lerle transform eder. SVD latent boyutlari -> LGBM (anchor HP). Test = 15-fold bagging.

KARAR = standalone rw-OOF (rapor) + ensemble.py'de NESTED rw-OOF + 0.25*std kapisi.
Determinizm: SVD random_state=42 (randomized solver deterministik), LGBM deterministic=True n_jobs=1.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import TruncatedSVD

import artifacts_io as aio
import cv
import text_utils as tu

MODEL = "txt_svd_gbdt"
SVD_COMPONENTS = 80            # 50-100 bandi ortasi; tek deger (HP taramasi yok, anchor felsefesi)
SVD_RANDOM_STATE = cv.SEED

# Metin-SVD GBDT (anchor LGBM ile ayni muhafazakar HP; metin latentleri sayisal feature gibi).
LGBM_PARAMS = dict(
    objective="regression_l2",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=cv.SEED,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 100


def make_fit_fold_svd():
    """fit_fold(X_tr,y_tr,X_val,y_val) -> (predict_fn, best_it). X = tek-kolon ('text') DataFrame.

    TF-IDF + TruncatedSVD dis-fold train metninde fit (fold-safe); LGBM SVD latentleri uzerinde.
    Early stopping ic-valid = dis-valid SVD donusumu (dis-valid'e fit YOK -> sizinti yok)."""
    def _txt(X):
        return np.asarray(X["text"].values, dtype=object)

    def fit_fold(X_tr, y_tr, X_val, y_val):
        vec = tu.make_tfidf(analyzer="word")            # kilitli TF-IDF (word 1-2gram, stopword'lu)
        Xtr_tfidf = vec.fit_transform(_txt(X_tr))
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SVD_RANDOM_STATE)
        Ztr = svd.fit_transform(Xtr_tfidf)
        Zval = svd.transform(vec.transform(_txt(X_val)))

        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(
            Ztr, y_tr,
            eval_set=[(Zval, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        best_it = m.best_iteration_ or LGBM_PARAMS["n_estimators"]

        def predict(X):
            Z = svd.transform(vec.transform(_txt(X)))
            return m.predict(Z, num_iteration=best_it)  # HAM; clip run_oof'ta

        return predict, int(best_it)

    return fit_fold


def main() -> None:
    cv.set_seed()
    train = cv.load_train(); test = cv.load_test(); folds = cv.load_folds()
    y = train[cv.TARGET_COL].values; sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    # HAM normalize metin -> tek-kolon DataFrame (run_oof iloc[tr_idx] ile fold-ici bolunur).
    txt_tr = tu.normalize_texts(train[cv.TEXT_COL].values)
    txt_te = tu.normalize_texts(test[cv.TEXT_COL].values)
    X = pd.DataFrame({"text": txt_tr})
    X_test = pd.DataFrame({"text": txt_te})
    print(f"[txt_svd] {MODEL}: TF-IDF(word 1-2gram) -> SVD({SVD_COMPONENTS}) -> LGBM "
          f"(fold-ici fit, random_state={SVD_RANDOM_STATE}).")

    out = cv.run_oof(make_fit_fold_svd(), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]
    genuine = out["genuine_fold_mse"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.compute_recency_weighted_mse(oof, y, w)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))
    single5fold_std = float(np.std(genuine[: cv.N_SPLITS]))

    # mevcut metin base txt_ridge ile karsilastir + korelasyon (cesitlilik gostergesi)
    oof_ridge = np.load(cv.ARTIFACTS_DIR / "oof_txt_ridge.npy")
    rw_ridge = cv.compute_recency_weighted_mse(oof_ridge, y, w)
    corr = float(np.corrcoef(oof, oof_ridge)[0, 1])

    note = (
        f"{MODEL} = TF-IDF(word 1-2gram) -> fold-ici TruncatedSVD({SVD_COMPONENTS}) -> LGBM "
        f"(nonlineer metin base, LEVER3). standalone rw-OOF={rw:.4f} (txt_ridge {rw_ridge:.4f}; "
        f"corr={corr:.3f}). Fold-safe (vectorizer+SVD dis-fold train'inde). Blend faydasi ensemble.py "
        f"NESTED rw-OOF + 0.25*std kapisi karar verir."
    )

    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, best_iter_mean)
    aio.write_cv_log(
        MODEL, cv_mean, cv_std, fold_mse, best_iters, best_iter_mean,
        genuine_fold_mse=genuine, single5fold_std=single5fold_std, note=note,
    )
    aio.log_model_score(MODEL, cv_mean, cv_std, rw, weighted_training=False, note=note)

    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, re_std, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6 and abs(re_std - cv_std) < 1e-6, "DoD-4 KIRIK."
    cv.assert_in_range(oof, f"oof_{MODEL}"); cv.assert_in_range(test_pred, f"test_{MODEL}")

    print(f"[txt_svd] standalone: unweighted_cv={cv_mean:.4f}  rw-OOF={rw:.4f}  "
          f"(txt_ridge rw={rw_ridge:.4f}; corr={corr:.3f})")
    print(f"[txt_svd] best_iteration_mean={best_iter_mean:.1f}; test mean={test_pred.mean():.3f} "
          f"std={test_pred.std():.3f}")
    print(f"[txt_svd] {'DAHA IYI' if rw < rw_ridge else 'daha kotu/esit'} standalone vs txt_ridge. "
          f"Blend: python src/ensemble.py (havuza ekle).")


if __name__ == "__main__":
    main()
