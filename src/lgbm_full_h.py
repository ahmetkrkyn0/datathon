"""
TIER-3 ROBUST-LOSS — lgbm_full_h: lgbm_full'un Huber-loss varyanti (alpha=5).
=============================================================================

    python src/lgbm_full_h.py

NE: lgbm_full ile BIREBIR ayni feature matrisi (yapisal + txt_ridge_pred + lexicon) ve ayni
  15-fold protokol; TEK fark objective='huber' (alpha=5). Cikti: oof/test_lgbm_full_h.npy + ledger.

NEDEN (mekanizma, reports/LOW_TAIL_LEVER.md zinciri): alt-kuyruk surprizleri (ex-ante ayirt
  edilemeyen dusuk-y satirlar; residual +-10..30) L2 egitiminde karesel gradyanla kutlenin
  (nufusun %95'i) fit'ini bozuyor. Kuyruk DUZELTILEMEZ (iki-asama Bayes-tutarlilik kaniti:
  her beta>0 kotulesti) ama kuyrugun kutleyi zehirlemesi ENGELLENEBILIR: Huber, |resid|>alpha
  bolgesinde gradyani sabitler (robust regresyon). Ilk sizintisiz olcum (repeat-0):
    L2 rw=88.4754 -> huber(a=5) rw=87.1618 (-1.31), unw 78.43->77.01 (-1.42).
  NOT: ayni mekanizmanin trimming (satir atma) versiyonu -2.19 vermisti ama filtre blend-OOF
  kullaniyordu (capraz-fold y sizintisi riski) -> REDDEDILDI; Huber filtresiz/sizintisiz esdegeri.

ALPHA SECIMI: repeat-0 fold-safe OOF taramasi (gate-kor, y'ye bakar; e5 alpha secimiyle ayni
  emsal): a=0.9(def) +1.15 KOTU, a=5 -1.31 EN IYI, a=10 -0.77, fair +1.44 KOTU -> alpha=5.

KARAR: standalone rw-OOF (rapor) + ensemble paired-test (mm/e5 olcutu) blend'e girisine karar
  verir. SUB-1 adayligi: finalize_submissions en dusuk rw'li tek-GBDT'yi secer; lgbm_full_h
  catboost_full'u (86.41) gecerse SUB-1 dogal degisir (ayni kural). Public'e BAKILMAZ.
Determinizm: lgbm_full ile ayni (SEED=42, deterministic=True, n_jobs=1).
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb

import artifacts_io as aio
import cv
import text_utils as tu
from anchor_lgbm_num import LGBM_PARAMS, EARLY_STOPPING_ROUNDS

MODEL = "lgbm_full_h"
HUBER_ALPHA = 5.0

OOF_TXT_PATH = cv.ARTIFACTS_DIR / "oof_txt_ridge.npy"
TEST_TXT_PATH = cv.ARTIFACTS_DIR / "test_txt_ridge.npy"


def make_fit_fold_huber(cat_features):
    """anchor_lgbm_num.make_fit_fold'un Huber-loss kopyasi (objective + alpha disinda AYNI)."""
    params = dict(LGBM_PARAMS)
    params["objective"] = "huber"
    params["alpha"] = HUBER_ALPHA

    def fit_fold(X_tr, y_tr, X_val, y_val):
        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",  # erken durdurma KARAR metriginin vekiliyle (MSE), huber'le degil
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        best_it = model.best_iteration_ or params["n_estimators"]

        def predict(X):
            return model.predict(X, num_iteration=best_it)

        return predict, int(best_it)

    return fit_fold


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    oof_txt = np.load(OOF_TXT_PATH)
    test_txt = np.load(TEST_TXT_PATH)
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
    print(f"[{MODEL}] {X.shape[1]} feature (lgbm_full ile birebir); objective=huber alpha={HUBER_ALPHA:g}")

    out = cv.run_oof(make_fit_fold_huber(cat_features), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    w = cv.recency_weights(train, test)
    rw = cv.compute_recency_weighted_mse(oof, y, w)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))

    # referans: L2 ikizi
    oof_l2 = np.load(cv.ARTIFACTS_DIR / "oof_lgbm_full.npy")
    rw_l2 = cv.compute_recency_weighted_mse(oof_l2, y, w)

    note = (
        f"{MODEL} = lgbm_full'un Huber(alpha={HUBER_ALPHA:g}) varyanti (TIER-3 robust-loss; alt-kuyruk "
        f"surprizlerinin L2 fit'ini zehirlemesini engeller; alpha repeat-0 fold-safe taramayla, e5 emsal). "
        f"standalone rw-OOF={rw:.4f} (L2 ikizi lgbm_full {rw_l2:.4f}). Blend karari ensemble paired-test."
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
    print(f"[{MODEL}] rw-OOF={rw:.4f}  (L2 ikizi lgbm_full {rw_l2:.4f}; fark {rw - rw_l2:+.4f})")
    print(f"[{MODEL}] test: mean={test_pred.mean():.3f} std={test_pred.std():.3f}")
    print(f"[{MODEL}] YAZILDI: artifacts/oof_{MODEL}.npy, test_{MODEL}.npy + ledger satirlari")


if __name__ == "__main__":
    main()
