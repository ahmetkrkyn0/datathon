"""
Faz 6 — catboost_full: CatBoost (native kategorik) FULL matris, 3-seed averaging.
================================================================================

    python src/catboost_full.py            # unweighted   -> catboost_full
    python src/catboost_full.py weighted   # recency-weighted egitim -> catboost_full_w

NEDEN: LGBM'den YAPISAL FARKLI ikinci GBDT (ordered boosting + native kategorik) -> blend
  cesitliligi (risk dagitimi). FULL matris lgbm_full ile AYNI parcalar (num+YIL+kategorik+
  missing-flag + txt_ridge_pred(nested-OOF) + lexicon(10)) ama kategorik CatBoost'a NATIVE
  string olarak verilir (one-hot/encode YOK -> hedef-bagimsiz, fold-safe). NaN sayisal
  CatBoost native islenir.

KARAR METRIGI = recency_weighted_oof_mse (review H1). weighted varyant (Pool weight=recency)
  dogrudan test dagilimina optimize eder; unweighted ile karsilastirilir, DUSUK olan tutulur.
  (Not: lgbm_full_w'de recency agirlik rw-OOF'u DUSURMEDI; burada tekrar olculur.)

SEED AVG (zaman): CatBoost tek-fit ~46-58s (kucuk veri, kotu paralelize). 3-seed=45 fit ~35-44dk
  cok uzun -> TEK seed (42). Seed-avg yalniz varyans-azaltma nicety; tek blend bileseni icin sart
  degil (MVP). Test = KANONIK fold-bagging (run_oof) -> 15 fold.
FOLD-SAFE: tum fit'ler run_oof ile dis-fold train'inde; txt_ridge nested-OOF artefakti.
Determinizm: random_seed sabit + thread_count SABIT (6) -> CatBoost CPU egitimi ayni
  (thread_count+seed) ile yeniden-uretilebilir (olculdu: 2 fit max|diff|=0). thread_count
  FARKLI olursa sonuc degisebilir; bu yuzden HARDCODE. allow_writing_files=False.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from catboost import CatBoostRegressor, Pool

import artifacts_io as aio
import cv
import text_utils as tu
from lgbm_full import OOF_TXT_PATH, TEST_TXT_PATH, _add_text

CB_SEEDS = (42,)  # tek seed (zaman; thread_count=6 ile reproducible). Seed-avg drop edildi (MVP).

# Muhafazakar CatBoost (anchor felsefesi: overfit kapisi; HP taramasi YOK). depth 6 + l2 3.0.
CB_PARAMS = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=3000,           # ust sinir; early stopping karar verir
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_strength=1.0,
    thread_count=6,            # SABIT (reproducible); thread=1 cok yavas (58s/fit)
    allow_writing_files=False,
    verbose=0,
)
EARLY_STOPPING_ROUNDS = 100

W_CLIP_LO, W_CLIP_HI = 0.10, 5.0


def make_fit_fold_cb(cat_features, w_full=None, n_total=cv.N_REPEATS * cv.N_SPLITS):
    """fit_fold -> (predict_fn, best_it). Seed-avg (CB_SEEDS); opsiyonel recency Pool weight (fold-ici).
    Gorunurluk: her fold sonunda ilerleme + sure yazar (uzun calisma -> nezaret)."""
    state = {"k": 0}

    def fit_fold(X_tr, y_tr, X_val, y_val):
        t0 = time.time()
        pos_tr = np.asarray(X_tr.index, dtype=int)
        pos_val = np.asarray(X_val.index, dtype=int)
        w_tr = None if w_full is None else w_full[pos_tr]
        w_val = None if w_full is None else w_full[pos_val]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features, weight=w_tr)
        val_pool = Pool(X_val, y_val, cat_features=cat_features, weight=w_val)

        models = []
        best_its = []
        for s in CB_SEEDS:
            m = CatBoostRegressor(random_seed=s, **CB_PARAMS)
            m.fit(train_pool, eval_set=val_pool,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, use_best_model=True)
            models.append(m)
            best_its.append(int(m.get_best_iteration() or CB_PARAMS["iterations"]))

        def predict(X):
            P = Pool(X, cat_features=cat_features)
            return np.mean([m.predict(P) for m in models], axis=0)  # HAM; clip run_oof'ta

        state["k"] += 1
        print(f"[catboost] fit {state['k']}/{n_total} bitti ({time.time()-t0:.1f}s, "
              f"best_it~{int(np.mean(best_its))})", flush=True)
        return predict, int(np.mean(best_its))

    return fit_fold


def build_cb_matrix(df):
    """CatBoost FULL matris: lgbm_full parcalari ama kategorik NATIVE string (Categorical->str)."""
    X_struct, cat_features = cv.build_structured_matrix(df, cv.structured_cat_dtypes(df))
    X_struct = X_struct.copy()
    for c in cat_features:
        X_struct[c] = X_struct[c].astype(str)  # CatBoost native kategorik = string (encode yok)
    return X_struct, cat_features


def main() -> None:
    cv.set_seed()
    weighted = len(sys.argv) > 1 and sys.argv[1].lower().startswith("w")
    MODEL = "catboost_full_w" if weighted else "catboost_full"

    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    oof_txt = np.load(OOF_TXT_PATH)
    test_txt = np.load(TEST_TXT_PATH)
    cv.assert_in_range(oof_txt, "oof_txt_ridge")
    cv.assert_in_range(test_txt, "test_txt_ridge")
    lex_tr = tu.extract_handcrafted_features(train[cv.TEXT_COL].values)
    lex_te = tu.extract_handcrafted_features(test[cv.TEXT_COL].values)

    X_struct, cat_features = build_cb_matrix(train)
    Xt_struct, cat_features_te = build_cb_matrix(test)
    assert list(X_struct.columns) == list(Xt_struct.columns) and cat_features == cat_features_te

    X = _add_text(X_struct, oof_txt, lex_tr)
    X_test = _add_text(Xt_struct, test_txt, lex_te)
    assert list(X.columns) == list(X_test.columns), "FULL train/test kolonlari hizali degil."

    w_full = None
    if weighted:
        w_full = np.clip(cv.recency_weights(train, test), W_CLIP_LO, W_CLIP_HI)
        print(f"[catboost] recency Pool weight: mean={w_full.mean():.4f} "
              f"min={w_full.min():.4f} max={w_full.max():.4f}")

    print(f"[catboost] {MODEL}: {X.shape[1]} feature, {len(cat_features)} native-kategorik, "
          f"seed-avg {CB_SEEDS} (thread_count={CB_PARAMS['thread_count']}), weighted={weighted}.")

    out = cv.run_oof(make_fit_fold_cb(cat_features, w_full), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]
    genuine = out["genuine_fold_mse"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.recency_weights(train, test)
    recency_mse = cv.compute_recency_weighted_mse(oof, y, rw)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))
    single5fold_std = float(np.std(genuine[: cv.N_SPLITS]))

    note = (
        f"{MODEL} = CatBoost native-kategorik FULL matris (num+YIL+kategorik+flag+txt_ridge+"
        f"lexicon), seed-avg {CB_SEEDS} thread_count={CB_PARAMS['thread_count']}, weighted={weighted}. "
        f"KARAR recency_weighted_oof_mse"
        f"={recency_mse:.4f} (lgbm_full 87.2663). Unweighted CV={cv_mean:.4f}. Fold-safe."
    )

    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, best_iter_mean)
    aio.write_cv_log(
        MODEL, cv_mean, cv_std, fold_mse, best_iters, best_iter_mean,
        genuine_fold_mse=genuine, single5fold_std=single5fold_std, note=note,
    )
    aio.log_model_score(MODEL, cv_mean, cv_std, recency_mse, weighted_training=weighted, note=note)

    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, re_std, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6 and abs(re_std - cv_std) < 1e-6, "DoD-4 KIRIK."
    cv.assert_in_range(oof, f"oof_{MODEL}")
    cv.assert_in_range(test_pred, f"test_{MODEL}")

    base_rw = 87.266319
    print(f"[catboost] unweighted cv_mse_mean = {cv_mean:.4f} +/- {cv_std:.4f}  (KARAR DEGIL)")
    print(f"[catboost] recency_weighted_oof_mse = {recency_mse:.4f}   (KARAR METRIGI; "
          f"lgbm_full {base_rw:.4f}, delta {recency_mse - base_rw:+.4f})")
    print(f"[catboost] best_iteration_mean = {best_iter_mean:.1f}")
    print(f"[catboost] test fold-bagging: mean={test_pred.mean():.3f} std={test_pred.std():.3f} "
          f"min={test_pred.min():.3f} max={test_pred.max():.3f}")
    verdict = "DUSURDU (degerli)" if recency_mse < base_rw else "dusurmedi"
    print(f"[catboost] >>> tek-model recency-weighted OOF {verdict} (karar: rw-OOF).")


if __name__ == "__main__":
    main()
