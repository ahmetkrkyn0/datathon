"""
Faz 6 — STEP2 (spekulatif edge): GUCLU metin modeli = word + char TF-IDF BIRLESIK -> Ridge.
=============================================================================================

    python src/text_strong.py

NEDEN: txt_ridge tek-basina ZAYIF (rw-OOF 168; word 147 / char 156 unweighted standalone). Ama
  word ve char sinyalleri farkli (kok vs morfoloji/yazim). BIRLESIK (hstack) Ridge tek-basina
  daha iyi olabilir; OLABILIR ise blend'e EK aday olur. Bu adim CEKIRDEGI RISKE ATMAZ: mevcut
  GBDT'lere/finallere DOKUNMAZ, yalnizca yeni oof_txt_ridge_wc artefakti uretir; ensemble havuzuna
  eklenir, NESTED rw-OOF DUSERSE tutulur (yoksa NNLS/greedy ~0 agirlik verir / silinir).

FOLD-SAFE: build_tfidf_ridge_oof ile AYNI nested yapı (her fit yalniz ic-train; dis-valid/test =
  inner modellerin ortalamasi). KARAR METRIGI recency-weighted OOF.
Determinizm: SEED=42; sabit TF-IDF HP.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import artifacts_io as aio
import cv
import text_utils as tu

MODEL = "txt_ridge_wc"
ALPHAS = (1.0, 2.0, 5.0, 10.0)
N_INNER = 5
CHAR_NGRAM = (3, 5)


def _fit_pair(texts_tr_idx, texts_all, y_tr, alpha):
    """word+char vectorizer ic-train'de fit; Ridge birlesik matriste. Doner: (predict_fn)."""
    wv = tu.make_tfidf(analyzer="word")
    cv_ = tu.make_tfidf(analyzer="char_wb", ngram_range=CHAR_NGRAM)
    Xw = wv.fit_transform(texts_all[texts_tr_idx])
    Xc = cv_.fit_transform(texts_all[texts_tr_idx])
    X = hstack([Xw, Xc]).tocsr()
    r = Ridge(alpha=alpha)
    r.fit(X, y_tr)

    def predict(idx):
        Q = hstack([wv.transform(texts_all[idx]), cv_.transform(texts_all[idx])]).tocsr()
        return r.predict(Q)

    return predict


def select_alpha(texts_norm, y, folds, sid):
    """repeat-0 5-fold OOF-MSE ile en iyi alpha (goreli karsilastirma, fold-safe)."""
    y = np.asarray(y, dtype=float)
    fold_of = cv.fold_of_rows(folds, sid, 0)
    res = {}
    for a in ALPHAS:
        oof = np.zeros(len(y))
        for f in range(cv.N_SPLITS):
            va = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            pred = _fit_pair(tr, texts_norm, y[tr], a)
            oof[va] = pred(va)
        res[float(a)] = float(np.mean((y - cv.clip_predictions(oof)) ** 2))
    best = min(res, key=res.get)
    return best, res


def build_oof(texts_norm, y, texts_norm_test, folds, sid, alpha):
    """Nested inner-KFold OOF + fold-bagged test (build_tfidf_ridge_oof ile ayni sema)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_test = len(texts_norm_test)
    all_tr = texts_norm
    oof_sum = np.zeros(n); oof_cnt = np.zeros(n); test_sum = np.zeros(n_test); ntm = 0
    # test'i ayri dizi: _fit_pair idx ile all_tr'a bakar -> test icin ayri predict gerekir.
    for r in range(cv.N_REPEATS):
        fold_of = cv.fold_of_rows(folds, sid, r)
        for f in range(cv.N_SPLITS):
            va = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            inner = KFold(n_splits=N_INNER, shuffle=True, random_state=cv.SEED + r)
            vacc = np.zeros(len(va)); tacc = np.zeros(n_test); k = 0
            for itr_rel, _ in inner.split(tr):
                idx = tr[itr_rel]
                wv = tu.make_tfidf(analyzer="word")
                cvz = tu.make_tfidf(analyzer="char_wb", ngram_range=CHAR_NGRAM)
                Xw = wv.fit_transform(all_tr[idx]); Xc = cvz.fit_transform(all_tr[idx])
                model = Ridge(alpha=alpha); model.fit(hstack([Xw, Xc]).tocsr(), y[idx])
                vacc += model.predict(hstack([wv.transform(all_tr[va]), cvz.transform(all_tr[va])]).tocsr())
                tacc += model.predict(hstack([wv.transform(texts_norm_test), cvz.transform(texts_norm_test)]).tocsr())
                k += 1
            oof_sum[va] += vacc / k; oof_cnt[va] += 1.0; test_sum += tacc / k; ntm += 1
    assert np.all(oof_cnt == cv.N_REPEATS)
    return cv.clip_predictions(oof_sum / oof_cnt), cv.clip_predictions(test_sum / ntm)


def main() -> None:
    cv.set_seed()
    train = cv.load_train(); test = cv.load_test(); folds = cv.load_folds()
    y = train[cv.TARGET_COL].values; sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    texts_tr = tu.normalize_texts(train[cv.TEXT_COL].values)
    texts_te = tu.normalize_texts(test[cv.TEXT_COL].values)

    best_alpha, ares = select_alpha(texts_tr, y, folds, sid)
    print("[text_strong] alpha (repeat-0 OOF-MSE): "
          + "  ".join(f"a={a}:{m:.3f}" for a, m in ares.items()) + f"  -> {best_alpha}")

    oof, test_pred = build_oof(texts_tr, y, texts_te, folds, sid, best_alpha)
    cv.assert_in_range(oof, "oof_txt_ridge_wc"); cv.assert_in_range(test_pred, "test_txt_ridge_wc")

    cv_mean, cv_std, _ = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.compute_recency_weighted_mse(oof, y, w)
    # mevcut txt_ridge ile karsilastir
    oof_old = np.load(cv.ARTIFACTS_DIR / "oof_txt_ridge.npy")
    rw_old = cv.compute_recency_weighted_mse(oof_old, y, w)
    corr = float(np.corrcoef(oof, oof_old)[0, 1])

    aio.save_oof_test(MODEL, oof, test_pred)
    aio.log_model_score(MODEL, cv_mean, cv_std, rw, weighted_training=False,
                        note=f"word+char birlesik Ridge alpha={best_alpha} (STEP2 spekulatif metin)")

    print(f"[text_strong] {MODEL} standalone: unweighted_cv={cv_mean:.4f}  rw-OOF={rw:.4f}  "
          f"(eski txt_ridge rw={rw_old:.4f}; corr={corr:.3f})")
    print(f"[text_strong] {'DAHA IYI' if rw < rw_old else 'daha kotu/esit'} standalone. "
          f"Blend faydasi icin: python src/ensemble.py (havuza eklendi).")


if __name__ == "__main__":
    main()
