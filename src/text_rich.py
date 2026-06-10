"""
FORENSICS LEVER — `txt_rich`: zenginlestirilmis nested-OOF metin meta-feature.
=============================================================================
GEREKCE (FORENSICS C3 reversal): adversarial-verify, "metin REDUNDANT" tezini curuttu. Metnin
degeri ETKILESIMLI (joint num+text -5.24, additive'den -4.38 iyi) ve mevcut txt_ridge metni
DOYURMAMIS: zengin TF-IDF (word 1-3 + char 2-6) num tabaninda txt_ridge USTUNE -0.87 uw marjinal
verdi (forensics7). Bu, BERT-elemesini gecersiz kilar VE dogrudan denenebilir bir lever.

Bu dosya, text_utils ile AYNI sizinti sozlesmesini (nested inner-KFold OOF + fold-bagged test)
kullanarak `oof_txt_rich.npy` / `test_txt_rich.npy` uretir. Tek fark: vectorizer = word(1-3) +
char_wb(2-6) birlesik (hstack), daha buyuk max_features. Hicbir fit dis-valid/test'e dokunmaz.

    python src/text_rich.py        # artefakt uretir + standalone rw-OOF raporlar

KARAR: blend'e ekleme karari ensemble.py'de (CANDIDATE_POOL'a 'txt_rich' eklenip nested rw-OOF
olculur; kapi 85.4945 - 0.25*3.0238 = 84.7385). Bu script yalniz feature'i URETIR + tek-basina olcer.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import artifacts_io as aio
import cv
import text_utils as tu

MODEL = "txt_rich"
ALPHA = 2.0
N_INNER = 5
WORD_NGRAM = (1, 3)
CHAR_NGRAM = (2, 6)
MAX_FEAT = 80000


def _make_vecs():
    stop = [tu.turkish_lower(w) for w in __import__("lexicon_tr").STOPWORDS]
    wv = TfidfVectorizer(analyzer="word", ngram_range=WORD_NGRAM, min_df=2, max_features=MAX_FEAT,
                         sublinear_tf=True, lowercase=False, stop_words=stop, dtype=np.float32)
    cvz = TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM, min_df=2, max_features=MAX_FEAT,
                          sublinear_tf=True, lowercase=False, dtype=np.float32)
    return wv, cvz


def build_rich_oof(texts_norm, y, texts_norm_test, folds, sid, alpha=ALPHA, n_inner=N_INNER):
    """Nested inner-KFold OOF + fold-bagged test (text_utils.build_tfidf_ridge_oof ile AYNI iskelet,
    zengin word+char vectorizer ile). Sizintisiz: her fit yalniz ic-train'de."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_test = len(texts_norm_test)
    oof_sum = np.zeros(n); oof_cnt = np.zeros(n); test_sum = np.zeros(n_test); n_test_models = 0

    for r in range(cv.N_REPEATS):
        fold_of = cv.fold_of_rows(folds, sid, r)
        for f in range(cv.N_SPLITS):
            val_idx = np.where(fold_of == f)[0]
            tr_idx = np.where(fold_of != f)[0]
            inner = KFold(n_splits=n_inner, shuffle=True, random_state=cv.SEED + r)
            val_acc = np.zeros(len(val_idx)); test_acc = np.zeros(n_test); k = 0
            for inner_tr_rel, _ in inner.split(tr_idx):
                idx = tr_idx[inner_tr_rel]  # SADECE ic-train
                wv, cvz = _make_vecs()
                Xtr = hstack([wv.fit_transform(texts_norm[idx]), cvz.fit_transform(texts_norm[idx])]).tocsr()
                model = Ridge(alpha=alpha)
                model.fit(Xtr, y[idx])
                Xval = hstack([wv.transform(texts_norm[val_idx]), cvz.transform(texts_norm[val_idx])]).tocsr()
                Xte = hstack([wv.transform(texts_norm_test), cvz.transform(texts_norm_test)]).tocsr()
                val_acc += model.predict(Xval)
                test_acc += model.predict(Xte)
                k += 1
            oof_sum[val_idx] += val_acc / k
            oof_cnt[val_idx] += 1.0
            test_sum += test_acc / k
            n_test_models += 1
    assert np.all(oof_cnt == cv.N_REPEATS), "txt_rich OOF kapsami bozuk."
    return cv.clip_predictions(oof_sum / oof_cnt), cv.clip_predictions(test_sum / float(n_test_models))


def main():
    cv.set_seed()
    train = cv.load_train(); test = cv.load_test(); folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)
    texts_tr = tu.normalize_texts(train[cv.TEXT_COL].values)
    texts_te = tu.normalize_texts(test[cv.TEXT_COL].values)

    print(f"[txt_rich] nested-OOF zengin metin (word{WORD_NGRAM}+char{CHAR_NGRAM}, max_feat={MAX_FEAT})...")
    oof, test_pred = build_rich_oof(texts_tr, y, texts_te, folds, sid)
    cv.assert_in_range(oof, "oof_txt_rich"); cv.assert_in_range(test_pred, "test_txt_rich")

    cm, cs, _ = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.compute_recency_weighted_mse(oof, y, w)
    proj = np.clip(np.load(cv.ARTIFACTS_DIR / "oof_txt_ridge.npy"), 0, 100)
    print(f"[txt_rich] standalone: unweighted_cv={cm:.4f} +/- {cs:.4f}  rw-OOF={rw:.4f}")
    print(f"[txt_rich] vs proje txt_ridge rw-OOF={cv.compute_recency_weighted_mse(proj, y, w):.4f}  "
          f"corr={np.corrcoef(oof, proj)[0,1]:.4f}")

    aio.save_oof_test(MODEL, oof, test_pred)
    aio.log_model_score(MODEL, cm, cs, rw, weighted_training=False,
                        note=f"FORENSICS lever: zengin word{WORD_NGRAM}+char{CHAR_NGRAM} nested-OOF metin "
                             f"(C3 reversal). standalone rw={rw:.4f}. blend karari ensemble.py'de.")
    print(f"[txt_rich] yazildi: artifacts/oof_{MODEL}.npy, test_{MODEL}.npy + ledger satiri.")


if __name__ == "__main__":
    main()
