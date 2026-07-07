"""
Faz 5 — NLP: Turkce metin yardimcilari (SPEC 05). Deterministik (SEED=42), fold-safe.
=====================================================================================

Iceri:
  * turkish_lower(s)               — Turkce-duyarli kucuk-harf (I->i, I->i tuzagi; SPEC 05 §1).
  * make_tfidf()                   — kilitli TfidfVectorizer (word 1-2gram, min_df=3, sublinear).
  * build_tfidf_ridge_oof(...)     — NESTED inner-KFold OOF + fold-bagged test (SPEC 05 §2).
  * extract_handcrafted_features() — 10 elle tasarlanmis Turkce sozluk/yapi ozelligi (Katman B).

SIZINTI SOZLESMESI (SPEC 05 §6, cv.py UST OTORITE):
  Vectorizer + Ridge ASLA tum-train'e veya train+test'e fit edilmez. build_tfidf_ridge_oof'ta
  her fit yalniz IC-train (dis-fold train'inin inner-train parcasi) uzerinde; dis-valid ve test
  o dis fold'un inner modellerinin ORTALAMASI. Boylece klasik stacking sizintisi yapisal olarak
  engellenir ve uretilen oof_txt_ridge diger oof_* ile satir-hizali, folds.parquet'e bagli kalir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import cv
import lexicon_tr

# TF-IDF kilitli HP (SPEC 05 §2). lowercase=False: metin DISARIDA turkish_lower'dan gecer
# (sklearn'in ASCII lower'i Turkce 'I' tuzagina duser). Stopword'ler de ayni normalizasyonda.
TFIDF_NGRAM = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_FEATURES = 20000
RIDGE_ALPHAS = (1.0, 2.0, 5.0)  # SPEC 05 §2: alpha inner-fold CV ile dogrulanir
DEFAULT_ALPHA = 2.0
N_INNER = 5  # nested ic-KFold


# --------------------------------------------------------------------------- #
# Turkce-duyarli kucuk-harf (SPEC 05 §1; metin VE sozluk AYNI fonksiyondan gecer)
# --------------------------------------------------------------------------- #
def turkish_lower(s: str) -> str:
    """'I'->'i', 'I'(noktali)->'i' map'i sonra lower(). str.lower() tek basina 'I'->'i' yapip
    Turkce eslesmeleri sessizce kacirir (LOWERCASE TUZAGI)."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def normalize_texts(texts) -> np.ndarray:
    """Metin dizisini turkish_lower'dan gecirip (n,) object-array dondurur (fancy-index icin)."""
    return np.array([turkish_lower(str(t)) for t in texts], dtype=object)


# --------------------------------------------------------------------------- #
# Kilitli vectorizer fabrikasi (analyzer parametrik: 'word' ana, 'char_wb' char-ngram ablation)
# --------------------------------------------------------------------------- #
def make_tfidf(analyzer: str = "word", ngram_range=TFIDF_NGRAM, min_df: int = TFIDF_MIN_DF,
               max_features: int = TFIDF_MAX_FEATURES) -> TfidfVectorizer:
    stop = list(turkish_lower(w) for w in lexicon_tr.STOPWORDS) if analyzer == "word" else None
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=False,        # metin disarida turkish_lower'lanmis
        stop_words=stop,
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Alpha secimi — basit OOF (repeat 0, ic-nesting YOK; yalniz goreli karsilastirma) (SPEC 05 §2)
# --------------------------------------------------------------------------- #
def select_alpha(texts_norm, y, folds, sid, alphas=RIDGE_ALPHAS, analyzer: str = "word"):
    """Her alpha icin repeat-0 5-fold OOF-MSE; en dusugu secer. Aday-ici fit yalniz dis-train'de
    (fold-safe). Doner: (best_alpha, {alpha: oof_mse})."""
    y = np.asarray(y, dtype=float)
    fold_of = cv.fold_of_rows(folds, sid, 0)
    results: dict[float, float] = {}
    for a in alphas:
        oof = np.zeros(len(y))
        for f in range(cv.N_SPLITS):
            val = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            vec = make_tfidf(analyzer=analyzer)
            Xtr = vec.fit_transform(texts_norm[tr])
            r = Ridge(alpha=a)
            r.fit(Xtr, y[tr])
            oof[val] = r.predict(vec.transform(texts_norm[val]))
        results[float(a)] = float(
            np.mean((y - cv.clip_predictions(oof)) ** 2)
        )
    best = min(results, key=results.get)
    return best, results


# --------------------------------------------------------------------------- #
# NESTED inner-KFold OOF + fold-bagged test (SPEC 05 §2, ANA)
# --------------------------------------------------------------------------- #
def build_tfidf_ridge_oof(texts_norm, y, texts_norm_test, folds, sid, alpha: float = DEFAULT_ALPHA,
                          n_repeats: int = cv.N_REPEATS, n_inner: int = N_INNER,
                          analyzer: str = "word"):
    """Sizintisiz nested OOF `txt_ridge_pred` + fold-bagged test tahmini.

    Her (repeat, dis-fold) icin: dis-train'i inner KFold ile boler; HER inner-train'de
    vectorizer+Ridge fit; dis-valid ve test tahmini o fold'un inner modellerinin ORTALAMASI.
    Hicbir model dis-valid'e veya test'e fit edilmez -> klasik stacking sizintisi engellenir.

    Doner: (oof (n,), test (n_test,)) — ikisi de clip[0,100].
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_test = len(texts_norm_test)

    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=float)
    test_sum = np.zeros(n_test, dtype=float)
    n_test_models = 0

    for r in range(n_repeats):
        fold_of = cv.fold_of_rows(folds, sid, r)
        for f in range(cv.N_SPLITS):
            val_idx = np.where(fold_of == f)[0]
            tr_idx = np.where(fold_of != f)[0]

            inner = KFold(n_splits=n_inner, shuffle=True, random_state=cv.SEED + r)
            val_acc = np.zeros(len(val_idx), dtype=float)
            test_acc = np.zeros(n_test, dtype=float)
            k = 0
            for inner_tr_rel, _ in inner.split(tr_idx):
                idx = tr_idx[inner_tr_rel]  # SADECE ic-train (dis-valid'e ASLA dokunmaz)
                vec = make_tfidf(analyzer=analyzer)
                Xtr = vec.fit_transform(texts_norm[idx])
                model = Ridge(alpha=alpha)
                model.fit(Xtr, y[idx])
                val_acc += model.predict(vec.transform(texts_norm[val_idx]))
                test_acc += model.predict(vec.transform(texts_norm_test))
                k += 1

            oof_sum[val_idx] += val_acc / k
            oof_cnt[val_idx] += 1.0
            test_sum += test_acc / k
            n_test_models += 1

    # Nested OOF kapsam guvencesi: her satir her repeat'te tam 1 kez dis-valid (SPEC 05 DoD).
    assert np.all(oof_cnt == n_repeats), "txt_ridge OOF kapsami bozuk: satir != n_repeats kez gorulmus."
    oof = cv.clip_predictions(oof_sum / oof_cnt)
    test = cv.clip_predictions(test_sum / float(n_test_models))
    return oof, test


# --------------------------------------------------------------------------- #
# Katman B — elle tasarlanmis Turkce sozluk/yapi ozellikleri (SPEC 05 §3)
# --------------------------------------------------------------------------- #
HANDCRAFTED_COLS = [
    "n_pos", "n_neg", "pos_minus_neg", "has_ancak",
    "len_word", "len_char", "n_sentence", "n_skill_mention",
    "pos_ratio", "neg_ratio",
]


def _count_terms(text: str, terms) -> int:
    """terms listesindeki her terimin substring olarak toplam gecis sayisi (kok/cekim yakalar)."""
    return int(sum(text.count(t) for t in terms))


def extract_handcrafted_features(texts) -> pd.DataFrame:
    """10 sabit Turkce sozluk/yapi ozelligi. Hedef-bagimsiz, fold-bagimsiz (SPEC 05 §3, Katman B).

    Metin turkish_lower'lanir; lexicon terimleri AYNI normalizasyondan gecer (LOWERCASE TUZAGI)."""
    pos_terms = [turkish_lower(t) for t in lexicon_tr.POSITIVE]
    neg_terms = [turkish_lower(t) for t in lexicon_tr.NEGATIVE]
    skill_terms = [turkish_lower(t) for t in lexicon_tr.SKILL]
    ancak = turkish_lower("ancak")

    rows = []
    for t in texts:
        low = turkish_lower(str(t))
        n_pos = _count_terms(low, pos_terms)
        n_neg = _count_terms(low, neg_terms)
        len_word = len(low.split())
        len_char = len(low)
        n_sentence = max(1, low.count("."))
        n_skill = _count_terms(low, skill_terms)
        rows.append((
            n_pos,
            n_neg,
            n_pos - n_neg,
            1 if ancak in low else 0,
            len_word,
            len_char,
            n_sentence,
            n_skill,
            n_pos / (len_word + 1.0),
            n_neg / (len_word + 1.0),
        ))
    return pd.DataFrame(rows, columns=HANDCRAFTED_COLS)
