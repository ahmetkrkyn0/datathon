"""
Faz 5 — NLP smoke test (LEAN; tek dosya). Hizli, kucuk altkume uzerinde:
  * turkish_lower Turkce I/i tuzagi dogru.
  * build_tfidf_ridge_oof NESTED OOF kapsami TAM (her satir tam n_repeats kez) + clip[0,100].
  * ic-fit'in dis-valid'e dokunmadigi (sizinti yok) yapisal olarak: oof[i] i'nin dis-fold'undan
    DISLANARAK uretildi -> determinizm + sonlu deger.
  * extract_handcrafted_features: 10 kolon, beklenen sayimlar (n_pos/n_neg/has_ancak/len_word).

    python src/test_nlp_smoke.py
"""

from __future__ import annotations

import numpy as np

import cv
import text_utils as tu


def test_turkish_lower():
    assert tu.turkish_lower("I") == "ı", "Turkce 'I' -> 'ı' olmali (ASCII lower degil)."
    assert tu.turkish_lower("İ") == "i", "Turkce noktali 'İ' -> 'i' olmali."
    assert tu.turkish_lower("GÜÇLÜ") == "güçlü"
    print("[smoke] turkish_lower OK")


def test_handcrafted():
    texts = [
        "Güçlü bir aday, etkileyici potansiyel. Ancak SQL tarafini gelistirmesi gerekiyor.",
        "Yüksek başarı ve mükemmel iletişim.",  # 'ancak' yok
    ]
    df = tu.extract_handcrafted_features(texts)
    assert list(df.columns) == tu.HANDCRAFTED_COLS and df.shape == (2, 10)
    assert df.loc[0, "has_ancak"] == 1 and df.loc[1, "has_ancak"] == 0
    assert df.loc[0, "n_pos"] >= 2 and df.loc[0, "n_neg"] >= 2  # guclu/etkileyici/potansiyel ; ancak/gelistir/gerekiyor
    assert df.loc[0, "n_skill_mention"] >= 1  # sql
    assert (df["len_word"] > 0).all() and (df["n_sentence"] >= 1).all()
    print(f"[smoke] handcrafted OK (n_pos0={df.loc[0,'n_pos']}, n_neg0={df.loc[0,'n_neg']})")


def test_nested_oof_coverage():
    train = cv.load_train().iloc[:600].reset_index(drop=True)
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    texts = tu.normalize_texts(train[cv.TEXT_COL].values)
    texts_te = tu.normalize_texts(cv.load_test()[cv.TEXT_COL].values[:200])

    folds = cv.get_folds(y, sid, seeds=(42,))  # 1 repeat -> hizli
    oof, test = tu.build_tfidf_ridge_oof(
        texts, y, texts_te, folds, sid, alpha=2.0, n_repeats=1, n_inner=3,
    )
    # Kapsam: build_tfidf_ridge_oof icindeki assert (oof_cnt==n_repeats) zaten gecti; sonuc dogrula.
    assert oof.shape == (600,) and test.shape == (200,)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    assert oof.min() >= 0.0 and oof.max() <= 100.0, "OOF clip[0,100] disinda."
    assert test.min() >= 0.0 and test.max() <= 100.0, "test clip[0,100] disinda."
    # Determinizm: ayni cagri ayni sonuc.
    oof2, _ = tu.build_tfidf_ridge_oof(
        texts, y, texts_te, folds, sid, alpha=2.0, n_repeats=1, n_inner=3,
    )
    assert np.allclose(oof, oof2), "build_tfidf_ridge_oof deterministik degil."
    print(f"[smoke] nested OOF coverage+clip+determinizm OK (corr={np.corrcoef(oof,y)[0,1]:.3f})")


if __name__ == "__main__":
    cv.set_seed()
    test_turkish_lower()
    test_handcrafted()
    test_nested_oof_coverage()
    print("[smoke] TUM TESTLER GECTI")
