"""
Faz 3 — Ham veri temizleme (SPEC 03 §2). clean_raw: ISTATISTIKSIZ, fold-bagimsiz,
sizintisiz SAF donusum. Tip duzeltmeleri + kolon drop + kategorik dtype atama.

    from cleaning import clean_raw
    clean = clean_raw(cv.load_train())     # student_id atilir, yillar HAM SAYISAL kalir

KURALLAR (SPEC 03):
  * DROP yalnizca `student_id` (sentetik non-predictive anahtar). Yillar
    (application_year/graduation_year) HAM SAYISAL feature olarak TUTULUR (review C1).
  * Hicbir istatistik (medyan/encoding/scaler) burada hesaplanmaz -> fold-bagimsiz, sizinti
    yapisal olarak imkansiz. Tum fit-li donusumler preprocessing.build_preprocessor'da.
  * BOM (﻿) kolon adlarindan temizlenir (cv.load_* utf-8-sig okur ama defensive strip).
  * `mentor_feedback_text` PASSTHROUGH (Faz 05 NLP'ye devredilir); ftfy/latin1 fix YAPILMAZ.
  * `career_success_score` (varsa) dokunulmaz; cagiran taraf X'ten ayirir.
"""

from __future__ import annotations

import pandas as pd

import cv


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Ham train/test cercevesini modele-hazir tiplere getirir (istatistiksiz, sizintisiz).

    - BOM strip kolon adlarinda.
    - DROP: `student_id` (varsa). Yillar TUTULUR (HAM SAYISAL).
    - Kategorik 5 kolon -> `category` dtype (str normalize sonrasi; native-kategorik destegi).
    - Yil 2 kolon + kalan sayisal kolonlar -> `float64` (sayimlar NaN tutabilsin diye float).
    - `mentor_feedback_text`, `career_success_score` dokunulmaz (passthrough).

    Doner: kopya DataFrame (girdi mutate edilmez).
    """
    df = df.copy()

    # BOM (﻿) header'da gelebilir -> aksi halde student_id adi eslesmez (SPEC 03 §1).
    df.columns = df.columns.str.replace("﻿", "", regex=False)

    # DROP: yalnizca student_id (ID/satir-sirasi leak). Yillar drop EDILMEZ (review C1).
    if cv.ID_COL in df.columns:
        df = df.drop(columns=[cv.ID_COL])

    # Kategorik dtype (str normalize -> cv.structured_cat_dtypes ile ayni evren mantigi).
    for c in cv.CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).astype("category")

    # Yillar: HAM SAYISAL float64 (raw feature; drop YOK).
    for c in cv.YEAR_COLS:
        if c in df.columns:
            df[c] = df[c].astype("float64")

    # Kalan sayisal feature'lar (yil-disi): float64 (count int -> float, NaN korunur).
    for c in cv.numeric_feature_columns(df):
        if c in df.columns:
            df[c] = df[c].astype("float64")

    return df
