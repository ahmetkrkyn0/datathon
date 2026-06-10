"""
TIER-3 (multimodal NN) — FOLD-SAFE tabular matris (XLM-R + tabular NN icin).
=============================================================================

    python src/tabular_mm.py     # artifacts/tabular_{train,test}.npy + tabular_cols.json yazar

NE: NN'in tabular dalina beslenecek HAM (olceksiz, imputesiz) sayisal matris + sutun adlari.
  Matris = sayisal(37) + yil(2, ham) + missing-flag(7) + one-hot kategorik(36) = 82 sutun.
  Sutun sirasi student_id sirasinda (oof_*.npy ile satir-hizali). Train ve test AYNI sutun uzayi.

NEDEN HAM (imputesiz/olceksiz): impute (fold-train medyani) + StandardScaler NN egitiminde
  PER-FOLD, yalniz dis-fold train'ine fit edilir (sizinti kacin; CLAUDE.md fold-safe). Bu yuzden
  builder GLOBAL hicbir istatistik (medyan/ortalama/std) HESAPLAMAZ; sadece HEDEF-BAGIMSIZ donusum
  yapar:
    * sayisal/yil  : ham deger, NaN KORUNUR (notebook fold-ici impute eder).
    * missing-flag : isna() -> 0/1 (hedefe bakmaz, satir-ici -> fold-safe; cv.NA_COLS ile birebir).
    * kategorik    : one-hot, kategori EVRENI train seviyelerinden (cv.structured_cat_dtypes;
                     test-only seviye YOK -> Faz 01). One-hot hedef-bagimsiz -> TARGET-ENCODING YOK.

FOLD-SAFE GARANTI: tek hedef-bagimli adim (impute/scale) notebook'ta fold-train'e fit edilir; bu
  builder yalniz one-hot + flag uretir (ikisi de hedef-bagimsiz, global hesaplanmasi sizinti DEGIL).
  cv.py'nin build_structured_matrix'i LightGBM icindir (native-kategorik, NaN-koruyan); NN one-hot +
  fold-ici impute ister -> bu ayri builder. Kolon rolleri cv.py'den (TEK kaynak).

Determinizm: sutun sirasi sorted-kategori + sabit kolon listesi -> PYTHONHASHSEED-bagimsiz.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cv

TAB_TRAIN_PATH = cv.ARTIFACTS_DIR / "tabular_train.npy"
TAB_TEST_PATH = cv.ARTIFACTS_DIR / "tabular_test.npy"
TAB_COLS_PATH = cv.ARTIFACTS_DIR / "tabular_cols.json"


def build_tabular_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """HAM fold-safe NN tabular matrisi. Doner: (X_train, X_test, col_names).

    X_* float32 ndarray (n, 82). NaN sayisal/yil sutunlarinda KORUNUR (notebook fold-ici impute eder).
    Train ve test AYNI sutun sirasi/uzayi (one-hot kategori evreni train'den, sabit).
    """
    num_cols = cv.numeric_feature_columns(train)          # 37, yil-disi sayisal
    year_cols = list(cv.YEAR_COLS)                         # 2, ham sayisal
    na_cols = list(cv.NA_COLS)                             # 7, missing-flag kaynagi
    cat_dtypes = cv.structured_cat_dtypes(train)           # train-seviye kategori evreni (fold-safe)

    def _one_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        parts = []
        # sayisal + yil (ham; NaN korunur)
        parts.append(df[num_cols].astype("float32").reset_index(drop=True))
        parts.append(df[year_cols].astype("float32").reset_index(drop=True))
        # missing-flag (hedef-bagimsiz; cv.NA_COLS ile birebir)
        fl = df[na_cols].isna().astype("float32")
        fl.columns = [f"{c}_missing" for c in na_cols]
        parts.append(fl.reset_index(drop=True))
        # one-hot kategorik (train kategori evreni -> sabit, test-only seviye yok)
        oh_blocks = []
        for c in cv.CATEGORICAL_COLS:
            s = df[c].astype(str).astype(cat_dtypes[c])
            d = pd.get_dummies(s, prefix=c, dtype="float32")
            # cat_dtypes ile TUM seviyeler garanti -> train/test ayni kolonlar (eksik seviye 0 kolon)
            expected = [f"{c}_{lvl}" for lvl in cat_dtypes[c].categories]
            d = d.reindex(columns=expected, fill_value=np.float32(0.0))
            oh_blocks.append(d.reset_index(drop=True))
        parts.extend(oh_blocks)
        out = pd.concat(parts, axis=1)
        return out, list(out.columns)

    Xtr_df, cols = _one_frame(train)
    Xte_df, cols_te = _one_frame(test)
    assert cols == cols_te, "train/test tabular kolonlari ayni degil (one-hot evren tutarsiz)."
    Xtr = Xtr_df.to_numpy(dtype=np.float32)
    Xte = Xte_df.to_numpy(dtype=np.float32)
    assert Xtr.shape[1] == Xte.shape[1] == len(cols)
    return Xtr, Xte, cols


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    Xtr, Xte, cols = build_tabular_matrix(train, test)

    cv.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TAB_TRAIN_PATH, Xtr)
    np.save(TAB_TEST_PATH, Xte)
    TAB_COLS_PATH.write_text(json.dumps(cols, ensure_ascii=False, indent=0), encoding="utf-8")

    n_num = len(cv.numeric_feature_columns(train))
    n_year = len(cv.YEAR_COLS)
    n_flag = len(cv.NA_COLS)
    n_oh = len(cols) - n_num - n_year - n_flag
    nan_cols = int(np.isnan(Xtr).any(axis=0).sum())
    print(f"[tabular_mm] train{Xtr.shape} test{Xte.shape} | "
          f"sayisal={n_num} yil={n_year} flag={n_flag} one-hot={n_oh} TOPLAM={len(cols)}")
    print(f"[tabular_mm] NaN iceren sutun sayisi (notebook fold-ici impute eder): {nan_cols}")
    print(f"[tabular_mm] YAZILDI: {TAB_TRAIN_PATH.name}, {TAB_TEST_PATH.name}, {TAB_COLS_PATH.name}")


if __name__ == "__main__":
    main()
