"""
Faz 3 — Fold-safe on isleme boru hatti (SPEC 03 §3,§5,§7). build_preprocessor: TUM `fit`
islemleri (medyan impute, OHE kategori ogrenme) YALNIZCA dis-fold train'inden yapilan,
deterministik, sizintisiz bir `ColumnTransformer` uretir.

    python src/preprocessing.py     # 1 smoke test + data/column_spec.json yazimi

DALLAR (SPEC 03 §7):
  * num_passthrough : NA'siz sayisal + YIL (ham sayisal) -> oldugu gibi (istatistik yok).
  * median_impute   : 6 kolon -> SimpleImputer(median), FOLD-ICI fit (leakageRules madde 4).
  * zero_impute     : YALNIZCA internship_duration_months -> constant 0 (MNAR; %82.14 count==0).
  * missing_flags   : 7 NA kolonu icin <col>_missing (isna, hedef-bagimsiz, fold-bagimsiz).
  * categorical     : model_family'ye gore native passthrough (CatBoost/HistGBR/LGBM-native)
                      VEYA OneHotEncoder(handle_unknown='ignore') (LGBM OHE ablasyonu).

MNAR AYRIMI (SPEC 03 §3, kanita dayali):
  * internship_duration_months -> 0+flag (NA'lerin %82.14'u internship_count==0 ile cakisir).
  * open_source_contribution_count -> MEDYAN+flag (github_avg_stars ile ayni 910 satirda NA;
    bu satirlarin %96.81'i aktif repo'ya sahip -> veri-toplama boslugu, yapisal sifir DEGIL).

Determinizm: SEED=42; kolon sirasi sabit; verbose_feature_names_out=False ile cikti adlari
ham/`<col>_missing` -> satir VE kolon hizasi (data/column_spec.json'a yazilir).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import cv

# --------------------------------------------------------------------------- #
# NA doldurma kararlari (SPEC 03 §3; union == cv.NA_COLS, kanita dayali).
# --------------------------------------------------------------------------- #
# Veri-toplama boslugu -> fold-ici medyan + flag (open_source DAHIL, review duzeltmesi).
MEDIAN_IMPUTE_COLS = (
    "english_exam_score",
    "github_avg_stars",
    "open_source_contribution_count",  # github_avg_stars ile ayni maske -> ayni strateji
    "hr_interview_score",
    "linkedin_profile_score",
    "portfolio_score",
)
# Yapisal sifir (MNAR) -> sabit 0 + flag (yalnizca internship; %82.14 count==0 kaniti).
ZERO_IMPUTE_COLS = ("internship_duration_months",)

# Sozlesme sigortasi: doldurma evreni tam olarak 7 NA kolonu (ne eksik ne fazla).
assert set(MEDIAN_IMPUTE_COLS) | set(ZERO_IMPUTE_COLS) == set(cv.NA_COLS), (
    "MEDIAN_IMPUTE_COLS + ZERO_IMPUTE_COLS, cv.NA_COLS ile birebir esit olmali."
)
assert not (set(MEDIAN_IMPUTE_COLS) & set(ZERO_IMPUTE_COLS)), "Bir kolon hem median hem zero olamaz."

COLUMN_SPEC_PATH = cv.DATA_DIR / "column_spec.json"
SPEC_VERSION = 1


# --------------------------------------------------------------------------- #
# Fold-bagimsiz NA bayrak ureticisi (SPEC 03 §3 / Guardrail 3).
# Sadece isna() -> hedefe bakmaz; fit'te istatistik ogrenmez -> sizinti imkansiz.
# --------------------------------------------------------------------------- #
class MissingFlagger(BaseEstimator, TransformerMixin):
    """Aldigi her kolon icin `<col>_missing` (int8) bayragi uretir. Hedef-bagimsiz, fold-bagimsiz."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else [
            f"x{i}" for i in range(np.asarray(X).shape[1])
        ]
        return self

    def transform(self, X):
        df = X if hasattr(X, "columns") else pd.DataFrame(np.asarray(X), columns=self.columns_)
        out = df.isna().astype("int8")
        out.columns = [f"{c}_missing" for c in df.columns]
        return out

    def get_feature_names_out(self, input_features=None):
        cols = self.columns_ if input_features is None else list(input_features)
        return np.asarray([f"{c}_missing" for c in cols], dtype=object)


# --------------------------------------------------------------------------- #
# Boru hatti montaji (SPEC 03 §7).
# --------------------------------------------------------------------------- #
def build_preprocessor(model_family: str = "native", feature_columns=None) -> ColumnTransformer:
    """Fold-safe `ColumnTransformer` doner. `fit_transform`'u DAIMA dis-fold train'inde cagir.

    Parametreler
    ------------
    model_family : {"native", "onehot"}
        "native" -> kategorikler passthrough (`category` dtype korunur; CatBoost/HistGBR/LGBM
        native split'leri kesfeder). "onehot" -> OneHotEncoder(handle_unknown='ignore') (LGBM
        OHE ablasyonu). Her ikisi de fold-ici fit.
    feature_columns : Iterable[str]
        X'in kolonlari (clean_raw sonrasi, target/text HARIC). Dal kolon listeleri buradan
        turetilir -> df'e bagimli degil, deterministik.

    Cikti kolonlari: ham sayisal/impute adlar + `<col>_missing` (+ OHE'de `<col>_<deger>`).
    """
    if feature_columns is None:
        raise ValueError("feature_columns zorunlu (X.columns; target/text disi).")
    feat = list(feature_columns)
    if model_family not in ("native", "onehot"):
        raise ValueError(f"model_family 'native' veya 'onehot' olmali, '{model_family}' degil.")

    cat_cols = [c for c in feat if c in cv.CATEGORICAL_COLS]
    median_cols = [c for c in MEDIAN_IMPUTE_COLS if c in feat]
    zero_cols = [c for c in ZERO_IMPUTE_COLS if c in feat]
    na_cols = [c for c in cv.NA_COLS if c in feat]  # 7 flag kaynagi (sabit sira: cv.NA_COLS)
    # NA'siz sayisal + YIL (ham): kategorik ve NA kolonlar disindaki her sey -> passthrough.
    num_passthrough = [c for c in feat if c not in cv.CATEGORICAL_COLS and c not in cv.NA_COLS]

    transformers = [
        ("num_passthrough", "passthrough", num_passthrough),
        ("median_impute", SimpleImputer(strategy="median"), median_cols),
        ("zero_impute", SimpleImputer(strategy="constant", fill_value=0.0), zero_cols),
        ("missing_flags", MissingFlagger(), na_cols),
    ]
    if model_family == "onehot":
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    else:  # native: kategorikler oldugu gibi (category dtype korunur)
        transformers.append(("categorical", "passthrough", cat_cols))

    ct = ColumnTransformer(
        transformers,
        remainder="drop",  # text/target dahil listelenmeyen her sey duser (sizinti-sigortasi)
        verbose_feature_names_out=False,  # ham adlar; cakisma yok (impute/passthrough/flag disjoint)
    )
    ct.set_output(transform="pandas")
    return ct


# --------------------------------------------------------------------------- #
# Kolon manifesti (SPEC 03 deliverable: data/column_spec.json) — turetilmis, versiyonlu.
# --------------------------------------------------------------------------- #
def column_spec(raw_df: pd.DataFrame) -> dict:
    """Ham df'ten turetilmis (hardcode degil) kolon manifesti. clean_raw kurallariyla hizali."""
    from cleaning import clean_raw

    clean = clean_raw(raw_df)
    feat = [c for c in clean.columns if c not in (cv.TARGET_COL, cv.TEXT_COL)]
    cats = [c for c in feat if c in cv.CATEGORICAL_COLS]
    numeric = [c for c in feat if c not in cats]  # YIL + NA kolonlar + NA'siz sayisal
    na_strategy = {
        c: ("zero_flag" if c in ZERO_IMPUTE_COLS else "median_flag") for c in cv.NA_COLS
    }
    return {
        "version": SPEC_VERSION,
        "seed": cv.SEED,
        "drop": [cv.ID_COL],
        "target": cv.TARGET_COL,
        "text": cv.TEXT_COL,
        "year_raw_numeric": list(cv.YEAR_COLS),
        "categorical": cats,
        "numeric": numeric,
        "na_columns": na_strategy,
        "median_impute": list(MEDIAN_IMPUTE_COLS),
        "zero_impute": list(ZERO_IMPUTE_COLS),
        "missing_flags": [f"{c}_missing" for c in cv.NA_COLS],
        "clip": [cv.CLIP_LO, cv.CLIP_HI],
    }


def write_column_spec(raw_df: pd.DataFrame, path=COLUMN_SPEC_PATH) -> dict:
    spec = column_spec(raw_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return spec


# --------------------------------------------------------------------------- #
# Smoke test (LEAN: tek kosu) — bir dis fold'da end-to-end + DoD ana iddialarini dogrula.
#   * fit YALNIZCA fold-train'de; valid/test transform.
#   * NaN kalmaz; student_id yok; yillar var; 7 _missing flag var.
#   * internship 0 ile, open_source MEDYAN ile (0 degil) doldurulur; maske == github_avg_stars.
#   * determinizm: iki kosu bit-ayni.
#   * data/column_spec.json yazilir.
# --------------------------------------------------------------------------- #
def main() -> None:
    from cleaning import clean_raw

    cv.set_seed()

    raw_train = cv.load_train()
    sid = raw_train[cv.ID_COL].values
    train = clean_raw(raw_train)
    test = clean_raw(cv.load_test())
    folds = cv.load_folds()

    y = train[cv.TARGET_COL].values
    X = train.drop(columns=[cv.TARGET_COL, cv.TEXT_COL])
    X_test = test.drop(columns=[cv.TEXT_COL])
    assert list(X.columns) == list(X_test.columns), "train/test feature kolonlari hizasiz."

    # Bir dis fold (repeat=0, fold=0) — fit yalniz fold-train'de.
    fold_of = cv.fold_of_rows(folds, sid, repeat=0)
    val_idx = np.where(fold_of == 0)[0]
    tr_idx = np.where(fold_of != 0)[0]
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]

    pre = build_preprocessor("native", feature_columns=X.columns)
    Xt_tr = pre.fit_transform(X_tr)          # FIT: yalniz dis-fold train
    Xt_val = pre.transform(X_val)
    Xt_test = pre.transform(X_test)

    # --- 1) NaN birakmaz (metin kolonu zaten X'te degil) ---
    for name, M in (("train", Xt_tr), ("valid", Xt_val), ("test", Xt_test)):
        assert int(M.isna().sum().sum()) == 0, f"{name}: impute sonrasi NaN var."

    # --- 2) student_id yok; yillar HAM SAYISAL var; 7 _missing flag var ---
    assert cv.ID_COL not in Xt_tr.columns, "student_id cikti matrisinde olmamali."
    for yc in cv.YEAR_COLS:
        assert yc in Xt_tr.columns, f"{yc} (ham sayisal) cikti matrisinde olmali."
    flags = [f"{c}_missing" for c in cv.NA_COLS]
    assert all(f in Xt_tr.columns for f in flags), "7 _missing bayraginin tamami olmali."
    assert len(flags) == 7

    # --- 3) MNAR ayrimi: internship 0 ile; open_source MEDYAN ile (0 degil) ---
    med_stats = dict(zip(MEDIAN_IMPUTE_COLS, pre.named_transformers_["median_impute"].statistics_))
    osc_median = med_stats["open_source_contribution_count"]
    assert osc_median != 0.0, "open_source medyani 0 olmamali (medyan+flag stratejisi)."

    na_int = X_val["internship_duration_months"].isna().to_numpy()
    if na_int.any():
        filled_int = Xt_val["internship_duration_months"].to_numpy()[na_int]
        assert np.allclose(filled_int, 0.0), "internship NA satirlari 0 ile dolmali (zero_flag)."

    na_osc = X_val["open_source_contribution_count"].isna().to_numpy()
    if na_osc.any():
        filled_osc = Xt_val["open_source_contribution_count"].to_numpy()[na_osc]
        assert np.allclose(filled_osc, osc_median), "open_source NA satirlari fold-medyani ile dolmali."
        assert not np.allclose(filled_osc, 0.0), "open_source 0 ile DOLMAMALI (median_flag)."

    # --- 4) open_source _missing maskesi == github_avg_stars _missing (bayt-bayt ayni) ---
    assert (X["open_source_contribution_count"].isna().to_numpy()
            == X["github_avg_stars"].isna().to_numpy()).all(), "ortak NA maskesi bozuk (ham veri)."
    assert (Xt_val["open_source_contribution_count_missing"].to_numpy()
            == Xt_val["github_avg_stars_missing"].to_numpy()).all(), "uretilen _missing maskeleri esit degil."

    # --- 5) Determinizm: iki ardisik fit_transform bit-ayni ---
    Xt_tr2 = build_preprocessor("native", feature_columns=X.columns).fit_transform(X_tr)
    pd.testing.assert_frame_equal(Xt_tr, Xt_tr2)

    # --- 6) OHE dali da calisir (NaN birakmaz, sayisal genisler) ---
    pre_ohe = build_preprocessor("onehot", feature_columns=X.columns)
    Xo_tr = pre_ohe.fit_transform(X_tr)
    Xo_val = pre_ohe.transform(X_val)
    assert int(Xo_tr.isna().sum().sum()) == 0 and int(Xo_val.isna().sum().sum()) == 0
    assert Xo_tr.shape[1] > Xt_tr.shape[1], "OHE native'den daha cok kolon uretmeli."

    # --- 7) Manifest yazimi (deliverable) ---
    spec = write_column_spec(raw_train)

    print(f"[faz3] native feature: {Xt_tr.shape[1]}  (train {Xt_tr.shape}, valid {Xt_val.shape}, test {Xt_test.shape})")
    print(f"[faz3] onehot feature: {Xo_tr.shape[1]}")
    print(f"[faz3] internship -> 0+flag (MNAR);  open_source -> medyan={osc_median:.3f}+flag (median_flag)")
    print(f"[faz3] 7 _missing flag OK;  open_source/github_avg_stars maske esit OK")
    print(f"[faz3] determinizm OK (iki fit_transform bit-ayni)")
    print(f"[faz3] column_spec.json yazildi -> {COLUMN_SPEC_PATH}  "
          f"(numeric={len(spec['numeric'])}, categorical={len(spec['categorical'])}, "
          f"flags={len(spec['missing_flags'])})")
    print(f"[faz3] SMOKE GECTI.")


if __name__ == "__main__":
    main()
