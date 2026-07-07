"""
submission_v2 — guclu model.

Metrik: MSE (DatathonNotes/overview.md). Hedef: career_success_score (0-100).

Pipeline:
  1. Feature engineering (EDA'dan): teknik/sosyal/portfolio ortalamalari,
     eksik-deger flag'leri, sayisal + kategorik (target/ordinal encoding).
  2. NLP: mentor_feedback_text -> TF-IDF (word 1-2gram + char 3-5gram)
     -> TruncatedSVD ile latent text feature'lari (agac modellerine uygun).
  3. Modeller: LightGBM / XGBoost / CatBoost / HistGB — 5-fold CV, OOF MSE.
  4. Ensemble: OOF uzerinde negatif-olmayan agirlikli ortalama (sirt).
  5. Tahminleri 0-100'e clip, submissions/submission_v2.csv yaz.

Calistir:  python src/train_model_v2.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "submissions" / "submission_v2.csv"

ID = "student_id"
TARGET = "career_success_score"
TEXT = "mentor_feedback_text"
SEED = 42
N_FOLDS = 5

CAT_COLS = ["department", "university_tier", "target_role", "hobby",
            "preferred_social_media_platform"]
TECH = ["coding_score", "problem_solving_score", "data_structures_score",
        "sql_score", "machine_learning_score", "backend_score",
        "frontend_score", "cloud_score", "devops_score"]
SOCIAL = ["communication_score", "teamwork_score", "leadership_score",
          "presentation_score"]
PORTFOLIO = ["portfolio_score", "github_repo_count", "github_avg_stars",
             "open_source_contribution_count", "linkedin_profile_score",
             "cv_quality_score"]
INTERVIEW = ["technical_interview_score", "hr_interview_score"]
# eksik degeri olan + flag faydali kolonlar (EDA bolum 2)
MISSING_FLAG = ["internship_duration_months", "english_exam_score",
                "github_avg_stars", "open_source_contribution_count",
                "hr_interview_score", "linkedin_profile_score",
                "portfolio_score"]

TR_STOP = ["ve", "bir", "bu", "ile", "için", "daha", "çok", "de", "da", "ama",
           "ancak", "olarak", "olan", "olduğu", "olabilir", "gibi", "kadar",
           "ise", "en", "hem", "ya", "veya", "konusunda", "sahip", "ki", "o"]


# ----------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["technical_avg"] = df[TECH].mean(axis=1)
    df["social_avg"] = df[SOCIAL].mean(axis=1)
    df["portfolio_avg"] = df[PORTFOLIO].mean(axis=1)
    df["interview_avg"] = df[INTERVIEW].mean(axis=1)
    df["tech_max"] = df[TECH].max(axis=1)
    df["tech_min"] = df[TECH].min(axis=1)
    df["tech_std"] = df[TECH].std(axis=1)
    # deneyim toplami
    df["total_experience"] = (
        df["internship_count"].fillna(0)
        + df["real_client_project_count"].fillna(0)
        + df["freelance_project_count"].fillna(0)
        + df["hackathon_count"].fillna(0)
    )
    df["feedback_len"] = df[TEXT].fillna("").str.len()
    for c in MISSING_FLAG:
        df[c + "_na"] = df[c].isnull().astype(int)
    df["n_missing"] = df[MISSING_FLAG].isnull().sum(axis=1)
    return df


def _vectorize(train_txt, test_txt):
    word = TfidfVectorizer(stop_words=TR_STOP, ngram_range=(1, 2),
                           min_df=5, max_features=20000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=5, max_features=20000, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(train_txt), char.fit_transform(train_txt)]).tocsr()
    Xte = hstack([word.transform(test_txt), char.transform(test_txt)]).tocsr()
    return Xtr, Xte


def text_svd(train_txt, test_txt, n_comp=40):
    """TF-IDF (word + char) -> SVD latent features. Fit yalnizca train'de."""
    Xtr, Xte = _vectorize(train_txt, test_txt)
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
    Str = svd.fit_transform(Xtr)
    Ste = svd.transform(Xte)
    print(f"  TF-IDF -> SVD: {Xtr.shape[1]} -> {n_comp} bilesen, "
          f"acikl. varyans={svd.explained_variance_ratio_.sum():.3f}")
    cols = [f"txt_svd_{i}" for i in range(n_comp)]
    return (pd.DataFrame(Str, columns=cols, index=train_txt.index),
            pd.DataFrame(Ste, columns=cols, index=test_txt.index))


def text_ridge_oof(train_txt, test_txt, y, alpha=8.0):
    """TF-IDF -> Ridge ile metinden hedef tahmini.
    OOF (sizintisiz): her satirin tahmini, o satiri gormemis Ridge'den gelir.
    Cikti: train icin OOF tahmin, test icin tum-train fit tahmini -> tek feature."""
    Xtr, Xte = _vectorize(train_txt, test_txt)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(Xtr.shape[0])
    for tr_idx, va_idx in kf.split(Xtr):
        r = Ridge(alpha=alpha, random_state=SEED)
        r.fit(Xtr[tr_idx], y[tr_idx])
        oof[va_idx] = r.predict(Xtr[va_idx])
    r_full = Ridge(alpha=alpha, random_state=SEED).fit(Xtr, y)
    te_pred = r_full.predict(Xte)
    mse = ((y - np.clip(oof, 0, 100)) ** 2).mean()
    print(f"  Ridge(metin) OOF MSE = {mse:.3f}  (tek basina metin sinyali)")
    return (pd.Series(oof, index=train_txt.index, name="txt_ridge"),
            pd.Series(te_pred, index=test_txt.index, name="txt_ridge"))


# Seed-averaging icin seed'ler (2 seed: hiz/varyans dengesi)
SEEDS = [42, 7]
# Sabit, makul iterasyon (OOF sizintisi olmasin diye early-stop YOK).
# Onceki sürüm (2000-2500 iter, 1 seed) zaten 76.9 verdi; burada
# kategorik native + seed-avg ile iyilesme bekleniyor, asiri agir degil.
N_JOBS = 4  # CPU oversubscription'i onle (3 model paralel calismasin diye)


def make_lgbm(seed):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_samples=30, random_state=seed,
        n_jobs=N_JOBS, verbose=-1)


def make_xgb(seed):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=1500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        min_child_weight=5, random_state=seed, n_jobs=N_JOBS,
        tree_method="hist")


def make_cat(seed):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, allow_writing_files=False,
        thread_count=N_JOBS)


def target_encode(tr_col, va_col, te_col, y, global_mean, smoothing=20):
    """Kategorik -> smoothed target mean (fold-icinde fit, sizinti yok)."""
    stats = pd.DataFrame({"x": tr_col.values, "y": y}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return (va_col.map(enc).fillna(global_mean).values,
            te_col.map(enc).fillna(global_mean).values)


def main():
    train = add_features(pd.read_csv(DATA / "train.csv"))
    test = add_features(pd.read_csv(DATA / "test_x.csv"))
    y = train[TARGET].values

    print("NLP feature'lari uretiliyor...")
    svd_tr, svd_te = text_svd(train[TEXT].fillna(""), test[TEXT].fillna(""))
    train = pd.concat([train, svd_tr], axis=1)
    test = pd.concat([test, svd_te], axis=1)
    # Ridge OOF metin tahmini (en guclu NLP sinyali)
    rdg_tr, rdg_te = text_ridge_oof(train[TEXT].fillna(""), test[TEXT].fillna(""), y)
    train["txt_ridge"] = rdg_tr.values
    test["txt_ridge"] = rdg_te.values

    # sayisal feature listesi (id/target/text/kategorik haric)
    drop = {ID, TARGET, TEXT, *CAT_COLS}
    num_cols = [c for c in train.columns
                if c not in drop and train[c].dtype != object]
    # CatBoost icin: sayisal + ham kategorik (native islenecek)
    cat_input_cols = num_cols + CAT_COLS
    print(f"Sayisal feature: {len(num_cols)} | LGBM/XGB target-encoded kategorik: "
          f"{len(CAT_COLS)} | CatBoost native kategorik: {len(CAT_COLS)}")
    print(f"Seed-averaging: {SEEDS}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gmean = y.mean()
    model_names = ["lgbm", "xgb", "cat"]
    oof = {m: np.zeros(len(train)) for m in model_names}
    test_pred = {m: np.zeros(len(test)) for m in model_names}

    # CatBoost kategorikleri string'e cevir (NaN'lari da)
    for c in CAT_COLS:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        # --- LGBM/XGB icin target-encoded matris ---
        Xtr_te, Xva_te, Xte_te = {}, {}, {}
        for c in CAT_COLS:
            va_enc, te_enc = target_encode(
                train.iloc[tr_idx][c], train.iloc[va_idx][c], test[c],
                y[tr_idx], gmean)
            tr_enc, _ = target_encode(
                train.iloc[tr_idx][c], train.iloc[tr_idx][c], test[c],
                y[tr_idx], gmean)
            Xtr_te[c + "_te"], Xva_te[c + "_te"], Xte_te[c + "_te"] = tr_enc, va_enc, te_enc
        Xtr = pd.concat([train.iloc[tr_idx][num_cols].reset_index(drop=True),
                         pd.DataFrame(Xtr_te)], axis=1)
        Xva = pd.concat([train.iloc[va_idx][num_cols].reset_index(drop=True),
                         pd.DataFrame(Xva_te)], axis=1)
        Xte = pd.concat([test[num_cols].reset_index(drop=True),
                         pd.DataFrame(Xte_te)], axis=1)

        # --- CatBoost icin ham (native kategorik) matris ---
        Ctr = train.iloc[tr_idx][cat_input_cols].reset_index(drop=True)
        Cva = train.iloc[va_idx][cat_input_cols].reset_index(drop=True)
        Cte = test[cat_input_cols].reset_index(drop=True)

        ns = len(SEEDS)
        for seed in SEEDS:
            # LGBM
            m = make_lgbm(seed); m.fit(Xtr, y[tr_idx])
            oof["lgbm"][va_idx] += m.predict(Xva) / ns
            test_pred["lgbm"] += m.predict(Xte) / (ns * N_FOLDS)
            # XGB
            m = make_xgb(seed); m.fit(Xtr, y[tr_idx])
            oof["xgb"][va_idx] += m.predict(Xva) / ns
            test_pred["xgb"] += m.predict(Xte) / (ns * N_FOLDS)
            # CatBoost (native kategorik)
            m = make_cat(seed); m.fit(Ctr, y[tr_idx], cat_features=CAT_COLS)
            oof["cat"][va_idx] += m.predict(Cva) / ns
            test_pred["cat"] += m.predict(Cte) / (ns * N_FOLDS)
        print(f"  fold {fold} bitti")

    print("\n=== TEKIL MODEL OOF MSE (seed-averaged) ===")
    for name in model_names:
        p = np.clip(oof[name], 0, 100)
        print(f"  {name:5s}: MSE = {((y - p) ** 2).mean():.4f}  "
              f"(RMSE {np.sqrt(((y - p) ** 2).mean()):.4f})")
    models = model_names

    # --- Ensemble: NNLS ile negatif-olmayan agirliklar ---
    M = np.column_stack([oof[m] for m in models])
    w, _ = nnls(M, y)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(models)) / len(models)
    ens_oof = np.clip(M @ w, 0, 100)
    ens_mse = ((y - ens_oof) ** 2).mean()
    print("\n=== ENSEMBLE ===")
    for name, wi in zip(models, w):
        print(f"  {name:5s} agirlik = {wi:.3f}")
    print(f"  ENSEMBLE OOF MSE = {ens_mse:.4f}  (RMSE {np.sqrt(ens_mse):.4f})")

    # --- Test tahmini ---
    Mte = np.column_stack([test_pred[m] for m in models])
    final = np.clip(Mte @ w, 0, 100).round(3)

    sub = pd.DataFrame({ID: test[ID], TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"\nYAZILDI -> {OUT}")
    print(f"satir: {len(sub)} | aralik: {final.min()}-{final.max()} | "
          f"ort: {final.mean():.2f}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
