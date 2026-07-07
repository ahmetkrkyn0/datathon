"""
submission_v3 — zaman kaymasi duzeltmeli + embedding'li model.

v2'den farklar (LB 87.12 -> hedef 83-84):
  1. YIL AGIRLIKLAMA: test'in %62'si 2024-2026 basvurusu; train uniform.
     w(yil) = P_test(yil)/P_train(yil) ile egitim sample_weight + LB-proxy
     agirlikli OOF raporu. (CV-LB farkinin ana sebebi bu kaymaydi.)
  2. EMBEDDINGS: mentor metni -> paraphrase-multilingual-MiniLM-L12-v2
     (384d, cache'li) -> SVD 50 bilesen + Ridge OOF tahmini.
  3. YENI FEATURE'LAR: role_skill_match (hedef role uygun teknik skor),
     proj_quality x tech_interview etkilesimleri, mezuniyet yili farki.
  4. ISOTONIC KALIBRASYON: 773 kayit tam 100'de (tavan). OOF uzerinde
     CV ile dogrulanir, iyilestiriyorsa uygulanir.

Calistir: python -u src/train_model_v3.py
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
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
OUT = ROOT / "submissions" / "submission_v3.csv"

ID = "student_id"
TARGET = "career_success_score"
TEXT = "mentor_feedback_text"
SEED = 42
N_FOLDS = 5
SEEDS = [42, 7]
N_JOBS = 4
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
MISSING_FLAG = ["internship_duration_months", "english_exam_score",
                "github_avg_stars", "open_source_contribution_count",
                "hr_interview_score", "linkedin_profile_score",
                "portfolio_score"]
# hedef rol -> en alakali teknik skor
ROLE_SKILL = {
    "Frontend Developer": "frontend_score",
    "Backend Developer": "backend_score",
    "DevOps Engineer": "devops_score",
    "Cloud Engineer": "cloud_score",
    "Data Scientist": "machine_learning_score",
    "AI Engineer": "machine_learning_score",
    "MLOps Engineer": "machine_learning_score",
    "Data Analyst": "sql_score",
    "Product Analyst": "sql_score",
    "Software Developer": "coding_score",
    "Cybersecurity Analyst": "devops_score",
}
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
    # --- v3 yenileri ---
    df["role_skill"] = [
        row[ROLE_SKILL[r]] if r in ROLE_SKILL else np.nan
        for r, (_, row) in zip(df["target_role"], df.iterrows())
    ]
    df["role_gap"] = df["role_skill"] - df["technical_avg"]
    df["pq_x_ti"] = df["project_quality_score"] * df["technical_interview_score"]
    df["pq_x_tech"] = df["project_quality_score"] * df["technical_avg"]
    df["ti_x_tech"] = df["technical_interview_score"] * df["technical_avg"]
    df["years_since_grad"] = df["application_year"] - df["graduation_year"]
    return df


def year_weights(train_years, test_years, cap=(0.3, 2.5)):
    """Importance weighting: w = P_test(yil) / P_train(yil), kirpilmis."""
    p_tr = train_years.value_counts(normalize=True)
    p_te = test_years.value_counts(normalize=True)
    w = train_years.map(p_te / p_tr).fillna(1.0).clip(*cap).values
    return w * len(w) / w.sum()  # ortalama 1'e normalle


def get_embeddings(train_txt, test_txt):
    """MiniLM embeddings — diske cache'lenir (ilk calistirmada uretilir)."""
    CACHE.mkdir(exist_ok=True)
    f_tr, f_te = CACHE / "emb_train.npy", CACHE / "emb_test.npy"
    if f_tr.exists() and f_te.exists():
        print("  embeddings cache'ten okundu")
        return np.load(f_tr), np.load(f_te)
    from sentence_transformers import SentenceTransformer
    print(f"  embedding modeli yukleniyor: {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    emb_tr = model.encode(train_txt.tolist(), batch_size=64,
                          show_progress_bar=False, normalize_embeddings=True)
    emb_te = model.encode(test_txt.tolist(), batch_size=64,
                          show_progress_bar=False, normalize_embeddings=True)
    np.save(f_tr, emb_tr)
    np.save(f_te, emb_te)
    print(f"  embeddings uretildi ve cache'lendi: {emb_tr.shape}")
    return emb_tr, emb_te


def _vectorize(train_txt, test_txt):
    word = TfidfVectorizer(stop_words=TR_STOP, ngram_range=(1, 2),
                           min_df=5, max_features=20000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=5, max_features=20000, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(train_txt), char.fit_transform(train_txt)]).tocsr()
    Xte = hstack([word.transform(test_txt), char.transform(test_txt)]).tocsr()
    return Xtr, Xte


def ridge_oof(Xtr, Xte, y, alpha, name):
    """Genel OOF Ridge: train icin sizintisiz tahmin, test icin full-fit."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(Xtr.shape[0])
    for tr_idx, va_idx in kf.split(np.arange(Xtr.shape[0])):
        r = Ridge(alpha=alpha, random_state=SEED)
        r.fit(Xtr[tr_idx], y[tr_idx])
        oof[va_idx] = r.predict(Xtr[va_idx])
    te = Ridge(alpha=alpha, random_state=SEED).fit(Xtr, y).predict(Xte)
    mse = ((y - np.clip(oof, 0, 100)) ** 2).mean()
    print(f"  {name} OOF MSE = {mse:.2f}")
    return oof, te


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
        iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        random_seed=seed, verbose=0, allow_writing_files=False,
        thread_count=N_JOBS)


def target_encode(tr_col, va_col, te_col, y, global_mean, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return (va_col.map(enc).fillna(global_mean).values,
            te_col.map(enc).fillna(global_mean).values)


def wmse(y, p, w):
    return float(np.average((y - p) ** 2, weights=w))


def main():
    train = add_features(pd.read_csv(DATA / "train.csv"))
    test = add_features(pd.read_csv(DATA / "test_x.csv"))
    y = train[TARGET].values

    # --- yil agirliklari ---
    w_fit = year_weights(train["application_year"], test["application_year"])
    print(f"Yil agirliklari: min={w_fit.min():.2f} max={w_fit.max():.2f} "
          f"(egitim + LB-proxy degerlendirme icin)")

    # --- NLP: TF-IDF + embeddings ---
    print("NLP feature'lari...")
    tr_txt, te_txt = train[TEXT].fillna(""), test[TEXT].fillna("")
    Xtf_tr, Xtf_te = _vectorize(tr_txt, te_txt)
    svd = TruncatedSVD(n_components=40, random_state=SEED)
    svd_tr, svd_te = svd.fit_transform(Xtf_tr), svd.transform(Xtf_te)
    for i in range(40):
        train[f"txt_svd_{i}"], test[f"txt_svd_{i}"] = svd_tr[:, i], svd_te[:, i]
    oof_tfidf, te_tfidf = ridge_oof(Xtf_tr, Xtf_te, y, 8.0, "Ridge(TF-IDF)")
    train["txt_ridge"], test["txt_ridge"] = oof_tfidf, te_tfidf

    emb_tr, emb_te = get_embeddings(tr_txt, te_txt)
    esvd = TruncatedSVD(n_components=50, random_state=SEED)
    e_tr, e_te = esvd.fit_transform(emb_tr), esvd.transform(emb_te)
    for i in range(50):
        train[f"emb_svd_{i}"], test[f"emb_svd_{i}"] = e_tr[:, i], e_te[:, i]
    oof_emb, te_emb = ridge_oof(emb_tr, emb_te, y, 1.0, "Ridge(embedding)")
    train["txt_emb_ridge"], test["txt_emb_ridge"] = oof_emb, te_emb

    # --- feature listeleri ---
    drop = {ID, TARGET, TEXT, *CAT_COLS}
    num_cols = [c for c in train.columns
                if c not in drop and train[c].dtype != object]
    cat_input_cols = num_cols + CAT_COLS
    print(f"Sayisal: {len(num_cols)} | kategorik: {len(CAT_COLS)} | seeds: {SEEDS}")

    for c in CAT_COLS:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    gmean = y.mean()
    names = ["lgbm", "xgb", "cat"]
    oof = {m: np.zeros(len(train)) for m in names}
    test_pred = {m: np.zeros(len(test)) for m in names}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
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
        Ctr = train.iloc[tr_idx][cat_input_cols].reset_index(drop=True)
        Cva = train.iloc[va_idx][cat_input_cols].reset_index(drop=True)
        Cte = test[cat_input_cols].reset_index(drop=True)
        wf = w_fit[tr_idx]

        ns = len(SEEDS)
        for seed in SEEDS:
            m = make_lgbm(seed); m.fit(Xtr, y[tr_idx], sample_weight=wf)
            oof["lgbm"][va_idx] += m.predict(Xva) / ns
            test_pred["lgbm"] += m.predict(Xte) / (ns * N_FOLDS)
            m = make_xgb(seed); m.fit(Xtr, y[tr_idx], sample_weight=wf)
            oof["xgb"][va_idx] += m.predict(Xva) / ns
            test_pred["xgb"] += m.predict(Xte) / (ns * N_FOLDS)
            m = make_cat(seed)
            m.fit(Ctr, y[tr_idx], cat_features=CAT_COLS, sample_weight=wf)
            oof["cat"][va_idx] += m.predict(Cva) / ns
            test_pred["cat"] += m.predict(Cte) / (ns * N_FOLDS)
        print(f"  fold {fold} bitti")

    print("\n=== TEKIL OOF (duz MSE | yil-agirlikli MSE ~ LB proxy) ===")
    for n in names:
        p = np.clip(oof[n], 0, 100)
        print(f"  {n:5s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    # --- NNLS blend (agirlikli uzayda) ---
    M = np.column_stack([oof[m] for m in names])
    sw = np.sqrt(w_fit)
    w_blend, _ = nnls(M * sw[:, None], y * sw)
    w_blend = w_blend / w_blend.sum()
    ens = np.clip(M @ w_blend, 0, 100)
    print("\n=== ENSEMBLE ===")
    for n, wi in zip(names, w_blend):
        print(f"  {n:5s} agirlik = {wi:.3f}")
    print(f"  duz MSE = {((y - ens) ** 2).mean():.4f} | "
          f"agirlikli (LB proxy) = {wmse(y, ens, w_fit):.4f}")

    # --- Isotonic kalibrasyon (CV ile durust dogrulama) ---
    iso_oof = np.zeros_like(ens)
    for tr_idx, va_idx in kf.split(ens):
        iso = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
        iso.fit(ens[tr_idx], y[tr_idx], sample_weight=w_fit[tr_idx])
        iso_oof[va_idx] = iso.predict(ens[va_idx])
    mse_before = wmse(y, ens, w_fit)
    mse_after = wmse(y, iso_oof, w_fit)
    print(f"\nIsotonic: agirlikli MSE {mse_before:.4f} -> {mse_after:.4f} "
          f"({'UYGULANIYOR' if mse_after < mse_before else 'atlandi'})")

    Mte = np.column_stack([test_pred[m] for m in names])
    final = np.clip(Mte @ w_blend, 0, 100)
    if mse_after < mse_before:
        iso = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
        iso.fit(ens, y, sample_weight=w_fit)
        final = iso.predict(final)
    final = np.clip(final, 0, 100).round(3)

    sub = pd.DataFrame({ID: test[ID], TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"\nYAZILDI -> {OUT}")
    print(f"satir: {len(sub)} | aralik: {final.min()}-{final.max()} | ort: {final.mean():.2f}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
