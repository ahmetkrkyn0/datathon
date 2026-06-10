"""Ortak feature pipeline — tune ve final egitim scriptleri bunu kullanir.

build_features(): train/test DataFrame'leri, y, yil agirliklari ve kolon
listelerini dondurur. Sonuc data/cache/'e pickle'lanir; sonraki cagrilar
cache'ten okur (NLP/embedding tekrar hesaplanmaz).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"

ID = "student_id"
TARGET = "career_success_score"
TEXT = "mentor_feedback_text"
SEED = 42
N_FOLDS = 5
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
# guclu duygu/yargi kelimeleri -> dogrudan binary bayrak (EDA'da ayirici cikti)
KEYWORDS = ["mükemmel", "olağanüstü", "etkileyici", "başarı", "güçlü",
            "yüksek", "umut verici", "potansiyel", "gelişmeye açık",
            "fazla pratik", "gelişim", "eksik", "yetersiz", "zayıf",
            "dikkat çekici", "üstün"]


def _add_tabular(df):
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
    skill_col = df["target_role"].map(ROLE_SKILL)
    df["role_skill"] = np.array(
        [df.iloc[i][sc] if isinstance(sc, str) else np.nan
         for i, sc in enumerate(skill_col)])
    df["role_gap"] = df["role_skill"] - df["technical_avg"]
    df["pq_x_ti"] = df["project_quality_score"] * df["technical_interview_score"]
    df["pq_x_tech"] = df["project_quality_score"] * df["technical_avg"]
    df["ti_x_tech"] = df["technical_interview_score"] * df["technical_avg"]
    df["years_since_grad"] = df["application_year"] - df["graduation_year"]
    txt_low = df[TEXT].fillna("").str.lower()
    for kw in KEYWORDS:
        df[f"kw_{kw.replace(' ', '_')}"] = txt_low.str.contains(kw, regex=False).astype(int)
    return df


def year_weights(train_years, test_years, cap=(0.3, 2.5)):
    p_tr = train_years.value_counts(normalize=True)
    p_te = test_years.value_counts(normalize=True)
    w = train_years.map(p_te / p_tr).fillna(1.0).clip(*cap).values
    return w * len(w) / w.sum()


def _vectorize(train_txt, test_txt):
    word = TfidfVectorizer(stop_words=TR_STOP, ngram_range=(1, 2),
                           min_df=5, max_features=20000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=5, max_features=20000, sublinear_tf=True)
    Xtr = hstack([word.fit_transform(train_txt), char.fit_transform(train_txt)]).tocsr()
    Xte = hstack([word.transform(test_txt), char.transform(test_txt)]).tocsr()
    return Xtr, Xte


def _ridge_oof(Xtr, Xte, y, alpha):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(Xtr.shape[0])
    for tr_idx, va_idx in kf.split(np.arange(Xtr.shape[0])):
        r = Ridge(alpha=alpha, random_state=SEED)
        r.fit(Xtr[tr_idx], y[tr_idx])
        oof[va_idx] = r.predict(Xtr[va_idx])
    te = Ridge(alpha=alpha, random_state=SEED).fit(Xtr, y).predict(Xte)
    return oof, te


def _embeddings(train_txt, test_txt):
    CACHE.mkdir(exist_ok=True)
    f_tr, f_te = CACHE / "emb_train.npy", CACHE / "emb_test.npy"
    if f_tr.exists() and f_te.exists():
        return np.load(f_tr), np.load(f_te)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    emb_tr = model.encode(train_txt.tolist(), batch_size=64,
                          normalize_embeddings=True)
    emb_te = model.encode(test_txt.tolist(), batch_size=64,
                          normalize_embeddings=True)
    np.save(f_tr, emb_tr)
    np.save(f_te, emb_te)
    return emb_tr, emb_te


def build_features(force=False):
    """Tam feature seti. Cache varsa direkt okur (force=True yeniden uretir)."""
    CACHE.mkdir(exist_ok=True)
    f_tr, f_te = CACHE / "feat_train.pkl", CACHE / "feat_test.pkl"
    if f_tr.exists() and f_te.exists() and not force:
        train, test = pd.read_pickle(f_tr), pd.read_pickle(f_te)
    else:
        train = _add_tabular(pd.read_csv(DATA / "train.csv"))
        test = _add_tabular(pd.read_csv(DATA / "test_x.csv"))
        y = train[TARGET].values
        tr_txt, te_txt = train[TEXT].fillna(""), test[TEXT].fillna("")

        Xtf_tr, Xtf_te = _vectorize(tr_txt, te_txt)
        svd = TruncatedSVD(n_components=40, random_state=SEED)
        s_tr, s_te = svd.fit_transform(Xtf_tr), svd.transform(Xtf_te)
        for i in range(40):
            train[f"txt_svd_{i}"], test[f"txt_svd_{i}"] = s_tr[:, i], s_te[:, i]
        o, t = _ridge_oof(Xtf_tr, Xtf_te, y, 8.0)
        train["txt_ridge"], test["txt_ridge"] = o, t

        emb_tr, emb_te = _embeddings(tr_txt, te_txt)
        esvd = TruncatedSVD(n_components=50, random_state=SEED)
        e_tr, e_te = esvd.fit_transform(emb_tr), esvd.transform(emb_te)
        for i in range(50):
            train[f"emb_svd_{i}"], test[f"emb_svd_{i}"] = e_tr[:, i], e_te[:, i]
        o, t = _ridge_oof(emb_tr, emb_te, y, 1.0)
        train["txt_emb_ridge"], test["txt_emb_ridge"] = o, t

        for c in CAT_COLS:
            train[c] = train[c].astype(str)
            test[c] = test[c].astype(str)
        train.to_pickle(f_tr)
        test.to_pickle(f_te)

    y = train[TARGET].values
    w_fit = year_weights(train["application_year"], test["application_year"])
    drop = {ID, TARGET, TEXT, *CAT_COLS}
    num_cols = [c for c in train.columns
                if c not in drop and train[c].dtype != object and c != TARGET]
    return train, test, y, w_fit, num_cols
