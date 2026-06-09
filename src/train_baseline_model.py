"""
Baseline model (gerçek tahmin) -> submission.csv

01_data_understanding_eda.ipynb'deki EDA bulgularını izler:
  - Hedef: career_success_score (0-100, regresyon)
  - Türetilen feature'lar: technical_avg / social_avg / portfolio_online_avg
    (notebook hücre 44, 46, 48) + mentor_feedback_length (hücre 29)
  - Eksik değerli kolonlar için _is_missing bayrakları (hücre 37)
  - Kategorikler: department, university_tier, target_role, hobby,
    preferred_social_media_platform  (one-hot)

Model: sklearn HistGradientBoostingRegressor
  (LightGBM kurulu olmadığı için; aynı gradient-boosting ailesi,
   eksik değerleri yerinde işler, ek kurulum gerektirmez)

Değerlendirme: 5-fold CV ile RMSE raporlanır, sonra tüm train ile
yeniden eğitilip test tahmini submission.csv'ye yazılır.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = ROOT / "submission.csv"

ID_COL = "student_id"
TARGET = "career_success_score"
SEED = 42

CATEGORICAL_COLS = [
    "department",
    "university_tier",
    "target_role",
    "hobby",
    "preferred_social_media_platform",
]

TECHNICAL_SKILL_COLS = [
    "coding_score", "problem_solving_score", "data_structures_score",
    "sql_score", "machine_learning_score", "backend_score",
    "frontend_score", "cloud_score", "devops_score",
]
SOCIAL_SKILL_COLS = [
    "communication_score", "teamwork_score", "leadership_score",
    "presentation_score",
]
PORTFOLIO_COLS = [
    "portfolio_score", "github_repo_count", "github_avg_stars",
    "open_source_contribution_count", "linkedin_profile_score",
    "cv_quality_score",
]
# Notebook hücre 37: eksik değerli ve hedefle ilişkili kolonlar
MISSING_FLAG_COLS = [
    "english_exam_score", "internship_duration_months",
    "github_avg_stars", "open_source_contribution_count",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """EDA'daki türetilmiş feature'ları ekler (train & test'e aynen)."""
    df = df.copy()
    df["technical_avg_score"] = df[TECHNICAL_SKILL_COLS].mean(axis=1)
    df["social_avg_score"] = df[SOCIAL_SKILL_COLS].mean(axis=1)
    df["portfolio_online_avg"] = df[PORTFOLIO_COLS].mean(axis=1)
    df["mentor_feedback_length"] = df["mentor_feedback_text"].fillna("").apply(len)
    for col in MISSING_FLAG_COLS:
        df[col + "_is_missing"] = df[col].isnull().astype(int)
    return df


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_x.csv")

    train = add_features(train)
    test = add_features(test)

    # Modele girecek kolonlar: tüm sayısallar (hedef ve id hariç) + kategorikler
    drop_cols = {ID_COL, TARGET, "mentor_feedback_text"}
    numeric_cols = [
        c for c in train.select_dtypes(include=["int64", "float64"]).columns
        if c not in drop_cols
    ]
    feature_cols = numeric_cols + CATEGORICAL_COLS

    X = train[feature_cols]
    y = train[TARGET].values
    X_test = test[feature_cols]

    # Kategorikleri one-hot, sayısallar olduğu gibi (HGB eksikleri yönetir)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=None,
        l2_regularization=1.0,
        random_state=SEED,
    )
    pipe = Pipeline([("prep", preprocess), ("model", model)])

    # ----- 5-fold CV ile RMSE -----
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rmses = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        pred = pipe.predict(X.iloc[va_idx])
        pred = np.clip(pred, 0, 100)
        rmse = root_mean_squared_error(y[va_idx], pred)
        rmses.append(rmse)
        print(f"  fold {fold}: RMSE = {rmse:.4f}")
    print(f"\nCV RMSE: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")

    # ----- Tüm train ile yeniden eğit, test tahmini -----
    pipe.fit(X, y)
    test_pred = np.clip(pipe.predict(X_test), 0, 100).round(2)

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"\nsubmission.csv yazildi -> {OUTPUT_PATH}")
    print(f"satir sayisi : {len(submission)}")
    print(f"skor araligi : {submission[TARGET].min()} - {submission[TARGET].max()}")
    print(f"ortalama     : {submission[TARGET].mean():.2f}")
    print("\nilk 5 satir:")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
