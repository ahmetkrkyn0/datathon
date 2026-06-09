"""
Rastgele submission üretici (baseline).

01_data_understanding_eda.ipynb boş olduğu için veri yapısı doğrudan
data/ klasöründeki dosyalardan çıkarıldı:

  - Hedef değişken : career_success_score  (sürekli, 0-100 aralığı)
  - ID kolonu      : student_id
  - Submission     : student_id,career_success_score   (sample_submission.csv ile aynı)

Train hedef istatistikleri: min=0, max=100, mean~77, std~15.
Bu script test_x.csv'deki her öğrenci için 0-100 aralığında rastgele
bir skor üretir ve submission.csv olarak kaydeder.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Ayarlar
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # src/ -> proje koku
DATA_DIR = ROOT / "data"
TEST_PATH = DATA_DIR / "test_x.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = ROOT / "submissions" / "submission_random.csv"

ID_COL = "student_id"
TARGET_COL = "career_success_score"

# Train hedefinin gerçek aralığı/istatistikleri (rastgeleyi gerçekçi tutmak için)
TARGET_MIN, TARGET_MAX = 0.0, 100.0
TARGET_MEAN, TARGET_STD = 76.94, 15.19

SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)

    test_df = pd.read_csv(TEST_PATH)
    ids = test_df[ID_COL]
    n = len(ids)

    # Train dağılımına benzer normal dağılımdan üret, 0-100'e kırp.
    scores = rng.normal(loc=TARGET_MEAN, scale=TARGET_STD, size=n)
    scores = np.clip(scores, TARGET_MIN, TARGET_MAX).round(2)

    submission = pd.DataFrame({ID_COL: ids, TARGET_COL: scores})
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"submission.csv yazildi -> {OUTPUT_PATH}")
    print(f"satir sayisi : {len(submission)}")
    print(f"skor araligi : {submission[TARGET_COL].min()} - {submission[TARGET_COL].max()}")
    print(f"ortalama     : {submission[TARGET_COL].mean():.2f}")
    print("\nilk 5 satir:")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
