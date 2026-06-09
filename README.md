# datathon

Career Success Score tahmini — regresyon datathonu.

**Hedef:** `career_success_score` (0–100 sürekli değer), öğrenci başına tahmin.
**Metrik:** RMSE.

## Klasör yapısı

```
datathon/
├── data/                 # Ham veri (train.csv, test_x.csv, sample_submission.csv)
├── notebooks/            # Keşifsel analiz
│   └── 01_data_understanding_eda.ipynb
├── src/                  # Çalıştırılabilir script'ler
│   ├── make_random_submission.py   # Rastgele baseline -> submissions/submission_random.csv
│   └── train_baseline_model.py     # Gerçek model -> submissions/submission.csv
├── submissions/          # Üretilen submission dosyaları
│   └── submission.csv
├── Roadmap/              # 7 fazlı çalışma planı (SPEC'ler)
└── DatathonNotes/        # Yarışma notları (overview, data, rules)
```

## Çalıştırma

Tüm script'ler proje kökünden çalıştırılır (path'ler köke göre çözülür):

```bash
# Gerçek model (5-fold CV ile RMSE raporlar, submissions/submission.csv yazar)
python src/train_baseline_model.py

# Rastgele baseline (submissions/submission_random.csv yazar)
python src/make_random_submission.py
```

## Model

`src/train_baseline_model.py` — `HistGradientBoostingRegressor` (sklearn).
Notebook'taki EDA'dan türetilen feature'lar: teknik/sosyal/portfolio skor ortalamaları,
mentor yorumu uzunluğu, eksik-değer bayrakları + kategorik one-hot.
Güncel skor: **CV RMSE ≈ 9.24**.

## Bağımlılıklar

`pandas`, `numpy`, `scikit-learn`, `matplotlib`. (LightGBM/XGBoost kurulu değil;
sklearn'in gradient boosting'i kullanılıyor.)
