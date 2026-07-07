"""Derin EDA — veriyi tanima. Metrik: MSE (overview.md).
Cikti tamamen stdout; grafik yok (terminal dostu)."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TARGET = "career_success_score"

train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test_x.csv")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)


def sec(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


sec("1) HEDEF")
y = train[TARGET]
print(y.describe())
print("0 olan:", (y == 0).sum(), "| 100 olan:", (y == 100).sum(),
      "| >100:", (y > 100).sum(), "| <0:", (y < 0).sum())
print("Sabit (mean) tahmin MSE  :", round(((y - y.mean()) ** 2).mean(), 3))
print("Sabit (median) tahmin MSE:", round(((y - y.median()) ** 2).mean(), 3))

sec("2) EKSIK DEGERLER (train)")
miss = pd.DataFrame({
    "n_missing": train.isnull().sum(),
    "pct": (train.isnull().mean() * 100).round(2),
})
print(miss[miss.n_missing > 0].sort_values("n_missing", ascending=False))
print("\nTest eksikleri:")
mt = test.isnull().sum()
print(mt[mt > 0].sort_values(ascending=False))

sec("3) KOLON TIPLERI")
num = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
num.remove(TARGET)
cat = train.select_dtypes(include=["object"]).columns.tolist()
print(f"Sayisal ({len(num)}):", num)
print(f"\nKategorik/metin ({len(cat)}):", cat)

sec("4) SAYISAL KORELASYON (target ile, |r| sirali)")
corr = train[num + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)
print(corr_sorted.round(3).to_string())

sec("5) KATEGORIK ETKI (target ortalamasi)")
for c in ["target_role", "university_tier", "department", "hobby",
          "preferred_social_media_platform"]:
    g = train.groupby(c)[TARGET].agg(["count", "mean"]).sort_values("mean", ascending=False)
    spread = g["mean"].max() - g["mean"].min()
    print(f"\n--- {c}  (kategori={g.shape[0]}, mean spread={spread:.2f}) ---")
    print(g.round(2).to_string())

sec("6) TRAIN vs TEST DAGILIM KAYMASI (drift)")
comp = pd.DataFrame({
    "train_mean": train[num].mean(), "test_mean": test[num].mean(),
    "train_std": train[num].std(), "test_std": test[num].std(),
})
comp["abs_mean_diff"] = (comp.train_mean - comp.test_mean).abs()
print(comp.sort_values("abs_mean_diff", ascending=False).head(10).round(3).to_string())

sec("7) MENTOR FEEDBACK (NLP sinyali)")
tx = train["mentor_feedback_text"].fillna("")
ln = tx.str.len()
print("metin uzunlugu describe:")
print(ln.describe().round(1).to_string())
print("uzunluk-target korelasyonu:", round(ln.corr(y), 3))
# yuksek vs dusuk grupta ayirici kelimeler
from sklearn.feature_extraction.text import TfidfVectorizer
hi = train[y >= y.quantile(0.90)]["mentor_feedback_text"].fillna("")
lo = train[y <= y.quantile(0.10)]["mentor_feedback_text"].fillna("")
tr_stop = ["ve", "bir", "bu", "ile", "icin", "için", "daha", "cok", "çok", "de", "da",
           "ama", "ancak", "olarak", "olan", "oldugu", "olabilir", "gibi", "kadar",
           "ise", "en", "hem", "ya", "veya", "konusunda", "bir", "sahip"]
vec = TfidfVectorizer(max_features=2000, stop_words=tr_stop, ngram_range=(1, 2), min_df=5)
vec.fit(pd.concat([hi, lo]))
words = np.array(vec.get_feature_names_out())
hi_m = np.asarray(vec.transform(hi).mean(axis=0)).ravel()
lo_m = np.asarray(vec.transform(lo).mean(axis=0)).ravel()
diff = hi_m - lo_m
order = diff.argsort()
print("\nYUKSEK basari grubuna ozgu (top 15):")
for i in order[::-1][:15]:
    print(f"  {words[i]:30s} {diff[i]:+.4f}")
print("\nDUSUK basari grubuna ozgu (top 15):")
for i in order[:15]:
    print(f"  {words[i]:30s} {diff[i]:+.4f}")

sec("DONE")
