# Tuna'ya — feature-düzeyi birleşme denemesi

OOF-düzeyinde birleştik (82.37, doğrulandı). Ama daha derin bir kapı kaldı: **feature-düzeyi
birleşme.** Şu ana kadar 5 model ailesi (XLM-R full-FT, quantile, regex, MLP, FT-Transformer)
denedim — hepsinin **residual korelasyonu ~0**, yani bizim 178-feature uzayında çıkarılabilecek
tüm sinyal zaten blend'de. Mimari değiştirmek yeni bilgi getirmiyor.

**Tek gerçek 'yeni bilgi' şansı:** senin feature'ların bizimkinde olmayan bir sinyal taşıyorsa,
ya da iki feature seti BİRLEŞİNCE yeni etkileşim doğarsa.

## Senin imzaların ki bizde YOK

1. **`e5_ridge` — multilingual-e5-large embedding.** Biz MiniLM/BERTurk kullandık, e5 farklı bir
   uzay. OOF olarak elimizde ama **ham embedding boyutları** (1024d veya SVD'si) feature olarak
   farklı etkileşim verebilir.
2. **native-kategorik kodlama** (LightGBM native NaN/kategorik). Biz target-encoding kullandık.
3. **ham yıl + 51-feature anchor** — senin minimal ama güçlü tabanın.

## Ne istiyorum

İdeal: senin **feature matrisini** (train + test, student_id ile hizalı) parquet/npy olarak.
- En değerlisi: `e5_ridge`'in ARKASINDAKİ ham e5 embedding (1024d) ya da SVD'si (~50d)
- Ayrıca varsa: senin anchor 51-feature'ın ham hali

Bunları bizim 178-feature'la TEK pipeline'da birleştirip, senin 5×3 fold'unla GBM/FTT eğitip
**residual korelasyonu** ölçeceğim. Eğer e5-feature residual'i açıklıyorsa (kor |>0.05|), gerçek
yeni sinyal var demektir ve blend'i kırabiliriz. Açıklamıyorsa, feature uzayı gerçekten tükenmiş
ve 82.37 iki ekibin de tavanı — o zaman lider 80.41 büyük ihtimalle public-overfit.

## Neden bu mantıklı

İki ekip bağımsız feature mühendisliği yaptı (senin 51-anchor + e5/Huber, bizim 178 +
segment-yıl-TE/kohort-z). OOF blend bu farkı yakaladı (−1.4). Ama feature-düzeyinde HİÇ
birleşmedik — belki orada hâlâ ortogonal sinyal var. Son denenmemiş kapı bu.

Eğer feature matrisi büyükse (e5 1024d ağır), sadece **e5 SVD-50 + en güçlü 20 anchor feature**
yeter. Hızlı test ederim.

— Ahmet
