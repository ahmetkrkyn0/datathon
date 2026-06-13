# Ahmet'e — feature paketi (e5 SVD-50 + anchor)

İstediğin feature'lar hazır: **`tuna_features_for_ahmet.npz`**. İçindekiler (hepsi student_id ile
satır-hizalı, train+test):

| Anahtar | Şekil | Ne |
|---|---|---|
| `e5_svd50_train/test` | (10000, 50) | multilingual-e5-large (1024d) → TruncatedSVD-50 (varyans %75) |
| `anchor_train/test` | (10000, 51) | bizim 51-feature anchor (37 sayısal + 2 ham yıl + 7 missing-flag + 5 kategorik) |
| `anchor_cols` | (51,) | kolon adları |
| `categorical_cols` | (5,) | kategorik kolonlar (int-koda çevrildi: department/tier/role/hobby/social) |
| `student_id_train/test` | (10000,) | hizalama anahtarı |

## FOLD-SAFE garanti (önemli)

- **e5 SVD-50:** SVD train+test havuzunda fit edildi AMA **hedefsiz** (sadece embedding geometrisi,
  y görmez) → sızıntı DEĞİL. e5 embedding'in kendisi gibi global-ama-hedefsiz. Kendi fold'unda
  güvenle kullanabilirsin.
- **anchor:** ham sayısal + yıl + isna()-flag + kategorik int-kod. Hepsi hedefsiz → fold-safe.
- 1024d ham embedding yerine SVD-50 verdim (sen "yeter, hızlı test ederim" dedin). 1024d istersen
  söyle, `artifacts/emb_{train,test}.npy` (10000×1024) ham hali de var.

## DÜRÜST ÖN-TEST (beklentini kalibre etsin)

Senin yapacağın testin bir ön-provasını **bizim tarafta** çalıştırdım: e5 SVD-50, bizim 82.24
blend'inin residual'ini açıklıyor mu?

```
e5 SVD-50 -> blend(82.24) residual:  corr = -0.0282   R2 = -0.0028
```

**Açıklamıyor** (|corr| < 0.05, senin eşiğin). Sebep: `e5_ridge` zaten blend'in bir üyesi →
e5'in tek-başına taşıdığı sinyal blend'de mevcut.

**AMA senin sorun farklı ve hâlâ geçerli:** sen e5'i **kendi 178-feature'ınla birleştirince yeni
ETKİLEŞİM** doğup doğmadığını soruyorsun. Bizim blend'de e5 var ama senin feature'larınla
etkileşimli değil. O kapı hâlâ açık — bu yüzden paketi gönderiyorum. Ama dürüst beklentim: e5
tek-başına residual açıklamadığına göre, etkileşimden büyük sinyal çıkma ihtimali düşük.

## Senin testin

Bu feature'ları kendi 178'inle TEK pipeline'da birleştir, **bizim 5×3 fold'unla** (folds.parquet
pakette var) GBM/FTT eğit, blend(82.24) residual'ine karşı corr ölç. Eşiğin: |corr| > 0.05.

- **Açıklarsa** → gerçek yeni sinyal, blend'i kırabiliriz. Yeni OOF'u `ahmetegidecekler` formatında
  gönder, paired-gate'ten geçiririz.
- **Açıklamazsa** → feature uzayı iki ekipte de tükenmiş, 82.24 ortak tavan. O zaman lider 80.41
  büyük olasılıkla public-overfit (senin de dediğin gibi).

Her iki sonuç da değerli. SUB-2 = 82.24 bu arada sağlam.

— Tuna tarafı
