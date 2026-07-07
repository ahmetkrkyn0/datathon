# Tuna'ya — fold-hizalı yeniden test SONUÇLARI

Senin §6'da yazdığın şeyi yaptım: modellerimi **senin `folds.parquet`'inle** (RepeatedStratified
5×3) yeniden eğittim. Fold-hizasızlık itirazın artık geçersiz — ve sonuç ilginç.

## Önce: fold-iyimserliği konusunda HAKLIYDIN

Bizim KFold(10) OOF'umuz gerçekten iyimser-yanlıymış:

```
Bizim tek-model rw-OOF:
  KFold(10) [eski]   = 83.85
  Senin fold'un [yeni] = 84.54   (+0.69 — KFold10 bu kadar şişikmiş)
```

Yani "10-fold daha çok train görür → iyimser" tespitin doğru. Bunu kabul ediyorum.

## Ama: "kazanç fold-artefaktı" iddian ÇÖKÜYOR

Asıl iddian, −1.58'lik birleşme kazancının önemli kısmının fold-hizasızlıktan geldiğiydi.
Fold'u **senin fold'unla** birebir hizaladıktan sonra ölçtüm — kazanç hiç azalmadı:

```
0.50*sub2 + 0.50*ourteam_tunafold   rw-OOF = 82.70   (sub2 84.26'ya göre -1.55)
```

Ve **senin kendi gate'ini, senin fold hücrelerinde** koştum:

| Senin gate'in | Senin eski ölçümün | Fold-hizalı yeni |
|---|---|---|
| Hücre tutarlılığı (≥12/15) | 11/15 ❌ | **15/15 ✅** |
| Paired t-test p (<0.01) | p=0.137 ❌ | **p=7.2e-07 ✅** |
| Bootstrap %95 CI (rw, B=5000) | — | **[+0.94, +2.20], P(≤0)=0.0000 ✅** |
| Yıl-paneli (8 yıl) | — | **8/8 pozitif ✅** |

15 hücrenin **hepsinde** pozitif (min +0.34, max +2.94). Senin 11/15'in, KFold(10) OOF'unun
senin fold'una hizasız olmasının yarattığı gürültüydü — düzeltince 15/15 oluyor.

## Sonuç

Birleşme kazancı (−1.55) **gerçek, hücre-tutarlı ve senin kendi eşiklerinden geçiyor.**
Engel teknikti (fold uyumsuzluğu), prensip değil — tam senin dediğin gibi. Şimdi engel kalktı.

## Sana gönderdiğim 3 dosya

- `ourteam_oof_tunafolds.npy` — senin fold'unla eğitilmiş OOF (5 model × 2 seed NNLS blend)
- `ourteam_test_tunafolds.npy` — aynı modelin test tahmini
- `preds_tunafolds.npz` — tek tek model OOF/test'leri (lgbm/xgb/cat/mlp/nn) + y + w_fit + years
  (kendi probe'unda istediğin gibi yeniden blendleyebilmen için)

Lütfen kendi `probe_compare_solutions.py`'ınla doğrula. Benim ölçümümle ±0.2 içinde
çıkması beklenir (artık aynı fold). Geçerse final blend'i 0.50/0.50 (veya senin probe'unun
seçtiği oran — eğri 0.45-0.55 arası düz) yapıp gönderelim → takıma ~−1.5.

— Ahmet
