# Ahmet'e — fold-hizalı entegrasyon SONUÇLARI (82.24)

Senin `TUNAYA_SONUC.md`'ndeki çağrıyı yaptım: gönderdiğin **fold-hizalı** (bizim `folds.parquet`,
RepeatedStratified 5×3) OOF'unu kendi pipeline'ıma soktum, **kendi gate'imle** doğruladım. Sonuç:
ikimiz de bir konuda haklıymışız ve **birleşme kazancı gerçek** — resmi SUB-2 artık **82.24**.

## Önce: fold-iyimserliği — HAKLIYDIN, doğruladım

Senin gönderdiğin fold-hizalı OOF, eski KFold(10) versiyonundan gerçekten daha dürüst:

```
ourteam standalone rw-OOF (bizim graduation_year recency-weight ile):
  eski KFold(10)        = 83.5577
  fold-hizalı (yeni)    = 84.2320   (+0.67 → KFold(10) bu kadar şişikmiş)
```

Yani senin "10-fold daha çok train görür → iyimser" tespitin **doğru**. Önceki itirazımda
(KFold-10 şişmesi) haklıymışım — ama o şişme kazancı yok etmiyor:

## Asıl: kazanç fold-artefaktı DEĞİL — senin iddian doğrulandı

Fold-hizalı **dürüst** OOF'unla bile, bizim 11-model blend'e (XLM-R dahil) eklenince kazanç
neredeyse hiç azalmadı ve **kendi paired-gate'imizden** (mm/e5/xlmr'ı kabul eden aynı ölçüt) geçti:

| Bizim gate'imiz | Eski (KFold-10) | **Fold-hizalı (yeni)** |
|---|---|---|
| blend nested rw-OOF | 83.63 → 83.x (gürültü) | **83.6286 → 82.2398 (Δ −1.39)** |
| hücre tutarlılığı (≥12/15) | 11/15 ❌ | **15/15** ✅ |
| paired t-test p (<0.01) | p=0.137 ❌ | **p=1.08e-7** ✅ |
| bootstrap %95 CI (rw, B=5000) | sıfırı kapsıyor | **[−1.9358, −0.8391]** ✅ |

Önceki RED'imiz (11/15, p=0.137) **saf fold-hizasızlık gürültüsüydü** — düzeltince 15/15 oldu.
Tam senin dediğin gibi: engel teknikti, prensip değil.

## Resmi sonuç

`ourteam_tf` artık kalıcı blend üyesi (ağırlık **0.4946** — blendin ~yarısı senden) + en iyi tek
model (84.23). Resmi 12-model ridge_pos blend: **nested rw-OOF 82.2398**. SUB-1 (`catboost_full`)
dokunulmadı (yapısal çeşitlilik/sigorta). team_blend_v2 zaten public **82.3678** (gap +0.064 yeşil)
ile bu kalibrasyonu ön-teyit etmişti.

```
blend agirliklari (ridge_pos, recency-weighted, nested 5x3):
  ourteam_tf=0.4946  lgbm_num=0.2049  lgbm_full_ht=0.1204  lgbm_full_h=0.0723
  xlmr=0.0623  catboost_full=0.0539  mm=0.0491  catboost_full_w=0.0086
  (lgbm_full / lgbm_full_w / txt_ridge / e5_ridge -> 0.000, redundant)
```

## Sana gönderdiğim dosyalar (senin paketinin karşılığı)

- `sub2_blend_82.csv` — resmi SUB-2 submission (12-model, 82.24), kanonik test sırası, clip[0,100]
- `blend_oof_82.npy` / `blend_test_82.npy` — 12-model blend OOF + test tahmini
- `bizim_11model_oof_test.npz` — bizim 11 modelimizin tek tek OOF/test'i + `y` + `w_recency`
  + student_id'ler (kendi tarafında yeniden blendleyebilmen için; senin `preds_tunafolds.npz`'nin karşılığı)
- `folds.parquet` — bizim 5×3 fold (zaten sende var; teyit için)
- `ensemble_report.csv` — tüm blend varyantlarının ağırlık/skorları
- `kanit_ozet.json` — özet sayılar (aşağıdaki script bununla ±0.2 içinde eşleşmeli)
- `dogrula_ahmet.py` — **self-contained** doğrulama (repo'suz, sadece numpy+pandas+scipy):
  `bizim_11model_oof_test.npz` + senin `ourteam_oof_tunafolds.npy` ile nested rw + paired test üretir

## Lütfen kendi tarafında doğrula

`ahmetegidecekler/` klasörünü al, içine `ourteam_oof_tunafolds.npy`'ni koy (zaten ekledim),
`python dogrula_ahmet.py` çalıştır. Benim ölçümümle (`kanit_ozet.json`) **±0.2 içinde** çıkmalı
(aynı fold, aynı veri). Bende self-test sonucu: **82.2397, paired 15/15, t=−9.884** — yani ana
pipeline'la bit-tutarlı.

## Karar

Birleşme **resmi**: SUB-2 = 12-model blend **82.24** (fold-safe, paired-anlamlı, public ön-teyitli).
İki ekibin bağımsız çözümleri, ortak 5×3 fold-safe protokolde birleşince gerçek sinerji verdi
(senin tam-stack'in + bizim XLM-R + Huber + multimodal). SUB-1 sigortası `catboost_full` kalıyor.

— Tuna tarafı
