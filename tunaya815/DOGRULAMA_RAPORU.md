# TARGET815 Dogrulama Raporu

## Sonuc

| Kontrol | Durum |
|---|---|
| Paket kendi klasorunden calisiyor | GECTI |
| Submission semasi ve ID sirasi | GECTI |
| OOF skorlarinin bagimsiz yeniden hesabi | GECTI |
| Uretim sonucu ile paket dosyalarinin hash eslesmesi | GECTI |
| Aggressive aday base'den daha iyi | GECTI |
| Tam bagimsiz 5x3 base-model kaniti | GECMEDI |
| Leaderboard 81.49 garantisi | YOK |

## Yeniden Uretilen Sonuclar

```text
base       rw-MSE = 82.23976
meta       rw-MSE = 81.78859
robust     rw-MSE = 81.61241
aggressive rw-MSE = 81.48840
```

Aggressive aday:

```text
delta = -0.75137
15/15 hucre base'den iyi
paired p = 4.04e-06
satir-bootstrap %95 delta CI = [-1.185, -0.316]
```

Fold-hizasi tam olan `repeat=0` kontrolunde:

```text
5/5 fold iyi
ortalama delta yaklasik -0.75
p = 0.0135
```

Alternatif application-year recency proxy:

```text
base       = 82.49276
aggressive = 81.76465
delta      = -0.72811
```

Yani kazanc alternatif agirlikta da devam ediyor, ancak mutlak skor 81.49 degil.

## Neden Tam Garanti Degil?

1. `fullft` ve `mmstrong` OOF dosyalari yalnizca Tuna fold `repeat=0` ile
   uretilmistir. Repeat 1 ve 2 hucreleri ayni satirlari farkli bolse de bu iki
   model icin yeni, bagimsiz base-model OOF tahminleri degildir.
2. Repeated-CV hucreleri ayni 10.000 satiri tekrar kullandigi icin 15 hucre
   tamamen bagimsiz gozlem gibi yorumlanmamalidir.
3. Birden fazla meta/gate zinciri incelendikten sonra en iyi zincir secildi.
   Bu, OOF skorunda bir miktar secim iyimserligi yaratabilir.
4. Testte gate'e giren satir orani train'den biraz yuksektir:

```text
fullft-up:       train %2.68, test %3.31
lgbm_full-down: train %9.67, test %12.11
mmstrong-up:    train %5.10, test %6.02
```

## Gercekci Beklenti

Tuna'nin onceki combinatorial submission'inda:

```text
nested OOF = 82.0164
public     = 82.1221
gap        = +0.106
```

Ayni gap tasinirsa aggressive aday icin kaba public beklentisi:

```text
81.488 + 0.106 = 81.594
```

Model/gate secim iyimserligi de hesaba katildiginda **81.55-81.80** daha
gercekci bir beklenti bandidir. 81.5 mumkun, ama garanti degildir.

## Onerilen Kullanim

- Yeni skor denemesi: `submissions/TARGET815_aggressive.csv`
- Daha yumusak varyant: `submissions/TARGET815_robust.csv`
- Finalde iki slot varsa iki TARGET815 dosyasini birlikte secmek yerine,
  aggressive adayi Tuna'nin daha once publicte dogrulanmis `combinatorial`
  submission'i ile eslestirmek daha iyi risk dagilimi saglar.

## Tekrar Kontrol Komutlari

```powershell
python -u src/build_target_815.py
python -u verify_target815.py
```
