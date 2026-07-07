# GATING MAXOUT — Birikimli Greedy Zincir Sonucu

- **raw blend nested rw-OOF**: 82.239761  (referans 82.2398)
- **mevcut en-iyi-gate referansi**: 82.157
- **FINAL zincir nested rw-OOF**: 82.050399
- **delta vs raw blend**: -0.189362  (DUSTU)
- **delta vs en-iyi-gate**: -0.106601
- **post unweighted cv**: mean=72.7240 std=2.6593
- **81.5 ulasildi mi?**: HAYIR

## Tek-hamle tani (zincir uzunlugu 1, base=blend)

| aile | model | yon | nested rw | delta |
|---|---|---|---|---|
| conf | txt_ridge | up | 82.141899 | -0.097862 |
| conf | xlmr | up | 82.195451 | -0.044310 |
| band | xlmr | up | 82.210715 | -0.029046 |
| band | mm | down | 82.224538 | -0.015223 |
| conf | e5_ridge | up | 82.238924 | -0.000837 |
| conf | xlmr | down | 82.239761 | +0.000000 |
| conf | e5_ridge | down | 82.239761 | +0.000000 |
| conf | txt_ridge | down | 82.239761 | +0.000000 |
| band | txt_ridge | up | 82.242350 | +0.002589 |
| band | txt_ridge | down | 82.242460 | +0.002699 |
| band | xlmr | down | 82.247736 | +0.007975 |
| band | e5_ridge | up | 82.253815 | +0.014054 |
| conf | mm | down | 82.262956 | +0.023195 |
| band | e5_ridge | down | 82.266194 | +0.026433 |
| conf | mm | up | 82.270304 | +0.030543 |
| band | mm | up | 82.279315 | +0.039554 |

## Birikimli zincir (greedy, her adim re-nested-fit)

| adim | hamle | kumulatif nested rw | adim-delta |
|---|---|---|---|
| 0 | base(blend) | 82.239761 | +0.000000 |
| 1 | conf:txt_ridge:up | 82.141899 | -0.097862 |
| 2 | band:xlmr:up | 82.074981 | -0.066918 |
| 3 | band:mm:down | 82.059759 | -0.015223 |
| 4 | conf:xlmr:up | 82.055357 | -0.004402 |
| 5 | band:e5_ridge:up | 82.050399 | -0.004958 |

**Zincir sirasi**: conf:txt_ridge:up -> band:xlmr:up -> band:mm:down -> conf:xlmr:up -> band:e5_ridge:up

## Asama-bazli nested param secimi (15 hucre ozeti)

- conf:txt_ridge:up[q_med=13.581,a={0.2: 5, 0.3: 10}]
- band:xlmr:up[q_med=91.332,a={0.5: 15}]
- band:mm:down[q_med=60.267,a={0.0: 1, 0.3: 5, 0.5: 9}]
- conf:xlmr:up[q_med=21.163,a={0.2: 4, 0.3: 1, 0.4: 2, 0.5: 8}]
- band:e5_ridge:up[q_med=97.666,a={0.5: 7, 0.7: 8}]

## Test'e uygulanan FROZEN zincir param (tum-OOF)

- `conf:txt_ridge:up q=13.5781 a=0.3 | band:xlmr:up q=91.3429 a=0.5 | band:mm:down q=60.2826 a=0.3 | conf:xlmr:up q=21.1246 a=0.5 | band:e5_ridge:up q=97.6573 a=0.7`
- frozen-OOF rw (TANI, in-sample yanli — KARAR DEGIL): 81.976551

## DURUST YORUM

- **Nested, in-sample DEGIL**: her (repeat,fold) hucresinin gate param'i (yon, q_thr, a) o hucre DISINDAKI OOF'tan tr-rw minimize ile secildi, hucreye uygulandi, 3-repeat ortalandi. Raporlanan final_rw bu nested_oof'un rw-OOF'udur (DURUST). frozen-OOF rw yalniz test param'i icin uretildi ve TANI olarak etiketlendi (in-sample yanli, KARAR DEGIL).
- **Kapi YOK**: bu calisma overfit kapisindan (paired-anlamli / 0.25*std) GECMEDI; kullanici 'kapisiz, en alta indir' dedigi icin yapildi. Bu bir KESIN-IYILESME IDDIASI DEGIL; gating mekanizmasinin nested-durust tavanini olcer.
- **Ne kadari gercek mekanizma vs grid-secim gurultusu**: gating'in cekirdek tezi (metin modelleri UCLARDA GBDT'den guvenilir -> base'i uca ceken kucuk a) gercek bir sinyaldir (public 82.24->82.11 tek-asamada teyit edildi). ANCAK birikimli zincirde her ek asamanin kazanci hizla kuculur ve grid (yon x q x a) secim serbestligi rw'yi bir miktar ASAGI-YANLI eğer; nested protokol bu yanliligi BUYUK olcude (ama %100 degil) temizler. Ek asamalarin marjinal deltasi nested-gurultu bandina (~0.0x MSE) yaklastikca, kazancin artan kismi mekanizma DEGIL grid-arama gurultusudur. Bu yuzden zincir, dusurmeyi BIRAKINCA (adim-delta >= -eps) durdurulur.
- **81.5 hedefi**: ULASILMADI. Final 82.0504; 81.5'e +0.5504 uzakta. Gating tek-kaldirac olarak blend'i ~82.0-82.1 bandina indirebiliyor; 81.5 bu mekanizmanin nested-durust erisiminde DEGIL. 81.5'in altina inmek farkli/daha guclu base modeller gerektirir, gate ince-ayari degil.
