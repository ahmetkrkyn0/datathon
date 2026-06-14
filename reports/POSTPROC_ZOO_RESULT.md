# POSTPROC ZOO — Zincirleme-Birikimli Sonuc (blend_ppstack)

KARAR METRIGI = nested recency-weighted OOF MSE (rw-OOF). Overfit kapisi YOK (kullanici karari): en ufak nested dusus bile alinir.

- base (blend) nested rw-OOF = **82.239761**
- final (blend_ppstack) nested rw-OOF = **82.193623**
- toplam delta = **-0.046138**
- uygulanan sira = `['pp_xlmrgate', 'pp_winsor']`

## 1) Her metodun BLEND uzerindeki tek-delta'si (bagimsiz)
| metod | tek-delta (rw) | nested secim ozeti |
|---|---|---|
| pp_xlmrgate | -0.044310 | a_mode=0.3 a_dist={0.2: 1, 0.3: 9, 0.4: 1, 0.5: 4} q_thr_med=17.326; frozen q_thr=21.125 a=0.5 |
| pp_winsor | -0.001828 | p=0.0%:1/15, p=0.1%:4/15, p=0.25%:10/15; frozen p=0.25% (lo=43.2523) |

En iyi tek post-process: **pp_xlmrgate** (-0.044310).

## 2) Greedy zincir (kumulatif)
| adim | metod | step-delta | kumulatif rw | frozen/nested info |
|---|---|---|---|---|
| 1 | pp_xlmrgate | -0.044310 | 82.195451 | a_mode=0.3 a_dist={0.2: 1, 0.3: 9, 0.4: 1, 0.5: 4} q_thr_med=17.326; frozen q_thr=21.125 a=0.5 |
| 2 | pp_winsor | -0.001828 | 82.193623 | p=0.0%:1/15, p=0.1%:4/15, p=0.25%:10/15; frozen p=0.25% (lo=43.2523) |

## 3) DURUST YORUM
- **Kapisiz secim:** Bu zincir overfit kapisindan (0.25*cv_std ~ kabul kapisi) GECMEDI; kullanicinin acik talimati uzerine kapisiz, en-ufak-dusus mantigiyla secildi.
- **Nested ama paired-gate yok:** Her donusum nested (hucre-disindan fit) -> in-sample iyimserlik elimine edildi. Ancak delta'lar cv_std'nin (~birkac MSE) cok altinda; paired blocked-bootstrap ile 'istatistiksel anlamli' DEGILLER. **Private'da bu kazancin isaretinin korunacagi GARANTI DEGIL** (private-belirsiz).
- **Ne kadari gercek vs meta-overfit:** xlmrgate kazanci (~-0.044) lokal+kanitlanmis bir mekanizmadan (metin uclarda GBDT'den iyi; public 82.24->82.11 ile tutarli) -> daha guvenilir. winsor kazanci (~-0.0018) MIKROSKOBIK; nested grid-secim gurultusunden ayirt edilemez, buyuk olasilikla meta-overfit/no-op sinirinda. Toplam delta'nin ASIL govdesi xlmrgate'ten gelir.
- **Cakisma:** Iki eksen (alt-kuyruk trim vs ust-yon gate) buyuk olcude ortogonal; zincir toplami ~ iki tek-delta'nin toplamina yakin -> ciddi cakisma yok.
- **SUB onerisi:** blend_ppstack'i ancak base blend ile YAPISAL es-degerli (ayni 12-model + ince postproc) gorup, private risk dagitiminda DIKKATLE kullan; SUB-1 (sade catboost) ayri tutulmali.
