# SP_RESULT — pseudo-base'leri 14-model havuzla birlestirme (GERCEK-nested)

KARAR = nested recency-weighted OOF (rw). DUSUK=iyi. Tum blend rw'ler NESTED (her hucre param hucre-DISINDAN); in-sample SADECE overfit-gap tanisi.

## Kiyas tabanlari

- base (14-model ridge_pos NESTED blend): **82.1640** (cv_std 2.6960, 0.25*std kabul-bandi 0.6740)
- combinatorial (su ana kadarki EN IYI gercek-nested hedef): **82.0164** (public 82.12; onceki ~40 meta/gating denemesi GERCEK-nested GECEMEDI, en iyi 82.0165)

## 1) Her aday base: standalone + tek-kolon blend katkisi + overfit-gap

| base | standalone rw | +kolon NESTED blend rw | delta(base) | in-sample | overfit-gap | corr(base blend) |
|---|---|---|---|---|---|---|
| `sp_pseudo_blend` | 83.8360 | 81.7754 | -0.3886 | 81.6392 | +0.1362 | 0.9898 |

> overfit-gap = nested - in-sample. Kucuk pozitif (~0) = saglikli; buyuk pozitif/negatif = meta-overfit suphesi. corr(base blend) yuksek (~0.99) AMA katki ortogonal kalintidan.

## 2) Nihai kombinasyon (en dusuk NESTED rw)

| blend | method | nested rw | vs base | vs combinatorial |
|---|---|---|---|---|
| nnls_full | - | 81.8950 | -0.2690 | -0.1214 |
| ridge_pos_full | - | 81.7754 | -0.3886 | -0.2410 |
| greedy_nnls | - | 81.8578 | -0.3062 | -0.1586 |

- **SECILEN:** `ridge_pos_full` (method=ridge_pos) nested rw = **81.7754**
- vs base 82.1640: delta **-0.3886** (kabul-kapisi 0.25*std=0.6740: **GECMEDI**)
- vs combinatorial 82.0164: delta **-0.2410** (**GECTI**)
- final overfit-gap (nested - in-sample) = **+0.1362**
- secilen modeller: `lgbm_full+lgbm_num+lgbm_full_w+catboost_full+catboost_full_w+txt_ridge+e5_ridge+mm+lgbm_full_h+lgbm_full_ht+xlmr+ourteam_tf+histgbr_full+lgbm_num_h+sp_pseudo_blend`
- final agirliklar (ridge_pos): lgbm_full=0.0000, lgbm_num=0.0536, lgbm_full_w=0.0000, catboost_full=0.0146, catboost_full_w=0.0026, txt_ridge=0.0000, e5_ridge=0.0001, mm=0.0121, lgbm_full_h=0.0082, lgbm_full_ht=0.0520, xlmr=0.0611, ourteam_tf=0.4760, histgbr_full=0.0000, lgbm_num_h=0.1345, sp_pseudo_blend=0.2693, intercept=-6.3458

## 3) DURUST yorum

sp_pseudo_blend havuza eklenince NESTED ridge_pos blend 82.1640 -> 81.7754 (-0.3886) ve combinatorial 82.0164'i **GERCEK-NESTED GECTI** (delta -0.2410). Bu, onceki ~40 meta/gating denemesinin (en iyi 82.0165) KIRAMADIGI tavani asan ILK base. Mekanizma: pseudo-labeling modeli, test'in gec-yila kaymis (covariate shift, adversarial AUC ~0.668) dagilimini test-pseudo-etiketleriyle DOGRUDAN ogrenir -> base GBDT'lerin kacirdiği ortogonal sinyal. Overfit-gap +0.1362 (kucuk) ve standalone OOF fold-safe (pseudo-y fold-disi test, valid egitimde yok) -> kazanc in-sample artefakti DEGIL.

UYARI (durustluk): (a) pseudo-y = mevcut 14-model blend'in TEST tahmini; sp base bu blend'in test-uzayi 'gorusunu' icsellestirir -> ridge_pos final intercept'i belirgin negatif (-6.35) ile pseudo kolonu seviye-duzeltiliyor (meta dengeli, kacak degil ama not). (b) Gercek dogrulama PUBLIC LB'dir: rw-OOF private-durust tahmin; sp_pseudo_blend public'e CIKARILIP gap olculmeli (saglikli |gap|<=1.5*std). fezadangelenler ekibi ayni recete ile LB 83.81->83.18 (-0.63 gercek atilim) elde etti -> bagimsiz teyit var. (c) Paired-anlamlilik testi (mm/e5/xlmr ile ayni olcut) kabul oncesi onerilir.
