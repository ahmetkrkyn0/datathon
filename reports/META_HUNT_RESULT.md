# META HUNT RESULT — ridge-disi meta-ogrenici + durust-nested gating

KARAR = nested recency-weighted OOF (rw). DUSUK=iyi. Hepsi NESTED-durust (her hucre param hucre-DISINDAN; sahte/in-sample YOK).

- POOL14 (14 model): `lgbm_full, lgbm_num, lgbm_full_w, catboost_full, catboost_full_w, txt_ridge, e5_ridge, mm, lgbm_full_h, lgbm_full_ht, xlmr, ourteam_tf, histgbr_full, lgbm_num_h`
- ridge_pos(ref) referansi: **82.1640**
- combinatorial (GERCEK-nested hedef): **82.0164** (public 82.12)

## 1) Meta-aile taramasi (nested rw, dusukten yuksege)

| meta konfig | nested rw |
|---|---|
| `lasso_a0.005` | 82.0763 **<-- EN IYI** |
| `lasso_a0.01` | 82.0786 |
| `enet_a0.005_l10.8` | 82.0842 |
| `ardregression` | 82.0868 |
| `enet_a0.005_l10.5` | 82.0920 |
| `enet_a0.01_l10.8` | 82.0961 |
| `enet_a0.005_l10.2` | 82.0998 |
| `bayesianridge` | 82.1052 |
| `enet_a0.01_l10.5` | 82.1142 |
| `lasso_a0.001` | 82.1164 |
| `enet_a0.01_l10.2` | 82.1322 |
| `lasso_a0.03` | 82.1435 |
| `ridge_pos(ref)` | 82.1640 |
| `enet_a0.03_l10.8` | 82.1697 |
| `enetpos_a0.005_l10.8` | 82.1744 |
| `enetpos_a0.01_l10.8` | 82.1784 |
| `enetpos_a0.005_l10.5` | 82.1800 |
| `enetpos_a0.005_l10.2` | 82.1900 |
| `lasso_a0.1` | 82.1950 |
| `enetpos_a0.01_l10.5` | 82.1965 |
| `enetpos_a0.03_l10.8` | 82.2048 |
| `enetpos_a0.01_l10.2` | 82.2125 |
| `enet_a0.03_l10.5` | 82.2220 |
| `enetpos_a0.03_l10.5` | 82.2564 |
| `enet_a0.03_l10.2` | 82.2843 |
| `huber_e1.5_a0.0001` | 82.2898 |
| `huber_e1.5_a0.001` | 82.2898 |
| `huber_e1.5_a0.01` | 82.2900 |
| `enetpos_a0.03_l10.2` | 82.3283 |
| `huber_e1.35_a0.001` | 82.3567 |
| `huber_e1.35_a0.01` | 82.3568 |
| `huber_e1.35_a0.0001` | 82.3573 |
| `huber_e1.2_a0.01` | 82.4019 |
| `huber_e1.2_a0.0001` | 82.4022 |
| `huber_e1.2_a0.001` | 82.4025 |

## 2) En iyi meta + durust-nested gating zinciri

- En iyi meta base: **`lasso_a0.005`** nested rw = **82.0763**
- Gating zinciri (1 asama): conf:e5_ridge:up(q=13,a=0.3)
- Gating-sonrasi final nested rw = **82.0165** (delta base -0.0598)

## 3) Combinatorial kiyas (GERCEK-nested)

- final 82.0165  vs  combinatorial 82.0164  -> delta **+0.0001**
- **GECILDI mi? HAYIR**

## 4) DURUST yorum

En iyi nested meta `lasso_a0.005` ridge_pos(82.1640) base'ini +0.0877 iyilestirdi (regularizasyon 14-model arasi redundansa karsi hafif kazanc), AMA gating-sonrasi 82.0165 hala combinatorial 82.0164'un +0.0001 USTUNDE -> combinatorial GERCEK-nested GECILMEDI. Base 14-model OOF doygun; meta birlestirme SEKLI tek basina combinatorial'i kiramadi. Bu DURUST sonuc: sahte (in-sample) kazanc URETILMEDI. combinatorial muhtemelen test-uzayi avantaji/gating kombinasyonundan geliyor; nested-durust base+gating tavani ~82.05-82.16.
