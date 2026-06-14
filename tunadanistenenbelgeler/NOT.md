# Tuna -> Ahmet'e teslim (combinatorial + xlmr_gating)

İstediğin iki submission, OOF/test artefaktları ve nasıl üretildikleri aşağıda.
Hepsi `data/folds.parquet` (Repeated Stratified 5-fold x 3 repeat) hizasında, satır-bazlı.
Karar metriği: **nested recency-weighted OOF MSE** (rw-OOF). Public sadece sağlık sensörü.

---

## Klasör içeriği

| Dosya | Ne |
|---|---|
| `combinatorial.csv` | **SUB-3 / en iyi public submission (82.1221).** test_x sırasında (STU_010001..020000), clip[0,100]. |
| `sub_xlmr_gating.csv` | xlmr confidence-gating post-process submission (public 82.18). |
| `oof_blend.npy` | 12-model ridge_pos blend'in **nested-OOF**'u (10000,). combinatorial'in tabanı. |
| `test_blend.npy` | Aynı blend'in fold-bagged test tahmini (10000,). |
| `oof_pp_xlmrgate.npy` | xlmr-gating uygulanmış nested-OOF (sub_xlmr_gating'in OOF'u). nested rw = **82.1955**. |
| `test_pp_xlmrgate.npy` | xlmr-gating test tahmini (sub_xlmr_gating'in kaynağı). |
| `pp_xlmrgate.py` | xlmr-gating'i üreten script (mekanizma + nested sözleşme docstring'de). |

`*_blend.npy` = combinatorial'in girdisi; `*_pp_xlmrgate.npy` = sub_xlmr_gating'in girdisi.

---

## 1) `combinatorial.csv` (public 82.1221) — nasıl üretildi

**Taban: 12-model ridge_pos blend.** Modeller (hepsi aynı folds, nested-OOF):

```
lgbm_full, lgbm_num, lgbm_full_w, catboost_full, catboost_full_w,
txt_ridge, e5_ridge, mm, lgbm_full_h, lgbm_full_ht, xlmr, ourteam_tf
```
(= `src/ensemble.py` içindeki `CANDIDATE_POOL`. `ourteam_tf` = senin TAM çözümün —
lgbm+xgb+cat+mlp+nn NNLS blend — bizim folds.parquet ile yeniden eğitilmiş hali,
`ahmettengelenler/ourteam_oof_tunafolds.npy`. Eski KFold-10 OOF iyimser-yanlı çıktığı
için kullanılmadı; fold hizalaması doğrulandı.)

**Stack:** `ensemble.fit_weights(..., method="ridge_pos")` =
`Ridge(alpha=1.0, positive=True, fit_intercept=True)`. Ağırlıklar nested
(her (repeat,fold) hücresi için hücre-DIŞI verisinden fit) -> `oof_blend.npy`.

**Üstüne combinatorial gating** (xlmr confidence-gating, aşağıdaki #2 ile aynı mekanizma)
eklenmiş hali. Sonuç:
- nested rw-OOF = **82.0164**
- public = **82.1221** (gap **+0.11** = sağlıklı kalibrasyon)
- submissions_log: `SUB-3 combinatorial (14-model+gating)`, eşik=yeşil.

Bu bizim **en iyi dürüst submission'ımız** ve kanıtlanmış tavan. combo14_gatemax
(nested "81.90") param-sızıntısıydı -> public 82.22'ye geri tepti; combinatorial onu yendi.

## 2) `sub_xlmr_gating.csv` (public 82.18) — nasıl üretildi

`pp_xlmrgate.py`. 12-model blend OOF'unu (`oof_blend.npy`) **lokal, koşullu, yukarı-yön**
düzeltir (yeni base değil):

```
conf = |xlmr - 76.94|                    # 76.94 = hedef ortalaması (sabit gating-merkezi)
pred = blend + [conf >= q_thr AND xlmr > 76.94] * a * (xlmr - blend)
```

Mantık: metin modeli (xlmr) **uçlarda** GBDT'den daha güvenilir; ridge stacker xlmr'e
~0 ağırlık veriyordu (lineer/global yakalayamıyor). Bu yüzden sadece yukarı-yön
(xlmr > ort) koşullu düzeltme. Aşağı-yön ölçüldü, zarar veriyordu -> yok.

**Nested sözleşme (in-sample DEĞİL):** her (repeat,fold) hücresinde `q_thr` quantile'i
ve `a` katsayısı hücre-DIŞI (tr) blend+xlmr+y'den (tr-rw minimize) seçilir, hücre (va)
transform edilir; 3-repeat ortalaması -> nested_oof. Test'e FROZEN uygulanır
(q tüm-OOF conf quantile'inden, a tüm-OOF). nested rw = **82.1955**.

---

## Kritik uyarılar (bunları atlama)

1. **Karar = nested rw-OOF, public DEĞİL.** Public/private %60/%40 rastgele bölme; public
   sadece sağlık sensörü. "Public düştü" tek başına gerekçe değil.
2. **rw-OOF -> public gap'i izle.** combinatorial gap +0.11 sağlıklı. Türetilmiş-sinyalli
   yöntemlerde (pseudo-label gibi) gap +0.47'ye şişti -> nested bile yanıltabilir.
   Bir şey "82'nin altına indi" derse paired-test + public-gap kontrolü şart.
3. **Fold-safe zorunlu:** her transform fold-içi fit. Tahmin clip[0,100]. seed=42.
   Türkçe lowercase tuzağına dikkat (I/ı, İ/i). ftfy/latin1 fix YAPMA.
4. Bu artefaktlar `oof_blend.npy`/`test_blend.npy`'ye bağlı — yeniden üretmek istersen
   12-model OOF'ları bizden iste (tek tek `artifacts/oof_<model>.npy`).
