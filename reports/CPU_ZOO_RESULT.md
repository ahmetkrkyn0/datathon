# CPU-ZOO HARVEST — Birikimli Greedy Suzgec Sonucu

**Metodoloji:** mevcut 12-model `ridge_pos` blend (nested rw-OOF) USTUNE 8 yeni CPU base'i birikimli-greedy ile dener. Bir aday ancak nested rw-OOF'u HERHANGI bir miktar dusuruyorsa kabul edilir. **OVERFIT KAPISI YOK** (0.25*std ve paired-test UYGULANMADI — kullanici bilerek istedi).

- BASE (12-model) nested rw-OOF: **82.2398**
- FINAL (12 + 1 secilen) nested rw-OOF: **82.2167**
- delta = **-0.0230** (GERCEK DUSUS)
- final ridge_pos intercept: -5.5902

## Aday tablosu

| model | single_rw | corr(base) | greedy'de secildi | ridge_pos agirlik |
|---|---|---|---|---|
| z_et | 112.6166 | +0.9386 | hayir | - |
| z_knnpca | 154.1634 | +0.8297 | hayir | - |
| z_ridgepoly | 99.0779 | +0.9612 | hayir | - |
| z_lgbquant | 93.2054 | +0.9856 | hayir | - |
| z_catdepth | 89.1834 | +0.9843 | EVET | +0.0000 |
| z_lgbdart | 91.9497 | +0.9881 | hayir | - |
| z_gpr_te | 92.5875 | +0.9716 | hayir | - |
| z_histmono | 89.2027 | +0.9839 | hayir | - |

## SECILDI (greedy sirasi): z_catdepth

Artefakt: `artifacts/oof_blend_cpuzoo.npy` + `test_blend_cpuzoo.npy` (mevcut `oof_blend.npy` DEGISMEDI; ayri ad).

### KRITIK: full-OOF ridge agirligi = 0.00000000

z_catdepth greedy'de nested rw-OOF'u -0.0230 dusurmus gozukuyor AMA tum-OOF (final-test) ridge_pos
refit'i ona **TAM 0.0 agirlik** veriyor (corr GBDT ailesiyle 0.96-0.99). Yani **uretilen
test_blend_cpuzoo, mevcut 12-model test_blend ile pratik olarak OZDES** (yeni kolonun test-katkisi
sifir). -0.0230'luk dusus tamamen nested meta-CV'nin fold-ici agirlik dalgalanmasindan
(bazi fold'larda kucuk pozitif agirlik sansli denk gelmesi) geliyor; global optimumda katki YOK.

### DURUST YORUM

Bu dusus nested meta-CV ile olculdu (in-sample DEGIL: her hucre agirligi o hucre DISINDAKI OOF satirlarindan fit edildi) -> ham winner's-curse degil. ANCAK greedy secimi nested rw-OOF'u DOGRUDAN optimize ettigi icin (8 adaydan en cok dusureni tekrar tekrar secmek) **secim-kaynakli kucuk iyimserlik** (meta-overfit) ihtimali var; her eklenen kolon ridge'e bir serbestlik derecesi daha katar. **0.25*std kabul kapisi ve paired-test UYGULANMADI** (kullanici 'overfit kapisi onemsiz' dedi) -> bu dusus mevcut blend uyelerinin (e5/mm/xlmr/ourteam_tf) gectigi paired anlamlilik denetiminden GECMEDI. Karar otoritesi (CLAUDE.md) hala paired+0.25*std; bu artefakt o denetime girmeden nihai SUB-2'ye KONULMAMALI. corr(base) dusuk olan adaylar (ortogonal) gercek-katki adayi, corr~0.99 olanlar muhtemelen redundant secim-gurultusudur.
