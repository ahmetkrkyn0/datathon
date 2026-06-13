# Tuna'ya — fold-hizalı yeniden test için istek

Analizini okudum, çok temiz iş — özellikle takım birleşmesini paired-gate'e sokup
RED gerekçesini sayısallaştırman doğru yaklaşım. Senin de §6'da yazdığın gibi:

> "Eğer Ahmet'in modelleri benim RepeatedStratified 5×3 fold'larımla yeniden
> eğitilseydi ve birleşme paired-gate'i geçseydi, kabul edilirdi... Engel teknikti
> (fold uyumsuzluğu + zaman), prensip değil."

İşte bunu yapmak istiyorum — engeli kaldıralım, sonra kararı veri versin.

## Neyi gördüm (kendi zeminimde)

Birleşmeyi kendi recency-weighted proxy'mde + bootstrap paired-test ile ölçtüm:

```
0.45*sub2 + 0.55*ourteam  rw-OOF = 82.73   (sub2 84.26'ya göre -1.53)
Bootstrap %95 CI = [+0.92, +2.18]   P(kazanç<=0) = 0.0000
Yıl-bazlı tutarlılık: 8/8 yılda birleşme daha iyi (2025: +1.86, 2026: +2.11)
```

Yani benim tarafımda kazanç hem büyük hem hücre-tutarlı. Senin 11/15 (p=0.137)
sonucunla çelişiyor — ve aradaki farkın sebebi büyük ihtimalle **tam senin
işaret ettiğin şey: benim OOF'um KFold(10), senin fold'una hizasız.** Senin
15 küçük hücren (~300 satır/hücre) yüksek varyanslı; benim 8 yıl-panelim daha temiz.

## Ne istiyorum (3 dosya)

Modellerimi **senin fold'unla** yeniden eğitip OOF üreteceğim ki paired-gate'i
**senin zemininde** test edelim. Bunun için:

1. **`folds.parquet`** (veya .npy/.csv) — senin RepeatedStratified 5×3 fold atamaların.
   İdeal format: 10000 satır × {fold_id veya her repeat için ayrı kolon}, train satır
   sırasıyla hizalı. Stratify kolonunu (yıl mı, segment mi, y-binned mi) ve `random_state`'i
   de söyle ki birebir tekrar üretebileyim.

2. **`probe_compare_solutions.py`** — paired-gate + rw-OOF + bootstrap kodun. Aynı
   eşikleri (≥12/15, p<0.01, bootstrap CI<0) birebir kullanmak istiyorum, kendi
   versiyonumu uydurmak yerine.

3. **`sub2_oof` ve `sub2_test`'in fold-hizalı hali** — bende `data/cache/sub2_oof.npy`
   (mean 76.82) var ama bunun senin son SUB-2 olduğundan emin değilim. Hangi commit/dosya
   olduğunu doğrula.

## Sonra ne olacak

- GBM ailemizi (LGBM+Cat+XGB) senin fold şemanla yeniden eğitirim → `ourteam_oof_tunafolds`.
- Birleşmeyi **senin probe + senin eşiklerinle** test ederiz.
- Geçerse: takıma -1.5 girer, ikimizin de en iyisi. Geçmezse: mesele kanıtla kapanır,
  SUB-2 kalır. İki durumda da kazanırız çünkü artık tahmin değil ölçüm var.

Eğitimi Colab+ (A100) üzerinde koşacağım, hız sorun değil. Sadece 3 dosyayı at yeter.
