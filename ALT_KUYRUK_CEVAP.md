# Ahmet'e — alt-kuyruk biası (y<40) hakkında

Bulgun **doğru ve doğrulandı**: 82.24 blend, y<40 öğrencilerini ortalama **18.9 puan yüksek**
tahmin ediyor (ort_resid −18.87; senin "+19" ile aynı, sadece işaret yönü y−pred). Tablo:

```
y[ 0, 40): n=143  ort_resid=-18.87   <- alt kuyruk, asiri tahmin
y[40, 50): n=353  ort_resid=-12.15
y[50, 60): n=895  ort_resid= -7.32
y[70, 80):        ort_resid= +0.69   <- isabet
y[90,100):        ort_resid= +5.19   <- ust kuyruk, eksik tahmin
```

**AMA bu monoton örüntü (düşükte yüksek, yüksekte düşük tahmin) bir model hatası değil —
regression-to-the-mean imzası.** Her MSE-optimal model bunu yapar. İki ölçüm bunu kanıtlıyor:

## 1) Huber/quantile bu kuyruğu blend'den daha iyi ÇÖZMÜYOR

y<40 bandında MSE (düşük=iyi):

| Model | y<40 MSE | y<40 ort_resid |
|---|---|---|
| **blend (82.24)** | **452.8** | −18.87 |
| ourteam_tf | 458.5 | −19.03 |
| lgbm_full_ht (**Huber**) | 501.8 | −20.30 |
| lgbm_full_h (**Huber**) | 513.7 | −20.67 |

Senin işaret ettiğin Huber modeli (`lgbm_full_ht`) alt-kuyrukta blend'den **daha kötü** (501 > 452).
Blend zaten tüm tek-modelleri geçiyor. Huber, L2'den marjinal iyi ama tavan −20 civarında sabit.

## 2) Kök sebep: y<40 ex-ante AYIRT EDİLEMEZ

- Blend'in en düşük %5 tahmini (500 öğrenci, pred≤55.8) → gerçek y ortalaması **51.5**, içinde
  gerçekten y<40 olan sadece **89**.
- 143 y<40 öğrencisinin yalnız **89'u** en düşük %5 tahminde (recall %62), precision %18.

Yani model y<40'ı yakalayamıyor çünkü **feature'ları y~50-55 öğrencilere benziyor.** Sinyal yok.
Quantile/asimetrik model bu satırlara düşük tahmin verirse, onlara benzeyen ama gerçekte 55 olan
~410 öğrenciye de düşük verir → o bantta MSE patlar → **net kayıp.**

## Sonuç (üzgünüm, bu kapı kapalı)

Bu, bizim `LOW_TAIL_LEVER.md`'de zaten test edip elediğimiz şeyin aynısı: sample-weight, regex,
two-stage P(y<50) — hepsi denendi, hiçbiri MSE'yi düşürmedi. Bias gerçek ama **indirgenemez
gürültü** (bilgi-seti limiti), düzeltilebilir model hatası değil.

**Denemeye DEĞER tek şey:** asimetrik-loss veya quantile bir modeli, MSE'yi düşürmesini bekleyerek
değil, **blend'e ORTOGONAL bir kolon** olarak ekleyip paired-gate'ten geçirmek. Tek-başına kötü
olsa bile (mm gibi) farklı hata yapıyorsa blend'e girebilir. Bunu test edebiliriz — ama beklentim
düşük (alt-kuyruk tüm modellerde aynı yönde yanılıyor, corr yüksek → ortogonal değil).

İstersen sen asimetrik bir model (örn. quantile q=0.4 LightGBM) bizim 5×3 fold'la üretip
`ahmetegidecekler` formatında gönder, paired-gate'ten geçiririz. Geçerse alırız; geçmezse temiz
defterleriz. SUB-2 = 82.24 (12-model) bu arada sağlam kalıyor.

— Tuna tarafı
