# Lider farkı analizi — neden 80.41'e inemiyoruz (2026-06-13)

**Durum:** Takım 12-model blend gerçek LB **82.3678**. Lider (strawveri) **80.41**. Fark **1.96 MSE** — büyük, yapısal.

## En önemli bulgu: alt-kuyruk biası (gerçek ama düzeltilemez)

Model düşük-skorlu öğrencileri sistematik YÜKSEK tahmin ediyor:

| Gerçek skor | n | MSE | Bias (tahmin−gerçek) |
|---|---|---|---|
| 0–40 | 143 | 447.8 | **+18.87** |
| 40–55 | 717 | 172.8 | +10.51 |
| 55–70 | 2246 | 70.3 | +3.53 |
| 70–85 | 3654 | 62.9 | −1.34 |
| 85–100 | 2467 | 77.7 | −4.61 |

y<40 (143 satır, %1.4) tek başına rw-MSE'ye **9.25 puan** katkı. Mükemmel çözülse rw 82.5→73.25 olurdu.
**AMA:** y<40 grubu feature'larda ayrışıyor (project_quality −1.0 std) ama out-of-fold sızıntısız
TAHMİN edilemiyor (residual-fit R²=−0.03). MSE'nin doğası: bu nadir grubu aşağı çekmek doğru
olanları bozuyor.

## Elenen açıklamalar (8 yol, hepsi başarısız)

| Hipotez | Test | Sonuç |
|---|---|---|
| student_id sızıntısı | id~y kor | ❌ −0.009 |
| train-test örtüşme | 5-feature key | ❌ 0 ortak |
| Hedef lineer formül | OLS in-sample | ❌ R²=0.58, MSE 97 |
| Metin sayı-sızıntısı | regex \d | ❌ 0 satır |
| Metin-ton feature (poz-neg) | residual kor | ❌ ~0 (zaten blend'de) |
| Alt-kuyruk residual-fit | fold-içi LGBM | ❌ R²=−0.03 |
| Post-process (shift/isotonic/piecewise) | recency-rw | ❌ hepsi +zarar |
| Adversarial/recency EĞİTİM ağırlığı | tek-LGBM | ❌ uniform en iyi |

## Test dağılım kayması (bilinen, ama sömürülemez)

Adversarial AUC 0.65. `application_year`: test %62'si 2024-26 (train %35). Recency-weight'i
DEĞERLENDİRMEDE kullanıyoruz (doğru), ama EĞİTİMDE uniform en iyi (year-norm+uniform dersi).
Modeli test'e ağırlıklandırmak veri azlığı yüzünden zarar veriyor.

## Sonuç: 3 gerçekçi açıklama

1. **Lider public-overfit (en olası):** 80.41 *public* (60%). 5/gün submission ile post-process'i
   public LB'ye fit ettilerse private'da çökebilir. Bizim disiplinimiz (CV-otoritesi, public'e fit
   etmeme) tam bu riske karşı — private'da 82.37 onların 80.41'ini geçebilir.
2. Erişimimizde olmayan veri/sızıntı (bulamadık, sistematik aradık).
3. Daha iyi mimari (mümkün ama 12-model + tüm denemeler tavanı gösteriyor).

**Karar:** 82.37 bizim dürüst tavanımız. SUB-1 (catboost_full sigorta) + SUB-2 (12-model) private'a
karşı çeşitli. Public'in peşinden koşmak (winner's-curse) iki ekibin de reddettiği şey.
