# ALT-KUYRUK (y≤50) KALDIRACLARI — 2 meşru deneme, dağılım-sadık CV ile **ELENDİ**

> **Amac:** Hata bütçesinin en yoğun tek dilimi olan alt kuyruğu (y≤50: nüfusun %5'i, klasik MSE
> bütçesinin %18.7'si, recency-ağırlıklı bütçenin %22.4'ü; bias −14.0 = model bu öğrencileri yukarı
> tahmin ediyor) iyileştirmek. **Karar metriği = recency-weighted OOF (rw-OOF).** Public'e bakılmadı.
>
> **SONUC (TL;DR):** İki CV-güvenli kaldıraç da **ELENDİ**. Alt-kuyruk hatası model kusuru değil;
> (a) MSE-optimal mean-reversion (geniş koşullu dağılım), (b) test'te alt kuyruk train'den seyrek
> (%2.4 vs %4.9) → dağılımı bozan müdahaleler rw-OOF'u **yükseltiyor**, (c) metin sinyali zaten tükenmiş.

---

## Teşhis — alt kuyruk öğrenilebilir ama zayıf sinyalli

| Gösterge | Değer | Yorum |
|---|---|---|
| y≤50 nüfus payı | %5.0 (497 satır) | seyrek |
| y≤50 rw hata payı | **%22.4** | hata bütçesinde orantısız |
| bias (y−pred) | **−14.0** | model +14 yukarı tahmin (mean-reversion) |
| alt-kuyruk-içi corr(y,pred) | 0.415 | zayıf ayrışma |
| feature ayrışması (technical_interview) | −0.74σ | sinyal var ama zayıf |
| **test'te alt kuyruk** | **%2.4** (train %4.9) | test recency-yoğun → alt kuyruk daha seyrek |

## Kaldıraç A — sample-weight (alt-kuyruk vurgusu) → ELENDİ

`sample_weight = 1 + k·1[y≤50]` (sentetik veri YOK, dağılım sabit → sızıntısız). repeat-0 lgbm_full:

| k | rw-OOF | alt-kuyruk MSE | üst MSE (y>50) |
|---|---|---|---|
| **0 (mevcut)** | **88.48** | 307.2 | **66.5** |
| 1 | 88.82 | 281.4 | 68.4 |
| 2 | 90.00 | 273.2 | 69.7 |
| 4 | 91.00 | 271.6 | 70.8 |

**Klasik bias-variance takası, net kaybeden.** Alt-kuyruk MSE düşüyor (307→272) ama üst band (nüfusun
%95'i) bozuluyor (66.5→70.8) → net rw-OOF **artıyor** (88.48→90.0+). Hiçbir k karar metriğini düşürmüyor
→ paired-test'e bile gerek kalmadan reddedildi. Sebep: test alt-kuyrukta seyrek; oraya optimize etmek
test'in gerçek dağılımından uzaklaşmak demek (rw-OOF tam bunu cezalandırır).

## Kaldıraç B — regex/olumsuz-kalıp özelliği (az örnekli olanlara) → ELENDİ (redundant)

Alt kuyrukta ayırt edici olumsuz kalıplar GERÇEKTEN var (log-odds): *eksiklikler, zorluklar, sınırlı,
gelişme, potansiyelin, çaba, yola, deneyimin* (üst kuyrukta: *mükemmel, olağanüstü, başarı*). Regex
sayacı kuruldu:

| Ölçüt | Değer | Yorum |
|---|---|---|
| corr(neg_regex, y) | −0.170 | sinyal y ile korele |
| **corr(neg_regex, blend_residual)** | **−0.002** | **sinyal ZATEN yakalanmış** |
| corr(neg_regex, residual \| y≤60) | +0.084 | alt kuyrukta bile ihmal edilebilir |
| corr(neg_regex, pos_minus_neg lexicon) | −0.322 | mevcut lexicon ile örtüşüyor |

**Mevcut `lexicon` (n_neg, pos_minus_neg) + `txt_ridge` (TF-IDF→Ridge) + `mm` (XLM-R-large) bu olumsuz
kalıpları çoktan işliyor.** Regex residual'da iz bırakmıyor → yeni bilgi yok → eklemek redundant kolon
(Occam reddeder). mm zaten alt bantta en çok yardım eden üyeydi (−2.6 MSE) — metin modalitesi tükenmiş.

## Kaldıraç C — iki-aşamalı alt-kuyruk düzeltmesi (p100 analoğu) → ELENDİ (Bayes-tutarlılık)

En hedefli saldırı: fold-safe P(y≤50) sınıflandırıcı (LGBM-binary, lgbm_full matrisi, repeat-0 OOF)
+ `pred' = clip(blend − β·P)` düzeltmesi. **Oracle muazzam:** y≤50 rw-ağırlık payı %6.55 × bias²
(14.03²) ≈ **12.9 rw puanı** (mükemmel bilgiyle 84.2→71.3!). Sınıflandırıcı güçlü: **AUC 0.910**.

| β | rw-OOF | delta |
|---|---|---|
| 0 (mevcut) | 84.2393 | — |
| 2 | 84.2723 | +0.033 |
| 4 | 84.4233 | +0.184 |
| 8 | 85.0791 | +0.840 |
| 14 | 86.9474 | +2.708 |

**Her β>0 kötüleştiriyor — optimum tam sıfır düzeltme.** Yorum: blend zaten E[y|x] = P·μ_düşük +
(1−P)·μ_yüksek karışımını içeriyor; sınıflandırıcının P'si AYNI feature'lardan geldiği için yüksek-P
satırları blend zaten düşük tahmin etmiş. −14 bias'ı yaratan satırlar **ex-ante ayırt edilemeyen**
sürprizler — bias düzeltilebilir bir hata değil, kısmi-bilgiyle MSE-optimal davranışın kendisi.
(p100 üst-kütle analoğunda da aynı ders: oracle −3.57'nin sadece %5.6'sı gerçekleşmişti; burada %0.)

## Augment (sentetik balance) — DENENMEDİ (ilkesel ret)

SMOTE/mixup/LLM ile sentetik alt-kuyruk: sürekli hedefte uydurma (x,y) çiftleri üretir → sızıntısız
yapılamaz, CLAUDE.md sıfır-overfit ilkesini ihlal eder, rw-OOF sahte iyileşir / private patlar. Kaldıraç
A zaten dağılım bozmanın net zarar verdiğini gösterdi; augment aynı zararı + uydurma-etiket riskini ekler.

## SONUC — alt kuyruk yapısal; müdahale edilmedi

Alt-kuyruk hatası **veri-kaynaklı ve MSE-optimal**: seyrek + geniş koşullu dağılım → mean-reversion
doğru davranış; metin sinyali tükenmiş; dağılım bozmak rw'yi yükseltiyor. **SUB-1/SUB-2 son hâli
korundu** (blend rw-OOF 84.24, e5+mm). Bu 3 kaldıracın reddi jüri için bilimsel-disiplin kanıtı
("denedik, dağılım-sadık CV reddetti").

## EK — blend residual artık SAF GÜRÜLTÜ (genel kapanış kanıtı)

Alt kuyruğun ötesinde "başka ne denenebilir" sorusuna objektif cevap: **blend residual'ında öğrenilebilir
yapı kalmadı.** Bir GBDT (HistGBR), tüm sayısal+yıl feature'larıyla blend residual'ını repeat-0 OOF
tahmin etmeye çalıştı:

| Ölçüt | Değer | Yorum |
|---|---|---|
| residual OOF tahmin R² | **−0.021** | negatif → yapı YOK (gürültüyü ezberlemekten kötü) |
| max(\|corr(feature, residual)\|) | 0.042 (project_quality) | gürültü seviyesi |

→ mm/e5 kalan ortogonal sinyali (metin modalitesi + neural sınıf) çekip aldı; **geriye açıklanabilir
sinyal kalmadı**. Yeni bir base model (asimetrik-loss mm dahil) blend'i anlamlı düşüremez — çekilecek
sinyal yok. Forensics "noise-floor" tezi artık BLEND seviyesinde de doğrulandı. Daha fazla deneme
negatif beklenen değerli (overfit riski > kazanç umudu) → CLAUDE.md "marjinal kazancı reddet" + "biraz
daha deneyeyim tuzağı" ilkeleriyle **DUR**.

### Artefakt / kanıt
- repeat-0 sample-weight taraması + regex-residual analizi + blend-residual-öğrenilebilirlik testi (bu rapor).
- Karar: hiçbir kaldıraç rw-OOF'u düşürmedi → kalıcı artefakt/blend üyesi ÜRETİLMEDİ (Occam). SUB-1/SUB-2 korundu.
