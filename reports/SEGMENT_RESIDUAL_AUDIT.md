# SEGMENT & RESIDUAL DENETIMI — "yakalanamayan kotu satirlar" ogrenilebilir mi?

> **Soru (kullanici):** Blend'in yakalayamadigi (yuksek-residual) kotu satirlari segmentlere ayir;
> "kagittan kaplan / gizli cevher / dusuk-guven uyumsuzluk" kaliplari yeni ogrenilebilir sinyal mi?
> **Yontem:** Her kalip nested-durust test edildi — KRITIK olcut: kalibin blend RESIDUAL (y - blend_oof)
> ile korelasyonu. corr(kalip, residual)~0 ise blend (metin kanallari) ZATEN yakalamis = yeni degil.
> Karar metrigi nested rw-OOF (taban 82.2398). Public'e bakilmadi.
>
> **SONUC (TL;DR):** 3 segment-hipotezi + el-analizi + mevcut LOW_TAIL_LEVER = HEPSI ayni cevap:
> kotu satirlar SEGMENTLENEBILIR (belirsizlik ex-ante gorulur) ama DUZELTILEMEZ (yon ogrenilemez).
> Aleatoric (indirgenemez) gurultu. Metin/tabular sinyali doygun.

---

## 1. Kotu satir segment haritasi (en kotu %5, |residual|>=17.7)

| y-bandi | n | kotu-oran | yon (resid ort) | yorum |
|---|---|---|---|---|
| 0-40 (alt-kuyruk) | 142 | **%47** | **-18.6** (model FAZLA basiyor) | gercek ~40, model ~59 sanir |
| 40-56 | 806 | %15 | -10.1 (fazla) | yukari sapma |
| 56-70 | 2158 | %3 | -3.4 | model iyi |
| 70-85 | 3655 | %3 | +1.3 | model iyi |
| 85-95 | 1876 | %5 | +4.2 (AZ basiyor) | tavana yakin ama 100 degil |
| 95-100 | 1362 | %3 | +4.5 (az) | sansur kutlesi |

**Tek tutarli desen: MEAN-REVERSION.** Alt-kuyruk (y<56) model +11.5 sisirir (947 satir = toplam
MSE'nin %26.7'si); ust band (y>85) model -4 az basar. Bu GBDT'nin dogal MSE-optimal davranisi
(genis kosullu dagilimda uclari ortalamaya cekmek MSE'yi minimize eder), KUSUR DEGIL.

## 2. Neden duzeltilemez — belirsizlik gorulur AMA yon ogrenilemez

| Test | Sonuc | Yorum |
|---|---|---|
| Alt-kuyruk (y<56) ex-ante tespit | **AUC 0.90** | kim riskli GORULEBILIR |
| Riskli %10 dilimde gercek alt-kuyruk orani | **%48.7** | dilimin YARISI alt-kuyruk DEGIL -> duzeltme yanlis-pozitif uretir |
| |residual| ex-ante tahmin (heteroskedastik) | Spearman **+0.33** | belirsizlik buyuklugu gorulur |
| SIGNED residual ex-ante tahmin | OOF-R2 **-0.036** | YON ogrenilemez (yukari mi asagi mi: feature'larda YOK) |
| Signed-residual duzeltme uygulama | 82.24 -> **85.28** | duzeltme blend'i KOTULESTIRIR |

**Cekirdek:** "Bu ogrenci belirsiz" diyebiliyoruz (AUC 0.90 / Spearman 0.33) ama "y'si yukari mi asagi
mi sapacak" diyemiyoruz (signed R2 -0.036). Iki ozdes-gorunen ogrenci y'de 8.3 ayrisiyor = aleatoric.

## 3. Uc segment-hipotezi nested-durust test (workflow, 3 bagimsiz agent)

| Hipotez | corr(y) | corr(blend) | **corr(RESIDUAL)** | nested delta | karar |
|---|---|---|---|---|---|
| Kagittan-kaplan (yuksek-tab + ancak/fakat+elestiri) | -0.26 | -0.31 | **-0.004** | +0.72 (kotu) | zaten yakalanmis |
| Gizli-cevher (dusuk-tab + tutku/potansiyel/grit) | -0.18 | -0.21 | **-0.009** | -0.008 (gurultu) | zaten yakalanmis |
| Dusuk-guven (tabular-ici uyumsuzluk, 13 tanim) | -0.06 | +0.15 | **+0.009** | +1.08 (felaket) | zaten yakalanmis |

**Her hipotezde ayni imza:** corr(y) GUCLU ama corr(residual)~0. Sinyal GERCEK ama blend metin
kanallari (txt_ridge/xlmr/e5/mm) onu ZATEN emmis. Fishing-yanli en-guclu tanim bile |corr_resid|<0.022
(0.05 esiginin altinda). Kagittan-kaplan ozellikle TERS cikti: blend>=85 & celiski (n=1041) resid_mean
**+0.146** (model SISIRMIYOR, hatta hafif az basiyor) -> mentor celiski metni varsa model puani zaten
dusurmus. "100 basip cakilacaklar" tezi veride YOK.

## 4. Mevcut tutarli kanitlar (bu denetim onlari pekistirdi)
- **LOW_TAIL_LEVER.md:** alt-kuyruk sample-weight + post-correction -> ELENDI (test'te alt-kuyruk seyrek
  %2.4 vs train %4.9, recency-yogun; bias-variance net kaybeden).
- **CEILING_AUDIT.md Sonda-1:** hicbir modalite residual'i ogrenemiyor (e5/TF-IDF/tabular hepsi R2<0).
- **two_stage_p100 / hurdle (6-aci E):** classifier-times-regressor RED (AUC 0.90 ama yanlis-pozitif
  MSE'yi patlatir; bu denetim NEDENINI gosterdi: riskli dilimin yarisi alt-kuyruk DEGIL).

## KARAR
Kotu satirlar segmentlenebilir ama duzeltilemez = aleatoric gurultu tabani (~8.3). Metin/tabular
sinyali DOYGUN (3 segment-hipotezi + BERTurk corr-0.934 + tabular-residual R2<0 hepsi ayni yon).
**Yeni sinyal icin tek meslru kalan: GPU ortogonal fonksiyon sinifi (TabPFN-v2).** O da gelmezse
combo14 public 82.122 pratik tavandir; finalizasyon (en saglam 2 submission + repro + sunum) dogru hamle.

---

## 5. ISI HARITASINDAN CIKAN ANLAM — verinin yapisi (pasif "doygun" degil, AKTIF teshis)

Uc isi haritasi (reports/figs/corr_y_*.png) + takip analizleri verinin URETIM YAPISINI ortaya koydu:

### Bulgu A — y ADDITIVE-DOMINANT bir agirlikli-toplam + gurultu
- y ~ lineer(37 sayisal feature) **R2=0.572** (in-sample); GBDT/blend R2=0.62 -> etkilesimler sadece +0.05.
- **y'de karmasik etkilesim AZ; cogu sinyal additive.** Cekirdek formul (standardize katsayi):
  project_quality_score (+6.6) > technical_interview (+4.0) > communication (+2.2) > real_client_project
  (+1.8) > portfolio (+1.8) > github_repo (+1.3) > problem_solving (+1.3) > cloud (+1.2).
- Lineer-fit residual std=9.93, blend residual std=8.53 -> blend additive yapinin OTESINDE ~1.4 std cikarmis.

### Bulgu B — segment-ici korelasyon COKUYOR (range restriction)
- Genel corr(project_quality,y)=+0.54 AMA y-bandi icinde +0.08/+0.04/-0.01. Sinyal "y hangi BANDA duser"i
  belirliyor; bant-ICINDE kim daha yuksek feature'larda YOK. **Bu mean-reversion'in koku** (uclar belirsiz).
- Alt-kuyrukta (0-40) korelasyon nispeten canli (technical 0.25) ama n=142 (zayif/guvenilmez).

### Bulgu C — residual'da KACIRILAN SINYAL YOK (0.59 corr artefakt)
- corr_resid isi haritasinda yuksek degerler (project_quality @ 95-100 = -0.59) RANGE-RESTRICTION
  ARTEFAKTI: dar y-bandinda (std 1.6) resid ~ -blend ~ -feature mekanik. DURUST tum-veri OOF-R2 = **+0.0007**
  (her feature). Blend her feature'i zaten emmis.

### Bulgu D — gurultu tabani gercekten ALEATORIC
- top-8 feature uzayinda en-yakin %10 komsu (neredeyse ozdes profil): komsu-y |ayrisma|=11.19, std=8.69.
  Ayni feature'larla iki ogrenci y'de ~8.7 ayrisiyor = jeneratorun noise terimi, cikarilamaz.
- (kNN-tabani mutlak MSE tahmini GUVENILMEZ: 10k satir 8-dim'de bile seyrek -> tabani sisirir; prompt'un
  "ikiz testi bilgi vermiyor" uyarisi dogrulandi. Gecerli kanit lineer-residual + OOF-R2, kNN degil.)

### Bulgu E — additive-icgoruden HAMLE denendi -> RED
- "y additive ise saf lineer base GBDT'lere ortogonal olur" hipotezi: z_ridgepoly (lineer base) ZATEN var,
  corr(GBDT)=0.96, blend residual-corr=-0.017, blend +0.115 (zarar). **GBDT'ler additive yapiyi lineer
  kadar iyi yakaliyor** -> saf lineer ortogonal degil.

### NET ANLAM
Bu sentetik veri = **dominant ADDITIVE formul** (project_quality 6.6x + technical 4.0x + ...) + **guclu
rastgele gurultu (sigma~8.5-8.7)**. Blend additive yapiyi + az olan etkilesimi + metni ZATEN yakalamis;
kalan ~8.5 std jeneratorun noise'u = tanim geregi cikarilamaz. "Tavan" pasif degil: verinin URETIM
YAPISININ matematiksel sonucu. Yakalanabilir-ama-yakalanmamis sinyal HICBIR modalitede (tabular/metin/
lineer/GBDT/transformer) OOF-R2>0 vermiyor. Lider farki (~80.4 vs 82.24, 1.8 MSE) muhtemelen public-luck
veya farkli-feature kullanimi; nested-durust olarak combo14 public 82.122 bu bilgi-setinin tavani.

## 6. KULLANICI FIKRI — kotu-satirlara combinatorial-gating fonksiyonu (RISK-GATE) -> RED
Fikir: yuksek-residual ("kotu") satirlari ex-ante tespit et (risk skoru), SADECE onlara gating gibi
duzeltme uygula. NESTED + fold-safe test (risk-skoru hucre-disi HistGBR, esik+oran hucre-disi tr-rw min):
| Risk-gate hedefi | nested rw-OOF | delta |
|---|---|---|
| kotu satirda blend->MEAN(76.94) cek | 82.4738 | +0.234 |
| kotu satirda blend->ourteam_tf (en guclu) cek | 82.2862 | +0.046 |
| kotu satirda blend->lgbm_full_ht (robust) cek | 82.3632 | +0.123 |
| kotu satirda blend->base-medyan cek | 82.3709 | +0.131 |

**HEPSI RED.** Yapisal neden: kotu satir TESPIT edilebilir (risk AUC 0.90) ama YON ogrenilemez
(signed R2 -0.036). Bir satir "kotu" isaretlenince gercek y blend'in USTUNDE de ALTINDA da olabilir
(yari-yari). Herhangi bir hedefe cekince yarisini dogru yarisini YANLIS yone itiyor -> net zarar.
combinatorial-gating blend-BANDINA isler (yon belli: yuksek-tahmin->asagi) ama RISK-segmentine islemez
(yon rastgele). Kullanicinin sezgisi mantikli ama veri reddediyor: kotu-satir YONU feature'larda YOK.

## 7. KULLANICI FIKRI — "yonu de tahmin et" (residual isareti) -> zayif sinyal VAR ama MSE-kazanca DONMEZ
Yon (y>blend? = model az mi tahmin etti) cok-yonlu nested tahmin edildi (zengin feature: structured +
tum base OOF + blend + base-spread):
| Test | Sonuc |
|---|---|
| Yon siniflandirma OOF-AUC | **0.5595** (0.5-ustu: yon TAMAMEN tahmin edilemez DEGIL) |
| En-emin %20 dilimde yon dogrulugu | **%63.2** (sanstan yuksek) |
| Signed residual regresyon OOF-R2 | -0.036 (buyukluk tahmin edilemez) |
| Yon-duzeltme tam uygula | 85.27 (kotu) |
| Guvenilir %30 yarim-duzeltme | 83.24 (kotu) |
| TAM-NESTED yon-nudge (clf+delta+esik hucre-disi optimize) | **83.02 (+0.78, kotu)** |

**DUZELTME (onceki "yon ogrenilemez" fazla kesindi):** Yon zayif tahmin EDILEBILIR (AUC 0.56, emin-dilim
%63). AMA MSE-kazanca donmez: yon %63 dogru = %37 YANLIS; yanlis-yone itilen satirin hatasi KARESEL
buyur (MSE), dogru %63'un kazanci %37'nin karesel cezasini KARSILAMAZ. Ustelik buyukluk de bilinmiyor
(R2 -0.036). MSE-esigi ~AUC 0.60; bu veride yon AUC 0.56 (esik-alti). Bu, TUM duzeltme/gating/correction
ailesinin (11 confidence-correction + risk-gate + yon-nudge) kok-RED nedeni: zayif sinyaller VAR ama
hicbiri MSE-esigini asmiyor. Karesel ceza zayif-sinyali yutuyor.

## 8. KULLANICI FIKRI — "sadece ASIRI-emin yon tahminlerini kullan" -> zarar sifirlandi ama kazanc YOK
Guven-dilimi arttikca yon-dogrulugu (OOF):
| en-emin % | n | yon dogru | ort|resid| |
|---|---|---|---|
| %10 | 1000 | %72.3 | 3.66 |
| %5 | 500 | %85.8 | 1.82 |
| %2 | 200 | **%99.0** | **0.42** |
| %1 | 100 | %100 | 0.26 |

NESTED uygula (sadece asiri-emin uc, q+delta hucre-disi optimize):
- tum satir yon-nudge: +0.78 | top%5-0.5: +0.029 | top%2-0.5 buyuk-nudge: **+0.003 (~SIFIR)**

**KALIBRASYON PARADOKSU (senin fikrin duvari her iki yonden kanitladi):** Guven dar tutulunca zarar
+0.78 -> +0.003'e dustu (yon %99 dogru artik). AMA tam-sifiri gecemez cunku en-emin satirlarda hata
ZATEN mikroskobik (ort|resid| 0.42). Model "eminim" dedigi yer = "zaten hatasizim" dedigi yer (guven
ve hata TERS-BAGLI). Duzeltilecek buyuk hata = guvenin DUSUK oldugu yer. Yon sinyali GERCEK (uctaki
%99) ama MSE-kazanca donmez: kazanilabilir-hata ile guven ters orantili. risk-gate (hata-segmenti, yon
yok) + bu (guven-segmenti, hata yok) = duvari her iki yonden test etti. Combinatorial-gating blend-banda
isler cunku orada hem yon belli HEM hata var; risk/guven segmentinde ikisi ayni anda olmuyor.

## 9. KULLANICI FIKRI — "train'deki sacmalik satirlara (model 70 dedi gercek 0) benzer test satirlari ara" -> RED
TRAIN'de sacmalik satirlar GERCEK: 32 satir (blend>=65 ama y<=45), model ort 69.7 dedi gercek 38.8 (+30.9 sisti).
Test'te bu profile benzer ara + asagi cek (NESTED: sacma-profili hucre-disi ogren, esik+cekme hucre-disi optimize):
| Test | Sonuc |
|---|---|
| Sacma-profiline en yakin %10 TRAIN satiri ort y | **74.4** (genel 76.9; NORMAL, dusuk DEGIL) |
| ^ ayni satirlarda residual | -0.5 (~0; model sistematik sismiyor) |
| Test'te "supheli" (blend>=65 + sacma-profile yakin) | 805 satir AMA cogu normal-y |
| NESTED sacma-profil duzeltme uygula | **82.5409 (+0.30, RED)** |

**KOK NEDEN:** Sacmalik y'nin nedeni feature'larda YOK. 2 kisi feature-ozdes -> biri 38.8 ikizi 74.4 (fark
= jeneratorun rastgele gurultu terimi sigma~8.7). Sacma-profiline benzeyen test satirlarinin %90'i normal-y;
asagi cekince yanlis cezalandi -> +0.30. Bu, oturumdaki 4. ayni-duvar fikri (risk-gate +0.23 / yon-nudge +0.78 /
guven-esigi +0.003 / sacma-profil +0.30): hepsi "kotu-satir tespit+duzelt", hepsi RED, AYNI kok-neden:
kotu satirlar GERCEK ama feature'larda gorunmez (aleatoric, jenerator-noise). combinatorial 82.122 saglam tavan.
