# FORENSICS — Sentetik Hedef Geri-Muhendislik (TIER-1)

> **Amac:** `career_success_score`'un nasil URETILDIGINI cozerek 85.4945 rw-OOF duvarini kiracak
> YAPISAL bir somuru bulmak. **Karar metrigi = recency-weighted nested OOF-MSE.** Kabul kapisi =
> 85.4945 − 0.25·3.0238 = **84.7385**. Public LB'ye BAKILMADI.
>
> **SONUC (TL;DR):** Duvar **noise floor**'dur, modelleme eksigi DEGIL. Censoring gercek ama
> somurulabilir degil (oracle tavani yalniz −3.57 rw). Tum yeni lever'ler (zengin metin dahil)
> kapidan UZAK kaldi. Iki bagimsiz GBDT ailesi residual'in ~%97'sinde hemfikir → kalan = gurultu.
> **Tek dogrulanan acik kapi:** censored-aware TRAINING objective ve transformer-metin denenmedi —
> ama ikisi de oracle tavani (−3.57) ile sinirli, kapidan (−0.756) cok uzakta beklenir.
>
> Tum bulgular adversarial-verify edildi (4 iddia, her biri curutmeye calisan skeptik ile). Iki
> iddia (censoring, no-exploit) ayakta kaldi; ikisi (irreducible-floor overreach, "metin redundant")
> DUZELTILDI — duzeltmeler bu rapora islendi ve C3'un onerdigi lever (zengin metin) FIILEN denendi.

---

## A. CENSORING VERDICT — **KESIN: hedef = clip(latent, 0, 100), tavanda sag-sansurlu**

| Kanit | Deger | Yorum |
|---|---|---|
| ==100 kutle | 773 (%7.73) | tavanda kesin yigin |
| [99.99, 100) | **2 satir** | tavan-alti cukur → sansur imzasi (kutle 100'e atlamis) |
| [99.5, 100) | 47 satir | sansur-oncesi seyrek bolge |
| ==0 | 1 satir | alt sinir pratikte deginilmemis |
| 0.01-grid uzerinde | %100 | hedef 2-ondalik kuantize (sentetik uretim izi) |
| Tobit MLE | σ=10.4, E[y\|sansurlu]=96.5 | latent>100 acikca var; clip ile 100'e basilmis |
| LGBM num-only residual @==100 | **+6.23** | model 100'lerin ALTINI tahmin → latent gercekte >100 |

**Modelleme imasi:** sansur GERCEK ama **somurulemez** (asagidaki E bolumu). Sebep: tavana yakin
heteroskedastik gurultu DUSUK (σ≈4.7) ve P(y=100) siniflandirici guclu (AUC 0.958) olsa bile
yanlis-pozitifleri 100'e itmek karesel ceza yiyor; net kazanc oracle tavanin (~−3.57 rw) cok
altinda kaliyor.

## B. NOISE FLOOR — duvar pratik bir tavandir (DUZELTILMIS iddia)

> ⚠️ **Adversarial duzeltme:** Ilk taslak "85.49 = indirgenemez floor, R² plato" diyordu. Bu
> OVERREACH idi: kapasite zinciri boyunca uw-MSE **dusmeye devam ediyor** (88.16 num → 82.92 +metin
> → 78.51 +kategorik → 75.75 blend; R² 0.62→0.67 YUKSELIYOR, plato DEGIL). Dolayisiyla "matematiksel
> olarak indirgenemez" diyemeyiz. Dogru ifade asagida.

- **Inter-model floor (en saglam kanit):** iki BAGIMSIZ GBDT ailesi (LGBM vs CatBoost) OOF'lari
  **std=1.53** icinde hemfikir; residual std=8.72. → residual varyansinin **~%96.9'u model-bagimsiz**
  (iki farkli aile ayni yere yakinsiyor) = denenen feature ailesi icin pratik gurultu tabani.
- **Gurultu yasasi:** interior residual ~Gauss (skew −0.27) ama **agir kuyruklu** (excess kurt ~1.0)
  ve **guclu heteroskedastik**: `res_std ≈ 13.58 − 0.10·latent` (dusuk skorda σ≈11.9, yuksekte ≈4.7).
  Heteroskedastisite MSE altinda SOMURU DEGIL (kosullu ortalama hala optimal) ama floor'u anlatir.
- **rw > uw farki bir DAGILIM ozelligidir, sizinti degil:** test recency-yogun; hedef varyansi
  test-agirlikli 265.7 vs train 230.6. uw-MSE × (265.7/230.6)=87.28 ≈ rw 85.49 (~%2 icinde) →
  fark neredeyse tamamen varyans-siserlemesiyle aciklaniyor.

**Ima:** Duvar, **denenen yapisal-feature + TF-IDF-metin ailesi icin** pratik bir tavandir; kalan
residual bu ailede agirlikla gurultudur. "Hic kirilmaz" demiyoruz — yeni bir MODALITE (asagida)
teorik olarak residual'a girebilir; ama denenen her sey gurultu bandinda kaldi.

## C. LATENT FORMUL — agirlikli, lineer+monoton, TEMIZ-agirlik YOK

- **Lineer+monoton tek-index:** Spearman(latent_hat, y)=0.728 ≈ Pearson 0.722 → guclu nonlineer
  donusum YOK; latent ≈ feature'larin agirlikli toplami + hafif konveks `project_quality²` (+0.019 R²).
- **Temiz/yuvarlak agirlik YOK:** ham OLS katsayilari dagisik (pq 0.302, tech_interview 0.198, ...),
  skor-agirlik CV=1.04 (esit-agirlik DEGIL). "Ortalama-of-scores" gibi basit bir formul izi yok →
  ozel-agirlikli feature kurmak GBDT'nin zaten yaptigini tekrar eder.
- **En guclu lineer katkilar:** `project_quality_score` ≫ `technical_interview_score` >
  `communication_score`, `real_client_project_count`, `cgpa`, `github_repo_count`.
- **Segment determinizmi YOK:** target_role ortalama 74–80, department 76.4–77.2 (neredeyse duz) →
  deterministik kova yok.
- **Deterministik alt-kume YOK:** |resid|<0.5 fazlasi (%8.38 vs %4.6 Gauss) TAMAMEN ust pred-bin'de
  (==100 sansur kutlesi + dusuk-gurultulu yuksek-latent); interior near0 y-degerleri SUREKLI yayilmis
  (38.7, 39.7, 40.5, ...), grid-yigini DEGIL → formul izi degil, gurultu kuyrugu.

## D. METIN ROLU — ETKILESIMLI, redundant DEGIL (C3 TERSINE CEVRILDI)

> ⚠️ **Adversarial duzeltme + FIILI test:** Ilk taslak "metin numerikle REDUNDANT, BERT yardim
> etmez" diyordu. Skeptik bunu hakli olarak curuttu:

- metin → num-sonrasi-residual **lineer** R² = −0.002 (ZERO). AMA bu **additivite** olcer, redundancy
  DEGIL: joint num+metin (82.92) hem num-only'den (88.16, −5.24) hem additive'den (87.30, −4.38) iyi.
  Redundant bir feature GBDT'ye −5.24 veremez → **metin numerigin TASIMADIGI bilgi tasiyor**, degeri
  **etkilesimli** (metin, numerigin hedefe esleme bicimini module ediyor).
- **Headroom GERCEK:** zengin TF-IDF (word 1-3 + char 2-6, 80k) num tabaninda mevcut txt_ridge USTUNE
  **−0.87 uw marjinal** verdi (corr 0.946 olmasina ragmen) → mevcut txt_ridge metni DOYURMAMIS.
- **C3'un imasi:** BERT/transformer-metin ELENMEZ; etkilesim-tasiyan, additive-olmayan sinyal tam da
  daha zengin bir encoder'in yardim EDEBILECEGI rejimdir. Bu yuzden lever'i FIILEN denedik (asagida).

## E. SOMURU TESTLERI — hepsi fold-safe; HICBIRI kapiyi gecmedi

| Somuru | Yontem | Sonuc (rw, fold-safe) | Karar |
|---|---|---|---|
| **Zengin metin (txt_rich)** | word(1-3)+char(2-6) nested-OOF → blend havuzu (C3 lever) | **85.4945 → 85.4116 (−0.083)** | **ELENDI** — gurultu bandi (kapi −0.756). txt_ridge'i 0.099 agirlikla degistirdi; net ~0. |
| Threshold-push (pred≥thr→100) | tum esikler | her esikte MSE ARTAR (precision max 0.84 @thr=98) | ELENDI |
| Two-stage P(y=100) (gecmis) | LGBM-binary AUC 0.958, nested α | −0.20 (kapidan uzak) | ELENDI (LEVERS) |
| Tobit E[min(latent,100)] | censored-normal MLE | in-sample MSE 103 > GBDT 76 (lineer-latent zayif) | ELENDI |
| Isotonic recalib | nested fold-safe | in-sample −1.35 AMA nested **+0.42** (kotu) | ELENDI (overfit'ti) |
| Affine recalib | a=1.01 b=−0.85 | Δ −0.006 (zaten kalibre) | ELENDI |
| 3. GBDT (histgbr), SVD-metin | gecmis lever'ler | agirlik ~0 / +0.02 | ELENDI (LEVERS) |

**Oracle tavani (ulasilmaz ust sinir):** gercek ==100'leri 100'e zorla → yalniz **−3.57 rw** (−3.07 uw).
Herhangi bir 100-duzeltmesi bu degerle sinirli; kapi −0.756 ister, en iyi gercekci deneme (−0.20)
bunun cok altinda. *(Not: ilk taslaktaki −3.75 figuru zayif num+text uw tabanina aitti; proje blend
uzerinde rw-oracle −3.57'dir.)*

## MODELLEME IMASI / KARAR

1. **Censored objective?** GBDT post-hoc duzeltmeleri tukendi. Tek denenmemis hal: censored/quantile
   TRAINING loss veya 100-kutlesini fit-sirasinda upweight. **Beklenti dusuk** — oracle tavani −3.57,
   kapi −0.756; sansur-yalniz herhangi bir yontem bu tavanla sinirli ve gercekci hali cok daha az verir.
   Yine de tek "tam test edilmemis" sansur kapisi; istenirse 1 fold-safe deneme degerinde.
2. **Transformer metin (BERT)?** C3 reversal ile ELENMEDI — etkilesim-tasiyan headroom var (txt_rich
   −0.87 uw bunu gosterdi) ama **blend'de yine gurultu bandinda** kaldi (−0.083). BERT, txt_rich'ten
   anlamli farkli/daha derin sinyal cikarirsa kapiyi gecebilir; aksi halde ayni duvara carpar. **Orta
   risk, tek somut acik lever** — denenebilir ama beklenti temkinli (txt_rich kanaati: metin sinyali
   GBDT etkilesimlerince buyuk olcude zaten yakalaniyor).
3. **Belirli feature donusumu / near-exact reconstruction?** YOK. Latent dagisik-agirlikli ve
   GBDT'nin yakaladigi; temiz formul/deterministik alt-kume yok.

**Net:** Inkremental lever'ler GERCEKTEN tukendi; duvar bu feature+TF-IDF ailesinin gurultu tabani.
Geriye kalan iki teorik kapi (censored-train, transformer-metin) oracle/headroom sinirlariyla
cevrili ve **kapiyi gecmeleri olasi degil** — ama "denenmedi" durustlugu icin acikca belgelenir.

---

### Uretilen artefaktlar / kanit
- Analiz: `src/forensics.py` … `forensics7.py` (+ `reports/forensics_dump{,2..7}.json`).
- Lever (denendi, ELENDI): `src/text_rich.py` → `artifacts/{oof,test}_txt_rich.npy`,
  `reports/model_scores.csv` (txt_rich satiri: standalone rw 162.27; blend +txt_rich nested rw 85.41).
- Adversarial verify: 4 iddia skeptik-refute; C2/C3 duzeltmeleri bu rapora islendi.
- **Nihai blend DEGISMEDI** (85.4945); txt_rich kapidan (84.7385) uzak → CANDIDATE_POOL'a EKLENMEDI
  (Occam + sifir-overfit). 2 final aynen korunur (SUB-1 catboost_full, SUB-2 blend).
