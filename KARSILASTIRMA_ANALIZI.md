# İki Çözümün Karşılaştırmalı Analizi — Tuna (STACK_84) vs Ahmet (CALISMA_GUNLUGU)

> Datathon 2026 — `career_success_score` (0–100) regresyon, metrik **MSE**.
> Bu doküman iki bağımsız çözümü **aynı pusulayla** (recency-weighted OOF, aynı `folds.parquet`)
> ölçer, mimari/metodoloji farklarını tartar ve takım birleşmesinin neden RED edildiğini
> sayısal kanıtla açıklar. Kaynaklar: `STACK_84.md`, `CALISMA_GUNLUGU.md`, repo artefaktları,
> git geçmişi (commit `c6b7bb5`), ve canlı ölçüm (`src/probe_compare_solutions.py`).

---

## 0. TL;DR (Yönetici Özeti)

| | **Tuna (SUB-2 blend)** | **Ahmet (ourteam_oof)** |
|---|---|---|
| **rw-OOF (aynı metrik, aynı y)** | 84.0212 | **83.5577** |
| unweighted CV | 74.4055 ± 2.83 | 74.0239 ± 2.43 |
| Public LB | **84.13** (gönderildi, gap +0.11 yeşil) | 82.999 (team_blend_opt, 46 sub'dan en-iyi) |
| Fold şeması | RepeatedStratified 5×3 | KFold(10) |
| Felsefe | **az ama kanıtlı** (paired-gate, ~40 RED) | **çok & çeşitli** (45 sürüm, geniş keşif) |
| Final akıbeti | **SEÇİLDİ** (SUB-2) | birleşme RED (paired 11/15, p=0.137) |

**İki cümlede:** Ahmet'in çözümü tek-model olarak rw-OOF'ta **0.46 daha iyi** ve birleştirildiğinde
karışım rw-OOF **82.44**'e iner (−1.58, gerçek bir kazanç). **Ama** bu kazanç (a) paired-anlamlılık
eşiğini geçemedi, (b) farklı fold şeması yüzünden iyimser-yanlı olabilir, (c) team_blend_opt 46
denemenin public-en-iyisi olduğu için winner's-curse riski taşıyor. Disiplin (CV-otoritesi + paired-gate)
birleşmeyi reddetti ve CV-doğrulanmış SUB-2 (84.02) korundu. **Bu RED, bir hata değil; iki çözümün
felsefe farkının tam da test edildiği yerdir.**

---

## 1. Aynı Pusulayla Ölçüm — "Hangisi Daha İyi?"

Adil kıyas için ikisini de **aynı metrik + aynı hedef vektörü** ile ölçtüm (canlı, `probe_compare_solutions.py`):

```
BIZIM blend (SUB-2)   rw-OOF = 84.0212   unw_cv = 74.4055 ± 2.8289
ARKADAS (ourteam_oof) rw-OOF = 83.5577   unw_cv = 74.0239 ± 2.4311
corr(bizim, arkadas)  = 0.9844
```

**Bulgular:**
- Ahmet'in tek-çözümü rw-OOF'ta **0.46 daha iyi** (83.56 vs 84.02) ve daha düşük CV-std (daha stabil görünüyor).
- **AMA `corr = 0.984`** — iki çözüm neredeyse aynı tahminleri yapıyor. Bu kritik: yüksek korelasyon, birleşmenin teorik kazanç tavanını sınırlar (ortogonal olsalardı kazanç çok daha büyük olurdu).
- Tek-model üstünlüğü gerçek ama küçük; CV-std'lerin örtüşme bandında (±2.4–2.8), yani "kesinlikle daha iyi" demek için paired test şart.

---

## 2. Mimari & Felsefe Farkı

### 2.1 Tuna (STACK_84) — "Az ama Kanıtlı"
- **Taban:** 51-feature anchor (37 sayısal + 2 ham yıl + 7 missing-flag + 5 **native-kategorik**), LightGBM native NaN/kategorik → sızıntı yapısal imkansız.
- **Metin:** TF-IDF→Ridge nested-OOF (`txt_ridge`) + FROZEN multilingual-e5-large→Ridge (`e5_ridge`).
- **Robust loss:** Huber(α=5) varyantları (`lgbm_full_h`, `lgbm_full_ht` sıkı-reg) — alt-kuyruk sürprizlerine karşı.
- **Neural:** XLM-R-large + tabular NN multimodal (`mm`).
- **Blend:** `Ridge(positive)` + intercept, **recency sample-weight**, **nested meta-CV** (dürüst, meta-overfit'siz).
- **Kabul kapısı:** her ekleme **paired-test** (≥12/15 hücre + p<0.01 + bootstrap CI<0). ~40 teknik ailesi denenip RED edildi (bkz §5).
- **Final:** SUB-1 = `catboost_full` (safe, bit-repro), SUB-2 = 10-model blend (en-iyi CV).

### 2.2 Ahmet (CALISMA_GUNLUGU) — "Çok & Çeşitli Keşif"
- **Taban:** 170+ feature, agresif feature mühendisliği. **İki damgası:**
  - 🌟 **Segment-yıl Target Encoding** (rol/tier/hobby × yıl, hiyerarşik smoothing): tekil **−1.4 MSE** (89.94→88.53). Üreticinin segment-bazlı yıl trendini yakalıyor (Cybersecurity −3.98 vs DevOps −0.86).
  - 🌟 **Kohort-göreli z-skorları** (key5 feature'ın rol×yıl içi z): tekil **−0.45**. "Ham skor değil kohort-içi göreli yer önemli" tezi.
- **Modeller:** LGBM(tuned) + CatBoost(GPU,tuned) + **XGBoost(bilerek UNTUNED)** + MLP + PyTorch TabNN.
- **Metin bataryası (5+ model):** BERTurk×3 (en iyi 129.6), mDeBERTa (130.6), **XLM-R-large (126.1, en iyi metin)**, MiniLM. Mean-pool + hedef standardizasyon mimari dersi (145→129.6).
- **Blend:** NNLS + Ridge-meta + yıl-affine kalibrasyon + **fit'siz sabit-oran** (en güvenilir bulunan).
- **Tuning:** Optuna (tune_v4/v13/v30). "Ağır regülarizasyon kazanıyor" + "yıl-norm+uniform rejimi" keşfi.

### 2.3 Ortak DNA (bağımsız ama aynı pusulayı buldular)
Her iki ekip de **bağımsız olarak** şu kritik kararlara ulaştı — çözümlerin sağlamlığının en güçlü işareti:
- **Recency-weighted OOF** ana karar metriği (zaman kayması: test %62 2024-26, train uniform).
- **Public LB'ye fit etmeme** disiplini.
- **Çeşitlilik > bireysel tuning** (Ahmet'in v8 dersi = Tuna'nın ortogonallik vurgusu).
- **XLM-R-large** metin için en güçlü tek model (ikisi de aynı sonuca vardı).
- **Ağır regülarizasyon** kazanıyor (Ahmet: l2=5-8; Tuna: `lgbm_full_ht` leaves=15/mc=80).

---

## 3. Tartı: Güçlü ve Zayıf Yönler

### Tuna'nın güçlü yönleri
- **İstatistiksel disiplin:** her karar paired-test + bootstrap CI ile; "repeat-0'da parlak, full-15'te buharlaşan" seçim-gürültüsü tuzağını sistematik yakalıyor (`f_tshaped_std`: −0.194 → +0.087 örneği).
- **Reproducibility:** SUB-1 bit-özdeş; tüm pipeline `SEED=42`, deterministik, pinli requirements.
- **Tavan denetimi:** 4 bağımsız sonda + cross-class residual corr=0.906 ile "bilgi-seti limiti" kanıtlandı.
- **Forward-chaining teyit:** rw-OOF sıralama kararları temporal split'te Spearman +0.964 ile doğrulandı.

### Tuna'nın zayıf yönleri
- **Feature mühendisliği muhafazakârlığı:** hiçbir FE grubu gate'i geçmedi → Ahmet'in segment-yıl TE (−1.4) ve kohort-z (−0.45) gibi yapısal keşifleri kaçırmış olabilir. (Not: Tuna bunları arkadaş-tabanında test etti, kendi güçlü tabanında +0.018 verdi → telafi-edilmiş.)
- Tek-model rw-OOF Ahmet'ten 0.46 geride.

### Ahmet'in güçlü yönleri
- **Yapısal keşifler:** segment-yıl TE + kohort-z, üreticinin sentetik kohort-yapısını yakalayan gerçek sinyal (toplam ~−1.85 tekil katkı). Bu, problemin doğasına dair en derin içgörü.
- **Geniş metin bataryası:** 5+ transformer modeli, mimari incelikleri (mean-pool + hedef-std).
- **Tek-model rw-OOF daha iyi (83.56) ve daha stabil (std 2.43).**

### Ahmet'in zayıf yönleri
- **Fold şeması uyumsuz:** KFold(10), Tuna'nın RepeatedStratified 5×3'ü değil → OOF'lar fold-hizasız; rw-OOF iyimser-yanlı olabilir (10-fold daha çok train görür).
- **Winner's-curse riski:** `team_blend_opt` **46 submission'dan public-en-iyisi** seçilmiş → public-overfit şüphesi (günlük §5'teki kendi "ağırlık-fit kazançları LB'ye taşınmıyor" dersiyle çelişir).
- **Çift-sayım/karmaşıklık:** mm2 sub2-farkında olunca çift-sayım nedeniyle atılmış; 45 sürümlük dallanma reproducibility'yi zorlaştırıyor.

---

## 4. Takım Birleşmesi: Sayısal Gerçek vs Karar (Analizin Kalbi)

Bu, iki çözümün **fiilen birleştirildiği** ve felsefe farkının test edildiği yer. Canlı ölçüm:

```
=== Basit sabit-oran karışımlar (fit YOK) ===
  0.45*bizim + 0.55*arkadas   rw-OOF = 82.4400   (bizim 84.02'ye göre −1.58)  ← en iyi
  0.50*bizim + 0.50*arkadas   rw-OOF = 82.4497   (−1.57)
=== Öğrenilmiş 2-model ridge_pos (nested) ===
  ridge_pos(bizim, arkadas)   nested rw-OOF = 82.4966   (−1.52)
```

**Görünürdeki çelişki:** Birleşme rw-OOF'u **84.02 → ~82.44**'e indiriyor (−1.58). Bu BÜYÜK ve gerçek
bir kazanç gibi görünüyor. Peki neden RED edildi?

**Commit `c6b7bb5`'in gerekçesi (üç bağımsız kırmızı bayrak):**

1. **Paired-anlamlılık eşiği geçilemedi.** 55/45 karışım paired CV'de **11/15 hücre** iyileşti
   (eşik ≥12), **p=0.137** (eşik <0.01). Yani −1.58'lik ortalama kazanç, hücreler arası tutarsız —
   istatistiksel olarak gürültüden ayrışmıyor. (Kıyas: e5/mm 15/15, p<1e-4 ile geçmişti.)

2. **Fold şeması hizasız.** Ahmet KFold(10), Tuna RepeatedStratified 5×3 kullanıyor. Ahmet'in OOF'u
   bizim folds.parquet'e tam hizalı değil → rw-OOF iyimser-yanlı (10-fold modeli her satır için daha
   çok komşu görmüş). 82.44'lük rakam bu yüzden şişmiş olabilir.

3. **Winner's-curse.** `team_blend_opt` 46 submission içinden public-LB-en-iyisi (82.999) seçilmiş.
   Bu, ikisinin de reddettiği "public'in peşinden koşma" tuzağının ta kendisi. Public 82.999 ile
   CV ~82.44 arasındaki uyum bile, public'e bakılarak seçildiği için kalibrasyon kanıtı sayılmaz.

4. **Korelasyon tavanı.** corr=0.984 → iki çözüm zaten neredeyse aynı şeyi öğreniyor. Gerçek
   ortogonal kazanç bu kadar yüksek korelasyonda mümkün değil; gözlenen −1.58'in önemli kısmı
   muhtemelen fold-hizasızlık artefaktı.

**Erişilebilir 3 CPU-tekniği de ayrıca test edildi** (Ahmet'in fikirlerini bizim güçlü tabanda):
segment-yıl TE +0.018 (5/15, p=0.37), quantile +0.034 (2/15), CatBoost-MAE +0.0005 (6/15, p=0.98)
→ **3'ü de RED.** Yorum: Ahmet'in kazançları **zayıf tabanda** büyüktü; Tuna'nın tabanı (ham yıl +
native-kategorik + e5 + Huber) bu sinyalleri zaten içerdiği için telafi-edilmiş.

> **Karar:** SUB-2 (84.02, CV-doğrulanmış, paired-gate'lerden geçmiş) **KALDI**. Birleşme reddedildi —
> kazanç gerçek olabilir ama **kanıt standardını** karşılamıyor; ve onu seçmek, her iki ekibin de
> kurduğu CV-otoritesi disiplinini public-overfit lehine bozmak olurdu.

---

## 5. Denenmiş-ve-Elenen Teknik Envanteri (RED Zincirleri)

İki çözüm de "ne yaramadı"yı titizlikle belgeledi — bu, sığ değil derin keşfin kanıtı.

### Tuna'nın RED zinciri (~40 aile, hepsi paired-gate ile)
- **Alternatif GBDT:** HistGBR (blend ağırlık 0), XGBoost-L2 (+0.032), tabular-Ridge → hepsi redundant.
- **Eğitim mekanizması (6):** Tweedie, GLS, DART, sansür-aware L2/Huber, seed-bagging → hiçbiri L2/Huber'i geçemedi.
- **Post-process:** blend_p100 (oracle −3.57 ama gerçekçi −0.20), per-year isotonic (+1.26 overfit), affine (−0.006).
- **Metin:** txt_svd_gbdt (182.66), txt_ridge_wc (corr 0.974 redundant), txt_rich → hepsi gürültü bandında.
- **Sistematik tarama (252 taktik → 14 ölçüm):** kNN-target, MLP, null-importance FS, monotone, lineer-tree → 14 RED.
- **Düşük-EV (5):** HL-Gaussian, sigma(x)+E[clip], frequency-encoding, RBF-SVR, booster.refit → 5 RED.
- **Deep-research (8):** PetFinder post-process, BCE-on-target, Caruana GES, LightGBMLSS, cleanlab → 8 RED.

### Ahmet'in RED zinciri (günlükten)
- **Feature:** sistematik ikili çarpımlar (+0.59 zarar), n_strong/n_weak (nötr), kohort-yüzdelik (+0.13 zarar), staj tutarsızlığı (zarar), F1 övgü-gömme regex (F3'ü bozdu).
- **Model:** pseudo-labeling (89.1 vs 88.6), era-uzmanlaşma (113 vs 105), formül avı, residual-boosting (overfit), anlaşmazlık-meta (82.98 ama yön bilgisi yok), el-yapımı kesinti filtreleri (+32 zarar!).
- **Metin:** embedding'i GBM feature yapmak (85.97 > 85.49), fold-eşli embedding (87.58 çöktü), metin uzunluğu (r=0.008).
- **Blend dersleri:** ağırlık-fit kazançları LB'ye taşınmıyor (v10), çift-sayım sınırı (v43), proxy çözünürlük sınırı (<0.2).

**Ortak ders:** Bu veri **sentetik ve kohort-yapılı**. Sinyalin çoğu birkaç yapısal keşifte (Ahmet:
segment-yıl/kohort-z; Tuna: e5+Huber+mm), geri kalanı indirgenemez gürültü. İki ekip de bağımsız
olarak "feature/model uzayı tükendi, tavan ~84" sonucuna vardı.

---

## 6. Final Karar Mantığı ve Öneri

### Niçin SUB-2 (Tuna 84.02) seçildi?
1. **CV-doğrulanmış:** her bileşeni paired-gate'ten geçti; rw-OOF forward-chaining ile teyitli (Spearman +0.964).
2. **Public-kalibre:** gap +0.11 (yeşil), 2 kez bağımsız doğrulandı — ama public **seçim** için kullanılmadı.
3. **Risk dağıtımı:** SUB-1 (catboost_full, yapısal-bağımsız tek-model) + SUB-2 (ensemble) → private %40 bölmesine karşı çeşitli.

### Birleşme niçin alınmadı?
Kazanç (−1.58) **gerçek olabilir** ama **kanıt standardını karşılamıyor**: paired 11/15 (p=0.137),
fold-hizasız (iyimser-yanlı), public-overfit-seçilmiş. Disiplin = public'in peşinden koşmamak.

### Dürüst bir değerlendirme (savunma için)
- **Eğer** Ahmet'in modelleri Tuna'nın RepeatedStratified 5×3 fold'larıyla **yeniden eğitilseydi** ve
  birleşme **paired-gate'i geçseydi**, kabul edilirdi — çünkü tek-model rw-OOF'u (83.56) gerçekten daha iyi.
  Engel teknikti (fold uyumsuzluğu + zaman), prensip değil. Bu, gelecek için en güçlü iyileştirme yolu.
- İki çözümün **bağımsız olarak aynı pusulayı bulması** (recency-weighted OOF, public'e fit etmeme,
  çeşitlilik>tuning, XLM-R, ağır reg) her iki sonucun da sağlamlığının en ikna edici kanıtı.

---

## 7. Tek Bakışta Özet Tablo

| Boyut | Tuna (SUB-2) | Ahmet (ourteam) | Kazanan |
|---|---|---|---|
| Tek-model rw-OOF | 84.02 | **83.56** | Ahmet (+0.46) |
| CV stabilite (std) | 2.83 | **2.43** | Ahmet |
| İstatistiksel disiplin | **paired-gate, ~40 RED** | geniş ama bazı public-seçim | Tuna |
| Yapısal içgörü | e5+Huber+mm | **segment-yıl TE + kohort-z** | Ahmet |
| Reproducibility | **bit-repro, SEED=42, pinli** | 45-sürüm dallanma | Tuna |
| Fold şeması | RepeatedStratified 5×3 | KFold(10) | — (Tuna daha sağlam) |
| Public LB (gönderilen) | **84.13 (gap +0.11)** | 82.999 (winner's-curse) | Tuna (kalibre) |
| Birleşme (rw-OOF) | — | — | 82.44 ama **paired RED** |
| **Final seçim** | **✅ SUB-2** | birleşme RED | **Tuna** |

---

### Metodolojik Not
Tüm sayısal kıyaslar `src/probe_compare_solutions.py` (salt-okunur) ile canlı üretildi; aynı
`folds.parquet`, aynı `compute_recency_weighted_mse`, aynı hedef vektörü kullanıldı. Ahmet'in
OOF'u KFold(10) ile üretildiği için rw-OOF değeri iyimser-yanlı olabilir (§4.2). team_blend_opt
public-LB değeri (82.999) bir **referans**tır, karar metriği değil (her iki ekibin de CV-otoritesi ilkesi).
