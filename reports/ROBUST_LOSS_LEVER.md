# ROBUST-LOSS LEVER — Huber(α=5) lgbm_full_h (TIER-3, data-yapısı kaldıracı) → **KABUL**

> **Amac:** Kullanıcının "data üzerinde bir şeyler yapalım — augmentation/azaltma/FE" yönlendirmesinden
> çıkan kaldıraç zinciri. Alt-kuyruk sürprizlerinin (CEILING_AUDIT/LOW_TAIL_LEVER ile kanıtlanmış
> düzeltilemez Bayes hatası) **eğitim loss'unu zehirlemesini** engellemek.
> **Karar metriği = nested rw-OOF + PAIRED-TEST.** Public'e BAKILMADI.
>
> **SONUC (TL;DR):** `lgbm_full_h` (lgbm_full'un Huber(alpha=5) varyantı) **KABUL** → SUB-2 blend
> **84.2393 → 84.0991** (−0.140; paired 13/15, t=−3.83, p=0.0019, CI [−0.26,−0.02]). Yeni en düşük
> tek-model rw (86.32). **SUB-1 catboost_full KALDI** (fark paired'de gürültü + yapısal çeşitlilik).
> CatBoost-Huber ve tüm data-tarafı FE denemeleri ELENDİ (aşağıda).

---

## A. KEŞİF ZİNCİRİ — "data'ya odaklan" nasıl Huber'e vardı

| Adım | Deneme | Sonuç (repeat-0 lgbm_full; baz unw 78.43 / rw 88.48) |
|---|---|---|
| 1 | composites re-test (Faz-4 açık işi, NLP tabanında) | unw −0.18, **rw +0.04 → ELENDİ** (metin-zengin tabanda kazanç buharlaştı) |
| 2 | kohort-rank FE (yıl-içi yüzdelik, 6 skor) | unw −0.11, **rw +0.08 → ELENDİ** |
| 3 | label-denoise trimming (yüksek-\|resid\| satır atma) | drop %10: rw **−2.19** AMA filtre blend-OOF kullanıyor → **çapraz-fold y sızıntısı riski → REDDEDİLDİ** |
| 4 | **Huber loss** (trimming'in filtresiz/sızıntısız eşdeğeri) | **a=5: unw −1.42 / rw −1.31** ✓ (a=0.9def +1.15 kötü, a=10 −0.77, fair +1.44 kötü) |

**Mekanizma:** Alt-kuyruk sürprizleri (ex-ante ayırt edilemeyen düşük-y satırlar; iki-aşama Bayes
kanıtı LOW_TAIL_LEVER §C) L2 eğitiminde **karesel gradyanla** kütlenin (nüfusun %95'i) fit'ini bozuyor.
Kuyruk düzeltilemez ama kuyruğun kütleyi zehirlemesi engellenebilir: Huber |resid|>α'da gradyanı
sabitler. Trimming'in kazancının sızıntısız kısmı Huber'le doğrulanmış oldu.

## B. ÜRETİM — lgbm_full_h (full 15-fit)

`src/lgbm_full_h.py` = lgbm_full ile birebir aynı matris (yapısal+txt_ridge+lexicon) ve 15-fold
protokol; tek fark `objective='huber', alpha=5` (erken durdurma eval'i L2 = karar-metriği vekili).
Alpha repeat-0 fold-safe taramayla seçildi (gate-kör; e5-alpha emsali). **İnceltilmiş grid teyidi**
(repeat-0 rw): a=3→87.26, a=4→87.42, **a=5→87.16 (optimum)**, a=6→87.40, a=7→87.55, a=10→87.71 —
kaba grid optimumu birebir yakalamış; alpha=5 kesinleşti.

| Model | unw-CV | rw-OOF |
|---|---|---|
| lgbm_full (L2 ikizi) | 77.03 | 87.2663 |
| **lgbm_full_h (Huber)** | **76.25** | **86.3222** (−0.944) |
| catboost_full (önceki en iyi tek) | 76.30 | 86.4149 |

## C. BLEND KATKISI — küçük ama paired-anlamlı

| Konfig | nested rw-OOF |
|---|---|
| base (e5+mm'li 8 üye) | 84.2393 |
| **+lgbm_full_h (9 üye)** | **84.0991** (−0.140) |
| ikame (lgbm_full→h, 8 üye) | 84.1099 (eş; additive emsal seçildi) |

Paired (15 hücre): **13/15 iyileşti, t=−3.825, p=1.9e-3**, 5000-örnek bootstrap %95 CI
**[−0.264, −0.018]** tamamen sıfır-altı. Kıyas: yıl-etkileşimi adayı (−0.099) AYNI testte 10/15,
p=0.094, CI sıfırı kapsıyordu → reddedilmişti. −0.14 mutlak olarak mütevazı; kabul gerekçesi
gürültü-bandından paired-testle net ayrışması (e5/mm emsali) + kullanıcı onayı.

Final ridge_pos ağırlıkları: lgbm_full_h **0.236** (en yüksek), lgbm_num 0.242, mm 0.204,
catboost_full 0.181, catboost_full_w 0.100, e5 0.098, lgbm_full 0.050 (0.117'den düştü — doğal
kısmi ikame), txt_ridge/lgbm_full_w 0.000.

## C2. LGBM_NUM-HUBER → ELENDİ (gate: redundans)

Blend'in en yüksek ağırlıklı üyesi lgbm_num'a (0.242, metinsiz çeşitlilik ankrajı) aynı mekanizma:
`src/lgbm_num_h.py`, full 15-fit standalone rw **92.1077** (L2 ikizi 92.8154, −0.708 — mekanizma
burada da çalışıyor). AMA blend katkısı **gate'i GEÇEMEDİ**:

| Ölçüt | Değer | Eşik (mm/e5/lgbm_full_h emsali) |
|---|---|---|
| blend delta | 84.0991 → 84.0251 (−0.074) | — |
| paired iyileşen | **11/15** | ≥12 ✗ |
| paired p | **0.012** | <0.01 ✗ |
| bootstrap %95 CI | **[−0.189, +0.045] sıfırı kapsıyor** | üst<0 ✗ |

Huber bilgisi havuzda lgbm_full_h ile zaten temsil ediliyor; metinsiz huber ikizi ağırlıkla redundans.
Artefakt + ledger satırı dokümantasyon için tutuldu (histgbr/txt_rich emsali); CANDIDATE_POOL DIŞI.
Yeniden denemek: ensemble.py havuzuna "lgbm_num_h" ekle.

## C3. LGBM_FULL_HT (sıkı-regularizasyon) → **KABUL** (gece vardiyası 2. kabulü)

ÖNCEDEN-KAYITLI 12-konfig HP gridi (gate-kör, tek-yön değişimler, repeat-0): tek anlamlı yön
**sıkılaştırma** çıktı — `leaves15_mc80` rw 87.16→86.63 (−0.53); diğer 11 konfig ya kötü ya gürültü.
Post-hoc kombinasyon YAPILMADI. Full-15'te çürüme yok (−0.54) → seçilim-gürültüsü değil, gerçek etki:
muhafazakâr anchor bile bu gürültü seviyesinde fazla kompleksmiş.

| Model | unw-CV | rw-OOF |
|---|---|---|
| lgbm_full_h | 76.25 | 86.3222 |
| **lgbm_full_ht (leaves=15, mc=80)** | **75.78** | **85.7810** (yeni en düşük tek-model) |

**Blend:** EKLE 84.0991 → **84.0212** (−0.078; İKAME −0.051 ölçülüp elendi). Paired: **13/15,
t=−4.211, p=8.7e-4, bootstrap CI [−0.1552, −0.0019]** — üç ölçüt de geçti. Dürüstlük notu: CI üst
sınırı İNCE (−0.002); kabulü taşıyan şey hücre-tutarlılığı (std 0.072 — aynı büyüklükteki lgbm_num_h
−0.074, std 0.099 tutarsızlıkla düşmüştü). Final ağırlıklar: ht 0.165 + h 0.134 (huber ailesi toplam
~0.30), mm 0.237, lgbm_num 0.204.

**SUB-1 kararı (kayıt):** ht standalone catboost'tan −0.64 önde AMA paired 11/15, t=−2.81, p=0.014 —
kendi gate standardımıza göre tek-model üstünlüğü kanıtlanamaz; üstelik SUB-1'in amacı SİGORTA
(blend lgbm-huber-ağırlıklı → maksimum-bağımsız CatBoost kalır). `finalize` dışlaması `_h/_ht`'ye
genişletildi.

## D. CATBOOST-HUBER → ELENDİ (implementasyon underfit)

| Konfig | repeat-0 rw-OOF |
|---|---|
| catboost_full L2 (full-15 referans) | 86.41 |
| Huber:delta=5 | **104.18** (felaket) |
| Huber:delta=8 | **107.98** |

CatBoost'un Huber'i bu bütçede (lr 0.03, 3000 iter, RMSE eval) ağır underfit ediyor. lr/iter/leaf-
estimation kurtarma denemesi = HP-balıkçılığı (researcher freedom) → yapılmadı, temiz red.

## E. SUB-1 KARARI — catboost_full KALDI

lgbm_full_h (86.3222) vs catboost_full (86.4149): fark −0.093, paired'de **GÜRÜLTÜ**
(7/15 hücre, t=−0.44, p=0.67). CLAUDE.md: "cv farkı <0.25*std ise birini bilerek daha basit/farklı
tut" → eşitlikte yapısal-farklı incumbent kazanır: blend lgbm-ağırlıklı; SUB-1'in CatBoost
(ordered-boosting, ayrı aile) kalması private %40 bölmesine karşı risk dağıtımını korur.
`finalize_submissions._is_sub1_eligible` `_h` dışlaması ile kodlandı (+ mm/e5 açık dışlaması).

## F. FINAL ETKİSİ

| Final | Önceki | Yeni |
|---|---|---|
| **SUB-1** (safe tek-GBDT) | catboost_full (86.4149) | **catboost_full (86.4149)** — DEĞİŞMEDİ |
| **SUB-2** (ensemble blend) | blend 84.2393 (e5+mm) | **blend 84.0991** (e5+mm+huber) |

|SUB-1 − SUB-2| ortalama fark 1.11 → yapısal çeşitlilik korundu. Format assert'leri geçti.
Kümülatif SUB-2 yolculuğu: 85.4945 (GBDT+txt) → 84.8464 (+e5) → 84.2393 (+mm) → **84.0991** (+huber).

### Artefakt / kanıt
- `src/lgbm_full_h.py` → `artifacts/{oof,test}_lgbm_full_h.npy`; ledger satırları (model_scores/cv_log/cv_scores).
- `src/ensemble.py` CANDIDATE_POOL + kabul notu; `src/finalize_submissions.py` `_h` dışlaması.
- FE/trimming/catboost-huber redleri: bu rapor §A/§D (artefaktsız; Occam).
- `submissions/sub2_blend.csv` güncellendi; `sub1_catboost_full.csv` değişmedi.
