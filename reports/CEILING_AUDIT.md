# TAVAN DENETİMİ — "blend 84.24 tavanda mı?" → **EVET (mevcut bilgi seti için)**

> **Soru:** SUB-2 (e5+mm blend, nested rw-OOF 84.2393) gerçek tavanda mı, yoksa çekilebilir sinyal
> kaldı mı? **Yöntem:** "tavandayız" iddiasını ÇÜRÜTMEYE çalışan 4 bağımsız sonda. Hepsi fold-safe,
> karar metriği nested rw-OOF, public'e BAKILMADI.
>
> **SONUC (TL;DR):** 4 sondanın 4'ü de iddiayı çürütemedi. Residual TÜM modalitelerde öğrenilemez;
> meta-seviye alternatifler (nonlineer/bağlamsal stacking) ya kötüleştiriyor ya gürültü-bandında;
> iki maksimum-farklı fonksiyon sınıfının residual'ları 0.906 korele (ortak hata = bilgi-seti limiti).
> **Kalan tek meşru kaldıraç: mm full 15-fit** (yeni sinyal DEĞİL, varyans azaltma; opsiyonel).

---

## Sonda 1 — Residual öğrenilebilirliği, TAM uzay → ÇÜRÜTÜLEMEDİ

Blend residual'ını (y − oof_blend) her modaliteyle OOF tahmin etme girişimi (repeat-0, fold-safe):

| Düzeltici | OOF R² | rw etkisi |
|---|---|---|
| e5 embedding (1024) → Ridge | **−0.011** | 84.24 → 85.27 (kötü) |
| TF-IDF word(1,2) → Ridge (fold-içi fit) | **−0.141** | 84.24 → 94.87 (kötü) |
| tabular 82 (num+cat+yıl+flag) → HistGBR | **−0.045** | 84.24 → 88.13 (kötü) |
| tabular+e5 (1106) → HistGBR | **−0.028** | 84.24 → 86.50 (kötü) |

**Hiçbir modalite residual'ı tahmin edemiyor; her düzeltici işleri kötüleştiriyor.** (Önceki sayısal-only
test eksikti; bu tam-uzay testi onu kapsayıp genişletir.)

## Sonda 2 — Nonlineer meta-stacker → ÇÜRÜTÜLEMEDİ

ridge_pos yerine kapasiteli meta (nested, 15 hücre, sample_weight=recency):

| Meta | nested rw-OOF | vs 84.2393 |
|---|---|---|
| GBDT-stacker (OOF kolonları) | 86.0992 | +1.86 (meta-overfit) |
| GBDT-stacker (+yıllar) | 85.8024 | +1.56 |

Lineer-pozitif meta zaten doğru kapasitede; nonlineerlik meta-overfit getiriyor.

## Sonda 3 — Bağlamsal (yıl-duyarlı) blend → ÇÜRÜTÜLEMEDİ

"Optimal karışım yıla göre değişiyor olabilir" hipotezi (recency hata payının %66'sı 2024-26'da):

| Varyant | nested rw-OOF | Paired test |
|---|---|---|
| per-year ridge_pos (bucket, ≥300 fallback) | 84.4097 (+0.17) | — (kötüleşti) |
| ridge + P×yıl etkileşimi | 84.1406 (−0.099) | **10/15, t=−1.80, p=0.094, CI [−0.34,+0.15] sıfırı KAPSIYOR** |

Tek sınır-aday (−0.099) paired testte **anlamsız** (mm'in geçişi 15/15, p=1.4e-4, CI tamamen negatifti —
kıyas net). CLAUDE.md "marjinal ~0.1 MSE reddedilir" kuralının ders-kitabı örneği → **REDDEDİLDİ**.

## Sonda 4 — Mutlak taban: çapraz-sınıf residual yapısı → TAVAN TEYİDİ

| Kanıt | Değer | Yorum |
|---|---|---|
| corr(resid_catboost, resid_mm) | **0.906** | GBDT vs neural-multimodal (maksimum farklı sınıflar) hatanın ~%82'sinde hemfikir → ortak bileşen = bilgi-seti limiti |
| corr(resid_blend, üyeler) | 0.94–0.99 | blend üye-farklarını zaten emmiş |
| ikiz testi (1-NN y-uyuşmazlığı) | 131–156 | BİLGİ VERMİYOR: 82-dim'de 10k satır seyrek (ort. mesafe 7.9σ), tahmin ortalama-farkla şişik |
| forensics heterosked. E[σ²] | 36.4 | güvenilmez kaynak (|resid| regresyonu); doğrudan öğrenilemezlik testi (Sonda 1) onu geçersiz kılar |

## Daha önce elenenler (tekrar denenMEdi)

histgbr (3. GBDT ailesi), txt_svd_gbdt, txt_rich, txt_ridge_wc, global+per-year isotonic, p100 iki-aşama,
fractional logit (atlandı), recency sample-weight (eğitim), alt-kuyruk sample-weight, alt-kuyruk regex.
Hepsi defterli: FORENSICS.md, LEVERS_SUMMARY.md, LOW_TAIL_LEVER.md, ensemble.py inline notları.

## Kalan TEK meşru kaldıraç — mm full 15-fit (opsiyonel robustluk)

mm OOF kolonu şu an **repeat-0 (tek repeat)**; diğer üyeler 3-repeat ortalaması. Etkisi ÖLÇÜLDÜ
(ikame deneyi): lgbm_full deterministik olduğundan repeat-0'ı birebir yeniden üretildi ve blend'de
3-repeat kolonu yerine konuldu → blend nested rw-OOF 84.2393 → 84.2684 (**zarar +0.029**, ağırlık 0.117).
mm ağırlığına (0.212) ölçeklenince mm 15-fit'in **görünür metrik kazancı ≈ 0.10** (eşit-gürültü
varsayımı; neural gürültüsü daha yüksekse biraz üstü). Ancak bunun çoğu ÖLÇÜM iyileşmesi (daha temiz
kolon → daha az kötümser metrik); **gerçek private kazancı** = test-bagging (5→15 model; lgbm'de
satır-başı std 0.377, ağırlık² ile ~0.004-0.01 MSE) + hafif daha iyi meta-ağırlık ≈ **<0.05 MSE**.
Ridge meta gürültülü kolonu zaten adaptif düşük-ağırlıklayıp emiyor (standalone 1-rep zararı +1.21
iken blend zararı sadece +0.03 — blend üye-gürültüsüne robust). SONUÇ: bir GPU koşusuna değmez;
CLAUDE.md marjinal-kazanç bandının içinde. Tavanı KIRMAZ; repeat-0 bilinçli/dürüst tercih olarak kalır.

## KARAR (ilk tur)

**Blend 84.2393, mevcut bilgi seti (yapısal + yıllar + Türkçe metin tüm kanalları + neural fusion) için
pratik tavandır.** Daha fazla model denemesi negatif beklenen değerli (overfit riski > kazanç umudu).
SUB-1 (catboost_full 86.41) + SUB-2 (blend 84.24) FİNAL. Kaynak bundan sonra sunum/repro'ya harcanmalı.

> **GÜNCELLEME:** Bu ilk-tur karardan SONRA, kullanıcının "data yapısına odaklan" yönlendirmesiyle
> Huber robust-loss bulundu ve blend 84.0991'e indi (reports/ROBUST_LOSS_LEVER.md) — yani ilk tur
> "tavan" hükmü EĞİTİM-MEKANİZMASI uzayını kapsamıyordu. Aşağıdaki gece vardiyası o uzayı da taradı.

---

## GECE VARDİYASI (2. tur) — eğitim-mekanizması uzayı: 6 sonda, 6 RED

Huber dersi üzerine ("veriyi değil eğitim mekanizmasını değiştir"), kalan mekanizma ailelerinin
sistematik taraması (hepsi repeat-0 fold-safe, lgbm_full tabanı; bazlar: L2 rw 88.4754 / huber 87.1618):

| Sonda | Gerekçe | repeat-0 rw | Karar |
|---|---|---|---|
| Tweedie p∈{1.1,1.3,1.5,1.7} (z=100−y) | yansıtılmış hedef ders-kitabı Tweedie şekli (%7.7 sıfır kütlesi + pozitif kuyruk) | 89.2–91.9 | **RED** — log-link, additive-lineer latent'e uymadı |
| GLS 1/σ² ağırlık (L2 ve huber üstünde) | heteroskedastisite (σ: 11.9→4.7) → verimli tahminci | 89.77 / 90.50 | **RED** — Huber'in gradyan-kapamasının üstüne aşırı-düzeltme |
| Sansür-farkında objective, cens-L2 | forensics'in "denenmedi" dediği açık kapı (y=100 tek-yönlü loss + clip-feval) | 88.25 | **RED** — huber'den zayıf |
| Sansür-farkında, cens-huber | aynı, huber iç-bölge | **86.93 (−0.24)** | **RED** — emsal kalibrasyon: −0.91'lik lgbm_num_h bile gate'ten döndü; −0.24 + corr≈0.99 şanssız |
| DART + huber | dropout-ağaç dekorelasyonu | 5341 (bozuk) | **RED** — bu bütçede viable değil; tune etmek HP-balıkçılığı |
| Seed-bagging ×3 (huber) | saf varyans azaltma | 86.82 (−0.34) | **RED** — mm-15-fit emsali: blend'e gerçek yansıma ~0.03-0.08 (marjinal bant), pipeline 3× pahalı |

Ayrıca aynı turda: lgbm_num_h (metinsiz huber ikizi) full-15 + paired gate → **RED** (11/15, p=0.012,
CI sıfırı kapsar; ROBUST_LOSS_LEVER §C2) ve α-inceltme → α=5 optimum teyit.

**Pseudo-labeling (ertesi sabah, kullanıcı sorusu üzerine ölçüldü) → RED:** fold-safe öğretmen
(fold-train'den eğitilir, mevcut bagged test-pred KULLANILMAZ — val-y sızıntısı) + öğrenci
(fold-train + 10k pseudo-test satırı), lgbm_full_ht tabanı. Sonuç: baz rw 86.6303 → PL w=0.3:
**86.92 (+0.29)**, w=1.0: **86.96 (+0.33)** — her iki ağırlıkta ZARAR. Mekanizma: (a) dağılım-adaptasyon
kanalı = recency-weight'in ambalajı (zaten ölçülüp başarısız); (b) distilasyon kanalı yüksek-gürültü
probleminde confirmation-bias'a dönüşüyor (gerçek 8k etiket, öğretmenin kendinden-emin-hatalı 10k
pseudo'suyla seyreltiliyor). Test aynı sentetik üreticiden → unlabeled'da sömürülecek ekstra yapı yok.

## 3. TUR — SİSTEMATİK ENVANTER TARAMASI (252 taktik → 24 denenmemiş → 14 ölçüm, 14 RED)

Kullanıcının "plato deme, sistematik bul" yönlendirmesiyle 5-mercekli çok-ajanlı envanter çıkarıldı
(Kaggle kazanan çözümleri, gürültülü-etiket, kovaryat-kayma, metin+tabular, taksonomi): **252 taktik
toplandı**, denenmişler düşüldü → **24 denenmemiş** aday. Yüksek/orta-EV'lilerin tamamı ölçüldü
(hepsi fold-safe; blend adayları paired-gate'ten):

| Aday (ajan-EV) | Ölçüm | Karar |
|---|---|---|
| linear_tree (yüksek) | huber-üstü −0.06 / L2-üstü 0.00 | RED — lineer yapı zaten hasat edilmiş |
| **Tabular Ridge — "en büyük envanter boşluğu"** (yüksek) | standalone rw 98.7; blend +0.085 | RED — lineer aile recency'de çöküyor |
| XGBoost-L2 4. aile (yüksek) | standalone 87.63; blend ön-deneme **+0.032** | RED (HistGBR emsali doğrulandı) |
| XGBoost-pseudohuber | rw 824 (bozuk; xgb base_score tuhaflığı) | RED, debug edilmedi |
| Monotone constraints (orta) | +2.3 zarar | RED |
| Meta: alpha grid (orta) | 0.03→30 tamamen düz | RED |
| Meta: serbest-işaret ridge | nested −0.084 | RED (NM'in zayıfı) |
| Meta: Nelder-Mead doğrudan rw-MSE | nested **−0.123 AMA paired 12/15, p=0.012, CI [−0.334,+0.093] sıfırı kapsar** | RED — meta-overfit; gate görevini yaptı |
| kNN-target tabular-uzay (orta) | +0.25 | RED |
| kNN-target **e5-uzayı** (orta) | standalone **−0.23 GERÇEK** (85.78→85.56 full-15) ama blend −0.060, paired 12/15 p=0.013 **CI [−0.141,+0.018] sıfırı kapsar** | RED — e5_ridge aynı bilgiyi taşıyor, pass-through yetersiz |
| Null-importance feature selection (orta) | 46/62 feature düştü → reduced-set **çöktü** (92.93) | RED — korelasyonlu skorlar split-imp'i bölüşüyor; LGBM içsel seçimi yeterli |
| Saf-tabular MLP (orta) | standalone 107.7; blend +0.003 | RED — mm'in değeri NN sınıfı değil METİN fine-tune'uymuş |
| Soft-relabel distilasyon (orta) | a=0.85: +0.07, a=0.7: +0.43 (monoton kötü → optimum a=1) | RED — öğretmen-yumuşatma bias ekliyor, gürültüyü azaltmıyor |
| Pseudo-labeling (önceki bölümde) | +0.29/+0.33 | RED |

Ölçülmeden kapatılanlar: DAE (ön-koşul zayıf: 39/496 korelasyonlu çift + MLP'nin tam yenilgisi),
2.tur Optuna (grid optimumu bulundu, blend-gate hiçbir üye-iyileştirmesini geçirmiyor → EV~0),
düşük-EV listesi (rank-avg, TTA, HL-Gaussian, AutoGluon vb. — gerekçeler workflow çıktısında).
AÇIK KALAN TEK ADAY: **TabPFN v2 (Colab GPU, notebooks/colab_tabpfn.py)** — ortogonal
prior-fitted-transformer sınıfı; lokal CPU infeasible (16.9GB), kullanıcı koşusu bekliyor.
NOT: xgboost/tabpfn pip kurulumları yalnız probe içindi; requirements.txt ve kanonik pipeline DEĞİŞMEDİ.

## 4. TUR — DÜŞÜK-EV KUYRUĞU SÜPÜRMESİ (envanteri SIFIRLAMA; 5 ölçüm + 1 doğrulama, 5 RED + 1 teyit)

Kalan LOKAL-CPU kuyruğunun sistematik kapatılması. Amaç skor değil ENVANTER SIFIRLAMA
("evren tarandı, denenmemiş kalmadı" = jüri kanıtı). Hepsi fold-safe; karar = nested rw-OOF +
PAIRED-GATE (mm_gate deseni); public'e BAKILMADI. Bazlar: lgbm_full_ht repeat-0 rw **86.6303**
(altyapı birebir yeniden üretildi), blend nested rw-OOF **84.0212**, ht full-15 rw **85.7810**.

| Aday | Ölçüm | Karar |
|---|---|---|
| **1. HL-Gaussian / regresyon-sınıflandırma** (y→64 bin, fold-içi quantile + y=100 ayrı bin, LGBM multiclass num_class=64, tahmin=Σpᵢ·binₘₑᵣₖₑᵤ) | repeat-0 rw **98.8549 (+12.22)** | **RED** — sürekli hedefi kuantize etmek ağır bilgi kaybı + multiclass bin-başı varyansı; düzgün-sürekli hedefte regresyon-sınıflandırma yapısal olarak kaybeder |
| **2. Sigma(x) + analitik E[clip]** (fold-safe σ(x) LGBM, hedef \|y−blend_oof\|; pred'=E[clip(N(μ,σ),0,100)] censored-normal kapalı-form; c·σ ölçek gridi) | en iyi c=0.5 **−0.0263**; PAIRED **13/15, t=−2.83, p=0.0135, CI [−0.067,+0.002] sıfırı KAPSIYOR** | **RED** — üç ölçütten ikisi geçmedi (p≥0.01, CI⊅0); ridge blend zaten clip-sonrası optimize, homoskedastik-gauss residual varsayımı %7.7 atom-kütleli hedefte tutmuyor; c<1 kazancı tesadüfi shrinkage |
| **3. Frequency/count encoding** (5 kategoriğe fold-train frekans kolonu, lgbm_full_ht) | repeat-0 **−0.1309 (umut)** → full-15 standalone **−0.0123**; blend +freq PAIRED **6/15, t=−0.20, p=0.846, CI [−0.052,+0.047]** | **RED** — repeat-0 −0.13 SEÇİM-GÜRÜLTÜSÜ (full-15'te %90 buharlaştı); küçük kardinalite (4–11), LGBM native-kategorik frekansı zaten split-yapısında tutuyor. CLAUDE.md repeat-0→full-15-gate disiplininin ders-kitabı yakalaması |
| **4. RBF-SVR mini-zoo** (tabular 82, fold-içi median-impute+StandardScaler, tam RBF-SVR 6.8s/fit; C∈{1,10,30}×γ∈{scale,0.01}) | en iyi (C=10,γ=0.01) rw **104.93** (grid 104.9–120.8); eşik ~92 | **RED (standalone)** — tabular-only SVR GBDT'ye (85.78) onlarca-MSE geride; saf-tabular MLP (107.7) yenilgisini yineler. mm'in değeri NN sınıfı DEĞİL metin fine-tune'uydu → blend ön-denemesine değmez |
| **5. Booster.refit recent-slice** (lgbm_full_ht fold-modeli, refit decay∈{0.1..0.9} fold-train 2024-26 alt-dilimi, n=2610) | refit-yok kontrol 86.6303 (birebir); tüm decay KÖTÜ — en iyi decay=0.9 **+2.71** (89.34), decay=0.1 +15.19 | **RED** — recency-adaptasyon ailesi (sample-weight/year-weight/pseudo-label/refit) yine kaybetti; yaprak recency-dilime kayınca kütle-fit'i bozuluyor, test aynı sentetik üreticiden → kaydırılacak ekstra yapı yok |
| **6. Forward-chaining cross-check** (SKOR DEĞİL, DOĞRULAMA; yıl≤2023 geçmiş / 2024-26 gelecek dilim, model EĞİTİLMEZ, mevcut OOF'lar) | **Spearman(gelecek-MSE, rw-OOF)=+0.964**, Spearman(geçmiş, rw-OOF)=+1.000, Spearman(geçmiş, gelecek)=+0.964 | **POZİTİF TEYİT** — rw-OOF sıralama-kararları bağımsız temporal split'te güçlü teyitli → karar metodolojisi recency-doğru, karar otoritesi sağlam (tek üye-rank gürültüsü catboost vs huber, zaten paired'de eşit) |

ENVANTER DURUMU: 4. turla lokal-CPU düşük-EV kuyruğu KAPANDI. Açık kalan tek aday hâlâ
**TabPFN v2 (Colab GPU)** — ortogonal prior-fitted-transformer; lokal CPU infeasible (16.9GB),
kullanıcı koşusu bekliyor. Geçici probe scriptleri `src/_probe4_*.py` + artefaktları `_tmp_*`
dokümantasyon sonrası temizlendi; kanonik pipeline / CANDIDATE_POOL / requirements.txt DEĞİŞMEDİ.

### Ek (kullanıcı önerisi) — Hiyerarşik 2-aşamalı tahmin (coarse classification → bin-içi regresyon) → RED

Kullanıcı fikri: "önce y'yi 10 gruba ayır, classification ile bin seç; sonra seçilen binde regresyon
yap (ör. 60-70 bini → 66.7)". Sonda 1 HL-Gaussian'ın (soft-expectation, +12.22) akrabası AMA farklı:
hard-routing + **bin-içi ayrı GBDT regresörü**. Fold-safe ölçüldü (binler fold-içi qcut9+==100=10 sınıf;
aşama-1 ht-konfig multiclass classifier; aşama-2 her bin için Huber α=5 regresör, <200 satırlı binlerde
global-huber fallback). **Sanity kontrolü mükemmel: düz-ht kontrol kolu repeat-0 rw 86.6303 birebir.**

| Varyant | repeat-0 rw | delta vs 86.6303 |
|---|---|---|
| KONTROL (düz lgbm_full_ht) | 86.6303 | baz (sanity OK) |
| **HARD (argmax bin → o binin regresörü)** | **127.82** | **+41.19** 💥 |
| **SOFT (Σ P(bin)·reg_bin)** | **91.31** | **+4.68** |

**RED — ve mekanizma öğretici:** (a) HARD felaketi (+41) tam da en-kötü-satırların feature-ayırt-edilemez
oluşundan (top-50 z-skoru ~0): classifier y=0 satırını feature'ına bakıp popülasyon-tipik orta bine
atıyor → o binin regresörü ~76 tahmin ediyor → ±40-60 taban-hata. Tek yanlış-routing, doğru bin-içi
regresyonu yok ediyor. Classification, sürekli hedefe **sert bilgi-darboğazı** koyuyor ve darboğaz tam
uç satırlarda yanılıyor. (b) SOFT yumuşatması taban-hatayı siliyor (127→91) ama bin-içi az-veri varyansı
(~820 vs 8000 satır) + HL-Gaussian'a yakınsama yüzünden hâlâ +4.68. **Ders:** hiyerarşik tahmin
modaliteler/rejimler GERÇEKTEN ayrıksa kazanır; burada hedef sürekli-düzgün ve bin'i belirleyen sinyal
feature'larda zaten zayıf → classification, regresyonun işini kayıplı tekrar ediyor. Tek-aşama sürekli
Huber, [0,100]'ü kesintisiz modelleyip belirsizlikte ortalamaya kaçıyor = MSE-optimal. Artefaktsız (Occam).

### Ek (kullanıcı önerisi) — "Katmanlardan tek tek geçir": Residual zinciri + Derin stacking → 2 RED

Kullanıcı fikri: saf-routing yerine satırı KATMANLARDAN sırayla geçir. İki meşru mimari ayrıştırıldı,
ikisi de fold-safe ölçüldü (nested OOF; her katman bir öncekinin **OOF** residual'ini avlar → sızıntısız):

**Mimari A — Residual zinciri (boosting-of-models):** L1=yapısal lgbm_num → L2 hedefi y−L1_OOF
(metin-zengin) → L3 hedefi y−(L1+L2)_OOF (full Huber). repeat-0:

| Adım | rw | ne oldu |
|---|---|---|
| L1 (yapısal) | 93.8465 | kasıtlı zayıf başlangıç (avlanacak residual olsun) |
| +L2 (residual avla) | **88.9761** | **−4.87 GERÇEK kazanç** (metin sinyali residual'da vardı; zincir doğru kurulmuş) |
| +L3 (kalan avla) | 88.9819 | **+0.006 KURU** (Sonda 1 residual-öğrenilemezlik emsali) |

**RED** — L2 mekanizmayı kanıtladı (−4.87, sızıntı olsa sahte-büyük olurdu) AMA zincir tek-aşama
lgbm_full_ht'yi (86.63) **2.35 geride** kaldı; L3 kuru. Sebep: GBDT zaten içsel residual-boosting
makinesidir; modelleri elle katmanlara bölmek, birlikte-eğitilen tek modelin verimini bozar (katmanlar
ardışık donar, birbirinin kapasitesini optimize edemez).

**Mimari B — 3-katman derin stacking:** L1=10 taban model → L2=4 meta-varyant (ridge_pos α∈{0.2,1,5},
nnls) → L3=meta-of-meta (lineer-pozitif). nested:

| Katman | nested rw | not |
|---|---|---|
| L2 ridge_pos (mevcut blend) | 84.0212 | α=0.2/1/5 **birebir aynı** (buzme meta-çözümü değiştirmiyor) |
| **L2 meta'ları arası min korelasyon** | **0.9988** | aynı P'den türüyorlar → kollinear |
| L3 ridge_pos / nnls | 84.0981 / 84.0685 | **+0.077 / +0.047 (ikisi de kötü)** |

**RED** — L2 meta'ları ρ=0.999 kollinear (aynı taban-matrisin lineer kombinasyonları, aynı uzayda) →
3. katman yeni bilgi taşıyamaz, sadece gürültü ekler. Lineer-pozitif meta zaten optimal kapasitede
(4. tur nonlineer-meta +1.86 overfit'inin hafif versiyonu). **Ortak ders:** mimari karmaşıklığı
var-olmayan sinyali çıkaramaz; bilgi-seti 84.02'de tükenmiş, ekstra katman = overfit yüzeyi/gürültü.
Blend 84.0212 değişmedi. Artefaktsız (Occam).

### Ek (kullanıcı/arkadaş önerisi) — "kalan 56.87'yi düşür" 5 taktik: Log-Cosh + K-Means lokal → 2 RED (+3 zaten-kapalı)

Bağlam: 340 kötü satır (|resid|>20, train'in %3.4'ü) toplam rw-MSE'nin **%35.2'sini** üretiyor; onlar
SİLİNİRSE kalan 9660 satırın rw-MSE'si 56.87 (ort. hata 5.65, %53.9'u ±5 içinde). Öneri "o 56'yı düşür"
idi — ama 56.87 SKOR DEĞİL (test'te o satırlar var, silinemez; gerçek skor 84.02). 5 öneriden 3'ü zaten
kapalıydı: **stacking+Ridge = blend'in ta kendisi** (10 üye, ridge_pos meta); **feature interactions =**
Faz-4 + 3.tur'da "kanıtlı zararlı" RED; **target transform (log1p/Yeo-Johnson) = mantık hatası**
(log1p büyük-uçtaki 76-100 yığılmasını AÇMAZ, EZER; +Jensen bias; CLAUDE.md "log/logit YOK"). Kalan 2
gerçekten yeni öneri fold-safe ölçüldü:

| Aday | Ölçüm | Karar |
|---|---|---|
| **Log-Cosh loss** (lgbm_full_ht custom obj; loss=s²·log cosh(r/s), grad=s·tanh(r/s), hess=sech²; init_score=fold-mean — ilk koşu underfit'i init_score yokluğundandı, düzeltildi; scale∈{1,3,5}) | en iyi scale=5 repeat-0 rw **90.94** (+4.31 vs Huber 86.63) | **RED** — Log-Cosh ≈ Huber'in pürüzsüz akrabası; "küçük-hata ±5 pürüzsüzlüğü" tezi tutmadı çünkü o bölge zaten MSE-benzeriydi, belirleyici uç-satır dizginlemesi → keskin-eşikli Huber(α=5) yumuşak geçişi yeniyor |
| **K-Means lokal modeller** (cluster-then-predict; tabular-uzayda fold-içi K-Means, her kümeye ayrı Huber GBDT uzman, <500 satır→global fallback; K∈{3,4}) | en iyi K=3 rw **90.47** (+3.84); K=4 91.77 | **RED** — (a) uzman başına ~%33 az veri → varyans; (b) küme yine feature'a dayalı, en-kötü satırlar (feature ayırt-edilemez, z~0) yanlış rejime düşüyor; global GBDT profil-bazlı ağırlığı ağaç dallarıyla zaten çözüyor. Hiyerarşik bin-routing (+4.68) emsali |

**Sonuç:** "56.87'yi düşür" çerçevesi yanlış-kurgu — kalan satırlar zaten ~mükemmel (ort. 5.65), orada
çekilecek sinyal yok; tavan o satırlardan değil, feature'dan öngörülemeyen %3.4 uç-satırdan geliyor.
Her iki yeni taktik de eşiği geçemedi. Blend 84.0212 değişmedi. Artefaktsız (Occam).

### Ek (kullanıcı/arkadaş önerisi) — 10 elle-feature (metin örüntü + yapısal etkileşim) → RED (repeat-0 yanılttı, blend söndürdü)

Arkadaş 10 hedef-bağımsız feature önerdi (4 metin: but-criticism regex, soft/hard ratio, potential-trap,
buzzword-density; 6 yapısal: academic-bubble, market-rejection, ghost-coder, **t-shaped-std**,
eager-incompetent, feedback-len). Hepsi y-bağımsız → fold-safe. FULL matrise eklenip lgbm_full_ht ile
ölçüldü. **İki aşamalı disiplin (repeat-0 ablation → full-15 + blend paired-gate) kritikti:**

**repeat-0 tekil ablation** (baz 86.6303): **`f_tshaped_std` −0.194** (en güçlü; teknik skor std = uzman
vs jenerist), soft_hard/potential/eager/feedback ~−0.07, **market_rejection +0.093 ZARARLI**, academic_bubble
0.00 (%0.6 seyrek), **TÜMÜ-10-birlikte +0.063** (feature'lar birbirini boğuyor). En umutlu ikisi blend-gate'e:

| Aday | full-15 standalone | blend EKLE Δ | paired-gate (EKLE) |
|---|---|---|---|
| **f_tshaped_std** (tek) | **+0.087** (repeat-0 −0.194 BUHARLAŞTI) | +0.006 | **6/15, p=0.67, CI [−0.042,+0.053] → ELENDI** |
| 5'li temiz set (market hariç) | +0.061 | +0.002 | 8/15, p=0.89, CI [−0.047,+0.051] → ELENDI |

**RED — ve metodolojik ders altın değerinde:** `f_tshaped_std` repeat-0'da −0.194 ile EN parlak aday
gibi göründü; full-15'te **+0.087'ye döndü** (seçim-gürültüsü, Sonda-3 frequency-encoding tuzağının
ikizi) ve blend'e net-yeni sinyal getirmedi (paired CI sıfırı kapsıyor). Mekanizma: t-shaped/soft-hard/
potential bilgisi mevcut üyelerde ZATEN var — GBDT'ler ham teknik skorları görüyor, txt_ridge+e5 metni
okuyor. Elle-feature, blend'in başka kanallardan aldığı bilgiyi tekrar ediyor (Faz-4 "composites/
interactions zararlı" + 3.tur txt_rich/lexicon-redundans emsali). **Kullanıcının "tek-model değil
blend'e bak" sezgisi bu RED'i kurtardı** — repeat-0'a bakıp kabul etseydik gürültü kabul edilirdi.
Blend 84.0212 değişmedi. Artefaktsız (Occam).

### Ek (kullanıcı önerisi) — Kötü-tahmin TANISI + kanal-uyuşmazlığı ("çelişki") feature'ları → RED

Kullanıcı `kotu_tahminler.csv`'yi (340 satır, |resid|>20, rw-MSE'nin %35.2'si) inceleyip "neden kötü,
ex-ante flagleyip cezalandırabilir miyiz" diye sordu. **Tanı (gerçek veriden):** kötü satırlarda mentor
metni SİSTEMATİK YANILTICI — metin-only model (txt_ridge) gerçeği **22.4 puan saptırıyor** (iyi satır
9.1); örn. STU_005695 (y=0) "liderlik... önemli varlık", STU_000614 (y=45) "etkileyici güçlü aday".
Y-SİZ aday flag bulundu: kanal-uyuşmazlığı |txt_ridge − blend| kötü **8.41** vs iyi **6.84**; "ancak/fakat"
dönüşü %79.7 vs %67.4. Bu uyuşmazlık feature'a çevrildi (5 çelişki, hepsi y-siz/OOF-tahminden → fold-safe):

| Çelişki feature | repeat-0 delta vs 86.6303 |
|---|---|
| d_txt_vs_num \| d_txt_vs_cat \| d_model_std \| d_txt_minus_num \| d_txt_minus_blend | **+1.90 … +2.79 (HEPSİ ZARARLI)**; TÜMÜ +1.99 |

**RED (net, repeat-0'da bile) — istatistiksel serap dersi:** Tanı "kötü satırlarda uyuşmazlık biraz
yüksek" dedi (8.41>6.84) AMA bu ≠ "uyuşmazlık kötülüğü tahmin eder". (a) dağılımlar örtüşüyor: yüksek
uyuşmazlıklı satırların çoğu (9660 iyi vs 340 kötü) aslında iyi tahmin ediliyor → feature %95 yanlış-alarm;
(b) feature modelin KENDİ girdilerinden türev (txt_ridge_pred + skorlar zaten matriste) → yeni bilgi yok,
ama iki gürültülü tahminin farkı = daha gürültülü → split-kalitesini bozup zarar; (c) metin-yanıltıcılığı
gerçek ama metin her satırda aynı pozitif şablonla yazılı → "yalan mı" sorusu ancak y bilinirse cevaplanır
= sızıntı. Gözle "mantıklı, ekleyelim" yerine fold-safe ölçüm serabı yakaladı. Blend 84.0212 değişmedi.

### Ek (kullanıcı önerisi) — 340 kötü satırın SEGMENT analizi → pattern SERAP (kalibrasyon çürüttü)

Kullanıcı "uç-yalanlar hariç orta-bölgede pattern var mı" diye sordu. 340 satır 4 segmente bölündü
(hata-yönü × büyüklük): asiri_orta(20-30) 160, asiri_uc(>30) 29, dusuk_orta 130, dusuk_uc 21. Segment
imzası (z-skor vs popülasyon) GÜÇLÜ göründü: **dusuk-tahmin (y yüksek, pred düşük) segmentlerinde
project_quality_score −0.81/−1.19σ, technical_interview −0.56/−1.02σ, communication −0.38/−0.74σ.**
"Kâğıtta zayıf ama gerçekte başarılı" profili gibi durdu.

**KALİBRASYON TESTİ pattern'i SERAP olarak çürüttü** (feature ölçümüne bile gerek kalmadı): "düşük
project_quality olan TÜM satırlarda model sistematik underpredict ediyor mu?" → **HAYIR**. project_quality
5 kuantil diliminde mean_resid = {+1.04, −0.21, +0.15, −0.27, −0.12} ≈ 0 (technical_interview da ±0.27).
**Model HER skor-diliminde mükemmel kalibre.** Segment imzası bir TAUTOLOJİ: "düşük tahmin edilen satır
= düşük skorlu" (model girdisine bakıp düşük diyor, doğru davranış); geriye-bakışta o azınlığın y'si
yüksek çıkıyor ama düşük-skorluların %95'i zaten doğru tahmin ediliyor → "düşük-skorlu" flag'i %95
yanlış-alarm. **Ders:** segment-imzası (geriye-bakış z-skoru) ≠ ileriye-dönük sinyal; kalibrasyon testi
(her dilimde resid~0) bunu feature yapmadan elendi. İYİ HABER: kalibrasyon sıfır-overfit'in kanıtı —
kötü satırlar sistematik kör-nokta değil, rastgele indirgenemez gürültü. Blend 84.0212 değişmedi.

### Ek (kullanıcı önerisi) — Regex söz-dizimi pattern → nested-OOF model → soft-voting üyesi → 2 RED

Kullanıcı "metindeki cevher/balon örüntülerini düz regex'le yakala (TF-IDF'in kaçırdığı kelime-SIRASI),
nested-OOF modele çevir, blend'e (ridge_pos soft-voting) üye ekle, gücüne göre +/− puan" önerdi. Regex
paleti y-siz (alan-bilgisi, lexicon ruhu). **Tanı: örüntüler CANLI** — daha-fazla (%46) y-farkı **−6.7**,
övgü-ancak-negatif (%31) −3.6, potansiyel-tuzağı (%9.6) −1.9. İki varyant nested-OOF ridge (txt_ridge
deseni; fold-safe) → blend 11. üye + paired-gate:

| Varyant | standalone rw | blend EKLE Δ | paired-gate |
|---|---|---|---|
| **A saf-regex** (6 söz-dizimi pattern + ton/sayım) | 213.64 | +0.031 | 6/15, p=0.035, CI [−0.018,+0.080] → ELENDI |
| **B regex×skor** (balon=övgü∧düşük-teknik, sessiz-cevher=nötr∧yüksek-skor) | 194.38 | +0.012 | 4/15, p=0.025, CI sıfırı kapsıyor → ELENDI |

**RED — regex sinyali GERÇEK ama YENİ DEĞİL:** örüntüler tanıda canlıydı (−6.7'ye kadar) AMA o sinyal
mevcut metin kanallarında ZATEN var — txt_ridge "daha fazla"yı TF-IDF görüyor, e5 embedding cümle-düzeyi
söz-dizimini kodluyor. Regex bunların kaba/gürültülü alt-kümesini çıkarıyor → soft-voting düşük ağırlık
verse de blend'e gürültü ekledi (kötüleşti). **Metin kanalının HER formu artık tüketildi:** TF-IDF
word/char (txt_ridge/txt_rich/txt_ridge_wc), e5 embedding, lexicon-10, 10 elle-feature, ve şimdi regex
söz-dizimi + regex×skor — hepsi denendi; çıkarılabilir metin sinyali e5+txt_ridge'de doygun. Mentor metni
belirsiz (aynı kelimeler her y'de), keskinlik yeni bilgi getirmiyor. Blend 84.0212 değişmedi.

### Ek (kullanıcı önerisi) — Asimetrik/quantile loss (340 aşırı-tahmini dolaylı hedefle) → RED

Kullanıcı "340'ı doğrudan hedefleyemiyorsak (y-siz ayırt edilemez — 4 kez kanıtlı) dolaylı yoldan"
deyince asimetrik loss seçildi. **Gerekçe veriden:** 340'ın 189'u aşırı-tahmin (model yüksek dedi);
rw-katkı aşırı 446K vs düşük 394K (%13 fazla). Asimetrik-Huber custom-obj (r=pred−y; r>0 overpredict
çarpan 2(1−τ), r<0 çarpan 2τ; τ<0.5 → düşük-yanlı), τ∈{0.35..0.50}, init_score=fold-mean:

| τ | repeat-0 rw | aşırı-tahmin (baz 189) |
|---|---|---|
| 0.50 (custom-obj simetrik baz) | 92.77 | 223 |
| **0.45 (en iyi)** | **91.80** (custom-sym üzerine −0.97) | 204 |
| 0.40 | 93.22 | 178 |
| 0.35 | 94.05 | 149 |
| — gerçek ht (dahili-huber, MUTLAK ref) | **86.63** | 189 |

**RED — çift-katmanlı ders:** (1) Asimetri MEKANİZMASI çalışıyor: τ↓ → aşırı-tahmin monoton düşüyor
(223→149) ve custom-obj-simetrik bazını τ=0.45'te −0.97 iyileştirdi — yani fikir teknik olarak işliyor.
(2) AMA net RED: en iyi τ=0.45 (91.80) mutlak ht 86.63'ten +5.17 kötü. İki sebep: (a) custom-obj+
init_score, LightGBM'in dahili boost_from_average huber'ini birebir üretemiyor (simetrik baz 92.77,
kurulum sınırı); (b) ASIL mekanizma — model ZATEN dengeli (rw-ort resid +0.043); asimetri 189 aşırı-
tahmini biraz kırpıyor (204) ama dengeli 9500 satırı sistematik kaydırıyor → çoğunluğa zarar > azınlık
kazancı (τ≤0.40'ta net pozitif). **340'ı dolaylı hedeflemenin temel sınırı:** dolaylılık tüm popülasyona
dokunur, dengeli %95'i bozar; Huber zaten optimal dengeyi bulmuş. Blend 84.0212 değişmedi.

### Ek (kullanıcı önerisi) — Kural-tabanlı post-process (feature aşırılığı → skor kaydır) → RED (tetik analizi)

Kullanıcı "aşırı sinyal veriyorsa kuralla yaptır" deyince hard if-then kuralları (10-önerinin kural
versiyonu) tetik-analiziyle test edildi: kural tetiklenen satırların ort residual'i (fold-safe blend_oof)
popülasyondan farklı mı? Anlamlı = |sapma|>3 ∧ n>30.

| Kural | n | ort resid | pop-dışı sapma |
|---|---|---|---|
| coding>90 & intern=0 | 242 | −0.16 | **−0.28 (≈0)** |
| pq<40 & ti<40 (çift-zayıf) | 421 | +1.00 | +0.88 (≈0) |
| coding>85 & pq<40 (uyumsuz) | 265 | +0.75 | +0.63 (≈0) |
| cgpa>3.5 & intern=0 & ghr<2 | 37 | +2.28 | +2.16 (küçük-n gürültü) |
| aşırı-övgü(≥4) & tech<50 | **0** | — | tetiklenmedi |
| tech>85 & metin-kısa | 19 | −2.73 | −2.85 (küçük-n gürültü) |

**RED (tetik analizi post-process'i gereksiz kıldı, segment-kalibrasyon emsali):** Büyük-n kuralların
(242/421/265) sapması ≈0 → GBDT o satırları ZATEN doğru tahmin ediyor; kural uygulasak doğru tahminleri
bozardık. Anlamlı-görünenler küçük-n (37, 19) = istatistiksel gürültü. **Mekanizma:** GBDT senin koyacağın
her if-then kuralını (coding_score × internship_count split'leri) veriden OPTIMAL eşiklerle zaten
öğrendi; elle keyfi eşik (90, 0, −10) daha kötü ve global uygulanır → istisnaları da cezalandırır.
Olgun GBDT'de elle kural her zaman redundant ya da zararlı. Blend 84.0212 değişmedi.

### Ek (TAKIM ENTEGRASYONU) — arkadaş (ahmet branch) tekniklerini bizim CV'de süzme → 3 RED

Kullanıcı arkadaşının çalışmasını (public 82.99 team_blend) entegre etmek istedi. ** courteam_oof (55/45
karışım) bizim CV'de:** rw-OOF 83.56 (sayısal −0.46) AMA paired GEÇMEDİ (11/15, p=0.137, CI [−1.65,+0.66]
sıfırı kapsıyor); ayrıca arkadaş `KFold(10,shuffle)` kullanmış (bizim RepeatedStratified-5×3 DEĞİL) →
fold-hizasız, birebir kıyas geçersiz; team_blend 46-submission'dan public'te en iyisi = public-overfit
şüphesi. Arkadaşın **gerçek model OOF'ları** branch'te yok (data/cache gitignore, GPU). Erişilebilir
3 CPU-feasible tekniği bizim folds.parquet + nested paired-gate ile süzüldü:

| Arkadaş tekniği | standalone rw | blend EKLE Δ | paired-gate |
|---|---|---|---|
| **segment-yıl TE** (rol/tier/hobby/sosyal × yıl, fold-içi sm=20) | 86.54 (+0.76 vs ht) | +0.018 | 5/15, p=0.37 → ELENDI |
| **quant** (LGBM quantile q=0.35/0.5/0.65 ort) | 86.68 | +0.034 | 2/15, p=0.007 → ELENDI |
| **catmae** (CatBoost MAE, full-15) | 87.15 | +0.0005 | 6/15, p=0.98 → ELENDI |

**RED — neden arkadaşta işe yarayıp bizde yaramadı:** Arkadaşın v15 notu segment-TE'yi 89.94→88.53 kazanç
diyor; bizde +0.018 ZARAR. Çünkü **bizim taban onunkinden güçlü** — yıllar ham feature + native-kategorik
+ e5 + huber zaten içeride; onun "kazanç"ları zayıf-tabanda kaçırılan sinyalin telafisi, bizim olgun
taban onları zaten almış. quant/catmae GBM ailesi → catboost_full/huber ile yüksek korelasyon, ridge
meta ~0 ağırlık verdi. Arkadaşın gerçek potansiyeli GPU-transformer'larında (BERTurk/mDeBERTa/XLM-R,
e5'ten farklı) ama lokal GPU yok → koşulamadı. **Karar: public 82.99 tek-başına + fold-uyumsuz +
public-overfit şüphesi → CLAUDE.md gereği SUB-2 84.02 (CV-doğrulanmış) KALIR.** Blend değişmedi.

### Ek (kullanıcı önerisi) — Confidence-thresholded + binned pseudo-labeling → RED

Klasik PL RED'i (+0.29) üzerine kullanıcı "en-emin (bariz) satırları pseudo-label, ayrıca tam-değer
yerine binning (aralık)" önerdi. Güven = bizim 5 base test-pred std'si (y-siz). Fold-safe öğretmen
(fold-train'den) → en-emin %K test pseudo-label → öğrenciye ağırlıkla ekle. repeat-0:

| Varyant | en iyi delta vs 86.6303 |
|---|---|
| **tam-değer PL** (K∈{0.1,0.2,0.4}, w∈{0.3,1.0}) | **+0.17** (K=0.2,w=0.3) … +2.07 (en kötü) |
| **binned PL** (kullanıcı fikri, bins=10/20) | +0.24 / +0.25 (tam-değerden DAHA kötü) |

**RED — hepsi:** En-emin satırlar ZATEN doğru tahmin ediliyor → pseudo-label yeni bilgi getirmiyor,
öğrenci kendi tahminini tekrar öğreniyor (distilasyon değil tautoloji). Binning kök-sorunu çözmüyor,
kuantizasyon-kaybı ekliyor (tam-değerden kötü). "Az ve emin" sezgisi doğru yöndeydi (K=0.2,w=0.3 en
hafif zarar) ama yetmedi: test aynı sentetik üreticiden, sömürülecek ekstra yapı yok. Blend 84.0212 değişmedi.

### Ek (kullanıcı önerisi) — Data-oynamaları: feature-binning + mixup + hafif-budama → 3 RED

Kullanıcı "data üzerinde başka ne yapılabilir" deyince 3 denenmemiş data-oynaması (lgbm_full_ht tabanı,
repeat-0 fold-safe):

| Teknik | repeat-0 delta | full-15 doğrulama | Karar |
|---|---|---|---|
| **A) feature-binning** (6 skor → 5-bin kategorik ek-feature) | −0.061 | (eşik-altı, gürültü-bandı) | RED |
| **B) mixup** (x'=λxᵢ+(1−λ)xⱼ, y' aynı; λ~Beta(0.2)) | +0.347 | (kötüleşti) | RED |
| **C) hafif-budama** (en düşük-imp 3 missing-flag çıkar) | **−0.190 (UMUT)** | **full-15 +0.067 (BUHARLAŞTI)**; blend İKAME 4/15 CI⊃0, EKLE 6/15 p=0.69 | RED |

**RED — ve C `f_tshaped_std`'nin birebir ikizi:** budama repeat-0'da −0.19 ile en parlak göründü;
full-15'te +0.067'ye döndü (3 missing-flag'in düşük-imp'i tek-repeat shuffle gürültüsü), blend-gate
geçmedi. Repeat-0→full-15-gate disiplini bir kez daha gerçek-serap ayrımı yaptı. **Mekanizma (üçü):**
GBDT zaten optimal binning yapıyor (açık-bin redundant); mixup'ın lineer-interpolasyonu sentetik-nonlineer
hedefe uymuyor (sahte satırlar gürültü); düşük-imp feature çıkarmak (Occam) bu gürültü-seviyesinde gerçek
kazanç vermiyor. Data-tarafı tüm müdahaleler (FE/temizleme/çoğaltma/encoding/budama) artık tüketildi.
Blend 84.0212 değişmedi.

### 5. TUR — DEEP-RESEARCH (profil-eşi kazanan çözümler) → 2 yeni aday ölçüldü, 2 RED

Cite-li deep-research (PetFinder/CommonLit/Feedback-ELL/ICR/Playground kazananları, bizim
[0,100]-sansürlü + metin+tablo + sentetik + shift profiline eşlenmiş). Rapordaki tekniklerin ÇOĞU
zaten denenmiş (embeddings→SVR=RED 104.93, pseudo-labeling=RED, 2-aşama-P×rank=RED p100, adv-validation
=yıl analizi, importance-weighting=recency_weights, robust-loss=huber). **2 gerçekten denenmemiş +
düşük-risk aday ölçüldü:**

| Aday (kaynak) | Ölçüm | Karar |
|---|---|---|
| **PetFinder post-process** (Giba 1st: üst-clip + global çarpan ×1.032, MSE-shrinkage düzelt) | global çarpan tanı = **1.00060** (shrinkage YOK); nested: mult +0.014, clip −0.018 (9/15 p=0.13 CI⊃0), both −0.008 | RED |
| **BCE-on-normalized-target** (PetFinder/PETS-SWINF: y/100, cross_entropy obj, sigmoid çıktı, MSE-validate) | leaves15 **+1.11**, leaves31 +2.07 vs huber 86.63 | RED |

**RED — neden bu kazanan-teknikler bizde işe yaramadı:** (a) PetFinder post-process'in çekirdeği
SVR'ın MSE-shrinkage kusurunu düzeltmek; bizim global optimal çarpan **1.0006** = blend ZATEN kalibre
(ridge_pos + recency-weight), düzeltilecek bias yok (kalibrasyon-testi emsali). (b) BCE/sigmoid [0,1]
sınırını respekt ediyor AMA sınır-dostu ≠ MSE-optimal; sigmoid-saturasyonu orta-yoğunlukta (76-90)
çözünürlük kaybı, log-link additive-latent'e uymuyor (Tweedie/fractional-logit emsali). **Fizibıl-dışı
adaylar:** snap-features (orijinal-veri gerektirir — bizde tamamen sentetik, altta public original yok);
AWP + per-target attention-pooling + DeBERTa/BERTurk fine-tune (GPU gerektirir, lokal GPU yok — arkadaşın
transformer'larıyla örtüşür). **Araştırma esas olarak mevcut yaklaşımı DOĞRULADI:** TF-IDF→meta-feature
füzyon, iki-aşama, adv-validation, embedding+blend, importance-weighting hepsi kazanan çözümlerde var ve
biz zaten kullanıyoruz/denedik. Blend 84.0212 değişmedi.

### 6. TUR — DEEP-RESEARCH v2 (negatif-prompt, denenmemiş + CPU-feasible) → 4 aday, 4 RED

Negatif-prompt'lu cite-li araştırma (denenen ~25 taktik yasaklandı → sadece yeni/CPU-feasible).
8 yeni aday önerildi; EV-sırasıyla en yüksek 4'ü ölçüldü (cleanlab/LightGBMLSS probe-only pip-kurulum,
requirements.txt DOKUNULMADI):

| Aday (kaynak) | Ölçüm | Karar |
|---|---|---|
| **Caruana GES** (greedy ensemble selection + bagging, MSE-doğrudan, Ridge/NNLS değil) | nested rw **84.82** vs ridge 84.02 (+0.80); paired 2/15 CI tamamen pozitif | RED |
| **Generator-artifact features** (decimal-digit/value-freq/off-lattice; orijinal-veri gerektirmez) | TANI: skorlar düzgün 2-decimal, frac×100 düzgün 0-99 dağılmış → **lattice-izi YOK** | Fizibıl-dışı (artifact yok) |
| **LightGBMLSS ZABeta** (zero-adjusted Beta dağılım-boosting; ==100→z=0 atom-kütle yapısal) | repeat-0 rw **257** (overflow, yakınsamadı; samples-E[z] de düzeltmedi) | RED — bu bütçede viable değil (kurtarma=HP-balıkçılığı) |
| **cleanlab regression** (confident-learning surprise-row; ilk koşu A-feature −8.14 = **SIZINTI** [cl_quality y-tabanlı]) | **nested-fold-safe**: down-weight **+2.68**, quarantine **+0.36** | RED |

**RED — ve cleanlab nested fold-safe DERSİ (kritik):** İlk koşuda down-weight −0.86 / feature −8.14 "iyi"
göründü; bunlar cl_quality'nin y-tabanlı olmasından gelen SIZINTIYDI. **Nested-fold-safe yapınca (her dış-fold
için cleanlab o foldun train'inde) iyileşme ZARARA döndü** (down-weight +2.68). İki ayrı katman: fold-safe
ÖLÇÜM-dürüstlüğünü sağlar (sızıntı yok); ama dürüst ölçüm gerçeği gösterdi — surprise-row atmak/down-weight
GENELLEMEZ, çünkü (a) test'te o satırlar VAR (train'den atmak test'ten silmez), (b) Huber zaten onları
optimal dizginliyor, cleanlab'in ekstra down-weight'i üst-üste-düzeltme. cleanlab'in DEĞERİ tahmin değil
TANI: bulduğu surprise-row'lar bizim 340 ile **%69.7 örtüştü** (model zaten doğru teşhis ediyor — jüri
kanıtı). GES: ridge_pos 10-modelde zaten MSE-optimal, GES'in discrete-seçimi kaba. ZABeta: SLTB emsali —
sansürlü-kütle yapısal modeli toplam-MSE'de huber-point'i geçemiyor (zaten yakınsamadı). **Probe paketleri
(cleanlab 2.9.0, lightgbmlss) kuruldu ama requirements.txt + kanonik pipeline DEĞİŞMEDİ.** Blend 84.0212 değişmedi.

### 7. TUR — REJECTED-BUNDLE testi (metodoloji-denetimi B-9 kapanışı) → ELENDI, gate aklandı

Metodoloji denetimi şu endişeyi kaldırmıştı: paired-gate (12/15 & p<0.01 & CI<0) küçük-GERÇEK
kazançları tek tek reddediyor olabilir; bırakılan kümülatif kazanç −0.2..−0.4 olabilir. **Tek
ön-kayıtlı test:** tek tek gürültü-bandında olan adaylar gerçekse TOPLAMLARI net görünür olur.
Ön-kayıtlı kriter (defterden, objektif): reddedilmiş üye-adayları arasında paired iyileşen≥11/15
olanlar = **kNN-e5** (12/15, p=0.013) + **lgbm_num_h** (11/15, p=0.012). İkisi BİRLİKTE havuza:

| Ölçüm | Değer |
|---|---|
| base blend → +bundle | 84.0212 → 83.9491 (**−0.072**) |
| **paired (tek test)** | **11/15, t=−2.57, p=0.022, CI [−0.203,+0.056] sıfırı kapsıyor → ELENDI** |
| tanı (tek tek) | +knn_e5 −0.019, +lgbm_num_h −0.070 |

**SONUÇ — B-9 endişesi EMPİRİK kapandı:** bırakılan kümülatif kazanç ~−0.07 (tahmin edilen
−0.2..−0.4 değil) ve o bile gürültüden ayırt edilemiyor. Gate'in sigorta primi sandığımızdan ucuzmuş;
gate doğru çalışıyordu. Dürüstlük notları: (a) kNN-e5 yeniden-üretimi (artefakt 3.turda silinmişti;
e5-uzayı kNN-target, k=50 gate-kör repeat-0 taramayla) 3.tur kurulumuyla birebir olmayabilir
(katkı −0.019 vs eski −0.060) — ama bundle taşıyıcısı lgbm_num_h eski ölçümle tutarlı (−0.070) ve
tek başına da CI'ı kapatamıyor; (b) güç-artırma (+3 repeat → 30 hücre) teknik olarak mümkün ama
−0.07 deltası 30 hücrede de CI'ı kapatamaz (maliyet saatler, beklenen kazanç ~0) → yapılmadı.
Blend 84.0212 değişmedi.

### 8. TUR — BLOCKED-BOOTSTRAP gate-denetimi (metodoloji #1: istatistik-temeli dürüstleştir)

İtiraf: paired-t 15 hücreyi BAĞIMSIZ varsayar ama 3 repeat aynı 10k satırı paylaşır (etkin-df ~4)
→ p-değerleri iyimser; mevcut satır-bootstrap da satırları bağımsız çekip fold-yapısını kırar. Düzeltme:
gate-kabullerini 3 bootstrap moduyla yeniden denetle (row=mevcut, blocked=15 hücreyi yeniden örnekle,
foldcluster=repeat-içi 5 foldu blok). Test: "havuz−üye" vs "havuz" (leave-one-out).

| Kabul | leave-1-out delta | ROW-CI | BLOCKED-CI |
|---|---|---|---|
| e5_ridge | −0.308 | [−0.54,−0.08] | **[−0.43,−0.18] p=0.000** ✓ |
| mm | −0.537 | [−0.95,−0.13] | **[−0.77,−0.31] p=0.000** ✓ |
| lgbm_full_ht | −0.078 | [−0.16,−0.002] | **[−0.11,−0.04] p=0.000** ✓ |
| **lgbm_full_h** | **−0.027** | [−0.10,+0.04] | **[−0.062,+0.006] p=0.057 (sıfırı KAPSIYOR)** |

**ŞAŞIRTICI BULGU (beklentinin tersi):** blocked-CI satır-CI'dan DAHA DAR çıktı — satır-bootstrap
fold-içi yapıyı kırıp recency-ağırlık varyansını ŞİŞİRİYOR (fazla muhafazakar). Blocked fold-bütünlüğünü
koruyup GERÇEK (daha düşük) varyansı gösteriyor. Yani gate'in istatistik-temeli sandığımdan SAĞLAM:
e5/mm/ht blocked altında p=0.000, halüsinasyon değil — dışsal public-gap kalibrasyonuyla uyumlu.

**GERÇEK DENETİM BULGUSU — lgbm_full_h:** blocked-CI sıfırı kapsıyor (leave-one-out anlamsız). Sebep:
h kabul edildiğinde (mm'li havuza) anlamlıydı, sonra ht eklendi ve ht huber-bilgisini zaten taşıyor →
h nihai havuzda redundant. **Occam testi:** h çıkarınca blend 84.0212 → 84.0478 (**+0.027, KÖTÜLEŞİR**).
delta pozitif (h hafif katkı veriyor) ama anlamsız. **KARAR: h TUTULUR** — çıkarmak kötüleştiriyor (Occam
"eşitlikte basit" der, burada eşitlik yok); ayrıca kalibre-kanıtlı SUB-2'yi 0.027'lik jest için riske
atmaya değmez. Bulgu belgelendi, aksiyon yok. Blend 84.0212 değişmedi.

**NET (gate güvenilirliği):** Blocked-bootstrap, paired-t'nin sahte-bağımsızlık endişesini KAPATTI —
gerçek varyans daha düşük, kabuller daha güçlü. Gate "halüsinasyon değil, güçlü-kontrollü"den
"istatistik-temeli denetlenmiş"e yükseldi. Kalan açık: güç-profili (false-reject oranı) + 2-nokta
public kalibrasyonu (red-edilen aday public-doğrulaması — submission ister).

### 9. TUR — GATE GÜÇ-EĞRİSİ (metodoloji #2: false-reject profilini ölç)

Soru: gate false-accept'i ~0 (sıkı); ama gerçek bir küçük kazancı (−0.1) yakalayacak GÜCÜ var mı?
Sentetik enjeksiyon (blend'e fold-safe y-yönlü kontrollü kazanç) DENENDİ ama BAŞARISIZ: tüm alpha'larda
delta +0.045 sabit kaldı — blend zaten kalibre olduğundan ridge(blend→y) düzeltmesi ≈ kimlik, geriye
sadece gürültü kaldı (bu da AYRI bir kalibrasyon kanıtı: blend'i y'ye yaklaştırmak mümkün değil çünkü
zaten optimal). Bunun yerine GERÇEK güç-eğrisi geçmiş ölçümlerden çıkarıldı (delta'ya göre sıralı):

| aday | gerçek-delta | iyileşen | sonuç |
|---|---|---|---|
| lgbm_full_h | −0.140 | 13/15 | **KABUL** |
| NM-meta | −0.123 | 12/15 | RED |
| yıl-etkileşim | −0.099 | 10/15 | RED |
| **lgbm_full_ht** | **−0.078** | **13/15** | **KABUL** |
| lgbm_num_h | −0.074 | 11/15 | RED |
| knn-e5 | −0.060 | 12/15 | RED |

**BULGU — gate delta-BÜYÜKLÜĞÜYLE değil TUTARLILIKLA karar veriyor:** −0.078 (ht) geçti ama daha BÜYÜK
−0.123 (NM) reddedildi. Ayırt eden delta değil, iyileşen-hücre + CI. Bu tam istenen: (a) **false-reject
düşük** — gerçek+tutarlı küçük kazanç (−0.078, 13/15) yakalanıyor; (b) **false-accept düşük** — büyük-ama-
dağınık kazanç (−0.123, 12/15, CI⊃0 = meta-overfit imzası) reddediliyor. Reddedilenlerimiz (−0.06..−0.12)
delta olarak ht'ye yakın AMA hepsi 10-12/15 TUTARSIZ → haklı red; biri 13/15 tutarlı olsaydı ht gibi geçerdi.
**Gate'in gücü "küçük kazanç yakalama" değil, "gerçek-tutarlı vs büyük-gürültülü ayrımı" — ve bunu doğru
yapıyor.** Gate denetimi (3 tur: bundle B-9 + blocked-bootstrap #1 + güç-eğrisi #2) TAMAMLANDI: gate
halüsinasyon değil, istatistik-temeli sağlam, güç-profili doğru. Blend 84.0212 değişmedi.

## NİHAİ KARAR (6. tur sonrası)

**Blend 84.0212; altı tur (bilgi-seti + eğitim-mekanizması + envanter + düşük-EV + takım-entegrasyonu +
profil-eşi-araştırma) boyunca ~30 taktik ailesi fold-safe ölçüldü, hepsi RED.** Deep-research (2 tur)
kazanan-çözüm tekniklerinin ya zaten kullanıldığını ya da bu olgun tabanda işe yaramadığını doğruladı.
Lokal-CPU'da denenmemiş meşru taktik kalmadı. Tek açık yüksek-potansiyel: GPU-transformer'lar (BERTurk/
mDeBERTa bizim folds ile, e5'ten ortogonal) — lokal GPU yok. SUB-1 (catboost_full 86.4149) + SUB-2
(blend 84.0212) FİNAL.

## NİHAİ KARAR (4. tur sonrası)

**Blend 84.0212; bilgi-seti + eğitim-mekanizması + sistematik-envanter + düşük-EV-kuyruğu
uzaylarının DÖRDÜNDE de doğrulanmış tavan.** 4. tur ayrıca karar metodolojisini BAĞIMSIZ temporal
split'le çapraz-doğruladı (Sonda 6: rw-OOF sıralaması recency-dilimle ρ=0.96 teyitli). Denenmemiş
lokal-CPU taktiği kalmadı. SUB-1 (catboost_full 86.4149) + SUB-2 (blend 84.0212) FİNAL.

## NİHAİ KARAR (3. tur sonrası)

**Blend 84.0212; bilgi-seti + eğitim-mekanizması + sistematik-envanter uzaylarının ÜÇÜNDE de
doğrulanmış tavan.** Üye-düzeyi kazançlar hâlâ bulunabiliyor (kNN-e5 −0.23 gerçekti) ama blend bu
bilgileri mevcut üyelerden zaten alıyor — pass-through gate'i geçemiyor. SUB-1 (catboost_full
86.4149) + SUB-2 (blend 84.0212) FİNAL.

## NİHAİ KARAR (2. tur sonrası — tarihçe)

**Blend 84.0991 artık HEM bilgi-seti HEM eğitim-mekanizması uzayında doğrulanmış tavandır.**
Ampirik dış doğrulama da geldi: public LB 84.1709, gap +0.072 (=0.025σ) YEŞİL — rw-OOF metodolojisi
gerçek dünyada 0.07 hassasiyetle kalibre. SUB-1 (catboost_full 86.41) + SUB-2 (blend 84.0991) FİNAL.


## EK (4. tur öncesi) — UPSTREAM/VERİ-MANİPÜLASYONU kapısı (kullanıcı sorusu) → KAPALI

**Yapısal gerçek:** GBDT'ler monoton feature-dönüşümlerine değişmez (log/scale/winsorize/rank → birebir
aynı model) → "veri düzeltme" ailesinin büyük bölümü bu stack'te tanım gereği no-op.

| Upstream probe | Ölçüm | Karar |
|---|---|---|
| Veri tutarlılık denetimi (aralık/mantık ihlali) | **0 ihlal** (negatif yok, >100 yok, interviews≤applications ✓, yıl ilişkileri ✓) | düzeltilecek şey YOK |
| MNAR 0-impute (internship_duration NA→0) | rw 86.63→**86.78 (+0.15)** | RED — NaN-yönü esnekliğini öldürüyor |
| Fold-median impute (7 NA kolonu) | rw 86.6268 (−0.003) | no-op — native NaN zaten kapsıyor |

Satır temizliği (dup yok 4.8σ; outlier-atma→huber), etiket-düzeltme (soft-relabel), encoding'ler,
metin ön-işleme: önceki turlarda ölçülüp kapatıldı. Upstream kapısı bütünüyle KAPALI.

---

## KAPISIZ TAM TARAMA — overfit-kapisi KALDIRILMIS exhaustive aramanin sonucu (~40 teknik)

> **Baglam:** Kullanici "overfit kapisini bos ver, 82.24/82.12 altina ne kadar inersek o kadar iyi"
> dedi. Tum kabul kapilari (0.25*std, paired-test) KALDIRILDI; tek olcut "nested rw-OOF dustu mu".
> Amac: kapinin GERCEKTEN gercek kazanci mi engelledigini yoksa gurultuyu mu eledigini test etmek.
> **SONUC: kapi dogru calisiyormus. Kapisiz bile, GERCEK-nested olarak combinatorial 82.0164 ASILAMADI.**

### Karar metrigi kalibrasyonu (bu turda olculdu)
- combinatorial nested rw-OOF **82.0164** -> public **82.1221** (gap **+0.106**). rw-OOF private-durust;
  public'i ~0.1 iyimser tahmin ediyor (saglikli, yesil bant). Public 81.9X icin nested ~81.85 gerekir.

### Cephe 1 — Post-process zoo (11 metod, blend 82.24 uzerine, kapisiz)
| Metod | nested delta | Karar |
|---|---|---|
| **xlmr confidence-gating** | **−0.044** | GERCEK (tek kazanc; public 82.24→82.11 kanitli) |
| winsor (alt-kuyruk trim %0.25) | −0.002 | gurultu (mikroskobik) |
| linscale/isoglobal/spline/binshrink/topgate/round100/gamma/multigate/bidirgate | 0 .. +0.43 | hepsi ELENDI (blend zaten kalibre; asagi-yon + cok-model gate ZARARLI) |

### Cephe 2 — CPU-zoo (8 yeni ortogonal base, kapisiz)
ExtraTrees/kNN-PCA/Ridge-poly/LGBM-quantile/CatBoost-derin/LGBM-DART/target-encode-GBDT/HistGBM-L1.
**Hepsi RED.** Greedy z_catdepth'i sectiyse de tum-OOF ridge agirligi = **0.00000000**. 8 adayin tamami
base'e corr **0.83–0.99** (hicbiri ortogonal degil). Saf CPU-tabular uzayi **doygun** — cekilecek yeni sinyal yok.

### Cephe 3 — Reddedilenler dirilisi (eski "kapi gecmedi" adaylari, kapisiz, guncel tabanda)
| Aday | eski delta (84.24 taban) | yeni delta (82.24 taban) | sonuc |
|---|---|---|---|
| Nelder-Mead meta | −0.123 | **+0.0007** | redundant olmus (L2 cezasi geneller) |
| kNN-target e5 | −0.060 | +0.011 | redundant (e5_ridge zaten blend'de) |
| ridge+P×yil | −0.099 | +0.000008 | redundant (xlmr/ourteam_tf yil-biasi yakaliyor) |
| seed-bagging ×5 | −0.34 std | +0.0018 | redundant (lgbm_full_ht zaten blend'de) |
| f_tshaped_std | repeat-0 −0.194 | −0.011 | mikroskobik kaldi |
| Sigma-clip (censored-normal) | −0.026 | −0.0026 | mikroskobik kaldi |
**Bulgu:** 84.24-tabanindaki "kacirilan kazanclar" GERCEK degil REDUNDANSIN giderilmesiydi — o sinyali
sonradan daha iyi modeller (xlmr/ourteam_tf/e5) zaten aldi. **Kapi dogru elemis.**

### Cephe 4 — Random-search (4000 tur, 8 DS taktigi: subset×NNLS/Dirichlet/power-mean/rank/trim/...)
"En iyi 82.06" cikti AMA TESHIS: kirli `rv_ftshaped` artefaktindan (in-sample blend kopyasi, corr 0.9999)
geliyordu = winner's-curse + kirli-OOF. Kirli artefakt havuzdan cikarildi (.INSAMPLE_KIRLI). Gercek kazanc YOK.

### Cephe 5 — Gating-maxout (6 aile × yon × param-grid × cok-asama zincir, NESTED)
En iyi GERCEK-nested zincir: `txt_ridge↑→xlmr↑→mm↓→xlmr↑→e5↑` = **blend_gatemax 82.0504** (−0.189 vs ham blend).
**Yeni bulgu:** txt_ridge gate sinyali xlmr'den GUCLU (−0.098 vs −0.044 tek-hamle). Asagi-yon yine 0'a indi (teyit).
81.5 ULASILMADI (raporun kendi notu): "gate tek-kaldirac blend'i ~82.0-82.1 bandina indirir; otesi daha guclu base gerektirir."

### Cephe 6 — combo14 (14-model base + gating; en kritik)
| Varyant | nested rw | tur |
|---|---|---|
| **combo14_gatemax** (odunc param, 12-model'den) | **81.9034** | ⚠️ HAFIF SIZINTILI (param o OOF'u gormustu) |
| **combo14_agg** (in-sample agresif 12-asama) | **81.7746** | 🚨 IN-SAMPLE SAHTE (stage 7-12 saf gurultu-ezber) |
| combo14_gateopt / basegate (bu-base-OZEL nested) | 82.0771 | ✅ GERCEK-nested (combinatorial'dan +0.06 kotu) |
**Kritik ders:** 81.90/81.77 cazip ama GUVENILMEZ. Dürüst-nested (param hucre-disindan) yapinca **82.0771**'de
duruyor. Aradaki 0.30 fark = saf overfit/sizinti. In-sample distance-kNN rw=0.0000 (ezber imzasi).

### Cephe 7 — 99→100 snap (kullanici fikri)
blend≥thr olanlari 100'e yuvarla. Her esikte rw YUKSELDI (thr=99: +0.008). Snap'lenen 350 satirin %89'u
gercekten 100 ama %11'i (39 satir) degil; onlari 100'e itince MSE patliyor. RED (pp_round100/topgate ile tutarli).

### Cephe 8 — Meta-hunt (6 GERCEK-nested meta-ogrenici ailesi, ridge_pos disi)
| Meta | en iyi nested rw | combinatorial'a gore |
|---|---|---|
| mh_regularized (Lasso α=0.01) | 82.0786 | +0.062 (en yakin, GECEMEDI) |
| mh_constrained (free WLS) | 82.1394 | +0.123 |
| mh_segmented (yil-bazli) | 82.2739 | +0.258 |
| mh_2layer (aile-stack) | 82.2953 | +0.279 |
| mh_nonlinear (siki-reg GBDT-meta) | 83.0523 | +1.04 (overfit; in-sample 0.0 ezber) |
| mh_blendfamily (power-mean) | 83.9272 | +1.91 |
**ridge_pos zaten optimal meta.** Lasso/ElasticNet/Huber/WLS/non-linear/segment/2-layer — hicbiri gecemedi.

## NIHAI KAPANIS (kapisiz tam tarama)
~40 teknik, 8 cephe, overfit-kapisi KALDIRILMIS. **GERCEK-nested olarak combinatorial 82.0164 (public 82.12)
ASILAMADI.** En yakin durust deneme 82.0504 (gating-maxout, +0.034). "Daha asagi" gorunen tek sayilar
(81.90 sizinti / 81.77 in-sample) GUVENILMEZ ve public'te geri tepmesi beklenir. **Model+CPU uzayi tam doygun**
(base 4 kez, gating 1, meta 1 ayri yoldan teyit). Tek acik kaldirac: GPU-transformer (TabPFN, ayri is).
**Kapi dogru calisiyormus** — kapisiz bile gercek kazanc yok; kapinin eledigi her sey gercekten gurultuydu.

---

## GUVEN-SKORU CEPHESI — confidence-aware duzeltme (4 yontem, hepsi RED)

> **Baglam:** Kullanici "her tahmine guven skoru ekleyelim, sonra dusuk-guven satirlarda manipulasyon/
> AI ile residual ogrenip rw-OOF dusurelim" dedi. Guven skoru kuruldu + OOF-valide edildi (GUCLU);
> uzerine 4 farkli duzeltme yontemi nested-test edildi. **Hepsi RED — residual ogrenilemez gurultu.**

### Guven skoru (pred_sub_xlmr_gating.csv) — VALIDE, GUCLU
`confidence = rank01(0.10*C1 + 0.90*C2)*100`. C1=model-anlasmasi(14 base std), C2=tahmin-bolgesi(rank01(pred)),
C3=metin-GBDT-uyum (agirlik 0, yardim etmedi). **OOF validasyon: Spearman(conf,-|err|)=+0.290;
en-yuksek-guven %10 OOF-MSE=17.25 vs en-dusuk %10=137.44 (8x ucurum).** Guven skoru GERCEKTEN calisiyor.
Kok klasore pred_sub_xlmr_gating.csv yazildi (48 kolon: 46 test + pred + confidence) — JURI/ANALIZ icin degerli.

### Bias teshisi (neden duzeltme calismaz)
Guven-bandina gore bias (y-pred): dusuk-guven %10 bias=+0.37 ama |hata|=9.04; yuksek-guven %10 bias=+0.12 |hata|=2.38.
**Bias minik (±0.37), hata buyuk (9+) = problem VARYANS, bias DEGIL.** Dusuk-guven satirlar ex-ante ayirt edilemez.

### 4 duzeltme yontemi (hepsi NESTED, kapisiz, base xlmr-gating 82.1955)
| Yontem | nested rw | sonuc |
|---|---|---|
| Toplu guven-bazli shift/shrink (probe_conf_correction) | 82.3395 (+0.144) | RED |
| Esik-secici manipulasyon (7 esik × 7 kural, probe_conf_threshold) | 82.3406 (+0.145) | RED |
| **AI residual-ogrenici TUM satir** (LGBM: feat+conf+std→resid, probe_resid_learner) | 82.1955 (delta 0, alpha=0) | RED |
| **AI residual-ogrenici SADECE dusuk-guven** (%15/30/50) | 82.1955 (delta 0, alpha=0) | RED |

### KRITIK kanit — residual OGRENILEMEZ
AI residual-ogrenici: Spearman(pred_resid, gercek_resid) = **−0.037 (tum)** / **−0.079 (dusuk-guven)** → NEGATIF.
Model residual'i feature'lardan tahmin EDEMIYOR; optimizer her varyantta alpha=0 ("ekleme") secti. residual std=8.53
≈ forensics "iki ozdes ogrenci y'de 8.3 ayrisir" = INDIRGENEMEZ GURULTU, model hatasi degil.
Esik-secici TANI: en iyi in-sample 82.1504 AMA nested 82.3406 (0.19 fark = saf overfit; nested yakaladi).

**SONUC:** Guven skoru tahmin-guvenilirligini mukemmel olcuyor (8x ucurum) AMA dusuk-guven satirlar
duzeltilemez cunku residual ogrenilebilir-yapi DEGIL gurultu. confidence-aware post-processing'in 4 formu
(shift/threshold-rule/AI-learner-all/AI-learner-lowconf) nested'de RED. LOW_TAIL + noise-floor tezinin
guven-ekseninde 4. dogrulamasi. Guven skoru JURI-ANALIZ degeri tasir; SKOR-duzeltme degeri YOK.

---

## 2026-06-14 (2): RAKIP-CALISMA + PSEUDO-LABELING — ilk GERCEK-nested kazanc

### fezadangelenler (bagimsiz ekip) raporu — bizim TAVAN tezimizi DOGRULADI
Bagimsiz ekip ayni problemde calismis (public ~82.9). Bizim her bulgumuzu teyit etti: tek-sinyal duvari
(BERT+tabular corr 0.997), symbolic-reg formul-yok, dusuk-skor kuyrugu irreducible (decile-0 MSE 221),
outlier yearW(=LB) +0.36 KOTU (covariate-shift), 14 Kaggle-writeup sentetik-kazanma 3 kosulu (orijinal-dataset/
generator-artifact/AutoGluon) — UCU de kapali. AutoGluon onlarda null/redundant -> bizde de denenmedi (atlandi).
Datathon-2024 kazanani (Anil Ozturk) cozumu+sunumu incelendi: CV 5.572->Public 5.856->Private 5.659
(private public'ten IYI, CV'ye yakin) -> CV-otorite/public-kovalama-yok stratejimizin CANLI kanitı.

### Pseudo-labeling (covariate-shift) — ILK gercek-nested 82.0164-alti aday
fezadangelenler pseudo ile LB +0.63 atilim yapmis. Biz 3 pseudo varyanti + sentiment + catshap denedik:
| Yontem | nested blend | overfit-gap | sonuc |
|---|---|---|---|
| **sp_pseudo_blend** (test'i test_blend ile etiketle, PW=0.7) | **81.7754** (-0.389) | +0.136 | GECTI (nuansli) |
| sp_pseudo_cshift (fold-disi labeler, PW grid) | 82.2178 (+0.054) | dev | RED |
| sp_sentiment (20 sablon-text kolon) | 82.2361 (+0.072) | +0.26 | RED |
| sp_catshap (CatBoost SHAP-recursive select) | (40dk kosu, artefakt yok=gecmedi) | — | RED/yarim |

**sp_pseudo_blend = bu oturumun TEK gercek-nested 82.0164-alti adayi.** Kanit: paired 15/15 hucre, t=-13.0,
p=3.2e-9, bootstrap CI [-0.573,-0.201] tamamen sifir-alti (mm/e5/xlmr'den GUCLU); ridge agirlik 0.269 (yolcu degil);
overfit-gap sadece +0.136; sizinti-denetimi temiz (pseudo-y test-satirindan, train-ortusme 0, valid hic gormedi,
corr(y)=0.82 sahte-degil). Submission: sub_sp_pseudo_blend.csv (nested 81.7754).
**NUANS (durust):** literal kapi (0.674) GECMEDI; corr~0.99 GBDT ailesiyle = ORTOGONAL yeni sinif DEGIL, blend'in
KENDI ciktisindan turetilmis covariate-shift adaptasyonu. Public-testi SART (turetilmis-sinyal public'te tutabilir
[fezadangelenler'de tuttu] VEYA combo14_gatemax gibi geri tepebilir [o param-sizintiliydi, bu degil; bu daha saglam]).
KARSIT: sp_pseudo_cshift (gercek labeler) RED -> recency-weight covariate-shift'i zaten telafi ediyor; sadece
blend-etiketli (kendine-referans) varyant tuttu -> public dogrulamasi kritik.

### Bu oturumun TUM redleri (overfit-gap imzasi: in-sample iyi/nested kotu, her cephede +0.1-1.8)
confidence-correction (11), CPU-zoo (8), EDA-FE (8), SMOTE/augment/extraction (9: jitter/mask/mixup/smote dahil),
meta-hunt (6), gating-maxout (6), outlier (buharlasti), 99-snap, AutoGluon (redundant). ~55+ yontem.
**Tek gecen: sp_pseudo_blend (pseudo-labeling).** combinatorial 82.12 + pseudo 81.78 = 2 gercek aday.

---

## OTURUM (2026-06-14, devam) — combo14_gatemax PUBLIC-TEYIDI + 6 yeni aci (A-F) + aci D

### combo14_gatemax public olcumu GELDI -> tahmin DOGRULANDI
Cephe-6'da "HAFIF SIZINTILI, public'te geri tepebilir" denen `combo14_gatemax` artik OLCULDU:
**nested 81.9034 -> public 82.2241 (gap +0.3207, SARI).** Daha basit `combo14` (gating'siz):
nested 82.0164 -> public **82.1221** (gap +0.106 YESIL). Yani gating zinciri nested-OOF'u
0.11 "iyilestirdi" ama public'i 0.10 KOTULESTIRDI -> klasik in-sample/frozen-param sismesi.
**En iyi GERCEK public hala combo14 = 82.1221.** Param-sizinti tezi deneysel kanitlandi.

### Adversarial aday-denetimi (14 baseline-yakin artefakt; survived=0)
Tum "81.7X-82.1X" artefaktlari nested-durustluk + adversarial cururtmeden gecirildi:
- `combo14_agg` 81.77 = in-sample sahte (script kendi itiraf eder), `spbest` 81.78 = pseudo-turetilmis
  literal-gate'ten gecmemis, `combo14_gatemax` 81.90 = frozen-param sizintili (public teyit etti).
- Nested-durust olanlar (`metabest` 82.0165, `basegate`/`gateopt` 82.077, `gm_*` 82.12+) DURUST ama
  baseline'i KIRMIYOR (`metabest` = combo14 ile ozdes nokta). **Havuzda public 82.12'yi kiran YOK.**

### Kullanicinin 6 yeni acisi (A-F) — kesin denetim
| Aci | Durum | Karar |
|---|---|---|
| A — LGBM cross_entropy/xentropy (y/100 fractional logit) | tam denenmis | RED (BCE-on-normalized +1.11/+2.07 MSE; log-link additive-latent'e uymuyor) |
| B — Hill-climbing/Caruana greedy ensemble | tam denenmis | RED (greedy_nnls 82.4765 > ridge_pos 82.2398; soft-shrinkage daha robust) |
| C — Post-cutoff CV (2024-26 tam holdout) | kismen denenmis | REDUNDANT (forward-chaining Spearman 0.964; recency-weight kaymayi zaten telafi eder) |
| D — Kolon-bazli rank-transform blend input | YENI, test edildi | **RED (+3.26/+3.29 MSE, 0/15 hucre, CI tamamen pozitif; rank seviye-bilgisini atiyor -> MSE zarar)** |
| E — classifier x regressor merge/hurdle (y=100 kutle) | tam denenmis | RED (two-stage Bayes-hurdle; bu BIREBIR denenmis varyant) |
| F — Sentetik jeneratoru tersine muhendislik | kismen | REDUNDANT (deterministik yil-kilidi app=grad|grad+1 zaten kesfedildi; baska kural yok) |

### Aci D detayi (src/probe_rank_blend.py) — NESTED-DURUST RED
Baseline HAM-input ridge_pos = 82.2398 (birebir mevcut blend). Rank-transform (inverse-map ve raw,
her ikisi nested + fold-safe): 85.50 / 85.53. Mekanizma: rank shift'e dayanikli AMA tahminlerin
mutlak-seviye bilgisini siler; MSE-regresyonda bu bilgi kritik. mh_blendfamily power/geo/trim
combiner'lari (83.9-85.1) ve z_lgbquant (+0.019 blend faydasi) ile TUTARLI: cikti-dagilimi
manipulasyonu (rank/qmap/quantile/power-mean) bu problemde sistematik MSE-zararli.

### Conformal / optimal-transport (prompt listesi) — MSE-faydasiz
Conformal prediction GERCEKTEN denenmemis AMA point-MSE'yi DEGISTIRMEZ (interval uretir, nokta-tahmin ayni).
Optimal-transport/distribution-matching = cikti-dagilimini test'e esleme = aci D'nin akrabasi (kanitli zarar).
Ikisi de MSE metrigi icin yapisal olarak faydasiz -> test edilmedi (gerekce: D + mh_blendfamily + z_lgbquant kaniti).

### KAPANIS (bu oturum)
CPU/post-hoc/meta/loss/distribution uzayi DOGRULANMIS doygun. combo14_gatemax public-teyidi
"81.X gorunen sayilar guvenilmez" tezini kanitladi. **Tek mesru kalan: GPU ortogonal fonksiyon
sinifi** (BERTurk DOGRUDAN-hedef fine-tune + TabPFN-v2). Kod hazir: colab_berturk_fine.ipynb,
colab_tabpfn.ipynb, src/new_model_gate.py (paired-test gate). Kullanici GPU'da calistirip .npy
getirir -> nested paired-test + public-gap ile yargilanir.

## OTURUM (2026-06-14, "bir sey bulacaksin" emri) — 6 EL-CEPHESI + 4 WORKFLOW-CEPHESI

Yapisal analizden (y additive-dominant R2 0.57->0.62, sigma~8.7) cikan denenmemis catlaklar agresif kazildi.
Karar metrigi nested rw-OOF (taban 82.2398). HER cephe gercek kod + nested protokol.

### El-cepheleri (hepsi RED)
| Cephe | yontem | sonuc | delta/R2 |
|---|---|---|---|
| Gec-yil hard meta-fit | meta SADECE grad>=2024 train ile (test-rejimi) | RED | +0.115 (recency sample-weight zaten daha iyi) |
| Residual ~ base-tahmin nonlinear | HistGBR(base OOF) -> residual | RED | OOF-R2 -0.013, duzeltme +1.30 |
| Residual ~ base pairwise farklar | HistGBR(66 base-cifti disagreement) | RED | OOF-R2 -0.003 (anlasmazlik=gurultu, sinyal degil) |
| Spline-GAM additive base | SplineTransformer(5knot,deg3)+Ridge fold-ici | RED | standalone 103.79, corr(GBDT)0.94, blend +0.054 |

**Ortak bulgu:** Tum base'ler gec-yilda esit oranda (~1.47x) bozuluyor -> kovariat-shift'e dayanikli
ayri base YOK. Base-anlasmazligi residual TASIMIYOR (disagreement-gating ailesi RED'inin kok-nedeni).
Spline-GAM/lineer additive yapiyi GBDT'lerle OZDES yakaliyor (corr 0.94-0.96) -> ortogonal degil.

### TabPFN token-fix
colab_tabpfn.ipynb -> tabpfn==2.0.9 (lisans kapisi YOK) + bagging varyanti (8 subset x 4k context).
GPU'da calistir -> oof_tabpfn/tabpfn_full/tabpfn_bag.npy -> src/new_model_gate.py ile nested gate.
Tek kalan ortogonal FONKSIYON SINIFI (yeni sinyal degil, farkli hata-dagilimi umudu).

### Workflow-cepheleri (4, hepsi RED — winners=0)
| Cephe | nested rw-OOF | delta | karar |
|---|---|---|---|
| Spesifik etkilesim taramasi (top-8 cift carpim/oran -> HistGBR base) | 82.2477 | +0.008 | gurultu bandi |
| Gurultu-yapisi (row-z std/span, kategorik-ici sapma, yil-percentile) | 82.2398 | **0.000** | ridge 0-agirlik = blend zaten iceriyor |
| Segment-uzman blend (blend-tahmin bandina gore ayri ridge_pos agirlik, nested) | 82.347 | +0.107 | kotulesti |
| Kategorik recency-target-encoding (m=30 smooth) -> residual | 82.2398 | **0.000** | ridge 0-agirlik |

Iki cephe TAM 0.000 delta (yeni feature ridge_pos'ta 0-agirlik) = blend her turevi zaten icermis.
**Bu oturum 10 bagimsiz RED (6 el + 4 workflow), onceki ~57'nin ustune.** GBDT-blend'in kacirdigi
ogrenilebilir sinyal YOK (matematiksel kesinlik). KARAR: combo14 public 82.122 bu bilgi-setinin tavani;
finalizasyona gec. TabPFN lokalde GPU yok -> kapsam disi (notebook hazir ama calistirilmiyor).

### Combinatorial-fonksiyon GENISLETME (kullanici: "combinatorial gibi fonksiyonlarla bir sey?")
combinatorial = 14-model ridge_pos + tek xlmr-gating (nested 82.0164 -> public 82.122, gap +0.106 saglikli).
Denenmemis varyantlar test edildi:
| Varyant | sonuc | not |
|---|---|---|
| Backward elimination (14-model'den cikar) | hicbiri iyilestirmedi | 14-FULL (82.1640) optimal, model cikmaz |
| Forward greedy (44-model HAM havuz) | 81.7442 KIRLI | turetilmis artefakt (gm_/sp_/pseudo) leakage = combo14_agg tuzagi; GECERSIZ |
| Forward greedy (26 TEMIZ base, repeat-0 SEC) | 82.0778 | seçim-yanlilik suphesi |
| ^ AYNI subset HELD-OUT (repeat 1,2 dogrula) | **82.1592** | **+0.081 SISME** = secim-yanliligi kaniti |

**KESIN:** OOF'ta model-subset secmek = combo14_gatemax'in public-sismesinin AYNI mekanizmasi
(repeat-0 82.08 cazip ama held-out 82.16'ya geri tepiyor). Hicbir subset/gating varyanti combinatorial
82.0164'u GERCEKTEN (secim-yanlilik-disi) gecmiyor. **Combinatorial-fonksiyon ailesi de DOYGUN.**
combinatorial 82.0164 (public 82.122) bu ailenin saglam tavanidir.

### gate_up fonksiyonu TAM TARAMA (kullanici: "duz uzerine basit fonksiyon neydi")
combinatorial'in "basit fonksiyonu" = gate_up: conf=|model-MU(76.94)|; mask=(conf>=q)&(model>MU);
out=base+mask*a*(model-base). Yani uzman-model ortalamadan yeterince yukarda ise base'i o modele a-orani cek.
Tek-gate NESTED tarama (her gate-model x yon, q,a hucre-disi optimize; base=14-model 82.1640):
| gate-model | up | down |
|---|---|---|
| xlmr (combinatorial'in kullandigi) | -0.092 | +0.030 |
| **e5_ridge** | **-0.114** (en iyi tek-gate) | +0.050 |
| txt_ridge | -0.066 | -0.074 (cift-yon!) |
| ourteam_tf | -0.040 | +0.028 |
| mm | +0.043 | +0.057 |

**Bulgu:** e5-up tek-gate (nested 82.0505) xlmr-up'tan (82.0717) iyi AMA combinatorial (82.0164) yine
0.034 daha iyi (combinatorial = blend_metabest 82.0165 ile ozdes = optimize gating kombinasyonu, tek-gate
degil). txt_ridge/e5_ridge base'de ridge_pos 0.0000 agirlik aliyor (metin lineer-redundant) AMA gate
olarak bilgi katiyor (esik-bazli farkli mekanizma). Bidirectional (gm_bidir) zaten denenmis: 82.1752 (kotu).
**gate_up ailesi DOYGUN:** tek-gate combinatorial-alti, cok-gate zinciri = combo14_gatemax (public 82.224 sisti).
Combinatorial 82.0164 (public 82.122) bu fonksiyon-ailesinin saglam tavani.

### KOMBINATORIYEL GATE-HARMAN (kullanici: "her seyi harmanla, tum gate-modelleriyle, kac cikar")
14 gate-model x 2 yon x kombinatoriyel zincir, 3 strateji + adversarial sisme-testi (kanonik paired-bootstrap).
| Strateji | full_nested_rw | gercek karar |
|---|---|---|
| greedy-zincir (6 gate) | 81.7432 | **CURUTULDU**: kanonik paired CI=[-0.613,+0.039] SIFIR KAPSAR; kazanc top-10 satirda %83 yogun, 1537 satir ZARAR -> combo14_gatemax imzasi (birkac yuksek-recency satir in-sample oturmus) |
| top-k-bagimsiz (4 gate) | 81.7909 | **CURUTULDU**: "held-out" YANILTICI (ayni y'de olctu); DURUST satir-disjoint held-out = 82.0837 = combinatorial'dan +0.067 KOTU. 81.79->82.08 fark = saf grid-optimizasyon iyimserligi. 4 gate, tek e5-up'tan hicbir sey kazandirmiyor |
| nested-tam-zincir | 82.2197 | combinatorial'i gecmedi (durust nested zaten kotu) |

**KESIN KANIT (combo14_gatemax mekanizmasi):** Kombinatoriyel gate-zincir nested-OOF'u 81.74'e "indirir"
AMA bu METRIK-OVERFIT: (q,a) gridi OOF-y gurultusune fit oluyor, kazanc birkac yuksek-agirlikli satirda
yogun, cogunluk satir zarar goruyor. DURUST satir-disjoint held-out + kanonik paired-bootstrap CI ikisini
de cururuttu (CI sifir kapsar, gercek deger combinatorial-ustu). Tek e5-up gate (durust 82.05) 4-6 gate
zincirinden FARKSIZ -> ekstra gate'ler sadece gurultu ezberi. **gate-harman ailesi DOYGUN.**
Combinatorial 82.0165 (public 82.122) HALA saglam tavan; kombinatoriyel genisletme onu GECMIYOR.

### NEDEN cok-gate gap'i ASIRI sisirir (kullanici: "normalde 0.1 artiyordu, 81.79 neden asiri geciyor")
4-gate zincir (81.7951) satir-satir kazanc analizi:
- gate 2711 satiri degistirdi; NET kazanc 3688 (weighted-SE)
- **EN COK kazandiran 10 satir = net'in %90'i; ilk 30 satir = net'in %156'si** (gerisi zarar)
- ZARAR goren 1464 satir vs kazanan 1247; zarar -14552 / kazanc +18240 (ikisi devasa, fark kucuk-kirilgan)
- kazandiran top-30 satir ort recency-w = **1.750** (genel 1.0) -> rw-metrigini orantisiz sisirir

**MEKANIZMA:** Cok-gate (q,a) esiklerini OOF-y'ye fit eder; kazanc birkac (10) yuksek-recency-agirlikli
satira YOGUNLASIR. Public'te o satirlar BASKA y -> ezberlenen esik uymaz -> kazanc coker, 1464-satir
zarar kalir -> GAP PATLAR (+0.3+). Combinatorial tek-genis-gate (xlmr-up) kullanir: kazanc BINLERCE
satira YAYILIR (dar degil) -> genellesir -> gap saglikli +0.1. KURAL: kazanc ne kadar AZ satira yogunsa
gap o kadar siser. Bu, "cok-gate=overfit / combo14_gatemax public-sismesi"nin SAYISAL kok-nedenidir.

### KAZANC-GENISLIGI KISITLI gate (kullanici: "daha spesifik / gate'i kisitla") -> gate-ailesinin MATEMATIKSEL KAPANISI
Fikir (gap-analizinden): ezber=kazanc-yogunlasmasi; o halde (q,a) nested secimde "kazanc genis satira
yayilsin (top-20 satir net-kazancin <esik kadarini tasisin)" kisiti koy, ezberi YAPISAL engelle.
Sonuc (e5-up/xlmr-up/txt-up, conc esigi 0.15->1.0):
- Kisitsiz (conc<1.0) bile tek-gate ~0 iyilestirme (e5-up +0.000, xlmr-up -0.003)
- Kazanc-genisligi kisiti eklenince: TUM gate'lerde TAM 0.000 (hicbir kazanc kalmiyor)

**MATEMATIKSEL KAPANIS:** Gate'in nested kazanci = %100 kazanc-yogunlasmasindan (ezber) geliyor.
Yogunlasmayi yasaklayinca kazanc SIFIRLANIR -> GENELLESEN gate kazanci YOK. combinatorial'in 82.0165'i
o GENIS tek-gate'in (xlmr-up) marjinal-yayilan payindan; cok-gate/yogun-gate sadece OOF-y ezberi.
Bu, kullanicinin kazanc-genisligi fikriyle ulasilan gate-ailesinin KESIN kapanisi: combinatorial 82.0165
(public 82.122) gate-fonksiyon uzayinin matematiksel tavani. Daha-spesifik/daha-genis hicbir gate onu gecmez.

### mmstrong (arkadas artefakti, 2026-06-14 21:15) — incelendi, blende katildi -> ELENDI
mmstrong_oof/test.npy (float32, 10k, clip[0,100], NaN yok). Standalone rw-OOF=90.4617 (mm 94.82'den IYI,
arkadas mm'i guclendirmis). AMA ortogonal degil: corr(ourteam_tf)=0.976 (redundant!), corr(mm)=0.950,
GBDT'lere ~0.95; sadece metin modellerine ortogonal (txt 0.73/e5 0.77/xlmr 0.84).
Blend katki (new_model_gate.py, mm/e5/xlmr ile AYNI kanonik paired-test):
| Katilma | blend rw-OOF | delta | karar |
|---|---|---|---|
| +EKLE | 82.2821 | +0.042 | paired 7/15, t=0.66, p=0.52, CI[-0.158,+0.251] sifir-kapsar -> ELENDI |
| mm->mmstrong IKAME | 82.2274 | -0.012 | gurultu-bandi (0.25*std~0.75 alti), CLAUDE.md marjinal-RED |
| ourteam_tf->mmstrong IKAME | 82.8982 | +0.659 | felaket |
**ELENDI:** standalone guclu ama yeni ORTOGONAL sinyal yok (ourteam_tf corr 0.976 + mm corr 0.95 o sinyali
zaten tasiyor). En iyi ikame -0.012 = gurultu. mm/e5/xlmr paired-testi GECMISTI (CI sifir-alti); mmstrong
GECMEDI (CI sifir-kapsar, p=0.52). SUB-1/SUB-2 degismez. Artefakt dokuman icin tutuldu (oof/test_mmstrong.npy).

### AGIRLIK OPTIMIZASYONU TAM TARAMA (kullanici: "agirliklari farkli sekilde optimize etsek")
12-model blend meta-yontem taramasi (nested rw-OOF; mevcut ridge_pos a=1.0 = 82.2398):
| Yontem | nested rw-OOF | not |
|---|---|---|
| ridge_pos a=0.1..30 | 82.2395-82.2398 | alpha DUYARSIZ (doygun) |
| NNLS | 82.5291 | +0.29 (intercept yok) |
| Lasso/ElasticNet pos | 82.2384/82.2385 | ~ayni |
| Huber meta | 82.4457 | +0.21 kotu |
| **BayesianRidge** | 82.1703 | -0.07 GORUNDU ama 3 NEGATIF agirlik (overfit); paired 9/15 p=0.073 CI[-0.24,+0.10] sifir-kapsar -> GECMEZ |
| SLSQP simplex (pos,sum=1) | 82.5494 | +0.31 (intercept yasak, blend intercept=-4.99 onemli) |
| **SLSQP direkt rw-MSE (pos+intercept)** | 82.2374 | ridge_pos ile OZDES (-0.0024 gurultu) |

**KANIT:** Dogrudan rw-MSE'yi optimize eden SLSQP, ridge_pos ile AYNI noktaya variyor -> ridge_pos
keyfi degil MATEMATIKSEL OPTIMAL. Tek "iyi gorunen" BayesianRidge negatif-agirlik overfit (positive
kisit YOK -> in-sample gurultuye uyum, public'te siser); kanonik paired-test cururuttu. **Agirlik
optimizasyon uzayi DOYGUN.** ridge_pos (positive, L2, intercept) optimal; meta-hunt 6-yontem RED'ini
4 yeni yontemle (BayesianRidge/SLSQP x2/Huber) genisletip teyit eder.

### Ahmet blend'ini ACMA (kullanici: "ourteam_tf 0.495 alti, icindeki alt-modellerle oynayalim")
ahmettengelenler/ icinde Ahmet'in 5 alt-modeli AYRI mevcut (oof_lgbm/xgb/cat/mlp/nn, tunafold-hizali
recon corr 0.9956). Kaliteler: lgbm 84.60 / xgb 85.65 / cat 86.24 / mlp 94.21 / nn 108.69. Hepsi
ourteam_tf'ye 0.947-0.998 korele. ourteam_tf'yi acip alt-modelleri kendi ridge_pos'umuza AYRI kolon verdik:
| Yaklasim | nested rw-OOF | delta |
|---|---|---|
| ourteam_tf BIRLESIK (mevcut) | 82.2398 | — |
| ourteam_tf -> 5 alt-model AYRI | 82.3435 | +0.104 (RED) |
| ourteam_tf + 5 alt-model HEPSI | 82.3162 | +0.076 (RED) |

**RED, NEDEN:** Ahmet'in NNLS'i 5 alt-modeli ZATEN optimal birlestirmis (ourteam_tf=84.23). Acip kendi
ridge'imizle yeniden birlestirmek: (1) cift-optimizasyon gurultusu (iki kez OOF-fit ust uste), (2) alt-modeller
asiri korele (lgbm-xgb-cat hepsi GBDT) -> ridge multikolinearite, kararsiz agirlik, (3) Ahmet'in recency-aware
NNLS'i bu alt-grup icin bizim ridge'den iyi kalibre. Iyi-paketlenmis bir ensemble'i parcalamak BOZUYOR.
ourteam_tf BIRLESIK kalir. clip-disi (max 105-113) alt-modeller clip'lendi (yine de RED).

### AHMET TARGET815 (meta 81.79 / robust 81.61 / aggressive 81.49) — INCELEME
ahmettengelenler2/: Ahmet'in 12-model+fullft+ftt meta-stack + gate-zinciri. OOF'lar BIZIM metrikle
BIREBIR tutuyor (aggressive 81.4884 = iddia; metrik dogru, ayni y/w/fold).
| Aday | rw-OOF | Yontem | Bizim teshis |
|---|---|---|---|
| meta | 81.7886 | 12-model + fullft + ftt -> Ridge(a=0.01) | **SAGLIKLI gorunuyor**: std 12.57~blend 12.66 (varyans sismemis=negatif-agirlik overfit DEGIL), degisim %54 satira yayilmis (genis-yumusak, max 3.8) -> gercek fullft/ftt katkisi OLABILIR |
| robust | 81.6124 | meta + 1 gate (fullft-up) | kismi-ezber |
| aggressive | 81.4884 | meta + 3-gate (fullft-up,lgbm-down,mmstrong-up) | **EZBER IMZASI**: top-50 satir net kazancin %105'i, 4884 satir ZARAR (combo14_gatemax profili, bizimkinden yumusak) |
**KRITIK BILINMEYEN: fullft (XLM-R full fine-tune) + ftt (FT-Transformer) BIZDE YOK + repeat-0-only**
(Ahmet'in KENDI uyarisi: repeat-1,2 hizali degil -> meta-hucrelerde olasi leakage). meta -0.45'i gercek mi
leakage mi, fullft'i 3-repeat uretmeden veya PUBLIC-GAP olcmeden kesinlesemez. mmstrong zaten test edildi
(faydasiz, corr ourteam_tf 0.976). KARAR: meta-stack umut-verici (gate-DISI, genis, varyans-stabil = BERTurk'te
aradigimiz XLM-R-fine-tune sinyali olabilir); aggressive gate-zinciri RISKLI (ezber). Submission RISK: meta
> robust > aggressive (ezber-siralamasi ters). Public-gap olcumu sart.

### BASKA TAKIM (bagimsiz) TABLOSU — CAPRAZ-DOGRULAMA (kullanici paylasti)
Bagimsiz bir takim ayri calisip BIZIMLE BIREBIR ayni sonuclara varmis:
A. POST-PROCESSING (6 teknik, hepsi onlarda da RED): variance-expansion 82.38 / quantile-match belirsiz /
   mean-shift / weighted-median / power-mean / partial-clip -> hepsi baseline-alti. BIZIM RED'lerle 6/6 ortusur.
B. BLEND: onlarin EN IYISI 'neg200 tek-basina 82.276'; blend/rank-avg HEP daha kotu. **BIZ ONLARDAN ILERIDEYIZ:**
   combinatorial public 82.122 < onlarin 82.276. Bizim ensemble (ourteam_tf + Turkce-NLP ortogonalligi)
   onlarin yapamadigi blend-kazancini cikarabilmis.
C. MODEL: 'turkish-e5-large FINE-TUNE = TEK GERCEK SINYAL KAPISI (3-4 saat A100, deadline riski)'.
   2-layer meta + XGB seed-bag -> bizde de RED (mh_2layer 82.295 / rv_seedbag).
**UC BAGIMSIZ ISARET ayni yone:** (biz BERTurk-denedik-frozen-vardi) + (Ahmet fullft XLM-R-fine-tune) +
(bu takim 'turkish-e5 fine-tune tek kapi') = KALAN TEK GERCEK KALDIRAC = Turkce transformer FULL FINE-TUNE
(frozen degil). Bu, Ahmet'in fullft meta-stack'inin (81.79, varyans-stabil, genis) neden GERCEK olabilecegini
guclendirir. ONCELIK: Ahmet'ten fullft+ftt'yi 3-repeat fold-safe iste -> leakage-safe dogrula.

### meta tabanina combinatorial-gate (kullanici: "meta ustune combinatorial gibi uygulayalim") -> EZBER, RED
Ahmet meta (81.79) tabanina tek-gate (nested, combinatorial deseni). En iyi: meta+e5-up = 81.6897 (-0.099).
GORUNDU ama DURUST test cururuttu:
- Kazanc-yayilimi: degisen 987 satir, top-10=net %196, top-50=%357 (ilk 10 satir net'in 2 kati = ASIRI yogun)
- 568 satir ZARAR (degisen yarisi)
- PAIRED: 10/15, t=-1.54, p=0.147, CI[-0.296,+0.074] SIFIR-KAPSAR -> KANONIK KAPI GECMEZ
**KONTRAST:** meta-stack -0.45 = %54 satira GENIS+varyans-stabil (gercek model katkisi); meta+gate -0.10 =
10 satira YOGUN ezber (combo14_gatemax imzasi). Gate-ekleme HER tabanda (blend de meta de) ezber. Ahmet'in
robust/aggressive'i de meta-ustu-gate = ayni ezber. meta'nin DEGERI GATE-SIZ halinde (fullft gercekse).
meta NEDEN iyilesti: kazanc orta-ust kutleden (56-95 bandi +0.76..+0.81), alt-kuyrukta KOTU (-2.08);
fullft+ftt metin-yonlu ince-ama-gercek sinyal (corr(y-blend) +0.083, %50/%50 satir, net-pozitif).

### "CORBA" testi (kullanici: "butun modellerle corba yapsak daha iner mi") -> HAYIR, kotulesir
Mevcut 12-model 82.2398 -> TUM 25 temiz-base corba 82.3181 (+0.078 KOTU). 25 modelin 15'i ridge'de
~0 agirlik (redundant). KANIT: iyilesme model-SAYISINDAN degil ORTOGONALLIKTEN gelir. Benzer model
eklemek (GBDT/metin varyanti, mevcutlarla korele) = gurultu, kotulesir. Ahmet meta -0.45'i FARKLI cunku
2 YENI FONKSIYON-SINIFI ekledi (fullft=XLM-R fine-tune, ftt=FT-Transformer) -> frozen'dan temelde farkli.
**"81.79 ciktiysa daha gider" YANILGISI:** (1) 81.79 DOGRULANMADI (fullft repeat-0-only leakage olabilir,
public 82.2'ye donebilir), (2) cok-model corba KOTULESTIRIR (kanit +0.078), (3) gercek kazanc tek
DOGRU-ORTOGONAL modelden (fullft gibi). Yol: corba YAPMA; fullft+ftt'yi dogrula (leakage-safe) + tek
gerçek-yeni ortogonal sinyal ekle. Combinatorial 82.122 hala saglam; meta 81.79 = umut AMA kanitlanmamis.

### e5ft (kullanici: "frozen e5/txt'yi TEK fine-tune modelle ikame et") — uretildi, gate -> ELENDI
colab_e5_fullft.ipynb -> e5-large DOGRUDAN-HEDEF full fine-tune (govde+head, repeat-0). Standalone rw-OOF
142.90 = frozen e5 (158.46) ve txt_ridge (168.02)'den COK IYI (fine-tune CALISTI, xlmr-ft 140 seviyesi).
AMA blende ELENDI: delta +0.015, paired 7/15, t=0.45, p=0.658, CI[-0.081,+0.114] sifir-kapsar.
NEDEN: corr(e5ft, xlmr)=0.960 -> mevcut XLM-R fine-tune ile NEREDEYSE OZDES (ikisi de Turkce-transformer-
fine-tune, ayni metni isler, ayni nis). e5ft'nin iyiligini xlmr ZATEN tasiyor. Frozen->fine-tune teknik
basari ama yeni ORTOGONAL sinyal yok (xlmr doldurmus).
**AHMET FULLFT ICIN UYARI:** fullft de XLM-R fine-tune -> muhtemelen corr(xlmr)~0.95+ -> meta 81.79'un bir
kismi xlmr-redundant olabilir. Bu, meta'nin gerceklugune ek suphe (leakage + redundans). KESIN karar:
fullft OOF/test'i al -> gate'siz paired-test (e5ft'yi test ettigim gibi). mm (birlikte fine-tune) zaten blendde.

### fullft+ftt (Ahmet, OOF+test geldi) — BIZIM nested + gate ustu
fullft (XLM-R fine-tune, standalone 146.39, corr xlmr 0.943) + ftt (FT-Transformer, 114.59, corr xlmr 0.66
ORTOGONAL ama GBDT'lere 0.91-0.94). BIZIM 3-repeat nested ridge_pos:
| Aday | nested rw-OOF | paired |
|---|---|---|
| 12-model base | 82.2398 | ref |
| +fullft | 82.1482 | 11/15, p=0.18 |
| +ftt | 82.2416 | ~0 |
| **+fullft+ftt** | **82.1245** | 11/15, t=-1.54, **p=0.145, CI[-0.358,+0.132] sifir-kapsar -> GECMEZ** (gurultu-bandi, kil payi) |
| +fullft+ftt +e5-up GATE | 82.0191 | 10/15, p=0.171, top-10=%216 net, 587 zarar -> **EZBER** |

**KESIN:** fullft+ftt GERCEK yonde (-0.115) ama paired-test'i KIL PAYI kacirir (11/15 yerine 12 lazim,
p 0.145; xlmr corr 0.943 -> metin nisi dolu). Ahmet meta 81.79 = bizim 82.12 + meta-feature ozetleri
(mean/std/min/max) + positive-OLMAYAN Ridge ~0.33 sismesi (e5ft emsali). Gate ustu = EZBER (her tabanda;
top-10 %216, public-siser). YENI DURUST NOKTA: 82.12 (gate-siz, fullft+ftt) — blend'den iyi, combinatorial
82.0165'ten kil payi kotu, paired kil payi kacar. combinatorial 82.122 public HALA en saglam dogrulanmis.

### GERCEK PUBLIC OLCUMU: fullft+ftt taban + e5-up gate = 82.0191 nested -> public 82.122074
KULLANICI SUBMIT ETTI. gap +0.103 = combinatorial (+0.106) ile BIREBIR (saglikli, SISMEDI).
**DUZELTME (kendi tahmimi):** "gate-yogunlasma (top-10 %216) -> siser" tahminim YANLIS cikti; bu gate
SAGLIKLI geldi. Yogunlasma-metrigi bu durumda asiri-temkinli (yanlis-alarm). AMA: public 82.122074 =
combinatorial 82.1221 ile BIREBIR AYNI -> yeni gate combinatorial'i GECMEDI, sadece ESITLEDI (nested de
82.0191 vs 82.0164 kil-payi geride). fullft (yeni fine-tune sinyal) + ftt + gate bile 82.122 tavanini
KIRAMADI. **82.122 = SAGLAM TAVAN (artik 2 bagimsiz yoldan public-dogrulandi: combinatorial + fullft-gate).**
