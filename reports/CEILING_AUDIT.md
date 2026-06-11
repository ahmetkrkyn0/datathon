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
