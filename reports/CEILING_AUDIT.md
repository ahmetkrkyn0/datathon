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
