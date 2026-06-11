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

## NİHAİ KARAR (2. tur sonrası)

**Blend 84.0991 artık HEM bilgi-seti HEM eğitim-mekanizması uzayında doğrulanmış tavandır.**
Ampirik dış doğrulama da geldi: public LB 84.1709, gap +0.072 (=0.025σ) YEŞİL — rw-OOF metodolojisi
gerçek dünyada 0.07 hassasiyetle kalibre. SUB-1 (catboost_full 86.41) + SUB-2 (blend 84.0991) FİNAL.
