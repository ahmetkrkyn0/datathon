# Faz 7 — Degerlendirme & Submission (Evaluation & Submission)

> Manda: **SIFIR OVERFIT.** Bu faz YENI MODEL URETMEZ; onceki fazlarin OOF artefaktlarini
> son kez denetler, tahminleri guvenli hale getirir (clip), submission butcesini disiplinle
> harcar, **iki final submission'i CV ile (public LB ile DEGIL) secer**, ve juriye reproducible
> tek-script + net anlati teslim eder. Bu faz, Faz 02 (Validation Strategy) protokolunun
> uygulamaya gecirildigi son kapidir; bir karar Faz 02'deki overfit kapisindan (0.25*std)
> gecmiyorsa burada da gecmez.

---

## Amac

Bitmis modelleri (Faz 06 OOF/test artefaktlari) en dusuk OOF-MSE'li ve **CV-LB gap'i en saglikli**
iki yapisal farkli adaya damitmak, tahminleri `clip[0,100]` ile guvenli kilmak, yarisma formatinda
(`student_id, career_success_score`) hatasiz submission uretmek ve juri sunumu + internet-kapali
reproducible .py script ile teslimi tamamlamak.

---

## "0 Overfit" Rolu — bu faz genellemeye nasil hizmet ediyor

Bu faz overfit'in yarismayi kaybettirdigi son ve en kritik noktadir: **final submission secimi**.
Tum modelleme dogru yapilsa bile, finalde "public LB'de en yuksek" submission secilirse public %60'a
overfit olunur ve private %40'ta cokulebilir. Bu faz overfit'e su somut savunmalarla hizmet eder:

1. **Karar otoritesi CV'dedir, public LB sadece saglik sensorudur.** Final 2 aday CV-MSE(mean) ile
   secilir; public LB yalnizca "CV'm yalan soyluyor mu?" sorusunu yanitlar (Faz 02 §3.3 altin kural).
2. **Yapisal cesitlilik = risk dagitimi.** SUB-1 (sade tek-model capa) ve SUB-2 (ensemble) birbirinden
   yapisal farkli secilir; ikisinin private'ta ayni anda cokme olasiligi dusuk.
3. **Submission butce disiplini** public'e bakip "biraz daha deneyeyim" tuzagini ortadan kaldirir
   (gunluk 5 hakkin >=3'u rezerv).
4. **Clip[0,100]** sansurlu hedefte ([0,100]) bedava/notr kazanc; uc tahminlerin karesel MSE cezasini keser.
5. **Internet-kapali reproducibility testi** ayni OOF-MSE'yi tekrar uretir; CV'nin gercek (gurultu degil)
   oldugunu kanitlar — sunumun ve guvenin temeli.

---

## Girdiler / Cikti Artefaktlari

### Onceki fazlardan alinan girdiler (somut dosya/degisken)
- `data/folds.parquet` (student_id, repeat, fold) — Faz 02. TUM OOF satir-hizalamasinin referansi.
- Her base model M icin `artifacts/oof_{M}.npy` (10000,) ve `artifacts/test_{M}.npy` (10000,) —
  Faz 06. M ∈ {`lgbm_num`, `lgbm_full`, `catboost_full`, `histgbr_full`, (ops.) `xgb_full`}.
- `artifacts/cv_scores.csv` — her base ve ensemble icin `model, cv_mse_mean, cv_mse_std,
  best_iteration_mean` — Faz 06.
- `txt_ridge_pred` OOF/test kolonu (nested inner-KFold) ve lexicon feature OOF'lari — Faz 05
  (zaten base modellerin feature setine girmis, burada dogrudan tekrar uretilmez).
- `data/test_x.csv` — `student_id` sirasi (STU_010001..020000) submission icin referans.
- `data/sample_submission.csv` — format/satir-sayisi/ID-kumesi assert referansi.
- `y_train` (career_success_score, train hedefi) — OOF-MSE hesabi icin.

### Bu fazin urettigi cikti artefaktlari
- `submissions/sub1_anchor.csv`, `submissions/sub2_ensemble.csv` — final 2 aday (clip'li).
- (gerektiginde gun-ici teyit submission'lari: `submissions/probe_{tarih}_{model}.csv`)
- `submissions_log.csv` — tarih, model_aciklama, commit_hash, cv_mse_mean, cv_mse_std,
  public_lb_mse, gap, esik_durumu(yesil/sari/kirmizi), secildi(bool).
- `artifacts/oof_ensemble.npy`, `artifacts/test_ensemble.npy` — final blend ciktilari.
- `artifacts/blend_weights.json` — Ridge/NNLS agirliklari + alpha (reproducibility).
- `reports/cv_lb_gap.png`, `reports/ablation_table.md`, `reports/final_selection.md`
  (SUB-1/SUB-2 gerekce notu) — juri sunumu girdileri.
- `requirements.txt` (pinli), `src/final_clean.py` (deterministik tam koşu, internet kapali).
- `presentation/` (6-7 slayt + Q&A kartlari), `checklist_teslim.md` (BTK uygunluk).

---

## Detayli Adimlar

### 1. OOF artefakt butunluk denetimi (sizinti ve hizalama)
1. `data/folds.parquet` yuklenir; her `oof_{M}.npy`'nin uzunlugu 10000 ve `student_id` sirasinin
   `folds.parquet` ile birebir ayni oldugu **assert** edilir (satir-hizalama, Faz 02 §2.1 sozlesmesi).
2. Her base model icin OOF-MSE yeniden hesaplanir (`mean((oof_M - y)^2)`) ve `cv_scores.csv`'deki
   `cv_mse_mean` ile **+/- 1e-6** toleransta eslesir — eslesmiyorsa artefakt bozuk, DUR.
3. Beklenen capa degerleri teyit (Faz 02 anchor): `lgbm_num` ~91.6, `*_full` (metin'li) ~83.
   `*_full` < `lgbm_num` - 0.25*std saglanmazsa metin enjeksiyonu sorgulanir (Faz 05'e geri don).
4. `oof_M` ve `test_M` icinde NaN/Inf yok; `test_M` dagilimi `oof_M` ile makul ortusur
   (mean/std fark < 1 puan; aksi halde test-time leakage veya dagilim kaymasi suphesi — Faz 02 §5).

### 2. Final blend (ensemble) — sadece OOF uzerinde
1. Base `oof_*` kolonlarini (X_meta) ve `y`'yi al; meta-model **Faz 02 §2.3** uyarinca AYNI dis
   fold semasiyla, base OOF zaten fold-disi oldugu icin **nested gerekmeden** egitilir.
2. Iki aday meta: (a) `sklearn.linear_model.Ridge` alpha CV ile [0.1, 1, 10] taranir;
   (b) **NNLS** (`scipy.optimize.nnls`) ile negatif-olmayan agirlikli ortalama. Negatif agirlik
   yorumsuz/overfit isareti -> NNLS varsayilan, Ridge yalniz NNLS'yi >1 cv_std gecerse.
3. `oof_ensemble` OOF-MSE hesaplanir; **kabul kapisi**: `ensemble_mse < min(base_mse) - 0.25*cv_std`.
   Gecmezse ensemble REDDEDILIR, en iyi tek base model SUB-2 olur (Occam).
4. GBM-stacker **varsayilan DEGIL**; sadece nested CV'de Ridge/NNLS'yi >1 cv_std gecerse (Faz 02 §2.3).
5. Agirliklar `blend_weights.json`'a yazilir; `test_ensemble = Σ w_M * test_M`, ardindan clip.

### 3. Tahmin sonrasi (post-processing) — clip zorunlu
1. **TUM** tahminler `np.clip(pred, 0, 100)` (hem OOF hem test). Faz 02 §4.4: sansurlu hedef +
   MSE outlier cezasi; clip her zaman iyilestirir veya notrdur.
2. Submission yazici, clip-disi (>100 veya <0) deger gorurse **assert ile hata firlatir** —
   sessiz format hatasi engellenir. (sample_submission'daki 123.94 yalniz format ornegidir.)
3. Log/logit donusumu YOK (skew -0.45, neredeyse normal; cift-sinirli yiginla iyi calismaz).
4. ==100 kutlesi icin opsiyonel ufak yukari-kalibrasyon SADECE OOF-MSE'yi 0.25*std net iyilestirirse;
   varsayilan saf clip. Iki-asamali (P(y=100) + regressor) clever-but-fragile -> OOF kanitina kapali.

### 4. Submission dosyasi uretimi ve format dogrulamasi
1. `pandas.DataFrame({'student_id': test_ids, 'career_success_score': clipped_pred})`,
   `test_ids` = `test_x.csv`'deki ORIJINAL sira (STU_010001..020000), `index=False`, UTF-8.
2. **Assert bloku:** (a) satir sayisi == 10000; (b) `student_id` kumesi sample_submission ile
   birebir esit; (c) sira sample_submission ile ayni; (d) kolon adlari tam `student_id,
   career_success_score`; (e) min>=0, max<=100, NaN yok.
3. SUB-1 (`sub1_anchor.csv`) ve SUB-2 (`sub2_ensemble.csv`) ayni yazici fonksiyonundan uretilir.

### 5. CV-LB gap denetimi ve submission butce yonetimi (gunde max 5)
1. Her gercek submission `submissions_log.csv`'ye yazilir (commit_hash dahil — reproducibility).
2. Public LB donunce `gap = public_lb_mse - cv_mse_mean` ve esik (Faz 02 §3.2):
   - **Yesil:** `|gap| <= 1.5*cv_std` -> CV'ye guven, devam.
   - **Sari:** `1.5*std < |gap| <= 3*std` -> sizinti/encoding gozden gecir, **public'e gore SECME**.
   - **Kirmizi:** `gap > 3*std` VE public < CV (public daha iyi gorunuyor) -> sizinti suphesi, **DUR**.
3. **Butce disiplini:** gunluk 5 hakkin >=3'u rezerv. Submission yalnizca: (a) yeni model ailesi
   ILK kez LB'ye cikarken gap olcmek, (b) final adaylari teyit. "Biraz daha deneyeyim" yasak.

### 6. Iki final submission secimi (CV ile, public ile DEGIL)
1. **SUB-1 (capa / safe):** en dusuk `cv_mse_mean`'li, EN BASIT / EN AZ feature'li, gap'i **yesil**
   tek guclu GBDT (tipik aday: `lgbm_full` veya `catboost_full`, seed-averaged) + clip.
2. **SUB-2 (en iyi CV):** en dusuk `cv_mse_mean`'li Ridge/NNLS **ensemble**, gap'i **yesil**.
3. **Yapisal farklilik zorunlu:** SUB-1 tek-model, SUB-2 ensemble. cv farki < 0.25*std ise birini
   bilerek DAHA BASIT/farkli tut (private'ta birlikte cokmeyi engeller).
4. **ASLA** public-LB-en-yuksek aday secilmez. Gerekce `reports/final_selection.md`'ye yazilir.

### 7. Reproducibility paketleme
1. Global `SEED=42` (numpy, lightgbm `deterministic=True` + sabit thread, catboost, sklearn,
   `PYTHONHASHSEED`). `folds.parquet` repo'da.
2. `requirements.txt` **pinli surumler** (lightgbm, catboost, scikit-learn, pandas, numpy, scipy,
   (ops.) transformers/torch). Internet kapali calisir; offline kaynaklar Kaggle Dataset olarak.
3. `src/final_clean.py` "deterministik tam koşu" ile bastan sona calisir; uretilen OOF-MSE
   `cv_scores.csv` ile **eslesir** (deterministik teyit). Hucreler mantik sirasinda, ara debug yok.
4. `oof_*.npy`, `test_*.npy`, `blend_weights.json`, `submissions_log.csv` repo'da arsivlenir.

### 8. Juri sunumu ve teslim
1. 6-7 slayt: (1) problem & metrik (MSE), (2) veri portresi, (3) **validation felsefesi = farkimiz**
   (repeated stratified 5x3, public'e kosmama), (4) sizinti-guvenli fold-ici pipeline + adversarial
   (yilli 0.66 / yilsiz 0.49), (5) **NLP ablation** (NUM-only 89.86 -> +metin 83), (6) ensemble &
   final-2 gerekce, (7) reproduce adimlari. + Q&A kartlari.
2. `cv_lb_gap.png` (CV vs public, esik bantlari) ve `ablation_table.md` slaytlara islenir.
3. `checklist_teslim.md`: BTK Akademi basvurusu tamam + takim adi BTK ile birebir ayni; final 2
   submission secimi 14 Haz 23:59 oncesi platformda onaylandi.

---

## Kararlar & Gerekceler

- **NNLS varsayilan, Ridge sarta bagli:** NNLS negatif-olmayan agirlik verir (yorumlanabilir, base'lerin
  hicbiri "negatif yonde" kullanilmaz), overfit yuzeyi minimum. Ridge daha esnek ama yalniz NNLS'yi
  >1 cv_std net gecerse; aksi halde fazladan serbestlik = overfit. GBM-stacker elendi: 10k OOF'ta agir
  stacking CV'ye overfit eder ve CV-private gap acar (Faz 02 §2.3).
- **SUB-1 sade tek-model:** ensemble her zaman daha iyi CV verse de, tek bir blend bilesenindeki gizli
  sizinti/varyans tum ensemble'i bozar. Yapisal farkli bir tek-model capa, %40 private bolmesine karsi
  sigorta. Iki adayi da ensemble yapmak risk yogunlastirir -> elendi.
- **Saf clip, kalibrasyon kapali:** iki-asamali model ve uc kalibrasyon "clever-but-fragile"; OOF kaniti
  olmadan eklenmez. Clip ise gercek hedef [0,100] oldugu icin matematiksel olarak notr/iyilestirici.
- **Public LB optimizasyon hedefi degil:** rastgele %60/%40 bolme + public ornekleme gurultusu nedeniyle
  public'e gore secim private'ta cokebilir. Tum secim CV-MSE(mean,std) ile (Faz 02 §3.3).
- **Internet-kapali reproducibility:** ilk 10 takim temiz/reproducible .py script ZORUNDA; "deterministik tam koşu"
  testi hem yarisma sartini hem sunum guvenilirligini karsilar.

---

## Leakage / Overfit Guardrail'lari

- **STACK/BLEND LEAK:** meta-model SADECE `oof_*` (fold-disi) kolonlari uzerinde egitilir; in-fold
  tahminle stacking asiri-iyimser OOF uretir, private'ta tutmaz (Faz 02 §2.3 / kanonik leakageRules).
- **TEST-TIME REFIT LEAK:** test tahmini fold modellerinin ortalamasiyla (fold-bagging) ya da tum-train
  refit'le uretilir; refit'te early-stopping iterasyonu OOF ortalama `best_iteration` ile **sabitlenir**,
  test/valid early stopping icin KULLANILMAZ (gizli sizinti).
- **CLIP-OOF TUTARLILIGI:** OOF-MSE de clip SONRASI hesaplanir ki rapor edilen skor submission ile ayni
  islemden gecmis olsun; clip OOF'ta uygulanip test'te unutulmaz (veya tersi).
- **SECIM-LEAK (en ince tuzak):** final aday secimi public LB'ye bakilarak yapilmaz; aksi halde public
  validasyon setine ornekleme-overfit edilmis olur. Secim defteri (`final_selection.md`) CV-only gerekce icerir.
- **FORMAT/ID LEAK:** `student_id` non-predictive sentetik anahtar, ASLA feature degil; submission'da
  sira sample_submission ile birebir korunur (yanlis sira = sessiz buyuk MSE).
- **METRIK-SIZINTI:** OOF-MSE Faz 02'deki ayni `folds.parquet` ve ayni y ile hesaplanir; baska bir
  bolme/agirlik ile "daha iyi" skor uretmek yasak (karsilastirilabilirligi bozar).

---

## Teslimler (Deliverables)

1. `submissions/sub1_anchor.csv`, `submissions/sub2_ensemble.csv` (clip'li, format-assert'li).
2. `submissions_log.csv` (gap takibi + esik durumu + secildi bayragi, commit_hash'li).
3. `artifacts/oof_ensemble.npy`, `artifacts/test_ensemble.npy`, `artifacts/blend_weights.json`.
4. `reports/cv_lb_gap.png`, `reports/ablation_table.md`, `reports/final_selection.md`.
5. `requirements.txt` (pinli), `src/final_clean.py` (deterministik tam koşu, internet kapali).
6. `presentation/` (6-7 slayt + Q&A kartlari), `checklist_teslim.md` (BTK uygunluk).

---

## Definition of Done

1. Tum `oof_*.npy`/`test_*.npy` butunluk denetiminden gecti: 10000 satir, `folds.parquet` ile
   hizali, OOF-MSE `cv_scores.csv` ile +/-1e-6 esit, NaN/Inf yok.
2. Final blend OOF-MSE, en iyi base'i `0.25*cv_std` kapisiyla net geciyor (veya gecmedigi icin SUB-2
   bilincli olarak tek base secildi); `blend_weights.json` yazildi.
3. Iki submission da format-assert'ten geciyor: 10000 satir, ID kumesi/sira sample_submission ile
   birebir, kolon adlari dogru, tum degerler clip[0,100], NaN yok.
4. `submissions_log.csv` her gercek submission'i + gap + esik durumunu iceriyor; hicbir aday public
   LB'ye gore secilmedi (gerekce `final_selection.md`'de CV-only).
5. SUB-1 ve SUB-2 yapisal farkli (tek-model vs ensemble) ve ikisinin de gap'i yesil.
6. `src/final_clean.py` internet kapali "deterministik tam koşu" ile hatasiz calisti ve OOF-MSE'yi
   yeniden uretti (deterministik teyit).
7. Sunum (6-7 slayt + Q&A) ve `checklist_teslim.md` (BTK basvuru + takim adi) tamam; final 2
   submission platformda 14 Haz 23:59 oncesi onaylandi.

---

## Riskler & Azaltim

| Risk | Etki | Azaltim |
|---|---|---|
| Public LB'ye gore aday secme cazibesi | Private'ta cokme | Secim CV-only; `final_selection.md` gerekce; public sadece yesil/sari/kirmizi sensoru |
| Ensemble bir base'in gizli sizintisini buyutur | Private MSE patlar | NNLS + kabul kapisi; SUB-1 sade tek-model capa = sigorta |
| Submission ID sirasi/kume hatasi | Sessiz buyuk MSE | Yazici assert bloku (sira, kume, kolon, [0,100]); sample_submission ile birebir |
| Script internet acikken calisip offline'da kirilir | Reproducibility ihlali, eleme riski | Offline kaynaklar Kaggle Dataset; "deterministik tam koşu" internet-kapali testi DoD'de |
| Clip OOF'ta uygulanip test'te unutulur (veya tersi) | Rapor edilen CV submission ile uyumsuz | Tek clip fonksiyonu hem OOF hem test'e; OOF-MSE clip sonrasi hesaplanir |
| Gun-ici 5 submission'in tukenmesi | Final teyit hakki kalmaz | Gunluk >=3 rezerv; probe submission'lar sadece (a) yeni aile / (b) final teyit |
| best_iteration test'te early stopping ile yeniden secilir | Gizli sizinti, optimistik CV | Refit'te OOF ortalama best_iteration SABIT; test early stopping yok |
| Son gun yeni feature/model eklemek | Test edilmemis overfit | Gun 5 "YENI RISK YOK" kurali; sadece dondurma + reproducibility + secim |

---

## Sure / Zaman Kutusu — 5 gunluk takvimdeki yeri

**Gun 5 (13-14 Haz) — Dondurma & Final Secim.** Bu faz takvimin son gunudur. **YENI RISK YOK:** yeni
feature/model/HP eklenmez. Sira: (1) OOF butunluk denetimi + final blend (sabah), (2) internet-kapali
"deterministik tam koşu" reproducibility testi, (3) SUB-1/SUB-2 secimi + format-assert, (4) submission_log +
cv_lb_gap grafigi + ablation tablosu, (5) juri sunumu (6-7 slayt) + Q&A kartlari, (6) BTK uygunluk
checklist, (7) **14 Haz 23:59 deadline oncesi platformda final 2 submission onayi**. Gun-ici probe
submission'lari (final teyit) butce disiplini icinde harcanir.

> Not: Faz 02 §3 (gap takibi) ve §6 (final secim) Gun 1'den itibaren `submissions_log.csv` ile islerken,
> bu faz o defteri Gun 5'te baglayan ve dondurmayi yapan icradir.

---

## Capraz Referanslar

- **Faz 02 — Validation Strategy:** UST OTORITE. §1 CV protokolu (`folds.parquet`, 5x3 stratified),
  §2 OOF/stacking sozlesmesi, §3 CV-LB gap esikleri + 0.25*std kabul kapisi, §4.4 clip[0,100],
  §5 adversarial, §6 iki final submission secim kurali, §7 reproducibility. Bu faz onun icrasidir.
- **Faz 06 — Modeling & Ensembling:** `oof_*.npy`/`test_*.npy`/`cv_scores.csv` ve seed-averaged base
  modeller buradan gelir; final blend bu fazda OOF uzerinde kurulur.
- **Faz 05 — NLP Text Features:** `txt_ridge_pred` ve lexicon OOF'lari (NUM-only 89.86 -> +metin 83);
  ablation tablosu sunum girdisi.
- **Faz 04 — Feature Engineering:** kompozit feature'lar ve adversarial-temiz feature matrisi; SUB-1
  "az feature'li capa" tanimi buradaki feature ablasyonuna dayanir.
- **Faz 01 — EDA:** veri portresi slaytinin (hedef dagilimi, ==100 kutlesi, eksik degerler) kaynagi.
- **Faz 00 — Masterplan:** submission butce politikasi, BTK uygunluk checklist ve deadline orkestrasyonu.
