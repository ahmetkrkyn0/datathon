# Faz 2 — Dogrulama Stratejisi (CV) — Anti-Overfit Omurgasi

> **Manda:** SIFIR OVERFIT. Yerel CV, private leaderboard'un **sapmasiz tahmincisi** olmali.
> Public LB PESINDE KOSULMAYACAK. Bu dosya tum sonraki fazlarin (03-07) **UST OTORITESIDIR:**
> bir model / feature / hiperparametre ancak buradaki protokole gore CV-MSE'yi iyilestiriyorsa kabul edilir.
> Kanonik stratejideki `cvProtocol` ve `leakageRules` ile birebir tutarlidir; o kurallari burada uygulanabilir adimlara cevirir.
> Faz 07 (diskte dolu) bu fazin **icrasidir;** bu yuzden artefakt isimleri (`artifacts/oof_{M}.npy`,
> `artifacts/test_{M}.npy`, `artifacts/cv_scores.csv`) ve **TEK kanonik test-uretim yolu** burada baglayici olarak sabitlenir.

---

## 1. Amac

Modellemeden ONCE, private leaderboard MSE'sini **sapmasiz ve dusuk-varyansli** tahmin eden, satir-hizali OOF altyapili, sizinti-gecirmez (leakage-proof) bir **Repeated Stratified 5-fold x 3-repeat** dogrulama omurgasi kurmak; boylece tum model/feature kararlari guvenilir bir CV-MSE(mean, std) sinyaline baglanir. Ayrica **OOF ve test tahmin artefaktlarinin isimlendirmesini ve test-uretim yolunu tek noktada sabitleyerek** rapor edilen CV-MSE'nin submission ile birebir ayni modeli temsil ettigini garanti etmek.

---

## 2. "0 Overfit" Rolu

Bu faz, "sifir overfit" hedefinin **fiziksel kalbidir.** Genellemeye dort somut mekanizmayla hizmet eder:

1. **Olcum cozunurlugu:** Tek 5-fold'un fold-std'si ~4.68 olculdu; bu, iki rakip model arasindaki tipik farktan buyuk. 3 repeat (efektif K=15) ile CV-mean standart hatasi `4.68/sqrt(15) ~ 1.21`'e iner. Boylece "gercek iyilesme" ile "gurultu" istatistiksel olarak ayirt edilir; gurultuyu kovalayip overfit etmeyiz.
2. **Sizinti karantinasi:** TUM fit edilen donusumler (impute, target-encoding, TF-IDF, scaler, Ridge-on-TFIDF) yalniz fold-ici train'den fit edilir. Bu, CV-MSE'nin optimistik bias'ini sifirlar -> CV ile private arasindaki fark minimize edilir.
3. **Tek ayrismaya kilit (adversarial sigorta):** Olculen tek dagilim kaymasi yil kolonlarindadir (adversarial AUC yilli 0.664, yilsiz 0.491). Yil kolonlari disardiginda train/test ayirt edilemez, yani **random stratified KFold private MSE'nin sadik temsilcisi olur.** Bu fazda bu varsayim her feature matrisi degisikliginde yeniden dogrulanir.
4. **Kabul kapisi (overfit kapisi):** `yeni_cv_mean < eski_cv_mean - 0.25*cv_std` esigi, marjinal/varyans-icindeki "iyilesmeleri" reddederek model karmasikligi birikimini ve CV'ye overfit'i kapida durdurur.

**Ek olarak (bu surumde sabitlendi):** test tahmini ve OOF tahmini **AYNI fold modellerinden** uretildigi icin (asagida §4.3 kanonik yol), rapor edilen CV-MSE submission'in tahmin ettigi seyin sapmasiz olcusudur. Iki farkli test-uretim yolu arasinda kalip "CV bir modeli, submission baska bir modeli temsil etsin" durumu — overfit'in en sinsi kaynaklarindan biri — yapisal olarak kapatilir.

---

## 3. Girdiler / Cikti Artefaktlari

### Onceki fazlardan alinanlar
- **Faz 01 (EDA):** Olculen hedef portresi — `career_success_score` min=0, max=100, mean=76.94, std~15.19, `==100` orani %7.73, `<=50` orani %4.97, skew -0.45. Kategorik kolonlarin train/test tam ortusmesi (5/5), `student_id` ortusme = 0, eksik-deger oranlarinin train/test esitligi, metinde rakam yoklugu. Adversarial bulgusu (yilli/yilsiz AUC).
- **Ham veri:** `data/train.csv` (10.000 satir, 47 kolon), `data/test_x.csv` (10.000 satir), `data/sample_submission.csv` (`student_id, career_success_score`; gercek byte teyidi: 2. satir `STU_010002,123.94` — bu deger yalnizca FORMAT ornegidir, gercek hedef [0,100]).

### Bu fazin urettigi artefaktlar (sonraki tum fazlari baglar)
- **`data/folds.parquet`** — kolonlar: `student_id`, `repeat` (0/1/2), `fold` (0..4). 30.000 satir (10.000 x 3 repeat). TUM modeller bu TEK dosyayi okur.
- **`src/cv.py`** — `get_folds()`, `make_strat_bins(y)`, `run_oof(model_factory, X, y, folds, ...)`, `compute_cv_mse(oof, y, folds)` deterministik fonksiyonlari (manuel repeated stratified — asagida).
- **`artifacts/oof_{M}.npy`** ve **`artifacts/test_{M}.npy`** — her base model `M` icin satir-hizali `(10000,)` OOF ve test tahmin vektorleri (Faz 06 doldurur). Kanonik model adlari Faz 07 ile birebir: `M ∈ {lgbm_num, lgbm_full, catboost_full, histgbr_full, (ops.) xgb_full}` ve blend icin `oof_ensemble`, `test_ensemble`.
- **`artifacts/cv_scores.csv`** — kolonlar: `model, cv_mse_mean, cv_mse_std, best_iteration_mean` (Faz 07'nin butunluk denetimi bu semayi varsayar). Her base ve ensemble icin bir satir; fold-bazli 15 MSE degeri ayrica `reports/cv_log.csv`'de saklanir.
- **`reports/cv_log.csv`** — model_adi, cv_mse_mean, cv_mse_std, 15 fold-MSE listesi, best_iteration ortalamasi (insan-okur ayrinti / denetim izi).
- **`reports/submissions_log.csv`** — gap takibi semasi (asagida §7).
- **`reports/adversarial.txt`** — yil-disi nihai feature matrisinde train/test AUC dogrulamasi.

---

## 4. Detayli Adimlar

### Adim 0 — Sabitler ve determinizm
- `SEED = 42` global. `np.random.seed(42)`, `random.seed(42)`, `os.environ["PYTHONHASHSEED"]="42"`. LightGBM `deterministic=True` + sabit `num_threads`, CatBoost `random_seed=42`, sklearn `random_state=42`.
- `requirements.txt` pinli (numpy, pandas, scikit-learn, lightgbm, catboost, scipy, pyarrow surumleri sabit).
- Veri okuma: `pd.read_csv(..., encoding="utf-8")`. BOM (`\ufeff`) header'da gorulduyse `encoding="utf-8-sig"`. Mojibake fix YAPILMAZ (veri temiz UTF-8).

### Adim 1 — Stratify binlerini uret (`make_strat_bins`)
```python
def make_strat_bins(y):
    # y: career_success_score (10000,), pandas Series
    bins = pd.Series(index=y.index, dtype="int64")
    is_100 = (y >= 100.0)                      # sansurlu ust kutle, ~%7.73
    bins[is_100] = 9                           # AYRI bin
    rest = y[~is_100]
    q = pd.qcut(rest, q=9, labels=False, duplicates="drop")  # ~9 esit-frekans bin
    bins[~is_100] = q.values
    return bins.values                         # ~10 ayrik sinif
```
- Toplam ~10 bin. `==100` kutlesi ve alt kuyruk (`<=50`) her fold'a dengeli dagilir.

### Adim 2 — Repeated Stratified fold atamasi (`get_folds`)
- `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=s)` her repeat icin `s in [42, 2026, 7]`.
- `RepeatedStratifiedKFold` yerine **manuel dongu** kullan (her repeat'in seed'ini acikca kontrol etmek ve `folds.parquet`'e yazmak icin).
```python
seeds = [42, 2026, 7]
rows = []
bins = make_strat_bins(y)
for r, s in enumerate(seeds):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    for f, (_, val_idx) in enumerate(skf.split(X, bins)):
        for i in val_idx:
            rows.append((student_id[i], r, f))
folds = pd.DataFrame(rows, columns=["student_id","repeat","fold"])
folds.to_parquet("data/folds.parquet", index=False)
```
- **Dogrulama assert'leri:** her (repeat, fold) icin `mean(y==100)` ve `mean(y<=50)` oranlari global oranlardan `+/-%1` icinde olmali; her satir her repeat'te tam 1 kez validation'a dusmeli.

### Adim 3 — OOF kosucu (`run_oof`) sozlesmesi
- Her base model `M` icin: 15 fit (5 fold x 3 repeat). Her fit fold-ici train'den fit edilen TUM transformerlari icerir (Faz 03/04/05'in `ColumnTransformer`/`Pipeline`'i).
- **`oof_M[i]`** = `i` satirinin gorulmedigi fold'dan tahmin; 3 repeat ortalamasi alinarak `(10000,)` nihai OOF.
- Fit edilen 15 fold modeli ve her birinin `best_iteration` degeri saklanir (kanonik test-uretimi ve `best_iteration_mean` icin gerekli).
- Cikti: `artifacts/oof_{M}.npy`, `artifacts/test_{M}.npy` (uretim yolu §4.3), 15 fold-MSE listesi -> `cv_mse_mean`, `cv_mse_std`, `best_iteration_mean`.
- **Clip:** her OOF ve test tahmini `np.clip(pred, 0, 100)` SONRA MSE hesaplanir (clip OOF'ta ve test'te ayni fonksiyondan; biri yapilip digeri unutulamaz).

### Adim 4 — Test tahmini uretimi (KANONIK YOL — audit fix, tek dogru yol)

> **Bu surumun cozdugu belirsizlik:** Faz 07 §Guardrail "TEST-TIME REFIT LEAK" hem fold-bagging hem tum-train refit'ten bahsediyordu. Burada **TEK kanonik yol sabitlenir** ki CV-MSE (fold-bagged OOF'a dayali) ile submission AYNI modelden gelsin.

**4.1 — Kanonik yol = FOLD-BAGGING (zorunlu, varsayilan).**
- `run_oof` icinde, her (repeat, fold) icin fit edilen model AYNI cagride **test_x uzerinde de tahmin uretir.**
- `test_M = clip( mean_{15 fold modeli}( pred_fold(test_x) ), 0, 100 )`.
- **Neden kanonik:** `oof_M` tam olarak bu 15 fold modelinden uretilir; dolayisiyla `test_M` da ayni 15 modelden geldiginde, `oof_M` uzerinden olculen CV-MSE submission'in tahmin ettigi modelin (fold-ensemble) sapmasiz olcusudur. CV ile submission **yapisal olarak ayni nesneyi** temsil eder. Ek refit/early-stopping karari gerekmez -> sizinti yuzeyi en kucuk.

**4.2 — Opsiyonel yol = TUM-TRAIN REFIT (varsayilan DEGIL; sadece OOF ile dogrulanirsa).**
- Tum train uzerinde tek model refit; early-stopping iterasyonu **OOF'tan gelen `best_iteration_mean` ile SABITLENIR** (refit'te test/valid ile early stopping YASAK — gizli sizinti).
- **Kullanim sarti:** refit modelinin tahmin dagilimi fold-bagged `test_M` ile makul ortusmeli (mean/std fark < 1 puan) VE bir holdout/OOF kontrolunde fold-bagging'i `0.25*cv_std` net gecmeli. Aksi halde KULLANILMAZ.
- **Onemli uyari:** tum-train refit kullanilirsa rapor edilen `cv_mse_mean` artik submission'in *tam* temsilcisi degildir (CV fold-bagged OOF'a, submission refit'e dayanir); bu durum `reports/submissions_log.csv`'de acikca isaretlenir. Bu yuzden varsayilan **her zaman fold-bagging'dir.**

**4.3 — Yazim sozlesmesi.** Hangi yol kullanilirsa kullanilsin cikti dosya adi `artifacts/test_{M}.npy` ve `(10000,)` boyuttadir; ureten yol `cv_scores.csv`'de degil, `submissions_log.csv` ve `blend_weights.json`/notlarda kayda gecer. **Varsayilan ve teslim edilen yol fold-bagging'dir.**

### Adim 5 — NLP meta-feature icin NESTED inner-KFold (kritik sizinti onlemi)
- `txt_ridge_pred` (TF-IDF word 1-2gram -> Ridge alpha~2) base feature DEGIL, **uretilen feature'dir.** Dis fold train'i icinde **5-fold inner OOF** ile uretilir; dis-valid ve test tahmini inner modellerin ortalamasidir. (Detay Faz 05.) Bu faz yalnizca **nested protokolun zorunlulugunu** ve `run_oof`'un buna gore tasarlanmasini sabitler. (Olculen 83.21 degeri tam bu nested-OOF ile uretildi.)

### Adim 6 — CV-MSE raporlama
```python
def compute_cv_mse(oof, y, folds):
    per_fold = []
    for r in folds.repeat.unique():
        for f in folds.fold.unique():
            m = (folds.repeat==r)&(folds.fold==f)
            per_fold.append(mean_squared_error(y[m], np.clip(oof[m],0,100)))
    return float(np.mean(per_fold)), float(np.std(per_fold))   # mean, std (15 deger)
```
- `cv_scores.csv` satiri: `model, cv_mse_mean, cv_mse_std, best_iteration_mean`. Faz 07 butunluk denetimi `oof_{M}.npy`'den yeniden hesapladigi MSE'yi bu `cv_mse_mean` ile `+/-1e-6` toleransta esler.

### Adim 7 — Anchor calistir (sadece-sayisal LGBM, `M=lgbm_num`)
- Faz 06'nin ilk capasi: sadece sayisal + temel FE + kategorik, `objective="regression_l2"`, fold-ici early stopping. Beklenen `cv_mse_mean ~ 91.6`, `std ~ 4.7`. Bu deger CV altyapisinin **dogru** kuruldugunun kanitidir (anchor referans). Cikti: `artifacts/oof_lgbm_num.npy`, `artifacts/test_lgbm_num.npy` (fold-bagging), `cv_scores.csv` satiri.

### Adim 8 — Adversarial validation (yil-disi nihai matriste)
- Train(0)/Test(1) etiketiyle LGBM siniflandirici, AYNI 5-fold. Nihai feature matrisinde (yil kolonlari/turevleri DAHIL DEGIL) **AUC ~0.5** dogrulanir. AUC > 0.6 -> suclu feature incele (oncelikle yil turevleri); 03/04 fazlarina geri bildir. Kayit: `reports/adversarial.txt`.

### Adim 9 — submissions_log semasini kur
- Asagidaki kolonlarla bos CSV olustur (§7); her submission'da tek satir eklenir.

---

## 5. Kararlar & Gerekceler

| Karar | Secim | Gerekce | Elenen alternatif |
|---|---|---|---|
| Protokol | Repeated Stratified 5-fold x 3 repeat (15 fit) | fold-std ~4.68; SE = std/sqrt(15) ~1.21 ile model ayirt cozunurlugu | Tek 5-fold (cozunurluk yetersiz, gurultu = sahte iyilesme) |
| K secimi | 5 (10 degil) | Fold basina 2000 satir; target-encoding/TF-IDF OOF gurultusu dusuk. Ayni 15-fit butcesiyle 5x3, 10-fold'tan dusuk varyansli | 10-fold tek tur (fold basina 1000 satir, varyans patlar) |
| Stratify | `==100` ayri bin + qcut(q=9) | Sansurlu ust kutle (%7.73) ve seyrek alt kuyruk (%4.97) fold'lara dengesiz duserse fold-MSE varyansi patlar | Duz KFold (kuyruk dagilimi kontrolsuz); GroupKFold (grup yok) |
| Seed cesitliligi | [42, 2026, 7] | Her repeat farkli shuffle; fold-atama varyansini ortalar | Tek seed 3 repeat (ayni bolme, sahte tekrar) |
| Master fold dosyasi | `data/folds.parquet` tek kaynak | Tum modeller satir-hizali OOF -> durust stacking | Her model kendi fold'unu uretir (hizalama bozulur, stack sizar) |
| **Test-uretim yolu** | **Fold-bagging (15 fold modelinin test ortalamasi) KANONIK** | `test_M` ile `oof_M` AYNI 15 modelden gelir -> CV-MSE submission'in sapmasiz olcusu; ek refit/early-stop karari yok = en kucuk sizinti yuzeyi (audit fix) | Tum-train refit varsayilan (CV fold-bagged OOF'a, submission refit'e dayanir -> CV submission'i tam temsil etmez); opsiyonel olarak korunur ama OOF-dogrulamasi ve log-isareti sarti |
| Artefakt isimleri | `oof_{M}.npy`, `test_{M}.npy`, `cv_scores.csv(model,cv_mse_mean,cv_mse_std,best_iteration_mean)` | Faz 07 (diskte dolu) bu somut isimleri ve semayi varsayiyor -> 02-07 uyumu (audit fix) | Faz 02'de isim/sema belirsiz birakmak (07 ile uyumsuzluk, butunluk denetimi kirilir) |
| Yil kolonlari | HAM kullanim YOK | Adversarial AUC yilli 0.664 -> random CV yalan soyler; yilsiz 0.491 -> CV sadik | Yili ham feature/target-encode (CV iyi, private coker) |
| Post-process | saf `clip[0,100]` | Sansurlu hedef + MSE outlier cezasi; clip bedava/notr kazanc | Log/logit donusum (skew -0.45, cift-sinirli kutleyle kotu) |
| Stacking | Ridge(alpha CV) / NNLS, OOF uzerinde, ayni dis fold | OOF zaten fold-disi; basit+regularize meta overfit etmez | GBM-stacker varsayilan (10k OOF'ta CV'ye overfit) |
| Kabul kapisi | `< eski - 0.25*std` | Gurultu bandinin disinda anlamli iyilesme sarti | "Public 0.1 arttı" (overfit davetiyesi) |

---

## 6. Leakage / Overfit Guardrail'lari

Kanonik `leakageRules`'un bu faza dusen, uygulanabilir karsiligi:

1. **FOLD-ICI FIT MUTLAK KURALI:** imputation (median/mean), target/mean encoding, TF-IDF vectorizer, StandardScaler, SVD, Ridge-on-TFIDF — HER fit SADECE dis-fold train'inden. `run_oof` icindeki transformerlar `Pipeline`/`ColumnTransformer` olarak fold dongusu ICINDE `.fit`'lenir; hicbir istatistik tum-train uzerinde hesaplanmaz.
2. **TARGET-ENCODING:** `department / target_role / university_tier / hobby / preferred_social_media_platform` icin **global** mean/target encoding KESINLIKLE YASAK. Yalniz OOF target-encoding + Bayesian smoothing (m~20-50) veya one-hot.
3. **TF-IDF / RIDGE-OOF:** vectorizer ve Ridge ASLA train+test birlesimine veya tum-train'e fit edilmez. `txt_ridge_pred` **nested inner-KFold** ile uretilir; aksi halde CV sahte iyimser olur.
4. **IMPUTATION:** 7 NA'li sayisal kolonun (internship_duration ~%16.6, github_avg_stars/open_source ~%9.1, english_exam ~%9.5, hr_interview ~%7.8, linkedin ~%6.7, portfolio ~%3.6) impute degeri fold-ici fit; `_missing` bayraklari eklenir.
5. **YIL KOLONU:** `application_year`/`graduation_year` ham veya yil-bazli agregasyon/target-encode YASAK; sadece shift-invariant turev (`years_since_graduation`) ve adversarial AUC ~0.5 kaldigi dogrulanirsa.
6. **ID / SIRA:** `student_id` (STU_xxxxxx) ASLA feature degil; sentetik uretim sirasi ezberi engellenir.
7. **STACK/BLEND:** meta-model SADECE out-of-fold level-1 (`oof_*`) tahminleri uzerinde egitilir; in-fold tahminle stacking optimistik OOF uretir.
8. **TEST-URETIM TUTARLILIGI (audit fix):** Test tahmini KANONIK olarak fold-bagging ile uretilir (15 fold modelinin test ortalamasi), boylece `test_M` ile `oof_M` ayni modellerden gelir. Tum-train refit **yalnizca opsiyonel**, kullanilirsa early-stopping iterasyonu OOF `best_iteration_mean` ile SABITLENIR (refit'te test/valid early stopping YASAK), dagilim-ortusme ve OOF-dogrulamasi gecmeli ve `submissions_log.csv`'de "refit" olarak isaretlenmelidir. **Iki yol karistirilmaz**; bir model icin ya hep fold-bagging ya (dogrulanmis) refit.
9. **CLIP-OOF/TEST TUTARLILIGI:** tek `clip(0,100)` fonksiyonu hem OOF hem test'e uygulanir; OOF-MSE clip SONRASI hesaplanir ki rapor edilen skor submission ile ayni islemden gecsin.
10. **OPTUNA META-LEAK:** HP aramasi <=50 trial, objective = repeated-CV mean, secimi cv_std ile cezala; nihai HP ayri seed/repeat ile dogrula (CV'nin kendisine overfit'i onle).
11. **TURKCE LOWERCASE:** metin ve lexicon AYNI Turkce-duyarli normalizasyon (I/ı, İ/i); karisik normalizasyon sessizce eslesme kacirir (Faz 05'e baglar).

---

## 7. Teslimler (Deliverables)

1. `data/folds.parquet` (student_id, repeat, fold) — 30.000 satir, stratify dogrulanmis.
2. `src/cv.py` — `make_strat_bins`, `get_folds`, `run_oof` (fold-bagging test-uretimi dahili), `compute_cv_mse` (deterministik, SEED=42).
3. `artifacts/cv_scores.csv` semasi (`model, cv_mse_mean, cv_mse_std, best_iteration_mean`) + anchor `lgbm_num` satiri (~91.6 / ~4.7).
4. `reports/cv_log.csv` (15 fold-MSE detayi) ve `reports/submissions_log.csv` semasi (bos, kolonlar tanimli).
5. `reports/adversarial.txt` — yil-disi nihai matris AUC ~0.5 kaydi.
6. Anchor icin `artifacts/oof_lgbm_num.npy`, `artifacts/test_lgbm_num.npy` (fold-bagging).
7. `requirements.txt` pinli surumler.

**submissions_log.csv kolonlari (Faz 07 ile birebir):**
```
tarih, model_aciklama, notebook_commit_hash, cv_mse_mean, cv_mse_std,
public_lb_mse, gap (=public_lb_mse - cv_mse_mean), esik_durumu(yesil/sari/kirmizi),
test_uretim_yolu(fold_bagging|refit), secildi(bool)
```

**Artefakt isim sozlesmesi (02-07 boyunca degismez):**
```
artifacts/oof_{M}.npy , artifacts/test_{M}.npy      # M ∈ {lgbm_num, lgbm_full, catboost_full, histgbr_full, (ops.) xgb_full}
artifacts/oof_ensemble.npy , artifacts/test_ensemble.npy
artifacts/cv_scores.csv  # kolonlar: model, cv_mse_mean, cv_mse_std, best_iteration_mean
artifacts/blend_weights.json  # Faz 06/07 doldurur
```

---

## 8. Definition of Done

1. `data/folds.parquet` uretildi; **assert geciyor:** her (repeat, fold)'da `mean(y==100)` ve `mean(y<=50)` global orandan `+/-%1` icinde; her satir her repeat'te tam 1 kez validation.
2. OOF altyapisi calisiyor: sadece-sayisal LGBM anchor (`lgbm_num`) icin `cv_mse_mean` ~**91.6**, `cv_mse_std` ~4.7 raporlu ve `cv_scores.csv`'ye yazildi. Bu, altyapinin dogru kuruldugunun olculebilir kanitidir.
3. **Test-uretim yolu fold-bagging olarak calisiyor:** `test_lgbm_num` 15 fold modelinin clip'li test ortalamasidir; `oof_lgbm_num` ayni 15 modelden gelir. (Tum-train refit kullanildiysa OOF-dogrulamasi + dagilim-ortusme gecti ve `submissions_log.csv`'de "refit" isaretli.)
4. `oof_{M}.npy`'den yeniden hesaplanan MSE, `cv_scores.csv`'deki `cv_mse_mean` ile `+/-1e-6` esit (Faz 07 butunluk denetiminin gecmesi garanti).
5. Tum tahminler `clip[0,100]` sonrasi MSE hesaplaniyor; clip-disi deger gorulurse `assert` hata firlatiyor; clip hem OOF hem test'e tek fonksiyondan uygulaniyor.
6. Adversarial AUC olculdu: yil-disi nihai feature matrisinde **~0.5** (>=0.6 ise yazili aksiyon notu, `reports/adversarial.txt`).
7. `submissions_log.csv` semasi hazir (test_uretim_yolu kolonu dahil); gap esikleri (saglikli `|gap|<=1.5*std`, sari 1.5-3*std, kirmizi >3*std & public<CV) MASTERPLAN'a baglandi.
8. Kabul kapisi (`< eski - 0.25*std`) tum sonraki fazlara duyuruldu; Occam tie-break (esitlikte basit model) kaydedildi.
9. `run_oof` reproducibility testi: ayni cagri iki kez calistirilinca **birebir ayni** `cv_mse_mean` ve `test_{M}.npy` uretiyor.

---

## 9. Riskler & Azaltim

| Risk | Etki | Azaltim |
|---|---|---|
| Fold-disi fit (sizinti) kazara yapilir | CV optimistik, private coker | TUM transform `run_oof` dongusu ICINDE `Pipeline.fit`; kod review + adversarial sanity |
| **CV ile submission farkli modelden gelir** (fold-bagging vs refit karistirilir) | CV submission'i temsil etmez -> CV-LB gap acilir, private'ta surpriz | **Kanonik yol fold-bagging** (§4.1); refit yalniz OOF-dogrulamali + log-isaretli; iki yol bir model icin karistirilmaz (Guardrail 8) |
| 3 repeat yeterli cozunurluk vermez | Gurultu kovalama | 0.25*std kabul kapisi gurultu bandini zaten kesiyor; gerekirse 5. repeat (zaman izin verirse) |
| `qcut` duplicate sinirlar | Bin sayisi <10, dengesizlik | `duplicates="drop"`; assert ile bin-orani kontrolu |
| Nested NLP OOF unutulur | NLP CV sahte iyimser (83 -> private'ta yukari) | `txt_ridge_pred` yalniz nested kosucu uzerinden; tek-fit yasak (Faz 05 guard) |
| Yil turevi ayrismayi yeniden acar | Random CV gecersiz | Her feature matris degisiminde adversarial AUC tekrar olc; >0.6 -> feature reddet |
| best_iteration refit'te yanlis sabitlenir | Final model under/overfit | (Sadece opsiyonel refit yolunda) OOF `best_iteration_mean` ile sabit; refit'te early stopping YASAK |
| Artefakt isim/sema 07 ile uyumsuz | Faz 07 butunluk denetimi kirilir, teslim gecikir | Isim sozlesmesi §7'de sabit (`oof_{M}.npy`, `test_{M}.npy`, `cv_scores.csv` semasi); 07 ile birebir |
| Public LB'ye karar kaymasi (insan zafiyeti) | Private'ta cokme | Altin kural: public = saglik sensoru; karar sadece CV; gunluk hakkin >=3'u rezerv |

---

## 10. Sure / Zaman Kutusu

- **Gun 1 (9 Haz):** Bu fazin ANA gunu. `folds.parquet` uret + stratify assert; `src/cv.py` (`run_oof` fold-bagging test-uretimiyle + `compute_cv_mse`); anchor `lgbm_num` (~91.6) -> `oof_lgbm_num.npy`/`test_lgbm_num.npy`/`cv_scores.csv`; `cv_log.csv` + `submissions_log.csv` semalari; adversarial ilk teyit; 1 submission (anchor, fold-bagging) -> ilk CV-LB gap olcumu.
- **Gun 2-4:** Bu fazin protokolu Faz 04/05/06 tarafindan tuketilir (her yeni feature/model AYNI `folds.parquet` + ayni artefakt isimleri + kanonik fold-bagging test-uretimi + kabul kapisindan gecer). Yeni feature matrisi her degisiminde adversarial AUC ~0.5 yeniden dogrulanir.
- **Gun 5 (13-14 Haz):** Reproducibility testi (Save & Run All, internet kapali) ayni OOF-MSE ve ayni `test_{M}.npy`'yi uretmeli; final 2 submission CV ile secilir (yapisal farkli: capa tek-model + en iyi CV ensemble). Faz 07 bu artefaktlarin icrasidir.

---

## 11. Capraz Referanslar

- **Faz 00 (MASTERPLAN):** Bu faz, kabul kapisi (0.25*std), gap esikleri ve final-2-submission kuralini MASTERPLAN'a baglar; tum fazlarin ust otoritesidir.
- **Faz 01 (EDA):** Hedef portresi, kategorik ortusme, missingness esitligi ve adversarial bulgusu bu fazin girdisidir; stratify binleri ve yil-disi karari oradan beslenir.
- **Faz 03 (Preprocessing):** Imputation + `_missing` bayraklari fold-ici fit (Guardrail 1, 4); ColumnTransformer `run_oof` dongusune girer.
- **Faz 04 (Feature Engineering):** Kompozitler/carpimlar ve OOF target-encoding bu fazin fold-ici fit ve 0.25*std kabul kapisina tabidir; yil turevleri adversarial AUC ile dogrulanir.
- **Faz 05 (NLP):** `txt_ridge_pred` nested inner-KFold OOF protokolu bu fazin `run_oof` sozlesmesine gore uretilir (Guardrail 3).
- **Faz 06 (Modeling/Ensembling):** Tum base modeller (`lgbm_num/lgbm_full/catboost_full/histgbr_full`) bu `folds.parquet` + OOF sozlesmesini ve **kanonik fold-bagging test-uretimini** kullanir; `oof_{M}.npy`/`test_{M}.npy`/`cv_scores.csv` buradan uretilir; seed-averaging ve Ridge/NNLS stack ayni dis fold semasiyla (Guardrail 7).
- **Faz 07 (Evaluation/Submission):** Bu fazin **icrasidir.** `oof_{M}.npy`/`test_{M}.npy`/`cv_scores.csv` butunluk denetimi, `submissions_log.csv`, gap esikleri, clip-assert ve final-2-submission secimi bu fazda tanimlanan isim/sema ve kanonik test-uretim yolu ile yurutulur. Faz 07'nin "TEST-TIME REFIT" guardrail'i bu faza gore okunur: **varsayilan fold-bagging**, refit yalniz OOF-dogrulamali opsiyon.
```

**Yazilan dosya yolu:** `c:\Users\tuna9\OneDrive\Masaüstü\datathon26\Roadmap\02-validation-strategy\SPEC.md`

**Audit bulgusunun cozumu (medium severity — test-uretim yolu belirsizligi):** Yeni SPEC'te `## 4. Detayli Adimlar` icindeki **Adim 4** TEK kanonik test-uretim yolunu sabitler: **fold-bagging (15 fold modelinin clip'li test ortalamasi) varsayilan/zorunlu** cunku `test_M` ile `oof_M` ayni 15 modelden gelir ve CV-MSE submission'i sapmasiz temsil eder. Tum-train refit ise **opsiyonel** olarak isaretlendi (sadece OOF-dogrulamasi + dagilim-ortusme gecerse, `best_iteration_mean` sabit, `submissions_log.csv`'de "refit" isaretli). Bu, Faz 07 §Guardrail "TEST-TIME REFIT LEAK"teki ikiligi tek otoriteye baglar. Ayrica artefakt isimleri (`artifacts/oof_{M}.npy`, `artifacts/test_{M}.npy`) ve `cv_scores.csv` semasi (`model, cv_mse_mean, cv_mse_std, best_iteration_mean`) ile model adlari (`lgbm_num/lgbm_full/catboost_full/histgbr_full/xgb_full`, `ensemble`) Faz 07 ile birebir yazildi; `submissions_log.csv`'ye `test_uretim_yolu` kolonu eklendi; Guardrail 8 iki yolun karistirilmamasini ve DoD madde 3-4 fold-bagging tutarliligini + 07 butunluk denetiminin gecmesini garanti eder.
