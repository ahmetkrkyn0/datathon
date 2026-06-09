# CLAUDE.md

> Bu dosya, bu depoda calisacak Claude/AI ajanlari icin TALIMATTIR. Asagidaki kurallar pazarliksizdir; bir karar verirken supheye dustugunde SIFIR OVERFIT ilkesine ve CV otoritesine geri don.

## Proje Ozeti

BTK Akademi / Google / Girisimcilik Vakfi **Datathon 2026** (Kaggle) yarismasi: her ogrenci icin surekli `career_success_score` (0-100) tahmini yapan bir **gozetimli regresyon** cozumu. Cozum, mentorun Turkce serbest-metin degerlendirmesini (`mentor_feedback_text`) sayisal/kategorik ozelliklerle birlikte kullanir ve tek hedefi **sifir overfit** ile genellenen, reproducible bir modeldir.

## Sert Kisitlar

1. **EN ONEMLI - SIFIR OVERFIT:** 1 numarali oncelik genelleme. Yerel CV (repeated stratified 5-fold x 3 OOF MSE), private leaderboard'un sapmasiz tahmincisi olmali; CV ile private arasi fark minimum tutulur. Tum karar otoritesi CV'dedir; public LB **optimizasyon hedefi DEGIL**, yalnizca saglik sensorudur. Public LB'nin pesinden KOSULMAZ.
2. **Metrik MSE (Mean Squared Error):** Dusuk = iyi. Buyuk hatalar karesel cezalanir; uc/outlier tahminlerden kacin. Tum tahminler `np.clip(pred, 0, 100)` ile sinirlanir.
3. **Turkce NLP alani ZORUNLU kullanilmali:** `mentor_feedback_text` anlamli sekilde modele enjekte edilir (ana yontem: TF-IDF -> Ridge -> nested-OOF `txt_ridge_pred` meta-feature + elle Turkce sentiment/yapi ozellikleri). Yarisma bu dogal dil alanini kullanmayi acikca bekler.

## Git Konvansiyonlari (PAZARLIKSIZ)

- **(a) AI co-author YASAK:** Bu depodaki HICBIR commit'e `Co-Authored-By: Claude` veya herhangi bir AI co-author / "Generated with" trailer'i EKLENMEZ. Commit mesajlari temiz, insan-yazimi gibi, sade tutulur.
- **(b) ATOMIK commit:** Her commit tek mantiksal degisiklik icermeli (tek feature grubu, tek bugfix, tek SPEC). Karisik/dev commit'lerden kacin.
- **(c) Calisma branch'i:** `tuna`. Tum gelistirme bu branch'te yapilir; ana branch'e dogrudan yazilmaz.

## Veri Ozeti

| Dosya | Aciklama |
|---|---|
| `train.csv` | 10.000 satir, 47 kolon (46 feature + `mentor_feedback_text` + hedef `career_success_score`) |
| `test_x.csv` | 10.000 satir, hedef yok |
| `sample_submission.csv` | `student_id`, `career_success_score` (icindeki 123.94 yalnizca format ornegidir; gercek hedef kesin [0,100]) |

- **Hedef (`career_success_score`):** surekli, [0, 100]. min=0.00, max=100.00, ortalama=76.94. Sol kuyruklu, yuksek degerlere yiginli; ==100 kutlesi ~%7.73, <=50 alt kuyruk ~%4.97.
- **Sayisal feature'lar:** cok sayida skor (`coding_score`, `problem_solving_score`, `sql_score`, `machine_learning_score`, `technical_interview_score`, `hr_interview_score`, `cgpa`, `english_exam_score`, `attendance_rate`, `github_repo_count`, `github_avg_stars`, `portfolio_score`, `communication_score`, `teamwork_score`, `leadership_score`, `presentation_score` vb.) + sayimlar (`internship_count`, `hackathon_count`, `certification_count`, `applications_sent`, `interviews_attended` vb.).
- **Kategorik:** `department`, `university_tier`, `target_role`, `hobby`, `preferred_social_media_platform`.
- **METIN (Turkce NLP):** `mentor_feedback_text` - bos deger yok, temiz UTF-8, metinde rakam yok (hazir-cevap sizintisi yok).
- **Eksik degerler:** bazi sayisal kolonlarda NA var: `internship_duration_months` (~%16.6, MNAR -> 0+bayrak), `github_avg_stars`/`open_source_contribution_count` (~%9.1, **medyan+bayrak; 0 ENJEKTE ETME**), `english_exam_score` (~%9.5), `hr_interview_score` (~%7.8), `linkedin_*` (~%6.7), `portfolio_score` (~%3.6). Imputer degeri DAIMA fold-ici fit edilir + `_missing` bayragi eklenir.
- **`student_id`:** sentetik non-predictive anahtar (train STU_000001-010000, test STU_010001-020000, ortusme 0). ASLA modele feature olarak verilmez.

## Roadmap Index

- [`Roadmap/00-masterplan/MASTERPLAN.md`](Roadmap/00-masterplan/MASTERPLAN.md) - Tum fazlari orkestre eden ana plan: kazanma tezi, mimari, 5 gunluk zaman cizelgesi, risk register, final submission politikasi.
- [`Roadmap/01-eda-data-understanding/SPEC.md`](Roadmap/01-eda-data-understanding/SPEC.md) - Veri portresi, hedef dagilimi, eksik deger ve adversarial validation (yil kolonlari kaymasi) analizi.
- [`Roadmap/02-validation-strategy/SPEC.md`](Roadmap/02-validation-strategy/SPEC.md) - **Ust otorite:** repeated stratified 5-fold x 3 protokolu, `data/folds.parquet`, OOF sozlesmesi, 0.25*std kabul kapisi.
- [`Roadmap/03-preprocessing-cleaning/SPEC.md`](Roadmap/03-preprocessing-cleaning/SPEC.md) - Sizintisiz fold-ici Pipeline iskeleti: imputation+`_missing` bayraklari, kategorik encoding.
- [`Roadmap/04-feature-engineering/SPEC.md`](Roadmap/04-feature-engineering/SPEC.md) - Odakli FE: kompozitler (tech_mean/soft_mean/...), `project_quality*tech_mean` carpimi, log1p, MNAR isleme.
- [`Roadmap/05-nlp-text-features/SPEC.md`](Roadmap/05-nlp-text-features/SPEC.md) - Turkce NLP: TF-IDF -> Ridge -> nested-OOF `txt_ridge_pred` + elle sentiment/yapi ozellikleri.
- [`Roadmap/06-modeling-ensembling/SPEC.md`](Roadmap/06-modeling-ensembling/SPEC.md) - LGBM + CatBoost + HistGBR base'leri, seed-averaging, Ridge/NNLS blend.
- [`Roadmap/07-evaluation-submission/SPEC.md`](Roadmap/07-evaluation-submission/SPEC.md) - Reproducibility testi, final 2 submission secimi, submission format kontrolu, juri sunumu.

## Calisma Normlari

- **CV-first:** Her model/HP/feature karari CV-MSE(mean, std) ile alinir. "Public LB iyilesti" gerekce DEGILDIR.
- **Public LB'ye guvenme:** Yalnizca saglik sensoru. Gap esikleri: saglikli `|gap|<=1.5*cv_std`; sari `1.5-3*std` (sizinti incele, public'e gore SECME); kirmizi `>3*std` ve public<CV -> DUR.
- **Fold-safe (sizintisiz) pipeline:** HER fit edilen donusum (imputation, target/mean encoding, TF-IDF, scaler, SVD, NLP-uzeri Ridge) SADECE o dis fold'un train parcasindan fit edilir, valid/test'e transform edilir. Global encoding YASAK; OOF target-encoding + Bayesian smoothing (m~20-50) kullanilir. Tum modeller AYNI `data/folds.parquet` dosyasini kullanir (satir-hizali OOF).
- **Yil kolonlari (temporal kayma):** `application_year` ve `graduation_year` HAM FEATURE olarak KULLANILMAZ (adversarial AUC yillarla 0.66, yilsiz ~0.49). Sadece shift-invariant turev denenir ve nihai matriste adversarial AUC ~0.5 kaldigi DOGRULANIRSA kalir.
- **Sabit seed / reproducibility:** `SEED=42` her yerde (numpy, lgbm, catboost, sklearn, `PYTHONHASHSEED`). LightGBM `deterministic=True` + sabit thread. `requirements.txt` pinli. `oof_*.npy` + `test_*.npy` + CV-MSE(mean, std) loglanir. Pipeline (`python src/<faz>.py`) internet kapali bastan sona ayni sonucu uretmeli.
- **Tahminleri [0,100] clip et:** Tum cikti `np.clip(pred, 0, 100)`. Submission yazici clip-disi deger gorurse `assert` ile hata firlatir. Log/logit donusumu YOK.
- **Kabul kapisi (overfit kapisi):** `yeni_cv_mse_mean < eski_cv_mse_mean - 0.25*cv_mse_std` olmali. Marjinal "0.1 MSE" kazanci REDDEDILIR. Esitlikte daha basit model kazanir (Occam).
- **Turkce metin normalizasyonu:** Turkce-duyarli lowercase (I/ı, İ/i tuzagina dikkat); metin ve lexicon AYNI normalizasyon. UTF-8 oku; ftfy/latin1 mojibake fix YAPMA (veriyi bozar).

## Klasor Yapisi

```
datathon26/
├── CLAUDE.md                       # Bu dosya - AI ajanlari icin talimat
├── requirements.txt                # Pinli bagimliliklar (reproducibility)
├── train.csv                       # Egitim verisi (10k satir, hedefli)
├── test_x.csv                      # Test verisi (10k satir, hedefsiz)
├── sample_submission.csv           # Ornek submission formati
├── data/
│   ├── folds.parquet               # student_id, repeat, fold (TUM modeller kullanir)
│   ├── oof_*.npy                   # Her base model icin OOF tahminleri
│   └── test_*.npy                  # Her base model icin fold-bagged test tahminleri
├── Roadmap/
│   ├── 00-masterplan/MASTERPLAN.md
│   ├── 01-eda-data-understanding/SPEC.md
│   ├── 02-validation-strategy/SPEC.md
│   ├── 03-preprocessing-cleaning/SPEC.md
│   ├── 04-feature-engineering/SPEC.md
│   ├── 05-nlp-text-features/SPEC.md
│   ├── 06-modeling-ensembling/SPEC.md
│   └── 07-evaluation-submission/SPEC.md
├── src/                      # Temiz, reproducible Python script(ler)i
└── submissions/
    ├── submissions_log.csv         # tarih, model, commit_hash, cv_mean, cv_std, public_lb, gap, secildi
    └── *.csv                       # student_id, career_success_score
```

## Submission Disiplini

- **Butce:** Gunde max 5 hak, ama her gun 5 harcanmaz; gunluk hakkin >=3'u rezerv. Submission yalnizca (a) yeni model ailesi ilk kez LB'ye cikip gap olcumu icin, (b) final adaylarini teyit icin yapilir. "Biraz daha deneyeyim" tuzagina dusme.
- **Defter:** Her submission `submissions/submissions_log.csv`'ye yazilir: tarih, model_aciklama, commit_hash, cv_mse_mean, cv_mse_std, public_lb_mse, gap=public-cv, secildi(bool).
- **Final 2 submission (ikisi de CV ile secilir, public-en-yuksek ASLA secilmez):**
  - **SUB-1 (CAPA/safe):** En dusuk cv_mean'li EN BASIT tek guclu GBDT + clip; gap saglikli. "Ne olursa olsun makul" adayi.
  - **SUB-2 (EN IYI CV):** En dusuk cv_mean'li Ridge-stack/NNLS ENSEMBLE; gap saglikli.
  - Ikisi YAPISAL farkli (sade tek-model vs ensemble) secilir -> private %40 bolmesine karsi risk dagitimi. cv farki `<0.25*std` ise birini bilerek daha basit/farkli tut.
- **Format:** `student_id` + `career_success_score`, test ID sirasi (STU_010001..020000), index'siz, UTF-8. `sample_submission` ile satir sayisi (10000) ve ID kumesi birebir esitligi `assert` edilir. Tum tahminler clip[0,100].
- **Uygunluk (elemeli):** BTK Akademi basvurusu tamam ve takim adi BTK ile birebir ayni olmali; teslim oncesi checklist.
```
