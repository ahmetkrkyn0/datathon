# Datathon 2026 — Career Success Score Tahmini

> BTK Akademi / Google / Girişimcilik Vakfı "Datathon 2026" yarışması için, her öğrencinin `career_success_score` değerini (0-100 sürekli) tahmin eden, **sıfır overfit** felsefesiyle inşa edilmiş bir regresyon çözümü.

---

## Yarışma & Görev

- **Yarışma:** BTK Akademi / Google / Girişimcilik Vakfı — Datathon 2026 (Kaggle)
- **Görev tipi:** Gözetimli **regresyon**. Her öğrenci için `career_success_score` (sürekli, **0-100**) tahmini.
- **Metrik:** **MSE (Mean Squared Error)** — düşük = iyi. Büyük hatalar karesel cezalandığından uç (outlier) tahminlerden kaçınılır ve tüm tahminler `[0, 100]` aralığına clip edilir.
- **Süre:** Deadline 14 Haziran 23:59. Günde en fazla 5 submission; finalde 2 submission seçilir. Skor %60 public / %40 private (rastgele bölme).

---

## Veri

| Dosya | Satır | Açıklama |
|---|---|---|
| `train.csv` | 10.000 | 46 feature + `mentor_feedback_text` + hedef `career_success_score` (47 kolon) |
| `test_x.csv` | 10.000 | Hedef yok — tahmin edilecek |
| `sample_submission.csv` | — | `student_id`, `career_success_score` (format örneği) |

**Hedef (`career_success_score`):** min=0.00, max=100.00, ortalama=76.94 — sol kuyruklu, yüksek değerlere yığılmış. `==100` kütlesi yaklaşık %7.73.

**Feature tipleri:**
- **Sayısal:** çok sayıda skor (`coding_score`, `problem_solving_score`, `sql_score`, `machine_learning_score`, `technical_interview_score`, `hr_interview_score`, `cgpa`, `english_exam_score`, `attendance_rate`, `github_repo_count`, `github_avg_stars`, `portfolio_score`, `communication_score`, `teamwork_score`, `leadership_score`, `presentation_score` vb.) + sayımlar (`internship_count`, `hackathon_count`, `certification_count`, `applications_sent`, `interviews_attended` vb.). Bazı kolonlarda eksik değer var.
- **Kategorik:** `department`, `university_tier`, `target_role`, `hobby`, `preferred_social_media_platform`.
- **Metin (Türkçe NLP):** `mentor_feedback_text` — mentorun öğrenci hakkındaki kısa Türkçe değerlendirmesi. Boş değer yok. Yarışma bu doğal dil alanını anlamlı kullanmayı açıkça bekliyor.

---

## Yaklaşım / Yol Haritası

Tek bir kuzey yıldızı var: **SIFIR OVERFIT.** Yerel CV (repeated stratified 5-fold OOF MSE), private leaderboard'un sapmasız tahmincisi olacak şekilde kurulur; **tüm karar otoritesi CV'dedir, public LB yalnızca bir sağlık sensörüdür.** Adversarial validation ile doğrulanan tek dağılım kayması yıl kolonlarındadır (`application_year`, `graduation_year`); bu yüzden yıllar ham feature olarak **kullanılmaz** ve random stratified KFold private MSE'yi sadık temsil eder.

Yol haritası 7 faza bölünmüştür:

1. **EDA & Veri Anlama** — veri portresi, dağılımlar, eksik değer haritası, adversarial validation teyidi.
2. **Validation Stratejisi** *(üst otorite)* — Repeated Stratified 5-fold × 3 repeat (15 fit), `data/folds.parquet`'e bir kez yazılır; tüm modeller aynı foldları kullanır. Kabul kapısı: yeni CV, eskiyi `0.25 × cv_std` ile geçmeli.
3. **Preprocessing & Temizlik** — sızıntı-güvenli, fold-içi fit edilen pipeline; MNAR eksik değer bayrakları + fold-içi imputation.
4. **Feature Engineering** — odaklı kompozitler (`tech_mean`, `soft_mean`, `interview_mean`, `profile_mean`), kanıtlanmış çarpımlar (`project_quality * tech_mean`), log1p dönüşümleri. Kitchen-sink değil.
5. **Türkçe NLP Metin Özellikleri** — word TF-IDF (1-2gram) → Ridge → nested inner-KFold OOF → tek sürekli `txt_ridge_pred` kolonu + elle tasarlanmış Türkçe sentiment/yapı özellikleri. Ölçülen kazanç: NUM-only ~89.86 → ~83 CV MSE.
6. **Modelleme & Ensembling** — LightGBM (L2) + CatBoost (native kategorik) + HistGradientBoosting, her biri seed-averaging; OOF üzerinde Ridge/NNLS blend.
7. **Değerlendirme & Submission** — reproducibility testi, CV-LB gap takibi, yapısal olarak farklı **2 final submission** (sade çapa tek-model + en iyi CV ensemble).

**Post-processing:** Ham hedef + L2 loss ile eğit, tüm tahminleri `np.clip(0, 100)`. Log/logit dönüşümü yok.

---

## Proje Yapısı

```
datathon26/
├── data/
│   ├── train.csv                # 10.000 satır, 47 kolon
│   ├── test_x.csv               # 10.000 satır, hedef yok
│   ├── sample_submission.csv    # submission format örneği
│   └── folds.parquet            # (üretilecek) master fold indeksleri — tüm modeller paylaşır
│
├── DatathonNotes/               # çalışma notları, ölçüm kayıtları, ablation tabloları
│
├── Roadmap/                     # faz bazlı plan ve spesifikasyonlar
│   ├── 00-masterplan/
│   │   └── MASTERPLAN.md         # hepsini orkestre eden ana plan
│   ├── 01-eda-data-understanding/SPEC.md
│   ├── 02-validation-strategy/SPEC.md
│   ├── 03-preprocessing-cleaning/SPEC.md
│   ├── 04-feature-engineering/SPEC.md
│   ├── 05-nlp-text-features/SPEC.md
│   ├── 06-modeling-ensembling/SPEC.md
│   └── 07-evaluation-submission/SPEC.md
│
└── README.md                    # bu dosya
```

---

## Nasıl İlerlenir / Durum

5 günlük plan, her günü somut bir teslimata bağlar:

| Gün | Tarih | Odak | Durum |
|---|---|---|---|
| 1 | 9 Haz | Temel & Anchor — folds.parquet, sızıntı-güvenli pipeline, sadece-sayısal LGBM anchor (~91.6 CV MSE) | Başladı |
| 2 | 10 Haz | Feature Engineering — kompozitler, çarpımlar, ablation kapısı | Planlandı |
| 3 | 11 Haz | Türkçe NLP — TF-IDF→Ridge OOF + lexicon, hedef ~83 CV MSE | Planlandı |
| 4 | 12 Haz | Modelleme & Ensemble — CatBoost + HistGBR, seed-averaging, blend | Planlandı |
| 5 | 13-14 Haz | Dondurma & Final — reproducibility testi, 2 final submission seçimi, jüri sunumu | Planlandı |

**İlerleme prensibi:** Her değişiklik CV kabul kapısından (`yeni_mse < eski_mse − 0.25 × cv_std`) geçmek zorundadır. Marjinal kazanımlar reddedilir; eşitlikte daha basit model kazanır (Occam). Public LB'ye bakıp "biraz daha deneyeyim" tuzağına düşülmez.

---

## Önemli Linkler

- **Ana Plan:** [`Roadmap/00-masterplan/MASTERPLAN.md`](Roadmap/00-masterplan/MASTERPLAN.md) — tüm fazları orkestre eden master plan, kilitli kararlar, risk register, jüri sunum hazırlığı.
- **Faz SPEC'leri:**
  - [`01 — EDA & Veri Anlama`](Roadmap/01-eda-data-understanding/SPEC.md)
  - [`02 — Validation Stratejisi`](Roadmap/02-validation-strategy/SPEC.md) *(üst otorite)*
  - [`03 — Preprocessing & Temizlik`](Roadmap/03-preprocessing-cleaning/SPEC.md)
  - [`04 — Feature Engineering`](Roadmap/04-feature-engineering/SPEC.md)
  - [`05 — Türkçe NLP Metin Özellikleri`](Roadmap/05-nlp-text-features/SPEC.md)
  - [`06 — Modelleme & Ensembling`](Roadmap/06-modeling-ensembling/SPEC.md)
  - [`07 — Değerlendirme & Submission`](Roadmap/07-evaluation-submission/SPEC.md)
- **Çalışma Notları:** [`DatathonNotes/`](DatathonNotes/) — ölçüm kayıtları, ablation tabloları, CV-LB gap defteri.
