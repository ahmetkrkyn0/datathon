# Faz 1 — EDA & Veri Anlama

## Amac
`train.csv` / `test_x.csv` / `sample_submission.csv` dosyalarini derinlemesine kesfederek hedef dagilimini, eksik deger semantigini, kategorik kardinaliteyi, sayisal sinyali, train↔test dagilim kaymasini (ozellikle yil kolonlari) ve `mentor_feedback_text` Turkce metin sinyalini KANITA dayali olarak haritalamak; boylece sonraki tum fazlarin (validation, preprocessing, FE, NLP, modelleme) karar gerekceleri bu fazda olculmus gercek sayilara dayansin.

## "0 Overfit" Rolu
Bu faz, kanonik stratejinin `northStar`'inin temel tasini doser: **CV'nin private leaderboard'un sapmasiz tahmincisi olabilmesi icin, train ile test arasindaki TEK dagilim kaymasinin yil kolonlarinda oldugunu olcerek dogrulamak.** EDA burada estetik bir kesif degil, bir *sigorta isleme* faaliyetidir:
- Adversarial validation ile "yillar cikarilinca train/test ayirt edilemez" (AUC ~0.50) hipotezini test ederek, **random stratified KFold'un private MSE'yi sadik temsil edebilecegini** kanitlar. Bu kanit yoksa, Faz 2'nin tum validation mimarisi temelsizdir.
- Hedefin `==100` kutlesini (%7.73) ve alt kuyrugunu (`<=50`, %4.97) olcerek, Faz 2'deki stratify bininin neden zorunlu oldugunu sayisal olarak gerekcelendirir (stratify olmazsa fold-MSE varyansi patlar).
- Eksik degerlerin MNAR (Missing Not At Random) yapisini olcerek (internship NA'lerinin %82'si `internship_count==0`), Faz 3'un "NA→0 + bayrak" kararini overfit'siz semantik temele oturtur.
- **Bu fazda hicbir model fit edilmez, hicbir karar leaderboard'a bakilarak alinmaz, hicbir istatistik hedefe gore secilmez** (lexicon dahil). EDA cikti uretir, karar vermez; kararlar olculmus delta'larla sonraki fazlarda 0.25·std kapisindan gecer.

## Girdiler / Cikti Artefaktlari

### Girdiler (ham veri — teyit edildi)
- `data/train.csv` → 10.000 satir × 47 kolon (`student_id` + 45 feature + `mentor_feedback_text` + hedef `career_success_score`).
- `data/test_x.csv` → 10.000 satir × 46 kolon (hedef yok; baska eksik/fazla kolon yok).
- `data/sample_submission.csv` → 2 satir ornek; kolonlar `student_id`, `career_success_score` (ikinci satirdaki `123.94` yalnizca FORMAT ornegidir, gercek hedef [0,100]).

### Ciktilar (sonraki fazlara devredilen somut artefaktlar)
- `reports/eda/eda_report.html` (veya script `src/01_eda.py`) — tum grafik ve tablolarin tek reproducible kaynagi (SEED=42, internet kapali calisir).
- `reports/eda/column_profile.csv` — her kolon icin: `dtype`, `role` (numeric/categorical/text/id/target/year), `n_missing_train`, `pct_missing_train`, `n_missing_test`, `pct_missing_test`, `nunique`, `corr_with_target` (sayisal icin), `is_year_suspect` (bool). Faz 3 ve Faz 4 bu dosyayi feature listesi ve impute plani icin okur.
- `reports/eda/missing_map.csv` — 7 NA'li sayisal kolon + MNAR analizi (NA iken `internship_count==0` orani vb.). Faz 3'un impute+bayrak plani buradan turetilir.
- `reports/eda/adversarial_auc.json` — `{auc_with_years, auc_without_years, top_features}`. Faz 2'nin adversarial sigortasinin baz cizgisi; Faz 4'te nihai feature matrisinde AUC~0.5 dogrulamasi bununla kiyaslanir.
- `reports/eda/target_profile.json` — `{mean, std, min, max, skew, pct_eq_100, pct_le_50, var, mean_by_grad_year}`. Faz 2 stratify bin tasarimi ve Faz 7 clip/post-processing kararlari buna dayanir.
- `reports/eda/text_profile.json` — metin uzunluk istatistikleri, benzersizlik orani, rakam yoklugu, anahtar kelime frekanslari. Faz 5 NLP plani buradan beslenir.
- **NOT:** Bu faz `data/folds.parquet` URETMEZ — o Faz 2'nin sorumlulugudur. EDA yalnizca onu BESLEYEN istatistikleri (stratify bin oranlari) saglar.

## Detayli Adimlar
Tum adimlar `pandas`, `numpy`, `matplotlib`/`seaborn`, `scikit-learn` ile; SEED=42, `pd.read_csv(..., encoding='utf-8')`.

1. **Yukleme & semantik kontrol.** Uc dosyayi UTF-8 ile oku. Shape assert: train `(10000,47)`, test `(10000,46)`. Train∖test kolon farkinin yalniz `career_success_score` oldugunu, test∖train farkinin bos oldugunu dogrula. `student_id` formati `STU_NNNNNN`; train `STU_000001..010000`, test `STU_010001..020000`, **kesisim = 0** (teyit edildi) → ID non-predictive sentetik anahtar, asla feature degil (`leakageRules` ID kurali).

2. **Kolon rol tablosu.** Her kolonu siniflandir: `id` (`student_id`), `target` (`career_success_score`), `text` (`mentor_feedback_text`), `categorical` (`department`, `university_tier`, `target_role`, `hobby`, `preferred_social_media_platform`), `year` (`application_year`, `graduation_year`), kalan hepsi `numeric`. `df.dtypes` + manuel override ile `column_profile.csv` uret.

3. **Hedef dagilimi analizi.** `career_success_score` icin: histogram (50 bin) + KDE, ECDF, boxplot. Olculen referans degerler (bu veri): mean **76.94**, std **15.19**, median **77.81**, min **0.00**, max **100.00**, skew **-0.451** (hafif sol kuyruk, neredeyse normal), `==100` orani **%7.73** (773 satir), `<=50` orani **%4.97** (497 satir), `==0` yalniz 1 satir, var(y) **230.63**. `==100` spike'ini ve sol kuyrugu ayri vurgula → Faz 2 stratify ve Faz 7 clip gerekcesi.

4. **Eksik deger haritasi + MNAR.** `df.isna().sum()` train & test ayri. Olculen 7 NA'li sayisal kolon (train pct): `internship_duration_months` **16.57%**, `english_exam_score` **9.53%**, `github_avg_stars` **9.10%**, `open_source_contribution_count` **9.10%**, `hr_interview_score` **7.80%**, `linkedin_profile_score` **6.68%**, `portfolio_score` **3.64%**. Test missingness neredeyse ozdes (or. internship 15.86%) → impute icin dagilim kaymasi yok. `mentor_feedback_text`'te NA YOK (train & test). **MNAR testi:** `internship_duration_months` NA iken `internship_count==0` orani **%82.14** (genel %30.7'ye karsi) → bu kolonda eksiklik "staj yok" semantigi tasir, ortalama-impute sahte "orta staj" uydurur. Diger 6 kolon icin de NA↔ilgili-count/score iliskisini logla (`missing_map.csv`).

5. **Kategorik kardinalite & yeni-seviye taramasi.** Her kategorik icin `nunique` ve seviye listesi. Olculen (train==test, test-only seviye YOK): `department` 7, `university_tier` 4 (Tier 1-4, ordinal!), `target_role` 11, `hobby` 8, `preferred_social_media_platform` 6. Kardinalite dusuk (4-11) → Faz 4'te one-hot vs OOF target-encoding ablation'i makul; test'te gorulmemis seviye riski YOK.

6. **Sayisal korelasyon analizi.** `df[num].corr()` ısı haritasi + hedefe `|corr|` sirali bar. Olculen top sinyal: `project_quality_score` **0.541** (tek basina baskin), `technical_interview_score` 0.340, `problem_solving_score` 0.290, `cloud_score` 0.277, `coding_score` 0.274, `devops_score` 0.272, `portfolio_score` 0.271. Hicbir feature 0.55 ustu degil → sinyal etkilesimlerde, GBDT lineerden ustun (`lockedDecisions` model gerekcesi). Multikolineerlik kumeleri (9 teknik skor) Faz 4 kompozit tasarimina (`tech_mean`) isaret eder.

7. **Sezgisel feature on-dogrulamasi (sadece olcum, uretim degil).** Faz 4 kararlarini onceden test et: `tech_mean` (9 teknik skor ort.) ↔ hedef **0.338** (her bireysel skordan guclu); `project_quality_score × tech_mean` ↔ hedef **0.606** (tum ham/kompozitlerin ustunde — Faz 4'un capa carpimi); `conv_rate = interviews_attended/applications_sent` ↔ hedef **0.011** (sinyalsiz gurultu → Faz 4'te URETILMEYECEK). Bu olcumler `eda_report`'a tablo olarak girer; feature'lar burada uretilmez.

8. **YIL KOLONLARI — dagilim kaymasi (en kritik EDA bulgusu).** `application_year` ve `graduation_year` icin train vs test value_counts karsilastirmasi. Olculen: train her iki kolonda da ~uniform (2019-2026 her yil ~1300), test agir 2024-2026'ya yiginli (or. `application_year` test: 2019→403, 2025→2197). **Kritik ek bulgu:** hedef ortalamasi graduation_year boyunca neredeyse sabit (2018: 77.6 → 2026: 74.0, yalniz 3.6 puan drift) → yillari atmak neredeyse hic hedef sinyali kaybettirmez, ama tek dagilim kaymasini yok eder. Bu, "yili ham feature kullanma" kararinin bedava oldugunu kanitlar.

9. **ADVERSARIAL VALIDATION (sigorta dogrulamasi).** train(0)/test(1) etiketiyle `HistGradientBoostingClassifier(max_iter=200, max_depth=4, lr=0.05, random_state=42)`, 5-fold `cross_val_predict(method='predict_proba')`, ROC-AUC. Olculen: **yillarla AUC=0.6654, yillarsiz AUC=0.4995.** Sonuc: yillar haric tum feature uzayinda train↔test ayrilamaz → random stratified KFold private'i sadik temsil eder (Faz 2 temeli). `adversarial_auc.json`'a yaz + feature importance ile suclu kolonlari (yillar) listele.

10. **Metin (`mentor_feedback_text`) on inceleme.** UTF-8 byte teyidi: `'ö'.encode('utf-8') == b'\xc3\xb6'` → dosya temiz UTF-8, **mojibake fix YAPILMAZ** (`ftfy`/latin1 veriyi bozar; konsoldaki bozuk gorunum yalnizca Windows terminal codepage'idir, dosya degil). Olculen: kelime uzunlugu mean **33.2** (min 17, max 59), karakter mean **273.5**, benzersiz metin **10000/10000** (tam sablon ama her satir farkli), **rakam iceren satir = 0** (hazir-cevap/hedef sizintisi yok → gercek semantik gerekli), test kelime ort. 33.1 (train ile ozdes). Anahtar kelime frekanslari (substring): `ancak` 5831, `geliştir` 6302, `potansiyel` 2041, `güçlü` 3097, `başarı` 2526, `mükemmel` 468, `olağanüstü` 184, `üstün` 74, `gerekiyor` 325, `eksik` 447. Bu frekanslar Faz 5 lexicon tasariminin baz cizgisi (lexicon ALAN-BILGISIYLE sabit, hedefe bakarak SECILMEZ).

11. **Raporlama.** Tum cikti artefaktlarini yaz; `eda_report.html`'i 7 bolume bol (Hedef / Eksik / Kategorik / Sayisal-Korelasyon / Yil-Kayma / Adversarial / Metin) — bu yapi dogrudan juri sunumunun "veri portresi" slaytina cevrilir.

## Kararlar & Gerekceler

- **Adversarial validation'i ZORUNLU yapmak (sadece grafik degil).** Cunku tum validation mimarisinin (random stratified KFold) gecerliligi "yillar disinda kayma yok" hipotezine baglidir. Olculmeden kabul edilen hipotez, CV-private gap'in gizli kaynagidir. *Alternatif (sadece marjinal dagilim grafigine bakmak)* elendi: cok degiskenli etkilesim kaymasini yakalamaz, AUC tek skorla net karar verir.
- **Yil kolonlarini "supheli" isaretleyip hedef-by-yil drift'ini olcmek.** Bunun amaci, yili atmanin maliyetini (hedef sinyali kaybi) nicelemek. Olculdu: ~3.6 puan zayif drift → atmak bedava. *Alternatif (yili tutup target-encode etmek)* elendi: adversarial AUC'yi yeniden 0.6+ acar, model test'te kayan nicelige kilitlenir.
- **Sezgisel feature'lari EDA'da olcup uretmemek.** `conv_rate` (corr 0.011) gibi gurultuyu Faz 4'e tasimadan once eleyerek kombinatoryal patlamayi ve overfit yuzeyini kaynaginda kesmek. *Alternatif (kitchen-sink: tum oranlari uret)* elendi: her gereksiz feature 0.25·std kapisina yuk bindirir ve overfit riskidir.
- **Metinde rakam yoklugunu acikca dogrulamak.** Eger metin hedefin sayisal izini tasisaydi (or. "skoru 85"), bu de-facto hedef sizintisi olurdu. 0 rakam → metin guvenli semantik sinyal; Faz 5 TF-IDF→Ridge meta-feature yaklasimi mesru.
- **`university_tier`'i ordinal isaretlemek.** Tier 1-4 dogal siralidir; Faz 3/4 bunu ordinal encode edebilir (one-hot yerine) — EDA bu yapiyi belgeler, kararı sonraki faza birakir.

## Leakage / Overfit Guardrail'lari
- **EDA tam dataset uzerinde calisir AMA hicbir istatistik modele/feature'a tasinmaz.** Korelasyon, eksik oran, kardinalite gibi *betimleyici* istatistikler tum train uzerinde hesaplanabilir cunku bunlar feature uretmez. Aksine: impute degerleri, target-encoding, TF-IDF vocab, scaler — bunlarin hicbiri bu fazda hesaplanmaz; hepsi Faz 2-5'te fold-ici fit edilir (`leakageRules` fold-ici kurali).
- **Lexicon hedefe bakarak SECILMEZ.** EDA'da anahtar kelime ↔ hedef korelasyonu *gozlem amacli* loglanabilir, ama Faz 5 lexicon'u ALAN-BILGISIYLE sabitlenir; korelasyonla kelime ayiklamak de-facto target leakage olur (`leakageRules` lexicon kurali).
- **`student_id` analizde gruplama/profilleme icin bile feature olarak dusunulmez** (sentetik, non-predictive, train/test araliklari ayrik).
- **Yil kolonu sigortasi:** adversarial AUC bu fazda olculur ve `adversarial_auc.json`'a sabitlenir; Faz 4 nihai matriste AUC~0.5 dogrulamasini bu baz cizgisine gore yapar.
- **Turkce lowercase tuzagi:** EDA'da metin normalizasyonu (`str.lower()`) yapilirsa Faz 5 ile AYNI Turkce-duyarli normalizasyon (`I`→`ı`, `İ`→`i`) kullanilir; tutarsiz normalizasyon sessizce frekans olcumlerini bozar (`leakageRules` lowercase kurali).

## Teslimler (Deliverables)
1. `src/01_eda.py` — SEED=42, internet kapali, bastan sona reproducible; tum sayisal bulgular hardcode degil hesaplanir.
2. `reports/eda/column_profile.csv`, `missing_map.csv`.
3. `reports/eda/target_profile.json`, `adversarial_auc.json`, `text_profile.json`.
4. `reports/eda/eda_report.html` — 7 bolumlu, juri sunumuna donusebilir.
5. EDA bulgular ozeti (script ici markdown/yorum) → Faz 2-5 SPEC'lerinin "Girdiler" bolumune referans.

## Definition of Done
- [ ] Train/test shape ve kolon farki assert ile dogrulandi (`(10000,47)` / `(10000,46)`, fark sadece hedef).
- [ ] `target_profile.json` uretildi; `pct_eq_100≈7.73`, `pct_le_50≈4.97`, `skew≈-0.45` raporlandi.
- [ ] `missing_map.csv` uretildi; 7 NA kolonu + internship MNAR orani (`≈82%`) belgelendi.
- [ ] 5 kategorigin kardinalite ve seviye listesi cikarildi; test-only seviye YOK teyidi.
- [ ] Hedefe `|corr|` sirali tablo uretildi; `project_quality_score≈0.54` dogrulandi.
- [ ] **Adversarial AUC olculdu: yillarla `≈0.66`, yillarsiz `≈0.50`** ve `adversarial_auc.json`'a yazildi (Faz 2 icin kritik DoD).
- [ ] Yil dagilim kaymasi (train uniform / test 2024-26 yiginli) ve hedef-by-yil zayif drift belgelendi.
- [ ] Metin: UTF-8 byte teyidi, benzersizlik (10000/10000), rakam yoklugu (0), uzunluk istatistikleri, anahtar kelime frekanslari `text_profile.json`'a yazildi.
- [ ] `python src/01_eda.py` internet kapali hatasiz calisti, ayni sayilari uretti (deterministik).

## Riskler & Azaltim
- **Risk: EDA'da gozlemlenen bir korelasyonun feature secimine "sizip" overfit'e yol acmasi.** → Azaltim: EDA yalniz betimler, karar vermez; tum feature kararlari Faz 4'te fold-ici delta + 0.25·std kapisindan gecer.
- **Risk: Konsolda gorulen mojibake'nin gercek veri bozulmasi saniip `ftfy` ile "duzeltmeye" calismak.** → Azaltim: ham byte teyidi (`\xc3\xb6`) DoD'a kondu; dosya temiz UTF-8, mojibake yalniz terminal codepage'i. Fix YASAK.
- **Risk: Adversarial AUC'nin tek seed'e bagli olmasi.** → Azaltim: 5-fold CV ortalamasi kullanilir; sinir durumda (0.55-0.60) ikinci seed ile teyit.
- **Risk: Yil kolonu disinda gozden kacan ikincil kayma.** → Azaltim: adversarial feature importance loglanir; yillar disinda AUC'ye katki yapan kolon cikarsa Faz 2/4 incelemesine bayrak atilir (su an AUC=0.50, ikincil kayma yok).
- **Risk: Stratify bin tasariminin EDA'da yanlis olcekte planlanmasi.** → Azaltim: `==100` ve `<=50` oranlari net raporlanir; Faz 2 `qcut(q=9)` + ayri `==100` bin'i bu oranlara gore boyutlandirir.

## Sure / Zaman Kutusu
**Gun 1 (9 Haz) — Temel & Anchor** icinde. EDA, Faz 2 (validation) ile ayni gun tamamlanir: sabah EDA + adversarial dogrulama (bu faz), ardindan `data/folds.parquet` uretimi (Faz 2) ve sadece-sayisal LGBM anchor (~91.6 CV MSE). EDA toplam ~3-4 saatlik bir blok; cunku karar agirligi adversarial sigorta ve stratify-bin olcumunde, kozmetik grafiklerde degil. Gun sonunda 1 anchor submission ile ilk CV-LB gap olculur.

## Capraz Referanslar
- **Faz 2 (Validation Strategy):** `adversarial_auc.json` (0.50 hipotezi) ve `target_profile.json` (==100/<=50 oranlari) dogrudan stratified-KFold + stratify-bin tasariminin girdisidir. Yil kararı (`lockedDecisions` "Yil kolonlari") bu fazda kanitlanir.
- **Faz 3 (Preprocessing):** `missing_map.csv` + MNAR analizi → "internship NA→0 + bayrak, diger 6 icin fold-ici medyan + bayrak" planinin temeli.
- **Faz 4 (Feature Engineering):** `column_profile.csv` korelasyonlari + `tech_mean`/`project_quality×tech_mean`/`conv_rate` on-olcumleri → kompozit ve carpim feature secimi; nihai adversarial AUC~0.5 dogrulamasi bu fazin baz cizgisine gore.
- **Faz 5 (NLP Text):** `text_profile.json` (uzunluk, benzersizlik, rakam yoklugu, anahtar kelime frekanslari, UTF-8 teyidi) → TF-IDF→Ridge meta-feature + sabit lexicon tasariminin girdisi.
- **Faz 7 (Evaluation/Submission):** hedef [0,100] siniri + `==100` kutlesi → `np.clip(0,100)` post-processing ve submission format assert gerekcesi; `sample_submission`'daki 123.94'un yalniz format ornegi oldugu teyidi.
- **Faz 0 (Masterplan) & Kanonik Strateji:** Bu fazin tum sayisal bulgulari `cvProtocol`, `leakageRules`, `lockedDecisions` ile tutarli ve onlari KANITLAR; celiskili bir bulgu cikarsa (or. adversarial AUC yillarsiz >0.55) Masterplan'a geri bildirim verilir.
