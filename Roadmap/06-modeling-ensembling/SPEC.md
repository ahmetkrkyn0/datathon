# Faz 6 — Modelleme & Ensembling

> Manda (02'den miras): **SIFIR OVERFIT.** Bu fazda uretilen her model, her hiperparametre ve her blend agirligi
> SADECE `02-validation-strategy`'deki Repeated Stratified 5-fold x 3 protokolu uzerinden, `data/folds.parquet`
> ile olculur. Public LB pesinde KOSULMAZ; karar otoritesi CV-MSE(mean, std)'dedir.

---

## Amac

Sizinti-guvenli, fold-ici fit edilen GBDT base learner'lari (LightGBM-L2, CatBoost, HistGBR) + Faz 05'in
`txt_ridge_pred` metin sinyalini birlestirip, seed-averaging ve regularize Ridge/NNLS blend ile **CV-MSE'yi
anchor ~91.6'dan ~83'e indiren, CV ile private leaderboard arasinda sapmasiz** bir tahmin uretmek.

---

## "0 Overfit" Rolu

Bu faz overfit'in en kolay sizdigi yerdir (model kapasitesi, HP arama, stacking) — bu yuzden tum savunma burada yogunlasir:

1. **Tum fit'ler fold-ici.** Hicbir base model, encoder, imputer veya metin transformer tum-train uzerinde fit edilmez (02 §2.2). CV-MSE bu sayede private'in sapmasiz tahmincisi kalir.
2. **Muhafazakar HP + dar Optuna.** Kapasite bilincli kisilir (num_leaves <=63, min_child_samples >=50, reg_lambda >=1); Optuna <=50 trial ve objective=repeated-CV mean (CV'nin kendisine overfit'i sinirlanir).
3. **Seed-averaging ONCE.** Varyans dusurmenin sifir-overfit-riskli yolu (sadece gurultu iptali) once uygulanir; karmasiklik (stacking) sona birakilir ve kabul kapisindan gecmek zorundadir.
4. **0.25*std kabul kapisi (02 §3.4).** Her base model, her feature, her blend bilesimi `yeni_cv_mean < eski_cv_mean - 0.25*cv_std` saglamadikca REDDEDILIR; esitlikte daha basit model kazanir.
5. **Yapisal cesitlilik = private risk dagitimi.** Final 2 submission (sade capa tek-model vs ensemble) ayni anda cokmesin diye yapisal olarak farkli secilir.

---

## Girdiler / Cikti Artefaktlari

### Girdiler (onceki fazlardan)
- `data/folds.parquet` — (`student_id`, `repeat`, `fold`); 5x3 stratified master fold (Faz 02). **TUM modeller bunu kullanir.**
- `data/train_fe.parquet` / `data/test_fe.parquet` — Faz 03+04 ciktisi feature matrisi (kompozitler `tech_mean`/`soft_mean`/`interview_mean`/`profile_mean`, `project_quality_score*tech_mean`, github log1p, MNAR bayraklari). **Yil kolonlari (application_year/graduation_year) ham haliyle YOK** (02 §5, leakageRules).
- `oof_txt_ridge.npy` / `test_txt_ridge.npy` — Faz 05'in nested inner-KFold OOF metin sinyali (`txt_ridge_pred`) + lexicon/yapi ozellikleri (`n_pos`, `n_neg`, `pos_minus_neg`, `has_ancak`, `len_word`, `n_sentence`).
- `02-validation-strategy/SPEC.md` — ust otorite CV protokolu, OOF sozlesmesi, gap esikleri.
- Sizinti-guvenli fold-ici Pipeline iskeleti (Faz 01/03'ten): impute + `_missing` bayrak, kategorik encoder.

### Ciktilar (sonraki fazlara)
- `artifacts/oof_<model>.npy` + `artifacts/test_<model>.npy` her base learner icin (`lgbm_num`, `lgbm_full`, `catboost_full`, `histgbr_full`, ops. `xgb_full`). Sekil (10000,), satir-hizali (`student_id` sirasi `folds.parquet` ile ozdes).
- `artifacts/cv_scores.csv` — model basina `cv_mse_mean`, `cv_mse_std`, `best_iteration_mean`, `n_seeds`.
- `artifacts/oof_blend.npy` + `artifacts/test_blend.npy` — Seviye-1 Ridge/NNLS blend ciktisi.
- `artifacts/blend_weights.json` — base modellerin nihai (negatif-olmayan) agirliklari + Ridge alpha.
- `models/<model>_refit/` — tum-train refit edilmis nihai modeller; `best_iteration` OOF ortalamasiyla SABIT.
- `submissions_log.csv` satirlari (Faz 02 semasi) — her LB cikan modelin gap kaydi.
- `07-evaluation-submission`'a teslim: SUB-1 (capa) ve SUB-2 (ensemble) aday tahmin vektorleri, clip[0,100] uygulanmis.

---

## Detayli Adimlar

1. **Ortak OOF kosucu (runner) yaz.** `folds.parquet`'i okuyan tek bir fonksiyon: her (repeat, fold) icin train/valid ayir, fold-ici fit/transform pipeline'i uygula, model egit, valid'e tahmin -> `oof`, test'e tahmin -> fold ortalamasina ekle. 3 repeat'in `oof`'u ortalanir. Cikti: `oof_<model>`, `test_<model>`, fold-MSE listesi (mean, std). Bu kosucu TUM base learner'lar icin paylasilir (02 §2.1 OOF sozlesmesi).

2. **LGBM-num anchor (Gun 1 capa).** `lightgbm.LGBMRegressor(objective='regression_l2')`, sadece sayisal+FE+kategorik (metin YOK). `deterministic=True`, sabit `num_threads`, `seed=42`. Fold-ici early stopping (`callbacks=[early_stopping(100)]`, valid set = o fold'un valid'i). Hedef: CV-MSE ~91.6 (std ~4.68) reproduce. Bu anchor referans noktasidir.

3. **Kategorik kodlama ablasyonu.** LGBM icin iki yol fold-ici karsilastir: (a) OOF target-encoding + Bayesian smoothing (m~20-50), (b) one-hot (kardinalite 4-11, dusuk). Kazanani 0.25*std kapisiyla sec. CatBoost'ta kategorikler native (kodlama yok) -> bagimsiz bias kaynagi.

4. **LGBM-full.** Anchor feature setine `txt_ridge_pred` + lexicon ozelliklerini ekle. Hedef CV-MSE ~83 (Faz 05 olcumu 83.21, std 4.05). Kabul kapisini gecip gecmedigini logla.

5. **CatBoost-full.** `CatBoostRegressor(loss_function='RMSE', random_seed=42)`, 5 kategorigi `cat_features` ile native ver, ordered boosting acik (gercek ensemble cesitliligi). `od_type='Iter'` ile fold-ici early stopping. Ayni feature seti.

6. **HistGBR-full.** `sklearn.ensemble.HistGradientBoostingRegressor(loss='squared_error', random_state=42, early_stopping=True, validation_fraction` yerine fold-ici valid kullan). Bagimliliksiz, saf-sklearn 3. cesit -> reproducibility sigortasi.

7. **Seed-averaging (ILK varyans dusurucu).** Her base model 3-5 seed (or. [42, 2026, 7, 13, 101]) ile fold-ici egitilir; OOF ve test tahminleri seed'ler arasi ortalanir. Bu sifir-overfit-riskli, en yuksek ROI'li adim.

8. **Optuna dar arama (yalniz LGBM).** `optuna` <=50 trial, objective = repeated-CV mean (`folds.parquet` uzerinde), secim cv_std ile cezalandirilir (or. `score = mean + 0.25*std`). Arama uzayi muhafazakar: num_leaves 31-63, max_depth 5-7, lr 0.02-0.05, min_child_samples 50-100, feature_fraction 0.7, bagging_fraction 0.8, reg_lambda >=1. Secilen HP ayri seed/repeat ile dogrulanir (HP meta-leak'e karsi).

9. **Seviye-1 blend.** `oof_*` kolonlari uzerinde, AYNI dis fold semasiyla (base OOF zaten fold-disi -> nested gerekmez) `Ridge(alpha=CV ile secilen)` VEYA `scipy.optimize.nnls` ile negatif-olmayan agirlikli ortalama. Test blend ayni agirliklarla `test_*` uzerinde. GBM-stacker VARSAYILAN DEGIL — sadece nested CV'de Ridge'i >1 cv_std gecerse.

10. **Lineer/NN opsiyon degerlendirmesi.** Ridge-on-FE baseline base'e cesitlilik katiyor mu fold-ici test et; katmiyorsa ekleme (Occam). Sinir agi (kucuk MLP) DEGERLENDIRILIR ama 10k satirda overfit + repro maliyeti yuksek -> sadece OOF-MSE'yi net (>0.5) gecerse ve ancak 2. submission'a.

11. **Tum-train refit.** Secilen base'leri ve blend'i tum train uzerinde refit et; LGBM/CatBoost/HistGBR `best_iteration` = OOF fold'larinin ortalama best_iteration'i ile SABIT (refit'te early stopping icin test/valid KULLANMA — gizli sizinti, leakageRules FINAL-FIT).

12. **Clip + final aday uretimi.** Tum tahminlere `np.clip(pred, 0, 100)`. SUB-1 = en dusuk cv_mean'li EN BASIT tek GBDT (muhtemelen tuned LGBM-full). SUB-2 = en dusuk cv_mean'li blend. Ikisinin yapisal farki dogrulanir.

---

## Kararlar & Gerekceler

- **Neden GBDT ana is gucu?** Hicbir feature baskin degil (max korr ~0.54), sinyal etkilesimlerde (`project_quality*tech_mean` korr 0.606). GBDT'ler lineerden ustun; `objective='regression_l2'` MSE metrigine birebir uyar.
- **Neden 3 farkli GBDT?** LGBM (leaf-wise L2), CatBoost (ordered boosting, native kategorik), HistGBR (saf sklearn) gercek-cesitli bias kaynaklari -> blend tek modeli guvenilir gecer. XGBoost opsiyonel cunku LGBM+CatBoost ustune nadiren cesitlilik katar.
- **Neden seed-averaging once, stacking sonra?** Seed-averaging yalniz varyansi iptal eder (sifir overfit ekler) -> en ucuz/guvenilir kazanc. Agir stacking 10k OOF'ta CV'ye overfit eder ve CV-private gap acar -> nested CV kapisina baglanir.
- **Neden Optuna sade ve cv_std cezali?** 10k satirda agresif tuning CV'nin kendisine overfit eder; regularizasyon-agir varsayilanlar genelleme tarafini secer. cv_std cezasi "sansli ortalama" trial'lari eler.
- **Neden ham hedef + clip, log/logit YOK?** Skew -0.45 (neredeyse normal), cift-sinirli yiginla log/logit iyi calismaz; clip [0,100] bedava/notr kazanc (gercek hedef kesin [0,100]).
- **Elenen alternatifler:** (a) Tek 10-fold -> fold basina 1000 satir, varyans patlar (02 §1). (b) GBM-stacker varsayilan -> CV-private gap acar. (c) Iki-asamali P(y=100)+regressor -> clever-but-fragile, OOF kaniti olmadan reddedilir. (d) Ham TF-IDF'i agaca verme -> 20k seyrek kolonla overfit (Faz 05 olcumu: Ridge-OOF 83.21 < ham TF-IDF).

---

## Leakage / Overfit Guardrail'lari

- **Fold-ici fit MUTLAK (02 §2.2, leakageRules):** imputer, target-encoder, scaler, her sey o dis fold'un train'inden fit edilir. Bu fazda yeni eklenen riskli yer: target-encoding'i kosucunun ICINDE fit ettiginden emin ol — global encode KESINLIKLE YASAK.
- **Stack/blend leak:** Seviye-1 meta-model SADECE `oof_*` (fold-disi) uzerinde egitilir; in-fold tahminle stacking asiri-iyimser OOF uretir ve private %40'ta tutmaz.
- **Final-fit leak:** refit'te `best_iteration` OOF ortalamasiyla SABIT; refit early stopping icin test/valid kullanma.
- **Optuna/HP meta-leak:** arama tum CV katlari uzerinde tekrar tekrar optimize edilirse CV'ye overfit; <=50 trial, secimi ayri seed/repeat ile dogrula.
- **Yil kolonu leak:** `lgbm_full`/`catboost_full` feature listesinde `application_year`/`graduation_year` ham veya target-encode HALDE OLMADIGINI assert et (adversarial AUC ~0.5 nihai matriste teyit, 02 §5).
- **student_id leak:** feature matrisinden `student_id` dropla (sentetik anahtar, non-predictive).
- **Determinizm:** `deterministic=True` (LGBM), sabit `random_seed`/`random_state`, `PYTHONHASHSEED`, sabit thread sayisi -> ayni OOF-MSE tekrar uretilebilir.

---

## Teslimler (Deliverables)

1. `artifacts/oof_<model>.npy`, `artifacts/test_<model>.npy` (lgbm_num, lgbm_full, catboost_full, histgbr_full; ops. xgb_full).
2. `artifacts/cv_scores.csv` (model x cv_mse_mean/std, best_iteration_mean, n_seeds).
3. `artifacts/oof_blend.npy`, `artifacts/test_blend.npy`, `artifacts/blend_weights.json`.
4. `models/<model>_refit/` tum-train refit modelleri (best_iteration sabit).
5. Ablation tablosu (markdown): anchor 91.6 -> +FE -> +txt_ridge_pred -> +CatBoost/HistGBR -> +seed-avg -> +blend, her satirda cv_mean/std ve 0.25*std kapi karari.
6. `submissions_log.csv`'ye eklenen gap kayitlari (FE'li, NLP'li, ensemble modelleri).
7. `07`'ye teslim: SUB-1 ve SUB-2 aday tahmin vektorleri (clip uygulanmis, ID-hizali).

---

## Definition of Done

1. LGBM-num anchor reproduce edildi: cv_mse_mean ~91.6, std ~4.68 (+/- gurultu) — `cv_scores.csv`'de.
2. LGBM-full cv_mse_mean ~83 (Faz 05 hedefi 83.21) ve 0.25*std kapisini ANCHOR'a gore gecti.
3. CatBoost-full + HistGBR-full base'leri uretildi, her biri 3-5 seed-averaged, OOF/test `.npy` diske yazildi.
4. Seviye-1 blend, en iyi tekil base'i kabul kapisindan (>0.25*std) gecti VEYA gecemediyse blend yerine en iyi tekil model SUB-2 olarak belgelendi (Occam).
5. Tum tahminler clip[0,100]; clip-disi deger yok (assert ile dogrulandi).
6. Adversarial AUC nihai feature matrisinde ~0.5 teyit (>0.6 ise suclu feature notu).
7. Tum-train refit best_iteration OOF ortalamasiyla sabit; refit'te valid/test kullanilmadi (kod incelemesiyle dogrulandi).
8. SUB-1 (sade capa) ve SUB-2 (ensemble) yapisal olarak FARKLI ve ikisi de CV ile secildi; `submissions_log.csv` gap'leri saglikli (|gap| <= 1.5*std).
9. Pipeline 'deterministik tam koşu' (`python src/...py`) internet-kapali ayni cv_mse_mean/std'yi uretti (reproducibility).

---

## Riskler & Azaltim

| Risk | Belirti | Azaltim |
|---|---|---|
| HP arama CV'ye overfit | Optuna best trial CV cok dusuk, public sapiyor | <=50 trial, cv_std cezali objective, ayri seed/repeat dogrulama |
| Stacking overfit | Blend OOF tekil base'lerden cok dusuk, gap kirmizi | GBM-stacker yasak; Ridge/NNLS varsayilan; >1 cv_std kapisi |
| Target-encoding sizintisi | FE'li model anormal dusuk CV, gap negatif (public>CV) | Encode kosucu ICINDE fold-ici; global encode assert ile yasak |
| Yil turevi geri sizar | Adversarial AUC nihai matriste >0.6 | Feature listesi assert; suclu feature cikar |
| CatBoost/HistGBR reproducibility | Tekrar kosulunca farkli OOF-MSE | random_seed/state sabit, deterministik flag, thread sabit |
| Seed-averaging marjinal | Ek seed cv_mean'i degistirmiyor | 3 seed sonrasi azalan getiri -> 3-5 ile sinirla, zaman koru |
| Refit best_iter yanlis | Refit modeli OOF'tan farkli davraniyor | best_iteration = fold ortalamasi; underfit/overfit'i OOF ile capraz dogrula |

---

## Sure / Zaman Kutusu

**Gun 4 (12 Haziran) — Modelleme & Ensemble.** Bu fazin ana gunu:
- CatBoost-full + HistGBR-full base'lerini ekle, her base 3-5 seed-averaging.
- Optuna <=50 trial (CV-mean objective, cv_std cezali) LGBM icin dar arama.
- `oof_*` uzerinde Ridge(alpha CV)/NNLS blend; GBM-stacker yalniz >1 cv_std gecerse.
- best_iteration OOF ortalamasiyla tum-train refit.
- 1-2 submit (ensemble) gap teyidi (>=3 hak rezerv).

Hazirlik bagimliligi: LGBM-num anchor Gun 1'de (Faz 01/02 ile birlikte), LGBM-full Gun 2-3'te FE ve NLP ciktisi gelir gelmez. Gun 5 (13-14 Haz) bu fazda YENI RISK YOK — sadece dondurma ve final 2 secimi (Faz 07).

---

## Capraz Referanslar

- **02-validation-strategy/SPEC.md** — UST OTORITE: CV protokolu (Repeated Stratified 5x3, `folds.parquet`), OOF sozlesmesi (§2), gap esikleri (§3), regularizasyon disiplini (§4), adversarial (§5), final 2 submission kurali (§6). Bu fazin tum olcumleri oradaki protokole tabidir.
- **03-preprocessing-cleaning/SPEC.md** — fold-ici imputer + `_missing` bayraklari, MNAR internship 0+bayrak; bu fazin pipeline'ina girer.
- **04-feature-engineering/SPEC.md** — `train_fe.parquet` kompozitleri ve `project_quality*tech_mean`; base learner feature setinin kaynagi. Kabul kapisi (0.25*std) ortak.
- **05-nlp-text-features/SPEC.md** — `oof_txt_ridge.npy` / `txt_ridge_pred` + lexicon ozellikleri; `lgbm_full`/`catboost_full`/`histgbr_full`'a girer. Char n-gram'in elendigi ablation buradan.
- **07-evaluation-submission/SPEC.md** — bu fazin SUB-1/SUB-2 adaylarini ve `submissions_log.csv` kayitlarini devralir; final dondurma ve reproducibility testi orada.
