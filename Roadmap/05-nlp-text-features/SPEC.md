# Faz 5 — NLP: Türkçe `mentor_feedback_text` Özellik Çıkarımı

## Amaç
Mentörün Türkçe serbest-metin değerlendirmesindeki **sayısal kolonlardan bağımsız** semantik sinyali, sızıntısız (fold-içi fit) tek bir sürekli meta-feature (`txt_ridge_pred`) + 8-12 elle tasarlanmış sözlük/yapı özelliğine sıkıştırarak GBDT'lere besleyip CV-MSE'yi ~89.86'dan ~83'e indirmek.

## "0 Overfit" Rolü
Bu faz, "0 overfit" north-star'ına metni **modele ham olarak vermeden** hizmet eder. Çekirdek tehlike: 10k satırda 20k+ seyrek TF-IDF kolonunu doğrudan ağaca vermek garantili overfit ve CV-private uçurumu üretir. Çözüm: metnin lineer sinyalini Ridge ile **tek temiz sayıya** indirgemek (`txt_ridge_pred`) — minimum overfit yüzeyi, maksimum yorumlanabilirlik. Ölçülen kanıt: NUM-only 89.86 → NUM+Ridge-OOF **83.21 (std 4.05)**; bu sonuç ham TF-IDF'i ağaca vermekten ve SVD30'dan (84.40) hem daha iyi hem daha düşük varyanslıdır. Tüm metin türevleri kanonik leakage kurallarına (`leakageRules`: TF-IDF/SVD/EMBEDDING LEAK, LEXICON LEAK, TÜRKÇE LOWERCASE TUZAĞI) ve `cvProtocol`'deki nested-OOF sözleşmesine bağlıdır; metinde rakam **olmadığı doğrulandı** (`has_digit=0.0`), yani hedef/skor sızıntısı yoktur — sinyal gerçek semantiktir.

## Girdiler / Çıktı Artefaktları
**Girdiler (önceki fazlardan):**
- `data/folds.parquet` (Faz 02): `(student_id, repeat, fold)`, 5×3 stratified. TÜM OOF üretimi bu dosyaya satır-hizalı. Asla yeniden fold üretme.
- `data/train.csv` / `data/test_x.csv` ham `mentor_feedback_text` (Faz 01 teyit: 10000/10000 unique, NaN yok, ~33 kelime, 1-4 cümle, rakamsız).
- Faz 03'ten temizleme yardımcıları (Türkçe-duyarlı lowercase fonksiyonu) — ortak `text_utils.py` içinde paylaşılır.

**Çıktılar (sonraki fazlara):**
- `data/oof_txt_ridge.npy` (10000,) — train için nested-OOF `txt_ridge_pred`.
- `data/test_txt_ridge.npy` (10000,) — test için inner-model ortalaması.
- `data/text_features_train.parquet` / `data/text_features_test.parquet` — elle tasarlanmış sözlük/yapı kolonları (Katman B), `student_id` anahtarlı.
- `src/text_utils.py` — `turkish_lower()`, `build_tfidf_ridge_oof()`, `extract_handcrafted_features()` (deterministik, SEED=42).
- `data/nlp_ablation.csv` — NLP'siz vs NLP'li OOF-MSE ablation tablosu (Faz 07 sunumunun çekirdeği).
- `src/lexicon_tr.py` — alan-bilgisiyle SABİTLENMİŞ pozitif/negatif/gelişim sözlükleri.

`txt_ridge_pred` ve elle özellikler **base model değil**; Faz 04'ün feature matrisine kolon olarak eklenir, Faz 06 base learner'ları (LGBM-full, CatBoost-full, HistGBR-full) bunları tüketir.

## Detaylı Adımlar

1. **Encoding & temizleme.** `pd.read_csv(..., encoding='utf-8')`. Dosya temiz UTF-8 (ham byte teyit: 'ö'=\xc3\xb6); ftfy/latin1 mojibake düzeltmesi **YAPILMAZ** (veriyi bozar — konsoldaki bozuk görünüm yalnızca Windows terminal codepage'idir, dosya sağlamdır). Türkçe-duyarlı lowercase: `s.replace('I','ı').replace('İ','i').lower()` — `str.lower()` tek başına 'I'→'i' yapıp Türkçe eşleşmeleri sessizce kaçırır. Metin VE sözlük **aynı** normalizasyondan geçer.

2. **Katman A — TF-IDF + Ridge → nested-OOF meta-feature (ANA, zorunlu).**
   - `TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=3, sublinear_tf=True, max_features=20000)`.
   - Türkçe stopword: yalnızca **içerik-taşımayan** fonksiyon kelimeleri (`ve, ile, bir, bu, için, da, de, daha, olan, gibi`) çıkarılır; **`ancak` ASLA stopword DEĞİL** (en güçlü negatif/koşul sinyali, %58.3 satırda — ölçülen `pos_minus_neg` etkisi -7.4). `min_df=3` zaten gürültü token'ı eler.
   - `Ridge(alpha=2.0)` (alpha inner-fold CV ile {1,2,5} arasından doğrulanır).
   - **Nested inner-KFold OOF:** dış fold train'i içinde 5-fold; her inner-train'de vectorizer+Ridge fit, inner-valid'e transform → dış fold train'inin OOF `txt_ridge_pred`'i. Dış-valid ve test tahmini, o dış fold'un inner modellerinin **ortalaması**. 3 repeat ortalaması nihai `oof_txt_ridge`. (83.21 sonucu tam bu nested-OOF ile üretildi.)
   - Tahminler `np.clip(0,100)`.

3. **Katman B — elle tasarlanmış Türkçe sözlük/yapı özellikleri (ANA yanında).** Substring/kök eşleştirme (`'geliştir' in token` → tüm çekimleri yakalar). Ölçülen satır kapsamı parantezde:
   - `n_pos` = pozitif token sayısı (lexicon: `etkileyici`(16.2%)/`güçlü`(31.0%)/`yüksek`/`başarı`(25.3%)/`mükemmel`(4.7%)/`olağanüstü`(1.8%)/`üstün`(0.7%)/`potansiyel`(20.4%)/`hakimiyet`(0.5%)/`dikkat çek`(31.9%)).
   - `n_neg` = negatif/gelişim token sayısı (`ancak`(58.3%)/`geliştir`(63.0%)/`gerekiyor`(3.2%)/`eksik`(4.5%)/`ihtiyaç`(4.2%)/`daha fazla`/`düşük`(2.7%)/`zayıf`(3.9%)).
   - `pos_minus_neg` = `n_pos - n_neg` (en güçlü tek skaler sentiment proxy'si).
   - `has_ancak` (0/1) — koşul/çekince işareti, %58.3.
   - `len_word` = kelime sayısı (range 17-59, ort 33).
   - `len_char` = karakter sayısı (143-447).
   - `n_sentence` = nokta sayısı tabanlı cümle sayımı (1-4, ort 2.44).
   - `n_skill_mention` = teknik beceri anahtarı sayımı (`sql/backend/frontend/devops/cloud/makine öğren/veri yapı/portföy/github`).
   - `pos_ratio` = `n_pos / (len_word+1)`, `neg_ratio` = `n_neg / (len_word+1)`.
   - Tek başına Katman B: 89.86 → 86.52 (ölçülen). Çıktı `text_features_*.parquet`'e yazılır.

4. **Char n-gram ABLATION (negatif sonuç, belgelenir).** `analyzer='char_wb', ngram_range=(3,5)` denenir, CV-MSE delta ölçülür ve `nlp_ablation.csv`'ye işlenir. Ölçülen: CV kötüleşti **89.38 → 90.71** (metin temiz şablon, yazım hatası yok → karakter sinyali yok). **Final pipeline'dan ÇIKARILIR**; "denedik, kötüleşti, çıkardık" anlatısı sunuma girer.

5. **SVD fallback (yalnızca Katman A geçmezse).** `TruncatedSVD(n_components=30)` ham TF-IDF üzerine, 30 kolon GBDT'ye. Yalnızca Ridge-OOF kabul kapısından geçemezse devreye alınır (ölçülen 84.40 < Ridge'in 83.21'i olduğundan **varsayılan değil**).

6. **Katman C — Turkish BERT (OPSİYONEL, sadece 2. submission).** `dbmdz/bert-base-turkish-cased` frozen, mean-pooled embedding → `TruncatedSVD(16-32)` → ek kolon(lar). **Fine-tune YOK** (overfit + repro riski). Embedding fold-içi SVD ile indirgenir (SVD fold-safe). Offline (Kaggle Dataset, internet kapalı). **Sadece** nested CV'de Ridge-meta+lexicon'u **>0.5 MSE net** geçerse alınır; geçemezse rapor edilip terk edilir.

7. **Entegrasyon kararı (concat, ayrı-model-blend DEĞİL).** `txt_ridge_pred` + Katman B kolonları, Faz 04 feature matrisine **concat** edilir; GBDT'ler bunları diğer feature'larla birlikte tüketir. "Ayrı metin modeli + blend" reddedildi: Ridge zaten metnin lineer sinyalini özetliyor; ikinci seviye blend gereksiz katman + overfit yüzeyi açar.

## Kararlar & Gerekçeler
- **Neden Ridge-OOF tek kolon, ham TF-IDF değil?** Ölçüldü: Ridge-OOF 83.21 (std 4.05) > SVD30 84.40 > ağaca ham TF-IDF. Ağaçlar binlerce seyrek kolonla overfit eder; Ridge lineer metin sinyalini sızıntısız tek sayıya sıkıştırır → min overfit yüzeyi + "mentör metni modeli" sunum anlatısı.
- **Neden char n-gram yok?** Ölçülen CV kötüleşmesi (89.38→90.71); metin temiz şablon, yazım hatası yok → alt-kelime/karakter sinyali katkısız.
- **Neden BERT ana hatta değil?** Marjinal kazanç, yüksek repro/overfit maliyeti; `cvProtocol` kabul kapısı (0.25*std) marjinali zaten reddeder. Yalnızca 2. submission'da yapısal çeşitlilik için opsiyonel.
- **Neden sözlük alan-bilgisiyle sabit?** Kelime-hedef korelasyonuyla seçim de-facto target leakage'tır (`leakageRules`: LEXICON LEAK). Sabit sözlük fold-bağımsız ve güvenli.
- **Neden concat, neden ayrı blend değil?** Tek-kolon meta + elle özellikler GBDT'nin etkileşim gücünden faydalanır; ekstra blend katmanı 10k'da CV'ye overfit eder.

## Leakage / Overfit Guardrail'ları
- **FOLD-İÇİ FİT MUTLAK:** vectorizer, Ridge, SVD, BERT-üstü SVD — hepsi yalnız **inner-train** fold'una fit, valid/test'e transform. Train+test birleşimine veya tüm train'e fit **YASAK**.
- **Nested inner-KFold zorunlu:** `txt_ridge_pred` dış-fold train'i için nested OOF ile üretilmezse CV sahte iyimser olur ve private'ta çöker (`cvProtocol`: NLP META-FEATURE OOF).
- **LEXICON LEAK:** sözlük hedefe bakarak seçilmez; `lexicon_tr.py`'da alan-bilgisiyle sabit, fold-bağımsız.
- **TÜRKÇE LOWERCASE TUZAĞI:** metin ve sözlük aynı `turkish_lower()`; karışık normalizasyon sessizce eşleşme kaçırır.
- **Mojibake fix YASAK:** dosya temiz UTF-8; ftfy/latin1 düzeltmesi veriyi bozar.
- **Vocab sızıntısı:** test metni vectorizer vocab'ına dahil edilmez (CV ölçümünde). Nihai modelde tüm train'e fit + test transform doğru; ama **CV ölçümünde fold-dışına fit yasak**.
- **`student_id` metne girmez**, rakamsız metin doğrulandı (skor sızıntısı yok).

## Teslimler (Deliverables)
1. `data/oof_txt_ridge.npy`, `data/test_txt_ridge.npy` (satır-hizalı, clip[0,100]).
2. `data/text_features_train.parquet`, `data/text_features_test.parquet` (Katman B, `student_id` anahtarlı).
3. `src/text_utils.py`, `src/lexicon_tr.py` (deterministik, SEED=42, pinned).
4. `data/nlp_ablation.csv` — satırlar: NUM-only / +Katman B / +Ridge-OOF / +char(negatif) / +BERT(ops.), her biri cv_mse_mean+std.
5. Reproduce notu: hangi alpha, hangi min_df, nested-fold şeması.

## Definition of Done
- [ ] `oof_txt_ridge.npy` üretildi ve `data/folds.parquet` ile satır-hizalı (10000, `student_id` eşleşmesi assert).
- [ ] NUM + `txt_ridge_pred` + Katman B'nin CV-MSE'si ≤ **84** (hedef ~83.21), std raporlandı; kabul kapısı (`yeni < eski − 0.25*std`) geçildi (anchor 89.86 referans).
- [ ] Tüm vektörleştirici/Ridge/SVD'nin **yalnız** inner-train fold'una fit edildiği koddan doğrulandı.
- [ ] `nlp_ablation.csv` dolduruldu (char n-gram negatif sonucu dahil).
- [ ] Tüm metin tahminleri clip[0,100]; adversarial AUC nihai matriste ~0.5 (yıl türevi eklenmedi).
- [ ] 1-2 submission ile CV-LB gap ölçüldü; `|gap| ≤ 1.5*cv_std` (sağlıklı).

## Riskler & Azaltım
- **Risk: nested-OOF hatası → sahte iyimser CV.** → Birim test: inner-fold indekslerinin dış-train ile kesişmemesi assert; OOF satır kapsamının %100 olması.
- **Risk: `ancak`'ın yanlışlıkla stopword'e girmesi → sinyal kaybı.** → Stopword listesi koda gömülü ve testle korunur; `has_ancak` ayrıca açık feature.
- **Risk: char n-gram cazibesi.** → Ablation tablosu negatif sonucu kanıtlar; eklenmesi kabul kapısından geçemez.
- **Risk: BERT repro/internet bağımlılığı.** → Offline Kaggle Dataset embedding; ana hatta değil, sadece 2. submission opsiyonel.
- **Risk: tüm-train fit ile fold-içi fit karıştırılması.** → Nihai refit ayrı fonksiyon; CV ölçüm fonksiyonu fold-dışına fit yasağını koda gömer.

## Süre / Zaman Kutusu
**Gün 3 (11 Haz) — Türkçe NLP.** Sabah: `text_utils.py`+`lexicon_tr.py`, Katman B özellikleri ve TF-IDF+Ridge nested-OOF. Öğleden sonra: char n-gram ablation (çıkar+belgele), `nlp_ablation.csv`, NLP'li LGBM ile 1-2 submission + gap ölçümü. Hedef akşam: CV ~89.86 → ~83 teyit. (BERT yalnızca Gün 5 2. submission değerlendirmesine bırakılır.)

## Çapraz Referanslar
- **Faz 02 (validation):** `data/folds.parquet` ve `cvProtocol` nested-OOF sözleşmesi — bu fazın tüm OOF'u oraya bağlı.
- **Faz 03 (preprocessing):** Türkçe-duyarlı lowercase ve encoding kuralları paylaşılır.
- **Faz 04 (feature engineering):** `txt_ridge_pred` + Katman B kolonları feature matrisine concat edilir; adversarial AUC ~0.5 birlikte doğrulanır.
- **Faz 06 (modeling/ensembling):** base learner'lar (LGBM/CatBoost/HistGBR-full) bu kolonları tüketir; BERT 2. submission çeşitliliği.
- **Faz 07 (evaluation/submission):** `nlp_ablation.csv` sunumun "en güçlü 30 saniyesi" (NLP'siz vs NLP'li); CV-LB gap defteri.
```
