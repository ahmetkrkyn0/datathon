# Datathon 2026 — Çalışma Günlüğü (Takım Harici Tüm Denemeler)

> Bu dosya, **kendi modellerimizle** (takım arkadaşının `sub2` katkısından önce)
> denediğimiz **tüm** feature, model, teknik ve karışımları kaydeder.
> Amaç: neyi denedik, ne işe yaradı, ne yaramadı — ve **neden**.

---

## 0. Temel Bilgiler

| | |
|---|---|
| **Görev** | `career_success_score` (0–100 sürekli) regresyonu |
| **Metrik** | **MSE** (RMSE değil! — skorların ~80-95 görünmesinin sebebi) |
| **Veri** | train 10.000 satır + test 10.000 satır, 45 kolon (sayısal + kategorik + Türkçe metin) |
| **Submission** | `student_id, career_success_score` |
| **Kota** | günde 5 submission, final 2 çözüm seçilir |
| **Skor hesabı** | %60 public / %40 private |

### En kritik metodolojik karar: Recency-Weighted OOF
Normal 5-fold CV **yanıltıcı** (MSE ~75 gösteriyor ama gerçek LB ~85+).
Sebep: **zaman kayması** — test setinin %62'si 2024-2026 başvurusu, train uniform (~%13/yıl).

**Çözüm:** her train satırına `w = P_test(yıl) / P_train(yıl)` ağırlığı (0.3–2.5 arası kırpılmış).
MSE bu ağırlıkla hesaplanır → **recency-weighted OOF**. Bu metrik gerçek LB'yi ±0.2-0.5 ile tahmin etti (5+ kez doğrulandı). **Tüm kararlar buna göre verildi, public LB'ye fit edilmedi.**

---

## 1. Skor Geçmişi (gönderilen submission'lar)

| Sürüm | RW-OOF Proxy | Gerçek LB | İçerik |
|-------|:---:|:---:|--------|
| v1 | — | ~95 | İlk HGB baseline (metrik karışıklığı dönemi) |
| v2 | 76.8 | **87.12** | İlk ensemble (CatBoost+LGBM+XGB, TF-IDF+SVD) |
| v4b | 86.79 | **86.575** | Optuna-tuned + Ridge-meta blend |
| v7 | 85.79 | **85.231** | 5-model + yıl kalibrasyonu |
| v10 | 85.37 | **85.30** | Süper-blend (ağırlık-fit dersi: kazanç taşınmadı) |
| v20 | 84.91 | **~84.73** | v19+v7 mix |
| v42 | 83.85 | **84.098** | 6'lı mega-karışım |
| v39 | 83.89 | (gönderilmedi) | 5'li mega-karışım |

*(Takım birleşmesi sonrası `team_blend_opt` → LB **82.999**; bu dosya takım-harici kısmı belgeler.)*

---

## 2. FEATURE ENGINEERING — Ne Denedik

### ✅ İŞE YARAYAN feature'lar (modele dahil)

| Feature | Açıklama | Kazanç |
|---------|----------|:---:|
| **Türetilmiş ortalamalar** | `technical_avg`, `social_avg`, `portfolio_avg`, `interview_avg`, tech_max/min/std | baseline |
| **Eksik-değer bayrakları** | 7 kolon için `_na` + `n_missing` (eksiklik bilgi taşıyor: staj süresi boş = staj yok) | orta |
| **total_experience** | staj+gerçek proje+freelance+hackathon toplamı | küçük |
| **role_skill / role_gap** | hedef role uygun teknik skor (Backend→backend_score) + tech_avg'den farkı | küçük |
| **Etkileşim çarpımları** | `pq_x_ti` (proje×mülakat, r=0.583!), `pq_x_tech`, `ti_x_tech` | orta |
| **years_since_grad** | başvuru − mezuniyet yılı | küçük |
| **Keyword bayrakları** | "mükemmel/güçlü/gelişmeye açık" vb. 16 Türkçe anahtar kelime | küçük |
| **🌟 SEGMENT-YIL TE** | (rol/tier/hobby/sosyal × yıl) target encoding, hiyerarşik smoothing sm=20 | **−1.4 tekil!** |
| **🌟 KOHORT-Z skorları** | key5 feature'ın {yıl, rol, rol×yıl} içi z-skoru + role_skill rolyıl-z | **−0.45 tekil** |
| **F3 "potansiyel tuzağı"** | güçlü övgü olmadan geçen "potansiyel" = kibar uyarı (arkadaş fikri) | −0.137 tekil |

#### En değerli iki yapısal keşif (detay):

**Segment-yıl TE** — Üretici her segmente farklı yıl trendi vermiş:
Cybersecurity geç-erken farkı −3.98, Frontend −3.76 vs DevOps −0.86, Cloud −1.66.
Global yıl-düzeltmeleri bunu kaçırıyordu. Fold-içi (segment×yıl) hücre ortalaması →
yıl ortalamasına büzülmüş TE. **Tek-LGBM'de 89.94 → 88.53.**

**Kohort-göreli z** — Öğrencinin ham skoru değil, **kohortu içindeki göreli yeri** önemli.
proje_kalite'nin rol×yıl hücresi içindeki z-skoru gibi. Üretici öğrencileri kohort-içi
yarıştırıyor. Feature-only (hedefsiz) → train+test havuzlu istatistikle sızıntısız.

### ❌ İŞE YARAMAYAN feature'lar (dürüst harness'ta elendi)

| Denenen | Sonuç | Neden |
|---------|-------|-------|
| Sistematik ikili çarpımlar (21 çift) | **+0.59 zarar** | seyreltme; ağaçlar zaten yakalıyor |
| Güçlü/zayıf alan sayaçları (`n_strong`, `n_weak`) | +0.0 | nötr |
| "En güçlü beceri" kategoriği (`best_skill_idx`) | +0.0 | nötr |
| Koşullu cgpa (tier-içi z, cgpa×tier_te) | +0.0 | cgpa global r≈0, koşullu da değersiz |
| Geniş kohort-z (key5 yerine key10) | +0.08 | aşırı genişletme seyreltir |
| Metin skalar kohort-z (bert3/mdeb rolyıl-z) | nötr | |
| Kohort yüzdelik rank (pct) | +0.13 zarar | |
| Top-decile bayrakları | gürültü bandı | |
| Tamsayı parmak izleri (`isint` bayrakları) | nötr | segment +6 bias var AMA ağaçlar emiyor |
| pq nonlineritesi (kare/küp/decile) | −0.04 (gürültü) | |
| Staj tutarsızlığı (süre var, sayı 0) | zarar | |
| Hacim feature'ları (total_stars, görüşme oranı) | residual'da yok | zaten yakalanmış |
| F1 "övgü-ama-gömme" regex | +0.015 (F3'ü bozuyor) | |
| F2 soft/hard kelime oranı | +0.02 | nötr |
| F4 buzzword yoğunluğu | −0.01 | nötr |

### 🔍 Residual taraması bulgusu
En iyi blend'in artıklarını her feature'a karşı bin'ledik. **En büyük bin-bias bile ±1.4**
(pq) — feature uzayı tükenmiş. "Modelin ıskaladığı sistematik şey" kalmadı.

---

## 3. MODELLER — Ne Denedik

### ✅ Kullanılan modeller (final ensemble üyeleri)
| Model | Rol | Not |
|-------|-----|-----|
| **LightGBM** (tuned) | ana güç | yıl-norm + uniform rejimde re-tune edildi |
| **CatBoost** (GPU, tuned) | ana güç | native kategorik + one_hot_max_size=16 |
| **XGBoost** | çeşitlilik | **bilerek UNTUNED** (v8 dersi) |
| **HistGradientBoosting** | erken sürümlerde | sonra elendi (ağırlık 0) |
| **MLP** (sklearn) | nöral çeşitlilik | düşük ağırlık |
| **PyTorch TabNN** | embedding'li tabular NN | solo zayıf (109) ama çeşitlilik |

### Tuning
- `tune_v4.py` — ilk Optuna (CatBoost GPU + LGBM)
- `tune_xgb.py` — XGBoost tuning (sonra blend'de kullanılmadı: v8 dersi)
- `tune_v13.py` — yeni rejim (yıl-norm+uniform) re-tune
- `tune_v30.py` — tam-pipeline harness'ıyla final re-tune
- **Bulgu:** ağır regülarizasyon kazanıyor (LGBM 31-34 yaprak, l2=5-8). Yeni rejimde
  optimum farklı: uniform ağırlıkla etkin temiz veri artıyor → daha az regülarizasyon.

### ❌ İşe yaramayan model/teknik denemeleri
| Denenen | Sonuç | Neden |
|---------|-------|-------|
| **Pseudo-labeling** | dürüst nested protokolde base'den KÖTÜ (89.1 vs 88.6) | naif tarama (84.8) sızıntılıydı; pseudo kendi hatasını pekiştiriyor |
| **Era-uzmanlaşma** (sadece son yıllarla eğitim) | 113 vs 105 | veri azlığı kaybettiriyor |
| **Formül avı** (lineer/polinom) | lineer her yerde kötü | hedef formülsel değil |
| **Two-stage tavan** P(y=100) | tek başına +0.36 ama yıl-norm ile çakışıyor | |
| **15-fold × 3-seed cilası** (v40) | 84.71 > v37 84.60 | fold artırmak kazanç değil |
| **Train-test kopya eşleşmesi** | yok (NN mesafe medyan 4.88 ≈ train-içi) | |
| **Çapraz-yıl öğrenci tekrarı** | yok (stabil-profil NN min 1.09) | |
| **Felaket dedektörü** P(y<50) | AUC 0.917 AMA blend'e katkı YOK | hatanın yönünü değil büyüklüğünü söylüyor; risk zaten fiyatlı |
| **Residual-boosting** (y−sub2'ye eğitim) | 85.07 (overfit) | |
| **Anlaşmazlık-meta** (üye std'si) | 82.98 | belirsizlik MSE'de yön bilgisi vermiyor |
| **El yapımı kesinti filtreleri** (−15/−10 vb.) | +32 zarar! | maskelerin gerçek residual'ı ±0.5, kesintiler 20-30x büyük |

---

## 4. NLP / METİN İŞLEME — Ne Denedik

`mentor_feedback_text` = Türkçe mentor değerlendirmesi. **Metin tavanı ~145 MSE**
(hedef varyansı 230) — yani metin orta-güçlü sinyal, asıl iş sayısalda.

### Metin model bataryası (hepsi OOF, sızıntısız)

| Model | Mimari | Metin-tek MSE | Akıbet |
|-------|--------|:---:|--------|
| TF-IDF + SVD40 | word 1-2gram + char 3-5gram | — | SVD feature olarak |
| TF-IDF + Ridge OOF | `txt_ridge` | 146.8 | feature |
| MiniLM embeddings | paraphrase-multilingual, SVD50 | 148.7 | feature |
| **BERTurk #1** (`bert_text_oof`) | CLS-pool, len128 | 145.0 | `txt_bert` (sonra atıldı) |
| **BERTurk #2** (`bert_emb_oof`) | mean-pool + standardize hedef, len160 | **132.0** | `txt_bert2` ✅ |
| **BERTurk #3** (`bert_foldmatched`) | mean-pool, **10-fold** | **129.6** | `txt_bert3` ✅ (en iyi BERTurk) |
| **mDeBERTa-v3** | `mdeberta_oof` | 130.6 | `txt_mdeb` ✅ |
| **XLM-R-large** (560M) | `xlmr_oof`, 5-fold | **126.1** | `txt_xlmr` ✅ (en iyi metin) |
| **Multimodal v1** | BERTurk+tabular joint, 5-fold | flat 85.8 | `mm` ✅ blend üyesi |
| **Multimodal v2** | +sub2-farkında, 10-fold (Colab/A100) | flat 78.3 | `mm2` (sub2 çift-sayımı→atıldı) |

**Mimari dersi:** mean-pool + hedef standardizasyonu BERTurk'ü 145→129.6'ya indirdi —
"metin tavanı" kısmen mimariydi. Arkadaşın notebook'undaki bu inceliklerden öğrenildi.

### ❌ Metin tarafında elenen yollar
| Denenen | Sonuç | Neden |
|---------|-------|-------|
| **Embedding'leri GBM feature yapmak** (v17) | 85.97 > v15 85.49 | fold'lar arası uzay hizasız |
| **Fold-eşli embedding** (v21) | 87.58 (çöktü) | ezber sızıntısı |
| Metin uzunluğu | r=0.008 | sinyal yok, kelimeler lazım |

**Sonuç:** metin sadece **skalar tahmin** (her modelin tek OOF çıktısı) ve **blend üyesi**
(mm) olarak değerli. Embedding-vektörü doğrudan GBM'e vermek hep bozdu.

---

## 5. ENSEMBLE / BLEND STRATEJİLERİ — Ne Öğrendik

### Kullanılan
- **NNLS** (negatif-olmayan ağırlık) — OOF üzerinde, ağırlıklı uzayda
- **Ridge-meta** — üye OOF'ları + yıl + pq → meta Ridge
- **Yıl-bazlı affine kalibrasyon** — nested-CV doğrulamalı (kazandırırsa uygula)
- **Fit'siz sabit-oran karışım** — farklı rejimlerin eşit ortalaması (EN GÜVENİLİR)

### 🔑 Kritik blend dersleri
1. **Ağırlık-fit kazançları LB'ye TAŞINMIYOR** (v10 dersi): NNLS süper-blend proxy'de
   +0.42 gösterdi, LB'de +0.07 verdi. Bireysel model kazançları taşınıyor, fit kazançları hayır.
2. **Çeşitlilik > bireysel tuning** (v8 dersi): tuned XGB, tuned LGBM'e benzeyince
   ağırlığı 0.31→0.06 düştü, ensemble kötüleşti. XGB'yi UNTUNED tut.
3. **Çift-sayım sınırı** (v43 dersi): bir modeli hem feature olarak İÇERİDE hem
   karışım üyesi olarak DIŞARIDA kullanma — kazanç erir.
4. **Proxy çözünürlük sınırı**: <0.2'lik karışım-içi proxy farkları artık LB'ye
   taşınmıyor (v42'de proxy ilk kez +0.25 iyimser çıktı).
5. **Yıl-norm hedef + UNIFORM ağırlık**: w_fit eğitimde HATA, sadece değerlendirmede
   kullan. Hedefi yıl (mean,std) ile standardize edip öğret → +0.66 tekil.

---

## 6. ANAHTAR DOSYALAR

### Pipeline
- `src/features.py` — ortak feature pipeline (cache'li): 170+ sayısal feature
- `src/explore.py` — derin EDA

### Model script'leri (kronolojik)
- `train_model_v2.py` → ilk ensemble
- `train_model_v7.py` → 5-model + kalibrasyon (LB 85.23)
- `train_model_v12.py` → yıl-norm + uniform keşfi
- `train_model_v15.py` → segment-yıl TE keşfi
- `train_model_v26.py` → kohort-z keşfi
- `train_model_v30.py` → re-tuned final
- `train_model_v37.py` → mm+xlmr feature olarak
- `train_model_v45.py` → +F3 potansiyel tuzağı
- `tune_v30.py` → tam-pipeline re-tune

### Metin/NN
- `bert_foldmatched.py`, `mdeberta_oof.py`, `xlmr_oof.py`, `mm_oof.py`, `nn_oof.py`
- `mm_colab_core.py` + `colab_mm2.ipynb` → Colab/A100 multimodal v2

### Blend/analiz
- `super_blend.py`, `blend_lab.py`, `check_submission.py`

### Cache (yeniden hesaplama gerektirmez)
- `data/cache/preds_v*.npz` — her sürümün OOF + test tahminleri
- `data/cache/*_oof.npy` / `*_test.npy` — metin model skalarları
- `data/cache/best_params*.json` — Optuna sonuçları

---

## 7. ÖZET: Üç Günde Ne Oldu?

**LB yolculuğu:** 95 → 87.1 → 86.6 → 85.2 → 84.7 → **84.10** (takım-harici en iyi: v42)

**En büyük 3 kazanç:**
1. Recency-weighted OOF metodolojisi (doğru pusula) — yoksa hiçbir karar güvenilir değildi
2. Segment-yıl TE (−1.4) + Kohort-z (−0.45) — üreticinin kohort-içi yarıştırma yapısı
3. Metin bataryası (5 model: BERTurk×3, mDeBERTa, XLM-R) + multimodal

**En büyük ders:** Bu veri **sentetik ve kohort-yapılı**. Sinyalin çoğu birkaç
yapısal keşifte; geri kalanı indirgenemez gürültü. Feature/model uzayı tükendiğinde
kazanç **bağımsız pipeline çeşitliliğinden** gelir (takım birleşmesi neden işe yaradı).
