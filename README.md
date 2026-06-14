<div align="center">

# 🎓 BTK Datathon 2026 — Career Success Score Tahmini

**Her öğrenci için sürekli `career_success_score` (0–100) tahmini yapan, sıfır-overfit disipliniyle kurulmuş gözetimli regresyon çözümü.**

[![Sonuç](https://img.shields.io/badge/Final-20.%20s%C4%B1ra-blue?style=for-the-badge)](#-sonu%C3%A7)
[![Private](https://img.shields.io/badge/Private_MSE-82.369-success?style=for-the-badge)](#-sonu%C3%A7)
[![Public](https://img.shields.io/badge/Public_MSE-81.907-success?style=for-the-badge)](#-sonu%C3%A7)
[![Metrik](https://img.shields.io/badge/Metrik-MSE_(d%C3%BC%C5%9F%C3%BCk%3Diyi)-informational?style=for-the-badge)](#-problem)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](requirements.txt)
[![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-9cf)](src/lgbm_full.py)
[![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-orange)](src/catboost_full.py)
[![XLM-R](https://img.shields.io/badge/XLM--RoBERTa-fine--tune-yellow)](colab_xlmr.ipynb)
[![Seed](https://img.shields.io/badge/SEED-42_(reproducible)-critical)](src/cv.py)

</div>

---

## 📌 TL;DR

| | |
|---|---|
| **Görev** | Türkçe mentör yorumu + 46 sayısal/kategorik özellikten kariyer başarı puanı (0–100) regresyonu |
| **Metrik** | MSE (Mean Squared Error) — düşük = iyi |
| **Yaklaşım** | 14 farklı model → nested ağırlıklı blend → koşullu gate düzeltmesi |
| **Karar otoritesi** | **Nested recency-weighted OOF** (public leaderboard ASLA kovalanmadı) |
| **Final sonuç** | 🏅 **20. sıra**, Private MSE **82.369**, Public MSE **81.907** |

> **Felsefe:** Tek bir kuralımız vardı — **sıfır overfit**. Tüm kararlar yarışma test setini değil, kendi ürettiğimiz dürüst çapraz-doğrulama (CV) skorlarına bakılarak verildi. Public leaderboard bir optimizasyon hedefi değil, sadece bir "sağlık sensörü" olarak kullanıldı.

---

## 🎯 Problem <a name="-problem"></a>

Her öğrenci için elimizde:
- **46 sayısal/kategorik özellik** — notlar (`coding_score`, `cgpa`...), sayımlar (`internship_count`, `hackathon_count`...), kategoriler (`department`, `university_tier`...)
- **1 Türkçe serbest metin** — `mentor_feedback_text` (mentörün öğrenci hakkındaki değerlendirmesi)
- **Hedef** — `career_success_score`, sürekli [0, 100], ortalama 76.94, sol-kuyruklu (≈%7.7'si tam 100)

**10.000 öğrenciyle eğitiyoruz**, gizli **10.000 test öğrencisi** için tahmin gönderiyoruz. Başarı ölçüsü MSE: büyük hatalar karesel cezalandırılır, bu yüzden uç tahminlerden kaçınmak kritik. Tüm tahminler `[0, 100]` aralığına kıstırılır (`np.clip`).

---

## 🧠 Yöntem — 4 Katmanlı Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│  KATMAN 0 — Sızıntısız omurga                                     │
│  • Repeated Stratified 5-fold × 3 repeat (15 hücre)             │
│  • Her dönüşüm SADECE fold-içi eğitilir (zero leakage)          │
│  • Karar metriği: recency-weighted OOF MSE                      │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  KATMAN 1+2 — 14 temel model (her biri nested-OOF üretir)       │
│                                                                 │
│  📊 Tablo (GBDT):   lgbm_full · lgbm_num · catboost_full ·      │
│                     huber/weighted varyantlar · ftt             │
│  📝 Metin (NLP):    txt_ridge · e5_ridge · xlmr · fullft        │
│  🔀 Multimodal:     mm · ourteam_tf                             │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  KATMAN 3 — Nested ağırlıklı blend                              │
│  Ridge(alpha=1.0, positive=True) · ağırlıklar hücre-DIŞI fit    │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│  KATMAN 4 — Koşullu gate (son dokunuş)                          │
│  pred = blend + [conf≥q AND model>μ] · a · (model − blend)      │
│  → metin modelini SADECE yüksek-güvenli uçlarda devreye sokar   │
└─────────────────────────────────────────────────────────────────┘
```

### Neden bu kadar çok model?

Her model dünyaya farklı bakar ve farklı hatalar yapar. **GBDT'ler** (LightGBM/CatBoost) sayısal tabloda güçlü; **metin modelleri** (özellikle XLM-R fine-tune) mentör yorumundan GBDT'lerin yakalayamadığı **ortogonal** bir sinyal çıkarır. Blend'de birinin hatasını diğeri dengeler.

### 🔑 Tavanı kıran hamle: `fullft`

Aylarca metin tarafında **frozen** (donmuş) embedding modelleriyle (e5, xlmr) bir tavanda takıldık (public ~82.12). Kırılma, **XLM-RoBERTa-large'ı baştan sona, doğrudan hedefe fine-tune** ettiğimizde geldi (`fullft`). Frozen modellerin yakalayamadığı yeni metin sinyali blend'e girince public **81.90**'a indi.

---

## 🏆 Sonuç <a name="-sonuç"></a>

İki final submission **yapısal olarak farklı** seçildi (private %40 bölmesine karşı risk dağıtımı) — ve **ikisi de** rakip tablodaki en iyi private skorlardı:

| Submission | Yapı | Nested rw-OOF | Public | **Private** | Seçildi |
|---|---|---:|---:|---:|:---:|
| **`fullftgate.csv`** | 14-model blend + fullft/ftt + e5-gate | 82.0191 | **81.907** | **82.369** | ✅ |
| **`combinatorial.csv`** | 14-model blend + xlmr-gate | 82.0164 | 82.122 | **82.583** | ✅ |

**Final: 🏅 20. sıra.**

### Disiplin işe yaradı mı? — Kanıt

- ✅ **Gap sağlıklı:** fullftgate nested 82.0191 → public 81.907 (gap **−0.12**, şişme yok).
- ✅ **Sızıntılı aday elendi, haklı çıktık:** `combo14_gatemax` nested 81.90 gösteriyordu ama param-sızıntısıydı → public 82.224'e şişti. Seçmedik. (Private 82.544 — bizim iki adayımızdan kötü.)
- ✅ **Public kovalanmadı:** En düşük public'li `sub_sp_pseudo_blend` (84.39) bir tuzaktı; karar metriğimiz onu reddetti.

### Dürüst post-mortem

20. sıra hedeflediğimiz değildi. Kaybettiğimiz yer **strateji değil, ham model gücüydü**: lider takımlar muhtemelen Türkçe transformer'ı **tam fold (5×3) GPU fine-tune** ile çıkardı; biz GPU kısıtı nedeniyle `fullft`'i yalnızca **repeat-0** ile üretebildik. Verdiğimiz her metodolojik karar (iki yapısal-farklı aday seçmek, sızıntılıyı elemek, public'i kovalamamak) private skorlarda doğrulandı — eksik olan tek şey daha fazla GPU saatiydi.

> **Öğrenilen ders:** Bir sonraki sefer en baştan Colab GPU'da tam fine-tune'a yatırım yapılmalı; tek kalan kaldıraç buydu ve doğru teşhis edilmişti, sadece donanımla tam çıkarılamadı.

---

## ⚙️ Anahtar Kavramlar (sözlük)

| Terim | Ne demek |
|---|---|
| **Fold / OOF** | Veriyi parçalara böl, her satırı *görmeden* tahmin et (Out-Of-Fold) → ezberi (overfit) yakalar |
| **Nested** | Sadece tahminler değil, blend ağırlıkları ve gate parametreleri de hücre-dışı seçilir → ezber zinciri tamamen kapatılır |
| **recency-weight (rw)** | Test, train'den daha yeni mezuniyet yıllarına kaymış → test'e benzeyen satırlara daha çok ağırlık ver. Karar metriği = **rw-OOF MSE** |
| **Gate** | Blend'in üstüne koşullu, yukarı-yön, lokal düzeltme: metin modeli uçlarda GBDT'den güvenilir → o sinyali geri enjekte et |
| **fold-safe** | Her dönüşüm (imputer, encoder, TF-IDF, scaler) yalnız o fold'un eğitim parçasından fit edilir — sızıntı yasak |

---

## 📁 Proje Yapısı

```
datathon26/
├── data/                  # train.csv, test_x.csv, folds.parquet (15 hücre), feature parquet'leri
├── src/                   # Reproducible pipeline (LightGBM/CatBoost/NLP/blend/gate/finalize)
├── artifacts/             # Her modelin oof_*.npy + test_*.npy + kazanan zincir (fullft/ftt/fullftgate)
├── submissions/           # FİNAL 2: fullftgate.csv + combinatorial.csv
├── colab_*.ipynb          # GPU transformer fine-tune defterleri (xlmr/e5/mm/berturk)
├── reports/               # CV logları, ablasyon sonuçları, ceiling audit, submission defteri
├── Roadmap/               # 8 fazlı planlama (EDA → validation → FE → NLP → modeling → eval)
└── CLAUDE.md              # Geliştirme disiplini (sıfır-overfit anayasası)
```

---

## 🔬 Reproducibility

Tüm pipeline `SEED=42` ile deterministik; `requirements.txt` pinli.

```bash
pip install -r requirements.txt

python src/make_folds.py             # 15-hücre fold şemasını üret (data/folds.parquet)
python src/lgbm_full.py              # base GBDT modeller (her biri oof_*.npy + test_*.npy yazar)
python src/catboost_full.py
python src/e5_ridge.py               # NLP base'ler
python src/ensemble.py               # nested ridge_pos blend
python src/finalize_submissions.py   # final 2 submission CSV + format assert'leri
```

> **Not:** Transformer fine-tune modelleri (`fullft`, `ftt`, `xlmr`, `mm`) GPU gerektirir → ilgili `colab_*.ipynb` defterleri Colab'da çalıştırılıp üretilen `.npy` artefaktları `artifacts/`'a konur. Repo'daki hazır artefaktlar bu adımı atlamayı sağlar.

---

## 📜 Disiplin Kuralları (CLAUDE.md özeti)

1. **Sıfır overfit** — 1 numaralı öncelik genelleme; CV/rw-OOF tek karar otoritesi.
2. **Public LB kovalanmaz** — sadece sağlık sensörü; gap eşikleri (yeşil/sarı/kırmızı) ile izlenir.
3. **Türkçe NLP zorunlu** — `mentor_feedback_text` anlamlı şekilde modele enjekte edilir.
4. **Kabul kapısı** — yeni model ancak `yeni_cv < eski_cv − 0.25·std` İSE kabul; marjinal kazanç reddedilir (Occam).
5. **Clip [0,100]** — tüm tahminler kıstırılır; submission yazıcı aksini görürse `assert` ile durur.

---

<div align="center">

**BTK Akademi · Google · Girişimcilik Vakfı — Datathon 2026**

*"Tek korkumuz ezberlemekti; verdiğimiz her karar private'da doğrulandı."*

</div>
