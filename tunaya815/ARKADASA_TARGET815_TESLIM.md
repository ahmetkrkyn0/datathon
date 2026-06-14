# TARGET 81.5 - Son Calisma ve Teslim Notu

## Kisa Ozet

Mevcut 12-model fold-hizali blend tabani:

| Aday | Nested rw-OOF MSE | Hucre sonucu | Paired p |
|---|---:|---:|---:|
| Mevcut 12-model base | 82.23976 | referans | - |
| Yeni Ridge meta | 81.78859 | 12/15 | 0.00263 |
| Robust zincir | 81.61241 | 14/15 | 0.0000596 |
| Aggressive zincir | **81.48840** | **15/15** | **0.00000404** |

Bu sonuca public leaderboard kullanmadan, recency-weighted OOF MSE ile ulastik.

## Ne Yaptik?

### 1. Yeni meta-stack

12-model blend'deki fold-hizali tahminlere iki farkli model eklendi:

- `fullft`: XLM-R-large full fine-tune metin modeli
- `ftt`: FT-Transformer tabular modeli

Model tahminlerinin ortalama, standart sapma, minimum, maksimum ve medyan
ozetleriyle birlikte:

```text
StandardScaler + Ridge(alpha=0.01)
```

kullanildi. Meta-model her dis fold icinde yeniden egitildi. Gate parametreleri
icin de dis-fold train bolumunde ayri inner-OOF tahminleri olusturuldu.

Meta-stack sonucu:

```text
rw-OOF: 82.23976 -> 81.78859
hucre: 12/15
p: 0.00263
```

### 2. Robust aday

Meta tahmininden sonra `fullft` modeli sadece yuksek-guvenli ust uclarda
kullanildi:

```text
fullft > merkez
|fullft - merkez| >= nested esik
tahmin = meta + 0.5 * (fullft - meta)
```

Sonuc:

```text
rw-OOF: 81.61241
hucre: 14/15
repeat dagilimi: 4/5, 5/5, 5/5
p: 5.96e-05
```

### 3. Aggressive 81.5 adayi

Asagidaki nested gate zinciri uygulandi:

```text
1. fullft yukari gate
2. lgbm_full asagi gate
3. mmstrong yukari gate
```

Tam OOF ile test tahminine dondurulan frozen parametreler:

| Gate | Yon | Guven esigi | Guc |
|---|---|---:|---:|
| fullft | yukari | 22.5125 | 0.5 |
| lgbm_full | asagi | 15.4849 | 0.4 |
| mmstrong | yukari | 20.6991 | 0.5 |

Sonuc:

```text
rw-OOF: 81.48840
base'e gore kazanc: -0.75137 MSE
hucre: 15/15
repeat dagilimi: 5/5, 5/5, 5/5
p: 4.04e-06
```

## Onemli Metodolojik Uyari

`fullft` ve `mmstrong` OOF tahminleri Tuna fold'larinin yalnizca
`repeat=0` bolumuyle uretilmistir. Her satir kendi modelinin egitiminden disarida
olsa da, repeat 1 ve 2 meta hucreleriyle birebir fold-hizali degildir.

Bu nedenle:

- **81.48840 guclu bir OOF proxy ve submission adayidir.**
- Kesin, tamamen 5x3 fold-safe kanit sayilmasi icin `fullft` ve `mmstrong`
  modellerinin repeat 1 ve repeat 2 icin de uretilmesi veya Tuna'nin kendi
  pipeline'inda yeniden test edilmesi gerekir.
- Public leaderboard sonucu gorulmeden "81.49 garanti" denmemelidir.

## Hangi Submission Gonderilmeli?

Birinci tercih:

```text
submissions/TARGET815_aggressive.csv
```

Daha muhafazakar ikinci tercih:

```text
submissions/TARGET815_robust.csv
```

## Paket Icerigi

```text
ARKADASA_TARGET815_TESLIM.md
submissions/TARGET815_aggressive.csv
submissions/TARGET815_robust.csv
target815_meta_oof.npy
target815_robust_oof.npy
target815_aggressive_oof.npy
src/build_target_815.py
```

Paket gerekli OOF/test girdilerini de icerir ve ana repodan bagimsiz calisir.
`tunaya815` klasor kokunde:

```powershell
python -u src/build_target_815.py
```

Beklenen ana cikti:

```text
meta       rw=81.78859
robust     rw=81.61241
aggressive rw=81.48840
```

Bagimsiz aritmetik dogrulama:

```powershell
python -u verify_target815.py
```

Bu script uretim kodunu kullanmadan kayitli OOF dosyalarindan skor, fold
sonuclari, bootstrap araligi ve submission semasini yeniden hesaplar.

## Dogruluk Denetimi Sonucu

- Dosya sirasi ve train/test ID hizasi dogrulandi.
- Tum OOF dizileri 10.000 satir, sonlu ve `[0,100]` araliginda.
- Aggressive skor bagimsiz hesapta tekrar **81.48840** cikti.
- Aggressive aday repeat-0 kontrolunde 5/5 fold kazandi.
- Satir-bootstrap delta %95 araligi sifirin altinda kaldi.
- Alternatif `application_year` agirlikli proxy skoru **81.76465** cikti.

Dolayisiyla dosya ve hesap dogru; fakat leaderboard icin 81.49 garanti degildir.
Tarihsel public gap dikkate alindiginda gercek public skorun yaklasik 81.6
cevresinde olmasi daha gercekci bir beklentidir.

## Kontroller

Iki submission da su kontrollerden gecti:

- 10.000 satir
- Kolonlar: `student_id`, `career_success_score`
- Test ID sirasi birebir ayni
- Tekrarlanan ID yok
- NaN yok
- Tahminler `[0, 100]` araliginda
