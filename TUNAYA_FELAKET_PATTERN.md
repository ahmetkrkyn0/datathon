# Tuna'ya — felaket-satır pattern fikrini test ettim

Önerini ("residual'ı kötü satırları tespit et, aralarında pattern var mı, feature çıkar")
tam olarak uyguladım. Sonuç: **fikrin yarısı çalışıyor, yarısı çalışmıyor — ve neden'i net.**

## DETECTION çalışıyor (AUC 0.70)

Felaket satırları (|res| üst %2.5, 250 satır) out-of-fold **tahmin edilebiliyor:**
- felaket-üyelik classifier AUC = **0.70** (eşik 0.65'in üstünde, gerçek pattern)
- en riskli %5 tahminde precision %8 (baz %2.5, **3.2x lift**)

Bu satırların ortak imzası net:
- **OVER-predict (140 satır):** yeni-yıl (+0.50 std) + düşük real_client_project + düşük technical_interview → model bu zayıf yeni öğrencileri fazla değerliyor
- **UNDER-predict (110 satır):** project_quality −0.95 std (çok düşük!) ama gerçek yüksek + yeni-yıl → "underdog" sürprizleri

## CORRECTION çalışmıyor (DIRECTION AUC 0.43)

İşte tıkanma: riskli sette **yön (over mı under mı) tahmin edilemiyor.**
- riskli set yön-AUC = **0.43** (rastgeleden KÖTÜ)
- riskli set: 529 over / 471 under (dengeli), ort_resid −2.2
- ikisi de "yeni yıl + düşük feature" görünüyor → feature'larda ayrılamıyorlar

Riskli satırları ortalamaya çekme nested testi: en iyi **−0.014** (gürültü), sonra zarar.
Over'ları düzeltirken under'ları bozuyor → net sıfır. Bu, senin "aşağı-gate zarar verir"
gözleminin aynı kökü.

## Mekanizma: aleatoric (indirgenemez) varyans

Üretici, yeni-yıl düşük-feature öğrencilerine feature'larla açıklanamayan rastgele bir
bileşen koymuş. Bazıları beklenmedik başaramıyor, bazıları başarıyor — hangisi olduğu
feature'larda YOK. Blend zaten koşullu ortalamayı veriyor (doğru); sapma gerçek gürültü.
MSE'de optimal tahmin = koşullu ortalama, ki onu zaten veriyoruz.

**Sonuç:** Fikrin detection'da değerli (pattern gerçek) ama correction'a dönüşmüyor
(yön indirgenemez). Bu, ikimizin 18+ kaldıraç testinin hepsiyle tutarlı — feature uzayı doymuş.

## Bu arada: gate-zinciri geliştirmesi (asıl kullanılabilir kazanç)

Senin xlmr-gating'ine **txt_ridge gate'i de ekledim** (önce txt_ridge, sonra xlmr, nested zincir):

```
12-model blend          nested rw (grad) = 82.2398
+ xlmr-gate (senin)                      = 82.1955
+ txt_ridge -> xlmr zinciri              = 82.0912   (-0.149 toplam)
```

İki proxy'de de tutarlı (app-proxy: 82.49 -> 82.35). Greedy zincir 3. sinyalde durdu
(overfit yok). combinatorial pipeline'ına txt_ridge gate'i eklersen final ~0.03-0.05 iyileşir.
İstersen gate-zinciri kodunu/OOF'unu yollarım.

— Ahmet
