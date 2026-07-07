"""
Faz 5 — NLP: ALAN-BILGISIYLE SABITLENMIS Turkce sozlukler (SPEC 05 §3, Guardrail LEXICON LEAK).
================================================================================================

Bu sozlukler hedefe BAKILMADAN, mentor-degerlendirme alan bilgisiyle elle sabitlenmistir.
Kelime-hedef korelasyonuyla secim DE-FACTO target leakage'tir (SPEC 05 §6: LEXICON LEAK) ->
fold-bagimsiz, sabit liste tek guvenli yontemdir. Substring/kok eslesmesi ('gelistir' in token
tum cekimleri yakalar) text_utils.extract_handcrafted_features icinde uygulanir.

NORMALIZASYON: tum terimler ve metin AYNI text_utils.turkish_lower'dan gecer (Turkce I/i tuzagi,
SPEC 05 §1). Burada terimler ZATEN kucuk-harf Turkce yazimda; eslesme aninda yine normalize edilir.
"""

from __future__ import annotations

# Pozitif sinyal (SPEC 05 §3; parantezdeki satir kapsami EDA'dan, secim icin DEGIL belge icin):
#   etkileyici(16.2%) / guclu(31.0%) / yuksek / basari(25.3%) / mukemmel(4.7%) /
#   olaganustu(1.8%) / ustun(0.7%) / potansiyel(20.4%) / hakimiyet(0.5%) / dikkat cek(31.9%)
POSITIVE = (
    "etkileyici",
    "güçlü",
    "yüksek",
    "başarı",
    "mükemmel",
    "olağanüstü",
    "üstün",
    "potansiyel",
    "hakimiyet",
    "dikkat çek",
)

# Negatif / gelisim-ihtiyaci sinyali (SPEC 05 §3):
#   ancak(58.3%) / gelistir(63.0%) / gerekiyor(3.2%) / eksik(4.5%) / ihtiyac(4.2%) /
#   daha fazla / dusuk(2.7%) / zayif(3.9%).  'ancak' ASLA stopword DEGIL (en guclu kosul sinyali).
NEGATIVE = (
    "ancak",
    "geliştir",
    "gerekiyor",
    "eksik",
    "ihtiyaç",
    "daha fazla",
    "düşük",
    "zayıf",
)

# Teknik beceri anahtarlari (n_skill_mention sayimi; SPEC 05 §3).
SKILL = (
    "sql",
    "backend",
    "frontend",
    "devops",
    "cloud",
    "makine öğren",
    "veri yapı",
    "portföy",
    "github",
)

# TF-IDF stopword'leri: yalnizca icerik-tasimayan fonksiyon kelimeleri (SPEC 05 §2).
# 'ancak' BILEREK DISARIDA (en guclu negatif/kosul sinyali). 'daha' burada elenir ama lexicon
# 'daha fazla' substring eslesmesini metin uzerinden ayrica yakalar (TF-IDF vocab'dan bagimsiz).
STOPWORDS = (
    "ve",
    "ile",
    "bir",
    "bu",
    "için",
    "da",
    "de",
    "daha",
    "olan",
    "gibi",
)
