"""
Faz 3 — Tahmin-sonrasi sozlesme (SPEC 03 §6). clip[0,100] + submission koruyucu.

TEK CLIP KAYNAGI cv.clip_predictions'tir (cv.py Guardrail 9: "hem OOF hem test ayni
fonksiyondan gecer"). Bu modul SPEC 03 deliverable adini (postprocess.clip_predictions)
saglayan ince bir sarmaldir -> mantik tek yerde, ikilenme yok. Faz 07 submission yazici
guard_predictions'u kullanir.

Hedef kesin [0,100] (min=0.00, max=100.00). Log/logit donusumu YOK (skew ~-0.45, neredeyse
normal; cift-sinirli yiginla logit kotu calisir). sample_submission'daki 123.94 yalniz
format ornegidir, hedef siniri DEGIL.
"""

from __future__ import annotations

import numpy as np

import cv

# SPEC 03 deliverable adlari -> cv'deki tek kaynaga delege (ikilenmis logic yok).
clip_predictions = cv.clip_predictions
assert_in_range = cv.assert_in_range


def guard_predictions(pred, name: str = "pred") -> np.ndarray:
    """Submission koruyucu: ham tahminde clip-disi deger varsa uyar, clip uygula, sonra assert.

    Akis (SPEC 03 §6):
      1. Ham tahminde [0,100] disi/NaN deger -> uyari (yazici farkinda olsun).
      2. clip_predictions ile [0,100]'e sinirla.
      3. clip sonrasi assert_in_range -> clip unutulduysa/bozulduysa HATA firlatir.
    """
    arr = np.asarray(pred, dtype=float)
    n_oob = int((~np.isfinite(arr)).sum() + (arr < cv.CLIP_LO).sum() + (arr > cv.CLIP_HI).sum())
    if n_oob:
        print(f"[postprocess][uyari] {name}: {n_oob} ham deger [0,100] disinda/NaN -> clip uygulanacak.")
    clipped = clip_predictions(arr)
    assert_in_range(clipped, name)  # clip sonrasi [0,100] teyit (Guardrail 9)
    return clipped
