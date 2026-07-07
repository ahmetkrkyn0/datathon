"""
TIER-3 (FARKLI FONKSIYON SINIFI) — mm: XLM-R-large + tabular NN multimodal entegrasyonu.
========================================================================================

    python src/mm_blend.py        # Colab artefaktlarini DOGRULA + standalone rapor + ledger

NE: `colab_mm_multimodal.ipynb` (GPU) bizim folds.parquet repeat-0 ile uretip indirilen
  `artifacts/oof_mm.npy` + `artifacts/test_mm.npy`'i ALIR (uretmez — neural egitim Colab'da),
  satir-hizasi/[0,100]/NaN denetimlerinden gecirir, standalone unweighted + recency-weighted
  OOF'u hesaplar ve reports/model_scores.csv'ye satir yazar. e5_ridge.py'nin RAPORLAMA yarisinin
  birebir analogu (uretim yarisi Colab'da; bkz. notebook).

NEDEN AYRI SINIF: GBDT/linear (lgbm/catboost/histgbr/ridge/e5_ridge) ailesi forensik 'noise-floor'
  dedi (GBDT-vs-GBDT artik uyumu). Neural multimodal ORTOGONAL fonksiyon sinifi -> tek mesru atis.
  Blend faydasi src/mm_gate.py'de PAIRED-TEST (e5 ile ayni olcut) + kullanici onayi ile karar verilir.

FOLD-SAFE (notebook'ta garanti, burada DENETLENIR):
  * NN her dis-fold'un SADECE fold-train'inde egitildi (folds.parquet repeat-0).
  * tabular impute (fold-train medyani) + StandardScaler PER-FOLD fold-train'e fit.
  * hedef z-score yalniz fold-train y'sinden. HEDEF-ENCODING YOK.
  * oof_mm[i] = i'nin val oldugu fold'un tahmini (her satir tam 1x val -> tam OOF).
  * test_mm = 5 fold modelinin test ortalamasi. clip[0,100] iki tarafa.

REPRODUCIBILITY: neural/GPU bit-deterministik DEGIL (cuDNN/atomik/bf16). oof_mm/test_mm .npy
  KANONIK artefakt; bu runner torch'a IHTIYAC DUYMAZ. SUB-2'ye girerse repro 'belgelenmis tolerans'.

KARAR METRIGI: standalone rw-OOF (rapor) + mm_gate.py NESTED rw-OOF paired-test. Public YOK.
"""

from __future__ import annotations

import numpy as np

import artifacts_io as aio
import cv

MODEL = "mm"
OOF_PATH = cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy"
TEST_PATH = cv.ARTIFACTS_DIR / f"test_{MODEL}.npy"


def _validate_artifacts(oof: np.ndarray, test_pred: np.ndarray, n_train: int, n_test: int) -> None:
    """Colab artefaktlarinin sozlesme denetimi (satir-hizasi/sekil/sonlu/[0,100])."""
    assert oof.shape == (n_train,), f"oof_mm sekli {oof.shape} != ({n_train},) — satir-hizasi bozuk."
    assert test_pred.shape == (n_test,), f"test_mm sekli {test_pred.shape} != ({n_test},)."
    assert np.isfinite(oof).all(), "oof_mm: NaN/Inf var."
    assert np.isfinite(test_pred).all(), "test_mm: NaN/Inf var."
    # clip[0,100] notebook'ta yapildi; burada TOLERANSLA dogrula (kanonik artefakt clip'li olmali).
    assert oof.min() >= -1e-9 and oof.max() <= 100 + 1e-9, f"oof_mm clip-disi ({oof.min():.3f}..{oof.max():.3f})."
    assert test_pred.min() >= -1e-9 and test_pred.max() <= 100 + 1e-9, "test_mm clip-disi."


def main() -> None:
    cv.set_seed()

    if not (OOF_PATH.exists() and TEST_PATH.exists()):
        raise SystemExit(
            f"[mm_blend] HATA: {OOF_PATH.name}/{TEST_PATH.name} yok.\n"
            f"  Once GPU'da uret: colab_mm_multimodal.ipynb (bizim folds.parquet repeat-0) ->\n"
            f"  indirilen oof_mm.npy + test_mm.npy'i artifacts/ icine koy, sonra bu script'i calistir."
        )

    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    # clip TEK kaynaktan (idempotent; kanonik artefakt zaten clip'li, yine de garanti).
    oof = cv.clip_predictions(np.load(OOF_PATH))
    test_pred = cv.clip_predictions(np.load(TEST_PATH))
    _validate_artifacts(oof, test_pred, len(train), len(test))
    cv.assert_in_range(oof, f"oof_{MODEL}")
    cv.assert_in_range(test_pred, f"test_{MODEL}")

    # --- standalone metrikler ---
    # NOT: oof_mm repeat-0 OOF (her satir 1x val). compute_cv_mse 3 repeat uzerinden ayni vektoru
    # 3 farkli partisyonla bucketler -> mean/std DIAGNOSTIK (15-hucre seviye-varyansi). KARAR METRIGI
    # rw-OOF (repeat-agnostik agirlikli MSE) -> repeat-0 OOF ile DOGRU.
    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    unw = float(np.mean((y - oof) ** 2))
    rw = cv.compute_recency_weighted_mse(oof, y, w)

    # mevcut en iyi tek-model (catboost_full) + e5_ridge metin kanali ile karsilastir (cesitlilik).
    refs = {}
    for m in ("catboost_full", "lgbm_full", "e5_ridge", "txt_ridge"):
        p = cv.ARTIFACTS_DIR / f"oof_{m}.npy"
        if p.exists():
            o = np.load(p)
            refs[m] = (cv.compute_recency_weighted_mse(o, y, w), float(np.corrcoef(oof, o)[0, 1]))

    note = (
        f"{MODEL} = XLM-R-large + tabular NN MULTIMODAL (FARKLI fonksiyon sinifi; GPU/Colab "
        f"colab_mm_multimodal.ipynb, bizim folds.parquet repeat-0 5-fit). standalone unw-OOF={unw:.4f} "
        f"rw-OOF={rw:.4f}. Fold-safe (NN fold-train; tabular per-fold impute+scale; hedef-encoding YOK). "
        f"Blend faydasi src/mm_gate.py PAIRED-TEST (e5 olcut) + onay ile karar. Repro: belgelenmis "
        f"tolerans (neural/GPU bit-deterministik degil)."
    )

    aio.save_oof_test(MODEL, oof, test_pred)  # idempotent re-write (clip garanti; kanonik)
    aio.write_cv_score(MODEL, cv_mean, cv_std, 0.0)
    aio.write_cv_log(MODEL, cv_mean, cv_std, fold_mse, [None] * len(fold_mse), 0.0,
                     genuine_fold_mse=None, single5fold_std=None, note=note)
    aio.log_model_score(MODEL, cv_mean, cv_std, rw, weighted_training=False, note=note)

    # DoD-4: reload -> ayni mean (clip idempotent)
    reloaded = np.load(OOF_PATH)
    re_mean, _, _ = cv.compute_cv_mse(cv.clip_predictions(reloaded), y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6, "DoD-4 KIRIK (reload mean farkli)."

    print(f"[mm_blend] MM standalone: unweighted-OOF={unw:.4f}  rw-OOF={rw:.4f}  "
          f"(diag cv_mean={cv_mean:.4f}+/-{cv_std:.4f})")
    for m, (rwm, c) in refs.items():
        tag = "DAHA IYI" if rw < rwm else "daha kotu"
        print(f"           vs {m:14s} rw-OOF={rwm:8.4f} ({tag})  corr={c:+.3f}")
    print(f"[mm_blend] test mean={test_pred.mean():.3f} std={test_pred.std():.3f} "
          f"min={test_pred.min():.3f} max={test_pred.max():.3f}")
    print(f"[mm_blend] YAZILDI: oof_{MODEL}.npy denetlendi + model_scores.csv satiri.")
    print(f"[mm_blend] SONRAKI: python src/mm_gate.py  (paired-test gated blend karari)")


if __name__ == "__main__":
    main()
