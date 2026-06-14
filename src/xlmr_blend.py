"""TIER-2 BIRLESME — xlmr_blend: Colab oof_xlmr/test_xlmr artefaktini DOGRULAR + standalone rw +
10-model blend'e (84.0212) BLEND ETKISINI olcer (asil karar metrigi). Sadece okur, defterler.
=================================================================================================

    python src/xlmr_blend.py

ON KOSUL: notebooks/colab_xlmr.ipynb Colab'da kosup artifacts/oof_xlmr.npy + test_xlmr.npy uretmeli.

KARAR: standalone rw DEGIL -> BLEND etkisi (nested rw-OOF, ensemble.py mekanigi). Tier-1 tum
  CPU teknikleri corr ~0.99 redundant cikti; mm zaten XLM-R tabanli -> txt_xlmr de yuksek-corr
  beklenir. Gate net karar verir. Determinizm: mm gibi belgelenmis tolerans (neural/GPU)."""

from __future__ import annotations

import numpy as np

import artifacts_io as aio
import cv
from ensemble import nested_rw_oof

MODEL = "xlmr"
OOF_PATH = cv.ARTIFACTS_DIR / "oof_xlmr.npy"
TEST_PATH = cv.ARTIFACTS_DIR / "test_xlmr.npy"

BASE10 = [
    "lgbm_full", "lgbm_num", "lgbm_full_w", "catboost_full", "catboost_full_w",
    "txt_ridge", "e5_ridge", "mm", "lgbm_full_h", "lgbm_full_ht",
]


def main() -> None:
    cv.set_seed()
    if not (OOF_PATH.exists() and TEST_PATH.exists()):
        raise SystemExit(
            f"[xlmr] HATA: {OOF_PATH.name}/{TEST_PATH.name} yok. Once Colab'da colab_xlmr.ipynb "
            f"kosup inen 2 .npy'yi artifacts/ icine koy."
        )

    train, test, folds = cv.load_train(), cv.load_test(), cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    oof = np.load(OOF_PATH).astype(float)
    test_pred = np.load(TEST_PATH).astype(float)
    assert len(oof) == len(train) and len(test_pred) == len(test), "xlmr artefakt boyutu uyumsuz."
    cv.assert_in_range(oof, "oof_xlmr")
    cv.assert_in_range(test_pred, "test_xlmr")

    cv_mean, cv_std, _ = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.compute_recency_weighted_mse(oof, y, w)

    # ortogonallik: mevcut metin/mm kanallariyla korelasyon
    others = {m: np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy") for m in ("mm", "e5_ridge", "txt_ridge")}
    corrs = {m: float(np.corrcoef(oof, v)[0, 1]) for m, v in others.items()}
    print(f"[xlmr] standalone rw-OOF={rw:.4f}  unw_cv={cv_mean:.4f} +/- {cv_std:.4f}")
    print("[xlmr] korelasyon: " + "  ".join(f"{m}={c:.3f}" for m, c in corrs.items())
          + f"  (mm-corr dususe -> ortogonal umut; >0.97 redundant riski)")

    # === ASIL KARAR: 10-model blend'e ekleyip nested rw-OOF etkisi ===
    oof_base = {m: np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy") for m in BASE10}
    P10 = np.column_stack([oof_base[m] for m in BASE10])
    rw10, _ = nested_rw_oof(P10, y, w, folds, sid, "ridge_pos")
    P11 = np.column_stack([oof_base[m] for m in BASE10] + [oof])
    rw11, _ = nested_rw_oof(P11, y, w, folds, sid, "ridge_pos")
    delta = rw11 - rw10
    print(f"\n[xlmr] === BLEND ETKISI (asil karar) ===")
    print(f"[xlmr] 10-model blend   nested rw-OOF = {rw10:.4f}  (resmi 84.02)")
    print(f"[xlmr] +xlmr (11-model)  nested rw-OOF = {rw11:.4f}  (delta {delta:+.4f})")
    verdict = "blend'i IYILESTIRIR -> paired-gate'e goz at" if delta < -0.01 else \
              "net katki YOK (gurultu/kotulesme) -> RED beklenir (Tier-1 emsali)"
    print(f"[xlmr] => {verdict}")

    note = (
        f"{MODEL} = XLM-R-large TEXT-ONLY nested-OOF (Colab, repeat-0; src colab_xlmr.ipynb). "
        f"standalone rw-OOF={rw:.4f}; corr(mm)={corrs['mm']:.3f}. BLEND etkisi: 10-model {rw10:.4f} "
        f"-> +xlmr {rw11:.4f} (delta {delta:+.4f}). {'KABUL-aday' if delta < -0.01 else 'RED-aday'} "
        f"(nihai: ensemble.py paired-test). Belgelenmis tolerans (neural/GPU)."
    )
    aio.save_oof_test(MODEL, oof, test_pred)
    aio.log_model_score(MODEL, cv_mean, cv_std, rw, weighted_training=False, note=note)
    print(f"\n[xlmr] defterlendi (model_scores.csv). Delta<0 ise: ensemble.py CANDIDATE_POOL'a 'xlmr' ekle.")


if __name__ == "__main__":
    main()
