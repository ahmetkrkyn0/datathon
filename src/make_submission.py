"""
Faz 7 — Submission yazici (SPEC 07 §4). Bir base/ensemble model'in test tahminini
yarisma formatinda guvenli submission CSV'sine cevirir + submissions_log defterine isler.

    python src/make_submission.py            # varsayilan MODEL=lgbm_full
    python src/make_submission.py catboost_full

SOZLESMELER:
  * KAYNAK = artifacts/test_{MODEL}.npy (Faz 06 fold-bagged test tahmini).
  * SIRA/KUME REFERANSI = test_x.csv student_id (STU_010001..020000) — KANONIK 10000-ID seti
    (SPEC 07 §4.1). sample_submission.csv yalniz 2-satirlik FORMAT ornegidir (123.94 = ornek deger,
    gercek hedef [0,100]); kolon ADLARI oradan dogrulanir, sira/kume ise test_x'ten alinir.
  * HIZALAMA = ID ile MERGE (pozisyonel varsayim YOK): test_pred student_id'ye baglanir, sonra
    kanonik test_x sirasina join edilir -> yanlis sira = sessiz buyuk MSE engellenir (SPEC 07 §6 FORMAT/ID LEAK).
  * clip[0,100] TEK kaynak (cv.clip_predictions); assert_in_range ile clip-disi deger -> hata.

DEFTER (reports/submissions_log.csv): bu diagnostic submission satiri (public_lb/gap bos; secildi=False).
  recency_weighted_oof_mse co-headline olarak loglanir (private-durust tahmin, review H1).
"""

from __future__ import annotations

import datetime as _dt
import sys

import numpy as np
import pandas as pd

import cv

SUBMISSIONS_DIR = cv.ROOT / "submissions"
SAMPLE_SUB_PATH = cv.DATA_DIR / "sample_submission.csv"
SUBMISSIONS_LOG_PATH = cv.REPORTS_DIR / "submissions_log.csv"

# submissions_log.csv sema (mevcut basliga recency_weighted_oof_mse eklenir; idempotent upsert).
LOG_COLS = [
    "tarih", "model_aciklama", "commit_hash", "cv_mse_mean", "cv_mse_std",
    "recency_weighted_oof_mse", "public_lb_mse", "gap", "esik_durumu",
    "test_uretim_yolu", "secildi",
]


def build_submission(model: str) -> pd.DataFrame:
    """test_{model}.npy -> clip[0,100] -> test_x student_id sirasina ID-merge ile hizali submission."""
    test_df = cv.load_test()  # data/test_x.csv (utf-8-sig)
    ref_ids = test_df[cv.ID_COL].to_numpy()  # KANONIK sira + kume

    pred = cv.clip_predictions(np.load(cv.ARTIFACTS_DIR / f"test_{model}.npy"))
    assert len(pred) == len(ref_ids), f"test_{model}.npy {len(pred)} != test_x {len(ref_ids)}"
    cv.assert_in_range(pred, f"test_{model}")

    # ID ile MERGE (pozisyonel varsayim yok): pred student_id'ye bagli, sonra kanonik siraya join.
    pred_df = pd.DataFrame({cv.ID_COL: ref_ids, cv.TARGET_COL: pred})
    ref = pd.DataFrame({cv.ID_COL: ref_ids})
    sub = ref.merge(pred_df, on=cv.ID_COL, how="left", validate="one_to_one")

    # --- FORMAT ASSERT BLOGU (SPEC 07 §4.2) ---
    assert len(sub) == 10000, f"submission {len(sub)} satir != 10000."
    assert list(sub.columns) == [cv.ID_COL, cv.TARGET_COL], f"kolon adlari yanlis: {list(sub.columns)}"
    assert sub[cv.ID_COL].is_unique, "student_id tekrar iceriyor."
    assert sub[cv.TARGET_COL].notna().all(), "career_success_score icinde NaN/eslesmeyen ID var."
    assert set(sub[cv.ID_COL]) == set(ref_ids), "student_id kumesi test_x ile birebir degil."
    assert (sub[cv.ID_COL].to_numpy() == ref_ids).all(), "sira test_x kanonik sirasiyla ayni degil."
    cv.assert_in_range(sub[cv.TARGET_COL].to_numpy(), "submission")

    # sample_submission yalniz FORMAT referansi (2-satir stub): kolon adlari + ID'leri alt-kume mi?
    sample = pd.read_csv(SAMPLE_SUB_PATH, encoding="utf-8-sig").dropna(how="all")
    assert list(sample.columns) == [cv.ID_COL, cv.TARGET_COL], (
        f"sample_submission kolonlari {list(sample.columns)} beklenenle uyusmuyor."
    )
    assert set(sample[cv.ID_COL]).issubset(set(sub[cv.ID_COL])), (
        "sample_submission ID'leri submission'da yok (format stub uyumsuz)."
    )
    return sub


def _recency_weighted_oof(model: str) -> float:
    """oof_{model}.npy + recency_weights -> private-durust co-headline (review H1)."""
    train = cv.load_train()
    test = cv.load_test()
    oof = np.load(cv.ARTIFACTS_DIR / f"oof_{model}.npy")
    rw = cv.recency_weights(train, test)
    return cv.compute_recency_weighted_mse(oof, train[cv.TARGET_COL].values, rw)


def _cv_scores(model: str) -> tuple[float, float]:
    """artifacts/cv_scores.csv'den (cv_mse_mean, cv_mse_std)."""
    df = pd.read_csv(cv.ARTIFACTS_DIR / "cv_scores.csv")
    row = df[df["model"] == model]
    assert len(row) == 1, f"cv_scores.csv'de {model} satiri {len(row)} adet (1 bekleniyor)."
    return float(row["cv_mse_mean"].iloc[0]), float(row["cv_mse_std"].iloc[0])


def log_submission(model: str, recency_mse: float) -> None:
    """submissions_log.csv'ye diagnostic satiri (idempotent upsert; public_lb/gap bos, secildi=False)."""
    cv_mean, cv_std = _cv_scores(model)
    row = {
        "tarih": _dt.date.today().isoformat(),
        "model_aciklama": model,
        "commit_hash": "",
        "cv_mse_mean": round(cv_mean, 6),
        "cv_mse_std": round(cv_std, 6),
        "recency_weighted_oof_mse": round(recency_mse, 6),
        "public_lb_mse": "",
        "gap": "",
        "esik_durumu": "",
        "test_uretim_yolu": "fold-bagging (15 model)",
        "secildi": False,
    }
    new = pd.DataFrame([row], columns=LOG_COLS)
    if SUBMISSIONS_LOG_PATH.exists():
        old = pd.read_csv(SUBMISSIONS_LOG_PATH)
        # mevcut (eski/dar) basligi yeni semaya tasi; ayni model satirini upsert et.
        old = old.reindex(columns=LOG_COLS)
        if "model_aciklama" in old:
            old = old[old["model_aciklama"] != model]
        df = new if old.dropna(how="all").empty else pd.concat([old, new], ignore_index=True)
    else:
        df = new
    SUBMISSIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUBMISSIONS_LOG_PATH, index=False, encoding="utf-8")


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "lgbm_full"

    sub = build_submission(model)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / f"{model}.csv"
    sub.to_csv(out_path, index=False, encoding="utf-8")

    recency_mse = _recency_weighted_oof(model)
    log_submission(model, recency_mse)

    p = sub[cv.TARGET_COL].to_numpy()
    print(f"[submission] {out_path.relative_to(cv.ROOT)} yazildi: {len(sub)} satir, "
          f"ID {sub[cv.ID_COL].iloc[0]}..{sub[cv.ID_COL].iloc[-1]} (test_x kanonik sira).")
    print(f"[submission] pred: mean={p.mean():.3f} std={p.std():.3f} min={p.min():.3f} "
          f"max={p.max():.3f}  (NaN yok, hepsi [0,100]).")
    print(f"[submission] defter: reports/submissions_log.csv'ye {model} satiri "
          f"(recency_weighted_oof_mse={recency_mse:.4f}, public_lb/gap bos, secildi=False).")
    print("[submission] FORMAT ASSERT'leri GECTI -> Kaggle'a upload'a hazir.")


if __name__ == "__main__":
    main()
