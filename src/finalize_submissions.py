"""
Faz 7-HAFIF — 2 FINAL submission secimi + CSV yazimi.
=====================================================

    python src/finalize_submissions.py

SECIM KARARI = recency_weighted_oof_mse (reports/model_scores.csv; review H1). PUBLIC LB'ye gore
HICBIR secim YOK (public-overfit tuzagi; final private'da).

  * SUB-1 (CAPA/safe)  = EN DUSUK recency-weighted TEK GBDT model (blend HARIC). Sade, tek model.
  * SUB-2 (EN IYI CV)  = recency-weighted ENSEMBLE (blend; NNLS/Ridge-stack). Yapisal farkli
    (tek-model vs ensemble) -> private %40 bolmesine karsi risk dagitimi (CLAUDE.md final politikasi).

CIKTI: submissions/sub1_<model>.csv , submissions/sub2_blend.csv
  (make_submission.build_submission ile: clip[0,100], test_x student_id sirasina ID-MERGE,
   assert 10000 satir + ID birebir + [0,100] + NaN yok). reports/submissions_log.csv secildi=True.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

import cv
import make_submission as ms


# SUB-1 (CAPA/safe) yalniz GERCEK tek guclu GBDT base modelinden secilir. Ledger'da ayrica
# tureyen/elenmis adaylar var (blend, blend_p100 post-process, txt_ridge* metin alt-modelleri,
# *_w recency varyantlari). Bunlar SUB-1 havuzunda OLMAMALI -> acik PREFIX-DISLAMA (robust).
#   - "blend"      : ensemble (SUB-2'nin kendisi)
#   - "blend_p100" : iki-asama post-process (LEVER2, elendi) — tek-model DEGIL
#   - "txt_ridge*" : metin alt-modeli (zayif standalone; sadece blend bileseni)
#   - "*_w"        : recency-weighted egitim varyanti (rw-OOF'u DUSURMEDI, elendi; bilerek SUB-1 disi)
SUB1_EXCLUDE_PREFIXES = ("blend", "txt_ridge")


def _is_sub1_eligible(model: str) -> bool:
    """SUB-1 (sade tek GBDT) adayligi: turetilmis/elenmis satirlari (blend*/txt_ridge*/*_w) ele."""
    if model.startswith(SUB1_EXCLUDE_PREFIXES):
        return False
    if model.endswith("_w"):  # recency-weighted varyant (elendi; SUB-1 sade-unweighted olmali)
        return False
    return True


def _pick_from_ledger():
    df = pd.read_csv(cv.REPORTS_DIR / "model_scores.csv")
    singles = df[df["model"].map(_is_sub1_eligible)].copy()
    assert not singles.empty, "model_scores.csv'de SUB-1 uygun tek-model yok."
    best_single = singles.sort_values("recency_weighted_oof_mse").iloc[0]["model"]
    has_blend = (df["model"] == "blend").any()
    return df, str(best_single), has_blend


def _write_submission(model: str, out_name: str) -> pd.DataFrame:
    sub = ms.build_submission(model)  # robust ID-merge + tum format assert'leri
    ms.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ms.SUBMISSIONS_DIR / out_name
    sub.to_csv(out_path, index=False, encoding="utf-8")
    p = sub[cv.TARGET_COL].to_numpy()
    print(f"[final] {out_path.relative_to(cv.ROOT)}: {len(sub)} satir, "
          f"ID {sub[cv.ID_COL].iloc[0]}..{sub[cv.ID_COL].iloc[-1]}; "
          f"pred mean={p.mean():.3f} std={p.std():.3f} min={p.min():.3f} max={p.max():.3f}.")
    return sub


def _log_selected(rows: list[dict]) -> None:
    path = ms.SUBMISSIONS_LOG_PATH
    new = pd.DataFrame(rows, columns=ms.LOG_COLS)
    if path.exists():
        old = pd.read_csv(path).reindex(columns=ms.LOG_COLS)
        keep_models = {r["model_aciklama"] for r in rows}
        old = old[~old["model_aciklama"].isin(keep_models)]
        df = new if old.dropna(how="all").empty else pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    df, best_single, has_blend = _pick_from_ledger()
    assert has_blend, "blend artefakti yok -> once python src/ensemble.py calistir."

    def _rw(m):
        return float(df[df["model"] == m]["recency_weighted_oof_mse"].iloc[0])

    print(f"[final] SUB-1 (safe tek-model) = {best_single}  rw-OOF={_rw(best_single):.4f}")
    print(f"[final] SUB-2 (ensemble blend) = blend       rw-OOF={_rw('blend'):.4f}")
    if best_single == "lgbm_full":
        print("[final] NOT: en iyi tek model lgbm_full (recency lever'lari rw-OOF'u dusurmedi).")

    sub1 = _write_submission(best_single, f"sub1_{best_single}.csv")
    sub2 = _write_submission("blend", "sub2_blend.csv")

    # iki final yapisal farkli mi? (tek-model vs ensemble) — risk dagitimi teyidi
    import numpy as np
    diff = float(np.mean(np.abs(sub1[cv.TARGET_COL].to_numpy() - sub2[cv.TARGET_COL].to_numpy())))
    print(f"[final] SUB-1 vs SUB-2 ortalama |fark| = {diff:.4f} (yapisal cesitlilik gostergesi).")

    today = _dt.date.today().isoformat()
    rows = []
    for role, model in (("SUB-1 (safe tek-model)", best_single), ("SUB-2 (ensemble blend)", "blend")):
        cm, cs = ms._cv_scores(model)
        rows.append({
            "tarih": today,
            "model_aciklama": model,
            "commit_hash": "",
            "cv_mse_mean": round(cm, 6),
            "cv_mse_std": round(cs, 6),
            "recency_weighted_oof_mse": round(_rw(model), 6),
            "public_lb_mse": "",
            "gap": "",
            "esik_durumu": "",  # cv.gap_status ciktisi icin rezerve; public LB sonucu girilince doldurulur
            "test_uretim_yolu": "fold-bagging (15 model)" if model != "blend" else "OOF-stack (recency-weighted)",
            "secildi": True,
            "not": role,
        })
    _log_selected(rows)
    print(f"[final] reports/submissions_log.csv guncellendi: 2 final secildi=True.")
    print("[final] FORMAT ASSERT'leri GECTI -> 2 final Kaggle'a hazir.")


if __name__ == "__main__":
    main()
