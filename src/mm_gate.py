"""
TIER-3 mm — PAIRED-TEST GATE (e5_ridge ile BIREBIR olcut).
==========================================================

    python src/mm_gate.py        # mm'in blend faydasini paired-test ile yargila (KARAR)

NE: `ridge_pos` blend'in NESTED rw-OOF'unu mm DAHIL ve mm HARIC iki kez hesaplar; farki
  (delta) hem (a) 15 (repeat,fold) hucresinde PAIRED test (mean/std, t, p, iyilesen-hucre sayisi)
  hem (b) satir-bootstrap %95 CI ile yargilar. e5_ridge kabulunde kullanilan AYNI metodoloji
  (reports/E5_EMBEDDING_LEVER.md §D): literal 0.25*std kapisi paired karsilastirma icin YANLIS
  olcut (o std blend'in MUTLAK-MSE seviye-varyansi); karar paired delta'nin KENDI belirsizligine
  gore verilir.

KARAR (otomatik on-yargi; nihai KABUL kullanici onayi):
  * GECTI  : paired delta robust-negatif (t<0, p kucuk, CI ust-siniri < 0, iyilesen-hucre cogunluk).
             -> ensemble.py CANDIDATE_POOL'a "mm" ekle (asagidaki talimat), src/ensemble.py +
                src/finalize_submissions.py calistir -> SUB-2 (e5+mm) guncellenir.
  * ELENDI : delta gurultu-bandi (CI sifiri kapsiyor / iyilesme yok / pozitif). -> TEMIZ REDDET,
             defterle. "neural multimodal da floor'u kiramadi" = juri kanit. SUB-2 DEGISMEZ.

mm DOKUNULMAZLIK: bu script CANDIDATE_POOL'u DEGISTIRMEZ (sadece teshis/karar uretir). Gercek
  blend degisimi e5 deseni gibi ACIK adim: gecerse ensemble.py'de "mm" satirini ac + calistir.
  SUB-1 (catboost_full) HER DURUMDA dokunulmaz.

Determinizm: bootstrap RNG SEED-sabit (np.random.default_rng(SEED)); paired hesap kapali-form.
"""

from __future__ import annotations

import numpy as np

import cv
import ensemble as ens

# e5'in halihazirda KABUL edilmis blend havuzu (mm HARIC). mm bunun UZERINE eklenince fayda olcu.
BASE_POOL = [m for m in ens.CANDIDATE_POOL if m != "mm"]
MM = "mm"
N_BOOT = 5000


def _load_pool(models, y, folds, sid):
    """Mevcut olan model OOF/test kolonlarini yukle (sirayi koru). Doner: (cands, P, T)."""
    cands = []
    for m in models:
        if (cv.ARTIFACTS_DIR / f"oof_{m}.npy").exists() and (cv.ARTIFACTS_DIR / f"test_{m}.npy").exists():
            cands.append(m)
    P = np.column_stack([np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy") for m in cands])
    T = np.column_stack([np.load(cv.ARTIFACTS_DIR / f"test_{m}.npy") for m in cands])
    return cands, P, T


def _blend_meta_oof(P, y, w, folds, sid, method="ridge_pos"):
    """ensemble.nested_rw_oof ile AYNI: nested meta_oof (durust). Doner: (rw, meta_oof)."""
    return ens.nested_rw_oof(P, y, w, folds, sid, method)


def _per_cell_rw_mse(resid_w, y, meta_oof, w, folds, sid):
    """Her (repeat,fold) hucresinde recency-weighted MSE (paired delta icin hucre-bazli)."""
    pos = {s: i for i, s in enumerate(sid)}
    out = {}
    for r in sorted(folds["repeat"].unique()):
        for f in sorted(folds["fold"].unique()):
            sub = folds[(folds["repeat"] == r) & (folds["fold"] == f)]
            idx = np.array([pos[s] for s in sub["student_id"].values])
            ww = w[idx]
            out[(r, f)] = float(np.sum(ww * (y[idx] - meta_oof[idx]) ** 2) / np.sum(ww))
    return out


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = np.asarray(train[cv.TARGET_COL].values, dtype=float)
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    if not (cv.ARTIFACTS_DIR / f"oof_{MM}.npy").exists():
        raise SystemExit(
            f"[mm_gate] HATA: oof_{MM}.npy yok. Once: colab notebook -> artifacts/ -> python src/mm_blend.py"
        )

    # --- blend WITHOUT mm (mevcut kabul edilmis havuz; = SUB-2 base) ---
    cands_base, P_base, _ = _load_pool(BASE_POOL, y, folds, sid)
    rw_base, oof_base = _blend_meta_oof(P_base, y, w, folds, sid)

    # --- blend WITH mm ---
    cands_mm, P_mm, _ = _load_pool(BASE_POOL + [MM], y, folds, sid)
    rw_mm, oof_mm = _blend_meta_oof(P_mm, y, w, folds, sid)

    delta_overall = rw_mm - rw_base
    print(f"[mm_gate] blend nested rw-OOF:  base(mm-siz)={rw_base:.4f}   +mm={rw_mm:.4f}   "
          f"delta={delta_overall:+.4f}")
    print(f"[mm_gate] havuz: base={'+'.join(cands_base)}  |  +mm={'+'.join(cands_mm)}")

    # --- (a) PAIRED test: 15 (repeat,fold) hucresinde per-cell rw-MSE deltasi ---
    cell_base = _per_cell_rw_mse(None, y, oof_base, w, folds, sid)
    cell_mm = _per_cell_rw_mse(None, y, oof_mm, w, folds, sid)
    keys = sorted(cell_base.keys())
    d = np.array([cell_mm[k] - cell_base[k] for k in keys])  # negatif = mm iyilestirdi
    n_cells = len(d)
    improved = int(np.sum(d < 0))
    d_mean, d_std = float(d.mean()), float(d.std(ddof=1))
    # paired t (H0: delta=0); std==0 korumasi
    if d_std > 0:
        t_stat = d_mean / (d_std / np.sqrt(n_cells))
        # iki-yonlu p (t-dagilimi, df=n-1); scipy varsa kesin, yoksa normal yaklasim
        try:
            from scipy import stats
            p_val = float(2 * stats.t.sf(abs(t_stat), df=n_cells - 1))
        except Exception:
            from math import erf, sqrt
            p_val = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
    else:
        t_stat, p_val = float("nan"), float("nan")

    print(f"[mm_gate] PAIRED (15 hucre): iyilesen={improved}/{n_cells}  "
          f"delta_mean={d_mean:+.4f} +/- {d_std:.4f}  t={t_stat:.3f}  p={p_val:.2e}")

    # --- (b) satir-bootstrap %95 CI: overall rw-OOF delta (mm vs base) ---
    rng = np.random.default_rng(cv.SEED)
    n = len(y)
    boot = np.empty(N_BOOT)
    rb = (y - oof_base) ** 2
    rm = (y - oof_mm) ** 2
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        ws = w[s]
        boot[b] = (np.sum(ws * rm[s]) / np.sum(ws)) - (np.sum(ws * rb[s]) / np.sum(ws))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_ge0 = float(np.mean(boot >= 0.0))
    print(f"[mm_gate] BOOTSTRAP (B={N_BOOT}) overall delta %95 CI = [{lo:+.4f}, {hi:+.4f}]  "
          f"P(delta>=0)={p_ge0:.4f}")

    # --- literal 0.25*std kapisi (referans; e5'te de TEK BASINA YETERSIZ oldugu belgeli) ---
    _, cv_std_base, _ = cv.compute_cv_mse(oof_base, y, folds, sid)
    literal_band = 0.25 * cv_std_base
    literal_pass = rw_mm < rw_base - literal_band
    print(f"[mm_gate] literal kapi (0.25*std={literal_band:.4f}): "
          f"{'GECER' if literal_pass else 'GECMEZ'} (e5'te de paired olcut esas alindi)")

    # --- otomatik on-yargi (e5 olcutu): robust-negatif paired delta ---
    paired_sig = (d_mean < 0) and (improved >= int(np.ceil(0.8 * n_cells))) and (p_val < 0.01) and (hi < 0)
    verdict = "GECTI (paired-anlamli)" if paired_sig else "ELENDI (gurultu-bandi / iyilesme yok)"
    print("\n[mm_gate] ================== KARAR (on-yargi; nihai onay kullanicida) ==================")
    print(f"[mm_gate]   overall delta = {delta_overall:+.4f}  |  paired t={t_stat:.3f} p={p_val:.2e}  "
          f"|  CI=[{lo:+.4f},{hi:+.4f}]  |  iyilesen {improved}/{n_cells}")
    print(f"[mm_gate]   >>> {verdict}")
    if paired_sig:
        print("[mm_gate]   AKSIYON (gecerse): ensemble.py CANDIDATE_POOL'a \"mm\" ekle -> "
              "python src/ensemble.py && python src/finalize_submissions.py (SUB-2 e5+mm).")
        print("[mm_gate]   SUB-1 (catboost_full) dokunulmaz. Repro: belgelenmis tolerans (neural).")
    else:
        print("[mm_gate]   AKSIYON: TEMIZ REDDET + defterle. 'neural multimodal da floor'u kiramadi'.")
        print("[mm_gate]   SUB-2 DEGISMEZ (e5 blend 84.85). CANDIDATE_POOL'a mm EKLENMEZ.")


if __name__ == "__main__":
    main()
