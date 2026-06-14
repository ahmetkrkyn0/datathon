"""
GENEL PAIRED-TEST GATE — yeni bir base model adayini mm_gate.py ile BIREBIR olcutle yargila.
==========================================================================================

    python src/new_model_gate.py berturk      # oof_berturk.npy + test_berturk.npy gate
    python src/new_model_gate.py tabpfn        # oof_tabpfn.npy  + test_tabpfn.npy  gate

NE: Mevcut KABUL EDILMIS blend havuzunun (ensemble.CANDIDATE_POOL) nested rw-OOF'unu yeni model
  DAHIL ve HARIC iki kez hesaplar; farki (delta) (a) 15 (repeat,fold) hucresinde PAIRED test
  (mean/std, t, p, iyilesen-hucre) + (b) satir-bootstrap %95 CI ile yargilar. mm/e5/xlmr/ourteam_tf
  kabulunde kullanilan AYNI metodoloji (reports/E5_EMBEDDING_LEVER.md §D). Literal 0.25*std kapisi
  paired karsilastirma icin yanlis olcut (blend MUTLAK-MSE seviye-varyansi); karar paired delta'nin
  KENDI belirsizligine gore.

KARAR (otomatik on-yargi; nihai KABUL kullanici onayi + public-gap teyidi):
  * GECTI  : paired delta robust-negatif (t<0, p<0.01, CI ust-siniri<0, iyilesen-hucre >=%80).
             -> ensemble.py CANDIDATE_POOL'a model adini ekle -> ensemble.py + finalize calistir.
  * ELENDI : delta gurultu-bandi (CI sifiri kapsiyor / iyilesme yok / pozitif). TEMIZ REDDET.

EK TESHIS: ortogonallik (yeni modelin mevcut uyelerle korelasyonu) + standalone rw-OOF. corr>0.97
  ise blende NET-YENI sinyal yok demektir (kabul edilse bile fayda redundant cikar).

SUB-1 (catboost_full) HER DURUMDA dokunulmaz. Bu script CANDIDATE_POOL'u DEGISTIRMEZ (teshis/karar).
Determinizm: bootstrap RNG SEED-sabit.
"""

from __future__ import annotations

import sys

import numpy as np

import cv
import ensemble as ens

N_BOOT = 5000


def _load_pool(models, sid):
    cands = []
    for m in models:
        if (cv.ARTIFACTS_DIR / f"oof_{m}.npy").exists() and (cv.ARTIFACTS_DIR / f"test_{m}.npy").exists():
            cands.append(m)
    P = np.column_stack([np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy") for m in cands])
    return cands, P


def _per_cell_rw_mse(y, meta_oof, w, folds, sid):
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
    if len(sys.argv) < 2:
        raise SystemExit("kullanim: python src/new_model_gate.py <model_adi>  (or. berturk | tabpfn)")
    NEW = sys.argv[1]

    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = np.asarray(train[cv.TARGET_COL].values, dtype=float)
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    if not (cv.ARTIFACTS_DIR / f"oof_{NEW}.npy").exists():
        raise SystemExit(
            f"[gate:{NEW}] HATA: oof_{NEW}.npy yok. Once colab notebook'u GPU'da calistir -> "
            f"oof_{NEW}.npy + test_{NEW}.npy -> artifacts/ kopyala."
        )

    BASE_POOL = [m for m in ens.CANDIDATE_POOL if m != NEW]

    # --- standalone + ortogonallik on-teshis ---
    oof_new = np.load(cv.ARTIFACTS_DIR / f"oof_{NEW}.npy")
    rw_new = cv.compute_recency_weighted_mse(oof_new, y, w)
    cm_new, _, _ = cv.compute_cv_mse(oof_new, y, folds, sid)
    print(f"[gate:{NEW}] standalone: rw-OOF={rw_new:.4f}  unweighted-CV={cm_new:.4f}")
    print(f"[gate:{NEW}] ortogonallik (corr ile mevcut uyeler; >0.97 = redundant sinyal):")
    for m in BASE_POOL:
        if (cv.ARTIFACTS_DIR / f"oof_{m}.npy").exists():
            om = np.load(cv.ARTIFACTS_DIR / f"oof_{m}.npy")
            c = float(np.corrcoef(oof_new, om)[0, 1])
            flag = "  <-- redundant" if c > 0.97 else ""
            print(f"    corr({NEW},{m:14s}) = {c:.3f}{flag}")

    # --- blend WITHOUT new ---
    cands_base, P_base = _load_pool(BASE_POOL, sid)
    rw_base, oof_base = ens.nested_rw_oof(P_base, y, w, folds, sid, "ridge_pos")

    # --- blend WITH new ---
    cands_new, P_new = _load_pool(BASE_POOL + [NEW], sid)
    rw_with, oof_with = ens.nested_rw_oof(P_new, y, w, folds, sid, "ridge_pos")

    delta_overall = rw_with - rw_base
    print(f"\n[gate:{NEW}] blend nested rw-OOF:  base={rw_base:.4f}   +{NEW}={rw_with:.4f}   "
          f"delta={delta_overall:+.4f}")
    print(f"[gate:{NEW}] havuz: base={'+'.join(cands_base)}  |  +{NEW}={'+'.join(cands_new)}")

    # --- (a) PAIRED test ---
    cell_base = _per_cell_rw_mse(y, oof_base, w, folds, sid)
    cell_with = _per_cell_rw_mse(y, oof_with, w, folds, sid)
    keys = sorted(cell_base.keys())
    d = np.array([cell_with[k] - cell_base[k] for k in keys])
    n_cells = len(d)
    improved = int(np.sum(d < 0))
    d_mean, d_std = float(d.mean()), float(d.std(ddof=1))
    if d_std > 0:
        t_stat = d_mean / (d_std / np.sqrt(n_cells))
        try:
            from scipy import stats
            p_val = float(2 * stats.t.sf(abs(t_stat), df=n_cells - 1))
        except Exception:
            from math import erf, sqrt
            p_val = float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))
    else:
        t_stat, p_val = float("nan"), float("nan")
    print(f"[gate:{NEW}] PAIRED (15 hucre): iyilesen={improved}/{n_cells}  "
          f"delta_mean={d_mean:+.4f} +/- {d_std:.4f}  t={t_stat:.3f}  p={p_val:.2e}")

    # --- (b) satir-bootstrap %95 CI ---
    rng = np.random.default_rng(cv.SEED)
    n = len(y)
    boot = np.empty(N_BOOT)
    rb = (y - oof_base) ** 2
    rm = (y - oof_with) ** 2
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        ws = w[s]
        boot[b] = (np.sum(ws * rm[s]) / np.sum(ws)) - (np.sum(ws * rb[s]) / np.sum(ws))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_ge0 = float(np.mean(boot >= 0.0))
    print(f"[gate:{NEW}] BOOTSTRAP (B={N_BOOT}) overall delta %95 CI = [{lo:+.4f}, {hi:+.4f}]  "
          f"P(delta>=0)={p_ge0:.4f}")

    # --- karar (mm/e5 olcutu) ---
    paired_sig = (d_mean < 0) and (improved >= int(np.ceil(0.8 * n_cells))) and (p_val < 0.01) and (hi < 0)
    verdict = "GECTI (paired-anlamli)" if paired_sig else "ELENDI (gurultu-bandi / iyilesme yok)"
    print(f"\n[gate:{NEW}] ================== KARAR (on-yargi; nihai onay + public-gap kullanicida) ===========")
    print(f"[gate:{NEW}]   overall delta={delta_overall:+.4f}  paired t={t_stat:.3f} p={p_val:.2e}  "
          f"CI=[{lo:+.4f},{hi:+.4f}]  iyilesen {improved}/{n_cells}")
    print(f"[gate:{NEW}]   >>> {verdict}")
    if paired_sig:
        print(f"[gate:{NEW}]   AKSIYON: ensemble.py CANDIDATE_POOL'a \"{NEW}\" ekle -> "
              "python src/ensemble.py && python src/finalize_submissions.py. SONRA public-gap olc.")
    else:
        print(f"[gate:{NEW}]   AKSIYON: TEMIZ REDDET + defterle. Blend DEGISMEZ. SUB-1/SUB-2 dokunulmaz.")


if __name__ == "__main__":
    main()
