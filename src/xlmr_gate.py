"""TIER-2 BIRLESME — xlmr PAIRED-TEST GATE (mm_gate / e5 ile BIREBIR olcut).
==========================================================================

    python src/xlmr_gate.py

NE: xlmr'in mevcut 10-model blend'e (84.0212) BLEND faydasini paired-test ile yargilar
  (mm_gate.py metodolojisi: 15-hucre paired t + 5000 satir-bootstrap %95 CI). Literal 0.25*std
  paired icin yanlis olcut; karar paired delta'nin KENDI belirsizligine gore. mm_gate yardimcilarini
  yeniden kullanir (TEK kaynak). KARAR otomatik on-yargi; nihai KABUL kullanici onayi.

GECTI -> ensemble.py CANDIDATE_POOL'a 'xlmr' ekle + ensemble.py/finalize. ELENDI -> temiz RED."""

from __future__ import annotations

import numpy as np

import cv
import ensemble as ens
from mm_gate import _load_pool, _blend_meta_oof, _per_cell_rw_mse, N_BOOT

NEW = "xlmr"
# mevcut RESMI 10-model havuz (= SUB-2 base). xlmr bunun UZERINE.
BASE_POOL = list(ens.CANDIDATE_POOL)


def main() -> None:
    cv.set_seed()
    train, test, folds = cv.load_train(), cv.load_test(), cv.load_folds()
    y = np.asarray(train[cv.TARGET_COL].values, dtype=float)
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    if not (cv.ARTIFACTS_DIR / f"oof_{NEW}.npy").exists():
        raise SystemExit(f"[xlmr_gate] HATA: oof_{NEW}.npy yok. Once Colab -> artifacts/ -> xlmr_blend.py")

    cands_base, P_base, _ = _load_pool(BASE_POOL, y, folds, sid)
    rw_base, oof_base = _blend_meta_oof(P_base, y, w, folds, sid)
    cands_new, P_new, _ = _load_pool(BASE_POOL + [NEW], y, folds, sid)
    rw_new, oof_new = _blend_meta_oof(P_new, y, w, folds, sid)

    delta_overall = rw_new - rw_base
    print(f"[xlmr_gate] blend nested rw-OOF:  base(10-model)={rw_base:.4f}   +xlmr={rw_new:.4f}   "
          f"delta={delta_overall:+.4f}")

    # (a) PAIRED 15-hucre
    cb = _per_cell_rw_mse(None, y, oof_base, w, folds, sid)
    cn = _per_cell_rw_mse(None, y, oof_new, w, folds, sid)
    keys = sorted(cb.keys())
    d = np.array([cn[k] - cb[k] for k in keys])
    n_cells = len(d); improved = int(np.sum(d < 0))
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
    print(f"[xlmr_gate] PAIRED (15 hucre): iyilesen={improved}/{n_cells}  "
          f"delta_mean={d_mean:+.4f} +/- {d_std:.4f}  t={t_stat:.3f}  p={p_val:.2e}")

    # (b) satir-bootstrap %95 CI
    rng = np.random.default_rng(cv.SEED)
    n = len(y); boot = np.empty(N_BOOT)
    rb = (y - oof_base) ** 2; rm = (y - oof_new) ** 2
    for b in range(N_BOOT):
        s = rng.integers(0, n, n); ws = w[s]
        boot[b] = (np.sum(ws * rm[s]) / np.sum(ws)) - (np.sum(ws * rb[s]) / np.sum(ws))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_ge0 = float(np.mean(boot >= 0.0))
    print(f"[xlmr_gate] BOOTSTRAP (B={N_BOOT}) overall delta %95 CI = [{lo:+.4f}, {hi:+.4f}]  "
          f"P(delta>=0)={p_ge0:.4f}")

    _, cv_std_base, _ = cv.compute_cv_mse(oof_base, y, folds, sid)
    literal_band = 0.25 * cv_std_base
    print(f"[xlmr_gate] literal kapi (0.25*std={literal_band:.4f}): "
          f"{'GECER' if rw_new < rw_base - literal_band else 'GECMEZ'} (paired olcut esas)")

    # otomatik on-yargi (mm/e5 ile AYNI esik): >=80% hucre + p<0.01 + CI ust-sinir<0
    paired_sig = (d_mean < 0) and (improved >= int(np.ceil(0.8 * n_cells))) and (p_val < 0.01) and (hi < 0)
    verdict = "GECTI (paired-anlamli)" if paired_sig else "ELENDI (gurultu-bandi / tutarsiz)"
    print("\n[xlmr_gate] ============ KARAR (on-yargi; nihai onay kullanicida) ============")
    print(f"[xlmr_gate]   overall delta={delta_overall:+.4f} | paired t={t_stat:.3f} p={p_val:.2e} "
          f"| CI=[{lo:+.4f},{hi:+.4f}] | iyilesen {improved}/{n_cells}")
    print(f"[xlmr_gate]   >>> {verdict}")
    if paired_sig:
        print("[xlmr_gate]   AKSIYON: ensemble.py CANDIDATE_POOL'a \"xlmr\" ekle -> python src/ensemble.py")
        print("[xlmr_gate]   -> sonra make_submission/finalize ile SUB-2 guncellenebilir. SUB-1 dokunulmaz.")
    else:
        print("[xlmr_gate]   AKSIYON: TEMIZ REDDET + defterle. SUB-2 (84.02) DEGISMEZ.")


if __name__ == "__main__":
    main()
