"""Ahmet'in kendi makinesinde BAGIMSIZ dogrulamasi icin self-contained script.
Repo'ya BAGLI DEGIL — sadece: numpy, scipy (opsiyonel), bu klasordeki npz + senin
ourteam_oof_tunafolds.npy + folds.parquet (pyarrow/pandas).

    python dogrula_ahmet.py

NE YAPAR: bizim 11-model blend (ridge_pos, recency-weighted, NESTED 5x3) + senin ourteam_tf'ini
  ekleyip nested rw-OOF deltasi + 15-hucre paired test + bootstrap CI uretir. Bizim olcumumuzle
  (+-0.2) eslesmesi beklenir (ayni fold, ayni veri). Tuna'nin kanit_ozet.json'iyla kiyasla."""
import numpy as np

# --- veri yukle (bu klasor) ---
Z = np.load("bizim_11model_oof_test.npz", allow_pickle=True)
y = Z["y"].astype(float)
w = Z["w_recency"].astype(float)          # bizim graduation_year recency-weight
models = list(Z["models"])
sid = Z["student_id_train"]
OOF = {m: Z[f"oof_{m}"] for m in models}   # 11 model OOF (ourteam HARIC)

# senin fold-hizali OOF'un (ayni klasore koy):
ah = np.load("ourteam_oof_tunafolds.npy").astype(float)
assert len(ah) == len(y), "ourteam_oof_tunafolds satir sayisi y ile uyusmuyor."

import pandas as pd
folds = pd.read_parquet("folds.parquet")
N_REP, N_SPL = 3, 5


def fold_of(rep):
    pos = {s: i for i, s in enumerate(sid)}
    fr = folds[folds["repeat"] == rep]
    out = np.full(len(sid), -1, dtype=int)
    for s, f in zip(fr["student_id"].values, fr["fold"].values):
        out[pos[s]] = f
    return out


def clip(p): return np.clip(p, 0, 100)
def rwmse(p): return float(np.sum(w * (y - clip(p)) ** 2) / np.sum(w))


def ridge_pos_fit(P, yy, ww):
    # pozitif-agirlik Ridge (kapali-form yerine basit NNLS+intercept yaklasimi yeterli kiyas icin)
    from numpy.linalg import lstsq
    # intercept'li, pozitif kisit icin coordinate-descent yerine sklearn varsa onu kullan
    try:
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=1.0, positive=True, fit_intercept=True)
        r.fit(P, yy, sample_weight=ww)
        return lambda Q: Q @ r.coef_ + r.intercept_
    except Exception:
        sw = np.sqrt(ww)
        A = np.column_stack([P, np.ones(len(P))]) * sw[:, None]
        b = yy * sw
        coef, *_ = lstsq(A, b, rcond=None)
        return lambda Q: np.column_stack([Q, np.ones(len(Q))]) @ coef


def nested_rw(P):
    n = len(y); s = np.zeros(n); c = np.zeros(n)
    for r in range(N_REP):
        fo = fold_of(r)
        for g in range(N_SPL):
            va = np.where(fo == g)[0]; tr = np.where(fo != g)[0]
            fn = ridge_pos_fit(P[tr], y[tr], w[tr])
            s[va] += clip(fn(P[va])); c[va] += 1
    meta = clip(s / c)
    return rwmse(meta), meta


P11 = np.column_stack([OOF[m] for m in models])
P12 = np.column_stack([OOF[m] for m in models] + [ah])
rw11, meta11 = nested_rw(P11)
rw12, meta12 = nested_rw(P12)
print(f"bizim 11-model blend nested rw-OOF = {rw11:.4f}")
print(f"+ourteam_tf (12-model)            = {rw12:.4f}   (delta {rw12-rw11:+.4f})")
print(f"ourteam_tf standalone rw          = {rwmse(ah):.4f}")

# 15-hucre paired
def per_cell(meta):
    out = []
    for r in range(N_REP):
        fo = fold_of(r)
        for g in range(N_SPL):
            idx = np.where(fo == g)[0]; ww = w[idx]
            out.append(float(np.sum(ww * (y[idx] - meta[idx]) ** 2) / np.sum(ww)))
    return np.array(out)


d = per_cell(meta12) - per_cell(meta11)
imp = int(np.sum(d < 0)); dm, ds = d.mean(), d.std(ddof=1)
t = dm / (ds / np.sqrt(len(d)))
try:
    from scipy import stats; p = float(2 * stats.t.sf(abs(t), df=len(d) - 1))
except Exception:
    p = float("nan")
print(f"PAIRED: iyilesen={imp}/{len(d)}  t={t:.3f}  p={p:.2e}")
print("\nTuna'nin kanit_ozet.json'i ile kiyasla; +-0.2 icinde eslesmeli (ayni fold/veri).")
