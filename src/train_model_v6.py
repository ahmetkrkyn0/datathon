"""
submission_v6 — pseudo-labeling (DURUST ic-ice protokol).

Motivasyon: test'in %42'si 2025-26 (train'de n=2188); pseudo-label ile
son-yil orneklem sayisi 3x'e cikar. Naif alfa taramasi 84.8 gosterdi ama
SIZINTILI idi (pseudo'lar val satirlarini gormus modellerden geliyordu).

Burada temiz protokol:
  Faz A (degerlendirme, 5-fold):
    her fold k icin:
      1. base modeller folds!=k ile egitilir -> test pseudo-etiketleri p_k
      2. ayni modeller folds!=k + test(p_k, agirlik alfa) ile yeniden egitilir
      3. val_k tahmini -> DURUST OOF
    alfa ∈ {0.7, 1.2} kiyaslanir, agirlikli MSE raporlanir.
  Faz B (final):
    en iyi alfa ile: full train + test(v4b tahminleri pseudo) -> 2 seed
    NNLS agirliklari Faz A'nin durust OOF'undan -> submission_v6.csv

Feature: v5 seti + txt_bert (BERT OOF tahmini, cache'ten).

Calistir: python -u src/train_model_v6.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v6.csv"

SEED = 42
N_FOLDS = 5
ALPHAS = [0.7, 1.2]
FINAL_SEEDS = [42, 7]
BLEND_W = np.array([0.422, 0.275, 0.303])  # v4 NNLS (lgbm, xgb, cat) — pseudo uretiminde

NAMES = ["lgbm", "xgb", "cat"]


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def main():
    print("Feature'lar yukleniyor...")
    train, test, y, w_fit, num_cols = F.build_features()
    train["txt_bert"] = np.load(CACHE / "bert_oof.npy")
    test["txt_bert"] = np.load(CACHE / "bert_test.npy")
    num_cols = num_cols + ["txt_bert"]
    cat_cols_in = num_cols + F.CAT_COLS
    print(f"  sayisal {len(num_cols)} (txt_bert dahil)")

    best = json.loads((CACHE / "best_params.json").read_text())
    cat_p, lgbm_p = best["cat"]["params"], best["lgbm"]["params"]

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def make(name, seed):
        if name == "lgbm":
            return LGBMRegressor(**lgbm_p, subsample_freq=1, random_state=seed,
                                 n_jobs=8, verbose=-1)
        if name == "xgb":
            return XGBRegressor(
                n_estimators=1500, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                min_child_weight=5, random_state=seed, n_jobs=8,
                tree_method="hist", device="cuda")
        return CatBoostRegressor(**cat_p, task_type="GPU", devices="0",
                                 one_hot_max_size=16, verbose=0,
                                 allow_writing_files=False, random_seed=seed)

    def fit_predict(name, seed, X, yy, ww, X_out_list):
        m = make(name, seed)
        if name == "cat":
            m.fit(X[cat_cols_in], yy, cat_features=F.CAT_COLS, sample_weight=ww)
            return [m.predict(Xo[cat_cols_in]) for Xo in X_out_list]
        m.fit(X[num_cols], yy, sample_weight=ww)
        return [m.predict(Xo[num_cols]) for Xo in X_out_list]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ---------------- FAZ A: durust degerlendirme ----------------
    oof_base = {m: np.zeros(len(train)) for m in NAMES}
    oof_ps = {a: {m: np.zeros(len(train)) for m in NAMES} for a in ALPHAS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        Ttr = train.iloc[tr_idx]
        Tva = train.iloc[va_idx]
        wf = w_fit[tr_idx]

        # 1) base: pseudo uret
        base_te = {}
        for n in NAMES:
            va_p, te_p = fit_predict(n, SEED, Ttr, y[tr_idx], wf, [Tva, test])
            oof_base[n][va_idx] = va_p
            base_te[n] = te_p
        pseudo = np.clip(np.column_stack([base_te[n] for n in NAMES]) @ BLEND_W, 0, 100)

        # 2) pseudo ile yeniden egit
        aug = pd.concat([Ttr, test], axis=0, ignore_index=True)
        for a in ALPHAS:
            yy = np.concatenate([y[tr_idx], pseudo])
            ww = np.concatenate([wf, np.full(len(test), a)])
            for n in NAMES:
                (va_p,) = fit_predict(n, SEED, aug, yy, ww, [Tva])
                oof_ps[a][n][va_idx] = va_p
        print(f"  fold {fold}/{N_FOLDS} bitti")

    print("\n=== FAZ A: DURUST agirlikli OOF ===")
    Mb = np.column_stack([oof_base[n] for n in NAMES])
    sw = np.sqrt(w_fit)
    wb, _ = nnls(Mb * sw[:, None], y * sw)
    wb /= wb.sum()
    print(f"  base (pseudo'suz): {wmse(y, np.clip(Mb @ wb, 0, 100), w_fit):.4f}")

    results = {}
    nnls_w = {}
    for a in ALPHAS:
        Ma = np.column_stack([oof_ps[a][n] for n in NAMES])
        wa, _ = nnls(Ma * sw[:, None], y * sw)
        wa /= wa.sum()
        score = wmse(y, np.clip(Ma @ wa, 0, 100), w_fit)
        results[a] = score
        nnls_w[a] = wa
        print(f"  alfa={a}: {score:.4f}  (blend {np.round(wa,3)})")

    a_best = min(results, key=results.get)
    print(f"\nEN IYI ALFA = {a_best} ({results[a_best]:.4f})")

    # ---------------- FAZ B: final model ----------------
    print("\nFaz B: final egitim (full train + pseudo)...")
    pseudo_final = pd.read_csv(ROOT / "submissions" / "submission_v4b.csv")[
        F.TARGET].values
    aug = pd.concat([train, test], axis=0, ignore_index=True)
    yy = np.concatenate([y, pseudo_final])
    ww = np.concatenate([w_fit, np.full(len(test), a_best)])

    test_pred = {n: np.zeros(len(test)) for n in NAMES}
    for seed in FINAL_SEEDS:
        for n in NAMES:
            (te_p,) = fit_predict(n, seed, aug, yy, ww, [test])
            test_pred[n] += te_p / len(FINAL_SEEDS)

    Mte = np.column_stack([test_pred[n] for n in NAMES])
    final = np.clip(Mte @ nnls_w[a_best], 0, 100).round(3)

    np.savez(CACHE / "preds_v6.npz", y=y, w_fit=w_fit,
             **{f"oof_{m}": oof_ps[a_best][m] for m in NAMES},
             **{f"test_{m}": test_pred[m] for m in NAMES})

    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"\nYAZILDI -> {OUT}")
    print(f"LB tahmini ~{results[a_best]:.2f} (durust protokol)")
    print(f"satir: {len(sub)} | aralik: {final.min()}-{final.max()} | ort: {final.mean():.2f}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
