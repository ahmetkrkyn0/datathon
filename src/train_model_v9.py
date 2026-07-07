"""
submission_v9 — YIL-NORMALIZE hedef + v7 tarifi + NN cesitliligi.

Bulgular zinciri:
  - Son yillarda feature etkileri ~%30 amplifiye (eski-model slope 1.31)
  - Yil-normalize hedef ((y-mu_yil)/sd_yil) tek LGBM'de +0.66 kazandirdi
  - Two-stage P100 kombo ile CAKISIYOR -> sadece yil-norm kullaniliyor
  - v8 dersi: tuned XGB cesitliligi olduruyor -> XGB untuned kaliyor

Mimari:
  1. 4 regresor yil-normalize hedefle: lgbm(tuned), xgb(UNTUNED),
     cat(tuned), mlp — 10-fold x 2 seed, tahminler y-olcegine cevrilir
  2. + torch NN (cache, y-olcegi) = 5 blend uyesi
  3. NNLS vs Ridge-meta (yil+pq) yarisir, iyi olan kazanir
  4. Ustune yil-bazli affine kalibrasyon (nested dogrulamali)

Calistir: python -u src/train_model_v9.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "submissions" / "submission_v9.csv"

SEED = 42
N_FOLDS = 10
SEEDS = [42, 7]
REG_NAMES = ["lgbm", "xgb", "cat", "mlp"]  # yil-norm hedefle egitilenler
ALL_NAMES = REG_NAMES + ["nn"]             # nn cache'ten (y-olcegi)


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def target_encode(tr_col, apply_col, y_tr, gmean, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y_tr}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + gmean * smoothing) / (stats["count"] + smoothing)
    return apply_col.map(enc).fillna(gmean).values


def main():
    print("Feature'lar yukleniyor...")
    train, test, y, w_fit, num_cols = F.build_features()
    train["txt_bert"] = np.load(CACHE / "bert_oof.npy")
    test["txt_bert"] = np.load(CACHE / "bert_test.npy")
    num_cols = num_cols + ["txt_bert"]
    cat_input_cols = num_cols + F.CAT_COLS
    gmean = y.mean()
    yr_tr = train["application_year"].values
    yr_te = test["application_year"].values
    print(f"  sayisal {len(num_cols)} | {N_FOLDS}-fold x {SEEDS} | yil-norm hedef")

    best = json.loads((CACHE / "best_params.json").read_text())
    cat_p, lgbm_p = best["cat"]["params"], best["lgbm"]["params"]

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def make(name, seed):
        if name == "lgbm":
            return LGBMRegressor(**lgbm_p, subsample_freq=1, random_state=seed,
                                 n_jobs=8, verbose=-1)
        if name == "xgb":  # BILEREK untuned (cesitlilik — v8 dersi)
            return XGBRegressor(
                n_estimators=1500, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                min_child_weight=5, random_state=seed, n_jobs=8,
                tree_method="hist", device="cuda")
        if name == "cat":
            return CatBoostRegressor(**cat_p, task_type="GPU", devices="0",
                                     one_hot_max_size=16, verbose=0,
                                     allow_writing_files=False, random_seed=seed)
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                         learning_rate_init=1e-3, batch_size=256,
                         max_iter=120, early_stopping=True,
                         n_iter_no_change=8, random_state=seed))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(train)) for m in REG_NAMES}
    test_pred = {m: np.zeros(len(test)) for m in REG_NAMES}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        # fold-ici yil istatistikleri (sizinti yok)
        st = pd.DataFrame({"yil": yr_tr[tr_idx], "y": y[tr_idx]}).groupby("yil")["y"].agg(["mean", "std"])
        mu_tr = pd.Series(yr_tr[tr_idx]).map(st["mean"]).values
        sd_tr = pd.Series(yr_tr[tr_idx]).map(st["std"]).values
        yn = (y[tr_idx] - mu_tr) / sd_tr
        mu_va = pd.Series(yr_tr[va_idx]).map(st["mean"]).values
        sd_va = pd.Series(yr_tr[va_idx]).map(st["std"]).values
        mu_te = pd.Series(yr_te).map(st["mean"]).values
        sd_te = pd.Series(yr_te).map(st["std"]).values

        Xtr = train.iloc[tr_idx][num_cols].copy()
        Xva = train.iloc[va_idx][num_cols].copy()
        Xte = test[num_cols].copy()
        for c in F.CAT_COLS:
            Xtr[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[tr_idx][c], y[tr_idx], gmean)
            Xva[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[va_idx][c], y[tr_idx], gmean)
            Xte[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           test[c], y[tr_idx], gmean)
        Ctr = train.iloc[tr_idx][cat_input_cols]
        Cva = train.iloc[va_idx][cat_input_cols]
        Cte = test[cat_input_cols]
        wf = w_fit[tr_idx]
        ns = len(SEEDS)

        for seed in SEEDS:
            for n in REG_NAMES:
                m = make(n, seed)
                if n == "cat":
                    m.fit(Ctr, yn, cat_features=F.CAT_COLS, sample_weight=wf)
                    va_p, te_p = m.predict(Cva), m.predict(Cte)
                elif n == "mlp":
                    m.fit(Xtr, yn)
                    va_p, te_p = m.predict(Xva), m.predict(Xte)
                else:
                    m.fit(Xtr, yn, sample_weight=wf)
                    va_p, te_p = m.predict(Xva), m.predict(Xte)
                # y-olcegine cevir
                oof[n][va_idx] += (va_p * sd_va + mu_va) / ns
                test_pred[n] += (te_p * sd_te + mu_te) / (ns * N_FOLDS)
        print(f"  fold {fold}/{N_FOLDS} bitti")

    # NN (cache, y-olcegi)
    oof_all = dict(oof)
    test_all = dict(test_pred)
    oof_all["nn"] = np.load(CACHE / "nn_oof.npy")
    test_all["nn"] = np.load(CACHE / "nn_test.npy")

    print("\n=== TEKIL OOF (duz | agirlikli ~ LB proxy) ===")
    for n in ALL_NAMES:
        p = np.clip(oof_all[n], 0, 100)
        print(f"  {n:5s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    M = np.column_stack([oof_all[m] for m in ALL_NAMES])
    Mte = np.column_stack([test_all[m] for m in ALL_NAMES])
    sw = np.sqrt(w_fit)
    wb, _ = nnls(M * sw[:, None], y * sw)
    wb /= wb.sum()
    nnls_oof = np.clip(M @ wb, 0, 100)
    nnls_score = wmse(y, nnls_oof, w_fit)

    extra_tr = train[["application_year", "project_quality_score"]].fillna(0).values
    extra_te = test[["application_year", "project_quality_score"]].fillna(0).values
    Xmeta = np.column_stack([M, extra_tr])
    ridge_oof_p = np.zeros(len(y))
    for tr_i, va_i in kf.split(Xmeta):
        r = Ridge(alpha=1.0)
        r.fit(Xmeta[tr_i], y[tr_i], sample_weight=w_fit[tr_i])
        ridge_oof_p[va_i] = r.predict(Xmeta[va_i])
    ridge_oof_p = np.clip(ridge_oof_p, 0, 100)
    ridge_score = wmse(y, ridge_oof_p, w_fit)

    print("\n=== BLEND ===")
    print(f"  NNLS      : {nnls_score:.4f}  ({dict(zip(ALL_NAMES, np.round(wb,3)))})")
    print(f"  Ridge meta: {ridge_score:.4f}")

    if ridge_score < nnls_score:
        ens = ridge_oof_p
        mdl = Ridge(alpha=1.0).fit(Xmeta, y, sample_weight=w_fit)
        final = np.clip(mdl.predict(np.column_stack([Mte, extra_te])), 0, 100)
        chosen, blend_score = "ridge-meta", ridge_score
    else:
        ens = nnls_oof
        final = np.clip(Mte @ wb, 0, 100)
        chosen, blend_score = "nnls", nnls_score

    # yil-bazli affine kalibrasyon
    cal_oof = ens.copy()
    for tr_i, va_i in kf.split(ens):
        for yil in np.unique(yr_tr):
            m_tr = tr_i[yr_tr[tr_i] == yil]
            m_va = va_i[yr_tr[va_i] == yil]
            if len(m_tr) > 50 and len(m_va) > 0:
                b, a = np.polyfit(ens[m_tr], y[m_tr], 1)
                cal_oof[m_va] = a + b * ens[m_va]
    cal_oof = np.clip(cal_oof, 0, 100)
    cal_score = wmse(y, cal_oof, w_fit)
    use_cal = cal_score < blend_score
    print(f"Yil-kalibrasyon: {blend_score:.4f} -> {cal_score:.4f} "
          f"({'UYGULANIYOR' if use_cal else 'atlandi'})")
    if use_cal:
        for yil in np.unique(yr_te):
            m_tr = yr_tr == yil
            m_te = yr_te == yil
            if m_tr.sum() > 50:
                b, a = np.polyfit(ens[m_tr], y[m_tr], 1)
                final[m_te] = a + b * final[m_te]
        final = np.clip(final, 0, 100)
    final = final.round(3)

    np.savez(CACHE / "preds_v9.npz", y=y, w_fit=w_fit, years=yr_tr,
             **{f"oof_{m}": oof_all[m] for m in ALL_NAMES},
             **{f"test_{m}": test_all[m] for m in ALL_NAMES})

    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    score = cal_score if use_cal else blend_score
    print(f"\nYAZILDI -> {OUT} (blend: {chosen})")
    print(f"LB tahmini ~{score:.2f} (proxy ~0.5 muhafazakar olabilir)")
    print(f"satir: {len(sub)} | aralik: {final.min()}-{final.max()} | ort: {final.mean():.2f}")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
