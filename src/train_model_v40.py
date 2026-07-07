"""
submission_v15 — YAPISAL KESIF: segment-yil target encoding.

Bulgu: yil trendi segmentlere gore cok farkli (Cybersecurity -3.98 vs
DevOps -0.86). Global yil-norm bunu kacirir. Cozum: rol/tier/hobby/sosyal
icin fold-ici (segment x yil) TE — hiyerarsik smoothing (hucre -> yil
ortalamasina, sm=20). Tek-LGBM kazanci: 89.94 -> 88.53.

Tarif: v12 (yn + uniform + eski tuned parametreler) + 4 segment-yil TE.
10-fold x 2 seed. Ridge-meta/NNLS yarisi + yil-calib kontrolu.
Cikti: submission_v40.csv + submission_v41.csv (0.5*v15 + 0.5*v7)

Calistir: python -u src/train_model_v15.py
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
OUT = ROOT / "submissions" / "submission_v40.csv"
OUT_MIX = ROOT / "submissions" / "submission_v41.csv"

SEED = 42
N_FOLDS = 15
SEEDS = [42, 7, 2024]
REG_NAMES = ["lgbm", "xgb", "cat", "mlp"]
ALL_NAMES = REG_NAMES + ["nn"]
SEG_COLS = ["target_role", "university_tier", "hobby",
            "preferred_social_media_platform"]
SM = 20


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def target_encode(tr_col, apply_col, y_tr, gmean, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y_tr}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + gmean * smoothing) / (stats["count"] + smoothing)
    return apply_col.map(enc).fillna(gmean).values


def cell_te(seg_tr, yil_tr, y_tr, seg_ap, yil_ap, sm=SM):
    """(segment x yil) TE — hucre ortalamasi yil ortalamasina buzulur."""
    df = pd.DataFrame({"s": seg_tr, "yil": yil_tr, "y": y_tr})
    cell = df.groupby(["s", "yil"])["y"].agg(["mean", "count"])
    yil_mean = df.groupby("yil")["y"].mean()
    cell["te"] = (cell["mean"] * cell["count"]
                  + cell.index.get_level_values("yil").map(yil_mean) * sm) / (cell["count"] + sm)
    s = pd.Series(list(zip(seg_ap, yil_ap))).map(cell["te"])
    return s.fillna(pd.Series(yil_ap).map(yil_mean)).values


def main():
    print("Feature'lar yukleniyor...")
    train, test, y, w_fit, num_cols = F.build_features()
    # optimal skalar seti: bert2 + bert3 + mdeb (bert1 atildi — gürültü)
    train["txt_bert2"] = np.load(CACHE / "bert2_oof.npy")
    test["txt_bert2"] = np.load(CACHE / "bert2_test.npy")
    train["txt_bert3"] = np.load(CACHE / "bert3_oof.npy")
    test["txt_bert3"] = np.load(CACHE / "bert3_test.npy")
    train["txt_mdeb"] = np.load(CACHE / "mdeb_oof.npy")
    test["txt_mdeb"] = np.load(CACHE / "mdeb_test.npy")
    train["txt_mm"] = np.load(CACHE / "mm_oof.npy")
    test["txt_mm"] = np.load(CACHE / "mm_test.npy")
    train["txt_xlmr"] = np.load(CACHE / "xlmr_oof.npy")
    test["txt_xlmr"] = np.load(CACHE / "xlmr_test.npy")
    num_cols = num_cols + ["txt_bert2", "txt_bert3", "txt_mdeb", "txt_mm", "txt_xlmr"]
    # KOHORT-GORELI z-skorlar (feature-only, hedef yok -> sizinti yok;
    # train+test havuzlanmis kohort istatistikleriyle tutarli olcek)
    KEY5 = ["project_quality_score", "technical_interview_score",
            "technical_avg", "portfolio_avg", "coding_score"]
    both = pd.concat([train, test], axis=0, ignore_index=True)
    b_years = both["application_year"].values
    b_roles = both["target_role"].astype(str).values
    n_tr = len(train)
    zcols = []
    for c in KEY5 + ["role_skill"]:
        groups = ([(str(v),) for v in b_years], "yil") , ([(r,) for r in b_roles], "rol"),                  ([(r, str(v)) for r, v in zip(b_roles, b_years)], "rolyil")
        for keys, name in groups:
            if c == "role_skill" and name != "rolyil":
                continue  # role_skill icin sadece rolyil-z (test edilen)
            g = pd.DataFrame({"b": ["|".join(k) for k in keys], "v": both[c].values})
            stats = g.groupby("b")["v"].agg(["mean", "std"])
            z = ((g["v"] - g["b"].map(stats["mean"])) /
                 g["b"].map(stats["std"]).replace(0, 1)).values
            col = f"{c}_z_{name}"
            train[col] = z[:n_tr]
            test[col] = z[n_tr:]
            zcols.append(col)
    num_cols = num_cols + zcols
    print(f"  kohort-z eklendi: {len(zcols)} kolon")
    gmean = y.mean()
    yr_tr = train["application_year"].values
    yr_te = test["application_year"].values
    segs = {c: (train[c].astype(str).values, test[c].astype(str).values)
            for c in SEG_COLS}

    bp = json.loads((CACHE / "best_params_v30.json").read_text())
    lgbm_p, cat_p = bp["lgbm"]["params"], bp["cat"]["params"]
    print(f"  {N_FOLDS}-fold x {SEEDS} | yn+uniform + segment-yil TE {SEG_COLS}")

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
        if name == "cat":
            return CatBoostRegressor(**cat_p, one_hot_max_size=16,
                                     task_type="GPU", devices="0", verbose=0,
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
        # klasik kategorik TE
        for c in F.CAT_COLS:
            Xtr[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[tr_idx][c], y[tr_idx], gmean)
            Xva[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           train.iloc[va_idx][c], y[tr_idx], gmean)
            Xte[c + "_te"] = target_encode(train.iloc[tr_idx][c],
                                           test[c], y[tr_idx], gmean)
        # YENI: segment-yil TE
        for c, (seg_tr_full, seg_te_full) in segs.items():
            Xtr[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx],
                                         seg_tr_full[tr_idx], yr_tr[tr_idx])
            Xva[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx],
                                         seg_tr_full[va_idx], yr_tr[va_idx])
            Xte[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx],
                                         seg_te_full, yr_te)
        ns = len(SEEDS)

        for seed in SEEDS:
            for n in REG_NAMES:
                m = make(n, seed)
                if n == "cat":
                    # CatBoost: sayisal matris + ham kategorik yerine
                    # bu surumde tum feature'lar sayisal (TE'ler dahil)
                    m.fit(Xtr, yn)
                    va_p, te_p = m.predict(Xva), m.predict(Xte)
                else:
                    m.fit(Xtr, yn)
                    va_p, te_p = m.predict(Xva), m.predict(Xte)
                oof[n][va_idx] += (va_p * sd_va + mu_va) / ns
                test_pred[n] += (te_p * sd_te + mu_te) / (ns * N_FOLDS)
        print(f"  fold {fold}/{N_FOLDS} bitti")

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
    s_nnls = wmse(y, np.clip(M @ wb, 0, 100), w_fit)

    extra_tr = train[["application_year", "project_quality_score"]].fillna(0).values
    extra_te = test[["application_year", "project_quality_score"]].fillna(0).values
    Xmeta = np.column_stack([M, extra_tr])
    r_oof = np.zeros(len(y))
    for tr_i, va_i in kf.split(Xmeta):
        r = Ridge(alpha=1.0)
        r.fit(Xmeta[tr_i], y[tr_i], sample_weight=w_fit[tr_i])
        r_oof[va_i] = r.predict(Xmeta[va_i])
    s_ridge = wmse(y, np.clip(r_oof, 0, 100), w_fit)

    print("\n=== BLEND ===")
    print(f"  NNLS      : {s_nnls:.4f}  ({dict(zip(ALL_NAMES, np.round(wb,3)))})")
    print(f"  Ridge meta: {s_ridge:.4f}")

    if s_ridge < s_nnls:
        ens = np.clip(r_oof, 0, 100)
        mdl = Ridge(alpha=1.0).fit(Xmeta, y, sample_weight=w_fit)
        final = np.clip(mdl.predict(np.column_stack([Mte, extra_te])), 0, 100)
        chosen, score = "ridge-meta", s_ridge
    else:
        ens = np.clip(M @ wb, 0, 100)
        final = np.clip(Mte @ wb, 0, 100)
        chosen, score = "nnls", s_nnls

    cal = ens.copy()
    for tr_i, va_i in kf.split(ens):
        for yil in np.unique(yr_tr):
            mt = tr_i[yr_tr[tr_i] == yil]
            mv = va_i[yr_tr[va_i] == yil]
            if len(mt) > 50 and len(mv) > 0:
                b, a = np.polyfit(ens[mt], y[mt], 1)
                cal[mv] = a + b * ens[mv]
    cal = np.clip(cal, 0, 100)
    s_cal = wmse(y, cal, w_fit)
    use_cal = s_cal < score
    print(f"Yil-kalibrasyon: {score:.4f} -> {s_cal:.4f} "
          f"({'UYGULANIYOR' if use_cal else 'atlandi'})")
    if use_cal:
        for yil in np.unique(yr_te):
            mt = yr_tr == yil
            me = yr_te == yil
            if mt.sum() > 50:
                b, a = np.polyfit(ens[mt], y[mt], 1)
                final[me] = a + b * final[me]
        final = np.clip(final, 0, 100)

    np.savez(CACHE / "preds_v40.npz", y=y, w_fit=w_fit, years=yr_tr,
             **{f"oof_{m}": oof_all[m] for m in ALL_NAMES},
             **{f"test_{m}": test_all[m] for m in ALL_NAMES})

    final = final.round(3)
    sub = pd.DataFrame({F.ID: test[F.ID], F.TARGET: final})
    sub.to_csv(OUT, index=False)
    print(f"\nYAZILDI -> {OUT} (blend: {chosen})")
    print(f"v15 proxy ~{min(score, s_cal):.2f}")

    v7 = pd.read_csv(ROOT / "submissions" / "submission_v7.csv")
    mix = (0.5 * final + 0.5 * v7[F.TARGET].values).round(3)
    pd.DataFrame({F.ID: test[F.ID], F.TARGET: mix}).to_csv(OUT_MIX, index=False)
    print(f"YAZILDI -> {OUT_MIX} (0.5*v15 + 0.5*v7)")


if __name__ == "__main__":
    main()
