"""
FOLD-HIZALI YENIDEN EGITIM (Colab/A100) — Tuna'nin fold semasiyla.

Amac: Tuna'nin "senin OOF'un KFold(10), benim fold'uma hizasiz -> rw-OOF
iyimser-yanli" itirazini kapatmak. v45 tarifimizi (LGBM+Cat+XGB+MLP,
segment-yil TE + kohort-z + year-norm + uniform) AYNEN tutup, sadece
fold kaynagini disaridan aliyoruz: Tuna'nin folds.parquet'i.

Cikti: ourteam_oof_tunafolds.npy + ourteam_test_tunafolds.npy
       (Tuna'nin probe'una verilince fold-hizasizlik itirazi dusser)

Colab girdileri (upload):
  train.csv, test_x.csv, feat_train.pkl, feat_test.pkl, best_params_v30.json,
  bert2/bert3/mdeb/mm/xlmr _oof/_test .npy, nn_oof/nn_test.npy,
  folds.parquet  (TUNA'DAN — fold atamalari)

Calistir: python -u refit_tunafolds_core.py
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# ---- Colab'da hepsi cwd'de; lokalde data/cache'den ----
HERE = Path(".")
def find(name):
    for p in [HERE/name, HERE/"data"/name, HERE/"data"/"cache"/name]:
        if p.exists():
            return p
    raise FileNotFoundError(name)

SEED = 42
SEEDS = [42, 7]
REG_NAMES = ["lgbm", "xgb", "cat", "mlp"]
ALL_NAMES = REG_NAMES + ["nn"]
SEG_COLS = ["target_role", "university_tier", "hobby",
            "preferred_social_media_platform"]
SM = 20
CAT_COLS = ["department", "university_tier", "target_role", "hobby",
            "preferred_social_media_platform"]  # features.py F.CAT_COLS ile birebir ayni
ID = "student_id"
TARGET = "career_success_score"


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def year_weights(train_years, test_years, cap=(0.3, 2.5)):
    p_tr = train_years.value_counts(normalize=True)
    p_te = test_years.value_counts(normalize=True)
    w = train_years.map(p_te / p_tr).fillna(1.0).clip(*cap).values
    return w * len(w) / w.sum()


def target_encode(tr_col, apply_col, y_tr, gmean, smoothing=20):
    stats = pd.DataFrame({"x": tr_col.values, "y": y_tr}).groupby("x")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + gmean * smoothing) / (stats["count"] + smoothing)
    return apply_col.map(enc).fillna(gmean).values


def cell_te(seg_tr, yil_tr, y_tr, seg_ap, yil_ap, sm=SM):
    df = pd.DataFrame({"s": seg_tr, "yil": yil_tr, "y": y_tr})
    cell = df.groupby(["s", "yil"])["y"].agg(["mean", "count"])
    yil_mean = df.groupby("yil")["y"].mean()
    cell["te"] = (cell["mean"] * cell["count"]
                  + cell.index.get_level_values("yil").map(yil_mean) * sm) / (cell["count"] + sm)
    s = pd.Series(list(zip(seg_ap, yil_ap))).map(cell["te"])
    return s.fillna(pd.Series(yil_ap).map(yil_mean)).values


def _read_folds_table():
    """folds dosyasini bul ve DataFrame dondur. parquet/csv destekler."""
    for name in ["folds.parquet", "folds.csv"]:
        try:
            fp = find(name)
        except FileNotFoundError:
            continue
        return pd.read_parquet(fp) if name.endswith(".parquet") else pd.read_csv(fp)
    raise FileNotFoundError("folds.parquet / folds.csv yok")


def load_folds(train_ids):
    """
    Tuna'nin folds.parquet'ini oku -> 15 (repeat x fold) icin validation indeksi.
    Gercek format: 30000 satir (long) = student_id x repeat(0-2); fold 0-4.
    student_id ile train satir sirasina HIZALANIR (kritik — sira degil ID).
    Donus: list of (val_idx) [15 adet], her val_idx train satir indeksleri.
    """
    df = _read_folds_table()
    id2row = {sid: i for i, sid in enumerate(train_ids)}
    splits = []
    if "repeat" in df.columns and "fold" in df.columns and "student_id" in df.columns:
        for r in sorted(df["repeat"].unique()):
            sub = df[df["repeat"] == r]
            for f in sorted(sub["fold"].unique()):
                ids = sub.loc[sub["fold"] == f, "student_id"].values
                rows = np.array([id2row[s] for s in ids], dtype=int)
                splits.append(rows)
    elif "fold" in df.columns and "student_id" in df.columns:  # tek repeat
        for f in sorted(df["fold"].unique()):
            ids = df.loc[df["fold"] == f, "student_id"].values
            splits.append(np.array([id2row[s] for s in ids], dtype=int))
    else:
        raise ValueError(f"folds kolonlari taninmiyor: {list(df.columns)}")
    n_rep = df["repeat"].nunique() if "repeat" in df.columns else 1
    print(f"  folds -> {len(splits)} (repeat x fold) split | {n_rep} repeat | "
          f"ort. val boyutu {np.mean([len(s) for s in splits]):.0f} | "
          f"her satir {len(splits)//5 if len(splits)>=5 else 1}x val olur")
    return splits


def main():
    print("Yukleniyor...")
    train = pd.read_csv(find("train.csv"), encoding="utf-8-sig")
    test = pd.read_csv(find("test_x.csv"), encoding="utf-8-sig")
    # feature dosyalari (build_features ciktisi) — parquet tercih, pkl fallback.
    # parquet surumler-arasi guvenli (Colab pandas 2.x vs lokal 3.x).
    def load_feat(stem):
        for ext, rdr in [(".parquet", pd.read_parquet), (".pkl", pd.read_pickle)]:
            try:
                return rdr(find(stem + ext))
            except FileNotFoundError:
                continue
        raise FileNotFoundError(stem + ".parquet/.pkl")
    train = load_feat("feat_train").copy()
    test = load_feat("feat_test").copy()
    y = train[TARGET].values.astype(float)
    # SADECE sayisal dtype kolonlar (str/object/kategorik ham kolonlar haric).
    # v45'te num_cols build_features'tan gelir; burada pkl'den guvenli turetim.
    num_cols = [c for c in train.columns
                if c not in (ID, TARGET)
                and pd.api.types.is_numeric_dtype(train[c])]

    # txt skalarlari
    for nm in ["bert2", "bert3", "mdeb", "mm", "xlmr"]:
        train[f"txt_{nm}"] = np.load(find(f"{nm}_oof.npy"))
        test[f"txt_{nm}"] = np.load(find(f"{nm}_test.npy"))
        num_cols.append(f"txt_{nm}")

    # F3 potansiyel tuzak
    for df_, fn in [(train, "train.csv"), (test, "test_x.csv")]:
        t = pd.read_csv(find(fn), encoding="utf-8-sig")["mentor_feedback_text"].fillna("").str.lower()
        strong = t.str.contains("mükemmel|olağanüstü|etkileyici|üstün")
        df_["nlp_potansiyel_tuzak"] = (t.str.contains("potansiyel") & ~strong).astype(int).values
    num_cols.append("nlp_potansiyel_tuzak")

    # kohort-z
    KEY5 = ["project_quality_score", "technical_interview_score",
            "technical_avg", "portfolio_avg", "coding_score"]
    both = pd.concat([train, test], axis=0, ignore_index=True)
    b_years = both["application_year"].values
    b_roles = both["target_role"].astype(str).values
    n_tr = len(train)
    for c in KEY5 + ["role_skill"]:
        groups = (([(str(v),) for v in b_years], "yil"),
                  ([(r,) for r in b_roles], "rol"),
                  ([(r, str(v)) for r, v in zip(b_roles, b_years)], "rolyil"))
        for keys, name in groups:
            if c == "role_skill" and name != "rolyil":
                continue
            g = pd.DataFrame({"b": ["|".join(k) for k in keys], "v": both[c].values})
            stats = g.groupby("b")["v"].agg(["mean", "std"])
            z = ((g["v"] - g["b"].map(stats["mean"])) /
                 g["b"].map(stats["std"]).replace(0, 1)).values
            col = f"{c}_z_{name}"
            train[col] = z[:n_tr]; test[col] = z[n_tr:]
            num_cols.append(col)

    gmean = y.mean()
    yr_tr = train["application_year"].values
    yr_te = test["application_year"].values
    w_fit = year_weights(train["application_year"], test["application_year"])
    segs = {c: (train[c].astype(str).values, test[c].astype(str).values) for c in SEG_COLS}

    bp = json.loads(find("best_params_v30.json").read_text())
    lgbm_p, cat_p = bp["lgbm"]["params"], bp["cat"]["params"]

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def make(name, seed):
        if name == "lgbm":
            return LGBMRegressor(**lgbm_p, subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
        if name == "xgb":
            return XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                min_child_weight=5, random_state=seed, n_jobs=8,
                                tree_method="hist", device="cuda")
        if name == "cat":
            return CatBoostRegressor(**cat_p, one_hot_max_size=16, task_type="GPU",
                                     devices="0", verbose=0, allow_writing_files=False,
                                     random_seed=seed)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                          learning_rate_init=1e-3, batch_size=256,
                                          max_iter=120, early_stopping=True,
                                          n_iter_no_change=8, random_state=seed))

    # >>> TEK FARK: fold'lar Tuna'dan (student_id ile hizali) <<<
    splits = load_folds(train[ID].values)
    # her satir birden cok repeat'te val olur -> OOF'u repeat sayisina bol
    val_count = np.zeros(len(train))
    for va in splits:
        val_count[va] += 1
    oof = {m: np.zeros(len(train)) for m in REG_NAMES}
    test_pred = {m: np.zeros(len(test)) for m in REG_NAMES}
    n_splits = len(splits)

    for si, va_idx in enumerate(splits, 1):
        tr_idx = np.setdiff1d(np.arange(len(train)), va_idx)
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
        for c in CAT_COLS:
            Xtr[c + "_te"] = target_encode(train.iloc[tr_idx][c], train.iloc[tr_idx][c], y[tr_idx], gmean)
            Xva[c + "_te"] = target_encode(train.iloc[tr_idx][c], train.iloc[va_idx][c], y[tr_idx], gmean)
            Xte[c + "_te"] = target_encode(train.iloc[tr_idx][c], test[c], y[tr_idx], gmean)
        for c, (seg_tr_full, seg_te_full) in segs.items():
            Xtr[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx], seg_tr_full[tr_idx], yr_tr[tr_idx])
            Xva[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx], seg_tr_full[va_idx], yr_tr[va_idx])
            Xte[c + "_yil_te"] = cell_te(seg_tr_full[tr_idx], yr_tr[tr_idx], y[tr_idx], seg_te_full, yr_te)

        ns = len(SEEDS)
        for seed in SEEDS:
            for n in REG_NAMES:
                m = make(n, seed)
                m.fit(Xtr, yn)
                va_p, te_p = m.predict(Xva), m.predict(Xte)
                # OOF: satir birden cok repeat'te val olabilir -> val_count'a bol
                oof[n][va_idx] += (va_p * sd_va + mu_va) / (ns * val_count[va_idx])
                test_pred[n] += (te_p * sd_te + mu_te) / (ns * n_splits)
        print(f"  split {si}/{n_splits} bitti")

    oof_all = dict(oof); test_all = dict(test_pred)
    oof_all["nn"] = np.load(find("nn_oof.npy"))
    test_all["nn"] = np.load(find("nn_test.npy"))

    print("\n=== TEKIL OOF (duz | recency-weighted) ===")
    for n in ALL_NAMES:
        p = np.clip(oof_all[n], 0, 100)
        print(f"  {n:5s}: {((y - p) ** 2).mean():8.4f} | {wmse(y, p, w_fit):8.4f}")

    # fit'siz sabit-oran (en guvenilir): esit agirlik + nn dahil basit ortalama
    M = np.column_stack([oof_all[m] for m in ALL_NAMES])
    Mte = np.column_stack([test_all[m] for m in ALL_NAMES])
    from scipy.optimize import nnls
    sw = np.sqrt(w_fit)
    wb, _ = nnls(M * sw[:, None], y * sw); wb /= wb.sum()
    ens_oof = np.clip(M @ wb, 0, 100)
    ens_test = np.clip(Mte @ wb, 0, 100)
    print(f"\n  NNLS blend rw-OOF = {wmse(y, ens_oof, w_fit):.4f}  w={dict(zip(ALL_NAMES, np.round(wb,3)))}")

    np.save("ourteam_oof_tunafolds.npy", ens_oof.astype(np.float32))
    np.save("ourteam_test_tunafolds.npy", ens_test.astype(np.float32))
    np.savez("preds_tunafolds.npz", y=y, w_fit=w_fit, years=yr_tr,
             **{f"oof_{m}": oof_all[m] for m in ALL_NAMES},
             **{f"test_{m}": test_all[m] for m in ALL_NAMES})
    print("\nKAYDEDILDI -> ourteam_oof_tunafolds.npy + ourteam_test_tunafolds.npy + preds_tunafolds.npz")
    print(">>> Bu 3 dosyayi Tuna'ya yolla; probe_compare_solutions.py ile paired-gate'i kendi zemininde calistirin.")


if __name__ == "__main__":
    main()
