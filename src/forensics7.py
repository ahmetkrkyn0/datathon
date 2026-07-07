"""
FORENSICS PART 7 — C3 reversal follow-up: is text's value interaction-driven, and is there
recoverable HEADROOM a richer text encoder could capture? + inter-model noise-floor confirmation.
==================================================================================================
Adversarial verify C3'u curuttu: metin REDUNDANT DEGIL — joint num+text (-5.24) num-only'den iyi
ve additive'den (-4.38) iyi -> deger ETKILESIMLI. Bu, BERT'i ELEME GEREKCESINI gecersiz kilar.

Bu script:
1. ETKILESIM TEYIDI: txt_oof'u num ile etkilesim olarak vermek (txt * num) marjinal kazanc verir mi?
   Yani metnin numerigi MODULE etmesi olcum altinda mi?
2. HEADROOM: mevcut txt_ridge'in metinden cikardigi sinyal tavani ne? num+txt_ridge vs
   num+(daha-zengin-tfidf). Zengin tfidf ek kazanc veriyorsa BERT plauisible.
3. NOISE-FLOOR (inter-model): bagimsiz iki GBDT (lgbm vs catboost) ne kadar AYNI? disagreement
   std << residual std ise kalan residual gercek gurultu (model-deficiency degil).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv  # noqa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; ART = ROOT / "artifacts"; REPORTS = ROOT / "reports"
SEED = 42
TARGET, ID, TEXT = "career_success_score", "student_id", "mentor_feedback_text"
CAT = ["department", "university_tier", "target_role", "hobby", "preferred_social_media_platform"]


def numeric_cols(df):
    drop = {ID, TARGET, TEXT, *CAT}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def tr_lower(s):
    return s.str.replace("I", "ı").str.replace("İ", "i").str.lower()


def main():
    np.random.seed(SEED)
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    te = pd.read_csv(DATA / "test_x.csv", encoding="utf-8-sig")
    y = tr[TARGET].values.astype(float)
    n = len(y)
    num = numeric_cols(tr)
    dump = {}

    import lightgbm as lgb
    from sklearn.model_selection import KFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- 3. INTER-MODEL NOISE FLOOR (gercek artefaktlar) ----
    print("== 3. INTER-MODEL DISAGREEMENT (noise-floor teyidi) ==")
    lgbm = np.clip(np.load(ART / "oof_lgbm_full.npy"), 0, 100)
    cat = np.clip(np.load(ART / "oof_catboost_full.npy"), 0, 100)
    disagree = np.std(lgbm - cat)
    resid_std = np.std(y - 0.5 * (lgbm + cat))
    print(f"  lgbm vs catboost disagreement std = {disagree:.4f}")
    print(f"  ensemble residual std = {resid_std:.4f}")
    print(f"  -> iki bagimsiz GBDT ailesi {disagree:.2f} icinde HEMFIKIR; residual {resid_std:.2f}.")
    print(f"     residual'in ~%{100*(1-(disagree/resid_std)**2):.1f}'i model-bagimsiz (irreducible gurultu sinyali).")
    dump["intermodel_disagree_std"] = float(disagree)
    dump["ensemble_resid_std"] = float(resid_std)
    dump["irreducible_frac_est"] = float(1 - (disagree / resid_std) ** 2)

    # ---- 1+2. TEXT HEADROOM: zengin tfidf ek kazanc veriyor mu? ----
    print("\n== 1/2. TEXT HEADROOM: txt_ridge (proje) vs ZENGIN tfidf, num tabaninda ==")
    txt = tr_lower(tr[TEXT].astype(str))
    # mevcut proje txt_ridge OOF
    proj_txt = np.clip(np.load(ART / "oof_txt_ridge.npy"), 0, 100)
    # ZENGIN tfidf (word 1-3 + char 3-6, daha buyuk) OOF, fold-safe
    rich_txt = np.zeros(n)
    for tri, vai in kf.split(txt):
        wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, max_features=80000, sublinear_tf=True)
        cvz = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, max_features=80000, sublinear_tf=True)
        Xtr = hstack([wv.fit_transform(txt.iloc[tri]), cvz.fit_transform(txt.iloc[tri])]).tocsr()
        Xva = hstack([wv.transform(txt.iloc[vai]), cvz.transform(txt.iloc[vai])]).tocsr()
        rg = Ridge(alpha=2.0, random_state=SEED).fit(Xtr, y[tri])
        rich_txt[vai] = rg.predict(Xva)
    rich_txt_c = np.clip(rich_txt, 0, 100)
    print(f"  proje txt_ridge OOF MSE (uw)   = {np.mean((y-proj_txt)**2):.4f}")
    print(f"  ZENGIN tfidf  OOF MSE (uw)     = {np.mean((y-rich_txt_c)**2):.4f}")
    print(f"  corr(proj, rich) = {np.corrcoef(proj_txt, rich_txt)[0,1]:.4f}")

    # num tabani + her metin meta'sini ayri ayri LGBM ile olc (etkilesim altinda)
    Xnum = tr[num].copy()
    params = dict(n_estimators=2500, learning_rate=0.02, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                  random_state=SEED, n_jobs=4, verbosity=-1)

    def oof_with(meta_dict):
        X = Xnum.copy()
        for k, v in meta_dict.items():
            X[k] = v
        oof = np.zeros(n)
        for tri, vai in kf.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])], eval_metric="l2",
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration_)
        return float(np.mean((y - np.clip(oof, 0, 100)) ** 2))

    mse_num = oof_with({})
    mse_proj = oof_with({"__txt": proj_txt})
    mse_rich = oof_with({"__txt": rich_txt})
    mse_both = oof_with({"__txt": proj_txt, "__txt2": rich_txt})
    print(f"\n  num-only            OOF MSE = {mse_num:.4f}")
    print(f"  num + proj_txt      OOF MSE = {mse_proj:.4f}  ({mse_proj-mse_num:+.4f})")
    print(f"  num + rich_txt      OOF MSE = {mse_rich:.4f}  ({mse_rich-mse_num:+.4f})")
    print(f"  num + proj + rich   OOF MSE = {mse_both:.4f}  (rich'in proj USTUNE marjinali: {mse_both-mse_proj:+.4f})")
    print(f"  -> rich'in proj ustune ek kazanci {mse_both-mse_proj:+.4f}: metin sinyali TF-IDF ile DOYMUS mu?")
    dump["text_headroom"] = dict(mse_num=mse_num, mse_proj=mse_proj, mse_rich=mse_rich, mse_both=mse_both,
                                 rich_marginal_over_proj=mse_both - mse_proj)

    (REPORTS / "forensics_dump7.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print("\n[forensics7] dump -> reports/forensics_dump7.json")


if __name__ == "__main__":
    main()
