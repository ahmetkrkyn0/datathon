"""
Faz 4 — Feature Engineering (Tablosal). SPEC 04 uygulanabilir karsiligi.
===========================================================================

    python src/features.py    # ablation'i kosar + artefaktlari yazar

FELSEFE (SPEC 04 §3, CLAUDE.md SIFIR OVERFIT):
  * AZ ama GUCLU feature. Her grup SADECE kabul kapisindan gecerse kalir
    (cv.acceptance_gate: yeni_cv < eski_cv - 0.25*eski_std). Public LB'ye BAKILMAZ.
  * Anchor (cv.build_structured_matrix; sayisal+yil+flag+native-kategorik) = 81.70 +/- 2.93
    -> gate tabani 81.702394 - 0.25*2.933211 = 80.969. Forward-selection: her grup,
    o ana kadar KABUL edilmis tabana gore 0.25*std ile olculur (esikte basit kazanir).
  * Determinist FE (kompozit/carpim/log1p/total/oran/yil-turevi) ham pandas, fold-BAGIMSIZ
    (hicbir istatistik ogrenmez) -> features_*.parquet'e onbelleklenir.
  * Fold-ICI olan TEK sey kategorik kodlama (OHE / OOF-TE). build_feature_pipeline()
    bunu ColumnTransformer ile uretir; SADECE dis-fold train'inden fit (run_oof sozlesmesi).
    OOF-TE = sklearn TargetEncoder (cross-fitted, leak-safe) -> GLOBAL TE degil (Guardrail 2).

CIKTILAR (SPEC 04 §8):
  * data/features_train.parquet, features_test.parquet : determinist feature onbellegi (superset).
  * reports/fe_ablation.csv : grup, cv_mse_mean, cv_mse_std, delta_vs_anchor, kabul.
  * config/feature_groups.yaml : kabul/ret bayraklari + secilen LGBM feature listesi + kategorik karar.
  * reports/adversarial_auc.txt : nihai matriste yil-disi (<0.60 monitor) ve yilli (~0.66 beklenen) AUC.

Determinizm: SEED=42; sabit kolon sirasi; LGBM deterministic=True (anchor HP).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

import cv
from anchor_lgbm_num import LGBM_PARAMS, EARLY_STOPPING_ROUNDS
from cleaning import clean_raw

# --------------------------------------------------------------------------- #
# Anchor tabani (artifacts/cv_scores.csv'de KILITLI; review C1: yillar dahil).
# Gate DAIMA bu sabit anchor'a (veya forward-selection'da KABUL edilmis tabana) gore.
# --------------------------------------------------------------------------- #
ANCHOR_MEAN = 81.702394
ANCHOR_STD = 2.933211
GATE_THRESHOLD = ANCHOR_MEAN - cv.ACCEPT_K * ANCHOR_STD  # 80.969...

# OOF target-encoding ic-kat sayisi (cross-fitting; train-fold ICINDE de leak-safe).
TE_INNER_CV = 5

CONFIG_DIR = cv.ROOT / "config"
FE_ABLATION_PATH = cv.REPORTS_DIR / "fe_ablation.csv"
FEATURE_GROUPS_PATH = CONFIG_DIR / "feature_groups.yaml"
ADVERSARIAL_AUC_PATH = cv.REPORTS_DIR / "adversarial_auc.txt"
FEATURES_TRAIN_PATH = cv.DATA_DIR / "features_train.parquet"
FEATURES_TEST_PATH = cv.DATA_DIR / "features_test.parquet"

# --------------------------------------------------------------------------- #
# Kompozit gruplari (SPEC 04 §5.1) — ham kolon adlari train basligindan birebir.
# --------------------------------------------------------------------------- #
TECH_COLS = [
    "coding_score", "problem_solving_score", "data_structures_score", "sql_score",
    "machine_learning_score", "backend_score", "frontend_score", "cloud_score", "devops_score",
]
SOFT_COLS = ["communication_score", "teamwork_score", "leadership_score", "presentation_score"]
INTERVIEW_COLS = ["technical_interview_score", "hr_interview_score"]
PROFILE_COLS = ["portfolio_score", "linkedin_profile_score", "cv_quality_score"]
# log1p sadece uzun-kuyruk sayimlar (SPEC 04 §5.3): tree-notr, Faz 05 lineer tarafi icin.
LOG1P_SRC = ["github_avg_stars", "open_source_contribution_count", "github_repo_count"]


# --------------------------------------------------------------------------- #
# Determinist FE — ham pandas, fold-BAGIMSIZ (istatistik ogrenmez -> sizinti imkansiz).
# Her fonksiyon yeni kolonlari iceren bir DataFrame doner (df ile index-hizali).
# --------------------------------------------------------------------------- #
def _composite_cols(df: pd.DataFrame) -> pd.DataFrame:
    """SPEC §5.1 kompozit ortalamalar (satir-bazli skipna mean). Olculen: tech_mean korr ~0.338."""
    return pd.DataFrame({
        "tech_mean": df[TECH_COLS].mean(axis=1, skipna=True),
        "soft_mean": df[SOFT_COLS].mean(axis=1, skipna=True),
        "interview_mean": df[INTERVIEW_COLS].mean(axis=1, skipna=True),
        "profile_mean": df[PROFILE_COLS].mean(axis=1, skipna=True),
    }, index=df.index)


def _group_std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """SPEC §5.1 grup-ici tutarsizlik (dengeli profil mi tek-yon mu). Once denenir; gecmezse cikar."""
    return pd.DataFrame({
        "tech_std": df[TECH_COLS].std(axis=1, skipna=True),
        "soft_std": df[SOFT_COLS].std(axis=1, skipna=True),
    }, index=df.index)


def _interaction_cols(df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """SPEC §5.2 anahtar carpimlar (4 baz). pq_x_tech olculen korr ~0.606 (en guclu tek feature)."""
    pq = df["project_quality_score"]
    return pd.DataFrame({
        "pq_x_tech": pq * comp["tech_mean"],
        "interview_x_tech": comp["interview_mean"] * comp["tech_mean"],
        "pq_x_soft": pq * comp["soft_mean"],
        "profile_x_interview": comp["profile_mean"] * comp["interview_mean"],
    }, index=df.index)


def _totals_ratio_cols(df: pd.DataFrame) -> pd.DataFrame:
    """SPEC §5.3 toplam aktivite + tek aday oran (kabul kapisina sokulur; gecmezse atilir).

    total_projects: open_source NA -> skipna toplamda 0 katki (medyan-impute kararini bozmaz;
    bu yalniz bir aggregate). interview_rate = interviews_attended/(applications_sent+1)."""
    return pd.DataFrame({
        "total_projects": df[[
            "real_client_project_count", "freelance_project_count", "open_source_contribution_count",
        ]].sum(axis=1, skipna=True),
        "total_credentials": df["certification_count"] + df["bootcamp_count"],
        "interview_rate": df["interviews_attended"] / (df["applications_sent"] + 1.0),
    }, index=df.index)


def _year_deriv_cols(df: pd.DataFrame) -> pd.DataFrame:
    """SPEC §5.6 yil-turevi (gated). years_since_graduation = application_year - graduation_year
    (shift-invariant); yillar zaten HAM matriste -> *ek* sinyal mi diye kabul kapisina sokulur."""
    return pd.DataFrame({
        "years_since_graduation": df["application_year"] - df["graduation_year"],
    }, index=df.index)


def _log1p_cols(df: pd.DataFrame) -> pd.DataFrame:
    """SPEC §5.3 log1p (uzun-kuyruk sayimlar). NaN -> NaN korunur (LGBM native; Faz 05 impute eder)."""
    return pd.DataFrame(
        {f"{c}_log1p": np.log1p(df[c]) for c in LOG1P_SRC}, index=df.index
    )


def build_derived(clean_df: pd.DataFrame) -> pd.DataFrame:
    """TUM determinist turetilmis kolonlarin SUPERSET'i (clean_raw ciktisindan; fold-bagimsiz).

    features_*.parquet bu superset'i onbellekler; hangi grup LGBM'e girecegi feature_groups.yaml
    (ablation sonucu) ile secilir. Faz 05 lineer tarafi log1p kolonlarini buradan okur.
    """
    comp = _composite_cols(clean_df)
    parts = [
        comp,
        _group_std_cols(clean_df),
        _interaction_cols(clean_df, comp),
        _totals_ratio_cols(clean_df),
        _year_deriv_cols(clean_df),
        _log1p_cols(clean_df),
    ]
    return pd.concat(parts, axis=1)


# Ablation tasarimi (SPEC §5.7) — DUZEN-BAGIMSIZ, kanit-temelli.
#   CORE = kompozit ortalamalar (olculen en guclu blok). Diger gruplar CORE'un USTUNE eklenip
#   anchor'a gore olculur (saf greedy degil: interactions'i ana-etkileri olmadan test etmek
#   yaniltici olur). pq_x_tech gibi carpimlar CORE main-effect'leri varken AGACA gereksizdir
#   (Guardrail 6: korr-gudumlu secim tuzagi) -> ablation bunu ACIKCA gosterir.
CORE_COLS = ["tech_mean", "soft_mean", "interview_mean", "profile_mean"]  # SPEC §5.1 kompozitler
AUG_GROUPS: list[tuple[str, list[str]]] = [
    ("group_std", ["tech_std", "soft_std"]),                                            # §5.1
    ("interactions", ["pq_x_tech", "interview_x_tech", "pq_x_soft", "profile_x_interview"]),  # §5.2
    ("totals_ratio", ["total_projects", "total_credentials", "interview_rate"]),        # §5.3
    ("year_deriv", ["years_since_graduation"]),                                          # §5.6
]
LOG1P_COLS = [f"{c}_log1p" for c in LOG1P_SRC]  # tree-notr; Faz 05 lineer tarafi icin parquet'te tutulur


# --------------------------------------------------------------------------- #
# build_feature_pipeline — SPEC 04 deliverable. Fold-ICI kategorik kodlama transformer'i.
#   native  -> None (kategorikler `category` dtype; LGBM native, kodlama yok).
#   onehot  -> OneHotEncoder(handle_unknown='ignore'), fold-ici fit.
#   oof_te  -> sklearn TargetEncoder (cross-fitted, leak-safe), fold-ici fit (GLOBAL TE DEGIL).
# --------------------------------------------------------------------------- #
def build_feature_pipeline(cat_mode: str, feature_columns, cat_cols):
    """Fold-safe `ColumnTransformer` doner (onehot/oof_te) ya da None (native).

    `fit_transform` DAIMA dis-fold train'inde cagrilir (run_oof sozlesmesi) -> sizintisiz.
    Sayisal/turetilmis kolonlar passthrough; SADECE kategorikler kodlanir.
    """
    if cat_mode == "native":
        return None
    if cat_mode not in ("onehot", "oof_te"):
        raise ValueError(f"cat_mode native/onehot/oof_te olmali, '{cat_mode}' degil.")

    cat_cols = list(cat_cols)
    num_cols = [c for c in feature_columns if c not in cat_cols]
    if cat_mode == "onehot":
        cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:  # oof_te — Bayesian smoothing ('auto' empirik-Bayes, SPEC m~20-50 ruhu), cross-fitted
        cat_tf = TargetEncoder(
            target_type="continuous", smooth="auto", cv=TE_INNER_CV, random_state=cv.SEED,
        )
    ct = ColumnTransformer(
        [("num", "passthrough", num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def make_fit_fold(cat_mode: str, feature_columns, cat_cols, lgbm_params=None):
    """fit_fold(X_tr,y_tr,X_val,y_val) -> (predict_fn, best_iteration). SADECE fold-ici veri.

    Kategorik kodlama (varsa) fold-ICI fit edilir; predict_fn ayni fold-transformer'i uygular.
    """
    params = dict(LGBM_PARAMS if lgbm_params is None else lgbm_params)
    cat_cols = list(cat_cols)

    def fit_fold(X_tr, y_tr, X_val, y_val):
        pipe = build_feature_pipeline(cat_mode, feature_columns, cat_cols)
        if pipe is None:  # native: category dtype dogrudan LGBM'e
            Xt_tr, Xt_val = X_tr, X_val
            cat_feat = cat_cols
        else:  # onehot/oof_te: fold-ici fit -> sayisal matris
            Xt_tr = pipe.fit_transform(X_tr, y_tr)
            Xt_val = pipe.transform(X_val)
            cat_feat = "auto"  # kodlama sonrasi kategorik yok

        model = LGBMRegressor(**params)
        model.fit(
            Xt_tr, y_tr,
            eval_set=[(Xt_val, y_val)],
            eval_metric="l2",
            categorical_feature=cat_feat,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        best_it = model.best_iteration_ or params["n_estimators"]

        def predict(X):
            Xt = X if pipe is None else pipe.transform(X)
            return model.predict(Xt, num_iteration=best_it)  # HAM; clip run_oof'ta

        return predict, int(best_it)

    return fit_fold


# --------------------------------------------------------------------------- #
# Matris kurma yardimcilari
# --------------------------------------------------------------------------- #
def build_feature_matrix(structured_X: pd.DataFrame, derived: pd.DataFrame, derived_cols: list[str]):
    """structured (anchor) + secilen turetilmis kolonlar -> tek matris (kategorikler korunur)."""
    if not derived_cols:
        return structured_X.reset_index(drop=True)
    extra = derived[derived_cols].reset_index(drop=True)
    return pd.concat([structured_X.reset_index(drop=True), extra], axis=1)


def _run_cfg(cat_mode, X, y, X_test, folds, sid, cat_cols):
    """Bir konfigurasyonu run_oof + compute_cv_mse ile olcer -> (mean, std)."""
    feature_columns = list(X.columns)
    out = cv.run_oof(make_fit_fold(cat_mode, feature_columns, cat_cols), X, y, X_test, folds, sid)
    mean, std, _ = cv.compute_cv_mse(out["oof"], y, folds, sid)
    return mean, std


# --------------------------------------------------------------------------- #
# Adversarial kayma-monitoru (SPEC §5.7 adim 4). train=0/test=1 siniflandirici AUC.
# Yilli tam matris ~0.66 BEKLENEN (review C1, alarm degil); yil-disi <0.60 monitor.
# --------------------------------------------------------------------------- #
def adversarial_auc(X_tr: pd.DataFrame, X_te: pd.DataFrame, cat_cols, n_splits: int = 5) -> float:
    """Birlesik train/test uzerinde stratified-KFold LGBM-classifier AUC ortalamasi (tani)."""
    cat_cols = [c for c in cat_cols if c in X_tr.columns]
    Xc = pd.concat([X_tr.reset_index(drop=True), X_te.reset_index(drop=True)], axis=0, ignore_index=True)
    label = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv.SEED)
    aucs = []
    for tr_idx, va_idx in skf.split(Xc, label):
        clf = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=cv.SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbosity=-1,
        )
        clf.fit(
            Xc.iloc[tr_idx], label[tr_idx],
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[lgb.log_evaluation(0)],
        )
        p = clf.predict_proba(Xc.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(label[va_idx], p))
    return float(np.mean(aucs))


# --------------------------------------------------------------------------- #
# Ablation surucusu + artefakt yazimi (SPEC §5.7)
# --------------------------------------------------------------------------- #
def main() -> None:
    cv.set_seed()

    raw_train = cv.load_train()
    raw_test = cv.load_test()
    folds = cv.load_folds()
    sid = raw_train[cv.ID_COL].values
    y = raw_train[cv.TARGET_COL].values

    clean_tr = clean_raw(raw_train)
    clean_te = clean_raw(raw_test)

    # Anchor (yapisal) matris — cv.build_structured_matrix (anchor ile BIREBIR ayni).
    cat_dtypes = cv.structured_cat_dtypes(raw_train)
    X_struct, cat_cols = cv.build_structured_matrix(raw_train, cat_dtypes)
    Xt_struct, _ = cv.build_structured_matrix(raw_test, cat_dtypes)
    assert list(X_struct.columns) == list(Xt_struct.columns)

    # Determinist turetilmis superset (parquet onbellegi + ablation kaynagi).
    derived_tr = build_derived(clean_tr)
    derived_te = build_derived(clean_te)
    assert list(derived_tr.columns) == list(derived_te.columns)
    print(f"[fe] anchor struct={X_struct.shape[1]} feature; derived superset={derived_tr.shape[1]} kolon")
    print(f"[fe] gate tabani: anchor {ANCHOR_MEAN:.4f} +/- {ANCHOR_STD:.4f} -> esik {GATE_THRESHOLD:.4f}")

    rows: list[dict] = []

    def gate(mean: float) -> bool:
        """Kabul kapisi: matris anchor'i 0.25*anchor_std ile gecmeli (< 80.969)."""
        return cv.acceptance_gate(mean, ANCHOR_MEAN, ANCHOR_STD)

    def record(group, mean, std, cols, note=""):
        keep = gate(mean)
        rows.append(dict(
            group=group, cv_mse_mean=round(mean, 6), cv_mse_std=round(std, 6),
            delta_vs_anchor=round(mean - ANCHOR_MEAN, 6), kabul=bool(keep), note=note,
        ))
        flag = "KABUL" if keep else "RET  "
        print(f"[fe] {flag} {group:<18} cv={mean:7.4f} +/- {std:5.4f}  d_anchor={mean-ANCHOR_MEAN:+7.4f}")
        return mean, std, cols, keep

    # --- G0: anchor (native) yeniden-uretim (DoD-2 reproducibility) ---
    g0_mean, g0_std = _run_cfg("native", X_struct, y, Xt_struct, folds, sid, cat_cols)
    repro_ok = abs(g0_mean - ANCHOR_MEAN) < 0.50
    rows.append(dict(group="anchor", cv_mse_mean=round(g0_mean, 6), cv_mse_std=round(g0_std, 6),
                     delta_vs_anchor=round(g0_mean - ANCHOR_MEAN, 6), kabul=True,
                     note=f"reproduce(anchor 81.70) ok={repro_ok}"))
    print(f"[fe] BASE  {'anchor':<18} cv={g0_mean:7.4f} +/- {g0_std:5.4f}  (yeniden-uretim, repro_ok={repro_ok})")
    assert repro_ok, f"Anchor yeniden-uretim sapti: {g0_mean:.4f} vs {ANCHOR_MEAN:.4f} (CV altyapisi/HP incele)."

    # candidates: name -> (mean, std, derived_cols, kabul). Anchor zaten taban.
    candidates: dict[str, tuple] = {"anchor": (g0_mean, g0_std, [], False)}

    # --- CORE = kompozitler (olculen en guclu blok); anchor'a gore ---
    m, s = _run_cfg("native", build_feature_matrix(X_struct, derived_tr, CORE_COLS), y,
                    build_feature_matrix(Xt_struct, derived_te, CORE_COLS), folds, sid, cat_cols)
    candidates["composites"] = record("composites", m, s, list(CORE_COLS),
                                      note="SPEC §5.1 kompozitler; FE tezinin cekirdegi")

    # --- Augmentasyonlar: her biri CORE'un USTUNE (ana-etkiler mevcutken), anchor'a gore ---
    for name, cols in AUG_GROUPS:
        trial_cols = CORE_COLS + cols
        m, s = _run_cfg("native", build_feature_matrix(X_struct, derived_tr, trial_cols), y,
                        build_feature_matrix(Xt_struct, derived_te, trial_cols), folds, sid, cat_cols)
        note = ("§5.2 carpimlar — CORE main-effect'leri varken AGACA gereksiz (Guardrail 6)"
                if name == "interactions" else f"CORE + {name}")
        candidates[f"composites+{name}"] = record(f"composites+{name}", m, s, list(trial_cols), note=note)

    # --- log1p: tree-notr (SPEC §5.3). LGBM kapisinda olc (beklenen RET) ama Faz 05 lineer
    #     tarafi icin parquet'te DAIMA tutulur (yaml: log1p_for_linear). ---
    trial_cols = CORE_COLS + LOG1P_COLS
    m, s = _run_cfg("native", build_feature_matrix(X_struct, derived_tr, trial_cols), y,
                    build_feature_matrix(Xt_struct, derived_te, trial_cols), folds, sid, cat_cols)
    candidates["composites+log1p"] = record("composites+log1p", m, s, list(trial_cols),
                                            note="tree-notr; Faz05 lineer icin parquet'te tutulur")

    # --- SECIM: gate'i gecen en dusuk-CV native matris; hicbiri gecmezse ANCHOR (Occam) ---
    passing = {k: v for k, v in candidates.items() if v[3]}  # kabul=True olanlar
    if passing:
        sel_name = min(passing, key=lambda k: passing[k][0])
    else:
        sel_name = "anchor"
    sel_mean, sel_std, accepted_cols, _ = candidates[sel_name]
    accepted_groups = [] if sel_name == "anchor" else [sel_name]
    # En guclu aday (gate'i gecmese de) — Faz 06'da NLP-zenginlestirilmis tabanda yeniden test icin.
    det_only = {k: v for k, v in candidates.items() if k != "anchor"}
    best_candidate = min(det_only, key=lambda k: det_only[k][0]) if det_only else "anchor"
    print(f"[fe] secilen native matris: {sel_name} (cv={sel_mean:.4f}); "
          f"en guclu aday: {best_candidate} (cv={candidates[best_candidate][0]:.4f}, "
          f"gate {'GECTI' if candidates[best_candidate][3] else 'gecemedi'})")

    # --- Kategorik kodlama ablasyonu (SPEC §5.5): secilen matris uzerinde OHE / OOF-TE vs native ---
    X_final = build_feature_matrix(X_struct, derived_tr, accepted_cols)
    Xt_final = build_feature_matrix(Xt_struct, derived_te, accepted_cols)
    cat_decision, cat_decision_mean = "native", sel_mean
    for cat_mode in ("onehot", "oof_te"):
        m, s = _run_cfg(cat_mode, X_final, y, Xt_final, folds, sid, cat_cols)
        # Kategorik kodlama native incumbent'i 0.25*std ile gecmeli (yoksa native: basit+sizintisiz).
        keep_cat = cv.acceptance_gate(m, sel_mean, sel_std)
        rows.append(dict(group=f"cat_{cat_mode}", cv_mse_mean=round(m, 6), cv_mse_std=round(s, 6),
                         delta_vs_anchor=round(m - ANCHOR_MEAN, 6), kabul=bool(keep_cat),
                         note=f"vs native incumbent ({sel_name})"))
        flag = "KABUL" if keep_cat else "RET  "
        print(f"[fe] {flag} {'cat_'+cat_mode:<18} cv={m:7.4f} +/- {s:5.4f}  d_anchor={m-ANCHOR_MEAN:+7.4f}")
        if keep_cat and m < cat_decision_mean:
            cat_decision, cat_decision_mean = cat_mode, m
    best_mean, best_std = sel_mean, sel_std
    print(f"[fe] kategorik kodlama karari: {cat_decision} "
          f"(native incumbent; OHE/OOF-TE ancak 0.25*std gecerse secilir)")

    # --- Nihai matris adversarial kayma-monitoru (yilli ~0.66 beklenen; yil-disi <0.60) ---
    auc_full = adversarial_auc(X_final, Xt_final, cat_cols)
    year_like = list(cv.YEAR_COLS) + [c for c in ("years_since_graduation",) if c in X_final.columns]
    X_noyear = X_final.drop(columns=[c for c in year_like if c in X_final.columns])
    Xt_noyear = Xt_final.drop(columns=[c for c in year_like if c in Xt_final.columns])
    auc_noyear = adversarial_auc(X_noyear, Xt_noyear, cat_cols)
    print(f"[fe] adversarial AUC: yilli={auc_full:.4f} (~0.66 beklenen)  "
          f"yil-disi={auc_noyear:.4f} (<0.60 saglikli)")

    # ======================= Artefakt yazimi ======================= #
    cv.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) fe_ablation.csv
    pd.DataFrame(rows, columns=[
        "group", "cv_mse_mean", "cv_mse_std", "delta_vs_anchor", "kabul", "note",
    ]).to_csv(FE_ABLATION_PATH, index=False)

    # 2) features_*.parquet — determinist SUPERSET (anchor struct + tum turetilmis) + student_id.
    #    Kategorikler `category` dtype korunur (build_structured_matrix evreni).
    feats_tr = pd.concat([X_struct.reset_index(drop=True), derived_tr.reset_index(drop=True)], axis=1)
    feats_te = pd.concat([Xt_struct.reset_index(drop=True), derived_te.reset_index(drop=True)], axis=1)
    feats_tr.insert(0, cv.ID_COL, raw_train[cv.ID_COL].values)
    feats_te.insert(0, cv.ID_COL, raw_test[cv.ID_COL].values)
    feats_tr.to_parquet(FEATURES_TRAIN_PATH, index=False)
    feats_te.to_parquet(FEATURES_TEST_PATH, index=False)

    # 3) feature_groups.yaml — secim bayraklari + LGBM feature listesi + kategorik karar.
    lgbm_feature_columns = list(X_final.columns)  # struct + KABUL edilen turetilmis (secilen native)
    final_mean, final_std = best_mean, best_std
    bc_mean, bc_std, bc_cols, bc_keep = candidates[best_candidate]
    groups_meta = {
        "seed": cv.SEED,
        "anchor": {"cv_mse_mean": round(g0_mean, 6), "cv_mse_std": round(g0_std, 6),
                   "locked_reference": {"mean": ANCHOR_MEAN, "std": ANCHOR_STD}},
        "gate": {"k": cv.ACCEPT_K, "threshold": round(GATE_THRESHOLD, 6),
                 "rule": "matris ancak cv < anchor - 0.25*anchor_std ise KABUL (duzen-bagimsiz; Occam)"},
        "selected_matrix": sel_name,
        "accepted_groups": accepted_groups,
        "rejected_groups": [r["group"] for r in rows
                            if not r["kabul"] and r["group"] not in ("anchor",)
                            and not r["group"].startswith("cat_")],
        # En guclu determinist aday (gate'i gecmese de): Faz 06'da NLP-zenginlestirilmis
        # tabanda yeniden test edilir (SPEC §9 DoD-2: FE-tek-basina dususu marjinal olabilir).
        "best_candidate": {"name": best_candidate, "cv_mse_mean": round(bc_mean, 6),
                           "cv_mse_std": round(bc_std, 6), "delta_vs_anchor": round(bc_mean - ANCHOR_MEAN, 6),
                           "gate_passed": bool(bc_keep), "derived_columns": list(bc_cols),
                           "reeval_in": "faz06 (NLP-augmented base)"},
        "categorical_encoding": cat_decision,
        "final_cv": {"cv_mse_mean": round(final_mean, 6), "cv_mse_std": round(final_std, 6),
                     "delta_vs_anchor": round(final_mean - ANCHOR_MEAN, 6)},
        "lgbm_feature_columns": lgbm_feature_columns,
        "categorical_columns": list(cat_cols),
        "accepted_derived_columns": list(accepted_cols),
        "log1p_for_linear": LOG1P_COLS,  # Faz 05 lineer tarafi (tree-notr; LGBM setine girmez)
        "derived_columns_all": list(derived_tr.columns),  # parquet superset
        "adversarial_auc": {"with_years": round(auc_full, 4), "without_years": round(auc_noyear, 4)},
    }
    with open(FEATURE_GROUPS_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(groups_meta, fh, sort_keys=False, allow_unicode=True)

    # 4) adversarial_auc.txt
    ADVERSARIAL_AUC_PATH.write_text(
        f"Faz 4 nihai feature matrisi — adversarial kayma-monitoru (train=0/test=1, 5-fold LGBM-clf AUC)\n"
        f"with_years    = {auc_full:.4f}   (review C1: ~0.66 BEKLENEN — kovaryat-kayma dedektoru, zarar degil)\n"
        f"without_years = {auc_noyear:.4f}   (<0.60 saglikli; >0.60 -> yillar disinda yeni kayma, suclu feature incele)\n"
        f"karar: {'OK — yil-disi <0.60' if auc_noyear < 0.60 else 'INCELE — yil-disi >=0.60'}\n",
        encoding="utf-8",
    )

    # --- Ozet ---
    print(f"\n[fe] === OZET ===")
    print(f"[fe] kabul gruplar: {accepted_groups or '(yok)'}")
    print(f"[fe] nihai CV: {final_mean:.4f} +/- {final_std:.4f}  "
          f"(delta_vs_anchor {final_mean - ANCHOR_MEAN:+.4f}; esik {GATE_THRESHOLD:.4f})")
    print(f"[fe] LGBM feature sayisi: {len(lgbm_feature_columns)}  (kategorik kodlama: {cat_decision})")
    print(f"[fe] yazildi: {FE_ABLATION_PATH.name}, {FEATURE_GROUPS_PATH.name}, "
          f"{FEATURES_TRAIN_PATH.name}/{FEATURES_TEST_PATH.name}, {ADVERSARIAL_AUC_PATH.name}")
    gain = ANCHOR_MEAN - final_mean
    if final_mean < GATE_THRESHOLD:
        print(f"[fe] SONUC: FE matris anchor'i ANLAMLI gecti ({gain:+.4f} MSE, esik altinda). DoD-2 OK.")
    else:
        print(f"[fe] SONUC: FE matris esigi gecemedi ({gain:+.4f} MSE) -> yalniz kabul gecen gruplar tutuldu "
              f"(Occam: marjinal kazanc REDDEDILDI). Anchor referans korunur.")


if __name__ == "__main__":
    main()
