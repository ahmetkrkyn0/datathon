"""
Faz 6 — LEVER2: ==100 iki-asama. P(y>=100) ile blend'i ust-kutleye dogru ceken POST-PROCESS.
==============================================================================================

    python src/two_stage_p100.py

NEDEN: Hedefin ~%7.73'u TAM 100 (sansurlu ust kutle). Regresyon blend bu satirlari ortalamaya
  ceker (under-predict). ORACLE (true-100 -> 100) rw-OOF 85.49->81.92 (tavan 3.57). Iki-asama:
  asama-1 P(y>=100) siniflandirici (FULL matris, AYNI foldlar, nested-OOF) -> asama-2 birlesim
  blend'i p ile 100'e dogru iter.

BIRLESIM (tek-parametre aile, alpha nested secilir): pred'(a) = blend + a*p*(100-blend).
  a=0 -> saf blend; a=1 -> pred' = p*100 + (1-p)*blend (promptun naif formu). alpha bir hucre
  DISINDAKI satirlardan nested secilir (meta-overfit yok); a*p ust ve alt sinir [0, ~1] -> clip[0,100].

KARAR = nested recency-weighted OOF + CLAUDE.md 0.25*std kapisi (blend std=3.0238 -> esik ~0.756).
  Kazanc 85.4945 - 0.756 = 84.7385'in ALTINA INMELI; aksi halde gurultu -> REDDEDILIR (Occam).
  REALITE: oracle 3.57 MUKEMMEL sinifsayicidir; gercek sinifsayici false-positive'leri de 100'e
  iter (o satirlara zarar) -> gercek kazanc tavandan cok dusuktur; kapi durust karar verir.

FOLD-SAFE: p100 cv.run_oof ile (her satir o satiri GORMEYEN modelden); test = 15-fold bagging.
  alpha nested-meta CV (ensemble.nested mantigiyla AYNI: her (repeat,fold) hucresi disindan fit).
Determinizm: LGBM deterministic=True, n_jobs=1, seed=42; AYNI folds.parquet.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier

import artifacts_io as aio
import cv
import text_utils as tu
from lgbm_full import OOF_TXT_PATH, TEST_TXT_PATH, _add_text

MODEL = "blend_p100"  # iki-asama uygulanmis blend (aday final SUB-2 yerine gececek mi -> kapi)

# P(y>=100) siniflandirici — anchor felsefesi (muhafazakar, HP taramasi yok), regresyonla esdeger karmasiklik.
CLF_PARAMS = dict(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=cv.SEED,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 100
ALPHA_GRID = np.linspace(0.0, 1.5, 16)  # 0..1.5; a=1 naif form, >1 asiri-itme (genelde kotu)


def make_fit_fold_clf(cat_features):
    """fit_fold(X_tr,y_tr_bin,X_val,y_val_bin) -> (predict_proba_fn, best_it). y HEDEFI BINARY (y>=100)."""
    def fit_fold(X_tr, y_tr, X_val, y_val):
        m = LGBMClassifier(**CLF_PARAMS)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        best_it = m.best_iteration_ or CLF_PARAMS["n_estimators"]

        def predict(X):
            return m.predict_proba(X, num_iteration=best_it)[:, 1]  # P(y>=100), [0,1]

        return predict, int(best_it)

    return fit_fold


def combine(blend, p, alpha):
    """pred'(alpha) = blend + alpha*p*(100-blend); clip[0,100] (tek kaynak)."""
    return cv.clip_predictions(blend + alpha * p * (100.0 - blend))


def nested_alpha_rw(blend, p, y, w, folds, sid):
    """alpha'yi her (repeat,fold) hucresi DISINDAN sec, hucreyi o alpha ile tahmin et -> nested rw-OOF.
    Doner: (nested_rw, meta_oof, alpha_per_cell ozet)."""
    n = len(y)
    s = np.zeros(n); c = np.zeros(n)
    chosen = []
    for r in range(cv.N_REPEATS):
        fold_of = cv.fold_of_rows(folds, sid, r)
        for g in range(cv.N_SPLITS):
            va = np.where(fold_of == g)[0]
            tr = np.where(fold_of != g)[0]
            # tr uzerinde en iyi alpha (recency-weighted)
            best_a, best_m = 0.0, np.inf
            for a in ALPHA_GRID:
                m = cv.compute_recency_weighted_mse(combine(blend[tr], p[tr], a), y[tr], w[tr])
                if m < best_m:
                    best_m, best_a = m, a
            s[va] += combine(blend[va], p[va], best_a); c[va] += 1.0
            chosen.append(best_a)
    assert np.all(c == cv.N_REPEATS)
    meta_oof = cv.clip_predictions(s / c)
    return cv.compute_recency_weighted_mse(meta_oof, y, w), meta_oof, chosen


def main() -> None:
    cv.set_seed()
    train = cv.load_train(); test = cv.load_test(); folds = cv.load_folds()
    y = train[cv.TARGET_COL].values; sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    blend_oof = np.load(cv.ARTIFACTS_DIR / "oof_blend.npy")
    blend_test = np.load(cv.ARTIFACTS_DIR / "test_blend.npy")
    base_rw = cv.compute_recency_weighted_mse(blend_oof, y, w)
    gate_std = float(__import__("pandas").read_csv(cv.REPORTS_DIR / "model_scores.csv")
                     .set_index("model").loc["blend", "cv_mse_std"])
    gate = base_rw - cv.ACCEPT_K * gate_std
    print(f"[p100] blend rw-OOF={base_rw:.4f}; kabul kapisi (0.25*std={gate_std:.4f}) -> ALTINA INMELI: {gate:.4f}")

    # --- FULL matris (lgbm_full ile AYNI) ---
    oof_txt = np.load(OOF_TXT_PATH); test_txt = np.load(TEST_TXT_PATH)
    lex_tr = tu.extract_handcrafted_features(train[cv.TEXT_COL].values)
    lex_te = tu.extract_handcrafted_features(test[cv.TEXT_COL].values)
    cat_dtypes = cv.structured_cat_dtypes(train)
    X_struct, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    Xt_struct, _ = cv.build_structured_matrix(test, cat_dtypes)
    X = _add_text(X_struct, oof_txt, lex_tr)
    X_test = _add_text(Xt_struct, test_txt, lex_te)

    # --- asama-1: P(y>=100) nested-OOF (AYNI foldlar; run_oof y_binary ile) ---
    y_bin = (y >= 100.0).astype(int)
    out = cv.run_oof(make_fit_fold_clf(cat_features), X, y_bin, X_test, folds, sid)
    p_oof, p_test = out["oof"], out["test"]  # run_oof clip[0,100] uygular ama proba [0,1] -> etkisiz
    # guvenlik: olasilik [0,1] (clip[0,100] proba'yi degistirmez ama acik tutuyoruz)
    p_oof = np.clip(p_oof, 0.0, 1.0); p_test = np.clip(p_test, 0.0, 1.0)
    auc = _safe_auc(y_bin, p_oof)
    print(f"[p100] asama-1 P(y>=100) OOF: mean={p_oof.mean():.4f} AUC={auc:.4f} "
          f"(pos-rate {y_bin.mean():.4f}); test mean={p_test.mean():.4f}")

    # --- asama-2: nested alpha + birlesim ---
    nested_rw, meta_oof, chosen = nested_alpha_rw(blend_oof, p_oof, y, w, folds, sid)
    # final test: alpha tum-OOF'tan (held-out yok) — standart; sonra test'e uygula
    best_a_full, best_m_full = 0.0, np.inf
    for a in ALPHA_GRID:
        m = cv.compute_recency_weighted_mse(combine(blend_oof, p_oof, a), y, w)
        if m < best_m_full:
            best_m_full, best_a_full = m, a
    final_test = combine(blend_test, p_test, best_a_full)

    delta = nested_rw - base_rw
    passed = cv.acceptance_gate(nested_rw, base_rw, gate_std)
    print(f"[p100] nested alpha sec (hucre disindan): ortalama a={np.mean(chosen):.3f} "
          f"(dagilim min={min(chosen):.2f} max={max(chosen):.2f})")
    print(f"[p100] full-OOF en iyi alpha={best_a_full:.3f} (naif a=1 ile karsilastir)")
    print(f"[p100] NESTED rw-OOF(iki-asama)={nested_rw:.4f}  (blend {base_rw:.4f}; delta {delta:+.4f})")
    print(f"[p100] >>> KABUL KAPISI: {'GECTI (degerli)' if passed else 'GECMEDI (gurultu/zarar -> REDDEDILDI, Occam)'}")

    aio.save_oof_test(MODEL, meta_oof, final_test)
    note = (
        f"blend_p100 = iki-asama P(y>=100) post-process [pred'=blend+a*p*(100-blend), a nested]. "
        f"asama-1 LGBM-binary OOF AUC={auc:.4f}. nested rw-OOF={nested_rw:.4f} (blend {base_rw:.4f}, "
        f"delta {delta:+.4f}; oracle tavan 81.92). kabul kapisi ({gate:.4f}) "
        f"{'GECTI' if passed else 'GECMEDI -> REDDEDILDI (Occam/sifir-overfit)'}. full-OOF alpha={best_a_full:.3f}."
    )
    aio.log_model_score(MODEL, float("nan"), gate_std, nested_rw, weighted_training=False, note=note)
    cv.assert_in_range(meta_oof, "oof_blend_p100"); cv.assert_in_range(final_test, "test_blend_p100")
    print(f"[p100] artefakt+ledger yazildi (oof/test_blend_p100.npy). KARAR: rw-OOF nested.")


def _safe_auc(y_bin, p):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_bin, p))
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
