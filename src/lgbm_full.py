"""
Faz 6 — lgbm_full: yapisal anchor + NLP (Faz 05 FULL base matris) tek GBDT.
=========================================================================

    python src/lgbm_full.py

FEATURE UZAYI (Faz 05 nlp.py ile BIREBIR; reports/nlp_ablation.csv "num+txt_ridge+lexicon"):
  lgbm_full = lgbm_num (sayisal+YIL+native-kategorik+missing-flag) + txt_ridge_pred + lexicon(10)
  Faz 05 bu assembly'i kurdu; burada AYNI parcalar yeniden birlestirilir (kanit: ayni CV ~77.03).

  * txt_ridge_pred FOLD-SAFE: TRAIN = artifacts/oof_txt_ridge.npy (nested-OOF; her satir o satiri
    GORMEYEN modelden), TEST = artifacts/test_txt_ridge.npy (15-fold-bagged). Bu nested-OOF stacking
    klasik sizintiyi yapisal olarak engeller (text_utils.build_tfidf_ridge_oof, SPEC 05 §6).
  * lexicon (Katman B, 10 ozellik): hedef-bagimsiz + fold-bagimsiz -> deterministik yeniden uretim
    (text_utils.extract_handcrafted_features); sizinti yok.
  * Model = ANCHOR LightGBM (anchor_lgbm_num.LGBM_PARAMS + make_fit_fold), AYNI 15-fold (run_oof),
    fold-ici early stopping, native kategorik. Test = KANONIK fold-bagging (15 model).

REFERANS (review C1 zinciri): anchor lgbm_num 81.70 -> +metin (FULL) 77.03 (Faz05 kapi GECTI).
  Co-headline: compute_recency_weighted_mse (test recency-yogun -> private-DURUST tahmin, review H1).

DoD-4: oof_lgbm_full.npy'den yeniden hesap +/-1e-6 esle; oof+test clip[0,100].
Determinizm: SEED=42, deterministic=True, n_jobs=1 (anchor HP).
"""

from __future__ import annotations

import numpy as np

import artifacts_io as aio
import cv
import text_utils as tu
from anchor_lgbm_num import make_fit_fold  # AYNI LGBM_PARAMS + fold-ici early stopping

MODEL = "lgbm_full"

# Faz 05 nested-OOF txt_ridge artefaktlari (fold-safe, satir-hizali folds.parquet'e).
OOF_TXT_PATH = cv.ARTIFACTS_DIR / "oof_txt_ridge.npy"
TEST_TXT_PATH = cv.ARTIFACTS_DIR / "test_txt_ridge.npy"

# Faz 05 OLCULEN FULL referans bandi (reports/nlp_ablation.csv: 77.0337). Sapma -> incele.
REF_LO, REF_HI = 75.0, 79.0


def _add_text(structured_X, txt_pred, lex_df):
    """Anchor yapisal matrise txt_ridge_pred + Katman B kolonlarini ekler (nlp.py._add_text ile ayni
    siralama: once txt_ridge_pred, sonra lexicon). Kategorikler korunur (numeric eklemeler)."""
    out = structured_X.reset_index(drop=True).copy()
    out["txt_ridge_pred"] = np.asarray(txt_pred, dtype=float)
    L = lex_df.reset_index(drop=True)
    for c in L.columns:
        out[c] = L[c].to_numpy()
    return out


def main() -> None:
    cv.set_seed()

    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    # --- txt_ridge fold-safe kolon (Faz 05 nested-OOF artefaktlari) ---
    oof_txt = np.load(OOF_TXT_PATH)
    test_txt = np.load(TEST_TXT_PATH)
    assert len(oof_txt) == len(train), f"oof_txt {len(oof_txt)} != train {len(train)}"
    assert len(test_txt) == len(test), f"test_txt {len(test_txt)} != test {len(test)}"
    cv.assert_in_range(oof_txt, "oof_txt_ridge")
    cv.assert_in_range(test_txt, "test_txt_ridge")

    # --- lexicon (Katman B) — hedef/fold-bagimsiz, deterministik yeniden uretim ---
    lex_tr = tu.extract_handcrafted_features(train[cv.TEXT_COL].values)
    lex_te = tu.extract_handcrafted_features(test[cv.TEXT_COL].values)

    # --- yapisal anchor matris (sabit kategori evreni; native-kategorik hizali) ---
    cat_dtypes = cv.structured_cat_dtypes(train)
    X_struct, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    Xt_struct, cat_features_te = cv.build_structured_matrix(test, cat_dtypes)
    assert list(X_struct.columns) == list(Xt_struct.columns) and cat_features == cat_features_te

    # --- FULL matris = num + txt_ridge + lexicon (Faz 05 base; 77.03 olculen) ---
    X = _add_text(X_struct, oof_txt, lex_tr)
    X_test = _add_text(Xt_struct, test_txt, lex_te)
    assert list(X.columns) == list(X_test.columns), "FULL train/test kolonlari hizali degil."
    print(f"[lgbm_full] {MODEL}: {X.shape[1]} feature "
          f"(= {X_struct.shape[1]} yapisal + 1 txt_ridge_pred + {lex_tr.shape[1]} lexicon). "
          f"{len(cat_features)} native-kategorik.")

    # --- 15-fold sizintisiz OOF + kanonik fold-bagging test (anchor config) ---
    out = cv.run_oof(make_fit_fold(cat_features), X, y, X_test, folds, sid)
    oof, test_pred = out["oof"], out["test"]
    best_iters = out["best_iterations"]
    genuine = out["genuine_fold_mse"]

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    # Co-headline (review H1): recency-agirlikli OOF-MSE = private-durust tahmin.
    rw = cv.recency_weights(train, test)
    recency_mse = cv.compute_recency_weighted_mse(oof, y, rw)
    best_iter_mean = float(np.mean([b for b in best_iters if b is not None]))
    single5fold_std = float(np.std(genuine[: cv.N_SPLITS]))

    note = (
        "lgbm_full = lgbm_num + txt_ridge_pred(nested-OOF) + lexicon(10) (Faz05 FULL base, "
        "reports/nlp_ablation.csv 77.0337). txt_ridge TRAIN=oof_txt_ridge.npy / TEST="
        "test_txt_ridge.npy (fold-safe). cv_mse_mean/std = compute_cv_mse(oof) (avg-oof, DoD-4). "
        f"recency_weighted_oof_mse={recency_mse:.4f} (private-durust co-headline, review H1). "
        "anchor lgbm_num 81.70 -> FULL kapi GECTI (-4.67)."
    )

    # --- Artefaktlar ---
    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, best_iter_mean)
    aio.write_cv_log(
        MODEL, cv_mean, cv_std, fold_mse, best_iters, best_iter_mean,
        genuine_fold_mse=genuine, single5fold_std=single5fold_std, note=note,
    )

    # --- DoD-4 ic tutarlilik: kaydedilen oof'tan yeniden hesap, +/-1e-6 esle ---
    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, re_std, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6 and abs(re_std - cv_std) < 1e-6, (
        f"DoD-4 KIRIK: oof_{MODEL}.npy'den yeniden hesap {re_mean:.6f} != {cv_mean:.6f}."
    )

    # --- clip teyidi (hem oof hem test [0,100]) ---
    cv.assert_in_range(oof, "oof_lgbm_full")
    cv.assert_in_range(test_pred, "test_lgbm_full")

    print(f"[lgbm_full] cv_mse_mean = {cv_mean:.4f}   (compute_cv_mse / avg-oof; cv_scores.csv'ye)")
    print(f"[lgbm_full] cv_mse_std  = {cv_std:.4f}")
    print(f"[lgbm_full] recency_weighted_oof_mse = {recency_mse:.4f}   "
          f"(CO-HEADLINE; private-durust tahmin, review H1)")
    print(f"[lgbm_full] genuine-15  : mean={np.mean(genuine):.4f}  std={np.std(genuine):.4f}")
    print(f"[lgbm_full] best_iteration_mean = {best_iter_mean:.1f}  (15 fold)")
    print(f"[lgbm_full] test fold-bagging: mean={test_pred.mean():.3f} std={test_pred.std():.3f} "
          f"min={test_pred.min():.3f} max={test_pred.max():.3f}")
    print(f"[lgbm_full] DoD-4 ic tutarlilik GECTI (oof.npy -> {re_mean:.6f}).")

    if REF_LO <= cv_mean <= REF_HI:
        print(f"[lgbm_full] cv_mse_mean {cv_mean:.2f} Faz05 FULL bandinda ({REF_LO}-{REF_HI}); "
              f"77.0337 olcumuyle uyumlu -> NLP enjeksiyonu DOGRU.")
    else:
        print(f"[lgbm_full][UYARI] cv_mse_mean {cv_mean:.2f} beklenen Faz05 bandi ({REF_LO}-{REF_HI}) "
              f"DISINDA -> assembly/artefaktlari incele.")


if __name__ == "__main__":
    main()
