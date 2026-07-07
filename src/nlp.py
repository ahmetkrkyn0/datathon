"""
Faz 5 — NLP surucusu (SPEC 05). Turkce `mentor_feedback_text` -> nested-OOF `txt_ridge_pred`
meta-feature + 10 elle tasarlanmis sozluk/yapi ozelligi (Katman B).

    python src/nlp.py

URETIR (SPEC 05 §7 + task):
  * artifacts/oof_txt_ridge.npy , artifacts/test_txt_ridge.npy  (nested-OOF + fold-bagged, clip[0,100])
  * data/text_features_train.parquet , data/text_features_test.parquet  (Katman B, student_id anahtarli)
  * reports/nlp_ablation.csv  (NUM-only / +lexicon / +txt_ridge / +full ; char-ngram negatif sonuc)

SIZINTI: tum TF-IDF+Ridge fit'leri text_utils.build_tfidf_ridge_oof icinde SADECE ic-train'e
(nested); diger her sey fold-bagimsiz (lexicon hedef-bagimsiz, sabit). Anchor lgbm_num=81.70
(yillar dahil) referansina gore kabul kapisi (cv.acceptance_gate, 0.25*std).

Determinizm: SEED=42 her yerde; LGBM anchor HP (deterministic=True).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io as aio
import cv
import text_utils as tu
from anchor_lgbm_num import make_fit_fold as lgbm_fit_fold

# Locked anchor (features.py / artifacts/cv_scores.csv; review C1: yillar dahil). Kabul kapisi tabani.
ANCHOR_MEAN = 81.702394
ANCHOR_STD = 2.933211

NLP_ABLATION_PATH = cv.REPORTS_DIR / "nlp_ablation.csv"
TEXT_FEATURES_TRAIN_PATH = cv.DATA_DIR / "text_features_train.parquet"
TEXT_FEATURES_TEST_PATH = cv.DATA_DIR / "text_features_test.parquet"
TXT_MODEL = "txt_ridge"


def _add_text(structured_X: pd.DataFrame, txt_pred=None, lex_df: pd.DataFrame | None = None):
    """Anchor yapisal matrise txt_ridge_pred ve/veya Katman B kolonlarini ekler (kategorikler korunur)."""
    out = structured_X.reset_index(drop=True).copy()
    if txt_pred is not None:
        out["txt_ridge_pred"] = np.asarray(txt_pred, dtype=float)
    if lex_df is not None:
        L = lex_df.reset_index(drop=True)
        for c in L.columns:
            out[c] = L[c].to_numpy()
    return out


def _run_lgbm(X, y, X_test, folds, sid, cat_features):
    """num+(metin) matrisini anchor LGBM ile run_oof + compute_cv_mse -> (mean, std)."""
    out = cv.run_oof(lgbm_fit_fold(cat_features), X, y, X_test, folds, sid)
    mean, std, _ = cv.compute_cv_mse(out["oof"], y, folds, sid)
    return mean, std


def main() -> None:
    cv.set_seed()

    raw_train = cv.load_train()
    raw_test = cv.load_test()
    folds = cv.load_folds()
    sid = raw_train[cv.ID_COL].values
    y = raw_train[cv.TARGET_COL].values

    # Turkce-duyarli normalize (SPEC 05 §1; metin VE sozluk ayni turkish_lower).
    texts_tr = tu.normalize_texts(raw_train[cv.TEXT_COL].values)
    texts_te = tu.normalize_texts(raw_test[cv.TEXT_COL].values)
    # Metinde rakam YOK teyidi (skor/hedef sizintisi yok; SPEC 05 §6).
    has_digit = any(any(ch.isdigit() for ch in t) for t in texts_tr)
    print(f"[nlp] metin: train={len(texts_tr)} test={len(texts_te)}  has_digit={has_digit}")
    assert not has_digit, "Metinde rakam bulundu -> hazir-cevap sizintisi riski (SPEC 05 §6)."

    # ===================== Katman A: alpha secimi + nested-OOF txt_ridge ===================== #
    best_alpha, alpha_res = tu.select_alpha(texts_tr, y, folds, sid)
    print(f"[nlp] alpha secimi (repeat-0 OOF-MSE): "
          + "  ".join(f"a={a}:{m:.3f}" for a, m in alpha_res.items())
          + f"  -> SECILEN alpha={best_alpha}")

    oof_txt, test_txt = tu.build_tfidf_ridge_oof(texts_tr, y, texts_te, folds, sid, alpha=best_alpha)
    txt_mean, txt_std, _ = cv.compute_cv_mse(oof_txt, y, folds, sid)
    cv.assert_in_range(oof_txt, "oof_txt_ridge")
    cv.assert_in_range(test_txt, "test_txt_ridge")
    print(f"[nlp] txt_ridge STANDALONE OOF-MSE = {txt_mean:.4f} +/- {txt_std:.4f}  "
          f"(metin-tek-basina; corr(y)={np.corrcoef(oof_txt, y)[0,1]:.3f})")

    # char n-gram ABLATION (negatif sonuc; lean n_repeats=1, SPEC 05 §4).
    oof_char, _ = tu.build_tfidf_ridge_oof(
        texts_tr, y, texts_te, folds, sid, alpha=best_alpha,
        n_repeats=1, n_inner=3, analyzer="char_wb",
    )
    char_mean, char_std, _ = cv.compute_cv_mse(oof_char, y, folds, sid)
    print(f"[nlp] char_wb(3-5) STANDALONE OOF-MSE = {char_mean:.4f}  "
          f"(word {txt_mean:.4f} ile karsilastir; char {'KOTU' if char_mean > txt_mean else 'iyi'})")

    # ===================== Katman B: elle sozluk/yapi ozellikleri ===================== #
    lex_tr = tu.extract_handcrafted_features(raw_train[cv.TEXT_COL].values)
    lex_te = tu.extract_handcrafted_features(raw_test[cv.TEXT_COL].values)
    print(f"[nlp] Katman B: {lex_tr.shape[1]} ozellik {list(lex_tr.columns)}")

    # ===================== Ablation: num / +lexicon / +txt_ridge / +full ===================== #
    cat_dtypes = cv.structured_cat_dtypes(raw_train)
    X_struct, cat_features = cv.build_structured_matrix(raw_train, cat_dtypes)
    Xt_struct, _ = cv.build_structured_matrix(raw_test, cat_dtypes)

    rows: list[dict] = []

    def record(row, mean, std, kind="lgbm_matrix", note=""):
        is_lgbm = kind == "lgbm_matrix"
        delta = (mean - ANCHOR_MEAN) if is_lgbm else None
        gate = bool(cv.acceptance_gate(mean, ANCHOR_MEAN, ANCHOR_STD)) if is_lgbm else None
        rows.append(dict(
            row=row, kind=kind, cv_mse_mean=round(mean, 6), cv_mse_std=round(std, 6),
            delta_vs_anchor=None if delta is None else round(delta, 6),
            gate_pass=gate, note=note,
        ))
        d = "" if delta is None else f" d_anchor={delta:+.4f}"
        g = "" if gate is None else (" KABUL" if gate else " RET")
        print(f"[nlp] {row:<22} cv={mean:7.4f} +/- {std:5.4f}{d}{g}  {note}")

    # num-only (anchor yeniden-uretim)
    m, s = _run_lgbm(X_struct, y, Xt_struct, folds, sid, cat_features)
    record("num_only", m, s, note=f"anchor reproduce (locked {ANCHOR_MEAN})")

    # num + lexicon (Katman B)
    Xl = _add_text(X_struct, lex_df=lex_tr)
    Xtl = _add_text(Xt_struct, lex_df=lex_te)
    m, s = _run_lgbm(Xl, y, Xtl, folds, sid, cat_features)
    record("num+lexicon", m, s, note="Katman B (10 sozluk/yapi)")

    # num + txt_ridge (meta-feature)
    Xr = _add_text(X_struct, txt_pred=oof_txt)
    Xtr = _add_text(Xt_struct, txt_pred=test_txt)
    m, s = _run_lgbm(Xr, y, Xtr, folds, sid, cat_features)
    record("num+txt_ridge", m, s, note="nested-OOF tek meta-kolon")

    # num + txt_ridge + lexicon (FULL; DoD hedefi)
    Xf = _add_text(X_struct, txt_pred=oof_txt, lex_df=lex_tr)
    Xtf = _add_text(Xt_struct, txt_pred=test_txt, lex_df=lex_te)
    full_mean, full_std = _run_lgbm(Xf, y, Xtf, folds, sid, cat_features)
    record("num+txt_ridge+lexicon", full_mean, full_std, note="FULL (faz06 base matris adayi)")

    # text-standalone (word vs char negatif sonuc)
    record("txt_ridge_word_standalone", txt_mean, txt_std, kind="txt_standalone",
           note=f"alpha={best_alpha}, word 1-2gram (metin-tek-basina)")
    record("txt_ridge_char_standalone", char_mean, char_std, kind="txt_standalone",
           note="char_wb(3-5) negatif sonuc -> final pipeline'dan CIKARILDI (SPEC 05 §4)")

    # ===================== Artefakt yazimi ===================== #
    cv.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cv.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv.DATA_DIR.mkdir(parents=True, exist_ok=True)

    aio.save_oof_test(TXT_MODEL, oof_txt, test_txt)  # artifacts/oof_txt_ridge.npy + test_txt_ridge.npy

    lex_tr_out = lex_tr.copy(); lex_tr_out.insert(0, cv.ID_COL, raw_train[cv.ID_COL].values)
    lex_te_out = lex_te.copy(); lex_te_out.insert(0, cv.ID_COL, raw_test[cv.ID_COL].values)
    lex_tr_out.to_parquet(TEXT_FEATURES_TRAIN_PATH, index=False)
    lex_te_out.to_parquet(TEXT_FEATURES_TEST_PATH, index=False)

    pd.DataFrame(rows, columns=[
        "row", "kind", "cv_mse_mean", "cv_mse_std", "delta_vs_anchor", "gate_pass", "note",
    ]).to_csv(NLP_ABLATION_PATH, index=False)

    # ===================== Ozet / DoD ===================== #
    print("\n[nlp] === OZET ===")
    print(f"[nlp] txt_ridge standalone CV-MSE = {txt_mean:.4f} +/- {txt_std:.4f}  (alpha={best_alpha})")
    print(f"[nlp] FULL (num+txt+lexicon) CV-MSE = {full_mean:.4f} +/- {full_std:.4f}  "
          f"(delta_vs_anchor {full_mean - ANCHOR_MEAN:+.4f})")
    gate_full = cv.acceptance_gate(full_mean, ANCHOR_MEAN, ANCHOR_STD)
    print(f"[nlp] kabul kapisi (< {ANCHOR_MEAN - cv.ACCEPT_K * ANCHOR_STD:.4f}): "
          f"{'GECTI' if gate_full else 'gecemedi'}   DoD<=84: {'OK' if full_mean <= 84.0 else 'HAYIR'}")
    print(f"[nlp] yazildi: artifacts/oof_{TXT_MODEL}.npy, test_{TXT_MODEL}.npy, "
          f"{TEXT_FEATURES_TRAIN_PATH.name}/{TEXT_FEATURES_TEST_PATH.name}, {NLP_ABLATION_PATH.name}")


if __name__ == "__main__":
    main()
