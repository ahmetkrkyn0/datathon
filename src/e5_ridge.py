"""
TIER-3 SPEKULATIF — e5_ridge: FROZEN multilingual-e5 embedding uzerinde NESTED-OOF Ridge.
==========================================================================================

    python src/e5_ridge.py

NE: artifacts/emb_{train,test}.npy (bert_embed.py'nin urettigi FROZEN 1024-dim e5 gomme) uzerinde,
  text_utils.build_tfidf_ridge_oof'taki NESTED-OOF Ridge desenini BIREBIR taklit ederek fold-safe
  bir meta-feature uretir:
      artifacts/oof_e5_ridge.npy   (n,)     nested-OOF, clip[0,100]
      artifacts/test_e5_ridge.npy  (n_test,) fold-bagged, clip[0,100]
  + reports/model_scores.csv'ye standalone rw-OOF satiri (KARAR METRIGI, review H1).

FOLD-SAFE (PAZARLIKSIZ): embedding FROZEN+GLOBAL (satir-bagimsiz, fold-leakage IMKANSIZ). Ridge
  yine de yalniz IC-train'e (dis-fold train'inin inner-KFold ic-train parcasi) fit edilir; dis-valid
  ve test o fold'un inner modellerinin ORTALAMASI. Klasik stacking sizintisi YAPISAL engellenir.
  txt_ridge ile AYNI folds.parquet dis-dongusu + AYNI inner-KFold(random_state=SEED+r) -> satir-hizali.

ALPHA: txt_ridge mantigiyla repeat-0 fold-safe OOF taramasi (yogun e5 -> daha genis grid). En dusuk
  OOF-MSE'li alpha secilir; sonra nested-OOF o alpha ile uretilir.

KARAR: standalone rw-OOF (rapor) + ensemble.py NESTED rw-OOF + 0.25*std kapisi (84.7385). Public YOK.
Determinizm: Ridge kapali-form (random_state'siz, deterministik); SEED=42; emb .npy kanonik.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import artifacts_io as aio
import cv

MODEL = "e5_ridge"
EMB_TRAIN_PATH = cv.ARTIFACTS_DIR / "emb_train.npy"
EMB_TEST_PATH = cv.ARTIFACTS_DIR / "emb_test.npy"

# Yogun 1024-dim BIRIM-NORM e5 gomme -> kucuk alpha bandi. txt_ridge ile AYNI MANTIK: repeat-0
# fold-safe OOF ile sec (gate'e/public'e DEGIL, y'ye repeat-0 OOF ile bakar -> overfit'siz).
# NOT: ilk grid (1..500) yogun birim-norm embedding icin fazla buyuktu (optimum alpha~0.1-0.3
# civarinda; repeat-0 OOF: a=0.1->140.75, a=1->142.95). Grid embedding-uygun daraltildi; secim
# yine VERI-ODAKLI (repeat-0 OOF minimumu), kapiya bakilarak DEGIL.
RIDGE_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0)
N_INNER = cv.N_SPLITS  # = 5, txt_ridge ile ayni nested ic-KFold sayisi


def select_alpha(E, y, folds, sid, alphas=RIDGE_ALPHAS):
    """txt_ridge.select_alpha taklidi: repeat-0 5-fold OOF-MSE (fold-safe), en dusugu sec.

    Ridge yalniz dis-train embedding'inde fit; dis-valid tahmin. Doner: (best_alpha, {alpha: mse})."""
    y = np.asarray(y, dtype=float)
    fold_of = cv.fold_of_rows(folds, sid, 0)
    results: dict[float, float] = {}
    for a in alphas:
        oof = np.zeros(len(y))
        for f in range(cv.N_SPLITS):
            val = np.where(fold_of == f)[0]
            tr = np.where(fold_of != f)[0]
            r = Ridge(alpha=a)
            r.fit(E[tr], y[tr])
            oof[val] = r.predict(E[val])
        results[float(a)] = float(np.mean((y - cv.clip_predictions(oof)) ** 2))
    best = min(results, key=results.get)
    return best, results


def build_e5_ridge_oof(E, y, E_test, folds, sid, alpha, n_repeats=cv.N_REPEATS, n_inner=N_INNER):
    """text_utils.build_tfidf_ridge_oof'un BIREBIR analogu (TF-IDF yerine FROZEN embedding girisi).

    Her (repeat, dis-fold): dis-train'i inner KFold(random_state=SEED+r) ile boler; HER ic-train'de
    Ridge fit; dis-valid ve test tahmini o fold'un inner modellerinin ORTALAMASI. Hicbir model
    dis-valid'e/test'e fit edilmez -> stacking sizintisi engellenir. Doner: (oof, test) clip[0,100].
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_test = len(E_test)

    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=float)
    test_sum = np.zeros(n_test, dtype=float)
    n_test_models = 0

    for r in range(n_repeats):
        fold_of = cv.fold_of_rows(folds, sid, r)
        for f in range(cv.N_SPLITS):
            val_idx = np.where(fold_of == f)[0]
            tr_idx = np.where(fold_of != f)[0]

            inner = KFold(n_splits=n_inner, shuffle=True, random_state=cv.SEED + r)
            val_acc = np.zeros(len(val_idx), dtype=float)
            test_acc = np.zeros(n_test, dtype=float)
            k = 0
            for inner_tr_rel, _ in inner.split(tr_idx):
                idx = tr_idx[inner_tr_rel]  # SADECE ic-train (dis-valid'e ASLA dokunmaz)
                model = Ridge(alpha=alpha)
                model.fit(E[idx], y[idx])
                val_acc += model.predict(E[val_idx])
                test_acc += model.predict(E_test)
                k += 1

            oof_sum[val_idx] += val_acc / k
            oof_cnt[val_idx] += 1.0
            test_sum += test_acc / k
            n_test_models += 1

    assert np.all(oof_cnt == n_repeats), "e5_ridge OOF kapsami bozuk: satir != n_repeats kez gorulmus."
    oof = cv.clip_predictions(oof_sum / oof_cnt)
    test = cv.clip_predictions(test_sum / float(n_test_models))
    return oof, test


def main() -> None:
    cv.set_seed()

    if not (EMB_TRAIN_PATH.exists() and EMB_TEST_PATH.exists()):
        raise SystemExit(
            f"[e5_ridge] HATA: {EMB_TRAIN_PATH.name}/{EMB_TEST_PATH.name} yok. "
            f"Once embedding cikar: python src/bert_embed.py"
        )

    train = cv.load_train()
    test = cv.load_test()
    folds = cv.load_folds()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    E = np.load(EMB_TRAIN_PATH).astype(np.float64)   # Ridge icin float64 (kosul sayisi/sayisal kararlilik)
    E_test = np.load(EMB_TEST_PATH).astype(np.float64)
    assert E.shape[0] == len(train) and E_test.shape[0] == len(test), "emb satir sayisi train/test ile uyusmuyor."
    assert E.shape[1] == E_test.shape[1], "train/test emb boyutu farkli."
    print(f"[e5_ridge] FROZEN e5 emb: train{E.shape} test{E_test.shape}")

    # --- alpha secimi (repeat-0 fold-safe OOF; txt_ridge mantigi) ---
    best_alpha, alpha_res = select_alpha(E, y, folds, sid)
    print("[e5_ridge] alpha secimi (repeat-0 OOF-MSE): "
          + "  ".join(f"a={a:g}:{m:.3f}" for a, m in alpha_res.items())
          + f"  -> SECILEN alpha={best_alpha:g}")

    # --- nested-OOF + fold-bagged test ---
    oof, test_pred = build_e5_ridge_oof(E, y, E_test, folds, sid, alpha=best_alpha)
    cv.assert_in_range(oof, f"oof_{MODEL}")
    cv.assert_in_range(test_pred, f"test_{MODEL}")

    cv_mean, cv_std, fold_mse = cv.compute_cv_mse(oof, y, folds, sid)
    rw = cv.compute_recency_weighted_mse(oof, y, w)

    # mevcut metin base'leri ile karsilastir (cesitlilik/redundancy gostergesi)
    oof_ridge = np.load(cv.ARTIFACTS_DIR / "oof_txt_ridge.npy")
    rw_ridge = cv.compute_recency_weighted_mse(oof_ridge, y, w)
    corr_ridge = float(np.corrcoef(oof, oof_ridge)[0, 1])
    cmp = "DAHA IYI" if rw < rw_ridge else "daha kotu/esit"

    note = (
        f"{MODEL} = FROZEN multilingual-e5-large (1024-dim) -> nested-OOF Ridge(alpha={best_alpha:g}) "
        f"(TIER-3, src/e5_ridge.py). standalone rw-OOF={rw:.4f} (txt_ridge {rw_ridge:.4f}, {cmp}; "
        f"corr(txt_ridge)={corr_ridge:.3f}). Fold-safe (frozen emb global + Ridge ic-train'inde). "
        f"Blend faydasi ensemble.py NESTED rw-OOF + 0.25*std kapisi karar verir."
    )

    aio.save_oof_test(MODEL, oof, test_pred)
    aio.write_cv_score(MODEL, cv_mean, cv_std, 0.0)
    aio.write_cv_log(MODEL, cv_mean, cv_std, fold_mse, [None] * len(fold_mse), 0.0,
                     genuine_fold_mse=None, single5fold_std=None, note=note)
    aio.log_model_score(MODEL, cv_mean, cv_std, rw, weighted_training=False, note=note)

    # DoD-4: yeniden yukleyince ayni mean
    reloaded = np.load(cv.ARTIFACTS_DIR / f"oof_{MODEL}.npy")
    re_mean, re_std, _ = cv.compute_cv_mse(reloaded, y, folds, sid)
    assert abs(re_mean - cv_mean) < 1e-6 and abs(re_std - cv_std) < 1e-6, "DoD-4 KIRIK (reload mean farkli)."

    print(f"[e5_ridge] standalone: unweighted_cv={cv_mean:.4f} +/- {cv_std:.4f}  rw-OOF={rw:.4f}  "
          f"(txt_ridge rw={rw_ridge:.4f}; corr={corr_ridge:.3f}) -> {cmp}")
    print(f"[e5_ridge] test mean={test_pred.mean():.3f} std={test_pred.std():.3f}")
    print(f"[e5_ridge] YAZILDI: artifacts/oof_{MODEL}.npy, test_{MODEL}.npy + model_scores.csv satiri")
    print(f"[e5_ridge] Blend karari: python src/ensemble.py (CANDIDATE_POOL'a e5_ridge ekli).")


if __name__ == "__main__":
    main()
