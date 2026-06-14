"""POST-PROCESS 9/11 — XLM-R confidence-gating (kanitlanmis mekanizma, nested param).

KARAR METRIGI = nested recency-weighted OOF MSE (rw-OOF). Yeni base DEGIL: mevcut 12-model
blend OOF'unu (artifacts/oof_blend.npy) LOKAL/KOSULLU duzelt -> rw-OOF dussun.

KANITLANMIS mekanizma (kullanici elde etti, public 82.24->82.11): metin modelleri UCLARDA
GBDT'den daha guvenilir. Ridge stacker xlmr'e ~0 agirlik veriyordu (lineer/global yakalayamaz)
-> bu LOKAL, kosullu, YUKARI-YON duzeltme:
    conf = |xlmr - 76.94|                     (76.94 = hedef ortalamasi, sabit gating-merkezi)
    pred = blend + [conf >= q_thr AND xlmr > 76.94] * a * (xlmr - blend)
Sadece YUKARI-yon (xlmr>ort): uc-ovgu metinde net; asagi-yon zarar verdigi olculmus.

NESTED SOZLESME (in-sample DEGIL): her (repeat,fold) hucresinde q_thr quantile'i ve a'yi
hucre DISINDAKI (tr) blend+xlmr+y'den (tr-rw'yi minimize ederek) sec, hucreyi (va) transform et.
3-repeat ortalama -> nested_oof. rw-OOF(nested_oof) = DURUST skor. test'e FROZEN (q tum-OOF
blend/xlmr conf quantile'inden, a tum-OOF) uygulanir.

Overfit kapisi YOK (kullanici karari): delta<0 ise (ne kadar kucuk olsa da) artefakt yaz + lowered.
"""
from __future__ import annotations

import numpy as np

import cv
import artifacts_io as aio

GATE_CENTER = 76.94                       # hedef ortalamasi (sabit gating-merkezi)
Q_GRID = (0.85, 0.90, 0.95)               # conf quantile esikleri (tr conf'tan)
A_GRID = (0.0, 0.2, 0.3, 0.4, 0.5)        # gate gucu; a=0 = 'gate yok' (no-op fallback)


def _apply_gate(blend: np.ndarray, xlmr: np.ndarray, q_thr: float, a: float) -> np.ndarray:
    """conf>=q_thr AND xlmr>merkez ise blend'i xlmr'e dogru a kadar it (YUKARI-yon). clip[0,100]."""
    conf = np.abs(xlmr - GATE_CENTER)
    mask = (conf >= q_thr) & (xlmr > GATE_CENTER)
    pred = blend + mask * a * (xlmr - blend)
    return cv.clip_predictions(pred)


def _select_params(blend_tr, xlmr_tr, y_tr, w_tr):
    """tr-rw'yi minimize eden (q_thr, a) hucre DISINDAN secilir.

    q_thr quantile'i tr conf dagilimindan (sadece yukari-yon adaylar uzerinden) hesaplanir;
    boylece test-frozen ile AYNI tanim. a==0 ise q etkisiz -> no-op (base'e geri).
    """
    conf_tr = np.abs(xlmr_tr - GATE_CENTER)
    up = xlmr_tr > GATE_CENTER
    pool = conf_tr[up]
    best = (np.inf, Q_GRID[0], 0.0)  # (rw, q_thr, a)
    for q in Q_GRID:
        q_thr = float(np.quantile(pool, q)) if pool.size else np.inf
        for a in A_GRID:
            pred = _apply_gate(blend_tr, xlmr_tr, q_thr, a)
            rw = float(np.sum(w_tr * (y_tr - pred) ** 2) / np.sum(w_tr))
            if rw < best[0]:
                best = (rw, q_thr, a)
    return best  # (tr_rw, q_thr, a)


def nested_apply(blend, xlmr, y, w, folds, sid):
    """NESTED: her hucrede (q,a) tr'den sec, va'ya uygula. 3-repeat ortalama -> nested_oof."""
    n = len(y)
    s = np.zeros(n)
    c = np.zeros(n)
    picks = []
    for r in range(cv.N_REPEATS):
        fo = cv.fold_of_rows(folds, sid, r)
        for g in range(cv.N_SPLITS):
            va = np.where(fo == g)[0]
            tr = np.where(fo != g)[0]
            _, q_thr, a = _select_params(blend[tr], xlmr[tr], y[tr], w[tr])
            s[va] += _apply_gate(blend[va], xlmr[va], q_thr, a)
            c[va] += 1
            picks.append((r, g, round(q_thr, 4), a))
    return cv.clip_predictions(s / c), picks


def main() -> None:
    cv.set_seed()
    train, test, folds = cv.load_train(), cv.load_test(), cv.load_folds()
    y = train[cv.TARGET_COL].values.astype(float)
    sid = train[cv.ID_COL].values
    w = cv.recency_weights(train, test)

    blend_oof = np.load(cv.ARTIFACTS_DIR / "oof_blend.npy").astype(float)
    blend_test = np.load(cv.ARTIFACTS_DIR / "test_blend.npy").astype(float)
    xlmr_oof = np.load(cv.ARTIFACTS_DIR / "oof_xlmr.npy").astype(float)
    xlmr_test = np.load(cv.ARTIFACTS_DIR / "test_xlmr.npy").astype(float)

    base_rw = cv.compute_recency_weighted_mse(blend_oof, y, w)
    print(f"base_rw (blend nested rw-OOF) = {base_rw:.6f}  (~82.24 beklenir)")

    post_oof, picks = nested_apply(blend_oof, xlmr_oof, y, w, folds, sid)
    post_rw = cv.compute_recency_weighted_mse(post_oof, y, w)
    delta = post_rw - base_rw

    # secilen parametre ozeti (hucreler arasi en sik (q,a) + a dagilimi)
    qs = [p[2] for p in picks]
    as_ = [p[3] for p in picks]
    from collections import Counter
    a_counts = Counter(as_)
    q_med = float(np.median(qs))
    a_mode = a_counts.most_common(1)[0][0]
    best_param = (
        f"a_mode={a_mode} a_dist={dict(sorted(a_counts.items()))} "
        f"q_thr_median={q_med:.3f} (merkez={GATE_CENTER}, sadece-yukari)"
    )

    cv_mean, cv_std, _ = cv.compute_cv_mse(post_oof, y, folds, sid)
    print(f"post_rw = {post_rw:.6f}")
    print(f"delta   = {delta:.6f}  ({'DUSTU' if delta < 0 else 'dusmedi/yukseldi'})")
    print(f"secilen param: {best_param}")
    print(f"post cv_mean={cv_mean:.4f} cv_std={cv_std:.4f}")

    lowered = delta < 0
    out_name = ""
    if lowered:
        # FROZEN: tum-OOF blend/xlmr conf'tan q, tum-OOF'tan a -> test'e uygula.
        _, q_thr_f, a_f = _select_params(blend_oof, xlmr_oof, y, w)
        post_test = _apply_gate(blend_test, xlmr_test, q_thr_f, a_f)
        out_name = "pp_xlmrgate"
        aio.save_oof_test(out_name, post_oof, post_test)
        aio.log_model_score(
            out_name, cv_mean, cv_std, post_rw,
            note=(f"postproc XLM-R confidence-gating (kanitlanmis mekanizma, nested param); "
                  f"delta {delta:.6f}; frozen q_thr={q_thr_f:.3f} a={a_f}; {best_param}"),
        )
        print(f"ARTEFAKT YAZILDI: oof_{out_name}.npy/test_{out_name}.npy (frozen q={q_thr_f:.3f} a={a_f})")
    else:
        aio.log_model_score(
            "pp_xlmrgate", cv_mean, cv_std, post_rw,
            note=(f"ELENDI postproc XLM-R confidence-gating; delta {delta:.6f} >= 0 "
                  f"(artefakt YAZILMADI); {best_param}"),
        )
        print("ELENDI: delta >= 0 -> artefakt YAZILMADI (sadece ledger note).")

    print(f"\nSONUC base_rw={base_rw:.6f} post_rw={post_rw:.6f} delta={delta:.6f} lowered={lowered}")


if __name__ == "__main__":
    main()
