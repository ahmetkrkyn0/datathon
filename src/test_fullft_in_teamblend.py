"""
Full-FT'yi (13. model) Tuna'nin 12-model blend'ine ekleyince kazanc var mi?
Tuna'nin KENDI gate'iyle (nested ridge_pos 5x3, recency-weighted, 15-hucre paired).

Girdi:
  tunadan gelenler/  -> 11 model oof/test npy + y + w_recency + folds.parquet
  ourteam_oof_tunafolds.npy + ourteam_test_tunafolds.npy  -> bizim (12. model)
  fullft_oof.npy + fullft_test.npy  -> Colab'dan gelecek (13. model)

Cikti: 12-model vs 13-model nested rw + paired test + (kazanc varsa) yeni submission.

Calistir: python -u src/test_fullft_in_teamblend.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "tunadan gelenler"
N_REP, N_SPL = 3, 5


def main():
    # Tuna'nin 11 modeli
    models = list(np.load(G / "models.npy", allow_pickle=True))
    y = np.load(G / "y.npy").astype(float)
    w = np.load(G / "w_recency.npy").astype(float)
    sid = np.load(G / "student_id_train.npy", allow_pickle=True)
    sid_te = np.load(G / "student_id_test.npy", allow_pickle=True)
    oof = {m: np.load(G / f"oof_{m}.npy") for m in models}
    test = {m: np.load(G / f"test_{m}.npy") for m in models}

    # 12. model: bizim ourteam_tf
    oof["ourteam_tf"] = np.load(ROOT / "ourteam_oof_tunafolds.npy").astype(float)
    test["ourteam_tf"] = np.load(ROOT / "ourteam_test_tunafolds.npy").astype(float)

    # 13. model: full-FT (varsa)
    has_ft = (ROOT / "fullft_oof.npy").exists()
    if has_ft:
        oof["fullft"] = np.load(ROOT / "fullft_oof.npy").astype(float)
        test["fullft"] = np.load(ROOT / "fullft_test.npy").astype(float)
        print("full-FT bulundu -> 13. model olarak test edilecek")
    else:
        print("full-FT henuz yok -> sadece 12-model dogrulamasi")

    folds = pd.read_parquet(G / "folds.parquet")
    pos = {s: i for i, s in enumerate(sid)}

    def fold_of(rep):
        fr = folds[folds["repeat"] == rep]
        out = np.full(len(sid), -1, dtype=int)
        for s, f in zip(fr["student_id"].values, fr["fold"].values):
            out[pos[s]] = f
        return out

    def clip(p):
        return np.clip(p, 0, 100)

    def rwmse(p):
        return float(np.sum(w * (y - clip(p)) ** 2) / np.sum(w))

    def ridge_pos_fit(P, yy, ww):
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=1.0, positive=True, fit_intercept=True)
        r.fit(P, yy, sample_weight=ww)
        return r

    def nested(names):
        P = np.column_stack([oof[m] for m in names])
        n = len(y)
        s = np.zeros(n)
        c = np.zeros(n)
        for r in range(N_REP):
            fo = fold_of(r)
            for g in range(N_SPL):
                va = np.where(fo == g)[0]
                tr = np.where(fo != g)[0]
                rr = ridge_pos_fit(P[tr], y[tr], w[tr])
                s[va] += clip(rr.predict(P[va]))
                c[va] += 1
        meta = clip(s / c)
        return rwmse(meta), meta

    def per_cell(meta):
        out = []
        for r in range(N_REP):
            fo = fold_of(r)
            for g in range(N_SPL):
                idx = np.where(fo == g)[0]
                ww = w[idx]
                out.append(float(np.sum(ww * (y[idx] - meta[idx]) ** 2) / np.sum(ww)))
        return np.array(out)

    base12 = models + ["ourteam_tf"]
    rw12, meta12 = nested(base12)
    print(f"\n12-model blend nested rw-OOF = {rw12:.4f}  (Tuna referans 82.2398)")

    if has_ft:
        print(f"\nfull-FT tek-model rw = {rwmse(oof['fullft']):.3f}  "
              f"duz MSE = {((y - clip(oof['fullft'])) ** 2).mean():.2f}")
        # residual korelasyonu
        res = y - meta12
        print(f"full-FT residual korelasyonu (12-model'e gore): "
              f"{np.corrcoef(res, oof['fullft'])[0, 1]:+.4f}")
        rw13, meta13 = nested(base12 + ["fullft"])
        print(f"\n13-model blend nested rw-OOF = {rw13:.4f}  (delta {rw13 - rw12:+.4f})")
        d = per_cell(meta13) - per_cell(meta12)
        imp = int(np.sum(d < 0))
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=len(d) - 1))
        print(f"PAIRED (13 vs 12): iyilesen={imp}/15  t={t:.3f}  p={p:.2e}")
        gate = imp >= 12 and p < 0.01 and rw13 < rw12
        print(f"\n>>> KAPI: {'GECTI - full-FT eklenmeli!' if gate else 'GECMEDI - metin tavani, 12-model kalsin'}")

        if gate:
            # final test tahmini + submission
            from sklearn.linear_model import Ridge
            P = np.column_stack([oof[m] for m in base12 + ["fullft"]])
            Pte = np.column_stack([test[m] for m in base12 + ["fullft"]])
            rr = Ridge(alpha=1.0, positive=True, fit_intercept=True).fit(P, y, sample_weight=w)
            final = clip(rr.predict(Pte))
            sub = pd.DataFrame({"student_id": sid_te, "career_success_score": final.round(3)})
            out = ROOT / "submissions" / "team_blend_13model.csv"
            sub.to_csv(out, index=False)
            print(f"YAZILDI -> {out} (mean {final.mean():.3f})")


if __name__ == "__main__":
    main()
