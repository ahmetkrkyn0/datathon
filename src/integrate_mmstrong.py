"""
mmstrong gelince ANINDA: orthogonallik + (değer varsa) 13-model blend + gating + submission.

Colab'dan inen mmstrong_oof.npy + mmstrong_test.npy proje köküne konunca çalıştır:
    python -u src/integrate_mmstrong.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "tunadan gelenler"
D = ROOT / "tunadanistenenbelgeler"
CENTER = 76.94


def main():
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    te = pd.read_csv(ROOT / "data" / "test_x.csv")
    sid = np.load(G / "student_id_train.npy", allow_pickle=True)
    sid_te = np.load(G / "student_id_test.npy", allow_pickle=True)
    pos = {s: i for i, s in enumerate(tr["student_id"].values)}
    order = np.array([pos[s] for s in sid])
    y = tr["career_success_score"].values.astype(float)[order]
    w = np.load(G / "w_recency.npy")
    folds = pd.read_parquet(ROOT / "folds.parquet")
    fpos = {s: i for i, s in enumerate(sid)}

    def fold_of(rep):
        fr = folds[folds.repeat == rep]
        out = np.full(len(sid), -1, int)
        for s, f in zip(fr.student_id.values, fr.fold.values):
            out[fpos[s]] = f
        return out

    def clip(p): return np.clip(p, 0, 100)
    def rw(p): return float(np.sum(w * (y - clip(p)) ** 2) / np.sum(w))

    # 12 model + ourteam
    models = list(np.load(G / "models.npy", allow_pickle=True))
    OOF = {m: np.load(G / f"oof_{m}.npy").astype(float) for m in models}
    TEST = {m: np.load(G / f"test_{m}.npy").astype(float) for m in models}
    OOF["ourteam_tf"] = np.load(ROOT / "ourteam_oof_tunafolds.npy").astype(float)
    TEST["ourteam_tf"] = np.load(ROOT / "ourteam_test_tunafolds.npy").astype(float)
    allm = models + ["ourteam_tf"]
    blend = np.load(D / "oof_blend.npy").astype(float)
    blend_te = np.load(D / "test_blend.npy").astype(float)

    # mmstrong (Colab) — train sırasında üretildi -> sid sırasına çevir
    mm_tr = np.load(ROOT / "mmstrong_oof.npy").astype(float)
    mm_te = np.load(ROOT / "mmstrong_test.npy").astype(float)
    mm_oof = mm_tr[order]

    print(f"=== mmstrong ORTOGONALLIK ===")
    print(f"  tek-model rw = {rw(mm_oof):.3f}  (blend {rw(blend):.3f})")
    res = y - blend
    rc = np.corrcoef(mm_oof, res)[0, 1]
    print(f"  blend corr = {np.corrcoef(mm_oof, blend)[0, 1]:.4f}")
    print(f"  residual corr = {rc:+.4f}  (|>0.05| = yeni sinyal)")

    # 13-model nested blend (ourteam + mmstrong eklenmiş)
    def nested(names_extra):
        P = np.column_stack([OOF[m] for m in allm] + names_extra)
        n = len(y); s = np.zeros(n); c = np.zeros(n)
        for r in range(3):
            fo = fold_of(r)
            for g in range(5):
                va = np.where(fo == g)[0]; trn = np.where(fo != g)[0]
                rr = Ridge(alpha=1.0, positive=True).fit(P[trn], y[trn], sample_weight=w[trn])
                s[va] += clip(rr.predict(P[va])); c[va] += 1
        return rw(s / c)

    rw12 = nested([])
    rw13 = nested([mm_oof])
    print(f"\n  12-model nested rw = {rw12:.4f}")
    print(f"  +mmstrong (13) rw  = {rw13:.4f}  ({rw13 - rw12:+.4f})")

    if rw13 >= rw12 - 0.02:
        print("\n  >>> DEGER YOK (residual doygunlugu). 82.12 final kalsin.")
        return

    print("\n  >>> DEGER VAR! 13-model blend kuruluyor + gating + submission...")
    # 13-model blend OOF + test (full ridge)
    P = np.column_stack([OOF[m] for m in allm] + [mm_oof])
    Pte = np.column_stack([TEST[m] for m in allm] + [mm_te])
    rr = Ridge(alpha=1.0, positive=True).fit(P, y, sample_weight=w)
    new_blend = clip(rr.predict(P))
    new_blend_te = clip(rr.predict(Pte))
    print(f"  13-model blend (full) rw = {rw(new_blend):.4f}")

    # üstüne tek xlmr-gate (Tuna'nin kanitlanmis, taşınan mekanizması)
    xlmr = OOF["xlmr"]; xlmr_te = TEST["xlmr"]

    def pick(base, sig):
        conf = np.abs(sig - CENTER); pool = conf[sig > CENTER]; best = (np.inf, 0, 0)
        for q in (0.85, 0.90, 0.95):
            qt = float(np.quantile(pool, q))
            for a in (0, 0.2, 0.3, 0.4, 0.5):
                mk = (conf >= qt) & (sig > CENTER); pr = clip(base + mk * a * (sig - base))
                v = rw(pr) if len(pr) == len(y) else 9e9
                if v < best[0]: best = (v, qt, a)
        return best[1], best[2]

    qt, a = pick(new_blend, xlmr)
    g_oof = clip(new_blend + ((np.abs(xlmr - CENTER) >= qt) & (xlmr > CENTER)) * a * (xlmr - new_blend))
    g_te = clip(new_blend_te + ((np.abs(xlmr_te - CENTER) >= qt) & (xlmr_te > CENTER)) * a * (xlmr_te - new_blend_te))
    print(f"  + xlmr-gate (q={qt:.2f} a={a}) rw = {rw(g_oof):.4f}")

    final = g_te.round(4)
    sub = pd.DataFrame({"student_id": sid_te, "career_success_score": final})
    out = ROOT / "submissions" / "SUB_13model_mmstrong_gated.csv"
    sub.to_csv(out, index=False)
    print(f"\n  YAZILDI -> {out} (mean {final.mean():.3f})")
    print(f"  tahmini gercek LB ~ {rw(g_oof) + 0.10:.2f} (proxy+gap)")


if __name__ == "__main__":
    main()
