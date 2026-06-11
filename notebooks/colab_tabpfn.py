# TabPFN v2 (prior-fitted transformer) — Colab GPU'da repeat-0 OOF + test uretimi (OPSIYONEL sonda)
# ===================================================================================================
# NE: GBDT'lere ORTOGONAL yeni fonksiyon sinifi denemesi (mm emsali: zayif standalone bile
#     ortogonallikle gate gecebilir). Lokalde CPU predict 16.9GB RAM istedi -> infeasible; Colab GPU
#     ile fold basina saniyeler. Kabul yine paired-gate + onay ile (lokalde src/ tarafinda).
#
# KOSUM (Colab, GPU runtime):
#   1) Bu dosyayi tek hucreye yapistir.
#   2) Yukle: data/folds.parquet, data/train.csv, artifacts/tabular_train.npy, artifacts/tabular_test.npy
#   3) Calistir -> oof_tabpfn.npy + test_tabpfn.npy indir -> repo'da artifacts/ icine koy.
#   4) Lokalde gate: mm_gate desenine uygun paired-test (Claude'a "tabpfn gate" de).
#
# FOLD SOZLESMESI: bizim folds.parquet repeat-0 (5 fold), satir sirasi train.csv student_id.
# Deterministik degil-ihtimali: TabPFN inference deterministik (sabit seed, egitim yok) ama GPU
# kernel farklari olabilir -> artefakt kanonik (.npy), mm ile ayni belgelenmis-tolerans sinifi.

# !pip -q install tabpfn==2.0.9
import numpy as np, pandas as pd, torch, time
from tabpfn import TabPFNRegressor

SEED = 42
train = pd.read_csv("train.csv", encoding="utf-8-sig")
y = train["career_success_score"].values.astype(np.float32)
sid = train["student_id"].values
T_tr = np.load("tabular_train.npy").astype(np.float32)   # 82 feat, NaN korunmus (TabPFN native isler)
T_te = np.load("tabular_test.npy").astype(np.float32)
folds = pd.read_parquet("folds.parquet")
pos = {s: i for i, s in enumerate(sid)}
fr0 = folds[folds["repeat"] == 0]
fold_of = np.full(len(train), -1, dtype=np.int64)
for s, f in zip(fr0["student_id"].values, fr0["fold"].values):
    fold_of[pos[s]] = int(f)
assert (fold_of >= 0).all()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)
oof = np.zeros(len(train)); test_pred = np.zeros(len(T_te))
for f in range(5):
    t0 = time.time()
    va = np.where(fold_of == f)[0]; tr = np.where(fold_of != f)[0]
    reg = TabPFNRegressor(device=DEV, random_state=SEED, ignore_pretraining_limits=True)
    reg.fit(T_tr[tr], y[tr])
    oof[va] = np.clip(reg.predict(T_tr[va]), 0, 100)
    test_pred += np.clip(reg.predict(T_te), 0, 100) / 5.0
    print(f"fold {f+1}/5: {time.time()-t0:.0f}s  val MSE={np.mean((y[va]-oof[va])**2):.4f}", flush=True)

test_pred = np.clip(test_pred, 0, 100)
print(f"\nTabPFN repeat-0 unw-OOF = {np.mean((y-oof)**2):.4f}  (kiyas: lgbm_full_h 76.25 / mm 83.30)")
np.save("oof_tabpfn.npy", oof.astype(np.float64))
np.save("test_tabpfn.npy", test_pred.astype(np.float64))
print("YAZILDI: oof_tabpfn.npy, test_tabpfn.npy -> indir, artifacts/ icine koy.")
# from google.colab import files; files.download("oof_tabpfn.npy"); files.download("test_tabpfn.npy")
