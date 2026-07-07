"""
GÜÇLÜ MULTIMODAL — XLM-R-large metin kulesi + derin tabular kule, JOINT.

Amaç: blend'e ORTOGONAL yeni sinyal. Mevcut blend GBM + scalar-text ağırlıklı;
bu model metni GÖREVE-KOŞULLU öğrenir (önceden çıkarılmış scalar değil) + tabular ile
birlikte. mm (85.8) bu yönün zayıf versiyonuydu; bu güçlü hali (large + derin tab +
çok-seed + year-norm) daha orthogonal olabilir.

- Metin: XLM-R-large mean-pool (1024d), layer-wise LR decay
- Tabular: 170 feature -> 512->256->128 derin MLP (BatchNorm+SiLU+Dropout)
- Joint head: concat -> 256 -> 1
- year-norm hedef + uniform (en iyi ders), Tuna folds (5-fold, repeat 0)
- 2 seed averaj (stabilite), bf16
- Çıktı: mmstrong_oof.npy + mmstrong_test.npy + ANINDA orthogonallik raporu

Girdi (Colab upload): train.csv, test_x.csv, folds.parquet,
  feat_train.parquet, feat_test.parquet (tabular feature),
  oof_blend.npy, test_blend.npy, y.npy, w_recency.npy, student_id_train/test.npy (orthogonallik testi)
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

MODEL = "xlm-roberta-large"
SEEDS = [42, 7]
N_FOLDS = 5
EPOCHS = 5
BATCH = 24
MAX_LEN = 192
LR_TXT, LR_DECAY, LR_TAB = 1.2e-5, 0.9, 1e-3
DEV = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ID, TARGET = "student_id", "career_success_score"


class DS(Dataset):
    def __init__(self, enc, tab, idx, y=None):
        self.enc, self.tab, self.idx, self.y = enc, tab, idx, y
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        it = {k: v[j] for k, v in self.enc.items() if k != "token_type_ids"}
        it["tab"] = torch.tensor(self.tab[j], dtype=torch.float32)
        if self.y is not None:
            it["y"] = torch.tensor(self.y[j], dtype=torch.float32)
        return it


class MM(nn.Module):
    def __init__(self, n_tab):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
        h = self.bert.config.hidden_size
        self.tab = nn.Sequential(
            nn.Linear(n_tab, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(h + 128, 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, ids, att, tab):
        hs = self.bert(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(torch.cat([pooled, self.tab(tab)], 1)).squeeze(-1)


def llrd(model):
    groups = [{"params": list(model.tab.parameters()) + list(model.head.parameters()), "lr": LR_TAB}]
    layers = model.bert.encoder.layer
    n = len(layers)
    for i, l in enumerate(layers):
        groups.append({"params": l.parameters(), "lr": LR_TXT * (LR_DECAY ** (n - 1 - i))})
    groups.append({"params": model.bert.embeddings.parameters(), "lr": LR_TXT * (LR_DECAY ** n)})
    return groups


@torch.no_grad()
def infer(model, enc, tab, bs=64):
    model.eval(); n = len(enc["input_ids"]); out = np.zeros(n, np.float32)
    for s in range(0, n, bs):
        ids = enc["input_ids"][s:s+bs].to(DEV); att = enc["attention_mask"][s:s+bs].to(DEV)
        tb = torch.tensor(tab[s:s+bs], dtype=torch.float32).to(DEV)
        with torch.autocast(device_type="cuda", dtype=AMP):
            out[s:s+bs] = model(ids, att, tb).float().cpu().numpy()
    return out


def main():
    train = pd.read_csv("train.csv", encoding="utf-8-sig")
    test = pd.read_csv("test_x.csv", encoding="utf-8-sig")
    y = train[TARGET].values.astype(np.float32)
    yr_tr = train["application_year"].values
    yr_te = test["application_year"].values
    feat_tr = pd.read_parquet("feat_train.parquet")
    feat_te = pd.read_parquet("feat_test.parquet")
    tabcols = [c for c in feat_tr.columns if pd.api.types.is_numeric_dtype(feat_tr[c])
               and c not in (TARGET,) and not c.startswith("txt_")]
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    TAB_TR = sc.fit_transform(imp.fit_transform(feat_tr[tabcols])).astype(np.float32)
    TAB_TE = sc.transform(imp.transform(feat_te[tabcols])).astype(np.float32)
    print(f"tabular: {TAB_TR.shape}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    def enc(s): return tok(s.fillna("").tolist(), truncation=True, padding="max_length",
                           max_length=MAX_LEN, return_tensors="pt")
    enc_tr = enc(train["mentor_feedback_text"]); enc_te = enc(test["mentor_feedback_text"])

    folds = pd.read_parquet("folds.parquet")
    pos = {s: i for i, s in enumerate(train[ID].values)}
    def val_idx(f):
        ids = folds[(folds.repeat == 0) & (folds.fold == f)][ID].values
        return np.array([pos[s] for s in ids])

    oof = np.zeros(len(train), np.float32); te_pred = np.zeros(len(test), np.float32)
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        for f in range(N_FOLDS):
            va = val_idx(f); trn = np.setdiff1d(np.arange(len(train)), va)
            st = pd.DataFrame({"yil": yr_tr[trn], "y": y[trn]}).groupby("yil")["y"].agg(["mean", "std"])
            mu = pd.Series(yr_tr[trn]).map(st["mean"]).values; sd = pd.Series(yr_tr[trn]).map(st["std"]).values
            # yn GLOBAL boyutlu (DS global indeks j ile erisir; enc_tr/TAB_TR global)
            yn = np.zeros(len(train), dtype=np.float32)
            yn[trn] = ((y[trn] - mu) / sd).astype(np.float32)
            mu_va = pd.Series(yr_tr[va]).map(st["mean"]).values; sd_va = pd.Series(yr_tr[va]).map(st["std"]).values
            mu_te = pd.Series(yr_te).map(st["mean"]).values; sd_te = pd.Series(yr_te).map(st["std"]).values

            model = MM(TAB_TR.shape[1]).to(DEV)
            opt = torch.optim.AdamW(llrd(model), weight_decay=0.01)
            dl = DataLoader(DS(enc_tr, TAB_TR, trn, yn), batch_size=BATCH, shuffle=True, drop_last=True, num_workers=2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                        total_steps=len(dl)*EPOCHS, pct_start=0.1)
            model.train()
            for ep in range(EPOCHS):
                for b in dl:
                    opt.zero_grad()
                    ids = b["input_ids"].to(DEV); att = b["attention_mask"].to(DEV)
                    tb = b["tab"].to(DEV); lab = b["y"].to(DEV)
                    with torch.autocast(device_type="cuda", dtype=AMP):
                        loss = nn.functional.smooth_l1_loss(model(ids, att, tb), lab)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step()
            vp = infer(model, {k: v[va] for k, v in enc_tr.items()}, TAB_TR[va])
            oof[va] += np.clip(vp*sd_va+mu_va, 0, 100) / len(SEEDS)
            te_pred += np.clip(infer(model, enc_te, TAB_TE)*sd_te+mu_te, 0, 100) / (len(SEEDS)*N_FOLDS)
            print(f"seed{seed} fold{f}: val MSE={((y[va]-np.clip(vp*sd_va+mu_va,0,100))**2).mean():.2f}")
            del model; torch.cuda.empty_cache()

    np.save("mmstrong_oof.npy", oof); np.save("mmstrong_test.npy", te_pred)
    print(f"\nMM-STRONG OOF MSE = {((y-oof)**2).mean():.2f}")

    # === ORTOGONALLIK TESTİ (submission harcamadan önce) ===
    try:
        w = np.load("w_recency.npy"); sid = np.load("student_id_train.npy", allow_pickle=True)
        o = np.array([pos[s] for s in sid])
        blend = np.load("oof_blend.npy").astype(float); yo = y[o].astype(float); oofs = oof[o].astype(float)
        def rw(p): return float(np.sum(w*(yo-np.clip(p,0,100))**2)/np.sum(w))
        res = yo - blend
        print(f"\n=== ORTOGONALLIK ===")
        print(f"  mm-strong tek-model rw = {rw(oofs):.3f}")
        print(f"  blend ile corr = {np.corrcoef(oofs,blend)[0,1]:.4f}")
        print(f"  residual corr = {np.corrcoef(oofs,res)[0,1]:+.4f}  (|>0.05| = YENİ SİNYAL)")
        from sklearn.linear_model import Ridge
        fo = {f: set(val_idx(f)) for f in range(N_FOLDS)}
        rowfold = np.array([next(f for f in range(N_FOLDS) if i in fo[f]) for i in range(len(yo))])
        P = np.column_stack([blend, oofs]); s = np.zeros(len(yo))
        for f in range(N_FOLDS):
            va = np.where(rowfold == f)[0]; trn = np.where(rowfold != f)[0]
            rr = Ridge(alpha=1.0, positive=True).fit(P[trn], yo[trn], sample_weight=w[trn]); s[va] = np.clip(rr.predict(P[va]), 0, 100)
        print(f"  blend+mmstrong nested rw = {rw(s):.4f}  (blend {rw(blend):.4f}, {rw(s)-rw(blend):+.4f})")
        verdict = "DEGER KATIYOR -> blende ekle!" if rw(s) < rw(blend) - 0.02 else "deger yok (residual doygunlugu)"
        print(f"  >>> {verdict}")
    except Exception as e:
        print("ortogonallik testi atlandı:", e)
    print("\nKAYDEDILDI -> mmstrong_oof.npy + mmstrong_test.npy")


if __name__ == "__main__":
    main()
