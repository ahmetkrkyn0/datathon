"""
PyTorch tabular NN (kategorik embedding'li) — OOF + test tahmini uretir.

sklearn MLP'den farklari: kategorik embedding, BatchNorm+SiLU+Dropout,
OneCycle schedule, agirlikli MSE loss (yil agirliklari), GPU, early stop.
Cikti: data/cache/nn_oof.npy + nn_test.npy  (v9 blend'ine girer)

Calistir: python -u src/nn_oof.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
SEED = 42
N_FOLDS = 10
SEEDS = [42, 7]
EPOCHS = 60
BATCH = 512
PATIENCE = 8


class TabNN(nn.Module):
    def __init__(self, n_num, cat_cards, emb_dim=8):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(c, min(emb_dim, (c + 1) // 2)) for c in cat_cards])
        e_total = sum(e.embedding_dim for e in self.embs)
        d = n_num + e_total
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x_num, x_cat):
        es = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat([x_num] + es, dim=1)).squeeze(-1)


def wmse(y, p, w):
    return float(np.average((y - np.clip(p, 0, 100)) ** 2, weights=w))


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    train, test, y, w_fit, num_cols = F.build_features()
    train["txt_bert"] = np.load(CACHE / "bert_oof.npy")
    test["txt_bert"] = np.load(CACHE / "bert_test.npy")
    num_cols = num_cols + ["txt_bert"]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xn_tr = sc.fit_transform(imp.fit_transform(train[num_cols])).astype(np.float32)
    Xn_te = sc.transform(imp.transform(test[num_cols])).astype(np.float32)

    cat_maps, Xc_tr, Xc_te, cards = [], [], [], []
    for c in F.CAT_COLS:
        cats = pd.concat([train[c], test[c]]).astype(str).unique()
        mp = {v: i for i, v in enumerate(cats)}
        Xc_tr.append(train[c].astype(str).map(mp).values)
        Xc_te.append(test[c].astype(str).map(mp).values)
        cards.append(len(cats))
    Xc_tr = np.column_stack(Xc_tr).astype(np.int64)
    Xc_te = np.column_stack(Xc_te).astype(np.int64)

    y01 = (y / 100.0).astype(np.float32)
    w32 = w_fit.astype(np.float32)

    te_num = torch.tensor(Xn_te).to(dev)
    te_cat = torch.tensor(Xc_te).to(dev)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    te_pred = np.zeros(len(test), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xn_tr), 1):
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = TabNN(Xn_tr.shape[1], cards).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            ds = TensorDataset(torch.tensor(Xn_tr[tr_idx]),
                               torch.tensor(Xc_tr[tr_idx]),
                               torch.tensor(y01[tr_idx]),
                               torch.tensor(w32[tr_idx]))
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=2e-3, total_steps=len(dl) * EPOCHS, pct_start=0.15)
            va_num = torch.tensor(Xn_tr[va_idx]).to(dev)
            va_cat = torch.tensor(Xc_tr[va_idx]).to(dev)
            best_mse, best_pred, best_te, bad = 1e9, None, None, 0

            for ep in range(EPOCHS):
                model.train()
                for xb_n, xb_c, yb, wb in dl:
                    xb_n, xb_c = xb_n.to(dev), xb_c.to(dev)
                    yb, wb = yb.to(dev), wb.to(dev)
                    opt.zero_grad()
                    p = model(xb_n, xb_c)
                    loss = (wb * (p - yb) ** 2).mean()
                    loss.backward()
                    opt.step()
                    sched.step()
                model.eval()
                with torch.no_grad():
                    vp = model(va_num, va_cat).cpu().numpy()
                v_mse = ((y01[va_idx] - vp) ** 2).mean()
                if v_mse < best_mse - 1e-6:
                    best_mse, bad = v_mse, 0
                    best_pred = vp
                    with torch.no_grad():
                        best_te = model(te_num, te_cat).cpu().numpy()
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            oof[va_idx] += best_pred / len(SEEDS)
            te_pred += best_te / (len(SEEDS) * N_FOLDS)
            del model
            torch.cuda.empty_cache()
        print(f"  fold {fold}/{N_FOLDS} bitti")

    oof100 = np.clip(oof * 100, 0, 100)
    te100 = np.clip(te_pred * 100, 0, 100)
    print(f"\nNN OOF: duz MSE = {((y - oof100) ** 2).mean():.4f} | "
          f"agirlikli = {wmse(y, oof100, w_fit):.4f}")
    np.save(CACHE / "nn_oof.npy", oof100)
    np.save(CACHE / "nn_test.npy", te100)
    print(f"KAYDEDILDI -> {CACHE}/nn_oof.npy, nn_test.npy")


if __name__ == "__main__":
    main()
