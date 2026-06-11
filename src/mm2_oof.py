"""
MULTIMODAL joint net: BERTurk kulesi + tabular kule -> ortak head.

Fikir (arkadas-notebook'tan, kendi implementasyonumuz): metin ve tabular
BIRLIKTE egitilince metin kulesi tabular'in aciklayamadigi kalan sinyali
ogrenir — bagimsiz metin skalarindan farkli bir gorus.

- Metin: BERTurk mean-pool (768d), lr 2e-5
- Tabular: 256->128 MLP (BatchNorm+SiLU+Dropout), lr 1e-3 (head dahil)
- 5-fold OOF (sizintisiz skalar), bf16, 4 epoch, early-stop yok (sabit)
- Tabular girdi: txt_* OOF kolonlari HARIC tum sayisal feature'lar

Cikti: data/cache/mm2_oof.npy + mm2_test.npy

Calistir: python -u src/mm_oof.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import features as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MODEL = "dbmdz/bert-base-turkish-cased"
SEED = 42
N_FOLDS = 10
EPOCHS = 4
BATCH = 24
LR_BERT = 2e-5
LR_HEAD = 1e-3
MAX_LEN = 160


class MMDS(Dataset):
    def __init__(self, enc, tab, idx, y=None):
        self.enc = enc
        self.tab = tab
        self.idx = idx
        self.y = y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        item = {k: v[j] for k, v in self.enc.items() if k != "token_type_ids"}
        item["tab"] = torch.tensor(self.tab[j], dtype=torch.float32)
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[j], dtype=torch.float32)
        return item


class MM(nn.Module):
    def __init__(self, n_tab):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
        h = self.bert.config.hidden_size
        self.tab = nn.Sequential(
            nn.Linear(n_tab, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(h + 128, 128), nn.SiLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, ids, att, tab):
        hs = self.bert(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(torch.cat([pooled, self.tab(tab)], dim=1)).squeeze(-1)


@torch.no_grad()
def infer(model, enc, tab, dev, bs=64):
    model.eval()
    n = len(enc["input_ids"])
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, bs):
        ids = enc["input_ids"][s:s+bs].to(dev)
        att = enc["attention_mask"][s:s+bs].to(dev)
        tb = torch.tensor(tab[s:s+bs], dtype=torch.float32).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out[s:s+bs] = model(ids, att, tb).float().cpu().numpy()
    return out


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    torch.manual_seed(SEED)
    from transformers import AutoTokenizer

    train, test, y_raw_arr, w_fit, num_cols = F.build_features()
    y_raw = train["career_success_score"].values.astype(np.float32)
    ym, ys = y_raw.mean(), y_raw.std()
    y = (y_raw - ym) / ys

    tab_cols = [c for c in num_cols if not c.startswith("txt_")]
    train["sub2_f"] = np.load(CACHE / "sub2_oof.npy")
    test["sub2_f"] = np.load(CACHE / "sub2_test.npy")
    tab_cols = tab_cols + ["sub2_f"]
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    TAB_TR = sc.fit_transform(imp.fit_transform(train[tab_cols])).astype(np.float32)
    TAB_TE = sc.transform(imp.transform(test[tab_cols])).astype(np.float32)
    print(f"tabular girdi: {TAB_TR.shape[1]} kolon")

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc_tr = tok(train["mentor_feedback_text"].fillna("").tolist(),
                 truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")
    enc_te = tok(test["mentor_feedback_text"].fillna("").tolist(),
                 truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    te_pred = np.zeros(len(test), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        model = MM(TAB_TR.shape[1]).to(dev)
        opt = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": LR_BERT},
            {"params": list(model.tab.parameters()) + list(model.head.parameters()),
             "lr": LR_HEAD},
        ], weight_decay=0.01)
        dl = DataLoader(MMDS(enc_tr, TAB_TR, tr_idx, y), batch_size=BATCH,
                        shuffle=True, drop_last=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[LR_BERT, LR_HEAD],
            total_steps=len(dl) * EPOCHS, pct_start=0.1)

        model.train()
        for ep in range(EPOCHS):
            for b in dl:
                opt.zero_grad()
                ids = b["input_ids"].to(dev)
                att = b["attention_mask"].to(dev)
                tb = b["tab"].to(dev)
                lab = b["labels"].to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(model(ids, att, tb), lab)
                loss.backward()
                opt.step()
                sched.step()

        va_p = infer(model, {k: v[va_idx] for k, v in enc_tr.items()},
                     TAB_TR[va_idx], dev)
        oof[va_idx] = va_p
        te_pred += infer(model, enc_te, TAB_TE, dev) / N_FOLDS
        v_mse = ((y_raw[va_idx] - np.clip(va_p * ys + ym, 0, 100)) ** 2).mean()
        print(f"  fold {fold}/{N_FOLDS}: val MSE = {v_mse:.2f}")
        del model
        torch.cuda.empty_cache()

    oof100 = np.clip(oof * ys + ym, 0, 100)
    te100 = np.clip(te_pred * ys + ym, 0, 100)
    print(f"\nMULTIMODAL OOF MSE = {((y_raw - oof100) ** 2).mean():.2f} "
          f"(referans: en iyi tabular-tek ~75.3, metin-tek 129.6)")
    np.save(CACHE / "mm2_oof.npy", oof100)
    np.save(CACHE / "mm2_test.npy", te100)
    print("KAYDEDILDI -> mm2_oof.npy, mm2_test.npy")


if __name__ == "__main__":
    main()
