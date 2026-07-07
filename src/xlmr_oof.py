"""
FOLD-ESLI BERTurk fine-tune + embedding cikarimi.

v17'nin basarisizliginin duzeltmesi: metin modeli GBM ile AYNI
KFold(10, shuffle, seed=42) split'inde egitilir. Her fold modelinden:
  - TUM train satirlarinin embedding'i (model kendi train fold'unu gormus,
    val fold'unu gormemis — GBM fold k icinde hepsi TEK uzayda)
  - TUM test satirlarinin embedding'i
  - skalar tahminler (OOF val + test ortalama)

Cikti: data/cache/bert_foldemb.npz
  emb_tr_f{k}: (10000, 768)  emb_te_f{k}: (10000, 768)  k=0..9
  + bert3_oof.npy / bert3_test.npy (skalar)

Calistir: python -u src/bert_foldmatched.py   (~55 dk GPU)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MODEL = "xlm-roberta-large"
SEED = 42
N_FOLDS = 5  # GBM scriptleriyle ayni!
EPOCHS = 3
BATCH = 12
LR = 1e-5
MAX_LEN = 160


class TxtDS(Dataset):
    def __init__(self, enc, idx, y=None):
        self.enc = enc
        self.idx = idx
        self.y = y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        item = {k: v[j] for k, v in self.enc.items() if k != "token_type_ids"}
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[j], dtype=torch.float32)
        return item


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(self.bert.config.hidden_size, 1)

    def pooled(self, ids, att):
        h = self.bert(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, ids, att):
        return self.head(self.drop(self.pooled(ids, att))).squeeze(-1)


@torch.no_grad()
def infer_all(model, enc, dev, bs=96):
    model.eval()
    n = len(enc["input_ids"])
    preds = np.zeros(n, dtype=np.float32)
    embs = np.zeros((n, 1024), dtype=np.float32)
    for s in range(0, n, bs):
        ids = enc["input_ids"][s:s+bs].to(dev)
        att = enc["attention_mask"][s:s+bs].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            p = model.pooled(ids, att)
            out = model.head(p).squeeze(-1)
        preds[s:s+bs] = out.float().cpu().numpy()
        embs[s:s+bs] = p.float().cpu().numpy()
    return preds, embs


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    torch.manual_seed(SEED)
    from transformers import AutoTokenizer

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test_x.csv")
    y_raw = train["career_success_score"].values.astype(np.float32)
    ym, ys = y_raw.mean(), y_raw.std()
    y = (y_raw - ym) / ys

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc_tr = tok(train["mentor_feedback_text"].fillna("").tolist(),
                 truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")
    enc_te = tok(test["mentor_feedback_text"].fillna("").tolist(),
                 truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = {}
    oof_pred = np.zeros(len(train), dtype=np.float32)
    te_pred = np.zeros(len(test), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
        model = Reg().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        dl = DataLoader(TxtDS(enc_tr, tr_idx, y), batch_size=BATCH, shuffle=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=len(dl) * EPOCHS, pct_start=0.1)
        model.train()
        for ep in range(EPOCHS):
            for b in dl:
                opt.zero_grad()
                ids = b["input_ids"].to(dev)
                att = b["attention_mask"].to(dev)
                lab = b["labels"].to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(model(ids, att), lab)
                loss.backward()
                opt.step()
                sched.step()

        # bu fold modelinden TUM train + test embedding/tahminleri
        p_tr, e_tr = infer_all(model, enc_tr, dev)
        p_te, e_te = infer_all(model, enc_te, dev)
        oof_pred[va_idx] = p_tr[va_idx]
        te_pred += p_te / N_FOLDS
        v_mse = ((y_raw[va_idx] - np.clip(p_tr[va_idx] * ys + ym, 0, 100)) ** 2).mean()
        print(f"  fold {fold+1}/{N_FOLDS}: val MSE = {v_mse:.2f}")
        del model
        torch.cuda.empty_cache()

    oof100 = np.clip(oof_pred * ys + ym, 0, 100)
    te100 = np.clip(te_pred * ys + ym, 0, 100)
    print(f"\nXLM-R-large (5-fold) skalar OOF MSE = {((y_raw - oof100) ** 2).mean():.2f}")
    np.save(CACHE / "xlmr_oof.npy", oof100)
    np.save(CACHE / "xlmr_test.npy", te100)
    print("KAYDEDILDI -> mdeb_oof/test.npy")


if __name__ == "__main__":
    main()
