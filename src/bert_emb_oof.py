"""
BERTurk fine-tune + EMBEDDING cikarimi (OOF, sizintisiz).

bert_text_oof.py'den fark: skalar tahminin yani sira fine-tune edilmis
modelin MEAN-POOLED son katman temsilini (768-d) kaydeder.
  - train: her satirin embedding'i, o satiri gormemis fold modelinden (OOF)
  - test : 5 fold modelinin embedding ortalamasi
Fikir: hedefe gore fine-tune edilmis embedding uzayi, metni GBM'lerin
tabular feature'larla ETKILESIME sokabilecegi zengin forma tasir
(skalar tahminden cok daha fazla bilgi).

Cikti: data/cache/bert_emb_train.npy (10000x768),
       bert_emb_test.npy (10000x768),
       bert2_oof.npy / bert2_test.npy (skalar, mean-pool mimarisiyle)

Calistir: python -u src/bert_emb_oof.py
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
MODEL = "dbmdz/bert-base-turkish-cased"
SEED = 42
N_FOLDS = 5
EPOCHS = 3
BATCH = 32
LR = 2e-5
MAX_LEN = 160


class TxtDS(Dataset):
    def __init__(self, enc, y=None):
        self.enc = enc
        self.y = y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items() if k != "token_type_ids"}
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[i], dtype=torch.float32)
        return item


class Reg(nn.Module):
    """AutoModel + mean-pool + linear head (embedding erisilebilir)."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(MODEL)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(self.bert.config.hidden_size, 1)

    def pooled(self, ids, att):
        h = self.bert(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, ids, att):
        return self.head(self.drop(self.pooled(ids, att))).squeeze(-1)


@torch.no_grad()
def infer(model, loader, dev):
    model.eval()
    preds, embs = [], []
    for b in loader:
        ids = b["input_ids"].to(dev)
        att = b["attention_mask"].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            p = model.pooled(ids, att)
            out = model.head(p).squeeze(-1)
        preds.append(out.float().cpu().numpy())
        embs.append(p.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(embs)


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
    tr_txt = train["mentor_feedback_text"].fillna("").tolist()
    te_txt = test["mentor_feedback_text"].fillna("").tolist()

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc_tr = tok(tr_txt, truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")
    enc_te = tok(te_txt, truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")
    te_loader = DataLoader(TxtDS(enc_te), batch_size=96)

    H = 768
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(train), dtype=np.float32)
    oof_emb = np.zeros((len(train), H), dtype=np.float32)
    te_pred = np.zeros(len(test), dtype=np.float32)
    te_emb = np.zeros((len(test), H), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        model = Reg().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        enc_f = {k: v[tr_idx] for k, v in enc_tr.items()}
        enc_v = {k: v[va_idx] for k, v in enc_tr.items()}
        dl = DataLoader(TxtDS(enc_f, y[tr_idx]), batch_size=BATCH, shuffle=True)
        va_loader = DataLoader(TxtDS(enc_v), batch_size=96)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=len(dl) * EPOCHS, pct_start=0.1)
        scaler = torch.amp.GradScaler()

        model.train()
        for ep in range(EPOCHS):
            for b in dl:
                opt.zero_grad()
                ids = b["input_ids"].to(dev)
                att = b["attention_mask"].to(dev)
                lab = b["labels"].to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(ids, att)
                    loss = torch.nn.functional.mse_loss(out, lab)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()

        vp, ve = infer(model, va_loader, dev)
        oof_pred[va_idx] = vp
        oof_emb[va_idx] = ve
        tp, te_e = infer(model, te_loader, dev)
        te_pred += tp / N_FOLDS
        te_emb += te_e / N_FOLDS
        v_mse = (((y_raw[va_idx] - (vp * ys + ym))) ** 2).mean()
        print(f"  fold {fold}: val MSE = {v_mse:.2f}")
        del model
        torch.cuda.empty_cache()

    oof100 = np.clip(oof_pred * ys + ym, 0, 100)
    te100 = np.clip(te_pred * ys + ym, 0, 100)
    print(f"\nBERT2 skalar OOF MSE = {((y_raw - oof100) ** 2).mean():.2f}")
    np.save(CACHE / "bert2_oof.npy", oof100)
    np.save(CACHE / "bert2_test.npy", te100)
    np.save(CACHE / "bert_emb_train.npy", oof_emb)
    np.save(CACHE / "bert_emb_test.npy", te_emb)
    print(f"KAYDEDILDI -> bert_emb_train/test.npy ({oof_emb.shape}), "
          f"bert2_oof/test.npy")


if __name__ == "__main__":
    main()
