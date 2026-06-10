"""
Turkce BERT fine-tune -> mentor metninden hedef tahmini (OOF, sizintisiz).

- Model: dbmdz/bert-base-turkish-cased (BERTurk) + regresyon basligi
- 5-fold: her fold'da egit, val fold'u + test'i tahmin et (test ortalanir)
- fp16 (RTX 4070), batch 32, lr 2e-5, 3 epoch, max_len 128
- Cikti: data/cache/bert_oof.npy + bert_test.npy
  (sonraki adim: bunlar feature olarak tabular ensemble'a girer)

Calistir: python -u src/bert_text_oof.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
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
MAX_LEN = 128


class TxtDS(Dataset):
    def __init__(self, enc, y=None):
        self.enc = enc
        self.y = y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[i], dtype=torch.float32)
        return item


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for b in loader:
            ids = b["input_ids"].to(device)
            att = b["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=att).logits.squeeze(-1)
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def main():
    assert torch.cuda.is_available(), "CUDA yok!"
    device = "cuda"
    torch.manual_seed(SEED)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test_x.csv")
    y = train["career_success_score"].values.astype(np.float32) / 100.0
    tr_txt = train["mentor_feedback_text"].fillna("").tolist()
    te_txt = test["mentor_feedback_text"].fillna("").tolist()

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc_tr = tok(tr_txt, truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")
    enc_te = tok(te_txt, truncation=True, padding="max_length",
                 max_length=MAX_LEN, return_tensors="pt")
    te_loader = DataLoader(TxtDS(enc_te), batch_size=64)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    te_pred = np.zeros(len(test), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=1, problem_type="regression").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        enc_f = {k: v[tr_idx] for k, v in enc_tr.items()}
        enc_v = {k: v[va_idx] for k, v in enc_tr.items()}
        tr_loader = DataLoader(TxtDS(enc_f, y[tr_idx]), batch_size=BATCH,
                               shuffle=True)
        va_loader = DataLoader(TxtDS(enc_v), batch_size=64)
        n_steps = len(tr_loader) * EPOCHS
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=n_steps, pct_start=0.1)
        scaler = torch.amp.GradScaler()

        model.train()
        for ep in range(EPOCHS):
            for b in tr_loader:
                opt.zero_grad()
                ids = b["input_ids"].to(device)
                att = b["attention_mask"].to(device)
                lab = b["labels"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids=ids, attention_mask=att).logits.squeeze(-1)
                    loss = torch.nn.functional.mse_loss(logits, lab)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()

        oof[va_idx] = predict(model, va_loader, device)
        te_pred += predict(model, te_loader, device) / N_FOLDS
        v_mse = ((y[va_idx] - oof[va_idx]) ** 2).mean() * 10000
        print(f"  fold {fold}: val MSE = {v_mse:.2f} (0-100 olcek)")
        del model
        torch.cuda.empty_cache()

    oof100 = np.clip(oof * 100, 0, 100)
    te100 = np.clip(te_pred * 100, 0, 100)
    mse = ((train["career_success_score"].values - oof100) ** 2).mean()
    print(f"\nBERT metin-tek-basina OOF MSE = {mse:.2f} "
          f"(Ridge TF-IDF referansi: 146.8)")

    np.save(CACHE / "bert_oof.npy", oof100)
    np.save(CACHE / "bert_test.npy", te100)
    print(f"KAYDEDILDI -> {CACHE}/bert_oof.npy, bert_test.npy")


if __name__ == "__main__":
    main()
