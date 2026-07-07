# === MULTIMODAL v2: BERTurk + Tabular(sub2 dahil), 10-fold OOF ===
# Girdi (uploaded): train.csv, test_x.csv, colab_pkg_tab_train.npy, colab_pkg_tab_test.npy
# Cikti: mm2_oof.npy, mm2_test.npy
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel

MODEL = "dbmdz/bert-base-turkish-cased"
SEED, N_FOLDS, EPOCHS, MAX_LEN = 42, 10, 4, 160
BATCH = 64  # A100
LR_BERT, LR_HEAD = 2e-5, 1e-3
DEV = "cuda"
torch.manual_seed(SEED); np.random.seed(SEED)
USE_BF16 = torch.cuda.is_bf16_supported()
AMP = torch.bfloat16 if USE_BF16 else torch.float16
print("amp:", AMP)

train = pd.read_csv("train.csv", encoding="utf-8-sig")
test = pd.read_csv("test_x.csv", encoding="utf-8-sig")
y_raw = train["career_success_score"].values.astype(np.float32)
ym, ys = y_raw.mean(), y_raw.std()
y = (y_raw - ym) / ys
TAB_TR = np.load("colab_pkg_tab_train.npy")
TAB_TE = np.load("colab_pkg_tab_test.npy")
print("tabular:", TAB_TR.shape)

tok = AutoTokenizer.from_pretrained(MODEL)
enc_tr = tok(train["mentor_feedback_text"].fillna("").tolist(), truncation=True,
             padding="max_length", max_length=MAX_LEN, return_tensors="pt")
enc_te = tok(test["mentor_feedback_text"].fillna("").tolist(), truncation=True,
             padding="max_length", max_length=MAX_LEN, return_tensors="pt")


class MMDS(Dataset):
    def __init__(self, enc, tab, idx, y=None):
        self.enc, self.tab, self.idx, self.y = enc, tab, idx, y
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        it = {k: v[j] for k, v in self.enc.items() if k != "token_type_ids"}
        it["tab"] = torch.tensor(self.tab[j], dtype=torch.float32)
        if self.y is not None:
            it["labels"] = torch.tensor(self.y[j], dtype=torch.float32)
        return it


class MM(nn.Module):
    def __init__(self, n_tab):
        super().__init__()
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
def infer(model, enc, tab, bs=128):
    model.eval()
    n = len(enc["input_ids"])
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, bs):
        ids = enc["input_ids"][s:s+bs].to(DEV)
        att = enc["attention_mask"][s:s+bs].to(DEV)
        tb = torch.tensor(tab[s:s+bs], dtype=torch.float32).to(DEV)
        with torch.autocast(device_type="cuda", dtype=AMP):
            out[s:s+bs] = model(ids, att, tb).float().cpu().numpy()
    return out


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype=np.float32)
te_pred = np.zeros(len(test), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
    model = MM(TAB_TR.shape[1]).to(DEV)
    opt = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": LR_BERT},
        {"params": list(model.tab.parameters()) + list(model.head.parameters()),
         "lr": LR_HEAD}], weight_decay=0.01)
    dl = DataLoader(MMDS(enc_tr, TAB_TR, tr_idx, y), batch_size=BATCH,
                    shuffle=True, drop_last=True, num_workers=2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_BERT, LR_HEAD], total_steps=len(dl)*EPOCHS, pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            ids = b["input_ids"].to(DEV); att = b["attention_mask"].to(DEV)
            tb = b["tab"].to(DEV); lab = b["labels"].to(DEV)
            with torch.autocast(device_type="cuda", dtype=AMP):
                loss = torch.nn.functional.mse_loss(model(ids, att, tb), lab)
            loss.backward(); opt.step(); sched.step()
    va_p = infer(model, {k: v[va_idx] for k, v in enc_tr.items()}, TAB_TR[va_idx])
    oof[va_idx] = va_p
    te_pred += infer(model, enc_te, TAB_TE) / N_FOLDS
    v_mse = ((y_raw[va_idx] - np.clip(va_p*ys+ym, 0, 100))**2).mean()
    print(f"fold {fold}/{N_FOLDS}: val MSE = {v_mse:.2f}")
    del model; torch.cuda.empty_cache()

oof100 = np.clip(oof*ys+ym, 0, 100)
te100 = np.clip(te_pred*ys+ym, 0, 100)
print(f"\nMM2 OOF MSE = {((y_raw-oof100)**2).mean():.2f} (mm v1 referans: 85.79)")
np.save("mm2_oof.npy", oof100)
np.save("mm2_test.npy", te100)
print("KAYDEDILDI -> mm2_oof.npy, mm2_test.npy")
