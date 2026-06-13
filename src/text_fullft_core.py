"""
HAM METIN FULL FINE-TUNE — XLM-R-large, hedefe DOGRUDAN, full unfreeze.

Mevcut metin modelleri (xlmr skalar rw=143) feature'lara kosullu uretilmis +
sadece mean-pool/kismi egitimdi. Bu surum:
  - XLM-R-large TAM unfreeze (tum katmanlar egitilir, frozen yok)
  - hedef = career_success_score DOGRUDAN (year-norm + uniform, bizim en iyi ders)
  - layer-wise LR decay (alt katman kucuk, ust buyuk LR)
  - Tuna'nin folds.parquet'i (hizali kalsin)
  - 6 epoch, mean-pool + [CLS] concat, dropout

Amac: blend'in residual'ini metinden AÇIKLAYABILIR yeni bir aci var mi?
(Beklenti dusuk: metin feature'larin ozeti gibi gorunuyor, ama tek test
edilmemis bilgi kaynagi bu.)

Cikti: fullft_oof.npy + fullft_test.npy
Colab girdileri: train.csv, test_x.csv, folds.parquet
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

MODEL = "xlm-roberta-large"
SEED, EPOCHS, MAX_LEN, BATCH = 42, 6, 192, 16
LR_TOP, LR_DECAY, LR_HEAD, WD = 1.5e-5, 0.9, 1e-3, 0.01
DEV = "cuda"
torch.manual_seed(SEED); np.random.seed(SEED)
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("amp:", AMP, "| model:", MODEL)

train = pd.read_csv("train.csv", encoding="utf-8-sig")
test = pd.read_csv("test_x.csv", encoding="utf-8-sig")
y_raw = train["career_success_score"].values.astype(np.float32)
yr_tr = train["application_year"].values
yr_te = test["application_year"].values

tok = AutoTokenizer.from_pretrained(MODEL)
def enc(txt):
    return tok(txt.fillna("").tolist(), truncation=True, padding="max_length",
              max_length=MAX_LEN, return_tensors="pt")
enc_tr = enc(train["mentor_feedback_text"])
enc_te = enc(test["mentor_feedback_text"])


class DS(Dataset):
    def __init__(self, e, idx, y=None):
        self.e, self.idx, self.y = e, idx, y
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        it = {k: v[j] for k, v in self.e.items() if k != "token_type_ids"}
        if self.y is not None:
            it["labels"] = torch.tensor(self.y[j], dtype=torch.float32)
        return it


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
        h = self.bb.config.hidden_size
        self.head = nn.Sequential(nn.Linear(2 * h, 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, 1))
    def forward(self, ids, att):
        hs = self.bb(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        mean = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
        cls = hs[:, 0]
        return self.head(torch.cat([mean, cls], 1)).squeeze(-1)


def llrd_params(model):
    """layer-wise LR decay: ust katman LR_TOP, asagi indikce *LR_DECAY."""
    groups = [{"params": model.head.parameters(), "lr": LR_HEAD}]
    enc_layers = model.bb.encoder.layer
    n = len(enc_layers)
    for i, layer in enumerate(enc_layers):
        lr = LR_TOP * (LR_DECAY ** (n - 1 - i))
        groups.append({"params": layer.parameters(), "lr": lr})
    groups.append({"params": model.bb.embeddings.parameters(),
                   "lr": LR_TOP * (LR_DECAY ** n)})
    return groups


@torch.no_grad()
def infer(model, e, bs=64):
    model.eval()
    n = len(e["input_ids"]); out = np.zeros(n, np.float32)
    for s in range(0, n, bs):
        ids = e["input_ids"][s:s+bs].to(DEV); att = e["attention_mask"][s:s+bs].to(DEV)
        with torch.autocast(device_type="cuda", dtype=AMP):
            out[s:s+bs] = model(ids, att).float().cpu().numpy()
    return out


# Tuna fold'lari (student_id hizali) — repeat 0'i kullan (5-fold OOF; full FT pahali)
fd = pd.read_parquet("folds.parquet")
id2row = {s: i for i, s in enumerate(train["student_id"].values)}
fd0 = fd[fd.repeat == 0]
splits = []
for f in sorted(fd0.fold.unique()):
    ids = fd0[fd0.fold == f].student_id.values
    splits.append(np.array([id2row[s] for s in ids]))
print(f"folds -> {len(splits)} (repeat 0), val ort {np.mean([len(s) for s in splits]):.0f}")

oof = np.zeros(len(train), np.float32)
te_pred = np.zeros(len(test), np.float32)

for fold, va in enumerate(splits, 1):
    tri = np.setdiff1d(np.arange(len(train)), va)
    # year-norm hedef (fold-ici) + uniform agirlik
    st = pd.DataFrame({"yil": yr_tr[tri], "y": y_raw[tri]}).groupby("yil")["y"].agg(["mean", "std"])
    mu = pd.Series(yr_tr[tri]).map(st["mean"]).values; sd = pd.Series(yr_tr[tri]).map(st["std"]).values
    yn = ((y_raw[tri] - mu) / sd).astype(np.float32)
    mu_va = pd.Series(yr_tr[va]).map(st["mean"]).values; sd_va = pd.Series(yr_tr[va]).map(st["std"]).values
    mu_te = pd.Series(yr_te).map(st["mean"]).values; sd_te = pd.Series(yr_te).map(st["std"]).values

    model = Net().to(DEV)
    opt = torch.optim.AdamW(llrd_params(model), weight_decay=WD)
    dl = DataLoader(DS(enc_tr, tri, yn), batch_size=BATCH, shuffle=True, drop_last=True, num_workers=2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=len(dl) * EPOCHS, pct_start=0.06)
    model.train()
    for ep in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            ids = b["input_ids"].to(DEV); att = b["attention_mask"].to(DEV); lab = b["labels"].to(DEV)
            with torch.autocast(device_type="cuda", dtype=AMP):
                loss = torch.nn.functional.smooth_l1_loss(model(ids, att), lab, beta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

    vp = infer(model, {k: v[va] for k, v in enc_tr.items()})
    oof[va] = np.clip(vp * sd_va + mu_va, 0, 100)
    te_pred += np.clip(infer(model, enc_te) * sd_te + mu_te, 0, 100) / len(splits)
    vmse = ((y_raw[va] - oof[va]) ** 2).mean()
    print(f"fold {fold}/{len(splits)}: val MSE = {vmse:.2f}")
    del model; torch.cuda.empty_cache()

print(f"\nFULL-FT XLM-R-large OOF MSE = {((y_raw - oof) ** 2).mean():.2f} "
      f"(xlmr-skalar referans 126.1)")
np.save("fullft_oof.npy", oof)
np.save("fullft_test.npy", te_pred)
print("KAYDEDILDI -> fullft_oof.npy + fullft_test.npy")
