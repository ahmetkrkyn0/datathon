"""colab_xlmr.ipynb URETICISI (tek-seferlik). Calistir: python notebooks/build_colab_xlmr.py
Gecerli .ipynb semasi garantisi icin notebook'u JSON olarak insa eder (repo kokune yazar)."""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""# TIER-2 BIRLESME — XLM-R-large TEXT-ONLY nested-OOF (`txt_xlmr`)

Ahmet'in en iyi metin modeli (XLM-R-large, gunluk 126.1 metin-tek MSE) bizim **fold-safe**
altyapida (folds.parquet **repeat-0**, student_id sirasi) text-only fine-tune edilir.

**mm'den FARK:** mm = XLM-R + tabular MLP **birlesik**. Bu = **saf metin** (tabular dal YOK)
-> mm ile korelasyon dususe ortogonal metin sinyali; gate karar verir (corr yuksekse RED).

**Cikti:** `oof_xlmr.npy` + `test_xlmr.npy` (clip[0,100], student_id sirasi). Indir -> repo `artifacts/`.

**Kullanim:**
1. Runtime -> GPU (T4/A100). 2. Sol panele yukle: `train.csv`, `test_x.csv`, `folds.parquet`
   (repo KANONIK dosyalar). 3. Hucreleri sirayla calistir. 4. Inen 2 .npy'yi `artifacts/`'a koy.
5. Lokalde: `python src/xlmr_blend.py` (dogrula + blend etkisi).

**DETERMINIZM:** SEED=42; bit-deterministik DEGIL (neural/GPU/bf16) -> mm gibi "belgelenmis tolerans".
""")

code("!pip -q install transformers==4.44.2 accelerate sentencepiece pyarrow")

code("""
from google.colab import files
up = files.upload()  # YUKLE: train.csv, test_x.csv, folds.parquet
""")

code('''
# === SABITLER + KANONIK VERI (src/cv.py ile BIREBIR okuma) ===
import numpy as np, pandas as pd
SEED = 42
TARGET = "career_success_score"; TEXT = "mentor_feedback_text"; ID = "student_id"
RECENCY_COL = "graduation_year"; CLIP_LO, CLIP_HI = 0.0, 100.0

train = pd.read_csv("train.csv", encoding="utf-8-sig")
test  = pd.read_csv("test_x.csv", encoding="utf-8-sig")
print("train", train.shape, "test", test.shape)
assert train[ID].iloc[0] == "STU_000001" and test[ID].iloc[0] == "STU_010001", "ID sirasi beklenmedik."
assert len(train) == 10000 and len(test) == 10000

y = train[TARGET].values.astype(np.float64)
txt_tr = [("" if t is None else str(t)) for t in train[TEXT].values]
txt_te = [("" if t is None else str(t)) for t in test[TEXT].values]
# metinde rakam YOK teyidi (hazir-cevap sizintisi yok; SPEC 05)
assert not any(any(ch.isdigit() for ch in t) for t in txt_tr), "metinde rakam var (sizinti riski)."
''')

code('''
# === BIZIM folds.parquet repeat-0 (mm notebook ile BIREBIR) ===
folds = pd.read_parquet("folds.parquet")
assert {"student_id","repeat","fold"}.issubset(folds.columns)
pos = {sid: i for i, sid in enumerate(train[ID].values)}
fr0 = folds[folds["repeat"] == 0]
fold_of = np.full(len(train), -1, dtype=np.int64)
for sid, f in zip(fr0["student_id"].values, fr0["fold"].values):
    fold_of[pos[sid]] = int(f)
assert (fold_of >= 0).all(), "repeat-0: fold atanmamis satir."
N_SPLITS = int(fold_of.max()) + 1
FOLDS = [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(N_SPLITS)]
print(f"repeat-0: {N_SPLITS} fold, val sayilari = {[len(v) for _,v in FOLDS]}")
assert sum(len(v) for _,v in FOLDS) == len(train)
''')

code('''
# === TEXT-ONLY XLM-R-large + fold-safe egitim (mm Cell-5 deseni; tabular dal YOK) ===
import torch, torch.nn as nn, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

MODEL_NAME = "xlm-roberta-large"
MAX_LEN, BATCH, EPOCHS = 192, 16, 3
LR_BERT, LR_HEAD = 1e-5, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP = torch.bfloat16 if USE_BF16 else torch.float16
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print("device:", DEVICE, "| model:", MODEL_NAME, "| amp:", AMP)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextData(Dataset):
    def __init__(self, texts, tgt=None, ym=0.0, ys=1.0):
        self.t=texts; self.y=tgt; self.ym=ym; self.ys=ys
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = tok(self.t[i], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        it = {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0)}
        if self.y is not None:
            it["label"] = torch.tensor((self.y[i]-self.ym)/self.ys, dtype=torch.float32)
        return it

class TextReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME); h = self.bert.config.hidden_size
        # mean+CLS pool (mm ile ayni metin temsili); tabular dal YOK
        self.head = nn.Sequential(nn.Linear(2*h,256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,1))
    def forward(self, input_ids, attention_mask):
        hs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        mean = (hs*m).sum(1)/m.sum(1).clamp(min=1e-9); cls = hs[:,0]
        return self.head(torch.cat([mean, cls], dim=1)).squeeze(-1)

@torch.no_grad()
def predict(model, texts, ym, ys):
    model.eval(); out=[]
    for b in DataLoader(TextData(texts), batch_size=64):
        ids=b["input_ids"].to(DEVICE); am=b["attention_mask"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=AMP): p = model(ids, am)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)*ys + ym

oof = np.zeros(len(train), dtype=np.float64); test_pred = np.zeros(len(test), dtype=np.float64)
for f,(tr_idx, va_idx) in enumerate(FOLDS):
    print(f"\\n== Fold {f+1}/{N_SPLITS} ==")
    ym, ys = float(y[tr_idx].mean()), float(y[tr_idx].std())   # FOLD-SAFE hedef z-score (yaln. fold-train)
    model = TextReg().to(DEVICE)
    dl = DataLoader(TextData([txt_tr[i] for i in tr_idx], y[tr_idx], ym, ys),
                    batch_size=BATCH, shuffle=True, drop_last=True)
    bp=[p for n,p in model.named_parameters() if n.startswith("bert.")]
    hp=[p for n,p in model.named_parameters() if not n.startswith("bert.")]
    opt = torch.optim.AdamW([{"params":bp,"lr":LR_BERT},{"params":hp,"lr":LR_HEAD}], weight_decay=0.01)
    tot = len(dl)*EPOCHS; sch = get_linear_schedule_with_warmup(opt, int(0.1*tot), tot)
    scaler = torch.cuda.amp.GradScaler(enabled=not USE_BF16)
    for ep in range(EPOCHS):
        model.train()
        for b in dl:
            opt.zero_grad()
            ids=b["input_ids"].to(DEVICE); am=b["attention_mask"].to(DEVICE); lab=b["label"].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=AMP):
                loss = nn.functional.mse_loss(model(ids, am), lab)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        vp = np.clip(predict(model, [txt_tr[i] for i in va_idx], ym, ys), CLIP_LO, CLIP_HI)
        print(f"   epoch {ep+1} val MSE={np.mean((y[va_idx]-vp)**2):.4f}")
    oof[va_idx] = np.clip(predict(model, [txt_tr[i] for i in va_idx], ym, ys), CLIP_LO, CLIP_HI)
    test_pred  += np.clip(predict(model, txt_te, ym, ys), CLIP_LO, CLIP_HI) / N_SPLITS
    del model; torch.cuda.empty_cache()

oof = np.clip(oof, CLIP_LO, CLIP_HI); test_pred = np.clip(test_pred, CLIP_LO, CLIP_HI)
print(f"\\n=== txt_xlmr unweighted-OOF (repeat-0, bizim fold) = {np.mean((y-oof)**2):.4f} ===")
''')

code('''
# === standalone rw-OOF (recency-weighted; cv.recency_weights ile BIREBIR) ===
tr_g = train[RECENCY_COL].value_counts(normalize=True)
te_g = test[RECENCY_COL].value_counts(normalize=True)
w = (train[RECENCY_COL].map(te_g).fillna(0.0) / train[RECENCY_COL].map(tr_g)).to_numpy(float); w = w/w.mean()
rw = float(np.sum(w*(y-oof)**2)/np.sum(w))
print(f"txt_xlmr standalone:  unweighted-OOF={np.mean((y-oof)**2):.4f}   recency-weighted-OOF={rw:.4f}")
print("  (referans metin tek-model rw: e5_ridge 158.46 / txt_ridge 168.02; mm rw 94.82)")
print("  NOT: repeat-0 standalone; NIHAI KARAR lokalde src/xlmr_blend.py BLEND etkisi + paired-test.")
''')

code('''
# === artefakt yaz + indir (repo artifacts/ icine koy) ===
assert oof.shape == (len(train),) and test_pred.shape == (len(test),)
assert np.isfinite(oof).all() and np.isfinite(test_pred).all()
assert oof.min()>=0 and oof.max()<=100 and test_pred.min()>=0 and test_pred.max()<=100
np.save("oof_xlmr.npy", oof.astype(np.float64))
np.save("test_xlmr.npy", test_pred.astype(np.float64))
print("YAZILDI: oof_xlmr.npy", oof.shape, "| test_xlmr.npy", test_pred.shape)
from google.colab import files
files.download("oof_xlmr.npy"); files.download("test_xlmr.npy")
''')

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).resolve().parents[1] / "colab_xlmr.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"YAZILDI: {out}  ({len(CELLS)} hucre)")
