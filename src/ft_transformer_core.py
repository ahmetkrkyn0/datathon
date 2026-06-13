"""
FT-Transformer (Feature Tokenizer + Transformer) — Colab/A100.

GBM'den FARKLI hipotez uzayi: her feature (sayisal + kategorik) bir token'a
gomulur, transformer attention ile feature-etkilesimlerini ogrenir. GBM'in
goremedigi duzgun/yuksek-dereceli etkilesimleri yakalayabilir -> blend'e ORTOGONAL
olma sansi (basit MLP'de residual kor 0 cikti; bu mimari farkli).

- Sayisal: QuantileTransformer(normal) -> her biri ogrenilen linear token
- Kategorik: embedding (department/tier/role/hobby/social/yil)
- [CLS] token + 3 transformer blok (MHA + FFN), head -> 1
- year-norm hedef + uniform (bizim en iyi ders), Tuna fold (5x3 -> repeat 0 hizli; tam 15 opsiyonel)
- bf16, AdamW, OneCycle

Cikti: ftt_oof.npy + ftt_test.npy
Colab girdileri: train.csv, test_x.csv, folds.parquet
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer

SEED = 42
N_REP = 3            # tam 5x3 icin 3; hizli icin 1
EPOCHS = 80
BATCH = 256
D_TOKEN = 64
N_BLOCKS = 3
N_HEADS = 8
DROP = 0.2
LR = 1e-3
WD = 1e-5
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.bfloat16 if (DEV == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

CAT_COLS = ["department", "university_tier", "target_role", "hobby",
            "preferred_social_media_platform", "application_year"]


def load():
    tr = pd.read_csv("train.csv", encoding="utf-8-sig")
    te = pd.read_csv("test_x.csv", encoding="utf-8-sig")
    y = tr["career_success_score"].values.astype(np.float32)
    num = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c])
           and c not in ("career_success_score",) + tuple(CAT_COLS)
           and c != "student_id"]
    return tr, te, y, num


class FTT(nn.Module):
    def __init__(self, n_num, cat_cards):
        super().__init__()
        self.n_num = n_num
        # sayisal: her feature icin ogrenilen agirlik+bias -> token
        self.num_w = nn.Parameter(torch.randn(n_num, D_TOKEN) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, D_TOKEN))
        # kategorik: embedding
        self.embs = nn.ModuleList([nn.Embedding(c, D_TOKEN) for c in cat_cards])
        self.cls = nn.Parameter(torch.randn(1, 1, D_TOKEN) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=D_TOKEN, nhead=N_HEADS, dim_feedforward=D_TOKEN * 2,
            dropout=DROP, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, N_BLOCKS)
        self.head = nn.Sequential(nn.LayerNorm(D_TOKEN), nn.GELU(),
                                  nn.Linear(D_TOKEN, 1))

    def forward(self, xn, xc):
        b = xn.size(0)
        # sayisal token: xn[b,n_num] -> [b,n_num,D]
        num_tok = xn.unsqueeze(-1) * self.num_w + self.num_b
        cat_tok = torch.stack([emb(xc[:, i]) for i, emb in enumerate(self.embs)], 1)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, num_tok, cat_tok], 1)
        x = self.tr(x)
        return self.head(x[:, 0]).squeeze(-1)


def main():
    print("device:", DEV, "| amp:", AMP)
    tr, te, y, num = load()
    yr_tr = tr["application_year"].values
    yr_te = te["application_year"].values

    # sayisal
    qt = QuantileTransformer(output_distribution="normal", random_state=SEED,
                             n_quantiles=min(1000, len(tr)))
    Xn_tr = qt.fit_transform(tr[num].fillna(tr[num].median())).astype(np.float32)
    Xn_te = qt.transform(te[num].fillna(tr[num].median())).astype(np.float32)
    # kategorik -> kod (train+test ortak)
    cat_cards = []
    Xc_tr = np.zeros((len(tr), len(CAT_COLS)), np.int64)
    Xc_te = np.zeros((len(te), len(CAT_COLS)), np.int64)
    for i, c in enumerate(CAT_COLS):
        both = pd.concat([tr[c], te[c]]).astype(str)
        cats = {v: k for k, v in enumerate(both.unique())}
        cat_cards.append(len(cats))
        Xc_tr[:, i] = tr[c].astype(str).map(cats).values
        Xc_te[:, i] = te[c].astype(str).map(cats).values
    print(f"sayisal {len(num)} | kategorik {CAT_COLS} kart {cat_cards}")

    folds = pd.read_parquet("folds.parquet")
    pos = {s: i for i, s in enumerate(tr["student_id"].values)}

    def val_idx(rep, fold):
        ids = folds[(folds.repeat == rep) & (folds.fold == fold)].student_id.values
        return np.array([pos[s] for s in ids])

    ym, ys = y.mean(), y.std()
    Xc_te_t = torch.tensor(Xc_te).to(DEV)
    Xn_te_t = torch.tensor(Xn_te).to(DEV)

    oof = np.zeros(len(tr), np.float32)
    oof_cnt = np.zeros(len(tr))
    test_pred = np.zeros(len(te), np.float32)
    n_models = 0

    for rep in range(N_REP):
        for fold in range(5):
            va = val_idx(rep, fold)
            trn = np.setdiff1d(np.arange(len(tr)), va)
            # year-norm hedef
            st = pd.DataFrame({"yil": yr_tr[trn], "y": y[trn]}).groupby("yil")["y"].agg(["mean", "std"])
            mu = pd.Series(yr_tr[trn]).map(st["mean"]).values
            sd = pd.Series(yr_tr[trn]).map(st["std"]).values
            yn = ((y[trn] - mu) / sd).astype(np.float32)
            mu_va = pd.Series(yr_tr[va]).map(st["mean"]).values
            sd_va = pd.Series(yr_tr[va]).map(st["std"]).values
            mu_te = pd.Series(yr_te).map(st["mean"]).values
            sd_te = pd.Series(yr_te).map(st["std"]).values

            ds = TensorDataset(torch.tensor(Xn_tr[trn]), torch.tensor(Xc_tr[trn]),
                               torch.tensor(yn))
            dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
            m = FTT(len(num), cat_cards).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=LR, total_steps=len(dl) * EPOCHS, pct_start=0.1)
            m.train()
            for ep in range(EPOCHS):
                for xn, xc, yb in dl:
                    xn, xc, yb = xn.to(DEV), xc.to(DEV), yb.to(DEV)
                    opt.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=AMP, enabled=(DEV == "cuda")):
                        loss = nn.functional.mse_loss(m(xn, xc), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step(); sched.step()
            m.eval()
            with torch.no_grad():
                xnv = torch.tensor(Xn_tr[va]).to(DEV); xcv = torch.tensor(Xc_tr[va]).to(DEV)
                with torch.autocast(device_type="cuda", dtype=AMP, enabled=(DEV == "cuda")):
                    vp = m(xnv, xcv).float().cpu().numpy()
                    tp = m(Xn_te_t, Xc_te_t).float().cpu().numpy()
            oof[va] += np.clip(vp * sd_va + mu_va, 0, 100)
            oof_cnt[va] += 1
            test_pred += np.clip(tp * sd_te + mu_te, 0, 100)
            n_models += 1
            vmse = ((y[va] - np.clip(vp * sd_va + mu_va, 0, 100)) ** 2).mean()
            print(f"rep{rep} fold{fold}: val MSE = {vmse:.2f}")
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()

    oof /= np.maximum(oof_cnt, 1)
    test_pred /= n_models
    print(f"\nFT-Transformer OOF MSE = {((y - oof) ** 2).mean():.2f}")
    np.save("ftt_oof.npy", oof)
    np.save("ftt_test.npy", test_pred)
    print("KAYDEDILDI -> ftt_oof.npy + ftt_test.npy")


if __name__ == "__main__":
    main()
