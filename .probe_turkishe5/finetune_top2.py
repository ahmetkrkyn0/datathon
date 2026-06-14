"""Five-fold Turkish E5 regression probe with only the top two layers unfrozen."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".probe_turkishe5"
MODEL_DIR = Path(
    r"C:\Users\ahmet\.cache\huggingface\hub"
    r"\models--ytu-ce-cosmos--turkish-e5-large"
    r"\snapshots\02e2362d503bbdeafcb17143b2165c0743f9fdb1"
)
SEED = 2026
EPOCHS = 3
MAX_LENGTH = 192
BATCH_SIZE = 32
TOP_LAYERS = 2
BACKBONE_LR = 2e-5
HEAD_LR = 8e-4
WEIGHT_DECAY = 0.01
DEVICE = "cuda"
AMP_DTYPE = torch.bfloat16


class TextDataset(Dataset):
    def __init__(
        self,
        encoding: dict[str, torch.Tensor],
        indices: np.ndarray,
        target: np.ndarray | None = None,
    ) -> None:
        self.encoding = encoding
        self.indices = indices
        self.target = target

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[item])
        result = {
            key: value[row]
            for key, value in self.encoding.items()
            if key != "token_type_ids"
        }
        if self.target is not None:
            result["labels"] = torch.tensor(
                self.target[row],
                dtype=torch.float32,
            )
        return result


class RegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for layer in self.backbone.encoder.layer[-TOP_LAYERS:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        cls = hidden[:, 0]
        return self.head(torch.cat((mean, cls), dim=1)).squeeze(-1)


def tokenize(
    tokenizer: AutoTokenizer,
    texts: pd.Series,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts.fillna("").astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


@torch.no_grad()
def predict(
    model: RegressionModel,
    encoding: dict[str, torch.Tensor],
    indices: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    loader = DataLoader(
        TextDataset(encoding, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    output = []
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            pred = model(ids, mask)
        output.append(pred.float().cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test_x.csv")
    y = train["career_success_score"].to_numpy(np.float32)
    train_year = train["application_year"].to_numpy()
    test_year = test["application_year"].to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    train_encoding = tokenize(tokenizer, train["mentor_feedback_text"])
    test_encoding = tokenize(tokenizer, test["mentor_feedback_text"])

    folds = pd.read_parquet(ROOT / "folds.parquet")
    id_to_row = {
        student_id: row
        for row, student_id in enumerate(train["student_id"].to_numpy())
    }
    fold_zero = folds[folds["repeat"] == 0]
    splits = [
        np.array(
            [
                id_to_row[student_id]
                for student_id in fold_zero[fold_zero["fold"] == fold][
                    "student_id"
                ]
            ],
            dtype=int,
        )
        for fold in sorted(fold_zero["fold"].unique())
    ]

    oof = np.zeros(len(train), dtype=np.float32)
    test_prediction = np.zeros(len(test), dtype=np.float32)
    all_rows = np.arange(len(train))
    for fold, valid in enumerate(splits):
        train_idx = np.setdiff1d(all_rows, valid)
        stats = (
            pd.DataFrame({"year": train_year[train_idx], "y": y[train_idx]})
            .groupby("year")["y"]
            .agg(["mean", "std"])
        )
        mean_train = pd.Series(train_year[train_idx]).map(stats["mean"]).to_numpy()
        std_train = pd.Series(train_year[train_idx]).map(stats["std"]).to_numpy()
        normalized = np.zeros(len(train), dtype=np.float32)
        normalized[train_idx] = (
            (y[train_idx] - mean_train) / np.maximum(std_train, 1e-6)
        ).astype(np.float32)
        fallback_mean = float(np.mean(y[train_idx]))
        fallback_std = float(np.std(y[train_idx]))
        mean_valid = (
            pd.Series(train_year[valid])
            .map(stats["mean"])
            .fillna(fallback_mean)
            .to_numpy()
        )
        std_valid = (
            pd.Series(train_year[valid])
            .map(stats["std"])
            .fillna(fallback_std)
            .to_numpy()
        )
        mean_test = (
            pd.Series(test_year)
            .map(stats["mean"])
            .fillna(fallback_mean)
            .to_numpy()
        )
        std_test = (
            pd.Series(test_year)
            .map(stats["std"])
            .fillna(fallback_std)
            .to_numpy()
        )

        model = RegressionModel().to(DEVICE)
        backbone_parameters = [
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": BACKBONE_LR},
                {"params": model.head.parameters(), "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        loader = DataLoader(
            TextDataset(train_encoding, train_idx, normalized),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[BACKBONE_LR, HEAD_LR],
            total_steps=len(loader) * EPOCHS,
            pct_start=0.08,
        )
        for epoch in range(EPOCHS):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                labels = batch["labels"].to(DEVICE, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    loss = torch.nn.functional.smooth_l1_loss(
                        model(ids, mask),
                        labels,
                        beta=1.0,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            print(
                f"fold={fold} epoch={epoch + 1}/{EPOCHS} "
                f"loss={np.mean(losses):.5f}",
                flush=True,
            )

        valid_normalized = predict(model, train_encoding, valid)
        oof[valid] = np.clip(
            valid_normalized * std_valid + mean_valid,
            0.0,
            100.0,
        )
        test_normalized = predict(
            model,
            test_encoding,
            np.arange(len(test)),
        )
        test_prediction += (
            np.clip(test_normalized * std_test + mean_test, 0.0, 100.0)
            / len(splits)
        )
        print(
            f"fold={fold} raw_mse={np.mean((y[valid] - oof[valid]) ** 2):.5f}",
            flush=True,
        )
        del model, optimizer, scheduler, loader
        gc.collect()
        torch.cuda.empty_cache()

    np.save(OUT / "top2_oof.npy", oof)
    np.save(OUT / "top2_test.npy", test_prediction)
    print(f"raw_oof_mse={np.mean((y - oof) ** 2):.5f}", flush=True)


if __name__ == "__main__":
    main()
