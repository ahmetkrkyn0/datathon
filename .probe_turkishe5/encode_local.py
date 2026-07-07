"""Encode mentor feedback with the locally cached Turkish E5 Large model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".probe_turkishe5"
MODEL_DIR = Path(
    r"C:\Users\ahmet\.cache\huggingface\hub"
    r"\models--ytu-ce-cosmos--turkish-e5-large"
    r"\snapshots\02e2362d503bbdeafcb17143b2165c0743f9fdb1"
)
BATCH_SIZE = 32
MAX_LENGTH = 192


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def encode(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            embedding = F.normalize(
                mean_pool(hidden, tokens["attention_mask"]).float(),
                p=2,
                dim=1,
            )
        chunks.append(embedding.cpu().numpy().astype(np.float16))
        done = min(start + BATCH_SIZE, len(texts))
        if done % 1024 < BATCH_SIZE or done == len(texts):
            print(f"encoded {done:5d}/{len(texts)}", flush=True)
    return np.concatenate(chunks)


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not MODEL_DIR.exists():
        raise FileNotFoundError(MODEL_DIR)

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test_x.csv")
    train_text = train["mentor_feedback_text"].fillna("").astype(str).tolist()
    test_text = test["mentor_feedback_text"].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        dtype=torch.bfloat16,
    ).cuda().eval()

    print(f"train={len(train_text)} test={len(test_text)}", flush=True)
    train_embedding = encode(train_text, tokenizer, model)
    test_embedding = encode(test_text, tokenizer, model)
    np.save(OUT / "train.npy", train_embedding)
    np.save(OUT / "test.npy", test_embedding)
    print(
        f"saved train={train_embedding.shape} test={test_embedding.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
