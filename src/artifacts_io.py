"""
Faz 2 — artefakt yazim sozlesmesi (SPEC §7). Idempotent upsert: bir model'i iki kez
calistirmak cv_scores.csv / cv_log.csv'de TEK satir birakir (reproducibility dostu).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import cv

CV_SCORES_PATH = cv.ARTIFACTS_DIR / "cv_scores.csv"
CV_LOG_PATH = cv.REPORTS_DIR / "cv_log.csv"

CV_SCORES_COLS = ["model", "cv_mse_mean", "cv_mse_std", "best_iteration_mean"]
CV_LOG_COLS = [
    "model",
    "cv_mse_mean",  # = compute_cv_mse(oof) mean (avg-oof; cv_scores.csv ile ayni, DoD-4)
    "cv_mse_std",   # = compute_cv_mse(oof) std (avg-oof 15 hucre)
    "n_folds",
    "best_iteration_mean",
    "genuine15_mean",   # 15 fold modelinin tek-basina val MSE ortalamasi (standart repeated-CV)
    "genuine15_std",    # 15 fold modelinin tek-basina val MSE std'si
    "single5fold_std",  # repeat0'in 5 fold-MSE std'si (SPEC §2 'tek 5-fold fold-std ~4.68' referansi)
    "fold_mse_list",        # compute_cv_mse 15 hucre MSE (toplami cv_mse_mean'e esit)
    "genuine_fold_mse_list",  # 15 genuine fold-MSE
    "best_iterations_list",   # 15 best_iteration
    "note",
]


def _upsert(path: Path, cols: list[str], row: dict) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row], columns=cols)
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["model"] != row["model"]]  # ayni model varsa cikar (upsert)
        df = new if df.empty else pd.concat([df, new], ignore_index=True)
    else:
        df = new
    df = df.sort_values("model").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def save_oof_test(model: str, oof: np.ndarray, test: np.ndarray) -> None:
    cv.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cv.ARTIFACTS_DIR / f"oof_{model}.npy", np.asarray(oof, dtype=float))
    np.save(cv.ARTIFACTS_DIR / f"test_{model}.npy", np.asarray(test, dtype=float))


def write_cv_score(model: str, cv_mse_mean: float, cv_mse_std: float, best_iteration_mean: float) -> None:
    _upsert(
        CV_SCORES_PATH,
        CV_SCORES_COLS,
        dict(
            model=model,
            cv_mse_mean=round(float(cv_mse_mean), 6),
            cv_mse_std=round(float(cv_mse_std), 6),
            best_iteration_mean=round(float(best_iteration_mean), 2),
        ),
    )


def write_cv_log(
    model: str,
    cv_mse_mean: float,
    cv_mse_std: float,
    fold_mse: list[float],
    best_iterations: list,
    best_iteration_mean: float,
    genuine_fold_mse: list[float] | None = None,
    single5fold_std: float | None = None,
    note: str = "",
) -> None:
    import numpy as np

    g_mean = g_std = None
    if genuine_fold_mse is not None and len(genuine_fold_mse):
        g_mean = round(float(np.mean(genuine_fold_mse)), 6)
        g_std = round(float(np.std(genuine_fold_mse)), 6)

    _upsert(
        CV_LOG_PATH,
        CV_LOG_COLS,
        dict(
            model=model,
            cv_mse_mean=round(float(cv_mse_mean), 6),
            cv_mse_std=round(float(cv_mse_std), 6),
            n_folds=len(fold_mse),
            best_iteration_mean=round(float(best_iteration_mean), 2),
            genuine15_mean=g_mean,
            genuine15_std=g_std,
            single5fold_std=None if single5fold_std is None else round(float(single5fold_std), 6),
            fold_mse_list=json.dumps([round(float(v), 6) for v in fold_mse]),
            genuine_fold_mse_list=(
                None if genuine_fold_mse is None
                else json.dumps([round(float(v), 6) for v in genuine_fold_mse])
            ),
            best_iterations_list=json.dumps([None if b is None else int(b) for b in best_iterations]),
            note=note,
        ),
    )
