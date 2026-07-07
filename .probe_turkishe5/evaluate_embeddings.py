"""Evaluate frozen Turkish E5 embeddings on the shared Tuna folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".probe_turkishe5"
TEAM = ROOT / "tunadan gelenler"
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
N_FOLDS = 5


def wmse(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average((y - pred) ** 2, weights=weight))


def fold_vectors(folds: pd.DataFrame, ids: np.ndarray) -> list[np.ndarray]:
    position = {student_id: i for i, student_id in enumerate(ids)}
    result = []
    for repeat in sorted(folds["repeat"].unique()):
        vector = np.full(len(ids), -1, dtype=np.int8)
        frame = folds[folds["repeat"] == repeat]
        for student_id, fold in zip(frame["student_id"], frame["fold"]):
            vector[position[student_id]] = int(fold)
        if np.any(vector < 0):
            raise ValueError(f"Missing assignments for repeat {repeat}")
        result.append(vector)
    return result


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model.predict(x_valid)


def nested_alpha(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    fold_vector: np.ndarray,
    weight: np.ndarray,
    year: np.ndarray,
    year_normalized: bool,
    weighted_fit: bool,
) -> float:
    scores = {alpha: [] for alpha in ALPHAS}
    inner_folds = fold_vector[train_idx]
    for inner_fold in range(N_FOLDS):
        inner_valid = train_idx[inner_folds == inner_fold]
        inner_train = train_idx[inner_folds != inner_fold]
        if not len(inner_valid):
            continue
        if year_normalized:
            stats = (
                pd.DataFrame({"year": year[inner_train], "y": y[inner_train]})
                .groupby("year")["y"]
                .agg(["mean", "std"])
            )
            mean_train = pd.Series(year[inner_train]).map(stats["mean"]).to_numpy()
            std_train = (
                pd.Series(year[inner_train]).map(stats["std"]).fillna(1.0).to_numpy()
            )
            target = (y[inner_train] - mean_train) / np.maximum(std_train, 1e-6)
            mean_valid = (
                pd.Series(year[inner_valid])
                .map(stats["mean"])
                .fillna(float(np.mean(y[inner_train])))
                .to_numpy()
            )
            std_valid = (
                pd.Series(year[inner_valid])
                .map(stats["std"])
                .fillna(float(np.std(y[inner_train])))
                .to_numpy()
            )
        else:
            target = y[inner_train]
        fit_weight = weight[inner_train] if weighted_fit else None
        for alpha in ALPHAS:
            pred = fit_predict(
                x[inner_train],
                target,
                x[inner_valid],
                alpha,
                fit_weight,
            )
            if year_normalized:
                pred = pred * std_valid + mean_valid
            scores[alpha].append(wmse(y[inner_valid], pred, weight[inner_valid]))
    return min(ALPHAS, key=lambda alpha: np.mean(scores[alpha]))


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    test_x: np.ndarray,
    weight: np.ndarray,
    year: np.ndarray,
    test_year: np.ndarray,
    fold_ids: list[np.ndarray],
    year_normalized: bool,
    weighted_fit: bool,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    repeated_oof = []
    test_predictions = []
    chosen_alphas = []
    for repeat, fold_vector in enumerate(fold_ids):
        oof = np.zeros(len(y), dtype=float)
        for fold in range(N_FOLDS):
            valid = np.where(fold_vector == fold)[0]
            train = np.where(fold_vector != fold)[0]
            alpha = nested_alpha(
                x,
                y,
                train,
                fold_vector,
                weight,
                year,
                year_normalized,
                weighted_fit,
            )
            chosen_alphas.append(alpha)
            if year_normalized:
                stats = (
                    pd.DataFrame({"year": year[train], "y": y[train]})
                    .groupby("year")["y"]
                    .agg(["mean", "std"])
                )
                mean_train = pd.Series(year[train]).map(stats["mean"]).to_numpy()
                std_train = (
                    pd.Series(year[train]).map(stats["std"]).fillna(1.0).to_numpy()
                )
                target = (y[train] - mean_train) / np.maximum(std_train, 1e-6)
                fallback_mean = float(np.mean(y[train]))
                fallback_std = float(np.std(y[train]))
                mean_valid = (
                    pd.Series(year[valid])
                    .map(stats["mean"])
                    .fillna(fallback_mean)
                    .to_numpy()
                )
                std_valid = (
                    pd.Series(year[valid])
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
            else:
                target = y[train]
            fit_weight = weight[train] if weighted_fit else None
            valid_pred = fit_predict(x[train], target, x[valid], alpha, fit_weight)
            test_pred = fit_predict(x[train], target, test_x, alpha, fit_weight)
            if year_normalized:
                valid_pred = valid_pred * std_valid + mean_valid
                test_pred = test_pred * std_test + mean_test
            oof[valid] = np.clip(valid_pred, 0.0, 100.0)
            test_predictions.append(np.clip(test_pred, 0.0, 100.0))
            print(
                f"repeat={repeat} fold={fold} alpha={alpha:g}",
                flush=True,
            )
        repeated_oof.append(oof)
    return (
        np.mean(repeated_oof, axis=0),
        np.mean(test_predictions, axis=0),
        chosen_alphas,
    )


def best_nested_blend(
    base: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    fold_ids: list[np.ndarray],
) -> tuple[np.ndarray, list[float]]:
    grid = np.linspace(0.0, 0.20, 21)
    repeated = []
    selected = []
    for fold_vector in fold_ids:
        oof = np.zeros(len(y), dtype=float)
        for fold in range(N_FOLDS):
            valid = fold_vector == fold
            train = ~valid
            alpha = min(
                grid,
                key=lambda value: wmse(
                    y[train],
                    (1.0 - value) * base[train] + value * candidate[train],
                    weight[train],
                ),
            )
            selected.append(float(alpha))
            oof[valid] = (
                (1.0 - alpha) * base[valid] + alpha * candidate[valid]
            )
        repeated.append(oof)
    return np.mean(repeated, axis=0), selected


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test_x.csv")
    x = np.load(OUT / "train.npy").astype(np.float32)
    test_x = np.load(OUT / "test.npy").astype(np.float32)
    y = np.load(TEAM / "y.npy").astype(float)
    weight = np.load(TEAM / "w_recency.npy").astype(float)
    ids = np.load(TEAM / "student_id_train.npy", allow_pickle=True)
    folds = pd.read_parquet(TEAM / "folds.parquet")
    fold_ids = fold_vectors(folds, ids)
    aggressive = np.load(ROOT / "target815_aggressive_oof.npy").astype(float)

    variants = (
        ("raw_uniform", False, False),
        ("raw_weighted", False, True),
        ("yearnorm_uniform", True, False),
        ("yearnorm_weighted", True, True),
    )
    results = []
    for name, year_normalized, weighted_fit in variants:
        print(f"\n=== {name} ===", flush=True)
        oof, test_pred, alphas = cross_validate(
            x,
            y,
            test_x,
            weight,
            train["application_year"].to_numpy(),
            test["application_year"].to_numpy(),
            fold_ids,
            year_normalized,
            weighted_fit,
        )
        np.save(OUT / f"{name}_oof.npy", oof.astype(np.float32))
        np.save(OUT / f"{name}_test.npy", test_pred.astype(np.float32))
        score = wmse(y, oof, weight)
        blend, blend_weights = best_nested_blend(
            aggressive,
            oof,
            y,
            weight,
            fold_ids,
        )
        blend_score = wmse(y, blend, weight)
        cell_wins = sum(
            wmse(y[v == f], blend[v == f], weight[v == f])
            < wmse(y[v == f], aggressive[v == f], weight[v == f])
            for v in fold_ids
            for f in range(N_FOLDS)
        )
        p_value = binomtest(cell_wins, 15, 0.5, alternative="greater").pvalue
        print(
            f"{name}: standalone={score:.5f} "
            f"nested_blend={blend_score:.5f} "
            f"delta={blend_score - wmse(y, aggressive, weight):+.5f} "
            f"wins={cell_wins}/15 p={p_value:.4f} "
            f"blend_mean={np.mean(blend_weights):.4f}",
            flush=True,
        )
        results.append(
            {
                "name": name,
                "standalone": score,
                "nested_blend": blend_score,
                "delta_vs_aggressive": blend_score
                - wmse(y, aggressive, weight),
                "cell_wins": cell_wins,
                "p_value": p_value,
                "ridge_alpha_mean": float(np.mean(alphas)),
                "blend_weight_mean": float(np.mean(blend_weights)),
            }
        )
    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
    print("\n", pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
