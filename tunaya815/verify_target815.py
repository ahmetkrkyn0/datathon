"""Independent arithmetic verification for the TARGET815 delivery package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
TEAM = ROOT / "tunadan gelenler"
BASE = ROOT / "tunadanistenenbelgeler"


def wmse(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average((y - np.clip(pred, 0, 100)) ** 2, weights=weight))


def fold_vectors(folds: pd.DataFrame, student_ids: np.ndarray) -> list[np.ndarray]:
    position = {student_id: i for i, student_id in enumerate(student_ids)}
    vectors = []
    for repeat in sorted(folds["repeat"].unique()):
        vector = np.full(len(student_ids), -1, dtype=int)
        frame = folds[folds["repeat"] == repeat]
        for student_id, fold in zip(frame["student_id"], frame["fold"]):
            vector[position[student_id]] = int(fold)
        if np.any(vector < 0):
            raise AssertionError(f"Missing fold assignments in repeat {repeat}")
        vectors.append(vector)
    return vectors


def cell_scores(
    y: np.ndarray,
    pred: np.ndarray,
    weight: np.ndarray,
    fold_ids: list[np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            wmse(y[vector == fold], pred[vector == fold], weight[vector == fold])
            for vector in fold_ids
            for fold in range(5)
        ]
    )


def verify_submission(path: Path, test_ids: np.ndarray) -> None:
    frame = pd.read_csv(path)
    expected_columns = ["student_id", "career_success_score"]
    assert list(frame.columns) == expected_columns
    assert len(frame) == len(test_ids) == 10_000
    assert np.array_equal(frame["student_id"].to_numpy(), test_ids)
    pred = frame["career_success_score"].to_numpy(float)
    assert np.isfinite(pred).all()
    assert pred.min() >= 0 and pred.max() <= 100
    assert not frame["student_id"].duplicated().any()
    print(
        f"OK submission: {path.name} | mean={pred.mean():.4f} "
        f"range=[{pred.min():.4f}, {pred.max():.4f}]"
    )


def main() -> None:
    y = np.load(TEAM / "y.npy").astype(float)
    weight = np.load(TEAM / "w_recency.npy").astype(float)
    train_ids = np.load(TEAM / "student_id_train.npy", allow_pickle=True)
    test_ids = np.load(TEAM / "student_id_test.npy", allow_pickle=True)
    folds = pd.read_parquet(TEAM / "folds.parquet")
    fold_ids = fold_vectors(folds, train_ids)

    predictions = {
        "base": np.load(BASE / "oof_blend.npy").astype(float),
        "meta": np.load(ROOT / "target815_meta_oof.npy").astype(float),
        "robust": np.load(ROOT / "target815_robust_oof.npy").astype(float),
        "aggressive": np.load(ROOT / "target815_aggressive_oof.npy").astype(float),
    }
    for name, pred in predictions.items():
        assert pred.shape == y.shape
        assert np.isfinite(pred).all()
        assert pred.min() >= 0 and pred.max() <= 100
        print(f"{name:10s} rw-MSE={wmse(y, pred, weight):.5f}")

    base_cells = cell_scores(y, predictions["base"], weight, fold_ids)
    for name in ("meta", "robust", "aggressive"):
        cells = cell_scores(y, predictions[name], weight, fold_ids)
        delta = cells - base_cells
        p_value = float(stats.ttest_rel(cells, base_cells).pvalue)
        repeat_wins = [
            int(np.sum(delta[i * 5 : (i + 1) * 5] < 0))
            for i in range(3)
        ]
        print(
            f"{name:10s} delta={wmse(y, predictions[name], weight) - wmse(y, predictions['base'], weight):+.5f} "
            f"wins={int(np.sum(delta < 0))}/15 repeats={repeat_wins} p={p_value:.3g}"
        )

    # Row bootstrap does not treat the 15 repeated folds as independent samples.
    rng = np.random.default_rng(20260614)
    delta = (y - predictions["aggressive"]) ** 2 - (y - predictions["base"]) ** 2
    bootstrap = np.empty(5_000, dtype=float)
    for i in range(len(bootstrap)):
        index = rng.integers(0, len(y), len(y))
        bootstrap[i] = np.average(delta[index], weights=weight[index])
    low, median, high = np.quantile(bootstrap, [0.025, 0.5, 0.975])
    print(
        f"aggressive row-bootstrap delta 95% CI=[{low:.5f}, {high:.5f}] "
        f"median={median:.5f}"
    )

    verify_submission(ROOT / "submissions" / "TARGET815_robust.csv", test_ids)
    verify_submission(ROOT / "submissions" / "TARGET815_aggressive.csv", test_ids)

    print("\nPROVENANCE WARNING:")
    print("- FT-Transformer was generated with all 3 repeats.")
    print("- fullft and mmstrong were generated with repeat 0 only.")
    print("- Therefore 81.48840 is a strong proxy, not a guaranteed leaderboard score.")


if __name__ == "__main__":
    main()
