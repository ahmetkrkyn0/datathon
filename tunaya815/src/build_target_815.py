"""Build the validated 81.5-target stack and submission files.

The base is Tuna's 12-model nested blend. A small unconstrained Ridge meta-model
adds two fold-aligned models (full-finetuned XLM-R and FT-Transformer), then
strictly nested confidence gates make local corrections.

Outputs:
  submissions/TARGET815_robust.csv
  submissions/TARGET815_aggressive.csv
  target815_meta_oof.npy
  target815_robust_oof.npy
  target815_aggressive_oof.npy
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
TEAM_DIR = ROOT / "tunadan gelenler"
DELIVERY_DIR = ROOT / "tunadanistenenbelgeler"
SUBMISSION_DIR = ROOT / "submissions"

N_REPEATS = 3
N_SPLITS = 5
META_ALPHA = 0.01
Q_GRID = (0.75, 0.80, 0.85, 0.90, 0.95)
A_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

ROBUST_CHAIN = (("fullft", True),)
AGGRESSIVE_CHAIN = (
    ("fullft", True),
    ("lgbm_full", False),
    ("mmstrong", True),
)


def clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 100.0)


def weighted_mse(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average((y - clip(pred)) ** 2, weights=weight))


def make_meta_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=META_ALPHA))


def load_inputs():
    y = np.load(TEAM_DIR / "y.npy").astype(float)
    weight = np.load(TEAM_DIR / "w_recency.npy").astype(float)
    train_ids = np.load(TEAM_DIR / "student_id_train.npy", allow_pickle=True)
    test_ids = np.load(TEAM_DIR / "student_id_test.npy", allow_pickle=True)
    model_names = list(np.load(TEAM_DIR / "models.npy", allow_pickle=True))

    oof = {
        str(name): np.load(TEAM_DIR / f"oof_{name}.npy").astype(float)
        for name in model_names
    }
    test = {
        str(name): np.load(TEAM_DIR / f"test_{name}.npy").astype(float)
        for name in model_names
    }
    oof["ourteam_tf"] = np.load(ROOT / "ourteam_oof_tunafolds.npy").astype(float)
    test["ourteam_tf"] = np.load(ROOT / "ourteam_test_tunafolds.npy").astype(float)
    oof["fullft"] = np.load(ROOT / "fullft_oof.npy").astype(float)
    test["fullft"] = np.load(ROOT / "fullft_test.npy").astype(float)
    oof["ftt"] = np.load(ROOT / "ftt_oof.npy").astype(float)
    test["ftt"] = np.load(ROOT / "ftt_test.npy").astype(float)
    oof["mmstrong"] = np.load(ROOT / "mmstrong_oof.npy").astype(float)
    test["mmstrong"] = np.load(ROOT / "mmstrong_test.npy").astype(float)

    base_oof = np.load(DELIVERY_DIR / "oof_blend.npy").astype(float)
    base_test = np.load(DELIVERY_DIR / "test_blend.npy").astype(float)
    folds = pd.read_parquet(TEAM_DIR / "folds.parquet")

    expected = len(y)
    for name, values in oof.items():
        if len(values) != expected:
            raise ValueError(f"OOF length mismatch for {name}: {len(values)} != {expected}")
    for name, values in test.items():
        if len(values) != len(test_ids):
            raise ValueError(
                f"Test length mismatch for {name}: {len(values)} != {len(test_ids)}"
            )

    return y, weight, train_ids, test_ids, folds, oof, test, base_oof, base_test


def fold_vectors(folds: pd.DataFrame, train_ids: np.ndarray) -> list[np.ndarray]:
    position = {student_id: i for i, student_id in enumerate(train_ids)}
    vectors = []
    for repeat in range(N_REPEATS):
        vector = np.full(len(train_ids), -1, dtype=int)
        frame = folds[folds["repeat"] == repeat]
        for student_id, fold in zip(frame["student_id"], frame["fold"]):
            vector[position[student_id]] = int(fold)
        if np.any(vector < 0):
            raise ValueError(f"Missing fold assignments for repeat {repeat}")
        vectors.append(vector)
    return vectors


def meta_features(
    predictions: dict[str, np.ndarray],
    base: np.ndarray,
) -> np.ndarray:
    members = list(np.load(TEAM_DIR / "models.npy", allow_pickle=True))
    names = [str(name) for name in members] + ["ourteam_tf", "fullft", "ftt"]
    matrix = np.column_stack([predictions[name] for name in names])
    return np.column_stack(
        [
            matrix,
            base,
            matrix.mean(axis=1),
            matrix.std(axis=1),
            matrix.min(axis=1),
            matrix.max(axis=1),
            np.median(matrix, axis=1),
        ]
    )


def apply_gate(
    base: np.ndarray,
    signal: np.ndarray,
    center: float,
    threshold: float,
    strength: float,
    upward: bool,
) -> np.ndarray:
    confidence = np.abs(signal - center)
    direction = signal > center if upward else signal < center
    mask = (confidence >= threshold) & direction
    return clip(base + mask * strength * (signal - base))


def select_gate(
    base: np.ndarray,
    signal: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    center: float,
    upward: bool,
) -> tuple[float, float]:
    confidence = np.abs(signal - center)
    direction = signal > center if upward else signal < center
    pool = confidence[direction]
    best = (np.inf, np.inf, 0.0)
    for quantile in Q_GRID:
        threshold = float(np.quantile(pool, quantile)) if pool.size else np.inf
        for strength in A_GRID:
            pred = apply_gate(
                base, signal, center, threshold, strength, upward
            )
            score = weighted_mse(y, pred, weight)
            if score < best[0]:
                best = (score, threshold, strength)
    return best[1], best[2]


def strict_nested_predictions(
    features: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    fold_ids: Sequence[np.ndarray],
    signals: dict[str, np.ndarray],
    chain: Sequence[tuple[str, bool]],
) -> np.ndarray:
    total = np.zeros(len(y), dtype=float)
    count = np.zeros(len(y), dtype=float)

    for repeat, vector in enumerate(fold_ids):
        for fold in range(N_SPLITS):
            valid = np.where(vector == fold)[0]
            train = np.where(vector != fold)[0]
            fold_center = float(np.average(y[train], weights=weight[train]))

            inner_pred = np.zeros(len(train), dtype=float)
            inner_cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=4200 + repeat * 10 + fold,
            )
            for inner_train, inner_valid in inner_cv.split(train):
                model = make_meta_model()
                model.fit(
                    features[train[inner_train]],
                    y[train[inner_train]],
                    ridge__sample_weight=weight[train[inner_train]],
                )
                inner_pred[inner_valid] = model.predict(
                    features[train[inner_valid]]
                )

            model = make_meta_model()
            model.fit(
                features[train],
                y[train],
                ridge__sample_weight=weight[train],
            )
            valid_pred = clip(model.predict(features[valid]))

            for signal_name, upward in chain:
                threshold, strength = select_gate(
                    inner_pred,
                    signals[signal_name][train],
                    y[train],
                    weight[train],
                    fold_center,
                    upward,
                )
                inner_pred = apply_gate(
                    inner_pred,
                    signals[signal_name][train],
                    fold_center,
                    threshold,
                    strength,
                    upward,
                )
                valid_pred = apply_gate(
                    valid_pred,
                    signals[signal_name][valid],
                    fold_center,
                    threshold,
                    strength,
                    upward,
                )

            total[valid] += valid_pred
            count[valid] += 1.0

    return clip(total / count)


def cell_scores(
    pred: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    fold_ids: Sequence[np.ndarray],
) -> np.ndarray:
    scores = []
    for vector in fold_ids:
        for fold in range(N_SPLITS):
            idx = vector == fold
            scores.append(weighted_mse(y[idx], pred[idx], weight[idx]))
    return np.asarray(scores)


def report(
    name: str,
    pred: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    fold_ids: Sequence[np.ndarray],
) -> None:
    score = weighted_mse(y, pred, weight)
    base_score = weighted_mse(y, base, weight)
    candidate_cells = cell_scores(pred, y, weight, fold_ids)
    base_cells = cell_scores(base, y, weight, fold_ids)
    delta = candidate_cells - base_cells
    p_value = float(stats.ttest_rel(candidate_cells, base_cells).pvalue)
    per_repeat = [
        int(np.sum(delta[i * N_SPLITS : (i + 1) * N_SPLITS] < 0))
        for i in range(N_REPEATS)
    ]
    print(
        f"{name:18s} rw={score:.5f} delta={score - base_score:+.5f} "
        f"cells={int(np.sum(delta < 0))}/15 repeats={per_repeat} p={p_value:.3g}"
    )


def frozen_test_prediction(
    train_features: np.ndarray,
    test_features: np.ndarray,
    meta_oof: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    oof_signals: dict[str, np.ndarray],
    test_signals: dict[str, np.ndarray],
    chain: Sequence[tuple[str, bool]],
    center: float,
) -> tuple[np.ndarray, list[tuple[str, str, float, float]]]:
    model = make_meta_model()
    model.fit(train_features, y, ridge__sample_weight=weight)
    train_stage = meta_oof.copy()
    test_stage = clip(model.predict(test_features))
    selected = []

    for signal_name, upward in chain:
        threshold, strength = select_gate(
            train_stage,
            oof_signals[signal_name],
            y,
            weight,
            center,
            upward,
        )
        train_stage = apply_gate(
            train_stage,
            oof_signals[signal_name],
            center,
            threshold,
            strength,
            upward,
        )
        test_stage = apply_gate(
            test_stage,
            test_signals[signal_name],
            center,
            threshold,
            strength,
            upward,
        )
        selected.append(
            (
                signal_name,
                "up" if upward else "down",
                threshold,
                strength,
            )
        )

    return test_stage, selected


def save_submission(path: Path, test_ids: np.ndarray, pred: np.ndarray) -> None:
    submission = pd.DataFrame(
        {
            "student_id": test_ids,
            "career_success_score": clip(pred).round(4),
        }
    )
    submission.to_csv(path, index=False)
    print(
        f"wrote {path} | rows={len(submission)} "
        f"mean={submission['career_success_score'].mean():.4f}"
    )


def main() -> None:
    (
        y,
        weight,
        train_ids,
        test_ids,
        folds,
        oof,
        test,
        base_oof,
        base_test,
    ) = load_inputs()
    fold_ids = fold_vectors(folds, train_ids)
    center = float(np.average(y, weights=weight))
    train_features = meta_features(oof, base_oof)
    test_features = meta_features(test, base_test)

    meta_oof = strict_nested_predictions(
        train_features, y, weight, fold_ids, oof, ()
    )
    robust_oof = strict_nested_predictions(
        train_features, y, weight, fold_ids, oof, ROBUST_CHAIN
    )
    aggressive_oof = strict_nested_predictions(
        train_features, y, weight, fold_ids, oof, AGGRESSIVE_CHAIN
    )

    report("base", base_oof, base_oof, y, weight, fold_ids)
    report("meta", meta_oof, base_oof, y, weight, fold_ids)
    report("robust", robust_oof, base_oof, y, weight, fold_ids)
    report("aggressive", aggressive_oof, base_oof, y, weight, fold_ids)

    robust_test, robust_params = frozen_test_prediction(
        train_features,
        test_features,
        meta_oof,
        y,
        weight,
        oof,
        test,
        ROBUST_CHAIN,
        center,
    )
    aggressive_test, aggressive_params = frozen_test_prediction(
        train_features,
        test_features,
        meta_oof,
        y,
        weight,
        oof,
        test,
        AGGRESSIVE_CHAIN,
        center,
    )
    print("robust frozen gates:", robust_params)
    print("aggressive frozen gates:", aggressive_params)

    np.save(ROOT / "target815_meta_oof.npy", meta_oof.astype(np.float32))
    np.save(ROOT / "target815_robust_oof.npy", robust_oof.astype(np.float32))
    np.save(
        ROOT / "target815_aggressive_oof.npy",
        aggressive_oof.astype(np.float32),
    )
    SUBMISSION_DIR.mkdir(exist_ok=True)
    save_submission(
        SUBMISSION_DIR / "TARGET815_robust.csv", test_ids, robust_test
    )
    save_submission(
        SUBMISSION_DIR / "TARGET815_aggressive.csv",
        test_ids,
        aggressive_test,
    )


if __name__ == "__main__":
    main()
