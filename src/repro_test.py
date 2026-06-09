"""
Faz 2 — Reproducibility testi (SPEC §8 DoD-9 / MASTERPLAN Gun 5).

    python src/repro_test.py

Anchor'i IKI kez ayri (taze) Python surecinde calistirir; her kosuda:
  * artifacts/oof_lgbm_num.npy ve test_lgbm_num.npy'nin SHA-256'si
  * artifacts/cv_scores.csv'deki cv_mse_mean / cv_mse_std
Iki kosu BIREBIR ayni olmali (deterministik). Aksi halde assert hata firlatir.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

import cv

ANCHOR = Path(__file__).resolve().parent / "anchor_lgbm_num.py"
OOF = cv.ARTIFACTS_DIR / "oof_lgbm_num.npy"
TEST = cv.ARTIFACTS_DIR / "test_lgbm_num.npy"
SCORES = cv.ARTIFACTS_DIR / "cv_scores.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_once(tag: str) -> dict:
    # Taze surec + sabit hash/thread env (HistGBR/CatBoost OpenMP/BLAS determinizmi icin de).
    env = dict(
        os.environ,
        PYTHONHASHSEED="42",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    r = subprocess.run(
        [sys.executable, str(ANCHOR)],
        cwd=str(ANCHOR.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"[{tag}] anchor kosusu basarisiz (rc={r.returncode}).")
    row = pd.read_csv(SCORES)
    row = row[row["model"] == "lgbm_num"].iloc[0]
    return dict(
        oof=sha256(OOF),
        test=sha256(TEST),
        cv_mse_mean=float(row["cv_mse_mean"]),
        cv_mse_std=float(row["cv_mse_std"]),
    )


def main() -> None:
    a = run_once("kosu-1")
    b = run_once("kosu-2")

    print("                    kosu-1                                  kosu-2")
    print(f"oof  sha256   {a['oof'][:24]}...   {b['oof'][:24]}...")
    print(f"test sha256   {a['test'][:24]}...   {b['test'][:24]}...")
    print(f"cv_mse_mean   {a['cv_mse_mean']:.6f}                             {b['cv_mse_mean']:.6f}")
    print(f"cv_mse_std    {a['cv_mse_std']:.6f}                              {b['cv_mse_std']:.6f}")

    assert a["oof"] == b["oof"], "oof_lgbm_num.npy iki kosuda FARKLI (determinizm kirik)."
    assert a["test"] == b["test"], "test_lgbm_num.npy iki kosuda FARKLI (determinizm kirik)."
    assert a["cv_mse_mean"] == b["cv_mse_mean"], "cv_mse_mean iki kosuda farkli."
    assert a["cv_mse_std"] == b["cv_mse_std"], "cv_mse_std iki kosuda farkli."

    print("\n[repro] DoD-9 GECTI: iki taze kosu BIREBIR ayni (oof/test SHA-256 + cv_mse_mean/std).")


if __name__ == "__main__":
    main()
