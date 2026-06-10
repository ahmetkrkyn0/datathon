"""
Faz 2/7 — Reproducibility testi (SPEC §8 DoD-9 / MASTERPLAN Gun 5).

    python src/repro_test.py            # anchor (lgbm_num) — DoD-9 varsayilan
    python src/repro_test.py finals     # catboost_full (SUB-1) + ensemble/blend (SUB-2)
    python src/repro_test.py full       # anchor + lgbm_full + finals (final pipeline kapsami)

Her hedef script'i IKI kez ayri (taze) Python surecinde calistirir; her kosuda:
  * artifacts/oof_{model}.npy ve test_{model}.npy'nin SHA-256'si
  * artifacts/cv_scores.csv'deki cv_mse_mean / cv_mse_std
Iki kosu BIREBIR ayni olmali (deterministik). Aksi halde assert hata firlatir.

NOT (CatBoost): thread_count=6 HARDCODE (catboost_full.py); determinizm ayni
(seed, thread_count) ciftine baglidir. Env'deki OMP=1 LightGBM/sklearn icindir,
CatBoost kendi havuzunu thread_count'tan kurar.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

import cv

SRC = Path(__file__).resolve().parent
SCORES = cv.ARTIFACTS_DIR / "cv_scores.csv"

# (model_adi, script) — model_adi: oof_/test_ npy adlari + cv_scores satiri.
TARGETS: dict[str, list[tuple[str, str]]] = {
    "anchor": [("lgbm_num", "anchor_lgbm_num.py")],
    "finals": [("catboost_full", "catboost_full.py"), ("blend", "ensemble.py")],
}
TARGETS["full"] = [TARGETS["anchor"][0], ("lgbm_full", "lgbm_full.py"), *TARGETS["finals"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_once(model: str, script: str, tag: str) -> dict:
    # Taze surec + sabit hash/thread env (HistGBR/sklearn OpenMP/BLAS determinizmi icin de).
    env = dict(
        os.environ,
        PYTHONHASHSEED="42",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    r = subprocess.run(
        [sys.executable, str(SRC / script)],
        cwd=str(SRC),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"[{tag}] {script} kosusu basarisiz (rc={r.returncode}).")
    row = pd.read_csv(SCORES)
    row = row[row["model"] == model].iloc[0]
    return dict(
        oof=sha256(cv.ARTIFACTS_DIR / f"oof_{model}.npy"),
        test=sha256(cv.ARTIFACTS_DIR / f"test_{model}.npy"),
        cv_mse_mean=float(row["cv_mse_mean"]),
        cv_mse_std=float(row["cv_mse_std"]),
    )


def check_model(model: str, script: str) -> None:
    print(f"\n[repro] === {model} ({script}) — 2 taze kosu ===", flush=True)
    a = run_once(model, script, f"{model}/kosu-1")
    b = run_once(model, script, f"{model}/kosu-2")

    print("                    kosu-1                                  kosu-2")
    print(f"oof  sha256   {a['oof'][:24]}...   {b['oof'][:24]}...")
    print(f"test sha256   {a['test'][:24]}...   {b['test'][:24]}...")
    print(f"cv_mse_mean   {a['cv_mse_mean']:.6f}                             {b['cv_mse_mean']:.6f}")
    print(f"cv_mse_std    {a['cv_mse_std']:.6f}                              {b['cv_mse_std']:.6f}")

    assert a["oof"] == b["oof"], f"oof_{model}.npy iki kosuda FARKLI (determinizm kirik)."
    assert a["test"] == b["test"], f"test_{model}.npy iki kosuda FARKLI (determinizm kirik)."
    assert a["cv_mse_mean"] == b["cv_mse_mean"], f"{model}: cv_mse_mean iki kosuda farkli."
    assert a["cv_mse_std"] == b["cv_mse_std"], f"{model}: cv_mse_std iki kosuda farkli."
    print(f"[repro] {model}: iki taze kosu BIREBIR ayni.", flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "anchor"
    assert mode in TARGETS, f"mode {mode!r} taninmadi; {list(TARGETS)} birinden biri olmali."
    for model, script in TARGETS[mode]:
        check_model(model, script)
    print(f"\n[repro] DoD-9 GECTI ({mode}): tum hedefler iki taze kosuda BIREBIR ayni "
          "(oof/test SHA-256 + cv_mse_mean/std).")


if __name__ == "__main__":
    main()
