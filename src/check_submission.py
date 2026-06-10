"""Submission dosyasi dogrulayici.

Kullanim: python src/check_submission.py submissions/submission_v4.csv
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def check(path):
    sub = pd.read_csv(path)
    samp = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    test = pd.read_csv(ROOT / "data" / "test_x.csv")

    ok = True
    def row(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"  [{'OK' if passed else 'HATA'}] {name} {detail}")

    row("kolonlar", list(sub.columns) == list(samp.columns), str(list(sub.columns)))
    row("satir sayisi 10000", len(sub) == 10000, f"({len(sub)})")
    row("ID'ler test ile ayni sirada",
        (sub["student_id"].values == test["student_id"].values).all())
    row("NaN yok", not sub["career_success_score"].isnull().any())
    mn, mx = sub["career_success_score"].min(), sub["career_success_score"].max()
    row("0-100 araliginda", (mn >= 0) and (mx <= 100), f"({mn:.2f} - {mx:.2f})")
    print(f"  ortalama: {sub['career_success_score'].mean():.2f}")
    print("SONUC:", "GONDERILEBILIR" if ok else "GONDERME — SORUN VAR")
    return ok


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "submissions/submission_v4.csv"
    check(p)
