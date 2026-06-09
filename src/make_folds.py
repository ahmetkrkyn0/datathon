"""
Faz 2 — data/folds.parquet uretimi (TUM modellerin tek fold kaynagi).

    python src/make_folds.py

Uretir: data/folds.parquet (student_id, repeat[0-2], fold[0-4]); 30.000 satir.
Dogrular (SPEC §8 DoD-1): her (repeat,fold)'da mean(y>=100)/mean(y<=50) global oranlardan
+/-%1 icinde; her satir her repeat'te TAM 1 kez validation. Deterministik (seeds 42/2026/7).
"""

from __future__ import annotations

import cv  # src/ icinden calistir: python src/make_folds.py


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    y = train[cv.TARGET_COL].values
    sid = train[cv.ID_COL].values

    folds = cv.build_and_save_folds(train)
    stats = cv.validate_folds(folds, y, sid)

    print(f"[folds] yazildi -> {cv.FOLDS_PATH}")
    print(f"[folds] satir: {len(folds)}  (= {len(train)} x {cv.N_REPEATS} repeat)")
    print(f"[folds] repeat'ler: {sorted(folds['repeat'].unique())}  fold'lar: {sorted(folds['fold'].unique())}")
    print(f"[folds] global pct(y>=100)={stats['global_pct_100']*100:.2f}%  pct(y<=50)={stats['global_pct_50']*100:.2f}%")

    # Her (repeat,fold) hucresi: boyut ve stratify oranlari (denetim izi).
    print("\n(repeat,fold)   n   pct>=100   pct<=50")
    for (r, f), c in sorted(stats["per_cell"].items()):
        print(f"   ({r},{f})    {c['n']:5d}    {c['pct_100']*100:5.2f}%    {c['pct_50']*100:5.2f}%")

    print("\n[folds] DoD-1 assert'leri GECTI (her satir her repeat'te tam 1 kez; oranlar +/-%1).")


if __name__ == "__main__":
    main()
