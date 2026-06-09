"""
Faz 2 — Adversarial validation (SPEC §4 Adim 8 / §8 DoD-6).

    python src/adversarial.py

Train(0)/Test(1) etiketiyle TEK siniflandirici (LGBM), AYNI kanonik 5-fold. YIL-DISI nihai
feature matrisinde AUC olculur:
  * numeric-only (37)            -> beklenen ~0.49 (Faz 01 EDA 0.4942; SPEC referansi)
  * yapisal num+kategorik+flag   -> beklenen ~0.535 (Faz 01 EDA 0.5347; anchor matrisi)

Karar: AUC < 0.55 -> GUVENLI bolge (train/test pratikte ayrilamaz; random stratified KFold
private MSE'nin sadik temsilcisi). AUC > 0.6 -> suclu feature incele (oncelikle yil turevleri),
03/04 fazlarina geri bildir. Yil kolonlari (application_year/graduation_year) matriste YOK.
review L3/H1 duzeltmesi: TEK siniflandirici + nihai-matris kontrolu kategorikler DAHIL.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import cv

ADVERSARIAL_PATH = cv.REPORTS_DIR / "adversarial.txt"

# Faz 01 EDA ile karsilastirma referanslari (HistGBM ile olculmustu).
EDA_REF = {"numeric_only": 0.4942, "structured_num_cat": 0.5347, "with_years": 0.6658}
SAFE_THRESHOLD = 0.55  # < safe ; > 0.6 incele (target_role ikincil kayma ~0.535 GUVENLI)

# Faz 01 HistGBM (max_iter=200,max_depth=4,lr=0.05) ile kiyaslanabilir, mutevazi LGBM.
CLF_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    random_state=cv.SEED,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)


def adv_auc(X, cat_features, n_splits: int = cv.N_SPLITS) -> float:
    """Train(0)/Test(1) ayriminda 5-fold OOF proba -> ROC-AUC. cat_features: native-kategorik."""
    n = len(X)
    half = n // 2
    label = np.zeros(n, dtype="int64")
    label[half:] = 1  # ilk yari train(0), ikinci yari test(1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv.SEED)
    oof = np.zeros(n, dtype=float)
    for tr, va in skf.split(np.zeros(n), label):
        clf = LGBMClassifier(**CLF_PARAMS)
        clf.fit(
            X.iloc[tr], label[tr],
            categorical_feature=(cat_features or "auto"),
            callbacks=[lgb.log_evaluation(0)],
        )
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
    return float(roc_auc_score(label, oof))


def _stack(train_part, test_part):
    import pandas as pd
    return pd.concat([train_part, test_part], axis=0, ignore_index=True)


def main() -> None:
    cv.set_seed()
    train = cv.load_train()
    test = cv.load_test()

    # 1) numeric-only (yil-disi) -> SPEC ~0.49
    num = cv.numeric_feature_columns(train)
    X_num = _stack(train[num].reset_index(drop=True), test[num].reset_index(drop=True))
    auc_num = adv_auc(X_num, cat_features=None)

    # 2) yapisal num+kategorik+flag (anchor matrisi, yil-disi) -> ~0.535
    cat_dtypes = cv.structured_cat_dtypes(train)
    Xs_tr, cat_features = cv.build_structured_matrix(train, cat_dtypes)
    Xs_te, _ = cv.build_structured_matrix(test, cat_dtypes)
    X_struct = _stack(Xs_tr, Xs_te)
    auc_struct = adv_auc(X_struct, cat_features=cat_features)

    verdict = (
        "GUVENLI (random stratified KFold private'i sadik temsil eder)"
        if max(auc_num, auc_struct) < SAFE_THRESHOLD
        else ("SARI: 0.55-0.60 incele" if max(auc_num, auc_struct) < 0.60 else "KIRMIZI: >0.60 -> suclu feature incele")
    )

    lines = [
        "Faz 2 — Adversarial Validation (yil-disi nihai matris; SPEC §4 Adim 8 / DoD-6)",
        "=" * 78,
        f"Siniflandirici : LGBMClassifier(n_estimators=200, max_depth=4, num_leaves=15, lr=0.05, "
        f"deterministic, seed={cv.SEED})",
        f"Protokol       : Train(0)/Test(1) etiketi, StratifiedKFold(5, shuffle, rs={cv.SEED}) OOF proba, ROC-AUC",
        f"Yil kolonlari  : MATRISTE YOK ({', '.join(cv.YEAR_COLS)} ham feature degil)",
        "",
        "AUC olcumleri (yil-disi):",
        f"  numeric-only (37 feature)              : {auc_num:.4f}   "
        f"(EDA ref {EDA_REF['numeric_only']:.4f}; SPEC ~0.49)",
        f"  yapisal num+kategorik+flag (49 feature): {auc_struct:.4f}   "
        f"(EDA ref {EDA_REF['structured_num_cat']:.4f}; anchor matrisi)",
        f"  [referans] yillarla (EDA)              : {EDA_REF['with_years']:.4f}  (yillar drop edildi)",
        "",
        f"Esik           : AUC < {SAFE_THRESHOLD} GUVENLI ; 0.55-0.60 SARI ; > 0.60 KIRMIZI (incele)",
        f"KARAR          : {verdict}",
        "",
        "Not: num+kategorik AUC'nin numeric-only'dan yuksekligi target_role'un train/test ikincil",
        "     kaymasidir (yil drift'inin kategorik yansimasi; AI Engineer %8.27->%11.71). 0.535<0.55",
        "     -> GUVENLI; target_role degerli sinyaldir, ATILMAZ. Faz 04 nihai matriste ~0.53 bekle",
        "     (tam 0.50 degil). AUC>0.6 olursa suclu feature (oncelikle yil turevleri) incelenir.",
    ]
    ADVERSARIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADVERSARIAL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[adversarial] yazildi -> {ADVERSARIAL_PATH}")

    assert auc_struct < 0.60, f"Yapisal matris AUC {auc_struct:.4f} >= 0.60 -> suclu feature incele (DUR)."


if __name__ == "__main__":
    main()
