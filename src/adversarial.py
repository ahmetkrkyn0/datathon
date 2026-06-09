"""
Faz 2 — Adversarial validation (SPEC §4 Adim 8 / §8 DoD-6) — KAYMA DEDEKTORU.

    python src/adversarial.py

Train(0)/Test(1) etiketiyle TEK siniflandirici (LGBM), AYNI kanonik 5-fold protokol.

ROL DUZELTMESI (review C1/H1): Adversarial AUC bir KOVARYAT-KAYMA dedektorudur, ZARAR
dedektoru DEGIL. "AUC yuksek -> feature'i at" mantigi HATALIDIR: yillar test'te mevcut,
tum test yil degerleri train'de var (ekstrapolasyon yok) ve public/private AYNI test
setinin rastgele bolmeleri -> tek feature public/private ayrismasi yaratamaz. Olculen:
yillari eklemek CV-MSE'yi 87.91->81.69 (-6.2), recency-proxy'yi 101.1->92.8 (-8.3)
IYILESTIRIR -> YILLAR TUTULUR. Kayma feature atarak degil, validation tarafinda yonetilir
(cv.recency_weights + cv.compute_recency_weighted_mse co-headline; simetrik gap_status).

Olcumler: numeric-only yil-disi (~0.49), yapisal yil-disi (~0.535, target_role ikincil
kaymasi) ve yapisal +yillar (anchor matrisi, ~0.66 BEKLENEN). review L3: tek siniflandirici,
kategorikler dahil; "ana kayma yillar; kucuk ikincil kategorik (target_role) kayma var".
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
# Yil-disi uzay icin kayma-monitoru esigi: yil-disi AUC > 0.60 ise yillar DISINDA yeni/
# beklenmedik bir kayma var demektir -> incele. (Yilli matrisin ~0.66 cikmasi BEKLENEN
# davranistir; feature reddi icin kriter DEGILDIR.)
DRIFT_MONITOR_THRESHOLD = 0.60

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

    # 1) numeric-only, yil-disi -> ~0.49 (kayma-monitoru: yil-disi sayisal uzay temiz mi?)
    num = cv.numeric_feature_columns(train)
    X_num = _stack(train[num].reset_index(drop=True), test[num].reset_index(drop=True))
    auc_num = adv_auc(X_num, cat_features=None)

    # 2) yapisal num+kategorik+flag, yil-disi -> ~0.535 (target_role ikincil kaymasi)
    cat_dtypes = cv.structured_cat_dtypes(train)
    Xs_tr, cat_features = cv.build_structured_matrix(train, cat_dtypes, with_years=False)
    Xs_te, _ = cv.build_structured_matrix(test, cat_dtypes, with_years=False)
    auc_struct = adv_auc(_stack(Xs_tr, Xs_te), cat_features=cat_features)

    # 3) yapisal +YILLAR (ANCHOR matrisi; with_years=True varsayilan) -> ~0.66 BEKLENEN
    Xy_tr, _ = cv.build_structured_matrix(train, cat_dtypes)
    Xy_te, _ = cv.build_structured_matrix(test, cat_dtypes)
    auc_years = adv_auc(_stack(Xy_tr, Xy_te), cat_features=cat_features)

    monitor_ok = max(auc_num, auc_struct) < DRIFT_MONITOR_THRESHOLD

    lines = [
        "Faz 2 — Adversarial Validation (kayma dedektoru; SPEC §4 Adim 8 / DoD-6)",
        "=" * 78,
        f"Siniflandirici : LGBMClassifier(n_estimators=200, max_depth=4, num_leaves=15, lr=0.05, "
        f"deterministic, seed={cv.SEED})",
        f"Protokol       : Train(0)/Test(1) etiketi, StratifiedKFold(5, shuffle, rs={cv.SEED}) OOF proba, ROC-AUC",
        f"Yil kolonlari  : MATRISTE VAR ({', '.join(cv.YEAR_COLS)} ham sayisal; review C1 duzeltmesi)",
        "",
        "AUC olcumleri:",
        f"  numeric-only, yil-disi (37 feature)        : {auc_num:.4f}   "
        f"(EDA ref {EDA_REF['numeric_only']:.4f})",
        f"  yapisal yil-disi num+kat+flag (49 feature) : {auc_struct:.4f}   "
        f"(EDA ref {EDA_REF['structured_num_cat']:.4f})",
        f"  yapisal +YILLAR = ANCHOR matrisi (51)      : {auc_years:.4f}   "
        f"(EDA ref {EDA_REF['with_years']:.4f}; BEKLENEN ~0.66)",
        "",
        "KARAR          : YILLAR TUTULDU. Adversarial AUC bir KOVARYAT-KAYMA dedektorudur,",
        "                 zarar dedektoru DEGIL; 'AUC<0.55 guvenli -> yuksekse at' mantigi",
        "                 HATALIYDI ve duzeltildi. Olcum: yillari eklemek CV-MSE'yi",
        "                 87.91->81.69 (-6.2), recency-agirlikli private-proxy'yi",
        "                 101.1->92.8 (-8.3) iyilestiriyor. Tum test yil degerleri train'de",
        "                 mevcut; public/private ayni test setinin rastgele bolmeleri ->",
        "                 tek feature public/private ayrismasi yaratamaz.",
        "",
        "Kaymanin yonetimi (feature atmak DEGIL, validation kalibrasyonu):",
        "  * Co-headline metrik: recency-agirlikli OOF-MSE (w=P_test(gy)/P_train(gy)) =",
        "    private-durust tahmin (cv.compute_recency_weighted_mse).",
        "  * Gap politikasi SIMETRIK: |gap|>3*std her iki yonde KIRMIZI/DUR (cv.gap_status).",
        "",
        f"Kayma-monitoru : yil-disi uzay AUC < {DRIFT_MONITOR_THRESHOLD} -> "
        f"{'TEMIZ' if monitor_ok else 'INCELE'} (yillar disinda beklenmedik yeni kayma yok).",
        "",
        "Not: Ana kayma YILLARDADIR; kucuk ikincil kategorik kayma target_role'dadir",
        "     (yil drift'inin kategorik yansimasi; AI Engineer %8.27->%11.71). target_role",
        "     degerli sinyaldir, TUTULUR. Yil-disi AUC>0.6 olursa yillar disinda yeni kayma",
        "     var demektir -> incele (feature reddi yine otomatik degil; olc + kapidan gecir).",
    ]
    ADVERSARIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADVERSARIAL_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[adversarial] yazildi -> {ADVERSARIAL_PATH}")

    assert monitor_ok, (
        f"Yil-disi uzay AUC {max(auc_num, auc_struct):.4f} >= {DRIFT_MONITOR_THRESHOLD} -> "
        "yillar disinda beklenmedik kayma; incele."
    )


if __name__ == "__main__":
    main()
