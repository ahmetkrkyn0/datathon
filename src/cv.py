"""
Faz 2 — Dogrulama Stratejisi (CV) — ANTI-OVERFIT OMURGASI / UST OTORITE
=======================================================================

Bu modul, tum sonraki fazlarin (03-07) UST OTORITESIDIR. Bir model / feature /
hiperparametre ancak buradaki protokole gore CV-MSE'yi iyilestiriyorsa kabul edilir.
Roadmap/02-validation-strategy/SPEC.md'nin uygulanabilir karsiligidir.

KILITLI PROTOKOL (SPEC §4-§6):
  * Repeated Stratified 5-fold x 3-repeat (seeds 42/2026/7) = 15 fit.
  * Stratify: y>=100 AYRI bin (~%7.73) + kalan icin qcut(q=9, duplicates="drop") -> ~10 sinif.
  * Master fold dosyasi: data/folds.parquet (student_id, repeat, fold) -> TUM modeller okur.
  * OOF sozlesmesi: oof_M[i] = i'nin gorulmedigi fold'dan tahmin, 3 repeat ortalamasi.
  * TEST URETIMI = KANONIK FOLD-BAGGING: test_M = clip(mean(15 fold modelinin test tahmini)).
    oof_M ile test_M AYNI 15 modelden gelir -> CV-MSE submission'in sapmasiz olcusu.
  * clip[0,100] TEK fonksiyondan (clip_predictions) hem OOF hem test'e; MSE clip SONRASI.
  * Kabul kapisi (sonraki fazlar): yeni_cv < eski_cv - 0.25*cv_std. Esitlikte basit model (Occam).
  * CO-HEADLINE METRIK: compute_cv_mse (fold-denge/siralama) YANINDA compute_recency_weighted_mse
    (train satirlari test graduation_year dagilimiyla importance-weighted) raporlanir; test
    recency-yogun oldugundan recency-weighted deger private-DURUST tahmindir (review H1).

PAZARLIKSIZ SIZINTI KURALI (SPEC §6):
  * FOLD-ICI FIT MUTLAK: impute/encoding/scaler/TF-IDF/Ridge — HER fit yalniz dis-fold
    train'inden (transformerlar fit_fold ICINDE Pipeline/ColumnTransformer olarak). Hicbir
    istatistik tum-train'de hesaplanmaz. run_oof bu sozlesmeye gore tasarlanmistir: fit_fold
    SADECE (X_tr, y_tr, X_val, y_val) alir; tum-train'e veya test'e erisemez.
  * student_id feature DEGIL. Yil kolonlari HAM SAYISAL feature olarak DAHILDIR (review C1
    duzeltmesi; olculdu: +yillar CV 87.91->81.69, recency-proxy 101.1->92.8).

Determinizm: SEED=42 her yerde. cv.py model-agnostiktir (sadece numpy/pandas/sklearn);
modele ozgu determinizm bayraklari (LightGBM deterministic=True vb.) fit_fold'u kuran
script'in sorumlulugundadir.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- #
# Sabitler & yol sozlesmesi
# --------------------------------------------------------------------------- #
SEED = 42
SEEDS = (42, 2026, 7)  # her repeat farkli shuffle (SPEC §4 Adim 2)
N_SPLITS = 5
N_REPEATS = 3

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
FOLDS_PATH = DATA_DIR / "folds.parquet"

# Kolon rolleri (Faz 01 column_profile.csv ile birebir; KILITLI karar).
ID_COL = "student_id"
TARGET_COL = "career_success_score"
TEXT_COL = "mentor_feedback_text"
CATEGORICAL_COLS = (
    "department",
    "university_tier",
    "target_role",
    "hobby",
    "preferred_social_media_platform",
)
# Yil kolonlari: HAM SAYISAL feature olarak TUTULUR (review C1). Adversarial AUC (yilli 0.666)
# bir KOVARYAT-KAYMA dedektorudur, ZARAR dedektoru DEGIL: tum test yil degerleri train'de mevcut
# (ekstrapolasyon yok) ve public/private AYNI test setinin rastgele bolmeleri -> tek feature
# public/private ayrismasi yaratamaz. Olculen: +yillar CV-MSE 87.913->81.689 (-6.2),
# recency-agirlikli OOF 101.09->92.75 (-8.3), temporal holdout 141.7->121.6. Kayma feature
# atarak degil, VALIDATION tarafinda yonetilir (recency_weights + compute_recency_weighted_mse).
YEAR_COLS = ("application_year", "graduation_year")

# 7 NA'li sayisal kolon (Faz 01 missing_map.csv). _missing bayraklari fold-bagimsiz
# (sadece isna(), hedefe bakmaz -> Guardrail 4). Anchor 'temel FE' bunlari icerir.
NA_COLS = (
    "internship_duration_months",
    "english_exam_score",
    "github_avg_stars",
    "open_source_contribution_count",
    "hr_interview_score",
    "linkedin_profile_score",
    "portfolio_score",
)

# clip siniri (SPEC: hedef kesin [0,100]).
CLIP_LO, CLIP_HI = 0.0, 100.0


# --------------------------------------------------------------------------- #
# Determinizm
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    """Global determinizm: PYTHONHASHSEED + thread sayisi + random + numpy. Script basinda cagrilir.

    NOT: PYTHONHASHSEED ve thread env'leri yorumlayicinin/BLAS'in BASLANGICINDA okunur; calisan
    surec icinde set etmek mevcut surecin hash-rastgeleligini/OpenMP havuzunu degistirmez. Bu yine
    de kasitli/belge amacli set edilir; GERCEK reproducibility kapisi repro_test.py'nin TAZE
    subprocess env'inde bu degiskenleri set etmesidir (DoD-9). LightGBM zaten n_jobs=1; ileride
    HistGBR/CatBoost (Faz 06) icin tek-thread BLAS/OpenMP burada belgelenir.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Clip — TEK kaynak (Guardrail 9): hem OOF hem test ayni fonksiyondan gecer.
# --------------------------------------------------------------------------- #
def clip_predictions(pred) -> np.ndarray:
    """Tum tahminleri [0,100]'e sinirlar. MSE DAIMA bu fonksiyondan SONRA hesaplanir."""
    return np.clip(np.asarray(pred, dtype=float), CLIP_LO, CLIP_HI)


def assert_in_range(pred, name: str = "pred") -> None:
    """Submission yazici (Faz 07) icin: clip-disi deger gorulurse hata firlat."""
    arr = np.asarray(pred, dtype=float)
    if not np.isfinite(arr).all():
        raise AssertionError(f"{name}: NaN/Inf deger var.")
    if arr.min() < CLIP_LO - 1e-9 or arr.max() > CLIP_HI + 1e-9:
        raise AssertionError(
            f"{name}: clip-disi deger ({arr.min():.4f}..{arr.max():.4f}); clip_predictions unutulmus."
        )


# --------------------------------------------------------------------------- #
# IO yardimcilari (UTF-8 BOM'lu dosyalar -> utf-8-sig)
# --------------------------------------------------------------------------- #
def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test_x.csv", encoding="utf-8-sig")


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """YIL-DISI sayisal feature listesi: id/target/text/kategorik/yil HARIC tum kolonlar (37).

    Yillar ayri tutulur (YEAR_COLS) ve build_structured_matrix'te with_years=True ile HAM
    SAYISAL eklenir; boylece adversarial/teshis kosullari yil-disi uzayi ayrica olcebilir."""
    drop = {ID_COL, TARGET_COL, TEXT_COL, *CATEGORICAL_COLS, *YEAR_COLS}
    return [c for c in df.columns if c not in drop]


def structured_cat_dtypes(train: pd.DataFrame) -> dict:
    """Kategorik kolonlar icin SABIT kategori evreni (train seviyeleri; test-only seviye YOK,

    Faz 01). Train ve test AYNI CategoricalDtype ile kodlanir -> native-kategorik kodlari
    hizali. Bu hedef-bagimsizdir (one-hot ile ESDEGER sizinti profili) -> fold-safe."""
    from pandas.api.types import CategoricalDtype

    return {
        c: CategoricalDtype(categories=sorted(train[c].astype(str).unique()))
        for c in CATEGORICAL_COLS
    }


def build_structured_matrix(
    df: pd.DataFrame,
    cat_dtypes: dict,
    with_flags: bool = True,
    with_years: bool = True,
):
    """Anchor 'yapisal' feature matrisi (lgbm_num): sayisal + YIL (ham sayisal) +
    native-kategorik + missing-flag. METIN yok. Native kategorik -> hedef-bagimsiz, fold-safe.

    with_years=True VARSAYILAN (review C1): application_year + graduation_year ham sayisal.
    with_years=False yalniz teshis (or. adversarial yil-disi olcum) icindir.

    Doner: (X, categorical_feature_names).
    NaN'lar sayisal kolonlarda KORUNUR (LightGBM native isler -> impute yok -> sizinti yok).
    """
    parts = [df[numeric_feature_columns(df)].reset_index(drop=True)]
    if with_years:
        parts.append(df[list(YEAR_COLS)].astype(float).reset_index(drop=True))
    if with_flags:
        fl = df[list(NA_COLS)].isna().astype("int8")
        fl.columns = [f"{c}_missing" for c in NA_COLS]
        parts.append(fl.reset_index(drop=True))
    cat_cols = {c: df[c].astype(str).astype(cat_dtypes[c]) for c in CATEGORICAL_COLS}
    parts.append(pd.DataFrame(cat_cols).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    return X, list(CATEGORICAL_COLS)


# --------------------------------------------------------------------------- #
# Adim 1 — Stratify binleri (SPEC §4 Adim 1)
# --------------------------------------------------------------------------- #
def make_strat_bins(y) -> np.ndarray:
    """y>=100 -> AYRI bin (9); kalan -> qcut(q=9). Toplam ~10 ayrik sinif."""
    y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
    bins = pd.Series(np.empty(len(y), dtype="int64"))
    is_100 = y >= 100.0
    bins[is_100] = 9  # sansurlu ust kutle AYRI bin
    rest = y[~is_100]
    q = pd.qcut(rest, q=9, labels=False, duplicates="drop")  # ~9 esit-frekans bin
    bins[~is_100] = q.values
    out = bins.values
    # Pozisyonel hizalama + kapsam guvencesi (bos/NaN bin yok; ==100 ayri bin).
    assert int(is_100.sum()) == int(np.sum(out == 9)), "==100 kutlesi bin 9'a tam dusmedi."
    assert not pd.isna(out).any(), "make_strat_bins: atanmamis (NaN) bin var."
    return out


# --------------------------------------------------------------------------- #
# Adim 2 — Repeated Stratified fold atamasi (SPEC §4 Adim 2)
#   RepeatedStratifiedKFold YERINE manuel dongu: her repeat'in seed'i acik kontrol.
# --------------------------------------------------------------------------- #
def get_folds(
    y,
    student_id,
    seeds=SEEDS,
    n_splits: int = N_SPLITS,
) -> pd.DataFrame:
    """data/folds.parquet icerigini uretir: (student_id, repeat, fold), n*len(seeds) satir."""
    y = np.asarray(y, dtype=float)
    student_id = np.asarray(student_id)
    n = len(y)
    bins = make_strat_bins(y)
    Xdummy = np.zeros((n, 1))
    rows = []
    for r, s in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        fold_of = np.full(n, -1, dtype="int64")
        for f, (_, val_idx) in enumerate(skf.split(Xdummy, bins)):
            fold_of[val_idx] = f
        assert (fold_of >= 0).all(), "Bazi satirlar hicbir fold'a atanmadi."
        for i in range(n):
            rows.append((student_id[i], r, int(fold_of[i])))
    return pd.DataFrame(rows, columns=["student_id", "repeat", "fold"])


def validate_folds(
    folds: pd.DataFrame,
    y,
    student_id,
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
    tol: float = 0.01,
) -> dict:
    """SPEC §4 Adim 2 / §8 DoD-1 dogrulama assert'leri.

    - Her satir her repeat'te TAM 1 kez validation.
    - Her (repeat, fold)'da mean(y>=100) ve mean(y<=50) global orandan +/-%1 icinde.
    """
    y = np.asarray(y, dtype=float)
    student_id = np.asarray(student_id)
    n = len(y)
    pos = {sid: i for i, sid in enumerate(student_id)}

    g100 = float(np.mean(y >= 100.0))
    g50 = float(np.mean(y <= 50.0))

    assert len(folds) == n * n_repeats, (
        f"folds satir sayisi {len(folds)} != {n * n_repeats} (n x repeat)."
    )
    assert set(folds["repeat"].unique()) == set(range(n_repeats))
    assert set(folds["fold"].unique()) == set(range(n_splits))

    per_cell = {}
    for r in range(n_repeats):
        fr = folds[folds["repeat"] == r]
        # her satir tam 1 kez (set esitligi + uzunluk -> tekrar yok, eksik yok)
        assert len(fr) == n, f"repeat {r}: {len(fr)} satir != {n}."
        assert set(fr["student_id"]) == set(student_id), (
            f"repeat {r}: student_id kumesi train ile birebir degil."
        )
        for f in range(n_splits):
            sub = fr[fr["fold"] == f]
            idx = np.array([pos[s] for s in sub["student_id"].values])
            ys = y[idx]
            p100 = float(np.mean(ys >= 100.0))
            p50 = float(np.mean(ys <= 50.0))
            assert abs(p100 - g100) <= tol, (
                f"(r={r},f={f}) mean(y>=100)={p100:.4f} global {g100:.4f} +/-{tol} disinda."
            )
            assert abs(p50 - g50) <= tol, (
                f"(r={r},f={f}) mean(y<=50)={p50:.4f} global {g50:.4f} +/-{tol} disinda."
            )
            per_cell[(r, f)] = dict(n=len(sub), pct_100=p100, pct_50=p50)

    return dict(global_pct_100=g100, global_pct_50=g50, per_cell=per_cell)


def build_and_save_folds(train: pd.DataFrame, path: Path = FOLDS_PATH) -> pd.DataFrame:
    """get_folds + validate_folds + parquet yazimi (deterministik)."""
    y = train[TARGET_COL].values
    sid = train[ID_COL].values
    folds = get_folds(y, sid)
    validate_folds(folds, y, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(path, index=False)
    return folds


def load_folds(path: Path = FOLDS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def fold_of_rows(folds: pd.DataFrame, student_id, repeat: int) -> np.ndarray:
    """Bir repeat icin satir-hizali (n,) fold vektoru (student_id sirasinda)."""
    student_id = np.asarray(student_id)
    pos = {sid: i for i, sid in enumerate(student_id)}
    fr = folds[folds["repeat"] == repeat]
    out = np.full(len(student_id), -1, dtype="int64")
    for sid, f in zip(fr["student_id"].values, fr["fold"].values):
        out[pos[sid]] = f
    assert (out >= 0).all(), f"repeat {repeat}: bazi satirlarin fold'u yok."
    return out


# --------------------------------------------------------------------------- #
# Adim 3-4 — OOF kosucu + KANONIK FOLD-BAGGING test uretimi (SPEC §4)
# --------------------------------------------------------------------------- #
def run_oof(
    fit_fold,
    X,
    y,
    X_test,
    folds: pd.DataFrame,
    student_id,
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
):
    """15 fit (5 fold x 3 repeat). Sizintisiz OOF + kanonik fold-bagging test.

    Parametreler
    ------------
    fit_fold : callable(X_tr, y_tr, X_val, y_val) -> (predict_fn, best_iteration)
        SADECE dis-fold train + valid alir (tum-train/test'e erisemez -> sizinti yapisal
        olarak imkansiz). predict_fn(X) HAM (clip'siz) tahmin dondurur; clip burada,
        ortalama SONRASI, tek noktadan uygulanir. best_iteration int veya None.
    X, y : train feature matrisi (n,p) ve hedef (n,), student_id sirasina hizali.
    X_test : test feature matrisi (n_test,p), AYNI kolonlar.
    folds : data/folds.parquet (student_id, repeat, fold).
    student_id : X/y satir sirasindaki student_id dizisi (folds eslemesi icin).

    Doner
    -----
    dict(oof, test, best_iterations, n_models, fold_order, genuine_fold_mse)
        oof  : (n,) clip'li, 3-repeat ortalamasi OOF.
        test : (n_test,) clip'li, 15 fold modelinin ortalamasi (fold-bagging).
        genuine_fold_mse : 15 fold modelinin tek-basina val MSE'si (clip'li) — tani/denetim.
            cv_mse_mean/std DAIMA compute_cv_mse(oof) ile (DoD-4); bu liste ek bilgidir.
    """
    is_df = isinstance(X, pd.DataFrame)
    X = X.reset_index(drop=True) if is_df else np.asarray(X)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_test = len(X_test)
    student_id = np.asarray(student_id)

    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=float)
    test_sum = np.zeros(n_test, dtype=float)
    best_iters: list[int | None] = []
    fold_order: list[tuple[int, int]] = []
    genuine_fold_mse: list[float] = []  # her fold modelinin KENDI val MSE'si (clip'li); std §2 referansi

    for r in range(n_repeats):
        fold_of = fold_of_rows(folds, student_id, r)
        for f in range(n_splits):
            val_idx = np.where(fold_of == f)[0]
            tr_idx = np.where(fold_of != f)[0]

            if is_df:
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[tr_idx], X[val_idx]

            predict_fn, best_it = fit_fold(X_tr, y[tr_idx], X_val, y[val_idx])

            # HAM tahminler (clip yok); clip ortalama sonrasi tek noktadan.
            val_pred = np.asarray(predict_fn(X_val), dtype=float)
            oof_sum[val_idx] += val_pred
            oof_cnt[val_idx] += 1.0
            test_sum += np.asarray(predict_fn(X_test), dtype=float)

            # Tani: bu fold modelinin tek-basina val MSE'si (clip SONRASI).
            genuine_fold_mse.append(float(mean_squared_error(y[val_idx], clip_predictions(val_pred))))
            best_iters.append(None if best_it is None else int(best_it))
            fold_order.append((r, f))

    # Her satir her repeat'te tam 1 kez validation -> tam n_repeats tahmin.
    assert np.all(oof_cnt == n_repeats), "OOF kapsami bozuk: bir satir != n_repeats kez gorulmus."
    assert test_sum.shape == (n_test,), f"test_sum sekli {test_sum.shape} != ({n_test},) — predict_fn yanlis boyut dondurdu."
    n_models = len(fold_order)
    assert n_models == n_repeats * n_splits

    oof = clip_predictions(oof_sum / oof_cnt)          # ortalama -> TEK clip
    test = clip_predictions(test_sum / float(n_models))  # fold-bagging -> AYNI clip
    return dict(
        oof=oof,
        test=test,
        best_iterations=best_iters,
        n_models=n_models,
        fold_order=fold_order,
        genuine_fold_mse=genuine_fold_mse,
    )


# --------------------------------------------------------------------------- #
# Adim 6 — CV-MSE raporlama (SPEC §6). 15 fold-MSE -> mean/std (clip SONRASI).
# --------------------------------------------------------------------------- #
def compute_cv_mse(oof, y, folds: pd.DataFrame, student_id):
    """15 (repeat,fold) hucresinde MSE -> (mean, std, per_fold[15]).

    DoD-4 garantisi: oof_{M}.npy'den yeniden cagrilinca AYNI mean uretir (clip idempotent).
    np.std populasyon std'sidir (ddof=0), SPEC §6 ile birebir.
    """
    oof = clip_predictions(oof)  # idempotent; clip SONRASI MSE garantisi
    y = np.asarray(y, dtype=float)
    student_id = np.asarray(student_id)
    pos = {sid: i for i, sid in enumerate(student_id)}

    # sorted() -> deterministik hucre sirasi (PYTHONHASHSEED-bagimsiz).
    repeats = sorted(folds["repeat"].unique())
    foldids = sorted(folds["fold"].unique())
    per_fold: list[float] = []
    covered = 0
    for r in repeats:
        for f in foldids:
            sub = folds[(folds["repeat"] == r) & (folds["fold"] == f)]
            idx = np.array([pos[s] for s in sub["student_id"].values])
            per_fold.append(float(mean_squared_error(y[idx], oof[idx])))
            covered += len(idx)
    # Eksik hucre / kayip satir erken yakala (malformed folds'a karsi).
    assert len(per_fold) == len(repeats) * len(foldids), "compute_cv_mse: eksik (repeat,fold) hucresi."
    assert covered == len(folds), "compute_cv_mse: hucrelerin satir toplami folds ile uyusmuyor."
    return float(np.mean(per_fold)), float(np.std(per_fold)), per_fold


# --------------------------------------------------------------------------- #
# Recency-agirlikli OOF-MSE — private-DURUST co-headline (review H1/C1).
# Test graduation_year dagilimi recency-yogun (2024-26); unweighted random CV bu yuzden
# private'i ~10+ MSE iyimser tahmin eder. Importance weighting (w = P_test/P_train) bunu
# duzeltir. Standart compute_cv_mse fold-denge/SIRALAMA icin kalir; recency-weighted deger
# mutlak private beklentisi icin CO-HEADLINE raporlanir.
# --------------------------------------------------------------------------- #
RECENCY_COL = "graduation_year"


def recency_weights(train_df: pd.DataFrame, test_df: pd.DataFrame, col: str = RECENCY_COL) -> np.ndarray:
    """Train satir agirliklari: w_i = P_test(col=v_i) / P_train(col=v_i), mean-normalize."""
    tr = train_df[col].value_counts(normalize=True)
    te = test_df[col].value_counts(normalize=True)
    w = (train_df[col].map(te).fillna(0.0) / train_df[col].map(tr)).to_numpy(dtype=float)
    assert np.isfinite(w).all() and w.sum() > 0, "recency_weights: bozuk agirlik (NaN/Inf/0)."
    return w / w.mean()


def compute_recency_weighted_mse(oof, y, w) -> float:
    """Recency-agirlikli OOF-MSE (clip SONRASI). compute_cv_mse'nin YANINDA co-headline."""
    oof = clip_predictions(oof)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * (y - oof) ** 2) / np.sum(w))


# --------------------------------------------------------------------------- #
# Kabul kapisi & gap esikleri — UST OTORITE karari (SPEC §8 DoD-8, MASTERPLAN).
# Sonraki tum fazlar (04-07) bu fonksiyonlari kullanir.
# --------------------------------------------------------------------------- #
ACCEPT_K = 0.25  # gurultu bandi carpani


def acceptance_gate(new_mean: float, old_mean: float, old_std: float, k: float = ACCEPT_K) -> bool:
    """Kabul kapisi: yeni model ancak `new_mean < old_mean - k*old_std` ise KABUL (overfit kapisi).

    Marjinal / gurultu-icindeki iyilesmeler REDDEDILIR. Esitlikte daha basit model kazanir
    (Occam) -> esitlik durumunda cagiran taraf basit modeli secer."""
    return new_mean < old_mean - k * old_std


def gap_status(public_lb_mse: float, cv_mse_mean: float, cv_mse_std: float) -> str:
    """CV-LB gap saglik sensoru (MASTERPLAN). Public LB SADECE sensor; karar DAIMA CV.

    SIMETRIK (review H1 duzeltmesi): yesil |gap| <= 1.5*std ; sari 1.5-3*std ;
    kirmizi |gap| > 3*std — HER IKI YONDE DUR:
      public << CV : sizinti suphesi (CV sahte iyimser) -> pipeline'i incele.
      public >> CV : dagilim kaymasi / CV iyimser -> recency-agirlikli metrige gec (re-kalibre);
                     bunu "sizinti yok" diye GECISTIRME.
    """
    gap = public_lb_mse - cv_mse_mean
    a = abs(gap)
    if a <= 1.5 * cv_mse_std:
        return "yesil"
    if a <= 3.0 * cv_mse_std:
        return "sari"
    return "kirmizi"
