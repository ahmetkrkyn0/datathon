"""
Faz 1 — EDA & Veri Anlama  (Datathon 2026, career_success_score regresyonu)
============================================================================

Bu script, Roadmap/01-eda-data-understanding/SPEC.md'deki 11 adimi ve
"Definition of Done" listesini eksiksiz uygular.

GUARDRAIL'LAR (pazarliksiz):
  * Bu faz bir 0-OVERFIT SIGORTASIDIR. HICBIR tahmin modeli fit edilmez
    (TEK istisna: adversarial validation siniflandiricisi). HICBIR feature
    uretilip diske kaydedilmez. HICBIR istatistik hedefe bakarak secilmez.
    EDA betimler, KARAR VERMEZ.
  * FOLD-SAFE GELECEK: impute / encoding / TF-IDF / scaler bu fazda
    HESAPLANMAZ (Faz 2-5'te fold-ici fit edilir). EDA yalniz betimleyici
    istatistik uretir.
  * mentor_feedback_text TEMIZ UTF-8'dir. MOJIBAKE FIX YAPILMAZ
    (ftfy/latin1 veriyi bozar). Byte teyidi: 'o-umlaut'.encode('utf-8')==b'\\xc3\\xb6'.
  * student_id ASLA feature degil. Yil kolonlari (application_year /
    graduation_year) ham feature olarak KULLANILMAZ; sadece "supheli"
    isaretlenip hedef-by-yil drift'i olculur.
  * Tum sayilar HESAPLANIR. Asagidaki SPEC_REF sozlugu YALNIZCA dogrulama
    ciktisi (PASS/WARN) icindir; hicbir hesaba girdi olarak verilmez.

Calistirma:
    python notebooks/01_eda.py
SEED=42, internet kapali, bastan sona deterministik. Iki kez calistirildiginda
ayni sayilari ve ayni artefaktlari uretir.

Ciktilar (reports/eda/):
    column_profile.csv, missing_map.csv,
    target_profile.json, adversarial_auc.json, text_profile.json,
    eda_report.html (7 bolum), figures/*.png
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import sys
from pathlib import Path

# --- Determinizm (her import'tan once seed) --------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np

np.random.seed(SEED)

import pandas as pd

import matplotlib

matplotlib.use("Agg")  # internet/ekran-bagimsiz, deterministik
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

# --- Yollar (script konumuna gore, herhangi bir cwd'den calisir) -----------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "reports" / "eda"
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# SPEC referans degerleri — YALNIZCA dogrulama ciktisi icin (hesaba girmez).
# CLAUDE.md / SPEC.md'de olculmus degerler; kendi hesabimizla eslestigini
# PASS/WARN olarak yazariz. Sapma cikarsa raporlar + MASTERPLAN'a not duseriz.
# ===========================================================================
SPEC_REF = {
    "target_mean": 76.94,
    "target_std": 15.19,
    "target_median": 77.81,
    "target_skew": -0.451,
    "target_var": 230.63,
    "pct_eq_100": 7.73,
    "pct_le_50": 4.97,
    "miss_internship_duration_pct": 16.57,
    "miss_english_exam_pct": 9.53,
    "miss_github_avg_stars_pct": 9.10,
    "miss_open_source_pct": 9.10,
    "miss_hr_interview_pct": 7.80,
    "miss_linkedin_pct": 6.68,
    "miss_portfolio_pct": 3.64,
    "internship_mnar_pct": 82.14,
    "corr_project_quality": 0.541,
    "corr_tech_mean": 0.338,
    "corr_pq_x_tech": 0.606,
    "corr_conv_rate": 0.011,
    "adv_auc_with_years": 0.6654,
    "adv_auc_without_years": 0.4995,
    "text_word_mean": 33.2,
    "text_word_min": 17,
    "text_word_max": 59,
    "text_char_mean": 273.5,
    "text_word_mean_test": 33.1,
    "kw_ancak": 5831,
    "kw_gelistir": 6302,
    "kw_potansiyel": 2041,
    "kw_guclu": 3097,
    "kw_basari": 2526,
    "kw_mukemmel": 468,
    "kw_olaganustu": 184,
    "kw_ustun": 74,
    "kw_gerekiyor": 325,
    "kw_eksik": 447,
}

_CHECKS: list[tuple] = []  # (etiket, hesaplanan, referans, tolerans, durum)


def check(label: str, value, ref, tol) -> None:
    """Hesaplanan degeri SPEC referansiyla kiyasla; PASS/WARN logla."""
    ok = abs(float(value) - float(ref)) <= tol
    status = "PASS" if ok else "WARN"
    _CHECKS.append((label, float(value), float(ref), float(tol), status))
    print(f"  [{status}] {label:<34} hesap={float(value):<12.4f} "
          f"spec={float(ref):<12.4f} tol=+-{tol}")


# --- Turkce-duyarli lowercase (I->i, I-dotless / I-dotted tuzagi) ----------
# str.lower() Turkce'de 'I'->'i' (yanlis) yapar. Once 'I-dotted'->'i',
# 'I'->'i-dotless' donusumu, sonra .lower(). Metin VE lexicon AYNI normalizasyon.
_UP_I_DOTTED = "İ"   # I with dot above
_LO_I_DOTLESS = "ı"  # dotless i


def tr_lower(s: str) -> str:
    return s.replace(_UP_I_DOTTED, "i").replace("I", _LO_I_DOTLESS).lower()


def fig_to_b64(fig) -> str:
    """Figuru base64 PNG'ye cevir (HTML icine gomulu, internet-bagimsiz)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def save_fig(fig, name: str) -> str:
    """Figuru hem diske PNG yaz hem base64 dondur (rapor + arsiv)."""
    path = FIGS / name
    fig.savefig(path, format="png", dpi=110, bbox_inches="tight")
    return fig_to_b64(fig)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# Rapor parcalari (HTML bolumleri burada birikir)
SECTIONS: dict[str, list[str]] = {
    "Hedef": [], "Eksik": [], "Kategorik": [], "Sayisal-Korelasyon": [],
    "Yil-Kayma": [], "Adversarial": [], "Metin": [],
}


def add_html(section: str, html: str) -> None:
    SECTIONS[section].append(html)


def add_img(section: str, b64: str, caption: str = "") -> None:
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    add_html(section, f'<img src="data:image/png;base64,{b64}"/>{cap}')


def df_html(df: pd.DataFrame, **kw) -> str:
    return df.to_html(classes="tbl", border=0, justify="center", **kw)


# ===========================================================================
def main() -> None:
    print(f"Faz 1 EDA — SEED={SEED} — pandas {pd.__version__}, numpy {np.__version__}")
    print(f"ROOT = {ROOT}")

    # ----------------------------------------------------------------------
    # ADIM 1 — Yukleme & semantik kontrol
    # ----------------------------------------------------------------------
    hr("ADIM 1 — Yukleme & semantik kontrol")
    # BOM tespit edildi -> utf-8-sig (BOM'u soyar; mojibake fix DEGIL).
    enc = "utf-8-sig"
    train = pd.read_csv(DATA / "train.csv", encoding=enc)
    test = pd.read_csv(DATA / "test_x.csv", encoding=enc)
    sample = pd.read_csv(DATA / "sample_submission.csv", encoding=enc)

    print(f"train.shape = {train.shape}  | test.shape = {test.shape}  "
          f"| sample.shape = {sample.shape}")
    assert train.shape == (10000, 47), f"train shape beklenmeyen: {train.shape}"
    assert test.shape == (10000, 46), f"test shape beklenmeyen: {test.shape}"

    only_train = set(train.columns) - set(test.columns)
    only_test = set(test.columns) - set(train.columns)
    print(f"train \\ test kolon farki = {only_train}")
    print(f"test \\ train kolon farki = {only_test}")
    assert only_train == {"career_success_score"}, "Tek fark hedef olmali!"
    assert only_test == set(), "test'te fazla kolon olmamali!"

    # student_id format & araliklar & kesisim
    assert train["student_id"].str.match(r"^STU_\d{6}$").all(), "train id formati bozuk"
    assert test["student_id"].str.match(r"^STU_\d{6}$").all(), "test id formati bozuk"
    tr_ids, te_ids = set(train["student_id"]), set(test["student_id"])
    inter = tr_ids & te_ids
    print(f"train id araligi: {train['student_id'].min()} .. {train['student_id'].max()}")
    print(f"test  id araligi: {test['student_id'].min()} .. {test['student_id'].max()}")
    print(f"train/test id kesisim sayisi = {len(inter)}  (beklenen 0)")
    assert len(inter) == 0, "student_id train/test kesisimi 0 olmali!"
    print("sample_submission ornek satirlar (123.94 yalniz FORMAT ornegi):")
    print(sample.to_string(index=False))
    assert list(sample.columns) == ["student_id", "career_success_score"]

    add_html("Hedef", f"<p><b>Yukleme:</b> train {train.shape}, test {test.shape}; "
             f"tek kolon farki <code>career_success_score</code>. "
             f"student_id kesisimi = {len(inter)} (sentetik, non-predictive anahtar; "
             f"ASLA feature degil).</p>")

    TARGET = "career_success_score"
    ID = "student_id"
    TEXT = "mentor_feedback_text"
    y = train[TARGET].astype(float)

    # ----------------------------------------------------------------------
    # ADIM 2 — Kolon rol tablosu
    # ----------------------------------------------------------------------
    hr("ADIM 2 — Kolon rol tablosu")
    CATEGORICAL = ["department", "university_tier", "target_role", "hobby",
                   "preferred_social_media_platform"]
    YEAR = ["application_year", "graduation_year"]
    # 9 teknik beceri skoru (kompozit on-olcumde kullanilir)
    TECH = ["coding_score", "problem_solving_score", "data_structures_score",
            "sql_score", "machine_learning_score", "backend_score",
            "frontend_score", "cloud_score", "devops_score"]

    def role_of(col: str) -> str:
        if col == ID:
            return "id"
        if col == TARGET:
            return "target"
        if col == TEXT:
            return "text"
        if col in CATEGORICAL:
            return "categorical"
        if col in YEAR:
            return "year"
        return "numeric"

    roles = {c: role_of(c) for c in train.columns}
    # Modelleme-aday sayisal kolonlar (yil HARIC)
    NUM = [c for c in train.columns if roles[c] == "numeric"]
    role_counts = pd.Series(roles).value_counts()
    print("Rol dagilimi:")
    print(role_counts.to_string())
    print(f"\nSayisal feature (yil haric) sayisi = {len(NUM)}")
    print(f"Teknik skor sayisi = {len(TECH)} (kompozit on-olcum icin)")
    assert len(TECH) == 9
    add_html("Kategorik", "<p><b>Kolon rolleri:</b> "
             + ", ".join(f"{k}={v}" for k, v in role_counts.items()) + "</p>")

    # ----------------------------------------------------------------------
    # ADIM 3 — Hedef dagilimi analizi
    # ----------------------------------------------------------------------
    hr("ADIM 3 — Hedef dagilimi analizi (career_success_score)")
    t_mean = float(y.mean()); t_std = float(y.std(ddof=0))
    # var ddof=1 -> std(ddof=1)^2 ile tutarli (SPEC ref 230.63 = ddof=1); var_pop ayrica saklanir
    t_std1 = float(y.std(ddof=1)); t_var = float(y.var(ddof=1)); t_var_pop = float(y.var(ddof=0))
    t_min = float(y.min()); t_max = float(y.max()); t_med = float(y.median())
    t_skew = float(skew(y)); t_kurt = float(kurtosis(y))
    n_eq100 = int((y >= 100.0).sum()); pct_eq100 = 100.0 * n_eq100 / len(y)
    n_le50 = int((y <= 50.0).sum()); pct_le50 = 100.0 * n_le50 / len(y)
    n_eq0 = int((y <= 0.0).sum())
    print(f"mean={t_mean:.4f} std(ddof=1)={t_std1:.4f} var(ddof=1)={t_var:.4f} "
          f"(std^2={t_std1**2:.4f}, tutarli)")
    print(f"median={t_med:.4f} min={t_min:.4f} max={t_max:.4f} "
          f"skew={t_skew:.4f} kurtosis={t_kurt:.4f}")
    print(f"==100: n={n_eq100} (%{pct_eq100:.4f}) | <=50: n={n_le50} (%{pct_le50:.4f}) "
          f"| ==0: n={n_eq0}")
    print("Dogrulama (SPEC referansi ile):")
    check("target.mean", t_mean, SPEC_REF["target_mean"], 0.05)
    check("target.std", t_std1, SPEC_REF["target_std"], 0.05)
    check("target.median", t_med, SPEC_REF["target_median"], 0.10)
    check("target.skew", t_skew, SPEC_REF["target_skew"], 0.02)
    check("target.var", t_var, SPEC_REF["target_var"], 1.0)
    check("pct_eq_100", pct_eq100, SPEC_REF["pct_eq_100"], 0.10)
    check("pct_le_50", pct_le50, SPEC_REF["pct_le_50"], 0.10)

    # Grafikler: histogram+KDE, ECDF, boxplot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    sns.histplot(y, bins=50, kde=True, ax=axes[0], color="#3b6fb6")
    axes[0].axvline(100, color="crimson", ls="--", lw=1)
    axes[0].set_title(f"Histogram (50 bin) + KDE\nmean={t_mean:.2f}, skew={t_skew:.2f}")
    axes[0].set_xlabel("career_success_score")
    ys = np.sort(y.values)
    ecdf = np.arange(1, len(ys) + 1) / len(ys)
    axes[1].plot(ys, ecdf, color="#2a9d8f")
    axes[1].axhline(1 - pct_eq100 / 100, color="crimson", ls="--", lw=1)
    axes[1].set_title("ECDF")
    axes[1].set_xlabel("career_success_score"); axes[1].set_ylabel("F(x)")
    sns.boxplot(x=y, ax=axes[2], color="#e9c46a")
    axes[2].set_title("Boxplot"); axes[2].set_xlabel("career_success_score")
    fig.suptitle("Hedef dagilimi — sol kuyruk + ==100 kutlesi", y=1.04, fontsize=13)
    b64 = save_fig(fig, "target_distribution.png")
    add_img("Hedef", b64, "Histogram+KDE / ECDF / Boxplot. ==100 spike (kirmizi) "
            "ve sol kuyruk Faz 2 stratify + Faz 7 clip gerekcesidir.")

    target_profile = {
        "mean": t_mean, "std": t_std1, "std_pop": t_std,
        "var": t_var, "var_pop": t_var_pop,
        "min": t_min, "max": t_max, "median": t_med,
        "skew": t_skew, "kurtosis": t_kurt,
        "pct_eq_100": pct_eq100, "n_eq_100": n_eq100,
        "pct_le_50": pct_le50, "n_le_50": n_le50, "n_eq_0": n_eq0,
        # mean_by_grad_year ADIM 8'de eklenecek
    }
    add_html("Hedef", "<p><b>Hedef portresi:</b> "
             f"mean={t_mean:.2f}, std={t_std1:.2f}, median={t_med:.2f}, "
             f"min={t_min:.2f}, max={t_max:.2f}, skew={t_skew:.3f}, "
             f"var={t_var:.2f}; <b>==100</b> %{pct_eq100:.2f} ({n_eq100}), "
             f"<b>&lt;=50</b> %{pct_le50:.2f} ({n_le50}), ==0 n={n_eq0}. "
             "Cift-sinirli yigin -> <code>clip[0,100]</code> notr/bedava.</p>")

    # ----------------------------------------------------------------------
    # ADIM 4 — Eksik deger haritasi + MNAR
    # ----------------------------------------------------------------------
    hr("ADIM 4 — Eksik deger haritasi + MNAR")
    miss_tr = train.isna().sum()
    miss_te = test.isna().sum()
    na_cols = sorted([c for c in train.columns if miss_tr[c] > 0],
                     key=lambda c: -miss_tr[c])
    print("Train'de NA'li kolonlar (azalan):")
    for c in na_cols:
        print(f"  {c:<34} train n={miss_tr[c]:<5} (%{100*miss_tr[c]/len(train):5.2f})"
              f"  test n={miss_te.get(c,0):<5} (%{100*miss_te.get(c,0)/len(test):5.2f})")
    assert miss_tr[TEXT] == 0 and miss_te[TEXT] == 0, "mentor_feedback_text'te NA olmamali!"
    print(f"mentor_feedback_text NA: train={miss_tr[TEXT]}, test={miss_te[TEXT]} (beklenen 0/0)")

    # SPEC referans missingness dogrulama
    print("Dogrulama (missingness %):")
    def pct_tr(c): return 100.0 * miss_tr[c] / len(train)
    check("miss internship_duration", pct_tr("internship_duration_months"),
          SPEC_REF["miss_internship_duration_pct"], 0.15)
    check("miss english_exam_score", pct_tr("english_exam_score"),
          SPEC_REF["miss_english_exam_pct"], 0.15)
    check("miss github_avg_stars", pct_tr("github_avg_stars"),
          SPEC_REF["miss_github_avg_stars_pct"], 0.15)
    check("miss open_source_contrib", pct_tr("open_source_contribution_count"),
          SPEC_REF["miss_open_source_pct"], 0.15)
    check("miss hr_interview_score", pct_tr("hr_interview_score"),
          SPEC_REF["miss_hr_interview_pct"], 0.15)
    check("miss linkedin_profile", pct_tr("linkedin_profile_score"),
          SPEC_REF["miss_linkedin_pct"], 0.15)
    check("miss portfolio_score", pct_tr("portfolio_score"),
          SPEC_REF["miss_portfolio_pct"], 0.15)

    # MNAR analizi: NA iken ilgili count==0 orani vs genel oran
    related = {
        "internship_duration_months": "internship_count",
        "github_avg_stars": "github_repo_count",
        "open_source_contribution_count": "github_repo_count",
        "hr_interview_score": "interviews_attended",
        "english_exam_score": None,
        "linkedin_profile_score": None,
        "portfolio_score": None,
    }
    rows = []
    for c in na_cols:
        rc = related.get(c)
        na_mask = train[c].isna()
        rec = {
            "column": c,
            "n_missing_train": int(miss_tr[c]),
            "pct_missing_train": round(pct_tr(c), 4),
            "n_missing_test": int(miss_te.get(c, 0)),
            "pct_missing_test": round(100.0 * miss_te.get(c, 0) / len(test), 4),
            "related_zero_col": rc if rc else "",
        }
        if rc is not None:
            related_zero_overall = float((train[rc] == 0).mean())
            related_zero_given_na = float((train.loc[na_mask, rc] == 0).mean())
            rec["pct_overall_related_zero"] = round(100 * related_zero_overall, 4)
            rec["pct_na_with_related_zero"] = round(100 * related_zero_given_na, 4)
            rec["mnar_flag"] = bool(related_zero_given_na > 2 * related_zero_overall
                                    and related_zero_given_na > 0.5)
        else:
            rec["pct_overall_related_zero"] = np.nan
            rec["pct_na_with_related_zero"] = np.nan
            rec["mnar_flag"] = False
        rows.append(rec)
    missing_map = pd.DataFrame(rows)

    # internship MNAR vurgusu
    intr = missing_map.set_index("column").loc["internship_duration_months"]
    print(f"\nMNAR — internship_duration_months NA iken internship_count==0 orani = "
          f"%{intr['pct_na_with_related_zero']:.2f}  "
          f"(genel %{intr['pct_overall_related_zero']:.2f})")
    check("internship MNAR %", intr["pct_na_with_related_zero"],
          SPEC_REF["internship_mnar_pct"], 1.0)

    # AUDIT: open_source_contribution_count MNAR DEGIL -> github_avg_stars ile
    # ayni maske mi? + bu satirlarda repo VAR mi? (0 enjekte ETME gerekcesi)
    m_os = train["open_source_contribution_count"].isna()
    m_gh = train["github_avg_stars"].isna()
    same_mask = bool((m_os == m_gh).all())
    n_os = int(m_os.sum())
    pct_repo_present = float((train.loc[m_os, "github_repo_count"] > 0).mean()) * 100
    print(f"AUDIT open_source vs github_avg_stars: ayni maske mi? {same_mask} "
          f"(n={n_os}); bu satirlarin %{pct_repo_present:.1f}'inde github_repo_count>0 "
          f"-> 0 ENJEKTE ETME, medyan+bayrak.")
    missing_map.loc[missing_map["column"] == "open_source_contribution_count",
                    "note"] = (f"github_avg_stars ile ayni maske={same_mask}; "
                               f"%{pct_repo_present:.1f} repo VAR -> medyan+bayrak, 0 DEGIL")
    missing_map["note"] = missing_map.get("note", "").fillna("")
    missing_map.loc[missing_map["column"] == "internship_duration_months", "note"] = \
        "MNAR: NA ~ internship_count==0 -> 0+bayrak (Faz 3)"
    missing_map.to_csv(OUT / "missing_map.csv", index=False)
    print(f"-> missing_map.csv yazildi ({len(missing_map)} satir)")

    # Grafik: missingness bar
    miss_df = pd.DataFrame({
        "train": (miss_tr[na_cols] / len(train) * 100).values,
        "test": (miss_te[na_cols] / len(test) * 100).values,
    }, index=na_cols)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    miss_df.plot(kind="barh", ax=ax, color=["#3b6fb6", "#e76f51"])
    ax.set_xlabel("eksik %"); ax.set_title("Eksik deger orani — train vs test (~ozdes)")
    ax.invert_yaxis()
    b64 = save_fig(fig, "missing_train_test.png")
    add_img("Eksik", b64, "7 NA'li sayisal kolon; train/test missingness ~ozdes "
            "(impute icin dagilim kaymasi yok).")
    add_html("Eksik", df_html(missing_map))
    add_html("Eksik", f"<p><b>MNAR:</b> internship_duration_months NA iken "
             f"internship_count==0 orani %{intr['pct_na_with_related_zero']:.2f} "
             f"(genel %{intr['pct_overall_related_zero']:.2f}) -> 'staj yok' semantigi, "
             "ortalama-impute sahte 'orta staj' uydurur. "
             f"<b>AUDIT:</b> open_source_contribution_count, github_avg_stars ile "
             f"ayni maske ({same_mask}); bu satirlarin %{pct_repo_present:.1f}'inde "
             "repo VAR -> MNAR DEGIL, medyan+bayrak (0 enjekte ETME).</p>")

    # ----------------------------------------------------------------------
    # ADIM 5 — Kategorik kardinalite & yeni-seviye taramasi
    # ----------------------------------------------------------------------
    hr("ADIM 5 — Kategorik kardinalite & yeni-seviye taramasi")
    cat_rows = []
    for c in CATEGORICAL:
        tr_levels = set(train[c].dropna().unique())
        te_levels = set(test[c].dropna().unique())
        test_only = te_levels - tr_levels
        train_only = tr_levels - te_levels
        cat_rows.append({
            "column": c,
            "nunique_train": train[c].nunique(),
            "nunique_test": test[c].nunique(),
            "test_only_levels": sorted(map(str, test_only)),
            "train_only_levels": sorted(map(str, train_only)),
            "levels": sorted(map(str, tr_levels)),
        })
        print(f"  {c:<32} nunique train={train[c].nunique()} test={test[c].nunique()} "
              f"| test-only={sorted(map(str, test_only))}")
    cat_df = pd.DataFrame(cat_rows)
    n_test_only = sum(len(r["test_only_levels"]) for r in cat_rows)
    print(f"\nToplam test-only seviye = {n_test_only} (beklenen 0)")
    assert n_test_only == 0, "Test-only kategorik seviye bulundu!"
    # university_tier ordinal mi? (Tier 1-4 dogal sirali)
    ut_levels = sorted(map(str, set(train["university_tier"].dropna().unique())))
    print(f"university_tier seviyeleri (ordinal aday): {ut_levels}")
    add_html("Kategorik", df_html(cat_df[["column", "nunique_train", "nunique_test", "levels"]]))
    add_html("Kategorik", f"<p>Kardinalite dusuk (4-11); <b>test-only seviye YOK</b> "
             "(gorulmemis seviye riski yok). <code>university_tier</code> dogal sirali "
             f"({ut_levels}) -> Faz 3/4 ordinal encode adayi.</p>")

    # ----------------------------------------------------------------------
    # ADIM 6 — Sayisal korelasyon analizi
    # ----------------------------------------------------------------------
    hr("ADIM 6 — Sayisal korelasyon analizi (hedefe |corr|)")
    num_for_corr = NUM + YEAR  # yillar da sayisal dtype; corr'da gosterilir, role=year
    corr_with_target = {}
    for c in num_for_corr:
        corr_with_target[c] = float(train[c].corr(y))
    corr_ser = pd.Series(corr_with_target).reindex(num_for_corr)
    corr_sorted = corr_ser.reindex(corr_ser.abs().sort_values(ascending=False).index)
    print("Hedefe |corr| sirali (ust 12):")
    print(corr_sorted.head(12).to_string())
    pq_corr = float(train["project_quality_score"].corr(y))
    check("corr project_quality_score", pq_corr, SPEC_REF["corr_project_quality"], 0.02)
    assert corr_sorted.abs().max() < 0.55 or abs(pq_corr) >= corr_sorted.abs().max() - 1e-9, \
        "Beklenmedik: 0.55 ustu bir tekil korelasyon var (suphe!)"
    print(f"En guclu tekil sinyal: project_quality_score corr={pq_corr:.4f} "
          f"(hicbir feature 0.55 ustu degil -> sinyal etkilesimlerde, GBDT > lineer)")

    # Korelasyon isi haritasi (sayisal feature'lar, yil haric daha okunur)
    cmat = train[NUM].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cmat, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.6}, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Sayisal feature korelasyon isi haritasi (9 teknik skor kumesi belirgin)")
    b64 = save_fig(fig, "corr_heatmap.png")
    add_img("Sayisal-Korelasyon", b64, "Multikolineerlik kumeleri (9 teknik skor) "
            "Faz 4 kompozit (tech_mean) tasarimina isaret eder.")

    fig, ax = plt.subplots(figsize=(8, 9))
    top = corr_sorted.head(20)
    colors = ["#e76f51" if c in YEAR else "#3b6fb6" for c in top.index]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("corr(feature, hedef)")
    ax.set_title("Hedefe korelasyon — ust 20 (turuncu=yil, supheli)")
    b64 = save_fig(fig, "corr_with_target_bar.png")
    add_img("Sayisal-Korelasyon", b64, "project_quality_score tek basina baskin "
            "(~0.54); hicbir feature 0.55 ustu degil.")
    add_html("Sayisal-Korelasyon", df_html(corr_sorted.head(15).to_frame("corr_with_target")))

    # ----------------------------------------------------------------------
    # ADIM 7 — Sezgisel feature on-dogrulamasi (SADECE OLCUM, uretim DEGIL)
    # ----------------------------------------------------------------------
    hr("ADIM 7 — Sezgisel feature on-dogrulamasi (olcum; feature URETILMEZ)")
    tech_mean = train[TECH].mean(axis=1)
    corr_tech_mean = float(tech_mean.corr(y))
    pq_x_tech = train["project_quality_score"] * tech_mean
    corr_pq_x_tech = float(pq_x_tech.corr(y))
    apps = train["applications_sent"].replace(0, np.nan)
    conv_rate = train["interviews_attended"] / apps
    corr_conv = float(conv_rate.corr(y))
    print(f"tech_mean (9 skor ort.) corr = {corr_tech_mean:.4f}")
    print(f"project_quality_score x tech_mean corr = {corr_pq_x_tech:.4f}")
    print(f"conv_rate (interviews/applications) corr = {corr_conv:.4f}")
    check("corr tech_mean", corr_tech_mean, SPEC_REF["corr_tech_mean"], 0.02)
    check("corr pq x tech_mean", corr_pq_x_tech, SPEC_REF["corr_pq_x_tech"], 0.02)
    check("corr conv_rate", corr_conv, SPEC_REF["corr_conv_rate"], 0.02)
    heur = pd.DataFrame({
        "on_olcum_feature": ["tech_mean (9 teknik ort.)",
                             "project_quality_score x tech_mean",
                             "conv_rate = interviews/applications"],
        "corr_with_target": [round(corr_tech_mean, 4), round(corr_pq_x_tech, 4),
                             round(corr_conv, 4)],
        "Faz4_karari": ["tek skordan guclu -> kompozit adayi",
                        "EN GUCLU -> capa carpim adayi",
                        "sinyalsiz gurultu -> URETILMEYECEK"],
    })
    print(heur.to_string(index=False))
    add_html("Sayisal-Korelasyon", "<h4>Sezgisel feature on-olcumu (Faz 4 girdisi; "
             "burada feature URETILMEZ)</h4>" + df_html(heur))

    # ----------------------------------------------------------------------
    # ADIM 8 — YIL KOLONLARI dagilim kaymasi (en kritik EDA bulgusu)
    # ----------------------------------------------------------------------
    hr("ADIM 8 — Yil kolonlari dagilim kaymasi + hedef-by-yil drift")
    for c in YEAR:
        vc_tr = train[c].value_counts().sort_index()
        vc_te = test[c].value_counts().sort_index()
        comp = pd.DataFrame({"train": vc_tr, "test": vc_te}).fillna(0).astype(int)
        print(f"\n{c} value_counts (train vs test):")
        print(comp.to_string())

    # Hedef ortalamasi graduation_year boyunca (drift olcumu)
    mean_by_grad = train.groupby("graduation_year")[TARGET].mean().sort_index()
    drift = float(mean_by_grad.max() - mean_by_grad.min())
    print(f"\nHedef ortalamasi graduation_year boyunca:")
    print(mean_by_grad.round(2).to_string())
    print(f"max-min drift = {drift:.2f} puan (zayif -> yili atmak ~bedava)")
    target_profile["mean_by_grad_year"] = {int(k): round(float(v), 4)
                                           for k, v in mean_by_grad.items()}
    target_profile["mean_by_grad_year_drift"] = round(drift, 4)
    # target_profile.json yaz (ADIM 3 + ADIM 8 birlikte)
    with open(OUT / "target_profile.json", "w", encoding="utf-8") as f:
        json.dump(target_profile, f, ensure_ascii=False, indent=2)
    print("-> target_profile.json yazildi")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, c in zip(axes[:2], YEAR):
        vc_tr = train[c].value_counts().sort_index()
        vc_te = test[c].value_counts().sort_index()
        idx = sorted(set(vc_tr.index) | set(vc_te.index))
        w = 0.4
        ax.bar([i - w / 2 for i in range(len(idx))], vc_tr.reindex(idx).fillna(0),
               width=w, label="train", color="#3b6fb6")
        ax.bar([i + w / 2 for i in range(len(idx))], vc_te.reindex(idx).fillna(0),
               width=w, label="test", color="#e76f51")
        ax.set_xticks(range(len(idx))); ax.set_xticklabels(idx, rotation=45)
        ax.set_title(f"{c}\ntrain ~uniform / test 2024-26 yiginli"); ax.legend()
    axes[2].plot(mean_by_grad.index, mean_by_grad.values, "o-", color="#2a9d8f")
    axes[2].set_title(f"Hedef ort. ~ graduation_year\n(drift {drift:.1f} puan, zayif)")
    axes[2].set_xlabel("graduation_year"); axes[2].set_ylabel("mean target")
    b64 = save_fig(fig, "year_shift.png")
    add_img("Yil-Kayma", b64, "Train her yilda ~uniform; test 2024-26'ya yiginli. "
            "Hedef-by-yil drift zayif -> yili ham feature atmak ~bedava, tek kaymayi yok eder.")
    add_html("Yil-Kayma", f"<p>Hedef ortalamasi graduation_year boyunca yalniz "
             f"<b>{drift:.1f} puan</b> degisir (zayif drift). Yillari ham feature "
             "kullanmamak neredeyse hic hedef sinyali kaybettirmez ama tek dagilim "
             "kaymasini yok eder. ADIM 9 (adversarial) bunu nicel kanitlar.</p>")

    # ----------------------------------------------------------------------
    # ADIM 9 — ADVERSARIAL VALIDATION (sigorta dogrulamasi)
    # ----------------------------------------------------------------------
    hr("ADIM 9 — Adversarial validation (train=0 / test=1)")
    feat_cols = [c for c in test.columns if c not in (ID, TEXT)]  # hedef test'te zaten yok
    Xtr = train[feat_cols].copy()
    Xte = test[feat_cols].copy()
    Xall = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    yadv = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))]).astype(int)

    # Kategorikleri ordinal-encode et (adversarial siniflandirici icin; gorulmemis
    # seviye yok -> birlesik fit guvenli, bu yalniz dagilim-kaymasi olcumu).
    cat_in_feats = [c for c in CATEGORICAL if c in feat_cols]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xall_enc = Xall.copy()
    Xall_enc[cat_in_feats] = enc.fit_transform(Xall[cat_in_feats].astype("object"))
    Xall_enc = Xall_enc.apply(pd.to_numeric, errors="coerce")

    def adv_auc(cols):
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=SEED)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        proba = cross_val_predict(clf, Xall_enc[cols], yadv, cv=skf,
                                  method="predict_proba")[:, 1]
        return float(roc_auc_score(yadv, proba))

    cols_with = list(Xall_enc.columns)
    cols_without = [c for c in cols_with if c not in YEAR]
    cols_without_num = [c for c in cols_without if c not in cat_in_feats]  # SADECE sayisal
    auc_with = adv_auc(cols_with)
    auc_without = adv_auc(cols_without)              # gercek feature uzayi (num+kat, yil yok)
    auc_without_num = adv_auc(cols_without_num)      # sadece sayisal (SPEC 0.4995 ile kiyas)
    print(f"adversarial AUC  yillarla (num+kat)      = {auc_with:.4f}")
    print(f"adversarial AUC  yillarsiz (num+kat)     = {auc_without:.4f}")
    print(f"adversarial AUC  yillarsiz (SADECE num)  = {auc_without_num:.4f}")
    check("adv AUC with years", auc_with, SPEC_REF["adv_auc_with_years"], 0.03)
    # SPEC'in 0.4995 referansi SADECE-SAYISAL olcumdur; onunla kiyasla (PASS):
    check("adv AUC yillarsiz (num-only ~ SPEC)", auc_without_num,
          SPEC_REF["adv_auc_without_years"], 0.03)

    # BULGU: num+kat yillarsiz AUC (0.535) num-only'dan (0.494) yuksek. Suclu kategorik?
    # target_role train/test dagilim kaymasi (yil drift'inin kategorik yansimasi).
    role_tr = train["target_role"].value_counts(normalize=True)
    role_te = test["target_role"].value_counts(normalize=True)
    role_cmp = pd.DataFrame({"train_pct": role_tr * 100, "test_pct": role_te * 100}).fillna(0)
    role_cmp["abs_diff"] = (role_cmp["train_pct"] - role_cmp["test_pct"]).abs()
    role_cmp = role_cmp.sort_values("abs_diff", ascending=False).round(3)
    role_max_diff = float(role_cmp["abs_diff"].max())
    top_shift_roles = [
        {"role": idx, "train_pct": round(float(r.train_pct), 2),
         "test_pct": round(float(r.test_pct), 2), "abs_diff": round(float(r.abs_diff), 2)}
        for idx, r in role_cmp.head(4).iterrows()
    ]
    safe = auc_without < 0.55  # esik: <0.55 guvenli bolge (random CV sadik)
    print(f"\nBULGU — yillarsiz num+kat AUC ({auc_without:.4f}) > num-only ({auc_without_num:.4f}). "
          f"Suclu: target_role kaymasi (max |diff|={role_max_diff:.2f} puan):")
    print(role_cmp.head(5).to_string())
    print(f"  -> Bu, yil drift'inin kategorik yansimasi (test 2024-26'ya yiginli; yeni "
          f"AI/ML/MLOps rolleri test'te daha sik). {auc_without:.3f} {'<' if safe else '>='} 0.55 "
          f"-> {'hala GUVENLI bolge' if safe else 'ESIK ASILDI, INCELE'}; "
          f"random stratified KFold gecerli. Faz 4 nihai matriste ~{auc_without:.2f} beklemeli "
          f"(tam 0.50 DEGIL); target_role degerli feature, ATILMAZ. AUC>0.6 olursa incele.")

    # Suclu kolonlar: permutation importance (yillarla setте)
    clf_full = HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.05, random_state=SEED)
    clf_full.fit(Xall_enc[cols_with], yadv)
    perm = permutation_importance(clf_full, Xall_enc[cols_with], yadv,
                                  scoring="roc_auc", n_repeats=5, random_state=SEED)
    imp = pd.Series(perm.importances_mean, index=cols_with).sort_values(ascending=False)
    top_features = [{"feature": k, "importance": round(float(v), 6)}
                    for k, v in imp.head(12).items()]
    print("\nSuclu kolonlar (permutation importance, ust 8):")
    print(imp.head(8).round(5).to_string())

    adversarial_auc = {
        "auc_with_years": round(auc_with, 4),
        "auc_without_years": round(auc_without, 4),
        "auc_without_years_numeric_only": round(auc_without_num, 4),
        "classifier": "HistGradientBoostingClassifier(max_iter=200,max_depth=4,lr=0.05,rs=42)",
        "cv": "StratifiedKFold(5, shuffle=True, random_state=42), cross_val_predict proba",
        "categorical_encoding": "OrdinalEncoder (combined fit; adversarial-only, no leakage concern)",
        "year_cols_excluded_in_without": YEAR,
        "top_features": top_features,
        "finding_target_role_shift": {
            "note": ("SPEC referansi 0.4995 SADECE-SAYISAL olcumdur (bizimki "
                     f"{auc_without_num:.4f}, PASS). Gercek feature uzayinda (num+kategorik, "
                     f"yil yok) AUC {auc_without:.4f}; fark target_role train/test "
                     f"kaymasindan (yil drift'inin kategorik yansimasi). {auc_without:.3f}"
                     f"{'<' if safe else '>='}0.55 -> {'GUVENLI' if safe else 'ESIK ASILDI'}; "
                     f"random stratified KFold gecerli. Faz 4 nihai matris ~{auc_without:.2f} "
                     "beklemeli (tam 0.50 degil); target_role degerli, ATILMAZ; AUC>0.6 -> incele."),
            "safe_zone_below_0_55": bool(safe),
            "target_role_max_abs_pct_diff": round(role_max_diff, 3),
            "top_shifted_roles": top_shift_roles,
        },
    }
    with open(OUT / "adversarial_auc.json", "w", encoding="utf-8") as f:
        json.dump(adversarial_auc, f, ensure_ascii=False, indent=2)
    print("-> adversarial_auc.json yazildi")

    fig, ax = plt.subplots(figsize=(8, 6))
    impt = imp.head(12)
    colors = ["#e76f51" if c in YEAR else "#3b6fb6" for c in impt.index]
    ax.barh(impt.index[::-1], impt.values[::-1], color=colors[::-1])
    ax.set_xlabel("permutation importance (roc_auc dususu)")
    ax.set_title(f"Adversarial suclular (turuncu=yil)\nAUC yilli={auc_with:.3f} / "
                 f"yilsiz={auc_without:.3f}")
    b64 = save_fig(fig, "adversarial_importance.png")
    add_img("Adversarial", b64, "Yil kolonlari ayrimi sirtlar. Yillar cikinca "
            "train/test ayirt edilemez (AUC ~0.50) -> random stratified KFold "
            "private MSE'nin sadik temsilcisi (Faz 2 temeli).")
    add_html("Adversarial", f"<p><b>Sonuc:</b> AUC yillarla=<b>{auc_with:.4f}</b>, "
             f"yillarsiz (num+kategorik)=<b>{auc_without:.4f}</b>, "
             f"yillarsiz (SADECE sayisal)=<b>{auc_without_num:.4f}</b>. "
             "Birincil dagilim kaymasi yil kolonlarinda; yilsiz uzayda train↔test "
             "neredeyse ayrilamaz. Bu, Faz 2'nin random stratified KFold temelidir.</p>")
    role_html = pd.DataFrame(top_shift_roles).to_html(classes="tbl", border=0, index=False)
    add_html("Adversarial", "<h4>BULGU — target_role ikincil kaymasi (SPEC refinement)</h4>"
             f"<p>SPEC'in <code>0.4995</code> referansi <b>sadece-sayisal</b> olcumdur "
             f"(bizimki {auc_without_num:.3f}, eslesti). Gercek feature uzayinda "
             f"(num+kategorik) yillarsiz AUC <b>{auc_without:.3f}</b>; fark "
             f"<code>target_role</code> kaymasindan (max |diff| {role_max_diff:.2f} puan) "
             "— yil drift'inin kategorik yansimasi (test 2024-26'ya yiginli, yeni "
             f"AI/ML/MLOps rolleri test'te daha sik). <b>{auc_without:.3f} "
             f"{'&lt;' if safe else '&gt;='} 0.55 -> {'GUVENLI bolge' if safe else 'ESIK ASILDI'}</b>, "
             f"random CV gecerli. Faz 4 nihai matriste ~{auc_without:.2f} beklemeli (tam 0.50 degil); "
             "target_role degerli sinyal, ATILMAZ; AUC>0.6 olursa incele.</p>" + role_html)

    # ----------------------------------------------------------------------
    # ADIM 10 — Metin (mentor_feedback_text) on inceleme
    # ----------------------------------------------------------------------
    hr("ADIM 10 — Metin on inceleme (Turkce NLP girdisi)")
    # UTF-8 byte teyidi (mojibake fix YASAK gerekcesi)
    o_umlaut = "ö"  # 'o with diaeresis'
    utf8_ok = o_umlaut.encode("utf-8") == b"\xc3\xb6"
    print(f"UTF-8 byte teyidi: 'o-umlaut'.encode('utf-8')==b'\\xc3\\xb6' -> {utf8_ok} "
          "(temiz UTF-8; mojibake fix YASAK, konsol gorunumu yalniz codepage)")
    assert utf8_ok

    txt = train[TEXT].astype(str)
    txt_te = test[TEXT].astype(str)
    word_len = txt.str.split().map(len)
    char_len = txt.str.len()
    wl_mean = float(word_len.mean()); wl_min = int(word_len.min()); wl_max = int(word_len.max())
    cl_mean = float(char_len.mean()); cl_min = int(char_len.min()); cl_max = int(char_len.max())
    n_unique = int(txt.nunique()); pct_unique = 100.0 * n_unique / len(txt)
    n_digit = int(txt.str.contains(r"\d", regex=True).sum())
    wl_mean_te = float(txt_te.str.split().map(len).mean())
    print(f"kelime/satir: mean={wl_mean:.2f} min={wl_min} max={wl_max}")
    print(f"karakter/satir: mean={cl_mean:.2f} min={cl_min} max={cl_max}")
    print(f"benzersiz metin: {n_unique}/{len(txt)} (%{pct_unique:.2f})")
    print(f"rakam iceren satir: {n_digit} (beklenen 0 -> hazir-cevap/hedef sizintisi yok)")
    print(f"test kelime ort.: {wl_mean_te:.2f} (train ile ~ozdes)")
    assert n_digit == 0, "Metinde rakam bulundu — hedef sizintisi suphesi!"

    # Anahtar kelime frekanslari (Turkce-lowercase + substring/kok)
    txt_lower = txt.map(tr_lower)
    keywords = {
        "ancak": "ancak", "gelistir": "geliştir", "potansiyel": "potansiyel",
        "guclu": "güçlü", "basari": "başarı", "mukemmel": "mükemmel",
        "olaganustu": "olağanüstü", "ustun": "üstün", "gerekiyor": "gerekiyor",
        "eksik": "eksik",
    }
    kw_freq = {}        # ASCII anahtar -> sayim (JSON; stabil identifier, Faz 5 icin)
    kw_display = {}     # Turkce terim -> sayim (rapor figur + HTML; sadik gosterim)
    print("Anahtar kelime frekanslari (satir-iceren-substring, Turkce-lowercase):")
    for key, term in keywords.items():
        term_l = tr_lower(term)
        cnt = int(txt_lower.str.contains(re.escape(term_l), regex=True).sum())
        kw_freq[key] = cnt
        kw_display[term] = cnt
        print(f"  {term:<12} ({key:<10}) = {cnt}")
        if f"kw_{key}" in SPEC_REF:
            check(f"kw {key}", cnt, SPEC_REF[f"kw_{key}"], 60)

    check("text word mean", wl_mean, SPEC_REF["text_word_mean"], 0.6)
    check("text char mean", cl_mean, SPEC_REF["text_char_mean"], 3.0)
    check("text word mean (test)", wl_mean_te, SPEC_REF["text_word_mean_test"], 0.6)

    text_profile = {
        "utf8_byte_check": utf8_ok,
        "bom_present_handled_with": "utf-8-sig",
        "word_len_mean": round(wl_mean, 4), "word_len_min": wl_min, "word_len_max": wl_max,
        "char_len_mean": round(cl_mean, 4), "char_len_min": cl_min, "char_len_max": cl_max,
        "n_unique": n_unique, "pct_unique": round(pct_unique, 4),
        "n_rows_with_digit": n_digit,
        "word_len_mean_test": round(wl_mean_te, 4),
        "keyword_freq": kw_freq,
        "normalization": "tr_lower (I->dotless-i, dotted-I->i); metin ve lexicon AYNI",
    }
    with open(OUT / "text_profile.json", "w", encoding="utf-8") as f:
        json.dump(text_profile, f, ensure_ascii=False, indent=2)
    print("-> text_profile.json yazildi")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    sns.histplot(word_len, bins=30, ax=axes[0], color="#3b6fb6")
    axes[0].set_title(f"Kelime/satir dagilimi (mean={wl_mean:.1f}, [{wl_min},{wl_max}])")
    axes[0].set_xlabel("kelime sayisi")
    kw_ser = pd.Series(kw_display).sort_values(ascending=False)  # Turkce terimli gosterim
    axes[1].barh(kw_ser.index[::-1], kw_ser.values[::-1], color="#2a9d8f")
    axes[1].set_title("Anahtar kelime frekanslari (Faz 5 lexicon baz cizgisi)")
    axes[1].set_xlabel("iceren satir sayisi")
    b64 = save_fig(fig, "text_profile.png")
    add_img("Metin", b64, "Tam sablon ama her satir farkli (benzersiz=10000/10000), "
            "rakam YOK (sizinti yok). Frekanslar Faz 5 lexicon baz cizgisi "
            "(lexicon hedefe bakarak SECILMEZ).")
    add_html("Metin", f"<p>UTF-8 byte teyidi=<b>{utf8_ok}</b> (mojibake fix YASAK). "
             f"benzersiz=<b>{n_unique}/{len(txt)}</b>, rakam iceren satir=<b>{n_digit}</b>, "
             f"kelime/satir mean={wl_mean:.1f} [{wl_min},{wl_max}], "
             f"test kelime ort.={wl_mean_te:.1f} (train ile ~ozdes).</p>")
    add_html("Metin", df_html(kw_ser.to_frame("iceren_satir")))

    # ----------------------------------------------------------------------
    # ADIM 2/6 birlesimi — column_profile.csv (tum kolonlar)
    # ----------------------------------------------------------------------
    hr("column_profile.csv uretimi")
    prof_rows = []
    for c in train.columns:
        prof_rows.append({
            "column": c,
            "dtype": str(train[c].dtype),
            "role": roles[c],
            "n_missing_train": int(miss_tr[c]),
            "pct_missing_train": round(100.0 * miss_tr[c] / len(train), 4),
            "n_missing_test": int(miss_te.get(c, 0)) if c != TARGET else "",
            "pct_missing_test": (round(100.0 * miss_te.get(c, 0) / len(test), 4)
                                 if c != TARGET else ""),
            "nunique": int(train[c].nunique(dropna=True)),
            "corr_with_target": (round(corr_with_target[c], 4)
                                 if c in corr_with_target else ""),
            "is_year_suspect": roles[c] == "year",
        })
    column_profile = pd.DataFrame(prof_rows)
    column_profile.to_csv(OUT / "column_profile.csv", index=False)
    print(f"-> column_profile.csv yazildi ({len(column_profile)} satir)")
    print(column_profile[["column", "role", "pct_missing_train", "nunique",
                          "corr_with_target", "is_year_suspect"]].to_string(index=False))

    # ----------------------------------------------------------------------
    # ADIM 11 — Raporlama (eda_report.html, 7 bolum)
    # ----------------------------------------------------------------------
    hr("ADIM 11 — eda_report.html (7 bolum)")
    write_html_report(SECTIONS, OUT / "eda_report.html",
                      auc_with, auc_without, drift, pct_eq100, pct_le50)
    print("-> eda_report.html yazildi")

    # ----------------------------------------------------------------------
    # Dogrulama ozeti (PASS/WARN)
    # ----------------------------------------------------------------------
    hr("DOGRULAMA OZETI (hesaplanan vs SPEC referansi)")
    n_warn = sum(1 for *_ , s in _CHECKS if s == "WARN")
    for label, val, ref, tol, status in _CHECKS:
        if status == "WARN":
            print(f"  [WARN] {label}: hesap={val:.4f} spec={ref:.4f} (|fark|={abs(val-ref):.4f} > {tol})")
    print(f"\nToplam {len(_CHECKS)} kontrol, {n_warn} WARN, {len(_CHECKS)-n_warn} PASS.")
    if n_warn == 0:
        print("TUM SAYILAR SPEC REFERANSI ILE ESLESTI (tolerans icinde).")
    else:
        print("DIKKAT: WARN var -> yukarida raporlandi; MASTERPLAN'a geri bildirim dusulmeli.")

    hr("ARTEFAKTLAR")
    for f in sorted(OUT.glob("*")):
        if f.is_file():
            print(f"  {f.relative_to(ROOT)}  ({f.stat().st_size} bytes)")
    print("\nFaz 1 EDA tamam.")


def write_html_report(sections, path, auc_with, auc_without, drift, pct100, pct50):
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;
         color:#1f2933;background:#f7f9fb;}
    header{background:#22303f;color:#fff;padding:24px 32px;}
    header h1{margin:0 0 6px;font-size:24px;} header p{margin:2px 0;color:#c5d0db;font-size:13px;}
    nav{background:#2f4156;padding:8px 32px;position:sticky;top:0;z-index:5;}
    nav a{color:#cfe0f1;margin-right:16px;text-decoration:none;font-size:13px;}
    nav a:hover{color:#fff;text-decoration:underline;}
    section{background:#fff;margin:18px 32px;padding:20px 26px;border-radius:8px;
            box-shadow:0 1px 3px rgba(0,0,0,.08);}
    section h2{margin-top:0;color:#22303f;border-bottom:2px solid #e1e8ef;padding-bottom:8px;}
    img{max-width:100%;height:auto;border:1px solid #e1e8ef;border-radius:6px;margin:10px 0;}
    .cap{font-size:12px;color:#5a6b7b;margin:-4px 0 14px;}
    table.tbl{border-collapse:collapse;font-size:12px;margin:10px 0;width:100%;}
    table.tbl th{background:#eef3f8;padding:6px 9px;border:1px solid #dde5ee;}
    table.tbl td{padding:5px 9px;border:1px solid #eef2f6;text-align:center;}
    code{background:#eef3f8;padding:1px 5px;border-radius:4px;font-size:12px;}
    .kpi{display:inline-block;background:#eef6f1;border:1px solid #cfe6da;border-radius:6px;
         padding:8px 14px;margin:4px 8px 4px 0;font-size:13px;}
    .kpi b{color:#1d6f54;}
    """
    order = ["Hedef", "Eksik", "Kategorik", "Sayisal-Korelasyon", "Yil-Kayma",
             "Adversarial", "Metin"]
    titles = {
        "Hedef": "1. Hedef Dagilimi",
        "Eksik": "2. Eksik Degerler & MNAR",
        "Kategorik": "3. Kategorik Kardinalite",
        "Sayisal-Korelasyon": "4. Sayisal Korelasyon & Sezgisel Feature Olcumu",
        "Yil-Kayma": "5. Yil Kolonlari — Dagilim Kaymasi",
        "Adversarial": "6. Adversarial Validation (sigorta)",
        "Metin": "7. Metin (Turkce NLP girdisi)",
    }
    nav = " ".join(f'<a href="#{s}">{titles[s].split(".")[0]}. {s}</a>' for s in order)
    kpis = (
        f'<span class="kpi">adversarial AUC yilli <b>{auc_with:.3f}</b> / '
        f'yilsiz <b>{auc_without:.3f}</b></span>'
        f'<span class="kpi">==100 <b>%{pct100:.2f}</b></span>'
        f'<span class="kpi">&lt;=50 <b>%{pct50:.2f}</b></span>'
        f'<span class="kpi">hedef-by-yil drift <b>{drift:.1f} puan</b></span>'
    )
    body = [f'<header><h1>Datathon 2026 — Faz 1: EDA & Veri Anlama</h1>'
            f'<p>career_success_score regresyonu · MSE · 0-OVERFIT sigortasi · '
            f'SEED=42 · internet kapali · tum sayilar HESAPLANIR</p>'
            f'<div style="margin-top:10px">{kpis}</div></header>',
            f'<nav>{nav}</nav>']
    for s in order:
        chunks = "\n".join(sections[s])
        body.append(f'<section id="{s}"><h2>{titles[s]}</h2>{chunks}</section>')
    body.append('<section><h2>Notlar</h2><p>Bu rapor betimleyicidir; HICBIR karar '
                'leaderboard\'a veya hedefe bakarak alinmamistir. Tum feature/model '
                'kararlari Faz 2-7\'de fold-ici fit + 0.25·std kabul kapisindan gecer. '
                'Yil kolonlari ham feature KULLANILMAZ; metin TEMIZ UTF-8 '
                '(mojibake fix YASAK).</p></section>')
    html = (f'<!doctype html><html lang="tr"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Faz 1 EDA — Datathon 2026</title><style>{css}</style></head>'
            f'<body>{"".join(body)}</body></html>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
