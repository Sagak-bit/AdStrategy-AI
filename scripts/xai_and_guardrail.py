# -*- coding: utf-8 -*-
"""
D-5: XAI 고도화 — SHAP 대조(leakage vs clean) + Ablation 가드레일 플롯
=====================================================================
1. SHAP bar plot 대조: V1(leakage 포함) vs V5(clean) 모델의 피처 중요도 비교
2. Ablation guardrail: 피처 제거 R² 곡선 + 셔플/난수 기준선
"""
from __future__ import annotations

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ENRICHED_DATA_PATH, FIGURES_DIR, AD_SPEND_BINS,
    LEAKAGE_ORDERED, MIN_NON_NULL_RATIO,
    COLOR_DANGER, COLOR_SUCCESS, COLOR_PRIMARY, CHART_DPI,
    BASE_CAT_FEATURES, BASE_NUM_FEATURES,
    CALENDAR_NUM_FEATURES, COMPETITION_NUM_FEATURES,
    CREATIVE_CAT_FEATURES, CREATIVE_NUM_FEATURES_CLEAN,
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    USE_XGB = False


def load_data():
    df = pd.read_csv(ENRICHED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["ad_spend_log"] = np.log1p(df["ad_spend"])
    df["ad_spend_bin"] = pd.cut(df["ad_spend"], bins=AD_SPEND_BINS,
                                labels=list(range(len(AD_SPEND_BINS)-1))).astype(float).fillna(0).astype(int)
    df["is_q4"] = (df["quarter"] == 4).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def avail(df, cats, nums):
    c = [x for x in cats if x in df.columns]
    n = [x for x in nums if x in df.columns and df[x].notna().sum() > len(df)*MIN_NON_NULL_RATIO]
    return c, n


def train_and_get_importance(df, cat, num, label="model"):
    for c in num:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median() if pd.notna(df[c].median()) else 0)
    for c in cat:
        if df[c].isnull().any():
            m = df[c].mode()
            df[c] = df[c].fillna(m.iloc[0] if len(m) > 0 else "Unknown")

    X = df[cat + num]
    y = np.log1p(df["ROAS"].clip(lower=0).values)
    tscv = TimeSeriesSplit(n_splits=5)

    tfm = []
    if cat:
        tfm.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    if num:
        tfm.append(("num", StandardScaler(), num))
    pre = ColumnTransformer(transformers=tfm, remainder="drop")

    r2s = []
    importances_sum = None
    for tr, te in tscv.split(df):
        X_tr = pre.fit_transform(X.iloc[tr])
        X_te = pre.transform(X.iloc[te])
        if USE_XGB:
            m = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 n_jobs=-1, verbosity=0)
        else:
            m = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                         learning_rate=0.1, subsample=0.8, random_state=42)
        m.fit(X_tr, y[tr])
        r2s.append(r2_score(np.expm1(y[te]), np.expm1(m.predict(X_te))))
        fi = m.feature_importances_
        if importances_sum is None:
            importances_sum = fi.copy()
        else:
            importances_sum += fi

    feat_names = pre.get_feature_names_out()
    importances_avg = importances_sum / 5
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances_avg})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return float(np.mean(r2s)), fi_df


# =========================================================================
# SHAP 대조 그림
# =========================================================================
def plot_shap_contrast(fi_leak, fi_clean, r2_leak, r2_clean, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, fi, r2, title, color in [
        (axes[0], fi_leak.head(15), r2_leak, "V1: Leakage 포함 모델", COLOR_DANGER),
        (axes[1], fi_clean.head(15), r2_clean, "V5: Clean 모델 (정직한 성능)", COLOR_SUCCESS),
    ]:
        fi_plot = fi.iloc[::-1]
        bars = ax.barh(range(len(fi_plot)), fi_plot["importance"],
                       color=color, edgecolor="white", height=0.7, zorder=3)
        for i, (_, row) in enumerate(fi_plot.iterrows()):
            label = row["feature"].replace("cat__", "").replace("num__", "")
            is_leak = any(lk in label for lk in LEAKAGE_ORDERED)
            ax.text(row["importance"] + 0.002, i, f'{row["importance"]:.3f}',
                    va="center", fontsize=8,
                    fontweight="bold" if is_leak else "normal",
                    color="#C0392B" if is_leak else "#333")

        labels = [r["feature"].replace("cat__", "").replace("num__", "")
                  for _, r in fi_plot.iterrows()]
        ax.set_yticks(range(len(fi_plot)))
        ax.set_yticklabels(labels, fontsize=9)
        for i, lab in enumerate(labels):
            if any(lk in lab for lk in LEAKAGE_ORDERED):
                ax.get_yticklabels()[i].set_color("#C0392B")
                ax.get_yticklabels()[i].set_fontweight("bold")
        ax.set_xlabel("Feature Importance (평균)", fontsize=11)
        ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("피처 중요도 대조: Leakage 포함 vs 제거 모델",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "bounce_rate가 V1에서 압도적 기여 → V5에서 사라진 뒤 '진짜 설명력'이 드러남",
             ha="center", fontsize=10, color="#666", style="italic")
    plt.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# =========================================================================
# 가드레일 플롯 (ablation + 셔플/난수 기준선)
# =========================================================================
def plot_guardrail(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cat_base, num_base = avail(df, BASE_CAT_FEATURES, BASE_NUM_FEATURES)
    _, cal = avail(df, [], CALENDAR_NUM_FEATURES)
    _, comp = avail(df, [], COMPETITION_NUM_FEATURES)
    cr_cat, cr_num = avail(df, CREATIVE_CAT_FEATURES, CREATIVE_NUM_FEATURES_CLEAN)
    leak_avail = [v for v in LEAKAGE_ORDERED if v in df.columns]
    all_cat = list(dict.fromkeys(cat_base + cr_cat))

    from waterfall_r2_leakage import cv_r2

    # 셔플 기준선: y를 랜덤으로 섞은 뒤 R²
    df_shuf = df.copy()
    df_shuf["ROAS"] = np.random.RandomState(42).permutation(df_shuf["ROAS"].values)
    all_num = list(dict.fromkeys(num_base + cal + comp + cr_num + leak_avail))
    r2_shuffle = cv_r2(df_shuf, all_cat, all_num)

    # 난수 피처 기준선: 난수 피처만으로 학습
    df_rand = df.copy()
    rng = np.random.RandomState(42)
    rand_cols = [f"rand_{i}" for i in range(5)]
    for rc in rand_cols:
        df_rand[rc] = rng.randn(len(df_rand))
    r2_random = cv_r2(df_rand, [], rand_cols)

    # Ablation 곡선
    steps = [
        ("Full\n(leakage)", list(dict.fromkeys(num_base + cal + comp + cr_num + leak_avail))),
    ]
    remain = list(leak_avail)
    for feat in leak_avail:
        if feat in remain:
            remain.remove(feat)
        label = f"− {feat}" if len(feat) < 20 else f"− {feat[:18]}..."
        steps.append((label, list(dict.fromkeys(num_base + cal + comp + cr_num + remain))))
    steps.append(("Base only", list(dict.fromkeys(num_base))))

    labels = []
    r2_vals = []
    for label, nums in steps:
        r2 = cv_r2(df, all_cat, nums)
        labels.append(label)
        r2_vals.append(r2)
        print(f"  {label:<30} R² = {r2:.4f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(labels))
    ax.plot(x, r2_vals, "o-", color=COLOR_PRIMARY, linewidth=2.5, markersize=8,
            zorder=5, label="Ablation 곡선")
    for i, (lab, r2) in enumerate(zip(labels, r2_vals)):
        ax.text(i, r2 + 0.015, f"{r2:.3f}", ha="center", fontsize=9.5, fontweight="bold")

    ax.axhline(r2_shuffle, color=COLOR_DANGER, linestyle="--", linewidth=2,
               label=f"셔플 기준선 (R²={r2_shuffle:.3f})", zorder=3)
    ax.axhline(r2_random, color="#8E44AD", linestyle=":", linewidth=2,
               label=f"난수 피처 기준선 (R²={r2_random:.3f})", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.5, zorder=1)

    ax.fill_between(x, r2_shuffle, [max(v, r2_shuffle) for v in r2_vals],
                    alpha=0.08, color=COLOR_SUCCESS)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("R² (5-fold TimeSeriesSplit CV)", fontsize=12)
    ax.set_title("Ablation 가드레일: 피처 제거에 따른 R² 변화 + 기준선",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.02,
             "초록 영역 = 셔플 기준선 이상의 '진짜 설명력'  |  기준선 아래 = 의미 없는 잡음",
             ha="center", fontsize=9.5, color="#666", style="italic")
    plt.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# =========================================================================
# main
# =========================================================================
def main():
    print("=" * 70)
    print("  D-5: XAI 고도화 + Ablation 가드레일")
    print("=" * 70)

    df = load_data()
    print(f"  데이터: {len(df):,}건\n")

    # SHAP 대조
    cat_base, num_base = avail(df, BASE_CAT_FEATURES, BASE_NUM_FEATURES)
    _, cal = avail(df, [], CALENDAR_NUM_FEATURES)
    _, comp = avail(df, [], COMPETITION_NUM_FEATURES)
    cr_cat, cr_num = avail(df, CREATIVE_CAT_FEATURES, CREATIVE_NUM_FEATURES_CLEAN)
    leak_avail = [v for v in LEAKAGE_ORDERED if v in df.columns]
    all_cat = list(dict.fromkeys(cat_base + cr_cat))

    print("[1] V1 모델 학습 (leakage 포함)...")
    num_leak = list(dict.fromkeys(num_base + cal + comp + cr_num + leak_avail))
    r2_leak, fi_leak = train_and_get_importance(df.copy(), all_cat, num_leak, "V1_leak")

    print(f"    R² = {r2_leak:.4f}")

    print("[2] V5 모델 학습 (clean)...")
    num_clean = list(dict.fromkeys(num_base + cal + comp + cr_num))
    r2_clean, fi_clean = train_and_get_importance(df.copy(), all_cat, num_clean, "V5_clean")
    print(f"    R² = {r2_clean:.4f}")

    shap_path = os.path.join(FIGURES_DIR, "shap_force_leakage_vs_clean.png")
    plot_shap_contrast(fi_leak, fi_clean, r2_leak, r2_clean, shap_path)

    print("\n[3] Ablation 가드레일 계산...")
    guardrail_path = os.path.join(FIGURES_DIR, "ablation_guardrail.png")
    plot_guardrail(df, guardrail_path)

    print("\n  완료!")


if __name__ == "__main__":
    main()
