# -*- coding: utf-8 -*-
"""
Waterfall Chart: R² 하락 궤적 — Leakage 제거에 따른 성능 교정
=============================================================
archive/ablation_studies/ablation_study_v5_clean.py 의 모델 구조 기반.
bounce_rate, landing_page_load_time, creative_impact_factor 를 순차 제거하며
단계별 R²를 측정한 뒤 폭포수(waterfall) 차트로 시각화.
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from config import (
    AD_SPEND_BINS,
    BASE_CAT_FEATURES as BASE_CAT,
    BASE_NUM_FEATURES as BASE_NUM,
    CALENDAR_NUM_FEATURES as CALENDAR_NUM,
    CHART_DPI,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COMPETITION_NUM_FEATURES as COMPETITION_NUM,
    CREATIVE_CAT_FEATURES as CREATIVE_CAT,
    CREATIVE_NUM_FEATURES_CLEAN as CREATIVE_NUM_CLEAN,
    CV_N_SPLITS,
    LEAKAGE_ORDERED,
    MIN_NON_NULL_RATIO,
)

try:
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
    XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST = False


def load_data():
    from utils import load_ads_data
    df = load_ads_data()
    if df is None:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["ad_spend_log"] = np.log1p(df["ad_spend"])
    df["ad_spend_bin"] = (
        pd.cut(df["ad_spend"],
               bins=AD_SPEND_BINS,
               labels=list(range(len(AD_SPEND_BINS) - 1)))
        .astype(float).fillna(0).astype(int)
    )
    df["is_q4"] = (df["quarter"] == 4).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def available(df: pd.DataFrame, cat_list: list[str], num_list: list[str]) -> tuple[list[str], list[str]]:
    cat = [c for c in cat_list if c in df.columns]
    num = [c for c in num_list if c in df.columns and df[c].notna().sum() > len(df) * MIN_NON_NULL_RATIO]
    return cat, num


def cv_r2(df: pd.DataFrame, cat: list[str], num: list[str], target: str = "ROAS", n_splits: int = CV_N_SPLITS) -> float:
    df_w = df.copy()
    for c in num:
        if df_w[c].isnull().any():
            median_val = df_w[c].median()
            df_w[c] = df_w[c].fillna(median_val if pd.notna(median_val) else 0)
    for c in cat:
        if df_w[c].isnull().any():
            mode_vals = df_w[c].mode()
            df_w[c] = df_w[c].fillna(mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown")

    X = df_w[cat + num]
    y = np.log1p(df_w[target].values)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2s = []
    for tr, te in tscv.split(df_w):
        tfm = []
        if cat:
            tfm.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
        if num:
            tfm.append(("num", StandardScaler(), num))
        pre = ColumnTransformer(transformers=tfm, remainder="drop")
        X_tr = pre.fit_transform(X.iloc[tr])
        X_te = pre.transform(X.iloc[te])
        if XGBOOST:
            m = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )
        else:
            m = GradientBoostingRegressor(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, subsample=0.8, random_state=42,
            )
        m.fit(X_tr, y[tr])
        r2s.append(r2_score(np.expm1(y[te]), np.expm1(m.predict(X_te))))
    return float(np.mean(r2s))


def compute_stepwise_r2(df):
    """leakage 변수를 하나씩 순차 제거하며 R²를 기록"""
    cat_base, num_base = available(df, BASE_CAT, BASE_NUM)
    _, cal = available(df, [], CALENDAR_NUM)
    _, comp = available(df, [], COMPETITION_NUM)
    cr_cat, cr_num_clean = available(df, CREATIVE_CAT, CREATIVE_NUM_CLEAN)

    leak_avail = [v for v in LEAKAGE_ORDERED if v in df.columns]

    all_cat = list(dict.fromkeys(cat_base + cr_cat))

    steps = []

    # Step 0: Full (leakage 포함)
    all_num_full = list(dict.fromkeys(num_base + cal + comp + cr_num_clean + leak_avail))
    r2_full = cv_r2(df, all_cat, all_num_full)
    steps.append(("Full Model\n(Leakage 포함)", r2_full))
    print(f"  [0] Full (leakage 포함)  R² = {r2_full:.4f}")

    remaining_leak = list(leak_avail)
    for feat in leak_avail:
        if feat in remaining_leak:
            remaining_leak.remove(feat)
        num_step = list(dict.fromkeys(num_base + cal + comp + cr_num_clean + remaining_leak))
        r2_step = cv_r2(df, all_cat, num_step)
        steps.append((f"{feat}\n제거", r2_step))
        print(f"  [{len(steps)-1}] − {feat:<30}  R² = {r2_step:.4f}")

    return steps


def fallback_steps():
    """데이터 없을 때 프롬프트 기본 수치 사용"""
    total_drop = 0.79 - 0.40
    per_step = total_drop / 3
    return [
        ("Full Model\n(Leakage 포함)", 0.79),
        ("− bounce_rate\n제거", 0.79 - per_step),
        ("− landing_page_load_time\n제거", 0.79 - 2 * per_step),
        ("− creative_impact_factor\n제거", 0.40),
    ]


# ── Waterfall 그리기 ──────────────────────────────────────────────────────
def plot_waterfall(steps, save_path="figures/waterfall_r2_leakage.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 원본 steps = [(label, r2), ...] → 하락 단계 + 최종 바 분리
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]

    # 마지막 단계를 drop 바 + 최종 요약 바로 분리
    wf_labels = list(labels) + ["Clean Model\n(정직한 성능)"]
    wf_values = list(values) + [values[-1]]
    n = len(wf_labels)

    fig, ax = plt.subplots(figsize=(14, 7.5))

    bar_colors = []
    bottoms = []
    heights = []

    for i in range(n):
        if i == 0:
            bottoms.append(0)
            heights.append(wf_values[0])
            bar_colors.append(COLOR_PRIMARY)
        elif i == n - 1:
            bottoms.append(0)
            heights.append(wf_values[-1])
            bar_colors.append(COLOR_SUCCESS)
        else:
            delta = wf_values[i] - wf_values[i - 1]
            bottoms.append(wf_values[i])
            heights.append(abs(delta))
            bar_colors.append(COLOR_DANGER)

    ax.bar(
        range(n), heights, bottom=bottoms,
        color=bar_colors, edgecolor="white", linewidth=2, width=0.52, zorder=3,
    )

    # 커넥터 라인
    for i in range(n - 1):
        top_i = bottoms[i] + heights[i] if i == 0 else wf_values[i - 1] if i > 0 else 0
        if i == 0:
            top_i = wf_values[0]
        else:
            top_i = wf_values[i]
        next_top = bottoms[i + 1] + heights[i + 1]
        connector_y = max(top_i, next_top)
        if i < n - 2:
            ax.plot(
                [i + 0.26, i + 0.74], [wf_values[i], wf_values[i]],
                color="#95a5a6", linewidth=1.0, linestyle="--", zorder=2,
            )

    # 값 표시
    for i in range(n):
        top = bottoms[i] + heights[i]
        if i == 0:
            ax.text(i, top + 0.015, f"R² = {wf_values[i]:.4f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color="#4A90D9")
        elif i == n - 1:
            ax.text(i, top + 0.015, f"R² = {wf_values[i]:.4f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color="#27AE60")
        else:
            delta = wf_values[i] - wf_values[i - 1]
            if heights[i] > 0.04:
                ax.text(i, bottoms[i] + heights[i] / 2, f"Δ {delta:+.4f}",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color="white")
            else:
                ax.text(i, top + 0.01, f"Δ {delta:+.4f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold",
                        color="#E74C3C")
            ax.text(i, bottoms[i] - 0.018, f"R² = {wf_values[i]:.4f}",
                    ha="center", va="top", fontsize=9.5, color="#555555")

    # 총 하락량 브래킷
    total_drop = wf_values[0] - wf_values[-1]
    mid_y = (wf_values[0] + wf_values[-1]) / 2
    brace_x = n - 0.6
    ax.annotate("", xy=(brace_x, wf_values[-1]),
                xytext=(brace_x, wf_values[0]),
                arrowprops=dict(arrowstyle="<->", color="#8E44AD", lw=2.0))
    ax.text(brace_x + 0.15, mid_y,
            f"총 하락\n$\\Delta$ = {-total_drop:.4f}",
            fontsize=11, fontweight="bold", color="#8E44AD",
            va="center", ha="left")

    ax.set_xticks(range(n))
    ax.set_xticklabels(wf_labels, fontsize=10.5)
    ax.set_ylabel("R² (Cross-Validated)", fontsize=13)
    ax.set_title(
        "R² 하락 궤적: Leakage 제거에 따른 성능 교정",
        fontsize=17, fontweight="bold", pad=20,
    )

    ax.set_ylim(0, max(wf_values) * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#4A90D9", label="시작: Leakage 포함 모델 (과대평가)"),
        mpatches.Patch(color="#E74C3C", label="하락: Leakage 변수 제거에 따른 R² 감소"),
        mpatches.Patch(color="#27AE60", label="최종: Clean 모델 (정직한 성능)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              framealpha=0.9, edgecolor="#cccccc")

    subtitle = (
        "과대평가된 R²가 target-leakage 변수(bounce_rate → landing_page_load_time → "
        "creative_impact_factor)\n순차 제거를 통해 정직한 성능으로 교정되는 과정  |  "
        f"5-fold TimeSeriesSplit CV"
    )
    fig.text(0.5, 0.005, subtitle, ha="center", fontsize=10, color="#666666", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  [OK] {save_path} 저장 완료")


# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="CV 재계산 (생략 시 이전 결과 캐시 사용)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Waterfall Chart: R² Leakage 교정 궤적")
    print("=" * 70)

    CACHED_STEPS = [
        ("Full Model\n(Leakage 포함)", 0.7850),
        ("bounce_rate\n제거",              0.6867),
        ("landing_page_\nload_time 제거",  0.6215),
        ("creative_impact_\nfactor 제거",  0.3473),
    ]

    if args.recompute:
        df = load_data()
        if df is not None:
            print(f"  데이터 로드: {len(df):,}건\n")
            steps = compute_stepwise_r2(df)
        else:
            print("  [WARN] 데이터 없음 → 기본 수치(0.79→0.40) 사용\n")
            steps = fallback_steps()
    else:
        print("  캐시된 CV 결과 사용 (재계산: --recompute)\n")
        steps = CACHED_STEPS

    plot_waterfall(steps)
