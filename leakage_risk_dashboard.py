# -*- coding: utf-8 -*-
"""
Leakage Risk Scoreboard
========================
숫자형 피처 각각에 대해 단변량 Ridge → ROAS R² 를 구하고,
비정상적으로 높은 피처(leakage 후보)를 시각적으로 강조하는 대시보드.
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
    CHART_DPI,
    COLOR_DANGER,
    COLOR_INFO,
    COLOR_WARNING,
    LEAKAGE_FEATURES as LEAKAGE_SUSPECTS,
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

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ARITHMETIC_RELATED = {"revenue", "CPA", "conversions", "CTR", "CPC", "clicks", "impressions"}
EXCLUDE = {"ROAS", "is_synthetic"}


def load_data() -> pd.DataFrame | None:
    from utils import load_ads_data
    return load_ads_data()


def compute_univariate_r2(df, target="ROAS"):
    from sklearn.model_selection import KFold

    df = df.loc[df[target].notna() & (df[target] > 0)].copy()

    # 극단적 이상값 제거 (1~99 백분위)
    lo, hi = df[target].quantile(0.01), df[target].quantile(0.99)
    df = df.loc[(df[target] >= lo) & (df[target] <= hi)].copy()
    y_log = np.log1p(df[target].values)
    print(f"  이상값 클리핑 후: {len(df):,}건  (ROAS {lo:.1f}~{hi:.1f})")

    num_cols = [
        c for c in df.select_dtypes("number").columns
        if c not in EXCLUDE
        and df[c].notna().sum() > len(df) * MIN_NON_NULL_RATIO
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for col in num_cols:
        col_mask = df[col].notna()
        X = df.loc[col_mask, [col]].values
        y_sub = y_log[col_mask.values]

        if len(y_sub) < 50:
            continue

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, X_sc, y_sub, cv=kf, scoring="r2")
        r2_mean = float(np.mean(scores))
        r2_std = float(np.std(scores))

        if col in LEAKAGE_SUSPECTS:
            tag = "leakage"
        elif col in ARITHMETIC_RELATED:
            tag = "arithmetic"
        else:
            tag = "normal"

        results.append({
            "feature": col,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "tag": tag,
        })

    return pd.DataFrame(results).sort_values("r2_mean", ascending=True).reset_index(drop=True)


def plot_dashboard(rdf: pd.DataFrame, save_path: str = "figures/leakage_risk_dashboard.png") -> None:
    if rdf.empty:
        print("  [WARN] 시각화할 데이터가 없습니다.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(rdf)
    fig_h = max(8, n * 0.36)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    color_map = {"leakage": COLOR_DANGER, "arithmetic": COLOR_WARNING}
    colors = [color_map.get(row["tag"], COLOR_INFO) for _, row in rdf.iterrows()]

    bars = ax.barh(
        range(n), rdf["r2_mean"], xerr=rdf["r2_std"],
        color=colors, edgecolor="white", linewidth=1.2,
        height=0.72, capsize=2, zorder=3,
    )

    # 값 라벨 — R²가 상위권인 피처만 표시 (상위 10개 + leakage)
    r2_max = rdf["r2_mean"].max()
    top_idx = min(9, n - 1) if n > 0 else 0
    show_threshold = sorted(rdf["r2_mean"], reverse=True)[top_idx] if n > 0 else 0

    for i, (_, row) in enumerate(rdf.iterrows()):
        r2 = row["r2_mean"]
        show = row["tag"] == "leakage" or r2 >= show_threshold
        if not show:
            continue
        offset = max(r2_max * 0.03, 0.003)
        ax.text(
            max(r2, 0) + offset, i, f"{r2:.3f}",
            va="center", ha="left", fontsize=9,
            fontweight="bold" if row["tag"] == "leakage" else "normal",
            color="#C0392B" if row["tag"] == "leakage" else "#333333",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(rdf["feature"], fontsize=9.5)

    # leakage 피처 y-label 강조
    for i, (_, row) in enumerate(rdf.iterrows()):
        if row["tag"] == "leakage":
            ax.get_yticklabels()[i].set_color("#C0392B")
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("단변량 R² (Ridge → log₁₊ROAS, 5-fold CV)", fontsize=12)
    ax.set_title(
        "누수 위험 점수보드: 피처별 단변량 R² → ROAS",
        fontsize=16, fontweight="bold", pad=18,
    )

    ax.axvline(0, color="black", linewidth=0.6)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 위험 임계선
    threshold = 0.15
    leakage_r2 = rdf.loc[rdf["tag"] == "leakage", "r2_mean"]
    if len(leakage_r2) > 0:
        threshold = max(0.10, leakage_r2.min() * 0.6)

    ax.axvline(threshold, color="#E74C3C", linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
    ax.text(threshold + 0.005, n - 0.5,
            f"위험 임계 (R²>{threshold:.2f})",
            fontsize=9, color="#E74C3C", va="top", style="italic")

    legend_handles = [
        mpatches.Patch(color="#E74C3C", label="Leakage 후보 (target에서 역산된 변수)"),
        mpatches.Patch(color="#F39C12", label="산술적 관련 (revenue, CPA 등 ROAS 구성요소)"),
        mpatches.Patch(color="#5DADE2", label="일반 피처"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.5,
              framealpha=0.9, edgecolor="#cccccc")

    subtitle = (
        "각 숫자형 피처 하나만으로 Ridge 회귀 → log(1+ROAS) 예측 시 R²  |  "
        "비정상적으로 높은 R²는 target leakage 의심"
    )
    fig.text(0.5, -0.005, subtitle, ha="center", fontsize=9.5, color="#666666", style="italic")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path} 저장 완료")


if __name__ == "__main__":
    print("=" * 70)
    print("  Leakage Risk Scoreboard: 피처별 단변량 R²")
    print("=" * 70)

    df = load_data()
    if df is None:
        print("  [ERROR] 데이터 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"  데이터: {df.shape[0]:,}행 × {df.shape[1]}컬럼\n")

    rdf = compute_univariate_r2(df)

    # 콘솔 출력
    print(f"  {'Feature':<30} {'R² mean':>10} {'R² std':>10} {'Tag':>12}")
    print("  " + "-" * 64)
    for _, row in rdf.iloc[::-1].iterrows():
        marker = "★" if row["tag"] == "leakage" else " "
        print(f"  {marker} {row['feature']:<28} {row['r2_mean']:>10.4f} {row['r2_std']:>10.4f} {row['tag']:>12}")

    # CSV 저장
    csv_path = "figures/leakage_risk_scores.csv"
    rdf.sort_values("r2_mean", ascending=False).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] {csv_path} 저장 완료")

    # 시각화
    plot_dashboard(rdf)
