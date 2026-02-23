# -*- coding: utf-8 -*-
"""
전략 시뮬레이션: 예산 재할당 효과 분석
========================================
예측 ROAS 상위 k% 캠페인에 예산을 가중 배분했을 때 vs 균등 배분 대비
기대 ROAS 개선율을 시계열 holdout으로 검증.

산출물:
  - figures/budget_reallocation_impact.png
  - figures/segment_confidence_heatmap.png
  - figures/platform_budget_response_curves.png
"""
from __future__ import annotations

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ENRICHED_DATA_PATH, FIGURES_DIR, CHART_DPI,
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_DANGER, COLOR_WARNING,
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_data():
    df = pd.read_csv(ENRICHED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ROAS"]).sort_values("date").reset_index(drop=True)
    df["_synth"] = df["is_synthetic"].round(0).astype(int) if "is_synthetic" in df.columns else 0
    return df


def prepare_features(df):
    features = ["ad_spend", "CPC", "CTR", "impressions", "clicks"]
    cat_cols = ["platform", "industry", "country", "campaign_type"]
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
        features.append(c + "_enc")
    features = [f for f in features if f in df.columns]
    return features, encoders


# =========================================================================
# 1. 예산 재할당 시뮬레이션
# =========================================================================
def budget_reallocation_simulation(df, k_pcts=(10, 20, 30, 50)):
    """
    Holdout(마지막 20%) 기반 예산 재할당 시뮬레이션.
    상위 k% 예측 ROAS 캠페인에 예산 집중 vs 균등 배분.
    """
    df = df.copy()
    features, _ = prepare_features(df)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    X_train = train[features].fillna(0)
    y_train = np.log1p(train["ROAS"].clip(lower=0))
    X_test = test[features].fillna(0)

    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    test["pred_roas"] = np.expm1(gb.predict(X_test))

    total_budget = test["ad_spend"].sum()
    if total_budget <= 0 or len(test) == 0:
        return pd.DataFrame([{"strategy": "데이터 부족", "weighted_ROAS": 0,
                              "improvement_pct": 0, "k_pct": 0}])
    actual_weighted_roas = (test["ROAS"] * test["ad_spend"]).sum() / total_budget

    results = [{"strategy": "균등 배분 (현재)", "weighted_ROAS": round(actual_weighted_roas, 2),
                "improvement_pct": 0.0, "k_pct": 0}]

    for k in k_pcts:
        threshold = test["pred_roas"].quantile(1 - k / 100)
        top_mask = test["pred_roas"] >= threshold
        top = test[top_mask]
        bottom = test[~top_mask]

        top_weight = 0.7
        bottom_weight = 0.3

        top_budget = total_budget * top_weight
        bottom_budget = total_budget * bottom_weight

        if len(top) > 0 and len(bottom) > 0:
            top_spend_per = top_budget / len(top)
            bottom_spend_per = bottom_budget / len(bottom)
            sim_roas = (
                (top["ROAS"] * top_spend_per).sum() +
                (bottom["ROAS"] * bottom_spend_per).sum()
            ) / total_budget
        else:
            sim_roas = actual_weighted_roas

        improvement = (sim_roas - actual_weighted_roas) / max(abs(actual_weighted_roas), 0.01) * 100
        results.append({
            "strategy": f"상위 {k}%에 70% 집중",
            "weighted_ROAS": round(sim_roas, 2),
            "improvement_pct": round(improvement, 1),
            "k_pct": k,
        })

    return pd.DataFrame(results)


def plot_reallocation(sim_df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = sim_df["strategy"]
    roas_vals = sim_df["weighted_ROAS"]
    colors = [COLOR_PRIMARY if i == 0 else COLOR_SUCCESS for i in range(len(sim_df))]

    bars = ax.bar(range(len(labels)), roas_vals, color=colors, edgecolor="white",
                  width=0.6, zorder=3)

    for i, (v, imp) in enumerate(zip(roas_vals, sim_df["improvement_pct"])):
        label = f"{v:.1f}"
        if imp != 0:
            label += f"\n({imp:+.1f}%)"
        ax.text(i, v + max(abs(roas_vals.max()), 1) * 0.02, label,
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("가중 평균 ROAS", fontsize=12)
    ax.set_title("예산 재할당 시뮬레이션: 상위 예측 ROAS에 집중 배분 효과",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.02,
             "Holdout(마지막 20%) 기준  |  '상위 k%에 70% 예산 집중' vs '현재 균등 배분'",
             ha="center", fontsize=9.5, color="#666", style="italic")
    plt.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# =========================================================================
# 2. 세그먼트 신뢰도 맵
# =========================================================================
def segment_confidence_map(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = df.copy()
    features, _ = prepare_features(df)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    X_train = train[features].fillna(0)
    y_train = np.log1p(train["ROAS"].clip(lower=0))
    X_test = test[features].fillna(0)

    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    test["pred"] = np.expm1(gb.predict(X_test))

    rows = []
    for plat in test["platform"].unique():
        for ind in test["industry"].unique():
            seg = test[(test["platform"] == plat) & (test["industry"] == ind)]
            if len(seg) >= 5:
                mae = mean_absolute_error(seg["ROAS"], seg["pred"])
                rmse = np.sqrt(mean_squared_error(seg["ROAS"], seg["pred"]))
                rows.append({"platform": plat, "industry": ind,
                             "MAE": round(mae, 2), "RMSE": round(rmse, 2),
                             "n": len(seg)})

    seg_df = pd.DataFrame(rows)
    if seg_df.empty:
        print("  [WARN] 세그먼트 데이터 부족")
        return

    pivot = seg_df.pivot(index="industry", columns="platform", values="MAE")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax,
                linewidths=1, linecolor="white", cbar_kws={"label": "MAE (낮을수록 신뢰도 높음)"})
    ax.set_title("세그먼트 신뢰도 맵: 플랫폼 × 산업별 예측 MAE",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# =========================================================================
# 3. 플랫폼별 예산-ROAS 반응 곡선
# =========================================================================
from scripts.platform_policy_params import PLATFORM_POLICY


def platform_response_curves(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    budgets = np.arange(100, 20001, 100)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {"Google Ads": COLOR_PRIMARY, "Meta Ads": COLOR_WARNING, "TikTok Ads": COLOR_DANGER}

    for plat, policy in PLATFORM_POLICY.items():
        plat_df = df[df["platform"] == plat]
        if len(plat_df) < 10:
            continue
        base_roas = plat_df["ROAS"].clip(lower=0).median()
        if pd.isna(base_roas) or base_roas <= 0:
            base_roas = 3.0

        roas_curve = []
        for b in budgets:
            if policy["curve"] == "sigmoid":
                x = (b - policy["min_effective_budget"]) / (policy["saturation_budget"] - policy["min_effective_budget"])
                x = np.clip(x, -2, 5)
                mult = policy["max_multiplier"] / (1 + np.exp(-5 * (x - 0.5)))
            elif policy["curve"] == "log_penalty":
                if b < policy["min_effective_budget"]:
                    mult = policy["penalty_below_min"] * (b / policy["min_effective_budget"])
                else:
                    mult = policy["max_multiplier"] * np.log1p(b / 1000) / np.log1p(20)
            else:
                mult = policy["max_multiplier"] * np.log1p(b / 500) / np.log1p(40)
                mult *= (1 + np.random.RandomState(int(b)).normal(0, policy["volatility"]))

            roas_curve.append(max(0, base_roas * mult))

        ax.plot(budgets, roas_curve, linewidth=2.5, color=colors[plat], label=plat, zorder=3)

        min_b = policy["min_effective_budget"]
        ax.axvline(min_b, color=colors[plat], linestyle=":", alpha=0.5, linewidth=1)
        ax.text(min_b + 50, max(roas_curve) * 0.95, f"최소\n${min_b}",
                fontsize=7.5, color=colors[plat], va="top")

    ax.set_xlabel("월 광고 예산 (USD)", fontsize=12)
    ax.set_ylabel("기대 ROAS (시뮬레이션)", fontsize=12)
    ax.set_title("플랫폼별 예산-ROAS 반응 곡선 (정책 시뮬레이션)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(0.5, -0.02,
             "Google: S형 포화 | Meta: 소예산 페널티 + 학습 구간 | TikTok: 선형 + 경매 변동  "
             "(실무 가정 기반 시뮬레이션, 실측 아님)",
             ha="center", fontsize=9, color="#666", style="italic")
    plt.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# =========================================================================
# main
# =========================================================================
def main():
    print("=" * 70)
    print("  D-4/D-5: 전략 시뮬레이션 + 세그먼트 신뢰도 + 반응 곡선")
    print("=" * 70)

    df = load_data()
    print(f"  데이터: {len(df):,}건\n")

    print("[1] 예산 재할당 시뮬레이션")
    sim_df = budget_reallocation_simulation(df)
    print(sim_df.to_string(index=False))
    plot_reallocation(sim_df, os.path.join(FIGURES_DIR, "budget_reallocation_impact.png"))

    print(f"\n[2] 세그먼트 신뢰도 맵")
    segment_confidence_map(df, os.path.join(FIGURES_DIR, "segment_confidence_heatmap.png"))

    print(f"\n[3] 플랫폼별 반응 곡선")
    platform_response_curves(df, os.path.join(FIGURES_DIR, "platform_budget_response_curves.png"))

    print(f"\n[4] 플랫폼 정책 파라미터:")
    for plat, pol in PLATFORM_POLICY.items():
        print(f"  {plat}: {pol['note']}")

    print("\n  완료!")


if __name__ == "__main__":
    main()
