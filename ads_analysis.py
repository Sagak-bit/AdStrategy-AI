# -*- coding: utf-8 -*-
"""
글로벌 광고 성과 데이터 분석
Top 4 분석:
1. 플랫폼 × 산업 × 국가 교차분석
2. 고성과 + 저성과 캠페인 비교분석
3. 시계열 트렌드 (산업별 분리)
4. 광고비 규모별 효율성
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    AD_SPEND_BINS,
    CHART_DPI,
    ENRICHED_DATA_PATH,
    BASE_DATA_PATH,
    FIGURES_DIR,
)
from utils import configure_windows_encoding, load_ads_data

logger = logging.getLogger(__name__)

configure_windows_encoding()
warnings.filterwarnings("ignore", category=FutureWarning)

# 한글 폰트 설정 (크로스플랫폼)
import platform as _pf

_os_name = _pf.system()
if _os_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif _os_name == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 데이터 로드 / 검증
# ============================================================================

def _load_and_validate() -> pd.DataFrame:
    """분석용 데이터를 로드하고 필수 컬럼을 검증한다."""
    df = load_ads_data()
    if df is None:
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {ENRICHED_DATA_PATH} 또는 {BASE_DATA_PATH}"
        )

    required = ["platform", "industry", "country", "campaign_type", "ROAS", "CPC", "ad_spend"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    return df


# ============================================================================
# 1. 플랫폼 × 산업 × 국가 교차분석
# ============================================================================

def analyse_cross(df: pd.DataFrame) -> dict[str, Any]:
    """플랫폼·산업·국가 교차분석 수행."""
    agg_cols = {"CTR": "mean", "CPC": "mean", "CPA": "mean", "ROAS": "mean",
                "ad_spend": "sum", "revenue": "sum"}

    platform_stats = df.groupby("platform").agg({**agg_cols, "conversions": "sum"}).round(2)
    platform_stats["Overall_ROAS"] = (platform_stats["revenue"] / platform_stats["ad_spend"]).round(2)

    industry_stats = df.groupby("industry").agg(agg_cols).round(2)
    industry_stats["Overall_ROAS"] = (industry_stats["revenue"] / industry_stats["ad_spend"]).round(2)

    country_stats = df.groupby("country").agg(agg_cols).round(2)
    country_stats["Overall_ROAS"] = (country_stats["revenue"] / country_stats["ad_spend"]).round(2)

    platform_industry = df.pivot_table(values="ROAS", index="industry", columns="platform", aggfunc="mean").round(2)
    platform_country = df.pivot_table(values="ROAS", index="country", columns="platform", aggfunc="mean").round(2)

    combo_stats = df.groupby(["platform", "industry", "country"]).agg(
        {"ROAS": "mean", "ad_spend": "sum", "revenue": "sum", "conversions": "sum"}
    ).reset_index()
    combo_stats["sample_count"] = df.groupby(["platform", "industry", "country"]).size().values
    combo_filtered = combo_stats[combo_stats["sample_count"] >= 5].sort_values("ROAS", ascending=False)

    return {
        "platform_stats": platform_stats,
        "industry_stats": industry_stats,
        "country_stats": country_stats,
        "platform_industry": platform_industry,
        "platform_country": platform_country,
        "combo_top10": combo_filtered.head(10),
        "combo_bottom10": combo_filtered.tail(10),
    }


# ============================================================================
# 2. 고성과 / 저성과 비교
# ============================================================================

def analyse_performers(df: pd.DataFrame) -> dict[str, Any]:
    """ROAS 상위/하위 10% 비교분석."""
    top_thresh = df["ROAS"].quantile(0.90)
    bot_thresh = df["ROAS"].quantile(0.10)
    top = df[df["ROAS"] >= top_thresh]
    bot = df[df["ROAS"] <= bot_thresh]

    compare_cols = ["CTR", "CPC", "CPA", "ROAS", "ad_spend", "revenue"]
    comparison = pd.DataFrame({
        "Top 10%": top[compare_cols].mean(),
        "Bottom 10%": bot[compare_cols].mean(),
        "Overall": df[compare_cols].mean(),
    }).round(2)

    def _dist(subset: pd.DataFrame, col: str) -> pd.Series:
        return (subset[col].value_counts(normalize=True) * 100).round(1)

    return {
        "top_threshold": top_thresh,
        "bot_threshold": bot_thresh,
        "comparison": comparison,
        "top_platform": _dist(top, "platform"),
        "bot_platform": _dist(bot, "platform"),
        "top_industry": _dist(top, "industry"),
        "bot_industry": _dist(bot, "industry"),
        "top_country": _dist(top, "country"),
        "bot_country": _dist(bot, "country"),
        "top_campaign": _dist(top, "campaign_type"),
        "bot_campaign": _dist(bot, "campaign_type"),
    }


# ============================================================================
# 3. 시계열 트렌드
# ============================================================================

def analyse_trends(df: pd.DataFrame) -> dict[str, Any]:
    """월별/분기별 트렌드 분석."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    agg_base = {"CTR": "mean", "CPC": "mean", "ROAS": "mean",
                "ad_spend": "sum", "revenue": "sum"}

    monthly = df.groupby("month").agg({**agg_base, "conversions": "sum"}).round(2)
    monthly["Overall_ROAS"] = (monthly["revenue"] / monthly["ad_spend"]).round(2)

    quarterly = df.groupby("quarter").agg(agg_base).round(2)
    quarterly["Overall_ROAS"] = (quarterly["revenue"] / quarterly["ad_spend"]).round(2)

    industry_monthly_roas = df.pivot_table(values="ROAS", index="month", columns="industry", aggfunc="mean").round(2)
    platform_monthly_roas = df.pivot_table(values="ROAS", index="month", columns="platform", aggfunc="mean").round(2)

    return {
        "monthly": monthly,
        "quarterly": quarterly,
        "industry_monthly_roas": industry_monthly_roas,
        "platform_monthly_roas": platform_monthly_roas,
    }


# ============================================================================
# 4. 광고비 규모별 효율성
# ============================================================================

def analyse_spend_efficiency(df: pd.DataFrame) -> dict[str, Any]:
    """광고비 구간별 효율성 분석."""
    df = df.copy()
    labels = ["~$1K", "$1K~3K", "$3K~5K", "$5K~10K", "$10K~20K", "$20K+"]
    df["ad_spend_bin"] = pd.cut(df["ad_spend"], bins=AD_SPEND_BINS, labels=labels)

    spend_stats = df.groupby("ad_spend_bin", observed=False).agg(
        {"CTR": "mean", "CPC": "mean", "CPA": "mean", "ROAS": "mean",
         "conversions": "mean", "revenue": "mean"}
    ).round(2)
    spend_stats["count"] = df.groupby("ad_spend_bin", observed=False).size().values

    platform_spend = df.pivot_table(values="ROAS", index="ad_spend_bin", columns="platform", aggfunc="mean",
                                    observed=False).round(2)

    corr = df[["ad_spend", "ROAS", "CTR", "CPC", "CPA", "conversions", "revenue"]].corr()["ad_spend"].drop(
        "ad_spend").round(3)

    return {
        "spend_stats": spend_stats,
        "platform_spend_roas": platform_spend,
        "ad_spend_correlation": corr,
    }


# ============================================================================
# 5. 캠페인 유형별 분석
# ============================================================================

def analyse_campaign_types(df: pd.DataFrame) -> dict[str, Any]:
    """캠페인 유형별 성과 분석."""
    campaign_stats = df.groupby("campaign_type").agg(
        {"CTR": "mean", "CPC": "mean", "CPA": "mean", "ROAS": "mean",
         "ad_spend": "sum", "revenue": "sum"}
    ).round(2)
    campaign_stats["Overall_ROAS"] = (campaign_stats["revenue"] / campaign_stats["ad_spend"]).round(2)

    industry_campaign = df.pivot_table(values="ROAS", index="industry", columns="campaign_type",
                                       aggfunc="mean").round(2)
    return {
        "campaign_stats": campaign_stats,
        "industry_campaign": industry_campaign,
    }


# ============================================================================
# 인사이트 요약
# ============================================================================

def summarise_insights(cross: dict, trends: dict) -> dict[str, str]:
    ps = cross["platform_stats"]
    ist = cross["industry_stats"]
    cs = cross["country_stats"]
    ms = trends["monthly"]
    return {
        "best_platform": f"{ps['Overall_ROAS'].idxmax()} (ROAS: {ps['Overall_ROAS'].max():.2f})",
        "best_industry": f"{ist['Overall_ROAS'].idxmax()} (ROAS: {ist['Overall_ROAS'].max():.2f})",
        "best_country": f"{cs['Overall_ROAS'].idxmax()} (ROAS: {cs['Overall_ROAS'].max():.2f})",
        "best_month": f"{ms['Overall_ROAS'].idxmax()}월 (ROAS: {ms['Overall_ROAS'].max():.2f})",
    }


# ============================================================================
# 시각화
# ============================================================================

def save_figures(
    df: pd.DataFrame,
    cross: dict,
    performers: dict,
    trends: dict,
    output_dir: str = FIGURES_DIR,
) -> None:
    """분석 결과 시각화를 PNG 파일로 저장."""
    os.makedirs(output_dir, exist_ok=True)
    dpi = CHART_DPI

    # --- Figure 1: 전체 성과 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, color, title) in zip(
        axes.flat,
        [
            ("platform", "#4A90D9", "Platform Average ROAS"),
            ("industry", "steelblue", "Industry Average ROAS"),
            ("country", "seagreen", "Country Average ROAS"),
            ("campaign_type", "coral", "Campaign Type Average ROAS"),
        ],
    ):
        data = df.groupby(col)["ROAS"].mean().sort_values()
        ax.barh(data.index, data.values, color=color)
        ax.set_xlabel("Average ROAS")
        ax.set_title(title)
        for i, v in enumerate(data.values):
            ax.text(v + 0.1, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_overall_performance.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # --- Figure 2: 교차분석 히트맵 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (pivot, title) in zip(axes, [
        (cross["platform_industry"], "Platform x Industry Avg ROAS"),
        (cross["platform_country"], "Platform x Country Avg ROAS"),
    ]):
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax, center=pivot.values.mean())
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_cross_analysis_heatmap.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # --- Figure 3: 월별 트렌드 ---
    monthly = trends["monthly"]
    ind_roas = trends["industry_monthly_roas"]
    plat_roas = trends["platform_monthly_roas"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(monthly.index, monthly["ROAS"], marker="o", linewidth=2, color="steelblue")
    axes[0, 0].set(xlabel="Month", ylabel="Average ROAS", title="Monthly Average ROAS Trend")
    axes[0, 0].set_xticks(range(1, 13))
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(monthly.index, monthly["CPC"], marker="s", linewidth=2, color="coral")
    axes[0, 1].set(xlabel="Month", ylabel="Average CPC ($)", title="Monthly Average CPC Trend")
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].grid(True, alpha=0.3)

    for industry in ind_roas.columns:
        axes[1, 0].plot(ind_roas.index, ind_roas[industry], marker="o", label=industry)
    axes[1, 0].set(xlabel="Month", ylabel="Average ROAS", title="Industry Monthly ROAS Trend")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    plat_colors = {"Google Ads": "#4285F4", "Meta Ads": "#1877F2", "TikTok Ads": "#000000"}
    for plat in plat_roas.columns:
        axes[1, 1].plot(plat_roas.index, plat_roas[plat], marker="o", label=plat,
                        color=plat_colors.get(plat, "gray"))
    axes[1, 1].set(xlabel="Month", ylabel="Average ROAS", title="Platform Monthly ROAS Trend")
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_monthly_trends.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # --- Figure 4: 광고비 효율성 ---
    labels = ["~$1K", "$1K~3K", "$3K~5K", "$5K~10K", "$10K~20K", "$20K+"]
    df_c = df.copy()
    df_c["ad_spend_bin"] = pd.cut(df_c["ad_spend"], bins=AD_SPEND_BINS, labels=labels)
    spend_roas = df_c.groupby("ad_spend_bin", observed=False)["ROAS"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(spend_roas)), spend_roas.values, color="steelblue")
    axes[0].set_xticks(range(len(spend_roas)))
    axes[0].set_xticklabels(spend_roas.index, rotation=45)
    axes[0].set(xlabel="Ad Spend Range", ylabel="Average ROAS", title="Average ROAS by Ad Spend Range")
    for i, v in enumerate(spend_roas.values):
        axes[0].text(i, v + 0.2, f"{v:.1f}", ha="center")

    axes[1].scatter(df["ad_spend"], df["ROAS"], alpha=0.3, s=10)
    axes[1].set(xlabel="Ad Spend ($)", ylabel="ROAS", title="Ad Spend vs ROAS Scatter")
    axes[1].set_xlim(0, df["ad_spend"].quantile(0.95))
    axes[1].set_ylim(0, df["ROAS"].quantile(0.95))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_spend_efficiency.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    # --- Figure 5: 고성과/저성과 비교 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    width = 0.35
    for ax, (top_key, bot_key, title) in zip(axes.flat, [
        ("top_platform", "bot_platform", "Platform Distribution: Top 10% vs Bottom 10%"),
        ("top_industry", "bot_industry", "Industry Distribution: Top 10% vs Bottom 10%"),
        ("top_country", "bot_country", "Country Distribution: Top 10% vs Bottom 10%"),
        ("top_campaign", "bot_campaign", "Campaign Type Distribution: Top 10% vs Bottom 10%"),
    ]):
        top_data = performers[top_key]
        bot_data = performers[bot_key]
        x = np.arange(len(top_data))
        ax.bar(x - width / 2, top_data.values, width, label="Top 10%", color="green", alpha=0.7)
        ax.bar(x + width / 2, bot_data.reindex(top_data.index).fillna(0).values, width,
               label="Bottom 10%", color="red", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(top_data.index, rotation=45)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_top_bottom_comparison.png"), dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("시각화 저장 완료: %s", output_dir)


# ============================================================================
# 통합 실행
# ============================================================================

def run_analysis() -> dict[str, Any]:
    """전체 분석을 실행하고 결과를 반환한다."""
    df = _load_and_validate()
    logger.info("분석 시작: %d건", len(df))

    cross = analyse_cross(df)
    performers_data = analyse_performers(df)
    trends_data = analyse_trends(df)
    spend = analyse_spend_efficiency(df)
    campaigns = analyse_campaign_types(df)
    insights = summarise_insights(cross, trends_data)

    save_figures(df, cross, performers_data, trends_data)

    return {
        "cross": cross,
        "performers": performers_data,
        "trends": trends_data,
        "spend": spend,
        "campaigns": campaigns,
        "insights": insights,
    }


# ============================================================================
# 콘솔 리포트 (독립 실행 시에만)
# ============================================================================

def print_report(results: dict[str, Any]) -> None:
    """분석 결과를 콘솔에 출력."""
    cross = results["cross"]
    perf = results["performers"]
    trends = results["trends"]
    spend = results["spend"]
    campaigns = results["campaigns"]
    insights = results["insights"]

    print("=" * 80)
    print("[1] Platform x Industry x Country Cross Analysis")
    print("=" * 80)
    print(cross["platform_stats"])
    print("\n[Best Platform by Industry (ROAS)]:")
    pi = cross["platform_industry"]
    for ind in pi.index:
        print(f"  - {ind}: {pi.loc[ind].idxmax()} (ROAS: {pi.loc[ind].max():.2f})")

    print("\n" + "=" * 80)
    print("[2] Top vs Bottom Performers")
    print("=" * 80)
    print(perf["comparison"])

    print("\n" + "=" * 80)
    print("[3] Monthly Trends")
    print("=" * 80)
    print(trends["monthly"])

    print("\n" + "=" * 80)
    print("[4] Ad Spend Efficiency")
    print("=" * 80)
    print(spend["spend_stats"])

    print("\n" + "=" * 80)
    print("[5] Campaign Type Analysis")
    print("=" * 80)
    print(campaigns["campaign_stats"])

    print("\n" + "=" * 80)
    print("[SUMMARY] Key Insights")
    print("=" * 80)
    for k, v in insights.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    results = run_analysis()
    print_report(results)
