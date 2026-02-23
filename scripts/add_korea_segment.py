# -*- coding: utf-8 -*-
"""
한국 시장 세그먼트 합성
=======================
India 행을 참조하여 한국 시장 특성을 반영한 합성 데이터를 생성합니다.

한국 광고 시장 스케일링 근거:
- 한국 디지털 광고 시장은 GDP 대비 높은 침투율 (eMarketer 2024)
- 평균 CPC가 인도 대비 약 1.3-1.5배 (Google Keyword Planner 기준)
- 모바일 중심 시장으로 TikTok/Meta 비중이 상대적으로 높음

산출물: data/enriched_ads_with_korea.csv
"""
from __future__ import annotations

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENRICHED_DATA_PATH, DATA_DIR

logger = logging.getLogger(__name__)

KOREA_SCALING = {
    "CPC": 1.35,
    "CPA": 1.20,
    "ad_spend": 1.15,
    "impressions": 0.85,
    "clicks": 0.90,
    "conversions": 0.95,
    "revenue": 1.10,
    "ROAS": 0.95,
    "CTR": 1.05,
    "audience_size": 0.70,
    "competition_index": 1.25,
    "auction_density": 1.30,
    "industry_avg_cpc": 1.35,
}

PLATFORM_WEIGHT_KOREA = {
    "Google Ads": 0.30,
    "Meta Ads": 0.35,
    "TikTok Ads": 0.35,
}


def generate_korea_segment(
    df: pd.DataFrame,
    n_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """India 행 기반으로 Korea 합성 행을 생성."""

    india = df[df["country"] == "India"].copy()
    if india.empty:
        logger.warning("India 데이터가 없어 전체 데이터를 참조합니다.")
        india = df.copy()

    if n_rows is None:
        n_rows = min(len(india), 500)

    rng = np.random.RandomState(random_state)

    platform_counts = {
        p: max(1, int(n_rows * w)) for p, w in PLATFORM_WEIGHT_KOREA.items()
    }
    remaining = n_rows - sum(platform_counts.values())
    if remaining > 0:
        platform_counts["TikTok Ads"] += remaining

    rows = []
    for platform, count in platform_counts.items():
        pool = india[india["platform"] == platform]
        if pool.empty:
            pool = india

        sampled = pool.sample(n=count, replace=True, random_state=rng).copy()
        sampled.index = range(len(rows), len(rows) + len(sampled))

        sampled["country"] = "Korea"
        sampled["is_synthetic"] = 1.0

        for col, scale in KOREA_SCALING.items():
            if col in sampled.columns:
                noise = rng.normal(1.0, 0.05, len(sampled))
                sampled[col] = sampled[col] * scale * noise

        if "cpc_vs_industry_avg" in sampled.columns:
            sampled["cpc_vs_industry_avg"] = (
                sampled["CPC"] / sampled["industry_avg_cpc"].clip(lower=0.01)
            )

        rows.append(sampled)

    korea_df = pd.concat(rows, ignore_index=True)

    numeric_cols = korea_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ("is_synthetic", "is_retargeting", "is_lookalike",
                    "is_video", "has_cta", "is_public_holiday",
                    "is_shopping_season", "is_major_event",
                    "is_month_start", "is_month_end"):
            continue
        korea_df[col] = korea_df[col].clip(lower=0)

    return korea_df


def main():
    print("=" * 60)
    print("  한국 시장 세그먼트 합성")
    print("=" * 60)

    df = pd.read_csv(ENRICHED_DATA_PATH)
    print(f"  원본 데이터: {len(df):,}건, 국가: {df['country'].nunique()}개")

    korea = generate_korea_segment(df, n_rows=500)
    print(f"  Korea 합성: {len(korea):,}건")

    combined = pd.concat([df, korea], ignore_index=True)

    out_path = os.path.join(DATA_DIR, "enriched_ads_with_korea.csv")
    combined.to_csv(out_path, index=False)
    print(f"  저장: {out_path} ({len(combined):,}건, 국가: {combined['country'].nunique()}개)")

    print("\n  국가별 분포:")
    for c, cnt in combined["country"].value_counts().items():
        print(f"    {c}: {cnt:,}건")

    print("\n  Korea 요약:")
    for col in ["ROAS", "CPC", "CPA", "ad_spend"]:
        if col in korea.columns:
            print(f"    {col}: mean={korea[col].mean():.2f}, median={korea[col].median():.2f}")

    print("\n  완료!")


if __name__ == "__main__":
    main()
