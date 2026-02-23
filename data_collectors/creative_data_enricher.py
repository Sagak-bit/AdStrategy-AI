# -*- coding: utf-8 -*-
"""
Phase 3: 크리에이티브/타겟팅 메타데이터 보강
- 기존 데이터에 크리에이티브(소재) 메타데이터 시뮬레이션 추가
- 타겟 오디언스 정보 추가
- 랜딩페이지/전환 퍼널 데이터 추가
- 실제 데이터가 없을 경우 플랫폼/산업별 합리적 분포 기반 시뮬레이션

실제 운영 데이터가 있다면 이 모듈 대신 직접 데이터를 사용하면 됨.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ===========================================================================
# 플랫폼/산업별 크리에이티브 분포 프로파일
# ===========================================================================

PLATFORM_AD_FORMAT: dict[str, dict[str, dict[str, float]]] = {
    "Google Ads": {
        "Search": {"text": 0.85, "responsive_search": 0.15},
        "Video": {"video_short": 0.3, "video_long": 0.5, "bumper": 0.2},
        "Shopping": {"product_listing": 0.7, "showcase": 0.3},
        "Display": {"image": 0.5, "responsive_display": 0.3, "html5": 0.2},
    },
    "Meta Ads": {
        "Search": {"image": 0.4, "carousel": 0.3, "video_short": 0.3},
        "Video": {"video_short": 0.5, "video_long": 0.2, "reels": 0.3},
        "Shopping": {"carousel": 0.4, "collection": 0.3, "dynamic_product": 0.3},
        "Display": {"image": 0.5, "carousel": 0.3, "video_short": 0.2},
    },
    "TikTok Ads": {
        "Search": {"video_short": 0.7, "spark_ad": 0.3},
        "Video": {"video_short": 0.6, "video_long": 0.15, "spark_ad": 0.25},
        "Shopping": {"video_short": 0.5, "live_shopping": 0.2, "product_card": 0.3},
        "Display": {"video_short": 0.6, "image": 0.2, "spark_ad": 0.2},
    },
}

VIDEO_LENGTH_MAP: dict[str, tuple[int, int]] = {
    "video_short": (6, 30),
    "video_long": (31, 120),
    "bumper": (5, 6),
    "reels": (15, 60),
    "spark_ad": (10, 60),
    "live_shopping": (60, 300),
}

VIDEO_FORMATS = frozenset(VIDEO_LENGTH_MAP.keys())

CTA_TYPES: dict[str, list[str]] = {
    "Search": ["Learn More", "Sign Up", "Get Quote", "Buy Now", "Contact Us"],
    "Video": ["Learn More", "Watch More", "Sign Up", "Shop Now", "Download"],
    "Shopping": ["Shop Now", "Buy Now", "Add to Cart", "View Deal", "Get Offer"],
    "Display": ["Learn More", "Sign Up", "Shop Now", "Get Started", "Try Free"],
}

HEADLINE_LENGTH: dict[str, tuple[int, int]] = {
    "Fintech": (45, 12), "EdTech": (50, 15), "Healthcare": (40, 10),
    "SaaS": (48, 13), "E-commerce": (35, 10),
}

AGE_GROUP_DIST: dict[str, dict[str, float]] = {
    "Fintech": {"18-24": 0.15, "25-34": 0.35, "35-44": 0.25, "45-54": 0.15, "55+": 0.10},
    "EdTech": {"18-24": 0.35, "25-34": 0.30, "35-44": 0.20, "45-54": 0.10, "55+": 0.05},
    "Healthcare": {"18-24": 0.10, "25-34": 0.20, "35-44": 0.25, "45-54": 0.25, "55+": 0.20},
    "SaaS": {"18-24": 0.10, "25-34": 0.35, "35-44": 0.30, "45-54": 0.20, "55+": 0.05},
    "E-commerce": {"18-24": 0.25, "25-34": 0.30, "35-44": 0.25, "45-54": 0.12, "55+": 0.08},
}

GENDER_DIST: dict[str, dict[str, float]] = {
    "Fintech": {"Male": 0.40, "Female": 0.30, "All": 0.30},
    "EdTech": {"Male": 0.25, "Female": 0.30, "All": 0.45},
    "Healthcare": {"Male": 0.20, "Female": 0.35, "All": 0.45},
    "SaaS": {"Male": 0.40, "Female": 0.25, "All": 0.35},
    "E-commerce": {"Male": 0.25, "Female": 0.35, "All": 0.40},
}

RETARGETING_RATE: dict[str, float] = {
    "Fintech": 0.30, "EdTech": 0.25, "Healthcare": 0.20, "SaaS": 0.40, "E-commerce": 0.45,
}

LOOKALIKE_RATE: dict[str, float] = {
    "Fintech": 0.35, "EdTech": 0.30, "Healthcare": 0.25, "SaaS": 0.40, "E-commerce": 0.50,
}

INTEREST_MAP: dict[str, list[str]] = {
    "Fintech": ["Finance", "Investment", "Banking", "Crypto", "Savings"],
    "EdTech": ["Education", "Career", "Skills", "Certification", "Students"],
    "Healthcare": ["Health", "Fitness", "Wellness", "Medical", "Insurance"],
    "SaaS": ["Business", "Technology", "Productivity", "Enterprise", "Startups"],
    "E-commerce": ["Shopping", "Fashion", "Electronics", "Home", "Lifestyle"],
}

SESSION_DURATION_MAP: dict[str, tuple[float, float]] = {
    "Fintech": (90, 30), "EdTech": (150, 50), "Healthcare": (120, 40),
    "SaaS": (180, 60), "E-commerce": (100, 35),
}

FUNNEL_MAP: dict[str, list[int]] = {
    "Fintech": [2, 3, 4, 5], "EdTech": [1, 2, 3], "Healthcare": [2, 3, 4],
    "SaaS": [2, 3, 4, 5], "E-commerce": [1, 2, 3],
}


class CreativeDataEnricher:
    """크리에이티브/타겟팅/퍼널 메타데이터를 기존 데이터에 추가 (벡터화)."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        logger.info("[Phase 3] Creative Data Enricher initialized (seed=%d)", seed)

    # ------------------------------------------------------------------
    # 벡터화 헬퍼: 그룹별 카테고리 샘플링
    # ------------------------------------------------------------------
    def _sample_categorical(
        self, n: int, dist: dict[str, float],
    ) -> np.ndarray:
        """확률 분포에서 n개 샘플링."""
        keys = list(dist.keys())
        probs = list(dist.values())
        return self.rng.choice(keys, size=n, p=probs)

    def _group_sample_categorical(
        self, groups: pd.Series, dist_map: dict[str, dict[str, float]],
        default_dist: dict[str, float] | None = None,
    ) -> np.ndarray:
        """그룹(예: industry)별로 서로 다른 분포에서 카테고리 샘플링."""
        result = np.empty(len(groups), dtype=object)
        for group_val, idx in groups.groupby(groups).groups.items():
            dist = dist_map.get(group_val, default_dist or {})
            if not dist:
                result[idx] = "Unknown"
                continue
            keys = list(dist.keys())
            probs = list(dist.values())
            result[idx] = self.rng.choice(keys, size=len(idx), p=probs)
        return result

    # ------------------------------------------------------------------
    # 크리에이티브 메타데이터 (벡터화)
    # ------------------------------------------------------------------
    def _generate_creative_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """크리에이티브 메타데이터를 벡터 연산으로 생성."""
        n = len(df)
        platforms = df["platform"].values
        campaign_types = df["campaign_type"].values
        industries = df["industry"].values

        ad_format = np.empty(n, dtype=object)
        for (plat, camp), idx in df.groupby(["platform", "campaign_type"]).groups.items():
            dist = PLATFORM_AD_FORMAT.get(plat, {}).get(camp, {"image": 1.0})
            keys = list(dist.keys())
            probs = list(dist.values())
            ad_format[idx] = self.rng.choice(keys, size=len(idx), p=probs)

        is_video = np.isin(ad_format, list(VIDEO_FORMATS)).astype(int)

        video_length = np.zeros(n, dtype=int)
        for fmt, (lo, hi) in VIDEO_LENGTH_MAP.items():
            mask = ad_format == fmt
            count = mask.sum()
            if count > 0:
                video_length[mask] = self.rng.randint(lo, hi + 1, size=count)

        cta_type = np.empty(n, dtype=object)
        for camp, idx in df.groupby("campaign_type").groups.items():
            opts = CTA_TYPES.get(camp, ["Learn More"])
            cta_type[idx] = self.rng.choice(opts, size=len(idx))

        headline_length = np.empty(n, dtype=int)
        for ind, idx in df.groupby("industry").groups.items():
            mu, sigma = HEADLINE_LENGTH.get(ind, (42, 12))
            headline_length[idx] = np.maximum(10, self.rng.normal(mu, sigma, size=len(idx)).astype(int))

        copy_sentiment = np.clip(self.rng.beta(5, 2, size=n), 0.0, 1.0).round(3)

        num_products = np.zeros(n, dtype=int)
        shop_mask = campaign_types == "Shopping"
        shop_n = shop_mask.sum()
        if shop_n:
            num_products[shop_mask] = self.rng.choice(
                [1, 2, 3, 4, 6, 8], size=shop_n, p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1],
            )
        other_mask = ~shop_mask
        other_n = other_mask.sum()
        if other_n:
            num_products[other_mask] = self.rng.choice([0, 1, 2], size=other_n, p=[0.5, 0.3, 0.2])

        return pd.DataFrame({
            "ad_format": ad_format,
            "is_video": is_video,
            "video_length_sec": video_length,
            "cta_type": cta_type,
            "has_cta": np.ones(n, dtype=int),
            "headline_length": headline_length,
            "copy_sentiment": copy_sentiment,
            "num_products_shown": num_products,
        })

    # ------------------------------------------------------------------
    # 타겟 오디언스 (벡터화)
    # ------------------------------------------------------------------
    def _generate_audience_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        industries = df["industry"]

        target_age_group = self._group_sample_categorical(
            industries, AGE_GROUP_DIST, AGE_GROUP_DIST["E-commerce"],
        )
        target_gender = self._group_sample_categorical(
            industries, GENDER_DIST, GENDER_DIST["E-commerce"],
        )

        base_mult = self.rng.uniform(100, 500, size=n)
        audience_size = np.maximum(1000, (df["ad_spend"].values * base_mult)).astype(int)

        is_retargeting = np.zeros(n, dtype=int)
        for ind, idx in industries.groupby(industries).groups.items():
            rate = RETARGETING_RATE.get(ind, 0.30)
            is_retargeting[idx] = (self.rng.random(len(idx)) < rate).astype(int)

        is_lookalike = np.zeros(n, dtype=int)
        for ind, idx in industries.groupby(industries).groups.items():
            rate = LOOKALIKE_RATE.get(ind, 0.30)
            is_lookalike[idx] = (self.rng.random(len(idx)) < rate).astype(int)

        target_interest = self._group_sample_categorical(
            industries,
            {k: {v: 1 / len(vs) for v in vs} for k, vs in INTEREST_MAP.items()},
            {"General": 1.0},
        )

        return pd.DataFrame({
            "target_age_group": target_age_group,
            "target_gender": target_gender,
            "audience_size": audience_size,
            "is_retargeting": is_retargeting,
            "is_lookalike": is_lookalike,
            "target_interest": target_interest,
        })

    # ------------------------------------------------------------------
    # 퍼널 데이터 (벡터화)
    #
    # !! TARGET LEAKAGE WARNING !!
    # bounce_rate와 landing_page_load_time은 ROAS 값을 입력으로 사용.
    # 예측 모델 학습 시 exclude_leakage=True 사용 권장.
    # ------------------------------------------------------------------
    def _generate_funnel_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        roas = df["ROAS"].values
        industries = df["industry"]

        base_load = np.maximum(0.5, 4.0 - roas * 0.1)
        load_time = np.maximum(0.5, self.rng.normal(base_load, 0.8)).round(2)

        base_bounce = np.maximum(20.0, 65.0 - roas * 2)
        bounce_rate = np.clip(self.rng.normal(base_bounce, 10), 10, 95).round(1)

        avg_session_duration = np.empty(n)
        for ind, idx in industries.groupby(industries).groups.items():
            mu, sigma = SESSION_DURATION_MAP.get(ind, (120, 40))
            avg_session_duration[idx] = np.maximum(10, self.rng.normal(mu, sigma, size=len(idx)))
        avg_session_duration = avg_session_duration.round(1)

        funnel_steps = np.empty(n, dtype=int)
        for ind, idx in industries.groupby(industries).groups.items():
            opts = FUNNEL_MAP.get(ind, [2, 3])
            funnel_steps[idx] = self.rng.choice(opts, size=len(idx))

        return pd.DataFrame({
            "landing_page_load_time": load_time,
            "bounce_rate": bounce_rate,
            "avg_session_duration": avg_session_duration,
            "funnel_steps": funnel_steps,
        })

    # ------------------------------------------------------------------
    # 크리에이티브 임팩트 팩터 (벡터화)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_impact_factors(
        df: pd.DataFrame,
        creative: pd.DataFrame,
        audience: pd.DataFrame,
        funnel: pd.DataFrame,
    ) -> np.ndarray:
        """크리에이티브/타겟팅 특성에 따른 ROAS 보정 계수를 벡터화 계산."""
        n = len(df)
        impact = np.ones(n)

        is_tiktok_video = (creative["is_video"].values == 1) & (df["platform"].values == "TikTok Ads")
        impact[is_tiktok_video] *= 1.08

        vlen = creative["video_length_sec"].values
        impact[(vlen >= 15) & (vlen <= 30)] *= 1.05
        impact[vlen > 60] *= 0.95

        impact[audience["is_retargeting"].values == 1] *= 1.25
        impact[audience["is_lookalike"].values == 1] *= 1.10

        lt = funnel["landing_page_load_time"].values
        impact[lt > 3.0] *= 0.90
        impact[lt < 1.5] *= 1.05

        br = funnel["bounce_rate"].values
        impact[br > 70] *= 0.85
        impact[br < 30] *= 1.10

        impact[creative["copy_sentiment"].values > 0.7] *= 1.03

        return np.round(impact, 4)

    # ------------------------------------------------------------------
    # 전체 데이터 보강
    # ------------------------------------------------------------------
    def enrich_ads_data(self, ads_df: pd.DataFrame, apply_impact: bool = True) -> pd.DataFrame:
        """광고 데이터에 크리에이티브/타겟팅/퍼널 메타데이터 추가."""
        logger.info("[Phase 3] Enriching %d rows...", len(ads_df))

        df = ads_df.reset_index(drop=True)

        creative = self._generate_creative_columns(df)
        audience = self._generate_audience_columns(df)
        funnel = self._generate_funnel_columns(df)

        enriched = pd.concat([df, creative, audience, funnel], axis=1)

        if apply_impact:
            impact = self._compute_impact_factors(df, creative, audience, funnel)
            enriched["creative_impact_factor"] = impact
            enriched["ROAS"] = enriched["ROAS"] * impact
            enriched["revenue"] = enriched["ad_spend"] * enriched["ROAS"]
            enriched["CPA"] = np.where(
                enriched["conversions"] > 0,
                enriched["ad_spend"] / enriched["conversions"],
                enriched["CPA"],
            )

        added = [c for c in enriched.columns if c not in ads_df.columns]
        logger.info("[Phase 3] Added %d columns: %s", len(added), added)
        return enriched

    def save_enriched(self, enriched_df: pd.DataFrame, output_path: str = "./data/ads_with_creative.csv") -> None:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        enriched_df.to_csv(output_path, index=False)
        logger.info("[Phase 3] Saved: %s (%d rows)", output_path, len(enriched_df))


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    enricher = CreativeDataEnricher(seed=42)

    if os.path.exists("global_ads_performance_dataset.csv"):
        ads_df = pd.read_csv("global_ads_performance_dataset.csv")
        enriched = enricher.enrich_ads_data(ads_df, apply_impact=True)

        print(f"\n--- Enriched Data Summary ---")
        print(f"Shape: {enriched.shape}")
        print(f"\nAd Format distribution:\n{enriched['ad_format'].value_counts().head(10)}")
        print(f"\nRetargeting rate: {enriched['is_retargeting'].mean():.1%}")
        print(f"Avg bounce rate: {enriched['bounce_rate'].mean():.1f}%")

        enricher.save_enriched(enriched, "./data/ads_with_creative.csv")
    else:
        print("[ERROR] Base dataset not found")
