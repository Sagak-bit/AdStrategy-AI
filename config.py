# -*- coding: utf-8 -*-
"""
프로젝트 공유 설정
==================
매직 넘버, 파일 경로, 피처 정의, 임계값 등을 중앙 집중 관리합니다.
모든 모듈은 이 파일을 참조하여 일관성을 유지합니다.
"""
from __future__ import annotations

import os
from typing import Final

# ============================================================================
# 경로
# ============================================================================
PROJECT_ROOT: Final[str] = os.path.dirname(os.path.abspath(__file__))

DATA_DIR: Final[str] = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR: Final[str] = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR: Final[str] = os.path.join(PROJECT_ROOT, "figures")

ENRICHED_DATA_PATH: Final[str] = os.path.join(DATA_DIR, "enriched_ads_final.csv")
BASE_DATA_PATH: Final[str] = os.path.join(PROJECT_ROOT, "global_ads_performance_dataset.csv")
TRENDS_DATA_PATH: Final[str] = os.path.join(DATA_DIR, "trends", "industry_trends.csv")
PREDICTOR_PKL_PATH: Final[str] = os.path.join(MODELS_DIR, "predictor_v2.pkl")

# ============================================================================
# 피처 정의
# ============================================================================
LEAKAGE_FEATURES: Final[frozenset[str]] = frozenset({
    "bounce_rate",
    "landing_page_load_time",
    "creative_impact_factor",
})

BASE_CAT_FEATURES: Final[list[str]] = [
    "platform", "campaign_type", "industry", "country",
]

BASE_NUM_FEATURES: Final[list[str]] = [
    "ad_spend", "month", "quarter", "ad_spend_log",
    "ad_spend_bin", "is_q4", "is_weekend", "day_of_week", "week_of_year",
]

CALENDAR_NUM_FEATURES: Final[list[str]] = [
    "is_public_holiday", "days_to_next_holiday",
    "is_shopping_season", "season_intensity", "is_major_event",
    "platform_event_impact", "is_month_start", "is_month_end",
]

COMPETITION_NUM_FEATURES: Final[list[str]] = [
    "industry_avg_cpc", "cpc_vs_industry_avg",
    "competition_index", "platform_growth_index", "auction_density",
]

CREATIVE_CAT_FEATURES: Final[list[str]] = [
    "ad_format", "cta_type", "target_age_group",
    "target_gender", "target_interest",
]

CREATIVE_NUM_FEATURES_CLEAN: Final[list[str]] = [
    "is_video", "video_length_sec", "has_cta",
    "headline_length", "copy_sentiment", "num_products_shown",
    "audience_size", "is_retargeting", "is_lookalike",
    "avg_session_duration", "funnel_steps",
]

LEAKAGE_ORDERED: Final[list[str]] = [
    "bounce_rate", "landing_page_load_time", "creative_impact_factor",
]

AD_SPEND_BINS: Final[list[float]] = [0, 1000, 3000, 5000, 10000, 20000, float("inf")]
AD_SPEND_BIN_LABELS: Final[list[int]] = [0, 1, 2, 3, 4, 5]

# ============================================================================
# 모델 학습 파라미터
# ============================================================================
TRAIN_TEST_SPLIT_RATIO: Final[float] = 0.8
DEFAULT_RANDOM_STATE: Final[int] = 42
CV_N_SPLITS: Final[int] = 5
MIN_NON_NULL_RATIO: Final[float] = 0.3

CONFIDENCE_Z_95: Final[float] = 1.96
CONFIDENCE_HIGH_THRESHOLD: Final[float] = 0.5
CONFIDENCE_MEDIUM_THRESHOLD: Final[float] = 0.3

# ============================================================================
# API / Agent
# ============================================================================
TOOL_CALL_MAX_ITERATIONS: Final[int] = 5
LLM_TEMPERATURE: Final[float] = 0.7
LLM_MAX_TOKENS: Final[int] = 2000
MAX_SCENARIOS: Final[int] = 5
MAX_INPUT_LENGTH: Final[int] = 2000

MIN_AD_SPEND: Final[float] = 0.0
MIN_CPC_FLOOR: Final[float] = 0.1

# ============================================================================
# 시각화
# ============================================================================
CHART_DPI: Final[int] = 200
CHART_HEIGHT_DEFAULT: Final[int] = 350

COLOR_PRIMARY: Final[str] = "#4A90D9"
COLOR_SUCCESS: Final[str] = "#27AE60"
COLOR_DANGER: Final[str] = "#E74C3C"
COLOR_WARNING: Final[str] = "#F39C12"
COLOR_INFO: Final[str] = "#5DADE2"
COLOR_MUTED: Final[str] = "#95a5a6"

# ============================================================================
# 예측 기본값 (보강 피처가 없을 때 사용)
# ============================================================================
PREDICT_ENRICHED_DEFAULTS: Final[dict[str, object]] = {
    "cpi_index": 110.0, "cpi_yoy_pct": 3.0, "unemployment_rate": 4.0,
    "gdp_growth_pct": 2.0, "exchange_rate_usd": 1.0,
    "is_public_holiday": 0, "days_to_next_holiday": 15,
    "is_major_event": 0, "platform_event_impact": 0,
    "is_month_start": 0, "is_month_end": 0,
    "trend_index": 50.0, "trend_momentum": 0.0,
    "is_video": 0, "video_length_sec": 0, "has_cta": 1,
    "headline_length": 45, "copy_sentiment": 0.7,
    "num_products_shown": 1, "is_retargeting": 0, "is_lookalike": 0,
    "landing_page_load_time": 2.5, "bounce_rate": 50.0,
    "avg_session_duration": 120.0, "funnel_steps": 3,
    "creative_impact_factor": 1.0,
    "industry_avg_cpc": 3.0, "cpc_vs_industry_avg": 1.0,
    "competition_index": 5.0, "platform_growth_index": 100,
    "auction_density": 5.0,
    "ad_format": "image", "cta_type": "Learn More",
    "target_age_group": "25-34", "target_gender": "All",
    "target_interest": "General",
}

# ============================================================================
# 데이터 파이프라인
# ============================================================================
MIN_DATASET_ROWS: Final[int] = 1000
MAX_NULL_PCT: Final[float] = 20.0
MAX_SYNTHETIC_PCT: Final[float] = 60.0
DEFAULT_TARGET_TOTAL_ROWS: Final[int] = 10000

REQUIRED_COLUMNS: Final[list[str]] = [
    "date", "platform", "campaign_type", "industry", "country",
    "impressions", "clicks", "CTR", "CPC", "ad_spend",
    "conversions", "CPA", "revenue", "ROAS",
]
