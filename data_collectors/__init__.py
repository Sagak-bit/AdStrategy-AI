# -*- coding: utf-8 -*-
"""
데이터 수집 파이프라인 패키지
- Phase 1: Kaggle 유사 광고 데이터셋 병합
- Phase 2: 외부 맥락 데이터 (경제지표, 시즌, 트렌드)
- Phase 3: 크리에이티브/타겟팅 메타데이터 보강
- Phase 4: 경쟁 환경 + 합성 데이터
"""
from data_collectors.kaggle_data_merger import KaggleDataMerger
from data_collectors.macro_economic_collector import MacroEconomicCollector
from data_collectors.holiday_calendar import HolidayCalendar
from data_collectors.google_trends_collector import GoogleTrendsCollector
from data_collectors.creative_data_enricher import CreativeDataEnricher
from data_collectors.competition_and_synthetic import (
    CompetitionDataCollector,
    SyntheticDataGenerator,
)

__all__ = [
    "KaggleDataMerger",
    "MacroEconomicCollector",
    "HolidayCalendar",
    "GoogleTrendsCollector",
    "CreativeDataEnricher",
    "CompetitionDataCollector",
    "SyntheticDataGenerator",
]
