# -*- coding: utf-8 -*-
"""
데이터 수집/보강 통합 파이프라인
모든 Phase의 데이터 수집기를 순차 실행하여 최종 보강 데이터셋 생성

사용법:
    python data_pipeline.py
    
출력:
    data/enriched_ads_final.csv  -- 모든 외부 데이터가 조인된 최종 데이터셋
"""

import pandas as pd
import numpy as np
import os
import sys
import io
import time
from datetime import datetime

# Windows 콘솔 인코딩
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 데이터 수집 모듈 임포트
from data_collectors.kaggle_data_merger import KaggleDataMerger
from data_collectors.macro_economic_collector import MacroEconomicCollector
from data_collectors.holiday_calendar import HolidayCalendar
from data_collectors.google_trends_collector import GoogleTrendsCollector
from data_collectors.creative_data_enricher import CreativeDataEnricher
from data_collectors.competition_and_synthetic import (
    CompetitionDataCollector, SyntheticDataGenerator, run_phase4
)


def run_pipeline(
    base_data_path='global_ads_performance_dataset.csv',
    output_dir='./data',
    target_total_rows=10000,
    extend_years=(2022, 2023),
    fred_api_key=None,
    seed=42,
):
    """
    전체 데이터 파이프라인 실행

    Parameters:
    -----------
    base_data_path : str - 원본 광고 데이터 CSV 경로
    output_dir : str - 출력 디렉토리
    target_total_rows : int - 최종 목표 행 수
    extend_years : tuple - 시계열 확장할 연도들
    fred_api_key : str - FRED API 키 (없으면 fallback 사용)
    seed : int - 랜덤 시드
    """
    start_time = time.time()

    print("=" * 80)
    print("  DATA ENRICHMENT PIPELINE")
    print("  광고 모델 성능 향상을 위한 데이터 수집/보강 파이프라인")
    print("=" * 80)
    print(f"  Base data: {base_data_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Target rows: {target_total_rows}")
    print(f"  Time extension: {extend_years}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # 원본 데이터 로드
    if not os.path.exists(base_data_path):
        print(f"[ERROR] Base dataset not found: {base_data_path}")
        return None

    base_df = pd.read_csv(base_data_path)
    print(f"\n[START] Original dataset: {len(base_df)} rows, {len(base_df.columns)} columns")

    # ===================================================================
    # Phase 1: Kaggle 데이터 병합 + 시계열 확장
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 1: Data Volume Expansion")
    print("=" * 80)

    merger = KaggleDataMerger(
        base_data_path=base_data_path,
        download_dir=os.path.join(output_dir, 'kaggle_downloads')
    )

    # 로컬에 다운로드된 CSV가 있으면 병합
    kaggle_dir = os.path.join(output_dir, 'kaggle_downloads')
    if os.path.exists(kaggle_dir):
        for root, dirs, files in os.walk(kaggle_dir):
            for f in files:
                if f.endswith('.csv'):
                    merger.merge_csv_file(os.path.join(root, f))

    # 시계열 확장
    if extend_years:
        merger.extend_time_range(target_years=extend_years)

    df = merger.base_df.copy()
    merger.save_merged(os.path.join(output_dir, 'phase1_expanded.csv'))
    print(f"[Phase 1 Done] {len(df)} rows")

    # ===================================================================
    # Phase 2-1: 거시경제 지표
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2-1: Macro Economic Indicators")
    print("=" * 80)

    macro = MacroEconomicCollector(
        fred_api_key=fred_api_key or os.environ.get('FRED_API_KEY', ''),
        output_dir=os.path.join(output_dir, 'macro')
    )
    macro.collect_all(start_date='2020-01-01')
    macro.save()
    df = macro.enrich_ads_data(df)
    print(f"[Phase 2-1 Done] {len(df)} rows, {len(df.columns)} columns")

    # ===================================================================
    # Phase 2-2: 시즌/공휴일 달력
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2-2: Holiday & Season Calendar")
    print("=" * 80)

    years_in_data = sorted(pd.to_datetime(df['date']).dt.year.unique().tolist())
    calendar = HolidayCalendar(years=years_in_data)
    calendar.build_calendar()
    calendar.save(os.path.join(output_dir, 'calendar', 'holiday_calendar.csv'))
    df = calendar.enrich_ads_data(df)
    df = calendar.get_platform_events_for_ads(df)
    print(f"[Phase 2-2 Done] {len(df)} rows, {len(df.columns)} columns")

    # ===================================================================
    # Phase 2-3: Google Trends 관심도
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2-3: Industry Trend Index")
    print("=" * 80)

    trends = GoogleTrendsCollector(output_dir=os.path.join(output_dir, 'trends'))
    trends._generate_fallback_trends()  # API 호출 대신 fallback 사용 (안전)
    trends.save()
    df = trends.enrich_ads_data(df)
    print(f"[Phase 2-3 Done] {len(df)} rows, {len(df.columns)} columns")

    # ===================================================================
    # Phase 3: 크리에이티브/타겟팅 메타데이터
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 3: Creative & Targeting Metadata")
    print("=" * 80)

    creative = CreativeDataEnricher(seed=seed)
    df = creative.enrich_ads_data(df, apply_impact=True)
    print(f"[Phase 3 Done] {len(df)} rows, {len(df.columns)} columns")

    # ===================================================================
    # Phase 4: 경쟁 환경 + 합성 데이터
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 4: Competition + Synthetic Data")
    print("=" * 80)

    comp = CompetitionDataCollector(seed=seed)
    df = comp.generate_competition_metrics(df)

    synth = SyntheticDataGenerator(seed=seed)
    if len(df) < target_total_rows:
        df = synth.augment_dataset(df, target_total=target_total_rows, balance=True)
    else:
        df = synth.balance_by_performance(df)

    print(f"[Phase 4 Done] {len(df)} rows, {len(df.columns)} columns")

    # ===================================================================
    # 최종 저장
    # ===================================================================
    print("\n" + "=" * 80)
    print("  FINAL: Saving enriched dataset")
    print("=" * 80)

    output_path = os.path.join(output_dir, 'enriched_ads_final.csv')
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Output: {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Column list:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        print(f"    {i+1:3d}. {col:<35} {str(dtype):<15} (nulls: {nulls})")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 80}")

    return df


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    result = run_pipeline(
        base_data_path='global_ads_performance_dataset.csv',
        output_dir='./data',
        target_total_rows=10000,
        extend_years=(2022, 2023),
        seed=42,
    )

    if result is not None:
        print(f"\nFinal dataset shape: {result.shape}")
        print(f"Memory usage: {result.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
