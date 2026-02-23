# -*- coding: utf-8 -*-
"""
데이터 수집/보강 통합 파이프라인
모든 Phase의 데이터 수집기를 순차 실행하여 최종 보강 데이터셋 생성

사용법:
    python data_pipeline.py
    
출력:
    data/enriched_ads_final.csv  -- 모든 외부 데이터가 조인된 최종 데이터셋
"""

from __future__ import annotations

import logging
import os
import sys
import time

import pandas as pd

from config import (
    BASE_DATA_PATH,
    DATA_DIR,
    DEFAULT_TARGET_TOTAL_ROWS,
    LEAKAGE_ORDERED,
    MAX_NULL_PCT,
    MAX_SYNTHETIC_PCT,
    MIN_DATASET_ROWS,
    REQUIRED_COLUMNS,
)
from utils import configure_windows_encoding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_pipeline")

configure_windows_encoding()

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
    base_data_path: str = BASE_DATA_PATH,
    output_dir: str = DATA_DIR,
    target_total_rows: int = DEFAULT_TARGET_TOTAL_ROWS,
    extend_years: tuple[int, ...] = (2022, 2023),
    fred_api_key: str | None = None,
    seed: int = 42,
    skip_leakage_features: bool = False,
) -> pd.DataFrame | None:
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
    skip_leakage_features : bool - True면 Phase 3에서 leakage 변수
        (bounce_rate, landing_page_load_time, creative_impact_factor) 생성을 건너뜀.
        Ablation Study에서 이 변수들이 ROAS의 역함수로 생성되어
        target leakage를 유발함이 확인되었음.
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
        logger.error("Base dataset not found: %s", base_data_path)
        return None

    base_df = pd.read_csv(base_data_path)
    print(f"\n[START] Original dataset: {len(base_df)} rows, {len(base_df.columns)} columns")

    # 각 Phase 시작 전 체크포인트 저장 (롤백 지원)
    checkpoints = {}

    def save_checkpoint(phase_name, dataframe):
        """Phase 성공 시 체크포인트 저장"""
        checkpoints[phase_name] = dataframe.copy()
        logger.info("Checkpoint saved: %s (%d rows, %d cols)", phase_name, len(dataframe), len(dataframe.columns))

    def rollback_to(phase_name):
        """실패 시 마지막 성공 체크포인트로 롤백"""
        if phase_name in checkpoints:
            logger.warning("Rolling back to checkpoint: %s", phase_name)
            return checkpoints[phase_name].copy()
        logger.error("No checkpoint found for %s", phase_name)
        return None

    save_checkpoint('base', base_df)

    # ===================================================================
    # Phase 1: Kaggle 데이터 병합 + 시계열 확장
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 1: Data Volume Expansion")
    print("=" * 80)

    try:
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
                        try:
                            merger.merge_csv_file(os.path.join(root, f))
                        except Exception as e:
                            logger.warning("Kaggle CSV 병합 실패 (%s): %s", f, e)

        # 시계열 확장
        if extend_years:
            merger.extend_time_range(target_years=extend_years)

        df = merger.base_df.copy()
        merger.save_merged(os.path.join(output_dir, 'phase1_expanded.csv'))
        save_checkpoint('phase1', df)
        print(f"[Phase 1 Done] {len(df)} rows")
    except Exception as e:
        logger.error("Phase 1 실패: %s", e)
        df = rollback_to('base')
        if df is None:
            return None
        print(f"[Phase 1 FAILED] Rollback to base ({len(df)} rows)")

    # ===================================================================
    # Phase 2-1: 거시경제 지표
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2-1: Macro Economic Indicators")
    print("=" * 80)

    try:
        macro = MacroEconomicCollector(
            fred_api_key=fred_api_key or os.environ.get('FRED_API_KEY', ''),
            output_dir=os.path.join(output_dir, 'macro')
        )
        macro.collect_all(start_date='2020-01-01')
        macro.save()
        df = macro.enrich_ads_data(df)
        save_checkpoint('phase2_1', df)
        print(f"[Phase 2-1 Done] {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error("Phase 2-1 실패: %s", e)
        print(f"[Phase 2-1 FAILED] 이전 체크포인트 유지 ({len(df)} rows)")

    # ===================================================================
    # Phase 2-2: 시즌/공휴일 달력
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2-2: Holiday & Season Calendar")
    print("=" * 80)

    try:
        years_in_data = sorted(pd.to_datetime(df['date']).dt.year.unique().tolist())
        calendar = HolidayCalendar(years=years_in_data)
        calendar.build_calendar()
        calendar.save(os.path.join(output_dir, 'calendar', 'holiday_calendar.csv'))
        df = calendar.enrich_ads_data(df)
        df = calendar.get_platform_events_for_ads(df)
        save_checkpoint('phase2_2', df)
        print(f"[Phase 2-2 Done] {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error("Phase 2-2 실패: %s", e)
        print(f"[Phase 2-2 FAILED] 이전 체크포인트 유지 ({len(df)} rows)")

    # ===================================================================
    # Phase 2-3: Google Trends 관심도
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2-3: Industry Trend Index")
    print("=" * 80)

    try:
        trends = GoogleTrendsCollector(output_dir=os.path.join(output_dir, 'trends'))
        trends.fetch_trends()
        trends.save()
        df = trends.enrich_ads_data(df)
        save_checkpoint('phase2_3', df)
        print(f"[Phase 2-3 Done] {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error("Phase 2-3 실패: %s", e)
        print(f"[Phase 2-3 FAILED] 이전 체크포인트 유지 ({len(df)} rows)")

    # ===================================================================
    # Phase 3: 크리에이티브/타겟팅 메타데이터
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 3: Creative & Targeting Metadata")
    print("  ⚠ WARNING: 이 Phase에서 생성되는 bounce_rate, landing_page_load_time,")
    print("    creative_impact_factor는 target leakage를 포함합니다.")
    print("    예측 모델 학습 시 exclude_leakage=True 사용을 권장합니다.")
    print("=" * 80)

    try:
        creative = CreativeDataEnricher(seed=seed)
        df = creative.enrich_ads_data(df, apply_impact=True)

        LEAKAGE_COLS = list(LEAKAGE_ORDERED)
        if skip_leakage_features:
            dropped = [c for c in LEAKAGE_COLS if c in df.columns]
            df = df.drop(columns=dropped, errors='ignore')
            logger.info("Leakage 변수 제거됨: %s", dropped)
            print(f"  [LEAKAGE] 제거된 변수: {dropped}")
        else:
            present = [c for c in LEAKAGE_COLS if c in df.columns]
            if present:
                logger.warning("Leakage 변수가 포함되어 있습니다: %s", present)
                print(f"  ⚠ [LEAKAGE WARNING] 다음 변수는 ROAS로부터 역산 생성됨: {present}")
                print(f"    -> 예측 모델 학습 시 exclude_leakage=True 사용 권장")
                print(f"    -> 파이프라인에서 제거하려면 skip_leakage_features=True 설정")

        save_checkpoint('phase3', df)
        print(f"[Phase 3 Done] {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error("Phase 3 실패: %s", e)
        print(f"[Phase 3 FAILED] 이전 체크포인트 유지 ({len(df)} rows)")

    # ===================================================================
    # Phase 4: 경쟁 환경 + 합성 데이터
    # ===================================================================
    print("\n" + "=" * 80)
    print("  PHASE 4: Competition + Synthetic Data")
    print("=" * 80)

    try:
        comp = CompetitionDataCollector(seed=seed)
        df = comp.generate_competition_metrics(df)

        synth = SyntheticDataGenerator(seed=seed)
        if len(df) < target_total_rows:
            df = synth.augment_dataset(df, target_total=target_total_rows, balance=True)
        else:
            df = synth.balance_by_performance(df)

        save_checkpoint('phase4', df)
        print(f"[Phase 4 Done] {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error("Phase 4 실패: %s", e)
        print(f"[Phase 4 FAILED] 이전 체크포인트 유지 ({len(df)} rows)")

    # ===================================================================
    # 데이터 품질 검증 (Validation)
    # ===================================================================
    print("\n" + "=" * 80)
    print("  VALIDATION: Data Quality Checks")
    print("=" * 80)

    validation_passed = True
    validation_results = []

    if len(df) < MIN_DATASET_ROWS:
        msg = f"[WARN] 행 수가 적음: {len(df)}건 (최소 {MIN_DATASET_ROWS:,}건 권장)"
        print(f"  {msg}")
        validation_results.append(msg)
    else:
        print(f"  [OK] 행 수: {len(df):,}건")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"[FAIL] 필수 컬럼 누락: {missing_cols}"
        print(f"  {msg}")
        validation_results.append(msg)
        validation_passed = False
    else:
        print(f"  [OK] 필수 컬럼 {len(REQUIRED_COLUMNS)}개 모두 존재")

    null_pct = (df.isnull().sum() / len(df) * 100)
    high_null = null_pct[null_pct > MAX_NULL_PCT]
    if len(high_null) > 0:
        msg = f"[WARN] 결측치 비율 20% 초과 컬럼: {dict(high_null.round(1))}"
        print(f"  {msg}")
        validation_results.append(msg)
    else:
        print(f"  [OK] 모든 컬럼 결측치 비율 {MAX_NULL_PCT}% 이하")

    # 체크 4: 값 범위 검증
    if 'ROAS' in df.columns:
        negative_roas = (df['ROAS'] < 0).sum()
        if negative_roas > 0:
            msg = f"[WARN] 음수 ROAS: {negative_roas}건"
            print(f"  {msg}")
            validation_results.append(msg)
        else:
            print(f"  [OK] ROAS 값 범위 정상 (min={df['ROAS'].min():.2f})")

    if 'CTR' in df.columns:
        invalid_ctr = ((df['CTR'] < 0) | (df['CTR'] > 100)).sum()
        if invalid_ctr > 0:
            msg = f"[WARN] 비정상 CTR (0~100 벗어남): {invalid_ctr}건"
            print(f"  {msg}")
            validation_results.append(msg)
        else:
            print(f"  [OK] CTR 값 범위 정상")

    # 체크 5: 범주형 값 검증
    expected_platforms = {'Google Ads', 'Meta Ads', 'TikTok Ads'}
    if 'platform' in df.columns:
        actual_platforms = set(df['platform'].unique())
        unexpected = actual_platforms - expected_platforms
        if unexpected:
            msg = f"[WARN] 예상 외 플랫폼 값: {unexpected}"
            print(f"  {msg}")
            validation_results.append(msg)
        else:
            print(f"  [OK] 플랫폼 값 정상: {sorted(actual_platforms)}")

    # 체크 6: 합성 데이터 비율
    if 'is_synthetic' in df.columns:
        synth_pct = df['is_synthetic'].mean() * 100
        print(f"  [INFO] 합성 데이터 비율: {synth_pct:.1f}%")
        if synth_pct > MAX_SYNTHETIC_PCT:
            msg = f"[WARN] 합성 데이터 비율이 {MAX_SYNTHETIC_PCT}%를 초과 ({synth_pct:.1f}%)"
            validation_results.append(msg)

    # 체크 7: 중복 행 검사
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        msg = f"[WARN] 중복 행: {dup_count}건"
        print(f"  {msg}")
        validation_results.append(msg)
    else:
        print(f"  [OK] 중복 행 없음")

    print(f"\n  검증 결과: {'PASS' if validation_passed else 'FAIL'} "
          f"({len(validation_results)}건의 경고/오류)")

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
    print(f"  Validation: {'PASS' if validation_passed else 'FAIL'}")
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
        base_data_path=BASE_DATA_PATH,
        output_dir=DATA_DIR,
        target_total_rows=DEFAULT_TARGET_TOTAL_ROWS,
        extend_years=(2022, 2023),
        seed=42,
    )

    if result is not None:
        print(f"\nFinal dataset shape: {result.shape}")
        print(f"Memory usage: {result.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
