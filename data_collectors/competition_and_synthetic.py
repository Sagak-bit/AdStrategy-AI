# -*- coding: utf-8 -*-
"""
Phase 4: 경쟁 환경 데이터 수집 + 합성 데이터 생성
- 4-1: 산업별/플랫폼별 경쟁 강도 지표 시뮬레이션
- 4-2: 플랫폼 알고리즘 변화 이벤트 (holiday_calendar에서 일부 중복)
- 4-3: CTGAN / SMOTE 기반 합성 데이터 생성
- 4-4: 데이터 밸런싱 (저/고성과 균형)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 합성 데이터 관련 (선택적)
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# ===========================================================================
# 경쟁 강도 추정 프로파일
# ===========================================================================

# 산업별 평균 CPC 추이 (연도별, USD)
# 출처: 업계 평균 추정치 (WordStream, SpyFu 리포트 기반)
INDUSTRY_AVG_CPC = {
    'Fintech': {2022: 3.85, 2023: 4.20, 2024: 4.55},
    'EdTech': {2022: 2.40, 2023: 2.65, 2024: 2.90},
    'Healthcare': {2022: 3.10, 2023: 3.35, 2024: 3.60},
    'SaaS': {2022: 4.50, 2023: 4.95, 2024: 5.40},
    'E-commerce': {2022: 1.20, 2023: 1.35, 2024: 1.50},
}

# 플랫폼별 광고주 수 성장률 (연도별 인덱스, 2022=100)
PLATFORM_ADVERTISER_INDEX = {
    'Google Ads': {2022: 100, 2023: 105, 2024: 110},
    'Meta Ads': {2022: 100, 2023: 98, 2024: 103},   # iOS 14.5 영향 후 회복
    'TikTok Ads': {2022: 100, 2023: 140, 2024: 180},  # 급성장
}

# 월별 경쟁 강도 지수 (1=낮음, 10=높음)
MONTHLY_COMPETITION = {
    1: 5, 2: 5, 3: 6, 4: 6, 5: 5, 6: 5,
    7: 4, 8: 5, 9: 6, 10: 7, 11: 9, 12: 10,  # Q4가 가장 치열
}


class CompetitionDataCollector:
    """광고 경쟁 환경 데이터를 생성하고 제공하는 클래스"""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        logger.info("[Phase 4-1] Competition Data Collector initialized (seed=%d)", seed)

    def generate_competition_metrics(self, ads_df: pd.DataFrame) -> pd.DataFrame:
        """광고 데이터에 경쟁 환경 메트릭 추가 (벡터화)."""
        logger.info("[Phase 4-1] Generating competition metrics...")
        df = ads_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        year = df["date"].dt.year
        month = df["date"].dt.month
        n = len(df)

        # 1. 산업 평균 CPC (벡터화 — 산업+연도 조합 lookup)
        industry_avg = np.full(n, 3.0)
        for ind, year_map in INDUSTRY_AVG_CPC.items():
            for yr, val in year_map.items():
                mask = (df["industry"] == ind) & (year == yr)
                industry_avg[mask] = val
        df["industry_avg_cpc"] = np.clip(
            industry_avg + self.rng.normal(0, 0.2, size=n), 0.5, 15.0,
        )

        # 2. CPC vs 산업 평균
        df["cpc_vs_industry_avg"] = np.where(
            df["industry_avg_cpc"] > 0, df["CPC"] / df["industry_avg_cpc"], 1.0,
        )

        # 3. 월별 경쟁 강도 (벡터화)
        base_comp = month.map(MONTHLY_COMPETITION).values.astype(float)
        industry_factor = df["industry"].map(
            {"Fintech": 1.2, "SaaS": 1.3, "Healthcare": 1.1, "E-commerce": 1.0, "EdTech": 0.85}
        ).fillna(1.0).values
        df["competition_index"] = np.clip(
            base_comp * industry_factor + self.rng.normal(0, 0.5, size=n), 1, 10,
        ).round(1)

        # 4. 플랫폼 성장 인덱스 (벡터화)
        growth = np.full(n, 100)
        for plat, year_map in PLATFORM_ADVERTISER_INDEX.items():
            for yr, val in year_map.items():
                mask = (df["platform"] == plat) & (year == yr)
                growth[mask] = val
        df["platform_growth_index"] = growth

        # 5. 경매 밀도
        df["auction_density"] = (df["competition_index"] * df["platform_growth_index"] / 100).round(2)

        # 임시 컬럼 정리
        for col in ("year", "month"):
            if col not in ads_df.columns and col in df.columns:
                df = df.drop(columns=[col])

        added = [c for c in df.columns if c not in ads_df.columns]
        logger.info("  Added %d competition columns: %s", len(added), added)
        return df


class SyntheticDataGenerator:
    """합성 데이터를 생성하여 데이터 볼륨과 밸런스를 개선하는 클래스"""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        logger.info("[Phase 4-2] Synthetic Data Generator initialized (seed=%d)", seed)
        logger.info("  SDV (CTGAN): %s", "Available" if SDV_AVAILABLE else "Not Available")
        logger.info("  imbalanced-learn: %s", "Available" if IMBLEARN_AVAILABLE else "Not Available")

    # -----------------------------------------------------------------------
    # CTGAN 기반 합성 데이터
    # -----------------------------------------------------------------------
    def generate_ctgan(self, df, num_rows=5000, epochs=100):
        """
        CTGAN을 사용하여 기존 데이터의 분포를 학습한 합성 데이터 생성

        Parameters:
        -----------
        df : pd.DataFrame - 학습용 원본 데이터
        num_rows : int - 생성할 합성 행 수
        epochs : int - CTGAN 학습 에폭
        """
        if not SDV_AVAILABLE:
            logger.info("[Phase 4-2] SDV not installed. Using statistical sampling fallback.")
            return self._statistical_sampling(df, num_rows)

        logger.info("[Phase 4-2] Generating %d synthetic rows with CTGAN...", num_rows)

        # 메타데이터 자동 감지
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # CTGAN 학습
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=epochs,
            verbose=True,
        )
        synthesizer.fit(df)

        # 합성 데이터 생성
        synthetic = synthesizer.sample(num_rows=num_rows)
        logger.info("  Generated: %d rows", len(synthetic))

        return synthetic

    # -----------------------------------------------------------------------
    # 통계적 샘플링 기반 합성 데이터 (CTGAN 대체)
    # -----------------------------------------------------------------------
    def _statistical_sampling(self, df: pd.DataFrame, num_rows: int = 5000) -> pd.DataFrame:
        """통계적 샘플링으로 합성 데이터 생성 (벡터화, 재현 가능)."""
        logger.info("[Phase 4-2] Generating %d synthetic rows (statistical sampling)...", num_rows)

        categorical_cols = ["platform", "campaign_type", "industry", "country"]
        numeric_cols = ["impressions", "clicks", "CTR", "CPC", "ad_spend",
                        "conversions", "CPA", "revenue", "ROAS"]
        date_col = "date"

        avail_cat = [c for c in categorical_cols if c in df.columns]
        avail_num = [c for c in numeric_cols if c in df.columns]

        # 1) 카테고리: 원본에서 재현 가능하게 샘플링
        cat_indices = self.rng.choice(len(df), size=num_rows, replace=True)
        cat_df = df[avail_cat].iloc[cat_indices].reset_index(drop=True)

        # 2) 각 카테고리 그룹의 수치 통계를 미리 계산
        group_stats: dict[tuple, dict[str, tuple[float, float]]] = {}
        for key, grp in df.groupby(avail_cat):
            stats = {}
            for col in avail_num:
                mu = grp[col].mean()
                sigma = max(grp[col].std(), mu * 0.05)
                stats[col] = (mu, sigma)
            group_stats[key if isinstance(key, tuple) else (key,)] = stats

        global_stats = {
            col: (df[col].mean(), max(df[col].std(), df[col].mean() * 0.05))
            for col in avail_num
        }

        # 3) 벡터화 샘플링: 그룹별로 한꺼번에 생성
        cat_keys = cat_df[avail_cat].apply(tuple, axis=1)
        num_data: dict[str, np.ndarray] = {col: np.empty(num_rows) for col in avail_num}

        for key, idx in cat_keys.groupby(cat_keys).groups.items():
            n_group = len(idx)
            stats = group_stats.get(key, global_stats)
            for col in avail_num:
                mu, sigma = stats.get(col, global_stats[col])
                num_data[col][idx] = self.rng.normal(mu, sigma, size=n_group)

        num_df = pd.DataFrame(num_data)

        for col in ["impressions", "clicks", "conversions"]:
            if col in num_df.columns:
                num_df[col] = np.maximum(1, np.round(num_df[col])).astype(int)
        for col in ["CPC", "CPA"]:
            if col in num_df.columns:
                num_df[col] = np.maximum(0.01, num_df[col]).round(2)
        if "ad_spend" in num_df.columns:
            num_df["ad_spend"] = np.maximum(100.0, num_df["ad_spend"]).round(2)
        if "revenue" in num_df.columns:
            num_df["revenue"] = np.maximum(0.0, num_df["revenue"]).round(2)
        if "CTR" in num_df.columns:
            num_df["CTR"] = np.clip(num_df["CTR"], 0.001, 0.2)

        # 4) 파생 지표 재계산 (일관성)
        if {"clicks", "impressions"} <= set(num_df.columns):
            num_df["CTR"] = (num_df["clicks"] / num_df["impressions"].clip(lower=1)).round(4)
        if {"ad_spend", "clicks"} <= set(num_df.columns):
            num_df["CPC"] = (num_df["ad_spend"] / num_df["clicks"].clip(lower=1)).round(2)
        if {"ad_spend", "conversions"} <= set(num_df.columns):
            num_df["CPA"] = (num_df["ad_spend"] / num_df["conversions"].clip(lower=1)).round(2)
        if {"revenue", "ad_spend"} <= set(num_df.columns):
            num_df["ROAS"] = (num_df["revenue"] / num_df["ad_spend"].clip(lower=100.0)).round(2)
            real_roas_max = df["ROAS"].quantile(0.99) if "ROAS" in df.columns else 50.0
            num_df["ROAS"] = num_df["ROAS"].clip(-real_roas_max, real_roas_max)

        # 5) 날짜
        synthetic_df = pd.concat([cat_df, num_df], axis=1)
        if date_col in df.columns:
            min_date = pd.to_datetime(df[date_col]).min()
            max_date = pd.to_datetime(df[date_col]).max()
            days_range = max(1, (max_date - min_date).days)
            random_days = self.rng.randint(0, days_range, size=num_rows)
            synthetic_df[date_col] = [(min_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in random_days]

        # 6) 원본에 있지만 아직 처리 안 된 추가 컬럼
        extra_cols = [c for c in df.columns if c not in synthetic_df.columns and c != date_col]
        for col in extra_cols:
            if df[col].dtype == "object":
                idx = self.rng.choice(len(df), size=num_rows, replace=True)
                synthetic_df[col] = df[col].values[idx]
            else:
                mu = df[col].mean()
                sigma = max(df[col].std(), 0.01)
                synthetic_df[col] = self.rng.normal(mu, sigma, size=num_rows)

        synthetic_df = synthetic_df[[c for c in df.columns if c in synthetic_df.columns]]
        logger.info("  Generated: %d rows, %d columns", len(synthetic_df), len(synthetic_df.columns))
        return synthetic_df

    # -----------------------------------------------------------------------
    # ROAS 기반 밸런싱 (고/저성과 균형)
    # -----------------------------------------------------------------------
    def balance_by_performance(self, df, target_col='ROAS', n_bins=5):
        """
        ROAS 분포를 균등하게 만들어 모델이 저/고성과 모두 잘 학습하도록 함.
        소수 구간을 오버샘플링하여 밸런싱.
        """
        logger.info("[Phase 4-2] Balancing data by %s distribution...", target_col)

        df_copy = df.copy()
        df_copy['perf_bin'] = pd.qcut(df_copy[target_col], q=n_bins, labels=False, duplicates='drop')

        bin_counts = df_copy['perf_bin'].value_counts()
        max_count = bin_counts.max()
        logger.info("  Before balancing: %s", bin_counts.to_dict())

        balanced_parts = []
        for bin_val in df_copy['perf_bin'].unique():
            bin_df = df_copy[df_copy['perf_bin'] == bin_val]
            if len(bin_df) < max_count:
                # 부족한 만큼 오버샘플링 (노이즈 추가)
                extra_needed = max_count - len(bin_df)
                extra = bin_df.sample(extra_needed, replace=True, random_state=self.seed)

                # 수치 컬럼에 약간의 노이즈 추가
                numeric_cols = extra.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != 'perf_bin':
                        noise = self.rng.normal(0, extra[col].std() * 0.05, size=len(extra))
                        extra[col] = extra[col] + noise

                balanced_parts.append(bin_df)
                balanced_parts.append(extra)
            else:
                balanced_parts.append(bin_df)

        balanced = pd.concat(balanced_parts, ignore_index=True)
        balanced = balanced.drop(columns=['perf_bin'])

        logger.info("  After balancing: %d -> %d rows", len(df), len(balanced))
        return balanced

    # -----------------------------------------------------------------------
    # 전체 합성 파이프라인
    # -----------------------------------------------------------------------
    def augment_dataset(self, df, target_total=10000, balance=True):
        """
        데이터셋을 목표 크기까지 합성 데이터로 보강

        Parameters:
        -----------
        df : pd.DataFrame - 원본 데이터
        target_total : int - 목표 행 수
        balance : bool - 성과 밸런싱 여부
        """
        logger.info("[Phase 4-2] Augmenting dataset: %d -> target %d", len(df), target_total)

        if len(df) >= target_total:
            logger.info("  Dataset already meets target. Skipping augmentation.")
            if balance:
                return self.balance_by_performance(df)
            return df

        need = target_total - len(df)
        synthetic = self._statistical_sampling(df, num_rows=need)

        augmented = pd.concat([df, synthetic], ignore_index=True)
        augmented['is_synthetic'] = [0] * len(df) + [1] * len(synthetic)

        logger.info("  Augmented: %d rows (original: %d, synthetic: %d)", len(augmented), len(df), len(synthetic))

        if balance:
            augmented = self.balance_by_performance(augmented)

        return augmented


# ===========================================================================
# 통합 파이프라인
# ===========================================================================
def run_phase4(ads_df: pd.DataFrame, target_total: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Phase 4 전체 실행."""
    logger.info("[Phase 4] Competition + Synthetic Data Pipeline")

    comp = CompetitionDataCollector(seed=seed)
    df = comp.generate_competition_metrics(ads_df)

    synth = SyntheticDataGenerator(seed=seed)
    df = synth.augment_dataset(df, target_total=target_total, balance=True)

    logger.info("[Phase 4] Final dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    if os.path.exists('global_ads_performance_dataset.csv'):
        ads_df = pd.read_csv('global_ads_performance_dataset.csv')
        print(f"Original: {len(ads_df)} rows")

        result = run_phase4(ads_df, target_total=10000, seed=42)

        # 저장
        os.makedirs('./data', exist_ok=True)
        result.to_csv('./data/ads_augmented.csv', index=False)
        print(f"\nSaved: ./data/ads_augmented.csv ({len(result)} rows)")

        # 통계 출력
        print(f"\n--- Augmented Data Stats ---")
        print(f"Total rows: {len(result)}")
        if 'is_synthetic' in result.columns:
            print(f"Original rows: {(result['is_synthetic'] == 0).sum()}")
            print(f"Synthetic rows: {(result['is_synthetic'] == 1).sum()}")
        print(f"ROAS mean: {result['ROAS'].mean():.2f}")
        print(f"ROAS std: {result['ROAS'].std():.2f}")
        print(f"Columns: {list(result.columns)}")
    else:
        print("[ERROR] Base dataset not found")
