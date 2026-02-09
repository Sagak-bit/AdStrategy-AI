# -*- coding: utf-8 -*-
"""
Phase 4: 경쟁 환경 데이터 수집 + 합성 데이터 생성
- 4-1: 산업별/플랫폼별 경쟁 강도 지표 시뮬레이션
- 4-2: 플랫폼 알고리즘 변화 이벤트 (holiday_calendar에서 일부 중복)
- 4-3: CTGAN / SMOTE 기반 합성 데이터 생성
- 4-4: 데이터 밸런싱 (저/고성과 균형)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

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

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        print("[Phase 4-1] Competition Data Collector initialized")

    def generate_competition_metrics(self, ads_df):
        """
        광고 데이터에 경쟁 환경 메트릭 추가
        - industry_avg_cpc: 산업 평균 CPC
        - cpc_vs_industry_avg: 내 CPC / 산업 평균
        - competition_index: 경쟁 강도 지수 (0-10)
        - platform_growth_index: 플랫폼 광고주 수 성장 인덱스
        - estimated_auction_density: 추정 경매 밀도
        """
        print(f"\n[Phase 4-1] Generating competition metrics...")
        df = ads_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # 1. 산업 평균 CPC
        df['industry_avg_cpc'] = df.apply(
            lambda row: INDUSTRY_AVG_CPC.get(row['industry'], {}).get(
                row['year'], 3.0
            ) + self.rng.normal(0, 0.2),
            axis=1
        )
        df['industry_avg_cpc'] = df['industry_avg_cpc'].clip(0.5, 15.0)

        # 2. 내 CPC vs 산업 평균
        df['cpc_vs_industry_avg'] = np.where(
            df['industry_avg_cpc'] > 0,
            df['CPC'] / df['industry_avg_cpc'],
            1.0
        )

        # 3. 월별 경쟁 강도 지수 (산업/플랫폼/월 조합)
        df['competition_index'] = df['month'].map(MONTHLY_COMPETITION)
        # 산업별 보정
        industry_comp_factor = {
            'Fintech': 1.2, 'SaaS': 1.3, 'Healthcare': 1.1,
            'E-commerce': 1.0, 'EdTech': 0.85,
        }
        df['competition_index'] = df.apply(
            lambda row: min(10, row['competition_index'] * industry_comp_factor.get(row['industry'], 1.0)
                           + self.rng.normal(0, 0.5)),
            axis=1
        ).clip(1, 10).round(1)

        # 4. 플랫폼 광고주 성장 인덱스
        df['platform_growth_index'] = df.apply(
            lambda row: PLATFORM_ADVERTISER_INDEX.get(row['platform'], {}).get(
                row['year'], 100
            ),
            axis=1
        )

        # 5. 추정 경매 밀도 (competition_index * platform_growth 정규화)
        df['auction_density'] = (
            df['competition_index'] * df['platform_growth_index'] / 100
        ).round(2)

        # 임시 컬럼 정리
        if 'year' not in ads_df.columns:
            df = df.drop(columns=['year'], errors='ignore')
        if 'month' not in ads_df.columns:
            df = df.drop(columns=['month'], errors='ignore')

        added = [c for c in df.columns if c not in ads_df.columns]
        print(f"  Added {len(added)} competition columns: {added}")

        return df


class SyntheticDataGenerator:
    """합성 데이터를 생성하여 데이터 볼륨과 밸런스를 개선하는 클래스"""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        print("[Phase 4-2] Synthetic Data Generator initialized")
        print(f"  SDV (CTGAN): {'Available' if SDV_AVAILABLE else 'Not Available'}")
        print(f"  imbalanced-learn: {'Available' if IMBLEARN_AVAILABLE else 'Not Available'}")

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
            print("[Phase 4-2] SDV not installed. pip install sdv")
            print("  Using statistical sampling fallback instead.")
            return self._statistical_sampling(df, num_rows)

        print(f"\n[Phase 4-2] Generating {num_rows} synthetic rows with CTGAN...")

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
        print(f"  Generated: {len(synthetic)} rows")

        return synthetic

    # -----------------------------------------------------------------------
    # 통계적 샘플링 기반 합성 데이터 (CTGAN 대체)
    # -----------------------------------------------------------------------
    def _statistical_sampling(self, df, num_rows=5000):
        """
        CTGAN 불가 시 통계적 샘플링으로 합성 데이터 생성.
        각 카테고리 조합에 대해 수치 컬럼의 분포를 학습하여 샘플링.
        """
        print(f"\n[Phase 4-2] Generating {num_rows} synthetic rows (statistical sampling)...")

        categorical_cols = ['platform', 'campaign_type', 'industry', 'country']
        numeric_cols = ['impressions', 'clicks', 'CTR', 'CPC', 'ad_spend',
                        'conversions', 'CPA', 'revenue', 'ROAS']
        date_col = 'date'

        # 사용 가능한 컬럼만 필터링
        avail_cat = [c for c in categorical_cols if c in df.columns]
        avail_num = [c for c in numeric_cols if c in df.columns]

        synthetic_rows = []
        for _ in range(num_rows):
            # 카테고리 변수: 원본 데이터에서 랜덤 조합 선택
            cat_sample = df[avail_cat].sample(1, random_state=None).iloc[0].to_dict()

            # 해당 카테고리 그룹의 수치 분포 추출
            mask = pd.Series(True, index=df.index)
            for col in avail_cat:
                mask = mask & (df[col] == cat_sample[col])

            group = df[mask]
            if len(group) < 3:
                group = df  # 샘플 부족 시 전체 데이터 사용

            # 수치 변수: 그룹 내 정규분포에서 샘플링
            num_sample = {}
            for col in avail_num:
                mean = group[col].mean()
                std = max(group[col].std(), mean * 0.05)  # 최소 5% 변동
                value = self.rng.normal(mean, std)
                if col in ['impressions', 'clicks', 'conversions']:
                    value = max(1, int(round(value)))
                elif col in ['CTR']:
                    value = max(0.001, min(0.2, value))
                elif col in ['CPC', 'CPA', 'ad_spend', 'revenue', 'ROAS']:
                    value = max(0.01, round(value, 2))
                num_sample[col] = value

            # 날짜: 원본 날짜 범위 내에서 랜덤
            if date_col in df.columns:
                min_date = pd.to_datetime(df[date_col]).min()
                max_date = pd.to_datetime(df[date_col]).max()
                days_range = (max_date - min_date).days
                random_days = self.rng.randint(0, max(1, days_range))
                cat_sample[date_col] = (min_date + pd.Timedelta(days=random_days)).strftime('%Y-%m-%d')

            # 파생 지표 재계산 (일관성 보장)
            if 'clicks' in num_sample and 'impressions' in num_sample and num_sample['impressions'] > 0:
                num_sample['CTR'] = round(num_sample['clicks'] / num_sample['impressions'], 4)
            if 'ad_spend' in num_sample and 'clicks' in num_sample and num_sample['clicks'] > 0:
                num_sample['CPC'] = round(num_sample['ad_spend'] / num_sample['clicks'], 2)
            if 'ad_spend' in num_sample and 'conversions' in num_sample and num_sample['conversions'] > 0:
                num_sample['CPA'] = round(num_sample['ad_spend'] / num_sample['conversions'], 2)
            if 'revenue' in num_sample and 'ad_spend' in num_sample and num_sample['ad_spend'] > 0:
                num_sample['ROAS'] = round(num_sample['revenue'] / num_sample['ad_spend'], 2)

            row = {**cat_sample, **num_sample}
            synthetic_rows.append(row)

        synthetic_df = pd.DataFrame(synthetic_rows)

        # 추가 컬럼 (원본에 있으나 위에서 처리하지 않은 것)
        extra_cols = [c for c in df.columns if c not in synthetic_df.columns and c not in [date_col]]
        for col in extra_cols:
            if df[col].dtype == 'object':
                synthetic_df[col] = df[col].sample(num_rows, replace=True).values
            else:
                synthetic_df[col] = self.rng.normal(
                    df[col].mean(), max(df[col].std(), 0.01), size=num_rows
                )

        # 컬럼 순서 맞추기
        synthetic_df = synthetic_df[[c for c in df.columns if c in synthetic_df.columns]]

        print(f"  Generated: {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns")
        return synthetic_df

    # -----------------------------------------------------------------------
    # ROAS 기반 밸런싱 (고/저성과 균형)
    # -----------------------------------------------------------------------
    def balance_by_performance(self, df, target_col='ROAS', n_bins=5):
        """
        ROAS 분포를 균등하게 만들어 모델이 저/고성과 모두 잘 학습하도록 함.
        소수 구간을 오버샘플링하여 밸런싱.
        """
        print(f"\n[Phase 4-2] Balancing data by {target_col} distribution...")

        df_copy = df.copy()
        df_copy['perf_bin'] = pd.qcut(df_copy[target_col], q=n_bins, labels=False, duplicates='drop')

        bin_counts = df_copy['perf_bin'].value_counts()
        max_count = bin_counts.max()
        print(f"  Before balancing: {bin_counts.to_dict()}")

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

        print(f"  After balancing: {len(df)} -> {len(balanced)} rows")
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
        print(f"\n[Phase 4-2] Augmenting dataset: {len(df)} -> target {target_total}")

        if len(df) >= target_total:
            print("  Dataset already meets target. Skipping augmentation.")
            if balance:
                return self.balance_by_performance(df)
            return df

        need = target_total - len(df)
        synthetic = self._statistical_sampling(df, num_rows=need)

        augmented = pd.concat([df, synthetic], ignore_index=True)
        augmented['is_synthetic'] = [0] * len(df) + [1] * len(synthetic)

        print(f"  Augmented: {len(augmented)} rows (original: {len(df)}, synthetic: {len(synthetic)})")

        if balance:
            augmented = self.balance_by_performance(augmented)

        return augmented


# ===========================================================================
# 통합 파이프라인
# ===========================================================================
def run_phase4(ads_df, target_total=10000, seed=42):
    """Phase 4 전체 실행"""
    print("\n" + "=" * 70)
    print("[Phase 4] Competition + Synthetic Data Pipeline")
    print("=" * 70)

    # 4-1: 경쟁 환경 데이터
    comp = CompetitionDataCollector(seed=seed)
    df = comp.generate_competition_metrics(ads_df)

    # 4-2: 합성 데이터 보강
    synth = SyntheticDataGenerator(seed=seed)
    df = synth.augment_dataset(df, target_total=target_total, balance=True)

    print(f"\n[Phase 4] Final dataset: {len(df)} rows, {len(df.columns)} columns")
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
