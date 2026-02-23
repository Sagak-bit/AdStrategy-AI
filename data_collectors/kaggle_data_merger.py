# -*- coding: utf-8 -*-
"""
Phase 1: Kaggle 유사 광고 데이터셋 검색/다운로드/병합
- Kaggle API를 통해 유사 데이터셋 자동 탐색
- 현재 스키마(14개 컬럼)에 맞춰 자동 매핑/병합
- 데이터 볼륨을 최소 10,000행 이상으로 확대
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kaggle API는 선택적 임포트
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


# ===========================================================================
# 현재 데이터셋의 스키마 정의
# ===========================================================================
TARGET_SCHEMA = {
    'date': 'datetime64[ns]',
    'platform': 'object',          # Google Ads, Meta Ads, TikTok Ads
    'campaign_type': 'object',     # Search, Video, Shopping, Display
    'industry': 'object',          # Fintech, EdTech, Healthcare, SaaS, E-commerce
    'country': 'object',           # UAE, UK, USA, Germany, Canada, India, Australia
    'impressions': 'float64',
    'clicks': 'float64',
    'CTR': 'float64',
    'CPC': 'float64',
    'ad_spend': 'float64',
    'conversions': 'float64',
    'CPA': 'float64',
    'revenue': 'float64',
    'ROAS': 'float64',
}

TARGET_COLUMNS = list(TARGET_SCHEMA.keys())

# 일반적인 컬럼명 매핑 사전 (다양한 데이터셋에서 사용되는 유사 이름)
COLUMN_MAPPING = {
    # date
    'date': 'date', 'Date': 'date', 'DATE': 'date',
    'reporting_date': 'date', 'campaign_date': 'date', 'day': 'date',
    'report_date': 'date', 'start_date': 'date',
    # platform
    'platform': 'platform', 'Platform': 'platform',
    'ad_platform': 'platform', 'channel': 'platform', 'Channel': 'platform',
    'media_source': 'platform', 'source': 'platform',
    'advertising_platform': 'platform', 'network': 'platform',
    # campaign_type
    'campaign_type': 'campaign_type', 'Campaign_Type': 'campaign_type',
    'campaign_objective': 'campaign_type', 'ad_type': 'campaign_type',
    'ad_format': 'campaign_type', 'objective': 'campaign_type',
    'campaign_goal': 'campaign_type', 'type': 'campaign_type',
    # industry
    'industry': 'industry', 'Industry': 'industry',
    'vertical': 'industry', 'sector': 'industry', 'category': 'industry',
    'business_type': 'industry', 'niche': 'industry',
    # country
    'country': 'country', 'Country': 'country', 'COUNTRY': 'country',
    'geo': 'country', 'region': 'country', 'location': 'country',
    'market': 'country', 'country_code': 'country',
    # impressions
    'impressions': 'impressions', 'Impressions': 'impressions',
    'imps': 'impressions', 'total_impressions': 'impressions',
    'Reached': 'impressions', 'reach': 'impressions',
    # clicks
    'clicks': 'clicks', 'Clicks': 'clicks',
    'total_clicks': 'clicks', 'link_clicks': 'clicks',
    'Website_Clicks': 'clicks',
    # CTR
    'CTR': 'CTR', 'ctr': 'CTR', 'click_through_rate': 'CTR',
    'click_rate': 'CTR',
    # CPC
    'CPC': 'CPC', 'cpc': 'CPC', 'cost_per_click': 'CPC',
    'avg_cpc': 'CPC', 'average_cpc': 'CPC',
    # ad_spend
    'ad_spend': 'ad_spend', 'Ad_Spend': 'ad_spend',
    'spend': 'ad_spend', 'Spend': 'ad_spend', 'cost': 'ad_spend',
    'Cost': 'ad_spend', 'total_spend': 'ad_spend', 'budget': 'ad_spend',
    'Amount_Spent': 'ad_spend', 'amount_spent': 'ad_spend',
    'Spent': 'ad_spend', 'total_cost': 'ad_spend',
    # conversions
    'conversions': 'conversions', 'Conversions': 'conversions',
    'total_conversions': 'conversions', 'purchases': 'conversions',
    'Approved_Conversion': 'conversions', 'leads': 'conversions',
    'Total_Conversion': 'conversions',
    # CPA
    'CPA': 'CPA', 'cpa': 'CPA', 'cost_per_acquisition': 'CPA',
    'cost_per_conversion': 'CPA', 'cost_per_action': 'CPA',
    'CPL': 'CPA', 'cost_per_lead': 'CPA',
    # revenue
    'revenue': 'revenue', 'Revenue': 'revenue',
    'total_revenue': 'revenue', 'sales': 'revenue',
    'income': 'revenue', 'conversion_value': 'revenue',
    'value': 'revenue',
    # ROAS
    'ROAS': 'ROAS', 'roas': 'ROAS',
    'return_on_ad_spend': 'ROAS', 'roi': 'ROAS',
}

# 추천 Kaggle 데이터셋 목록
RECOMMENDED_DATASETS = [
    {
        'name': 'sinderpreet/analyze-the-marketing-spending',
        'description': 'Marketing Spending Analysis - 멀티채널 광고 지출/성과',
    },
    {
        'name': 'manishkc06/digital-marketing-campaign-dataset',
        'description': 'Digital Marketing Campaign - 디지털 마케팅 캠페인 데이터',
    },
    {
        'name': 'agirlhasnoname/online-advertising-effectiveness',
        'description': 'Online Advertising Effectiveness - 온라인 광고 효과',
    },
    {
        'name': 'fayomi/advertising',
        'description': 'Advertising Dataset - 광고 기본 데이터',
    },
    {
        'name': 'jsonk11/social-media-advertising-dataset',
        'description': 'Social Media Advertising - 소셜미디어 광고 (creative type 포함)',
    },
    {
        'name': 'madislemsalu/facebook-ad-campaign',
        'description': 'Facebook Ad Campaign - Meta 광고 세부 데이터',
    },
]


class KaggleDataMerger:
    """Kaggle 데이터셋을 검색, 다운로드, 매핑, 병합하는 클래스"""

    def __init__(self, base_data_path='global_ads_performance_dataset.csv',
                 download_dir='./data/kaggle_downloads'):
        self.base_data_path = base_data_path
        self.download_dir = download_dir
        self.base_df = None
        self.merged_datasets = []

        os.makedirs(download_dir, exist_ok=True)

        # 기본 데이터 로드
        if os.path.exists(base_data_path):
            self.base_df = pd.read_csv(base_data_path)
            self.base_df['date'] = pd.to_datetime(self.base_df['date'])
            print(f"[Phase 1] Base dataset loaded: {len(self.base_df)} rows, {len(self.base_df.columns)} columns")
        else:
            print(f"[WARNING] Base dataset not found: {base_data_path}")

    # -----------------------------------------------------------------------
    # Kaggle API 연동
    # -----------------------------------------------------------------------
    def search_kaggle_datasets(self, query='advertising campaign performance',
                               max_results=10):
        """Kaggle에서 유사 광고 데이터셋 검색"""
        if not KAGGLE_AVAILABLE:
            print("[WARNING] Kaggle API not installed. pip install kaggle")
            print("[INFO] Showing recommended datasets instead:")
            for ds in RECOMMENDED_DATASETS:
                print(f"  - {ds['name']}: {ds['description']}")
            return RECOMMENDED_DATASETS

        api = KaggleApi()
        api.authenticate()

        print(f"\n[Phase 1] Searching Kaggle for: '{query}'")
        datasets = api.dataset_list(search=query, sort_by='relevance',
                                    max_size=50 * 1024 * 1024)  # 50MB 이하

        results = []
        for ds in datasets[:max_results]:
            info = {
                'name': str(ds),
                'title': ds.title if hasattr(ds, 'title') else str(ds),
                'size': ds.totalBytes if hasattr(ds, 'totalBytes') else 0,
                'downloads': ds.downloadCount if hasattr(ds, 'downloadCount') else 0,
            }
            results.append(info)
            print(f"  [{len(results)}] {info['name']}")
            print(f"      Title: {info['title']}, Downloads: {info['downloads']}")

        return results

    def download_kaggle_dataset(self, dataset_name):
        """Kaggle 데이터셋 다운로드"""
        if not KAGGLE_AVAILABLE:
            print(f"[ERROR] Kaggle API required. Run: pip install kaggle")
            return None

        api = KaggleApi()
        api.authenticate()

        target_dir = os.path.join(self.download_dir, dataset_name.replace('/', '_'))
        os.makedirs(target_dir, exist_ok=True)

        print(f"\n[Phase 1] Downloading: {dataset_name}")
        api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
        print(f"  -> Saved to: {target_dir}")

        # CSV 파일 찾기
        csv_files = []
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))

        print(f"  -> Found {len(csv_files)} CSV file(s)")
        return csv_files

    # -----------------------------------------------------------------------
    # 자동 컬럼 매핑 및 변환
    # -----------------------------------------------------------------------
    def auto_map_columns(self, df):
        """
        외부 데이터셋의 컬럼을 현재 스키마에 자동 매핑
        Returns: (mapped_df, mapping_report)
        """
        mapping = {}
        unmapped = []

        for col in df.columns:
            col_clean = col.strip()
            if col_clean in COLUMN_MAPPING:
                target = COLUMN_MAPPING[col_clean]
                if target not in mapping.values():
                    mapping[col_clean] = target
            else:
                # 퍼지 매칭: 소문자 비교
                col_lower = col_clean.lower().replace(' ', '_').replace('-', '_')
                matched = False
                for src, tgt in COLUMN_MAPPING.items():
                    if src.lower().replace(' ', '_') == col_lower:
                        if tgt not in mapping.values():
                            mapping[col_clean] = tgt
                            matched = True
                            break
                if not matched:
                    unmapped.append(col_clean)

        report = {
            'total_columns': len(df.columns),
            'mapped': len(mapping),
            'unmapped': unmapped,
            'mapping': mapping,
            'coverage': len(mapping) / len(TARGET_COLUMNS)
        }

        return mapping, report

    def transform_dataset(self, df, mapping):
        """매핑된 컬럼으로 데이터 변환 및 파생 컬럼 계산"""
        # 매핑 적용
        df_mapped = df.rename(columns=mapping)

        # 존재하는 타겟 컬럼만 선택
        available = [c for c in TARGET_COLUMNS if c in df_mapped.columns]
        df_result = df_mapped[available].copy()

        # 날짜 변환
        if 'date' in df_result.columns:
            df_result['date'] = pd.to_datetime(df_result['date'], errors='coerce')
            df_result = df_result.dropna(subset=['date'])

        # 파생 컬럼 계산 (누락된 경우)
        if 'CTR' not in df_result.columns and 'clicks' in df_result.columns and 'impressions' in df_result.columns:
            df_result['CTR'] = df_result['clicks'] / df_result['impressions'].replace(0, np.nan)

        if 'CPC' not in df_result.columns and 'ad_spend' in df_result.columns and 'clicks' in df_result.columns:
            df_result['CPC'] = df_result['ad_spend'] / df_result['clicks'].replace(0, np.nan)

        if 'CPA' not in df_result.columns and 'ad_spend' in df_result.columns and 'conversions' in df_result.columns:
            df_result['CPA'] = df_result['ad_spend'] / df_result['conversions'].replace(0, np.nan)

        if 'ROAS' not in df_result.columns and 'revenue' in df_result.columns and 'ad_spend' in df_result.columns:
            df_result['ROAS'] = df_result['revenue'] / df_result['ad_spend'].replace(0, np.nan)

        if 'revenue' not in df_result.columns and 'ROAS' in df_result.columns and 'ad_spend' in df_result.columns:
            df_result['revenue'] = df_result['ROAS'] * df_result['ad_spend']

        # 플랫폼 이름 정규화
        if 'platform' in df_result.columns:
            platform_norm = {
                'google': 'Google Ads', 'google ads': 'Google Ads', 'google_ads': 'Google Ads',
                'adwords': 'Google Ads', 'sem': 'Google Ads',
                'facebook': 'Meta Ads', 'meta': 'Meta Ads', 'meta ads': 'Meta Ads',
                'instagram': 'Meta Ads', 'fb': 'Meta Ads', 'facebook ads': 'Meta Ads',
                'tiktok': 'TikTok Ads', 'tiktok ads': 'TikTok Ads', 'tik tok': 'TikTok Ads',
            }
            df_result['platform'] = df_result['platform'].astype(str).str.lower().str.strip().map(
                lambda x: platform_norm.get(x, x)
            )
            # 알려진 플랫폼만 유지
            valid_platforms = ['Google Ads', 'Meta Ads', 'TikTok Ads']
            df_result = df_result[df_result['platform'].isin(valid_platforms)]

        # 캠페인 유형 정규화
        if 'campaign_type' in df_result.columns:
            type_norm = {
                'search': 'Search', 'sem': 'Search', 'text': 'Search',
                'video': 'Video', 'youtube': 'Video', 'reels': 'Video',
                'shopping': 'Shopping', 'catalog': 'Shopping', 'product': 'Shopping',
                'display': 'Display', 'banner': 'Display', 'image': 'Display',
            }
            df_result['campaign_type'] = df_result['campaign_type'].astype(str).str.lower().str.strip().map(
                lambda x: type_norm.get(x, x)
            )
            valid_types = ['Search', 'Video', 'Shopping', 'Display']
            df_result = df_result[df_result['campaign_type'].isin(valid_types)]

        # 산업 정규화
        if 'industry' in df_result.columns:
            industry_norm = {
                'fintech': 'Fintech', 'finance': 'Fintech', 'financial': 'Fintech', 'banking': 'Fintech',
                'edtech': 'EdTech', 'education': 'EdTech', 'e-learning': 'EdTech', 'online education': 'EdTech',
                'healthcare': 'Healthcare', 'health': 'Healthcare', 'medical': 'Healthcare', 'pharma': 'Healthcare',
                'saas': 'SaaS', 'software': 'SaaS', 'tech': 'SaaS', 'technology': 'SaaS', 'b2b': 'SaaS',
                'e-commerce': 'E-commerce', 'ecommerce': 'E-commerce', 'retail': 'E-commerce', 'shopping': 'E-commerce',
            }
            df_result['industry'] = df_result['industry'].astype(str).str.lower().str.strip().map(
                lambda x: industry_norm.get(x, x)
            )
            valid_industries = ['Fintech', 'EdTech', 'Healthcare', 'SaaS', 'E-commerce']
            df_result = df_result[df_result['industry'].isin(valid_industries)]

        # 국가 정규화
        if 'country' in df_result.columns:
            country_norm = {
                'us': 'USA', 'usa': 'USA', 'united states': 'USA', 'america': 'USA',
                'uk': 'UK', 'united kingdom': 'UK', 'britain': 'UK', 'england': 'UK',
                'de': 'Germany', 'germany': 'Germany', 'deutschland': 'Germany',
                'in': 'India', 'india': 'India',
                'ca': 'Canada', 'canada': 'Canada',
                'ae': 'UAE', 'uae': 'UAE', 'united arab emirates': 'UAE',
                'au': 'Australia', 'australia': 'Australia',
            }
            df_result['country'] = df_result['country'].astype(str).str.lower().str.strip().map(
                lambda x: country_norm.get(x, x)
            )
            valid_countries = ['UAE', 'UK', 'USA', 'Germany', 'Canada', 'India', 'Australia']
            df_result = df_result[df_result['country'].isin(valid_countries)]

        # 숫자 컬럼 음수 제거
        numeric_cols = ['impressions', 'clicks', 'CTR', 'CPC', 'ad_spend',
                        'conversions', 'CPA', 'revenue', 'ROAS']
        for col in numeric_cols:
            if col in df_result.columns:
                df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
                df_result = df_result[df_result[col] >= 0]

        # NaN/Inf 제거
        df_result = df_result.replace([np.inf, -np.inf], np.nan)
        df_result = df_result.dropna(subset=[c for c in ['ad_spend', 'ROAS'] if c in df_result.columns])

        return df_result

    # -----------------------------------------------------------------------
    # 병합
    # -----------------------------------------------------------------------
    def merge_dataset(self, external_df, source_name='unknown'):
        """외부 데이터셋을 기본 데이터에 병합"""
        if self.base_df is None:
            print("[ERROR] Base dataset not loaded")
            return None

        # 자동 매핑
        mapping, report = self.auto_map_columns(external_df)
        print(f"\n  [Mapping Report - {source_name}]")
        print(f"    Total columns: {report['total_columns']}")
        print(f"    Mapped: {report['mapped']}/{len(TARGET_COLUMNS)} ({report['coverage']:.0%})")
        print(f"    Mapping: {json.dumps(report['mapping'], indent=6)}")
        if report['unmapped']:
            print(f"    Unmapped: {report['unmapped'][:10]}...")

        # 최소 필수 컬럼 체크
        mapped_targets = set(report['mapping'].values())
        required_min = {'ad_spend'}  # 최소한 광고비는 있어야 함
        if not required_min.issubset(mapped_targets):
            print(f"    [SKIP] Missing required columns: {required_min - mapped_targets}")
            return None

        # 변환
        transformed = self.transform_dataset(external_df, mapping)
        if len(transformed) == 0:
            print(f"    [SKIP] No valid rows after transformation")
            return None

        print(f"    Valid rows after transform: {len(transformed)}")

        # 기본 데이터와 동일한 컬럼으로 정렬
        for col in TARGET_COLUMNS:
            if col not in transformed.columns:
                transformed[col] = np.nan

        transformed = transformed[TARGET_COLUMNS]

        # 기본 데이터에 추가
        original_len = len(self.base_df)
        self.base_df = pd.concat([self.base_df, transformed], ignore_index=True)

        # 중복 제거
        self.base_df = self.base_df.drop_duplicates(
            subset=['date', 'platform', 'campaign_type', 'industry', 'country', 'ad_spend'],
            keep='first'
        )

        added = len(self.base_df) - original_len
        print(f"    Added {added} new rows (total: {len(self.base_df)})")

        self.merged_datasets.append({
            'source': source_name,
            'rows_added': added,
            'timestamp': datetime.now().isoformat()
        })

        return transformed

    def merge_csv_file(self, csv_path, source_name=None):
        """CSV 파일 로드 후 병합"""
        if source_name is None:
            source_name = os.path.basename(csv_path)

        print(f"\n[Phase 1] Processing: {source_name}")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)}")
            return self.merge_dataset(df, source_name)
        except Exception as e:
            print(f"  [ERROR] Failed to load {csv_path}: {e}")
            return None

    # -----------------------------------------------------------------------
    # 시계열 확장 (과거 데이터 합성)
    # -----------------------------------------------------------------------
    def extend_time_range(self, target_years=(2022, 2023)):
        """
        기존 2024 데이터의 분포를 기반으로 2022-2023 데이터 합성.
        계절성 + 노이즈를 반영하여 현실적인 과거 데이터 생성.
        """
        if self.base_df is None:
            return

        print(f"\n[Phase 1] Extending time range to {target_years}")
        original_len = len(self.base_df)

        extended_rows = []
        for year in target_years:
            for _, row in self.base_df.iterrows():
                new_row = row.copy()

                # 날짜 변경 (같은 월/일, 다른 연도)
                try:
                    orig_date = pd.to_datetime(row['date'])
                    new_date = orig_date.replace(year=year)
                    new_row['date'] = new_date
                except (ValueError, AttributeError):
                    continue

                # 연도별 시장 성숙도 반영 (과거일수록 약간 낮은 성과)
                year_factor = 1.0 - (2024 - year) * 0.05  # 연 5%씩 성장
                noise = np.random.normal(1.0, 0.1)  # 10% 노이즈

                for col in ['impressions', 'clicks', 'conversions', 'revenue', 'ad_spend']:
                    if pd.notna(new_row.get(col)):
                        new_row[col] = max(0, new_row[col] * year_factor * noise)

                # 파생 지표 재계산
                if new_row.get('clicks', 0) > 0 and new_row.get('impressions', 0) > 0:
                    new_row['CTR'] = new_row['clicks'] / new_row['impressions']
                if new_row.get('clicks', 0) > 0 and new_row.get('ad_spend', 0) > 0:
                    new_row['CPC'] = new_row['ad_spend'] / new_row['clicks']
                if new_row.get('conversions', 0) > 0 and new_row.get('ad_spend', 0) > 0:
                    new_row['CPA'] = new_row['ad_spend'] / new_row['conversions']
                if new_row.get('ad_spend', 0) > 0 and new_row.get('revenue', 0) > 0:
                    new_row['ROAS'] = new_row['revenue'] / new_row['ad_spend']

                extended_rows.append(new_row)

        if extended_rows:
            extended_df = pd.DataFrame(extended_rows)
            self.base_df = pd.concat([self.base_df, extended_df], ignore_index=True)
            added = len(self.base_df) - original_len
            print(f"  Added {added} synthetic historical rows")
            print(f"  Total dataset: {len(self.base_df)} rows")
            print(f"  Date range: {self.base_df['date'].min()} ~ {self.base_df['date'].max()}")

    # -----------------------------------------------------------------------
    # 저장 및 리포트
    # -----------------------------------------------------------------------
    def save_merged(self, output_path='data/merged_ads_dataset.csv'):
        """병합된 데이터 저장"""
        if self.base_df is None:
            return

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        self.base_df.to_csv(output_path, index=False)
        print(f"\n[Phase 1] Merged dataset saved: {output_path}")
        print(f"  Total rows: {len(self.base_df)}")
        print(f"  Columns: {list(self.base_df.columns)}")

    def get_merge_report(self):
        """병합 리포트 생성"""
        if self.base_df is None:
            return {}

        report = {
            'total_rows': len(self.base_df),
            'date_range': {
                'min': str(self.base_df['date'].min()),
                'max': str(self.base_df['date'].max()),
            },
            'platforms': self.base_df['platform'].value_counts().to_dict(),
            'industries': self.base_df['industry'].value_counts().to_dict(),
            'countries': self.base_df['country'].value_counts().to_dict(),
            'campaign_types': self.base_df['campaign_type'].value_counts().to_dict(),
            'merged_sources': self.merged_datasets,
            'null_counts': self.base_df.isnull().sum().to_dict(),
        }

        print("\n" + "=" * 70)
        print("[Phase 1 MERGE REPORT]")
        print("=" * 70)
        print(f"Total rows: {report['total_rows']}")
        print(f"Date range: {report['date_range']['min']} ~ {report['date_range']['max']}")
        print(f"\nPlatform distribution:")
        for k, v in report['platforms'].items():
            print(f"  {k}: {v} ({v/report['total_rows']*100:.1f}%)")
        print(f"\nIndustry distribution:")
        for k, v in report['industries'].items():
            print(f"  {k}: {v} ({v/report['total_rows']*100:.1f}%)")
        print(f"\nMerged sources: {len(report['merged_sources'])}")
        for src in report['merged_sources']:
            print(f"  - {src['source']}: +{src['rows_added']} rows")

        return report


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    merger = KaggleDataMerger(
        base_data_path='global_ads_performance_dataset.csv',
        download_dir='./data/kaggle_downloads'
    )

    # 1. Kaggle 데이터셋 검색 (API 설치 시)
    datasets = merger.search_kaggle_datasets('advertising campaign performance')

    # 2. 로컬 CSV 파일이 있다면 병합
    local_csvs = [
        # 수동으로 다운로드한 CSV 파일 경로를 여기에 추가
        # './data/kaggle_downloads/facebook_ads.csv',
        # './data/kaggle_downloads/google_ads_campaign.csv',
    ]
    for csv_path in local_csvs:
        if os.path.exists(csv_path):
            merger.merge_csv_file(csv_path)

    # 3. 시계열 확장 (2022-2023 데이터 합성)
    merger.extend_time_range(target_years=(2022, 2023))

    # 4. 저장
    merger.save_merged('data/merged_ads_dataset.csv')

    # 5. 리포트
    merger.get_merge_report()
