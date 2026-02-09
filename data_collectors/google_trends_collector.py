# -*- coding: utf-8 -*-
"""
Phase 2-3: Google Trends 산업별 관심도 수집
- pytrends 패키지를 통한 Google Trends API 호출
- 산업별(Fintech, EdTech, Healthcare, SaaS, E-commerce) 키워드 관심도
- 국가별 + 월별 관심도 지수 수집
- fallback: API 불가 시 합리적인 추정 데이터 생성
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# pytrends (선택적)
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False


# ===========================================================================
# 산업별 검색 키워드 정의
# ===========================================================================
INDUSTRY_KEYWORDS = {
    'Fintech': {
        'primary': ['fintech', 'mobile banking', 'digital payment'],
        'secondary': ['cryptocurrency', 'neobank', 'buy now pay later'],
        'category': 7,  # Finance
    },
    'EdTech': {
        'primary': ['edtech', 'online learning', 'e-learning'],
        'secondary': ['online course', 'LMS', 'virtual classroom'],
        'category': 958,  # Online Education
    },
    'Healthcare': {
        'primary': ['telehealth', 'digital health', 'health app'],
        'secondary': ['telemedicine', 'health insurance', 'mental health app'],
        'category': 45,  # Health
    },
    'SaaS': {
        'primary': ['SaaS', 'cloud software', 'business software'],
        'secondary': ['CRM software', 'project management tool', 'automation software'],
        'category': 5,  # Computers & Electronics
    },
    'E-commerce': {
        'primary': ['online shopping', 'e-commerce', 'buy online'],
        'secondary': ['same day delivery', 'flash sale', 'discount code'],
        'category': 18,  # Shopping
    },
}

# 국가별 Google Trends 지역 코드
COUNTRY_GEO_MAP = {
    'USA': 'US',
    'UK': 'GB',
    'Germany': 'DE',
    'Canada': 'CA',
    'India': 'IN',
    'UAE': 'AE',
    'Australia': 'AU',
}


class GoogleTrendsCollector:
    """Google Trends에서 산업별 관심도를 수집하는 클래스"""

    def __init__(self, output_dir='./data/trends'):
        self.output_dir = output_dir
        self.trends_data = pd.DataFrame()
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Phase 2-3] Google Trends Collector initialized")
        print(f"  pytrends available: {PYTRENDS_AVAILABLE}")

    # -----------------------------------------------------------------------
    # pytrends를 통한 실제 데이터 수집
    # -----------------------------------------------------------------------
    def fetch_trends(self, timeframe='2022-01-01 2024-12-31', sleep_sec=5):
        """
        Google Trends에서 산업별/국가별 관심도 수집
        
        주의: Google Trends API에는 rate limit이 있음.
        너무 빠르게 호출하면 429 에러 발생. sleep_sec으로 조절.
        """
        if not PYTRENDS_AVAILABLE:
            print("[WARNING] pytrends 미설치. pip install pytrends")
            print("[INFO] Fallback 추정 데이터를 생성합니다.")
            return self._generate_fallback_trends()

        print(f"\n[Phase 2-3] Fetching Google Trends data...")
        print(f"  Timeframe: {timeframe}")
        pytrends = TrendReq(hl='en-US', tz=0, timeout=(10, 25))

        all_data = []

        for industry, config in INDUSTRY_KEYWORDS.items():
            keywords = config['primary'][:3]  # 최대 3개 키워드

            for country_name, geo in COUNTRY_GEO_MAP.items():
                print(f"  Fetching: {industry} / {country_name}...", end=' ')
                try:
                    pytrends.build_payload(
                        kw_list=keywords,
                        cat=config.get('category', 0),
                        timeframe=timeframe,
                        geo=geo,
                    )

                    # 시간별 관심도
                    interest = pytrends.interest_over_time()

                    if not interest.empty and 'isPartial' in interest.columns:
                        interest = interest.drop(columns=['isPartial'])

                    if not interest.empty:
                        # 키워드별 평균을 산업 관심도로
                        interest['trend_index'] = interest[keywords].mean(axis=1)
                        interest['industry'] = industry
                        interest['country'] = country_name

                        monthly = interest.resample('MS').agg({
                            'trend_index': 'mean',
                            'industry': 'first',
                            'country': 'first',
                        }).reset_index()
                        monthly.rename(columns={'index': 'date'}, inplace=True)

                        all_data.append(monthly)
                        print(f"OK ({len(monthly)} months)")
                    else:
                        print("No data")

                    time.sleep(sleep_sec)  # Rate limit 방지

                except Exception as e:
                    print(f"Error: {str(e)[:50]}")
                    time.sleep(sleep_sec * 2)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            # 정규화 (0-100 범위)
            df['trend_index'] = df.groupby(['industry', 'country'])['trend_index'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8) * 100
            )
            self.trends_data = df
            print(f"\n[Phase 2-3] Total trends records: {len(df)}")
            return df
        else:
            print("[Phase 2-3] No data from API, using fallback")
            return self._generate_fallback_trends()

    # -----------------------------------------------------------------------
    # Fallback 추정 데이터
    # -----------------------------------------------------------------------
    def _generate_fallback_trends(self):
        """
        Google Trends API 불가 시 합리적인 추정 관심도 데이터 생성.
        각 산업의 실제 성장 트렌드를 반영한 시뮬레이션.
        """
        print(f"\n[Phase 2-3] Generating estimated trend data...")

        # 산업별 기본 관심도 + 성장률 + 계절성
        industry_profiles = {
            'Fintech': {
                'base': 55, 'growth_rate': 0.015,  # 월 1.5% 성장
                'seasonality': {1: 1.1, 2: 0.95, 3: 1.15, 4: 1.1,  # 세금 시즌
                                5: 0.9, 6: 0.85, 7: 0.85, 8: 0.9,
                                9: 1.0, 10: 1.05, 11: 1.1, 12: 1.15},
            },
            'EdTech': {
                'base': 50, 'growth_rate': 0.012,
                'seasonality': {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9,
                                5: 0.7, 6: 0.6, 7: 0.65, 8: 1.3,  # Back to school
                                9: 1.4, 10: 1.1, 11: 1.0, 12: 0.8},
            },
            'Healthcare': {
                'base': 60, 'growth_rate': 0.01,
                'seasonality': {1: 1.2, 2: 1.0, 3: 0.95, 4: 0.9,
                                5: 0.85, 6: 0.85, 7: 0.9, 8: 0.9,
                                9: 1.0, 10: 1.15, 11: 1.2, 12: 1.1},  # Open enrollment
            },
            'SaaS': {
                'base': 45, 'growth_rate': 0.018,
                'seasonality': {1: 1.15, 2: 1.05, 3: 1.1, 4: 1.0,
                                5: 0.95, 6: 0.9, 7: 0.85, 8: 0.9,
                                9: 1.1, 10: 1.15, 11: 1.1, 12: 0.95},
            },
            'E-commerce': {
                'base': 65, 'growth_rate': 0.008,
                'seasonality': {1: 0.85, 2: 0.8, 3: 0.85, 4: 0.9,
                                5: 0.95, 6: 1.0, 7: 1.05, 8: 0.95,
                                9: 0.95, 10: 1.05, 11: 1.4, 12: 1.5},  # Q4 쇼핑
            },
        }

        # 국가별 디지털 성숙도 계수
        country_digital_factor = {
            'USA': 1.0, 'UK': 0.95, 'Germany': 0.85, 'Canada': 0.9,
            'India': 1.2, 'UAE': 0.8, 'Australia': 0.88,
        }

        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='MS')
        rows = []

        for industry, profile in industry_profiles.items():
            for country, digital_factor in country_digital_factor.items():
                for i, date in enumerate(dates):
                    month = date.month
                    seasonality = profile['seasonality'][month]
                    growth = (1 + profile['growth_rate']) ** i

                    base_trend = profile['base'] * growth * seasonality * digital_factor
                    noise = np.random.normal(0, 3)  # 약간의 랜덤 변동

                    trend_index = max(0, min(100, base_trend + noise))

                    rows.append({
                        'date': date,
                        'year': date.year,
                        'month': month,
                        'industry': industry,
                        'country': country,
                        'trend_index': round(trend_index, 2),
                    })

        df = pd.DataFrame(rows)
        self.trends_data = df
        print(f"  Generated {len(df)} trend records")
        print(f"  Industries: {df['industry'].nunique()}")
        print(f"  Countries: {df['country'].nunique()}")
        print(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")

        return df

    # -----------------------------------------------------------------------
    # 관련 검색어 수집 (보너스)
    # -----------------------------------------------------------------------
    def fetch_related_queries(self, industry='E-commerce', country='USA'):
        """특정 산업/국가의 관련 검색어 수집"""
        if not PYTRENDS_AVAILABLE:
            return None

        try:
            pytrends = TrendReq(hl='en-US', tz=0)
            keywords = INDUSTRY_KEYWORDS[industry]['primary'][:1]
            geo = COUNTRY_GEO_MAP[country]

            pytrends.build_payload(
                kw_list=keywords,
                timeframe='today 12-m',
                geo=geo,
            )

            related = pytrends.related_queries()
            return related
        except Exception as e:
            print(f"  Error: {e}")
            return None

    # -----------------------------------------------------------------------
    # 저장 및 조인
    # -----------------------------------------------------------------------
    def save(self, output_path=None):
        """트렌드 데이터 저장"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'industry_trends.csv')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.trends_data.to_csv(output_path, index=False)
        print(f"[Phase 2-3] Saved: {output_path} ({len(self.trends_data)} rows)")
        return output_path

    def enrich_ads_data(self, ads_df):
        """광고 데이터에 트렌드 관심도 조인"""
        if self.trends_data.empty:
            self._generate_fallback_trends()

        ads = ads_df.copy()
        ads['date'] = pd.to_datetime(ads['date'])
        ads['year'] = ads['date'].dt.year
        ads['month'] = ads['date'].dt.month

        # 트렌드 데이터에 year/month 보장
        trends = self.trends_data.copy()
        if 'year' not in trends.columns:
            trends['date'] = pd.to_datetime(trends['date'])
            trends['year'] = trends['date'].dt.year
            trends['month'] = trends['date'].dt.month

        enriched = ads.merge(
            trends[['industry', 'country', 'year', 'month', 'trend_index']],
            on=['industry', 'country', 'year', 'month'],
            how='left'
        )

        # 트렌드 변화율 (전월 대비)
        enriched = enriched.sort_values(['industry', 'country', 'year', 'month'])
        enriched['trend_momentum'] = enriched.groupby(
            ['industry', 'country']
        )['trend_index'].pct_change() * 100

        enriched['trend_momentum'] = enriched['trend_momentum'].fillna(0).clip(-50, 50)

        # 임시 컬럼 정리
        if 'year' not in ads_df.columns:
            enriched = enriched.drop(columns=['year'], errors='ignore')
        if 'month' not in ads_df.columns:
            enriched = enriched.drop(columns=['month'], errors='ignore')

        added_cols = [c for c in enriched.columns if c not in ads_df.columns]
        print(f"[Phase 2-3] Enriched ads data with {len(added_cols)} trend columns: {added_cols}")

        return enriched


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    collector = GoogleTrendsCollector(output_dir='./data/trends')

    # 트렌드 수집 (API 가능 시 실제 데이터, 불가 시 fallback)
    if PYTRENDS_AVAILABLE:
        trends_df = collector.fetch_trends(
            timeframe='2022-01-01 2024-12-31',
            sleep_sec=5
        )
    else:
        trends_df = collector._generate_fallback_trends()

    # 저장
    collector.save()

    # 광고 데이터 보강 예시
    if os.path.exists('global_ads_performance_dataset.csv'):
        ads_df = pd.read_csv('global_ads_performance_dataset.csv')
        enriched = collector.enrich_ads_data(ads_df)
        print(f"\nEnriched dataset shape: {enriched.shape}")
        enriched.to_csv('./data/ads_with_trends.csv', index=False)
