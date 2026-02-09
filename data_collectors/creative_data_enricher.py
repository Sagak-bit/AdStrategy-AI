# -*- coding: utf-8 -*-
"""
Phase 3: 크리에이티브/타겟팅 메타데이터 보강
- 기존 데이터에 크리에이티브(소재) 메타데이터 시뮬레이션 추가
- 타겟 오디언스 정보 추가
- 랜딩페이지/전환 퍼널 데이터 추가
- 실제 데이터가 없을 경우 플랫폼/산업별 합리적 분포 기반 시뮬레이션

실제 운영 데이터가 있다면 이 모듈 대신 직접 데이터를 사용하면 됨.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


# ===========================================================================
# 플랫폼/산업별 크리에이티브 분포 프로파일
# ===========================================================================

# 플랫폼별 광고 포맷 분포
PLATFORM_AD_FORMAT = {
    'Google Ads': {
        'Search': {'text': 0.85, 'responsive_search': 0.15},
        'Video': {'video_short': 0.3, 'video_long': 0.5, 'bumper': 0.2},
        'Shopping': {'product_listing': 0.7, 'showcase': 0.3},
        'Display': {'image': 0.5, 'responsive_display': 0.3, 'html5': 0.2},
    },
    'Meta Ads': {
        'Search': {'image': 0.4, 'carousel': 0.3, 'video_short': 0.3},
        'Video': {'video_short': 0.5, 'video_long': 0.2, 'reels': 0.3},
        'Shopping': {'carousel': 0.4, 'collection': 0.3, 'dynamic_product': 0.3},
        'Display': {'image': 0.5, 'carousel': 0.3, 'video_short': 0.2},
    },
    'TikTok Ads': {
        'Search': {'video_short': 0.7, 'spark_ad': 0.3},
        'Video': {'video_short': 0.6, 'video_long': 0.15, 'spark_ad': 0.25},
        'Shopping': {'video_short': 0.5, 'live_shopping': 0.2, 'product_card': 0.3},
        'Display': {'video_short': 0.6, 'image': 0.2, 'spark_ad': 0.2},
    },
}

# 광고 포맷별 동영상 길이 (초)
VIDEO_LENGTH_MAP = {
    'video_short': (6, 30),     # 6-30초
    'video_long': (31, 120),    # 31-120초
    'bumper': (5, 6),           # 6초 범퍼
    'reels': (15, 60),          # 릴스
    'spark_ad': (10, 60),       # 스파크 광고
    'live_shopping': (60, 300), # 라이브
}

# CTA 버튼 유형
CTA_TYPES = {
    'Search': ['Learn More', 'Sign Up', 'Get Quote', 'Buy Now', 'Contact Us'],
    'Video': ['Learn More', 'Watch More', 'Sign Up', 'Shop Now', 'Download'],
    'Shopping': ['Shop Now', 'Buy Now', 'Add to Cart', 'View Deal', 'Get Offer'],
    'Display': ['Learn More', 'Sign Up', 'Shop Now', 'Get Started', 'Try Free'],
}

# 산업별 헤드라인 길이 분포 (평균, 표준편차)
HEADLINE_LENGTH = {
    'Fintech': (45, 12),
    'EdTech': (50, 15),
    'Healthcare': (40, 10),
    'SaaS': (48, 13),
    'E-commerce': (35, 10),
}

# 타겟 연령대 분포 (산업별)
AGE_GROUP_DIST = {
    'Fintech': {'18-24': 0.15, '25-34': 0.35, '35-44': 0.25, '45-54': 0.15, '55+': 0.10},
    'EdTech': {'18-24': 0.35, '25-34': 0.30, '35-44': 0.20, '45-54': 0.10, '55+': 0.05},
    'Healthcare': {'18-24': 0.10, '25-34': 0.20, '35-44': 0.25, '45-54': 0.25, '55+': 0.20},
    'SaaS': {'18-24': 0.10, '25-34': 0.35, '35-44': 0.30, '45-54': 0.20, '55+': 0.05},
    'E-commerce': {'18-24': 0.25, '25-34': 0.30, '35-44': 0.25, '45-54': 0.12, '55+': 0.08},
}

# 타겟 성별 분포 (산업별)
GENDER_DIST = {
    'Fintech': {'Male': 0.40, 'Female': 0.30, 'All': 0.30},
    'EdTech': {'Male': 0.25, 'Female': 0.30, 'All': 0.45},
    'Healthcare': {'Male': 0.20, 'Female': 0.35, 'All': 0.45},
    'SaaS': {'Male': 0.40, 'Female': 0.25, 'All': 0.35},
    'E-commerce': {'Male': 0.25, 'Female': 0.35, 'All': 0.40},
}

# 리타겟팅 비율 (산업별)
RETARGETING_RATE = {
    'Fintech': 0.30,
    'EdTech': 0.25,
    'Healthcare': 0.20,
    'SaaS': 0.40,
    'E-commerce': 0.45,
}

# 유사 타겟(Lookalike) 사용 비율
LOOKALIKE_RATE = {
    'Fintech': 0.35,
    'EdTech': 0.30,
    'Healthcare': 0.25,
    'SaaS': 0.40,
    'E-commerce': 0.50,
}


class CreativeDataEnricher:
    """크리에이티브/타겟팅/퍼널 메타데이터를 기존 데이터에 추가"""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        print("[Phase 3] Creative Data Enricher initialized")

    # -----------------------------------------------------------------------
    # 크리에이티브 메타데이터 생성
    # -----------------------------------------------------------------------
    def generate_creative_metadata(self, row):
        """단일 행에 대한 크리에이티브 메타데이터 생성"""
        platform = row.get('platform', 'Google Ads')
        campaign_type = row.get('campaign_type', 'Search')
        industry = row.get('industry', 'E-commerce')

        # 1. 광고 포맷
        format_dist = PLATFORM_AD_FORMAT.get(platform, {}).get(campaign_type, {'image': 1.0})
        formats = list(format_dist.keys())
        probs = list(format_dist.values())
        ad_format = self.rng.choice(formats, p=probs)

        # 2. 동영상 길이 (해당 시)
        is_video = 'video' in ad_format or ad_format in ['reels', 'spark_ad', 'live_shopping', 'bumper']
        if is_video and ad_format in VIDEO_LENGTH_MAP:
            min_len, max_len = VIDEO_LENGTH_MAP[ad_format]
            video_length = self.rng.randint(min_len, max_len + 1)
        else:
            video_length = 0

        # 3. CTA 유형
        cta_options = CTA_TYPES.get(campaign_type, ['Learn More'])
        cta_type = self.rng.choice(cta_options)
        has_cta = 1  # 대부분 있음

        # 4. 헤드라인 길이
        h_mean, h_std = HEADLINE_LENGTH.get(industry, (42, 12))
        headline_length = max(10, int(self.rng.normal(h_mean, h_std)))

        # 5. 카피 감성 (0: 부정, 0.5: 중립, 1: 긍정) - 대부분 긍정
        copy_sentiment = min(1.0, max(0.0, self.rng.beta(5, 2)))

        # 6. 표시 상품 수 (Shopping 캠페인에서 주로)
        if campaign_type == 'Shopping':
            num_products = self.rng.choice([1, 2, 3, 4, 6, 8], p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
        else:
            num_products = self.rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])

        return {
            'ad_format': ad_format,
            'is_video': int(is_video),
            'video_length_sec': video_length,
            'cta_type': cta_type,
            'has_cta': has_cta,
            'headline_length': headline_length,
            'copy_sentiment': round(copy_sentiment, 3),
            'num_products_shown': num_products,
        }

    # -----------------------------------------------------------------------
    # 타겟 오디언스 데이터 생성
    # -----------------------------------------------------------------------
    def generate_audience_metadata(self, row):
        """단일 행에 대한 타겟 오디언스 메타데이터 생성"""
        industry = row.get('industry', 'E-commerce')
        ad_spend = row.get('ad_spend', 5000)

        # 1. 타겟 연령대
        age_dist = AGE_GROUP_DIST.get(industry, AGE_GROUP_DIST['E-commerce'])
        target_age_group = self.rng.choice(list(age_dist.keys()), p=list(age_dist.values()))

        # 2. 타겟 성별
        gender_dist = GENDER_DIST.get(industry, GENDER_DIST['E-commerce'])
        target_gender = self.rng.choice(list(gender_dist.keys()), p=list(gender_dist.values()))

        # 3. 오디언스 크기 (광고비에 비례, 약간의 변동)
        base_audience = ad_spend * self.rng.uniform(100, 500)
        audience_size = int(max(1000, base_audience))

        # 4. 리타겟팅 여부
        retarget_rate = RETARGETING_RATE.get(industry, 0.30)
        is_retargeting = int(self.rng.random() < retarget_rate)

        # 5. 유사 타겟 사용 여부
        lookalike_rate = LOOKALIKE_RATE.get(industry, 0.30)
        is_lookalike = int(self.rng.random() < lookalike_rate)

        # 6. 관심사 카테고리 (산업별 대표)
        interest_map = {
            'Fintech': ['Finance', 'Investment', 'Banking', 'Crypto', 'Savings'],
            'EdTech': ['Education', 'Career', 'Skills', 'Certification', 'Students'],
            'Healthcare': ['Health', 'Fitness', 'Wellness', 'Medical', 'Insurance'],
            'SaaS': ['Business', 'Technology', 'Productivity', 'Enterprise', 'Startups'],
            'E-commerce': ['Shopping', 'Fashion', 'Electronics', 'Home', 'Lifestyle'],
        }
        interests = interest_map.get(industry, ['General'])
        target_interest = self.rng.choice(interests)

        return {
            'target_age_group': target_age_group,
            'target_gender': target_gender,
            'audience_size': audience_size,
            'is_retargeting': is_retargeting,
            'is_lookalike': is_lookalike,
            'target_interest': target_interest,
        }

    # -----------------------------------------------------------------------
    # 랜딩페이지/퍼널 데이터 생성
    # -----------------------------------------------------------------------
    def generate_funnel_metadata(self, row):
        """단일 행에 대한 랜딩페이지/전환 퍼널 메타데이터 생성"""
        platform = row.get('platform', 'Google Ads')
        industry = row.get('industry', 'E-commerce')
        roas = row.get('ROAS', 5.0)

        # 1. 랜딩페이지 로딩 시간 (초)
        # ROAS가 높을수록 로딩이 빠른 경향 (잘 최적화된 사이트)
        base_load = max(0.5, 4.0 - roas * 0.1)
        load_time = max(0.5, self.rng.normal(base_load, 0.8))

        # 2. 이탈률 (bounce rate, %)
        # ROAS 높으면 이탈률 낮음
        base_bounce = max(20, 65 - roas * 2)
        bounce_rate = max(10, min(95, self.rng.normal(base_bounce, 10)))

        # 3. 평균 세션 시간 (초)
        session_map = {
            'Fintech': (90, 30), 'EdTech': (150, 50), 'Healthcare': (120, 40),
            'SaaS': (180, 60), 'E-commerce': (100, 35),
        }
        s_mean, s_std = session_map.get(industry, (120, 40))
        avg_session_duration = max(10, self.rng.normal(s_mean, s_std))

        # 4. 전환 퍼널 단계 수
        funnel_map = {
            'Fintech': [2, 3, 4, 5], 'EdTech': [1, 2, 3],
            'Healthcare': [2, 3, 4], 'SaaS': [2, 3, 4, 5],
            'E-commerce': [1, 2, 3],
        }
        funnel_options = funnel_map.get(industry, [2, 3])
        funnel_steps = self.rng.choice(funnel_options)

        return {
            'landing_page_load_time': round(load_time, 2),
            'bounce_rate': round(bounce_rate, 1),
            'avg_session_duration': round(avg_session_duration, 1),
            'funnel_steps': funnel_steps,
        }

    # -----------------------------------------------------------------------
    # ROAS에 대한 크리에이티브 영향 시뮬레이션
    # -----------------------------------------------------------------------
    def apply_creative_impact(self, row, creative_meta, audience_meta, funnel_meta):
        """
        크리에이티브/타겟팅 특성에 따른 ROAS 보정 계수를 계산.
        실제 인과 관계를 시뮬레이션하여 모델이 학습할 수 있도록 함.
        """
        impact = 1.0

        # 비디오가 TikTok에서 더 효과적
        if creative_meta['is_video'] and row.get('platform') == 'TikTok Ads':
            impact *= 1.08

        # 짧은 비디오(15-30초)가 최적
        vlen = creative_meta['video_length_sec']
        if 15 <= vlen <= 30:
            impact *= 1.05
        elif vlen > 60:
            impact *= 0.95

        # 리타겟팅은 ROAS 높임
        if audience_meta['is_retargeting']:
            impact *= 1.25

        # 유사 타겟도 약간 도움
        if audience_meta['is_lookalike']:
            impact *= 1.10

        # 로딩 속도 영향
        if funnel_meta['landing_page_load_time'] > 3.0:
            impact *= 0.90
        elif funnel_meta['landing_page_load_time'] < 1.5:
            impact *= 1.05

        # 이탈률 영향
        if funnel_meta['bounce_rate'] > 70:
            impact *= 0.85
        elif funnel_meta['bounce_rate'] < 30:
            impact *= 1.10

        # 카피 감성 영향
        if creative_meta['copy_sentiment'] > 0.7:
            impact *= 1.03

        return round(impact, 4)

    # -----------------------------------------------------------------------
    # 전체 데이터 보강
    # -----------------------------------------------------------------------
    def enrich_ads_data(self, ads_df, apply_impact=True):
        """
        광고 데이터에 크리에이티브/타겟팅/퍼널 메타데이터 추가

        Parameters:
        -----------
        ads_df : pd.DataFrame
        apply_impact : bool
            True면 크리에이티브 영향을 반영하여 ROAS 등 지표를 약간 보정
        """
        print(f"\n[Phase 3] Enriching ads data with creative/targeting metadata...")
        print(f"  Input: {len(ads_df)} rows")

        creative_cols = []
        audience_cols = []
        funnel_cols = []
        impact_factors = []

        for idx, row in ads_df.iterrows():
            creative = self.generate_creative_metadata(row)
            audience = self.generate_audience_metadata(row)
            funnel = self.generate_funnel_metadata(row)

            creative_cols.append(creative)
            audience_cols.append(audience)
            funnel_cols.append(funnel)

            if apply_impact:
                impact = self.apply_creative_impact(row, creative, audience, funnel)
                impact_factors.append(impact)

        # 크리에이티브 메타데이터 추가
        creative_df = pd.DataFrame(creative_cols)
        audience_df = pd.DataFrame(audience_cols)
        funnel_df = pd.DataFrame(funnel_cols)

        enriched = pd.concat([
            ads_df.reset_index(drop=True),
            creative_df,
            audience_df,
            funnel_df,
        ], axis=1)

        # ROAS 보정 (크리에이티브 영향 반영)
        if apply_impact and impact_factors:
            enriched['creative_impact_factor'] = impact_factors
            # ROAS와 revenue를 크리에이티브 영향 반영하여 약간 보정
            enriched['ROAS'] = enriched['ROAS'] * enriched['creative_impact_factor']
            enriched['revenue'] = enriched['ad_spend'] * enriched['ROAS']
            # CPA도 재계산
            enriched['CPA'] = np.where(
                enriched['conversions'] > 0,
                enriched['ad_spend'] / enriched['conversions'],
                enriched['CPA']
            )

        added_cols = [c for c in enriched.columns if c not in ads_df.columns]
        print(f"  Added {len(added_cols)} new columns: {added_cols}")
        print(f"  Output: {len(enriched)} rows, {len(enriched.columns)} columns")

        return enriched

    def save_enriched(self, enriched_df, output_path='./data/ads_with_creative.csv'):
        """보강된 데이터 저장"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        enriched_df.to_csv(output_path, index=False)
        print(f"[Phase 3] Saved: {output_path} ({len(enriched_df)} rows)")


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    enricher = CreativeDataEnricher(seed=42)

    if os.path.exists('global_ads_performance_dataset.csv'):
        ads_df = pd.read_csv('global_ads_performance_dataset.csv')
        enriched = enricher.enrich_ads_data(ads_df, apply_impact=True)

        print(f"\n--- Enriched Data Summary ---")
        print(f"Shape: {enriched.shape}")
        print(f"\nAd Format distribution:")
        print(enriched['ad_format'].value_counts().head(10))
        print(f"\nTarget Age distribution:")
        print(enriched['target_age_group'].value_counts())
        print(f"\nRetargeting rate: {enriched['is_retargeting'].mean():.1%}")
        print(f"Lookalike rate: {enriched['is_lookalike'].mean():.1%}")
        print(f"Avg bounce rate: {enriched['bounce_rate'].mean():.1f}%")
        print(f"Avg session duration: {enriched['avg_session_duration'].mean():.0f}s")

        enricher.save_enriched(enriched, './data/ads_with_creative.csv')
    else:
        print("[ERROR] Base dataset not found")
