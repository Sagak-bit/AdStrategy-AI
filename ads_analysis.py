# -*- coding: utf-8 -*-
"""
글로벌 광고 성과 데이터 분석
Top 4 분석:
1. 플랫폼 × 산업 × 국가 교차분석
2. 고성과 + 저성과 캠페인 비교분석
3. 시계열 트렌드 (산업별 분리)
4. 광고비 규모별 효율성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
import io

# Windows 콘솔 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('global_ads_performance_dataset.csv')

print("=" * 80)
print("[Global Ads Performance Data Analysis Report]")
print("=" * 80)

# ============================================================================
# 0. 데이터 기본 정보
# ============================================================================
print("\n" + "=" * 80)
print("[0] Basic Data Information")
print("=" * 80)

print(f"\n* Total Records: {len(df):,}")
print(f"* Period: {df['date'].min()} ~ {df['date'].max()}")
print(f"\n* Platforms: {df['platform'].unique().tolist()}")
print(f"* Campaign Types: {df['campaign_type'].unique().tolist()}")
print(f"* Industries: {df['industry'].unique().tolist()}")
print(f"* Countries: {df['country'].unique().tolist()}")

# 기본 통계
print("\n* Key Metrics Statistics:")
print(df[['impressions', 'clicks', 'CTR', 'CPC', 'ad_spend', 'conversions', 'CPA', 'revenue', 'ROAS']].describe().round(2))

# ============================================================================
# 1. 플랫폼 × 산업 × 국가 교차분석
# ============================================================================
print("\n" + "=" * 80)
print("[1] Platform x Industry x Country Cross Analysis")
print("=" * 80)

# 1-1. 플랫폼별 전체 성과
print("\n[1-1] Platform Performance:")
platform_stats = df.groupby('platform').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'CPA': 'mean',
    'ROAS': 'mean',
    'ad_spend': 'sum',
    'revenue': 'sum',
    'conversions': 'sum'
}).round(2)
platform_stats['Overall_ROAS'] = (platform_stats['revenue'] / platform_stats['ad_spend']).round(2)
print(platform_stats)

# 1-2. 산업별 전체 성과
print("\n[1-2] Industry Performance:")
industry_stats = df.groupby('industry').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'CPA': 'mean',
    'ROAS': 'mean',
    'ad_spend': 'sum',
    'revenue': 'sum'
}).round(2)
industry_stats['Overall_ROAS'] = (industry_stats['revenue'] / industry_stats['ad_spend']).round(2)
print(industry_stats.sort_values('Overall_ROAS', ascending=False))

# 1-3. 국가별 전체 성과
print("\n[1-3] Country Performance:")
country_stats = df.groupby('country').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'CPA': 'mean',
    'ROAS': 'mean',
    'ad_spend': 'sum',
    'revenue': 'sum'
}).round(2)
country_stats['Overall_ROAS'] = (country_stats['revenue'] / country_stats['ad_spend']).round(2)
print(country_stats.sort_values('Overall_ROAS', ascending=False))

# 1-4. 플랫폼 × 산업 교차분석 (ROAS 기준)
print("\n[1-4] Platform x Industry Cross Analysis (Avg ROAS):")
platform_industry = df.pivot_table(
    values='ROAS', 
    index='industry', 
    columns='platform', 
    aggfunc='mean'
).round(2)
print(platform_industry)

# 각 산업별 최적 플랫폼
print("\n[BEST] Best Platform by Industry (ROAS):")
for industry in platform_industry.index:
    best_platform = platform_industry.loc[industry].idxmax()
    best_roas = platform_industry.loc[industry].max()
    print(f"  - {industry}: {best_platform} (ROAS: {best_roas:.2f})")

# 1-5. 플랫폼 × 국가 교차분석
print("\n[1-5] Platform x Country Cross Analysis (Avg ROAS):")
platform_country = df.pivot_table(
    values='ROAS', 
    index='country', 
    columns='platform', 
    aggfunc='mean'
).round(2)
print(platform_country)

# 1-6. 플랫폼 × 산업 × 국가 3차원 분석 (Top 10 조합)
print("\n[1-6] Platform x Industry x Country - TOP 10 ROAS Combinations:")
combo_stats = df.groupby(['platform', 'industry', 'country']).agg({
    'ROAS': 'mean',
    'ad_spend': 'sum',
    'revenue': 'sum',
    'conversions': 'sum'
}).reset_index()
combo_stats['sample_count'] = df.groupby(['platform', 'industry', 'country']).size().values
# 샘플 수가 5개 이상인 것만 필터링 (신뢰성)
combo_stats_filtered = combo_stats[combo_stats['sample_count'] >= 5].sort_values('ROAS', ascending=False)
print(combo_stats_filtered.head(10).to_string(index=False))

print("\n[1-6-2] Platform x Industry x Country - BOTTOM 10 ROAS Combinations:")
print(combo_stats_filtered.tail(10).to_string(index=False))

# ============================================================================
# 2. 고성과 + 저성과 캠페인 비교분석
# ============================================================================
print("\n" + "=" * 80)
print("[2] Top vs Bottom Performers Analysis")
print("=" * 80)

# ROAS 기준 상위/하위 10%
top_10_threshold = df['ROAS'].quantile(0.90)
bottom_10_threshold = df['ROAS'].quantile(0.10)

top_performers = df[df['ROAS'] >= top_10_threshold]
bottom_performers = df[df['ROAS'] <= bottom_10_threshold]

print(f"\n* ROAS Top 10% threshold: {top_10_threshold:.2f} or higher ({len(top_performers)} records)")
print(f"* ROAS Bottom 10% threshold: {bottom_10_threshold:.2f} or lower ({len(bottom_performers)} records)")

# 2-1. 상위/하위 그룹 기본 통계 비교
print("\n[2-1] Top vs Bottom Group Comparison (Mean):")
comparison = pd.DataFrame({
    'Top 10%': top_performers[['CTR', 'CPC', 'CPA', 'ROAS', 'ad_spend', 'revenue']].mean(),
    'Bottom 10%': bottom_performers[['CTR', 'CPC', 'CPA', 'ROAS', 'ad_spend', 'revenue']].mean(),
    'Overall': df[['CTR', 'CPC', 'CPA', 'ROAS', 'ad_spend', 'revenue']].mean()
}).round(2)
print(comparison)

# 2-2. 플랫폼 분포 비교
print("\n[2-2] Platform Distribution:")
print("  Top 10%:")
top_platform = top_performers['platform'].value_counts(normalize=True).round(3) * 100
for p, v in top_platform.items():
    print(f"    - {p}: {v:.1f}%")

print("  Bottom 10%:")
bottom_platform = bottom_performers['platform'].value_counts(normalize=True).round(3) * 100
for p, v in bottom_platform.items():
    print(f"    - {p}: {v:.1f}%")

# 2-3. 산업 분포 비교
print("\n[2-3] Industry Distribution:")
print("  Top 10%:")
top_industry = top_performers['industry'].value_counts(normalize=True).round(3) * 100
for i, v in top_industry.items():
    print(f"    - {i}: {v:.1f}%")

print("  Bottom 10%:")
bottom_industry = bottom_performers['industry'].value_counts(normalize=True).round(3) * 100
for i, v in bottom_industry.items():
    print(f"    - {i}: {v:.1f}%")

# 2-4. 국가 분포 비교
print("\n[2-4] Country Distribution:")
print("  Top 10%:")
top_country = top_performers['country'].value_counts(normalize=True).round(3) * 100
for c, v in top_country.items():
    print(f"    - {c}: {v:.1f}%")

print("  Bottom 10%:")
bottom_country = bottom_performers['country'].value_counts(normalize=True).round(3) * 100
for c, v in bottom_country.items():
    print(f"    - {c}: {v:.1f}%")

# 2-5. 캠페인 유형 분포 비교
print("\n[2-5] Campaign Type Distribution:")
print("  Top 10%:")
top_campaign = top_performers['campaign_type'].value_counts(normalize=True).round(3) * 100
for c, v in top_campaign.items():
    print(f"    - {c}: {v:.1f}%")

print("  Bottom 10%:")
bottom_campaign = bottom_performers['campaign_type'].value_counts(normalize=True).round(3) * 100
for c, v in bottom_campaign.items():
    print(f"    - {c}: {v:.1f}%")

# 2-6. 고성과 캠페인의 주요 조합
print("\n[2-6] Top 10% - Most Common Combinations:")
top_combo = top_performers.groupby(['platform', 'industry', 'campaign_type']).size().reset_index(name='count')
top_combo = top_combo.sort_values('count', ascending=False).head(10)
print(top_combo.to_string(index=False))

print("\n[2-6-2] Bottom 10% - Most Common Combinations:")
bottom_combo = bottom_performers.groupby(['platform', 'industry', 'campaign_type']).size().reset_index(name='count')
bottom_combo = bottom_combo.sort_values('count', ascending=False).head(10)
print(bottom_combo.to_string(index=False))

# ============================================================================
# 3. 시계열 트렌드 분석 (산업별 분리)
# ============================================================================
print("\n" + "=" * 80)
print("[3] Time Series Trend Analysis")
print("=" * 80)

# 날짜 변환
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['month_name'] = df['date'].dt.strftime('%Y-%m')

# 3-1. 월별 전체 트렌드
print("\n[3-1] Monthly Overall Performance:")
monthly_stats = df.groupby('month').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'ROAS': 'mean',
    'ad_spend': 'sum',
    'revenue': 'sum',
    'conversions': 'sum'
}).round(2)
monthly_stats['Overall_ROAS'] = (monthly_stats['revenue'] / monthly_stats['ad_spend']).round(2)
print(monthly_stats)

# 3-2. 분기별 성과
print("\n[3-2] Quarterly Performance:")
quarterly_stats = df.groupby('quarter').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'ROAS': 'mean',
    'ad_spend': 'sum',
    'revenue': 'sum'
}).round(2)
quarterly_stats['Overall_ROAS'] = (quarterly_stats['revenue'] / quarterly_stats['ad_spend']).round(2)
print(quarterly_stats)

# 3-3. 산업별 월간 ROAS 트렌드
print("\n[3-3] Industry Monthly ROAS Trend:")
industry_monthly_roas = df.pivot_table(
    values='ROAS',
    index='month',
    columns='industry',
    aggfunc='mean'
).round(2)
print(industry_monthly_roas)

# 3-4. 각 산업의 최고/최저 성과 월
print("\n[BEST/WORST] Industry Best/Worst Month (ROAS):")
for industry in df['industry'].unique():
    industry_data = industry_monthly_roas[industry]
    best_month = industry_data.idxmax()
    worst_month = industry_data.idxmin()
    print(f"  - {industry}: Best={best_month}M({industry_data.max():.2f}) / Worst={worst_month}M({industry_data.min():.2f})")

# 3-5. 플랫폼별 월간 트렌드
print("\n[3-5] Platform Monthly ROAS Trend:")
platform_monthly_roas = df.pivot_table(
    values='ROAS',
    index='month',
    columns='platform',
    aggfunc='mean'
).round(2)
print(platform_monthly_roas)

# ============================================================================
# 4. 광고비 규모별 효율성 분석
# ============================================================================
print("\n" + "=" * 80)
print("[4] Ad Spend Efficiency Analysis")
print("=" * 80)

# 광고비 구간 설정
df['ad_spend_bin'] = pd.cut(df['ad_spend'], 
                            bins=[0, 1000, 3000, 5000, 10000, 20000, float('inf')],
                            labels=['~$1K', '$1K~3K', '$3K~5K', '$5K~10K', '$10K~20K', '$20K+'])

# 4-1. 광고비 구간별 성과
print("\n[4-1] Performance by Ad Spend Range:")
spend_bin_stats = df.groupby('ad_spend_bin').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'CPA': 'mean',
    'ROAS': 'mean',
    'conversions': 'mean',
    'revenue': 'mean'
}).round(2)
spend_bin_stats['count'] = df.groupby('ad_spend_bin').size().values
print(spend_bin_stats)

# 4-2. 광고비 구간별 ROAS 분포
print("\n[4-2] ROAS Distribution by Ad Spend Range (Median, 25th, 75th):")
spend_roas_dist = df.groupby('ad_spend_bin')['ROAS'].agg(['median', 
    lambda x: x.quantile(0.25), 
    lambda x: x.quantile(0.75)]).round(2)
spend_roas_dist.columns = ['Median', '25th', '75th']
print(spend_roas_dist)

# 4-3. 플랫폼별 광고비 효율성
print("\n[4-3] Platform Efficiency by Ad Spend Range (Avg ROAS):")
platform_spend_roas = df.pivot_table(
    values='ROAS',
    index='ad_spend_bin',
    columns='platform',
    aggfunc='mean'
).round(2)
print(platform_spend_roas)

# 4-4. 상관관계 분석
print("\n[4-4] Correlation with Ad Spend:")
correlations = df[['ad_spend', 'ROAS', 'CTR', 'CPC', 'CPA', 'conversions', 'revenue']].corr()['ad_spend'].drop('ad_spend').round(3)
print(correlations.sort_values(ascending=False))

# ============================================================================
# 5. 캠페인 유형별 분석 (보너스)
# ============================================================================
print("\n" + "=" * 80)
print("[5] Campaign Type Analysis (Bonus)")
print("=" * 80)

# 5-1. 캠페인 유형별 전체 성과
print("\n[5-1] Campaign Type Performance:")
campaign_stats = df.groupby('campaign_type').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'CPA': 'mean',
    'ROAS': 'mean',
    'ad_spend': 'sum',
    'revenue': 'sum'
}).round(2)
campaign_stats['Overall_ROAS'] = (campaign_stats['revenue'] / campaign_stats['ad_spend']).round(2)
print(campaign_stats.sort_values('Overall_ROAS', ascending=False))

# 5-2. 산업별 최적 캠페인 유형
print("\n[5-2] Industry x Campaign Type (Avg ROAS):")
industry_campaign = df.pivot_table(
    values='ROAS',
    index='industry',
    columns='campaign_type',
    aggfunc='mean'
).round(2)
print(industry_campaign)

print("\n[BEST] Best Campaign Type by Industry:")
for industry in industry_campaign.index:
    best_type = industry_campaign.loc[industry].idxmax()
    best_roas = industry_campaign.loc[industry].max()
    print(f"  - {industry}: {best_type} (ROAS: {best_roas:.2f})")

# ============================================================================
# 6. 핵심 인사이트 요약
# ============================================================================
print("\n" + "=" * 80)
print("[SUMMARY] Key Insights")
print("=" * 80)

# 최고 성과 플랫폼
best_platform = platform_stats['Overall_ROAS'].idxmax()
best_platform_roas = platform_stats['Overall_ROAS'].max()

# 최고 성과 산업
best_industry = industry_stats['Overall_ROAS'].idxmax()
best_industry_roas = industry_stats['Overall_ROAS'].max()

# 최고 성과 국가
best_country = country_stats['Overall_ROAS'].idxmax()
best_country_roas = country_stats['Overall_ROAS'].max()

# 최고 성과 월
best_month = monthly_stats['Overall_ROAS'].idxmax()
best_month_roas = monthly_stats['Overall_ROAS'].max()

print(f"""
[BEST PERFORMERS]
  - Platform: {best_platform} (Overall ROAS: {best_platform_roas})
  - Industry: {best_industry} (Overall ROAS: {best_industry_roas})
  - Country: {best_country} (Overall ROAS: {best_country_roas})
  - Month: {best_month} (Overall ROAS: {best_month_roas})

[CAUTION]
  - Results are based on simple mean/sum; consider sample size for decisions
  - Performance varies significantly by Platform x Industry x Country combinations
  - High ROAS with low Revenue absolute value needs scalability review
""")

# ============================================================================
# 시각화 저장
# ============================================================================
print("\n" + "=" * 80)
print("[VISUALIZATION] Generating charts...")
print("=" * 80)

# Figure 1: 플랫폼별 성과 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1-1. 플랫폼별 ROAS
ax1 = axes[0, 0]
platform_roas = df.groupby('platform')['ROAS'].mean().sort_values(ascending=True)
bars = ax1.barh(platform_roas.index, platform_roas.values, color=['#1877F2', '#4285F4', '#000000'])
ax1.set_xlabel('Average ROAS')
ax1.set_title('Platform Average ROAS')
for i, v in enumerate(platform_roas.values):
    ax1.text(v + 0.1, i, f'{v:.2f}', va='center')

# 1-2. 산업별 ROAS
ax2 = axes[0, 1]
industry_roas = df.groupby('industry')['ROAS'].mean().sort_values(ascending=True)
ax2.barh(industry_roas.index, industry_roas.values, color='steelblue')
ax2.set_xlabel('Average ROAS')
ax2.set_title('Industry Average ROAS')
for i, v in enumerate(industry_roas.values):
    ax2.text(v + 0.1, i, f'{v:.2f}', va='center')

# 1-3. 국가별 ROAS
ax3 = axes[1, 0]
country_roas = df.groupby('country')['ROAS'].mean().sort_values(ascending=True)
ax3.barh(country_roas.index, country_roas.values, color='seagreen')
ax3.set_xlabel('Average ROAS')
ax3.set_title('Country Average ROAS')
for i, v in enumerate(country_roas.values):
    ax3.text(v + 0.1, i, f'{v:.2f}', va='center')

# 1-4. 캠페인 유형별 ROAS
ax4 = axes[1, 1]
campaign_roas = df.groupby('campaign_type')['ROAS'].mean().sort_values(ascending=True)
ax4.barh(campaign_roas.index, campaign_roas.values, color='coral')
ax4.set_xlabel('Average ROAS')
ax4.set_title('Campaign Type Average ROAS')
for i, v in enumerate(campaign_roas.values):
    ax4.text(v + 0.1, i, f'{v:.2f}', va='center')

plt.tight_layout()
plt.savefig('01_overall_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 01_overall_performance.png saved")

# Figure 2: 교차분석 히트맵
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2-1. 플랫폼 × 산업 히트맵
ax1 = axes[0]
sns.heatmap(platform_industry, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1, center=platform_industry.values.mean())
ax1.set_title('Platform x Industry Avg ROAS')

# 2-2. 플랫폼 × 국가 히트맵
ax2 = axes[1]
sns.heatmap(platform_country, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2, center=platform_country.values.mean())
ax2.set_title('Platform x Country Avg ROAS')

plt.tight_layout()
plt.savefig('02_cross_analysis_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 02_cross_analysis_heatmap.png saved")

# Figure 3: 월별 트렌드
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3-1. 월별 ROAS 트렌드
ax1 = axes[0, 0]
ax1.plot(monthly_stats.index, monthly_stats['ROAS'], marker='o', linewidth=2, color='steelblue')
ax1.set_xlabel('Month')
ax1.set_ylabel('Average ROAS')
ax1.set_title('Monthly Average ROAS Trend')
ax1.set_xticks(range(1, 13))
ax1.grid(True, alpha=0.3)

# 3-2. 월별 CPC 트렌드
ax2 = axes[0, 1]
ax2.plot(monthly_stats.index, monthly_stats['CPC'], marker='s', linewidth=2, color='coral')
ax2.set_xlabel('Month')
ax2.set_ylabel('Average CPC ($)')
ax2.set_title('Monthly Average CPC Trend')
ax2.set_xticks(range(1, 13))
ax2.grid(True, alpha=0.3)

# 3-3. 산업별 월간 ROAS 트렌드
ax3 = axes[1, 0]
for industry in industry_monthly_roas.columns:
    ax3.plot(industry_monthly_roas.index, industry_monthly_roas[industry], marker='o', label=industry)
ax3.set_xlabel('Month')
ax3.set_ylabel('Average ROAS')
ax3.set_title('Industry Monthly ROAS Trend')
ax3.set_xticks(range(1, 13))
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)

# 3-4. 플랫폼별 월간 ROAS 트렌드
ax4 = axes[1, 1]
colors = {'Google Ads': '#4285F4', 'Meta Ads': '#1877F2', 'TikTok Ads': '#000000'}
for platform in platform_monthly_roas.columns:
    ax4.plot(platform_monthly_roas.index, platform_monthly_roas[platform], 
             marker='o', label=platform, color=colors.get(platform, 'gray'))
ax4.set_xlabel('Month')
ax4.set_ylabel('Average ROAS')
ax4.set_title('Platform Monthly ROAS Trend')
ax4.set_xticks(range(1, 13))
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_monthly_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 03_monthly_trends.png saved")

# Figure 4: 광고비 규모별 효율성
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4-1. 광고비 구간별 ROAS
ax1 = axes[0]
spend_roas = df.groupby('ad_spend_bin')['ROAS'].mean()
ax1.bar(range(len(spend_roas)), spend_roas.values, color='steelblue')
ax1.set_xticks(range(len(spend_roas)))
ax1.set_xticklabels(spend_roas.index, rotation=45)
ax1.set_xlabel('Ad Spend Range')
ax1.set_ylabel('Average ROAS')
ax1.set_title('Average ROAS by Ad Spend Range')
for i, v in enumerate(spend_roas.values):
    ax1.text(i, v + 0.2, f'{v:.1f}', ha='center')

# 4-2. 광고비 vs ROAS 산점도
ax2 = axes[1]
ax2.scatter(df['ad_spend'], df['ROAS'], alpha=0.3, s=10)
ax2.set_xlabel('Ad Spend ($)')
ax2.set_ylabel('ROAS')
ax2.set_title('Ad Spend vs ROAS Scatter')
ax2.set_xlim(0, df['ad_spend'].quantile(0.95))
ax2.set_ylim(0, df['ROAS'].quantile(0.95))

plt.tight_layout()
plt.savefig('04_spend_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 04_spend_efficiency.png saved")

# Figure 5: 고성과/저성과 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5-1. 플랫폼 분포 비교
ax1 = axes[0, 0]
x = np.arange(len(top_platform))
width = 0.35
ax1.bar(x - width/2, top_platform.values, width, label='Top 10%', color='green', alpha=0.7)
ax1.bar(x + width/2, bottom_platform.reindex(top_platform.index).fillna(0).values, width, label='Bottom 10%', color='red', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(top_platform.index, rotation=45)
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Platform Distribution: Top 10% vs Bottom 10%')
ax1.legend()

# 5-2. 산업 분포 비교
ax2 = axes[0, 1]
x = np.arange(len(top_industry))
ax2.bar(x - width/2, top_industry.values, width, label='Top 10%', color='green', alpha=0.7)
ax2.bar(x + width/2, bottom_industry.reindex(top_industry.index).fillna(0).values, width, label='Bottom 10%', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(top_industry.index, rotation=45)
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Industry Distribution: Top 10% vs Bottom 10%')
ax2.legend()

# 5-3. 국가 분포 비교
ax3 = axes[1, 0]
x = np.arange(len(top_country))
ax3.bar(x - width/2, top_country.values, width, label='Top 10%', color='green', alpha=0.7)
ax3.bar(x + width/2, bottom_country.reindex(top_country.index).fillna(0).values, width, label='Bottom 10%', color='red', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(top_country.index, rotation=45)
ax3.set_ylabel('Percentage (%)')
ax3.set_title('Country Distribution: Top 10% vs Bottom 10%')
ax3.legend()

# 5-4. 캠페인 유형 분포 비교
ax4 = axes[1, 1]
x = np.arange(len(top_campaign))
ax4.bar(x - width/2, top_campaign.values, width, label='Top 10%', color='green', alpha=0.7)
ax4.bar(x + width/2, bottom_campaign.reindex(top_campaign.index).fillna(0).values, width, label='Bottom 10%', color='red', alpha=0.7)
ax4.set_xticks(x)
ax4.set_xticklabels(top_campaign.index, rotation=45)
ax4.set_ylabel('Percentage (%)')
ax4.set_title('Campaign Type Distribution: Top 10% vs Bottom 10%')
ax4.legend()

plt.tight_layout()
plt.savefig('05_top_bottom_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 05_top_bottom_comparison.png saved")

print("\n" + "=" * 80)
print("[COMPLETE] All analysis finished!")
print("=" * 80)
print("""
Generated Files:
  - 01_overall_performance.png    : Overall performance overview
  - 02_cross_analysis_heatmap.png : Cross analysis heatmaps
  - 03_monthly_trends.png         : Monthly trends
  - 04_spend_efficiency.png       : Ad spend efficiency
  - 05_top_bottom_comparison.png  : Top vs Bottom comparison
""")
