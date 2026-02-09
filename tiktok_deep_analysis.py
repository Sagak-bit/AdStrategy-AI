# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_csv('global_ads_performance_dataset.csv')

print('=' * 80)
print('TikTok Ads Deep Analysis')
print('=' * 80)

# 1. 플랫폼별 전체 성과 비교
print('\n[1] Platform Performance Comparison')
print('-' * 60)
platform_stats = df.groupby('platform').agg({
    'CTR': 'mean',
    'CPC': 'mean', 
    'CPA': 'mean',
    'ROAS': 'mean',
    'ad_spend': ['sum', 'mean'],
    'revenue': 'sum',
    'conversions': 'sum',
    'clicks': 'sum'
}).round(2)
platform_stats.columns = ['Avg_CTR', 'Avg_CPC', 'Avg_CPA', 'Avg_ROAS', 'Total_Spend', 'Avg_Spend', 'Total_Revenue', 'Total_Conv', 'Total_Clicks']
platform_stats['Overall_ROAS'] = (platform_stats['Total_Revenue'] / platform_stats['Total_Spend']).round(2)
platform_stats['Conv_Rate'] = (platform_stats['Total_Conv'] / platform_stats['Total_Clicks'] * 100).round(2)
print(platform_stats[['Avg_ROAS', 'Overall_ROAS', 'Avg_CPC', 'Avg_CPA', 'Conv_Rate']])

# 비율 계산
tiktok_roas = platform_stats.loc['TikTok Ads', 'Avg_ROAS']
google_roas = platform_stats.loc['Google Ads', 'Avg_ROAS']
meta_roas = platform_stats.loc['Meta Ads', 'Avg_ROAS']
print(f'\nTikTok vs Google: {tiktok_roas/google_roas:.2f}x better')
print(f'TikTok vs Meta: {tiktok_roas/meta_roas:.2f}x better')

tiktok_cpc = platform_stats.loc['TikTok Ads', 'Avg_CPC']
google_cpc = platform_stats.loc['Google Ads', 'Avg_CPC']
print(f'TikTok CPC is {(1-tiktok_cpc/google_cpc)*100:.0f}% cheaper than Google')

# 2. 산업별 TikTok vs 타 플랫폼 비교
print('\n[2] Industry-wise Platform ROAS Comparison')
print('-' * 60)
industry_platform = df.pivot_table(values='ROAS', index='industry', columns='platform', aggfunc='mean').round(2)
industry_platform['TikTok_vs_Google'] = (industry_platform['TikTok Ads'] / industry_platform['Google Ads']).round(2)
industry_platform['TikTok_vs_Meta'] = (industry_platform['TikTok Ads'] / industry_platform['Meta Ads']).round(2)
print(industry_platform)

print('\n[Summary] TikTok advantage by industry:')
for industry in industry_platform.index:
    tg = industry_platform.loc[industry, 'TikTok_vs_Google']
    tm = industry_platform.loc[industry, 'TikTok_vs_Meta']
    print(f'  {industry}: {tg}x vs Google, {tm}x vs Meta')

# 3. 국가별 TikTok vs 타 플랫폼 비교
print('\n[3] Country-wise Platform ROAS Comparison')
print('-' * 60)
country_platform = df.pivot_table(values='ROAS', index='country', columns='platform', aggfunc='mean').round(2)
country_platform['TikTok_vs_Google'] = (country_platform['TikTok Ads'] / country_platform['Google Ads']).round(2)
print(country_platform)

# 4. 캠페인 유형별 TikTok 성과
print('\n[4] TikTok Ads Performance by Campaign Type')
print('-' * 60)
tiktok_campaign = df[df['platform'] == 'TikTok Ads'].groupby('campaign_type').agg({
    'ROAS': 'mean',
    'CPC': 'mean',
    'CPA': 'mean',
    'CTR': 'mean'
}).round(2)
print(tiktok_campaign.sort_values('ROAS', ascending=False))

# 5. 광고비 구간별 TikTok 효율
print('\n[5] TikTok Ads Efficiency by Ad Spend Range')
print('-' * 60)
tiktok_df = df[df['platform'] == 'TikTok Ads'].copy()
tiktok_df['spend_bin'] = pd.cut(tiktok_df['ad_spend'], bins=[0,1000,3000,5000,10000,50000], labels=['~1K','1K-3K','3K-5K','5K-10K','10K+'])
spend_roas = tiktok_df.groupby('spend_bin')['ROAS'].agg(['mean','median','count']).round(2)
print(spend_roas)

# 6. TikTok 최고 성과 조합 Top 10
print('\n[6] TikTok Ads Best Performing Combinations (Industry x Country)')
print('-' * 60)
tiktok_combo = tiktok_df.groupby(['industry', 'country']).agg({
    'ROAS': 'mean',
    'ad_spend': 'sum'
}).round(2).reset_index()
tiktok_combo['sample'] = tiktok_df.groupby(['industry', 'country']).size().values
tiktok_combo = tiktok_combo[tiktok_combo['sample'] >= 5].sort_values('ROAS', ascending=False)
print(tiktok_combo.head(10).to_string(index=False))

# 7. TikTok 고성과 캠페인 특성
print('\n[7] TikTok Ads Top 10% Campaign Characteristics')
print('-' * 60)
top_threshold = tiktok_df['ROAS'].quantile(0.9)
tiktok_top = tiktok_df[tiktok_df['ROAS'] >= top_threshold]
print(f'Top 10% ROAS threshold: {top_threshold:.2f}')
print(f'Top 10% Average ROAS: {tiktok_top["ROAS"].mean():.2f}')
print(f'Number of top campaigns: {len(tiktok_top)}')

print('\nIndustry distribution in top 10%:')
for ind, pct in (tiktok_top['industry'].value_counts(normalize=True) * 100).items():
    print(f'  {ind}: {pct:.1f}%')

print('\nCountry distribution in top 10%:')
for cnt, pct in (tiktok_top['country'].value_counts(normalize=True) * 100).items():
    print(f'  {cnt}: {pct:.1f}%')

print('\nCampaign type distribution in top 10%:')
for ct, pct in (tiktok_top['campaign_type'].value_counts(normalize=True) * 100).items():
    print(f'  {ct}: {pct:.1f}%')

# 8. 월별 TikTok 성과 트렌드
print('\n[8] TikTok Ads Monthly ROAS Trend')
print('-' * 60)
tiktok_df['date'] = pd.to_datetime(tiktok_df['date'])
tiktok_df['month'] = tiktok_df['date'].dt.month
monthly = tiktok_df.groupby('month')['ROAS'].mean().round(2)
print(monthly)
print(f'\nBest month: {monthly.idxmax()} (ROAS: {monthly.max():.2f})')
print(f'Worst month: {monthly.idxmin()} (ROAS: {monthly.min():.2f})')
