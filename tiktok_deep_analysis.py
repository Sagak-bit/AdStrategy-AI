# -*- coding: utf-8 -*-
"""
TikTok Ads 심층 분석
===================
TikTok Ads의 경쟁 우위를 통계적으로 검증하고,
Google Ads / Meta Ads 대비 성과를 다각도로 분석합니다.

출력:
  - figures/11_tiktok_platform_comparison.png
  - figures/12_tiktok_industry_advantage.png
  - figures/13_tiktok_optimal_strategy.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import sys
import io
import os
import warnings

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (크로스플랫폼)
import platform as _pf
_os_name = _pf.system()
if _os_name == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif _os_name == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# 데이터 로드
data_path = 'data/enriched_ads_final.csv' if os.path.exists('data/enriched_ads_final.csv') \
    else 'global_ads_performance_dataset.csv'
df = pd.read_csv(data_path)

print('=' * 80)
print('TikTok Ads Deep Analysis (Enhanced)')
print(f'데이터: {data_path} ({len(df):,}건)')
print('=' * 80)

# ============================================================================
# 1. 플랫폼별 성과 비교 (통계 검증 포함)
# ============================================================================
print('\n[1] Platform Performance Comparison (with Statistical Tests)')
print('-' * 60)
platform_stats = df.groupby('platform').agg({
    'CTR': 'mean',
    'CPC': 'mean',
    'CPA': 'mean',
    'ROAS': ['mean', 'median', 'std', 'count'],
    'ad_spend': ['sum', 'mean'],
    'revenue': 'sum',
    'conversions': 'sum',
    'clicks': 'sum'
}).round(2)
platform_stats.columns = [
    'Avg_CTR', 'Avg_CPC', 'Avg_CPA',
    'Avg_ROAS', 'Med_ROAS', 'Std_ROAS', 'N',
    'Total_Spend', 'Avg_Spend', 'Total_Revenue', 'Total_Conv', 'Total_Clicks'
]
platform_stats['Overall_ROAS'] = (platform_stats['Total_Revenue'] / platform_stats['Total_Spend']).round(2)
platform_stats['Conv_Rate'] = (platform_stats['Total_Conv'] / platform_stats['Total_Clicks'] * 100).round(2)
print(platform_stats[['Avg_ROAS', 'Med_ROAS', 'Std_ROAS', 'N', 'Avg_CPC', 'Conv_Rate']])

# 비율 및 통계 검정
tiktok = df[df['platform'] == 'TikTok Ads']['ROAS']
google = df[df['platform'] == 'Google Ads']['ROAS']
meta = df[df['platform'] == 'Meta Ads']['ROAS']

print(f'\n  TikTok Avg ROAS: {tiktok.mean():.3f} (N={len(tiktok)})')
print(f'  Google Avg ROAS: {google.mean():.3f} (N={len(google)})')
print(f'  Meta   Avg ROAS: {meta.mean():.3f} (N={len(meta)})')

# 최소 표본 크기 검증 함수
MIN_SAMPLE_SIZE = 20  # 통계 검정 최소 표본

def check_sample_size(group_a, group_b, name_a, name_b, min_n=MIN_SAMPLE_SIZE):
    """표본 크기 검증 및 검정력(power) 경고"""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < min_n or n_b < min_n:
        print(f'  [WARNING] 표본 크기 부족: {name_a}={n_a}, {name_b}={n_b} (최소 {min_n} 필요)')
        print(f'    -> 검정 결과의 통계적 검정력(power)이 낮을 수 있음')
        return False
    # Cohen's d 기준 small effect(0.2) 탐지를 위한 최소 표본 ~394/group
    # medium effect(0.5) 탐지를 위한 최소 표본 ~64/group
    if n_a < 64 or n_b < 64:
        print(f'  [INFO] 표본 크기: {name_a}={n_a}, {name_b}={n_b}')
        print(f'    -> medium effect(d=0.5) 탐지 가능, small effect(d=0.2) 탐지에는 부족할 수 있음')
    else:
        print(f'  [OK] 표본 크기 충분: {name_a}={n_a}, {name_b}={n_b}')
    return True

# 표본 크기 검증 수행
print(f'\n  [Sample Size Validation]')
valid_tg = check_sample_size(tiktok, google, 'TikTok', 'Google')
valid_tm = check_sample_size(tiktok, meta, 'TikTok', 'Meta')

# Welch's t-test
t_tg, p_tg = ttest_ind(tiktok, google, equal_var=False)
t_tm, p_tm = ttest_ind(tiktok, meta, equal_var=False)

# Mann-Whitney U (비모수)
u_tg, pu_tg = mannwhitneyu(tiktok, google, alternative='greater')
u_tm, pu_tm = mannwhitneyu(tiktok, meta, alternative='greater')

# Cohen's d (pooled SD 사용)
d_tg = (tiktok.mean() - google.mean()) / np.sqrt((tiktok.std()**2 + google.std()**2) / 2)
d_tm = (tiktok.mean() - meta.mean()) / np.sqrt((tiktok.std()**2 + meta.std()**2) / 2)

# 효과 크기 해석 함수
def interpret_cohens_d(d):
    """Cohen's d 해석"""
    abs_d = abs(d)
    if abs_d < 0.2: return '무시할 수준(negligible)'
    elif abs_d < 0.5: return '소(small)'
    elif abs_d < 0.8: return '중(medium)'
    else: return '대(large)'

print(f'\n  [TikTok vs Google] t={t_tg:.3f}, p={p_tg:.2e}, Cohen\'s d={d_tg:.3f} ({interpret_cohens_d(d_tg)})')
print(f'    Mann-Whitney U (one-sided): U={u_tg:.0f}, p={pu_tg:.2e}')
caveat_tg = " (단, 표본 부족으로 검정력 제한)" if not valid_tg else ""
print(f'    => {"TikTok이 통계적으로 유의하게 높음" if p_tg < 0.05 and tiktok.mean() > google.mean() else "유의한 차이 없음"}{caveat_tg}')

print(f'\n  [TikTok vs Meta]   t={t_tm:.3f}, p={p_tm:.2e}, Cohen\'s d={d_tm:.3f} ({interpret_cohens_d(d_tm)})')
print(f'    Mann-Whitney U (one-sided): U={u_tm:.0f}, p={pu_tm:.2e}')
caveat_tm = " (단, 표본 부족으로 검정력 제한)" if not valid_tm else ""
print(f'    => {"TikTok이 통계적으로 유의하게 높음" if p_tm < 0.05 and tiktok.mean() > meta.mean() else "유의한 차이 없음"}{caveat_tm}')

# CPC 비교
tiktok_cpc = platform_stats.loc['TikTok Ads', 'Avg_CPC']
google_cpc = platform_stats.loc['Google Ads', 'Avg_CPC']
print(f'\n  TikTok CPC ${tiktok_cpc:.2f} vs Google CPC ${google_cpc:.2f} '
      f'({(1 - tiktok_cpc / google_cpc) * 100:.0f}% cheaper)')

# ============================================================================
# 2. 산업별 TikTok 우위 분석
# ============================================================================
print('\n[2] Industry-wise TikTok Advantage')
print('-' * 60)
industry_platform = df.pivot_table(values='ROAS', index='industry', columns='platform', aggfunc='mean').round(3)
industry_platform['TikTok_vs_Google'] = (industry_platform['TikTok Ads'] / industry_platform['Google Ads']).round(3)
industry_platform['TikTok_vs_Meta'] = (industry_platform['TikTok Ads'] / industry_platform['Meta Ads']).round(3)

# 각 산업에서 통계 검정 (Bonferroni 보정 적용)
n_industry_tests = len(df['industry'].unique())
bonferroni_alpha = 0.05 / n_industry_tests
print(f'\n  산업별 TikTok vs Google 통계 검정 (Bonferroni α={bonferroni_alpha:.4f}, {n_industry_tests}개 비교):')
industry_tests = []
for industry in df['industry'].unique():
    ind_tiktok = df[(df['platform'] == 'TikTok Ads') & (df['industry'] == industry)]['ROAS']
    ind_google = df[(df['platform'] == 'Google Ads') & (df['industry'] == industry)]['ROAS']
    if len(ind_tiktok) < MIN_SAMPLE_SIZE or len(ind_google) < MIN_SAMPLE_SIZE:
        print(f'    {industry:>15}: [SKIP] 표본 부족 (TikTok n={len(ind_tiktok)}, Google n={len(ind_google)}, 최소 {MIN_SAMPLE_SIZE} 필요)')
        continue
    if len(ind_tiktok) >= 10 and len(ind_google) >= 10:
        t, p = ttest_ind(ind_tiktok, ind_google, equal_var=False)
        d = (ind_tiktok.mean() - ind_google.mean()) / np.sqrt((ind_tiktok.std()**2 + ind_google.std()**2) / 2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < bonferroni_alpha else "n.s."
        print(f'    {industry:>15}: TikTok={ind_tiktok.mean():.2f} vs Google={ind_google.mean():.2f}  '
              f'd={d:.3f}  p={p:.3f} {sig}')
        industry_tests.append({'industry': industry, 't': t, 'p': p, 'd': d,
                                'tiktok_mean': ind_tiktok.mean(), 'google_mean': ind_google.mean()})

# ============================================================================
# 3. 국가별 플랫폼 비교
# ============================================================================
print('\n[3] Country-wise Platform ROAS Comparison')
print('-' * 60)
country_platform = df.pivot_table(values='ROAS', index='country', columns='platform', aggfunc='mean').round(3)
country_platform['Best_Platform'] = country_platform.idxmax(axis=1)
print(country_platform)

# ============================================================================
# 4. TikTok 캠페인 유형별 성과
# ============================================================================
print('\n[4] TikTok Ads Performance by Campaign Type')
print('-' * 60)
tiktok_df = df[df['platform'] == 'TikTok Ads'].copy()
tiktok_campaign = tiktok_df.groupby('campaign_type').agg({
    'ROAS': ['mean', 'median', 'std', 'count'],
    'CPC': 'mean',
    'CPA': 'mean',
    'CTR': 'mean'
}).round(3)
tiktok_campaign.columns = ['Avg_ROAS', 'Med_ROAS', 'Std_ROAS', 'N', 'Avg_CPC', 'Avg_CPA', 'Avg_CTR']
print(tiktok_campaign.sort_values('Avg_ROAS', ascending=False))

# ============================================================================
# 5. TikTok 최적 예산 구간
# ============================================================================
print('\n[5] TikTok Ads Optimal Ad Spend Range')
print('-' * 60)
tiktok_df['spend_bin'] = pd.cut(tiktok_df['ad_spend'],
                                  bins=[0, 1000, 3000, 5000, 10000, 50000],
                                  labels=['~$1K', '$1K-3K', '$3K-5K', '$5K-10K', '$10K+'])
spend_roas = tiktok_df.groupby('spend_bin', observed=True)['ROAS'].agg(['mean', 'median', 'std', 'count']).round(3)
print(spend_roas)

# ============================================================================
# 6. TikTok Top 10 조합
# ============================================================================
print('\n[6] TikTok Ads Best Performing Combinations')
print('-' * 60)
tiktok_combo = tiktok_df.groupby(['industry', 'country']).agg({
    'ROAS': ['mean', 'median'],
    'ad_spend': 'sum'
}).round(2).reset_index()
tiktok_combo.columns = ['industry', 'country', 'mean_ROAS', 'median_ROAS', 'total_spend']
tiktok_combo['sample'] = tiktok_df.groupby(['industry', 'country']).size().values
tiktok_combo = tiktok_combo[tiktok_combo['sample'] >= 5].sort_values('mean_ROAS', ascending=False)
print(tiktok_combo.head(10).to_string(index=False))

# ============================================================================
# 7. TikTok 고성과 캠페인 프로파일링
# ============================================================================
print('\n[7] TikTok Top 10% Campaign Profiling')
print('-' * 60)
top_threshold = tiktok_df['ROAS'].quantile(0.9)
tiktok_top = tiktok_df[tiktok_df['ROAS'] >= top_threshold]
tiktok_bottom = tiktok_df[tiktok_df['ROAS'] <= tiktok_df['ROAS'].quantile(0.1)]

print(f'  Top 10% ROAS threshold: {top_threshold:.2f} ({len(tiktok_top)}건)')
print(f'  Top 10% Mean ROAS: {tiktok_top["ROAS"].mean():.2f}')

for col_name, col in [('Industry', 'industry'), ('Country', 'country'), ('Campaign Type', 'campaign_type')]:
    print(f'\n  [{col_name} Distribution] Top 10% vs Bottom 10%:')
    top_dist = tiktok_top[col].value_counts(normalize=True) * 100
    bot_dist = tiktok_bottom[col].value_counts(normalize=True) * 100
    all_vals = sorted(set(list(top_dist.index) + list(bot_dist.index)))
    for v in all_vals:
        t = top_dist.get(v, 0)
        b = bot_dist.get(v, 0)
        diff = t - b
        print(f'    {v:>20}: Top={t:5.1f}%  Bottom={b:5.1f}%  (diff={diff:+5.1f}%)')

# ============================================================================
# 8. 월별 트렌드
# ============================================================================
print('\n[8] TikTok Ads Monthly ROAS Trend')
print('-' * 60)
tiktok_df['date'] = pd.to_datetime(tiktok_df['date'], format='mixed')
tiktok_df['month'] = tiktok_df['date'].dt.month
monthly = tiktok_df.groupby('month')['ROAS'].agg(['mean', 'median', 'std', 'count']).round(3)
print(monthly)
print(f'\n  Best month: {monthly["mean"].idxmax()} (ROAS: {monthly["mean"].max():.3f})')
print(f'  Worst month: {monthly["mean"].idxmin()} (ROAS: {monthly["mean"].min():.3f})')

# ============================================================================
# 시각화
# ============================================================================
print('\n' + '=' * 80)
print('[VISUALIZATION] Generating TikTok analysis charts...')
print('=' * 80)

# --- Figure 11: 플랫폼 종합 비교 ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('TikTok vs Google vs Meta: Comprehensive Comparison', fontsize=16, fontweight='bold')

# 11-1: ROAS 분포 비교 (바이올린)
ax = axes[0, 0]
platforms_list = ['Google Ads', 'Meta Ads', 'TikTok Ads']
colors_plat = ['#4285F4', '#1877F2', '#000000']
data_groups = [df[df['platform'] == p]['ROAS'].clip(0, df['ROAS'].quantile(0.99)).values for p in platforms_list]
parts = ax.violinplot(data_groups, showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_plat[i])
    pc.set_alpha(0.6)
ax.set_xticks(range(1, 4))
ax.set_xticklabels(platforms_list, fontsize=10)
ax.set_ylabel('ROAS')
ax.set_title(f'ROAS Distribution by Platform\n(TikTok vs Google: p={p_tg:.2e})', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 11-2: 주요 KPI 레이더 차트 (대안: grouped bar)
ax = axes[0, 1]
kpi_metrics = ['Avg_ROAS', 'Conv_Rate']
kpi_data = platform_stats[kpi_metrics].reindex(platforms_list)
x = np.arange(len(kpi_metrics))
width = 0.25
for i, platform in enumerate(platforms_list):
    vals = kpi_data.loc[platform].values
    ax.bar(x + i * width, vals, width, label=platform, color=colors_plat[i], alpha=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(['Avg ROAS', 'Conv Rate (%)'], fontsize=10)
ax.set_title('Key KPI Comparison', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 11-3: CPC/CPA 비교
ax = axes[1, 0]
cost_metrics = ['Avg_CPC', 'Avg_CPA']
cost_data = platform_stats[cost_metrics].reindex(platforms_list)
x = np.arange(len(cost_metrics))
for i, platform in enumerate(platforms_list):
    vals = cost_data.loc[platform].values
    ax.bar(x + i * width, vals, width, label=platform, color=colors_plat[i], alpha=0.8)
    for j, v in enumerate(vals):
        ax.text(x[j] + i * width, v + 0.5, f'${v:.1f}', ha='center', fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(['Avg CPC ($)', 'Avg CPA ($)'], fontsize=10)
ax.set_title('Cost Efficiency Comparison', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 11-4: 효과 크기 시각화
ax = axes[1, 1]
effect_sizes = [d_tg, d_tm]
effect_labels = ['TikTok\nvs Google', 'TikTok\nvs Meta']
effect_colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in effect_sizes]
bars = ax.bar(effect_labels, effect_sizes, color=effect_colors, edgecolor='white', linewidth=2)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
ax.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium effect')
for bar, d in zip(bars, effect_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'd={d:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel("Cohen's d")
ax.set_title("Effect Size: TikTok Advantage", fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIGURES_DIR, '11_tiktok_platform_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] 11_tiktok_platform_comparison.png saved')

# --- Figure 12: 산업별 TikTok 우위 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('TikTok Industry-wise Advantage', fontsize=16, fontweight='bold')

# 12-1: 산업별 플랫폼 ROAS
ax = axes[0]
industry_pivot = df.pivot_table(values='ROAS', index='industry', columns='platform', aggfunc='mean')
industry_pivot = industry_pivot.reindex(columns=platforms_list)
industry_pivot.plot(kind='barh', ax=ax, color=colors_plat, alpha=0.8, edgecolor='white')
ax.set_xlabel('Mean ROAS')
ax.set_title('Industry × Platform ROAS', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)

# 12-2: TikTok 우위 비율 (산업별)
ax = axes[1]
if industry_tests:
    test_df = pd.DataFrame(industry_tests).sort_values('d', ascending=True)
    colors_test = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in test_df['p']]
    ax.barh(test_df['industry'], test_df['d'], color=colors_test, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
    for i, row in test_df.iterrows():
        sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
        ax.text(row['d'] + 0.01, row['industry'], f"d={row['d']:.2f} {sig}", va='center', fontsize=9)
    ax.set_xlabel("Cohen's d (TikTok - Google)")
    ax.set_title('TikTok vs Google Effect Size by Industry\n(Green = p < 0.05)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(FIGURES_DIR, '12_tiktok_industry_advantage.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] 12_tiktok_industry_advantage.png saved')

# --- Figure 13: TikTok 최적 전략 ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('TikTok Ads: Optimal Strategy Analysis', fontsize=16, fontweight='bold')

# 13-1: 예산 구간별 ROAS
ax = axes[0, 0]
if len(spend_roas) > 0:
    ax.bar(range(len(spend_roas)), spend_roas['mean'].values, color='#000000', alpha=0.7,
           yerr=spend_roas['std'].values, capsize=5, edgecolor='white')
    ax.set_xticks(range(len(spend_roas)))
    ax.set_xticklabels(spend_roas.index, fontsize=10)
    for i, (m, n) in enumerate(zip(spend_roas['mean'], spend_roas['count'])):
        ax.text(i, m + spend_roas['std'].iloc[i] + 0.3, f'{m:.1f}\n(n={n})', ha='center', fontsize=9)
ax.set_xlabel('Ad Spend Range')
ax.set_ylabel('Mean ROAS (±1 SD)')
ax.set_title('TikTok ROAS by Budget Range', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 13-2: 캠페인 유형별 ROAS
ax = axes[0, 1]
camp_data = tiktok_campaign.sort_values('Avg_ROAS', ascending=True)
colors_camp = sns.color_palette("viridis", len(camp_data))
ax.barh(camp_data.index, camp_data['Avg_ROAS'], color=colors_camp, edgecolor='white')
for i, (roas, n) in enumerate(zip(camp_data['Avg_ROAS'], camp_data['N'])):
    ax.text(roas + 0.1, i, f'{roas:.2f} (n={n})', va='center', fontsize=10)
ax.set_xlabel('Mean ROAS')
ax.set_title('TikTok ROAS by Campaign Type', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 13-3: 월별 트렌드
ax = axes[1, 0]
ax.plot(monthly.index, monthly['mean'], marker='o', linewidth=2.5, color='black',
        label='Mean ROAS', zorder=5)
ax.fill_between(monthly.index,
                monthly['mean'] - monthly['std'],
                monthly['mean'] + monthly['std'],
                alpha=0.2, color='gray', label='±1 SD')
best_m = monthly['mean'].idxmax()
worst_m = monthly['mean'].idxmin()
ax.annotate(f'Best: {best_m}M', xy=(best_m, monthly.loc[best_m, 'mean']),
            xytext=(best_m + 0.5, monthly.loc[best_m, 'mean'] + 1),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
ax.annotate(f'Worst: {worst_m}M', xy=(worst_m, monthly.loc[worst_m, 'mean']),
            xytext=(worst_m + 0.5, monthly.loc[worst_m, 'mean'] - 1),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
ax.set_xlabel('Month')
ax.set_ylabel('ROAS')
ax.set_title('TikTok Monthly ROAS Trend (±1 SD)', fontweight='bold')
ax.set_xticks(range(1, 13))
ax.legend()
ax.grid(alpha=0.3)

# 13-4: Top 10 조합 히트맵
ax = axes[1, 1]
heatmap_data = tiktok_df.pivot_table(values='ROAS', index='industry', columns='country', aggfunc='mean')
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Mean ROAS'})
ax.set_title('TikTok: Industry × Country ROAS Heatmap', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIGURES_DIR, '13_tiktok_optimal_strategy.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  [OK] 13_tiktok_optimal_strategy.png saved')

print('\n' + '=' * 80)
print('[COMPLETE] TikTok Deep Analysis finished!')
print('=' * 80)
print("""
Generated Files:
  - figures/11_tiktok_platform_comparison.png : TikTok vs 타 플랫폼 종합 비교
  - figures/12_tiktok_industry_advantage.png  : 산업별 TikTok 우위 분석
  - figures/13_tiktok_optimal_strategy.png    : TikTok 최적 전략 (예산/캠페인/트렌드)
""")
