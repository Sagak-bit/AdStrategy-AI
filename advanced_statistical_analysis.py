# -*- coding: utf-8 -*-
"""
고급 통계 분석 모듈
===================
평가 항목 강화를 위한 통계적 유의성 검증, 데이터 품질 감사,
크로스 데이터 분석, 합성 데이터 영향 분석을 수행합니다.

분석 내용:
  1. 데이터 품질 감사 (결측치, 이상치, 분포)
  2. 통계적 유의성 검증 (ANOVA, t-test, 효과 크기)
  3. 거시경제 × 광고 성과 크로스 분석
  4. 합성 데이터 vs 실제 데이터 영향 비교
  5. 다중 회귀 기반 성과 요인 분해

출력:
  - figures/06_data_quality_audit.png
  - figures/07_statistical_significance.png
  - figures/08_macro_cross_analysis.png
  - figures/09_synthetic_impact.png
  - figures/10_factor_decomposition.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, pearsonr, spearmanr
import warnings
import sys
import io
import os

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


def load_data():
    """데이터 로드 (enriched 우선, fallback으로 기본 데이터)"""
    enriched_path = 'data/enriched_ads_final.csv'
    base_path = 'global_ads_performance_dataset.csv'

    if os.path.exists(enriched_path):
        df = pd.read_csv(enriched_path)
        print(f"[DATA] Enriched 데이터 로드: {len(df)}건, {len(df.columns)}컬럼")
        return df, True
    elif os.path.exists(base_path):
        df = pd.read_csv(base_path)
        print(f"[DATA] 기본 데이터 로드: {len(df)}건, {len(df.columns)}컬럼")
        return df, False
    else:
        raise FileNotFoundError("데이터 파일을 찾을 수 없습니다.")


# ============================================================================
# 1. 데이터 품질 감사
# ============================================================================
def data_quality_audit(df):
    """데이터 품질 종합 감사"""
    print("\n" + "=" * 80)
    print("[1] DATA QUALITY AUDIT")
    print("=" * 80)

    report = {}

    # 1-1. 결측치 분석
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).query('missing_count > 0').sort_values('missing_pct', ascending=False)

    print("\n[1-1] Missing Values:")
    if len(missing_report) == 0:
        print("  * 결측치 없음 (모든 컬럼 완전)")
    else:
        print(missing_report)
    report['missing'] = missing_report

    # 1-2. 이상치 탐지 (IQR 방식)
    print("\n[1-2] Outlier Detection (IQR Method):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    key_metrics = [c for c in ['ROAS', 'CPC', 'CPA', 'CTR', 'ad_spend', 'revenue', 'conversions']
                   if c in numeric_cols]

    outlier_counts = {}
    for col in key_metrics:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        outlier_counts[col] = {'count': n_outliers, 'pct': round(pct, 2),
                                'lower': round(lower, 2), 'upper': round(upper, 2)}
        print(f"  {col:>15}: {n_outliers:>5}건 ({pct:.1f}%)  [범위: {lower:.2f} ~ {upper:.2f}]")
    report['outliers'] = outlier_counts

    # 1-3. 분포 정규성 검정 (Shapiro-Wilk, 샘플 5000 이하)
    print("\n[1-3] Normality Test (Shapiro-Wilk, sample=min(5000, N)):")
    for col in key_metrics:
        sample = df[col].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        stat, p_value = stats.shapiro(sample)
        normality = "정규" if p_value > 0.05 else "비정규"
        print(f"  {col:>15}: W={stat:.4f}, p={p_value:.2e}  => {normality}")

    # 1-4. 범주형 변수 분포
    print("\n[1-4] Categorical Variable Distribution:")
    cat_cols = [c for c in ['platform', 'industry', 'country', 'campaign_type'] if c in df.columns]
    for col in cat_cols:
        print(f"\n  [{col}]")
        dist = df[col].value_counts()
        for val, cnt in dist.items():
            pct = cnt / len(df) * 100
            print(f"    {val:>20}: {cnt:>5}건 ({pct:.1f}%)")

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Data Quality Audit', fontsize=16, fontweight='bold')

    for idx, col in enumerate(key_metrics[:6]):
        ax = axes[idx // 3, idx % 3]
        data = df[col].dropna()

        # 히스토그램 + KDE
        ax.hist(data, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='white')
        try:
            kde_x = np.linspace(data.min(), data.quantile(0.99), 200)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data.clip(data.quantile(0.01), data.quantile(0.99)))
            ax.plot(kde_x, kde(kde_x), color='red', linewidth=2)
        except Exception:
            pass

        # 통계 정보 표시
        ax.axvline(data.mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.2f}')
        ax.set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(data.quantile(0.01), data.quantile(0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, '06_data_quality_audit.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [OK] 06_data_quality_audit.png saved")

    return report


# ============================================================================
# 2. 통계적 유의성 검증
# ============================================================================
def statistical_significance_tests(df):
    """플랫폼/산업/국가 간 성과 차이의 통계적 유의성 검증"""
    print("\n" + "=" * 80)
    print("[2] STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    results = {}

    # 2-1. One-way ANOVA: 플랫폼별 ROAS 차이
    print("\n[2-1] One-way ANOVA: Platform → ROAS")
    print("-" * 60)
    platforms = df['platform'].unique()
    groups = [df[df['platform'] == p]['ROAS'].dropna().values for p in platforms]

    f_stat, p_value = f_oneway(*groups)
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  결론: {'플랫폼 간 ROAS 차이가 통계적으로 유의함 (p < 0.05)' if p_value < 0.05 else '유의한 차이 없음'}")

    # 효과 크기 (eta-squared)
    grand_mean = df['ROAS'].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((df['ROAS'] - grand_mean)**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    print(f"  효과 크기 (eta²): {eta_sq:.4f} ({'대' if eta_sq > 0.14 else '중' if eta_sq > 0.06 else '소'})")
    results['anova_platform_roas'] = {'F': f_stat, 'p': p_value, 'eta_sq': eta_sq}

    # 2-2. 사후 검정: Pairwise t-tests (Bonferroni 보정)
    print("\n[2-2] Pairwise t-tests (Bonferroni corrected):")
    print("-" * 60)
    n_comparisons = len(platforms) * (len(platforms) - 1) // 2
    alpha_corrected = 0.05 / n_comparisons

    pairwise_results = []
    for i in range(len(platforms)):
        for j in range(i + 1, len(platforms)):
            g1 = df[df['platform'] == platforms[i]]['ROAS'].dropna()
            g2 = df[df['platform'] == platforms[j]]['ROAS'].dropna()

            t_stat, p_val = ttest_ind(g1, g2, equal_var=False)  # Welch's t-test

            # Cohen's d (효과 크기)
            pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
            cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < alpha_corrected else "n.s."
            print(f"  {platforms[i]:>15} vs {platforms[j]:<15}: "
                  f"t={t_stat:>7.3f}, p={p_val:.2e}, d={cohens_d:>6.3f} {sig}")

            pairwise_results.append({
                'group1': platforms[i], 'group2': platforms[j],
                't': t_stat, 'p': p_val, 'cohens_d': cohens_d,
                'significant': p_val < alpha_corrected
            })

    results['pairwise_platform'] = pairwise_results

    # 2-3. ANOVA: 산업별 ROAS 차이
    print("\n[2-3] One-way ANOVA: Industry → ROAS")
    print("-" * 60)
    industries = df['industry'].unique()
    groups_ind = [df[df['industry'] == ind]['ROAS'].dropna().values for ind in industries]
    f_stat_ind, p_val_ind = f_oneway(*groups_ind)
    print(f"  F-statistic: {f_stat_ind:.4f}")
    print(f"  p-value: {p_val_ind:.2e}")
    print(f"  결론: {'산업 간 ROAS 차이가 통계적으로 유의함' if p_val_ind < 0.05 else '유의한 차이 없음'}")
    results['anova_industry_roas'] = {'F': f_stat_ind, 'p': p_val_ind}

    # 2-4. ANOVA: 국가별 ROAS 차이
    print("\n[2-4] One-way ANOVA: Country → ROAS")
    print("-" * 60)
    countries = df['country'].unique()
    groups_cnt = [df[df['country'] == c]['ROAS'].dropna().values for c in countries]
    f_stat_cnt, p_val_cnt = f_oneway(*groups_cnt)
    print(f"  F-statistic: {f_stat_cnt:.4f}")
    print(f"  p-value: {p_val_cnt:.2e}")
    print(f"  결론: {'국가 간 ROAS 차이가 통계적으로 유의함' if p_val_cnt < 0.05 else '유의한 차이 없음'}")
    results['anova_country_roas'] = {'F': f_stat_cnt, 'p': p_val_cnt}

    # 2-5. 카이제곱 검정: 플랫폼 × 고성과 캠페인 연관성
    print("\n[2-5] Chi-squared Test: Platform × High-Performance Association")
    print("-" * 60)
    df_temp = df.copy()
    df_temp['is_high_roas'] = (df_temp['ROAS'] >= df_temp['ROAS'].quantile(0.75)).astype(int)
    contingency = pd.crosstab(df_temp['platform'], df_temp['is_high_roas'])
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2 / (len(df_temp) * (min(contingency.shape) - 1)))
    print(f"  Chi² = {chi2:.4f}, df = {dof}, p = {p_chi:.2e}")
    print(f"  Cramer's V = {cramers_v:.4f} ({'강' if cramers_v > 0.3 else '중' if cramers_v > 0.1 else '약'}한 연관)")
    print(f"  결론: {'플랫폼과 고성과 캠페인 간 유의한 연관 있음' if p_chi < 0.05 else '유의한 연관 없음'}")
    results['chi2_platform_highperf'] = {'chi2': chi2, 'p': p_chi, 'cramers_v': cramers_v}

    # 2-6. Two-way ANOVA 효과 (Platform × Industry)
    print("\n[2-6] Two-way Interaction: Platform × Industry → ROAS")
    print("-" * 60)
    interaction = df.groupby(['platform', 'industry'])['ROAS'].agg(['mean', 'std', 'count']).round(3)
    print(interaction)

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Significance Tests', fontsize=16, fontweight='bold')

    # 2-1 시각화: 플랫폼별 ROAS 분포 (바이올린 플롯)
    ax1 = axes[0, 0]
    platform_data = [df[df['platform'] == p]['ROAS'].dropna().values for p in platforms]
    parts = ax1.violinplot(platform_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#4285F4', '#1877F2', '#000000'][i % 3])
        pc.set_alpha(0.6)
    ax1.set_xticks(range(1, len(platforms) + 1))
    ax1.set_xticklabels(platforms, fontsize=9)
    ax1.set_ylabel('ROAS')
    ax1.set_title(f'Platform ROAS Distribution\n(ANOVA F={f_stat:.2f}, p={p_value:.2e})', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 2-2 시각화: 효과 크기 비교
    ax2 = axes[0, 1]
    pair_labels = [f"{r['group1'][:6]}\nvs\n{r['group2'][:6]}" for r in pairwise_results]
    pair_d = [abs(r['cohens_d']) for r in pairwise_results]
    pair_sig = [r['significant'] for r in pairwise_results]
    colors = ['#2ecc71' if s else '#e74c3c' for s in pair_sig]
    bars = ax2.bar(pair_labels, pair_d, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax2.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
    ax2.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
    ax2.set_ylabel("|Cohen's d|")
    ax2.set_title("Effect Size: Pairwise Comparisons\n(Green=Significant, Red=Not)", fontweight='bold')
    ax2.legend(fontsize=8)
    for bar, d in zip(bars, pair_d):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{d:.3f}', ha='center', va='bottom', fontsize=9)

    # 2-3 시각화: 산업별 ROAS 박스플롯
    ax3 = axes[1, 0]
    industry_data = {ind: df[df['industry'] == ind]['ROAS'].dropna().values for ind in industries}
    bp = ax3.boxplot(industry_data.values(), labels=industry_data.keys(),
                     patch_artist=True, notch=True)
    palette = sns.color_palette("Set2", len(industries))
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('ROAS')
    ax3.set_title(f'Industry ROAS Distribution\n(ANOVA F={f_stat_ind:.2f}, p={p_val_ind:.2e})', fontweight='bold')
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(axis='y', alpha=0.3)

    # 2-4 시각화: 플랫폼 × 산업 히트맵 (평균 ROAS)
    ax4 = axes[1, 1]
    pivot = df.pivot_table(values='ROAS', index='industry', columns='platform', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4,
                center=pivot.values.mean(), linewidths=0.5,
                cbar_kws={'label': 'Mean ROAS'})
    ax4.set_title('Platform × Industry Mean ROAS\n(Interaction Effect)', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, '07_statistical_significance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [OK] 07_statistical_significance.png saved")

    return results


# ============================================================================
# 3. 거시경제 × 광고 성과 크로스 분석
# ============================================================================
def macro_cross_analysis(df):
    """거시경제 지표와 광고 성과 간 상관관계 분석"""
    print("\n" + "=" * 80)
    print("[3] MACRO ECONOMIC × AD PERFORMANCE CROSS ANALYSIS")
    print("=" * 80)

    macro_cols = [c for c in ['cpi_index', 'cpi_yoy_pct', 'unemployment_rate',
                               'gdp_growth_pct', 'exchange_rate_usd'] if c in df.columns]

    if not macro_cols:
        print("  [SKIP] 거시경제 데이터 컬럼 없음 (enriched 데이터 필요)")
        return None

    ad_metrics = [c for c in ['ROAS', 'CPC', 'CPA', 'CTR', 'ad_spend'] if c in df.columns]

    results = {}

    # 3-1. 상관계수 행렬
    print("\n[3-1] Correlation Matrix: Macro Indicators × Ad Metrics")
    print("-" * 60)

    corr_data = {}
    for macro in macro_cols:
        corr_data[macro] = {}
        for metric in ad_metrics:
            valid = df[[macro, metric]].dropna()
            if len(valid) > 30:
                r, p = pearsonr(valid[macro], valid[metric])
                rho, p_sp = spearmanr(valid[macro], valid[metric])
                corr_data[macro][metric] = {'pearson_r': round(r, 4), 'pearson_p': round(p, 4),
                                              'spearman_rho': round(rho, 4), 'spearman_p': round(p_sp, 4)}
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {macro:>20} × {metric:<10}: r={r:>7.4f} (p={p:.2e}) {sig}")

    results['correlations'] = corr_data

    # 3-2. CPI vs ROAS 심화 분석
    if 'cpi_yoy_pct' in df.columns:
        print("\n[3-2] CPI Inflation Impact on ROAS:")
        print("-" * 60)
        df_temp = df.copy()
        df_temp['cpi_bin'] = pd.qcut(df_temp['cpi_yoy_pct'].dropna(), q=4,
                                       labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                                       duplicates='drop')
        cpi_impact = df_temp.groupby('cpi_bin')['ROAS'].agg(['mean', 'median', 'std', 'count']).round(3)
        print(cpi_impact)
        results['cpi_impact'] = cpi_impact

    # 3-3. 실업률 vs 광고 효율
    if 'unemployment_rate' in df.columns:
        print("\n[3-3] Unemployment Rate Impact on Ad Efficiency:")
        print("-" * 60)
        df_temp = df.copy()
        df_temp['unemp_bin'] = pd.qcut(df_temp['unemployment_rate'].dropna(), q=3,
                                         labels=['Low', 'Medium', 'High'],
                                         duplicates='drop')
        unemp_impact = df_temp.groupby('unemp_bin')[['ROAS', 'CPC', 'CPA']].mean().round(3)
        print(unemp_impact)
        results['unemp_impact'] = unemp_impact

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Macro Economic × Ad Performance Cross Analysis', fontsize=16, fontweight='bold')

    # 3-1 시각화: 상관계수 히트맵
    ax1 = axes[0, 0]
    corr_matrix = df[macro_cols + ad_metrics].corr().loc[macro_cols, ad_metrics]
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                ax=ax1, linewidths=0.5, cbar_kws={'label': 'Pearson r'})
    ax1.set_title('Macro × Ad Metrics Correlation', fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)

    # 3-2 시각화: CPI vs ROAS 산점도
    ax2 = axes[0, 1]
    if 'cpi_yoy_pct' in df.columns:
        sample = df[['cpi_yoy_pct', 'ROAS']].dropna()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        ax2.scatter(sample['cpi_yoy_pct'], sample['ROAS'], alpha=0.2, s=10, color='steelblue')

        # 회귀선
        z = np.polyfit(sample['cpi_yoy_pct'], sample['ROAS'], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(sample['cpi_yoy_pct'].min(), sample['cpi_yoy_pct'].max(), 100)
        ax2.plot(x_line, p_line(x_line), color='red', linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        r, p = pearsonr(sample['cpi_yoy_pct'], sample['ROAS'])
        ax2.set_title(f'CPI YoY% vs ROAS\n(r={r:.3f}, p={p:.2e})', fontweight='bold')
        ax2.set_xlabel('CPI Year-over-Year %')
        ax2.set_ylabel('ROAS')
        ax2.legend()
    ax2.grid(alpha=0.3)

    # 3-3 시각화: GDP 성장률 vs 국가별 ROAS
    ax3 = axes[1, 0]
    if 'gdp_growth_pct' in df.columns:
        country_macro = df.groupby('country').agg({
            'gdp_growth_pct': 'mean', 'ROAS': 'mean'
        }).dropna()
        ax3.scatter(country_macro['gdp_growth_pct'], country_macro['ROAS'],
                    s=100, zorder=5, color='darkorange', edgecolors='black')
        for idx, row in country_macro.iterrows():
            ax3.annotate(idx, (row['gdp_growth_pct'], row['ROAS']),
                         textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax3.set_xlabel('GDP Growth %')
        ax3.set_ylabel('Mean ROAS')
        ax3.set_title('GDP Growth vs Country-level ROAS', fontweight='bold')
    ax3.grid(alpha=0.3)

    # 3-4 시각화: 경쟁 밀도 vs ROAS (있는 경우)
    ax4 = axes[1, 1]
    if 'competition_index' in df.columns:
        sample = df[['competition_index', 'ROAS']].dropna()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        ax4.scatter(sample['competition_index'], sample['ROAS'], alpha=0.2, s=10, color='purple')
        z = np.polyfit(sample['competition_index'], sample['ROAS'], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(sample['competition_index'].min(), sample['competition_index'].max(), 100)
        ax4.plot(x_line, p_line(x_line), color='red', linewidth=2)
        r, p = pearsonr(sample['competition_index'], sample['ROAS'])
        ax4.set_title(f'Competition Index vs ROAS\n(r={r:.3f}, p={p:.2e})', fontweight='bold')
        ax4.set_xlabel('Competition Index')
        ax4.set_ylabel('ROAS')
    elif 'exchange_rate_usd' in df.columns:
        sample = df[['exchange_rate_usd', 'ROAS']].dropna()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        ax4.scatter(sample['exchange_rate_usd'], sample['ROAS'], alpha=0.2, s=10, color='purple')
        ax4.set_title('Exchange Rate vs ROAS', fontweight='bold')
        ax4.set_xlabel('Exchange Rate (USD)')
        ax4.set_ylabel('ROAS')
    ax4.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, '08_macro_cross_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [OK] 08_macro_cross_analysis.png saved")

    return results


# ============================================================================
# 4. 합성 데이터 영향 분석
# ============================================================================
def synthetic_data_impact(df):
    """합성 데이터가 분석 결과에 미치는 영향 비교"""
    print("\n" + "=" * 80)
    print("[4] SYNTHETIC DATA IMPACT ANALYSIS")
    print("=" * 80)

    if 'is_synthetic' not in df.columns:
        print("  [SKIP] is_synthetic 컬럼 없음 (합성 데이터 구분 불가)")
        return None

    real = df[df['is_synthetic'] == 0]
    synth = df[df['is_synthetic'] == 1]

    print(f"\n  실제 데이터: {len(real):,}건 ({len(real)/len(df)*100:.1f}%)")
    print(f"  합성 데이터: {len(synth):,}건 ({len(synth)/len(df)*100:.1f}%)")

    results = {
        'real_count': len(real),
        'synthetic_count': len(synth),
        'real_pct': round(len(real) / len(df) * 100, 1),
    }

    # 4-1. 주요 지표 비교
    print("\n[4-1] Key Metrics Comparison: Real vs Synthetic")
    print("-" * 60)
    metrics = [c for c in ['ROAS', 'CPC', 'CPA', 'CTR', 'ad_spend', 'revenue'] if c in df.columns]
    comparison = {}
    for m in metrics:
        real_vals = real[m].dropna()
        synth_vals = synth[m].dropna()
        all_vals = df[m].dropna()

        if len(real_vals) > 0 and len(synth_vals) > 0:
            t_stat, p_val = ttest_ind(real_vals, synth_vals, equal_var=False)
            comparison[m] = {
                'real_mean': round(real_vals.mean(), 3),
                'synth_mean': round(synth_vals.mean(), 3),
                'all_mean': round(all_vals.mean(), 3),
                'diff_pct': round((synth_vals.mean() - real_vals.mean()) / real_vals.mean() * 100, 2),
                't_stat': round(t_stat, 3),
                'p_value': round(p_val, 4),
                'significant': p_val < 0.05,
            }
            sig = "*" if p_val < 0.05 else ""
            print(f"  {m:>10}: Real={real_vals.mean():.3f}  Synth={synth_vals.mean():.3f}  "
                  f"Diff={comparison[m]['diff_pct']:>+6.2f}%  (t={t_stat:.2f}, p={p_val:.3f}) {sig}")

    results['metric_comparison'] = comparison

    # 4-2. 분석 결론 영향도: 플랫폼 순위 비교
    print("\n[4-2] Platform Ranking Impact:")
    print("-" * 60)
    for dataset_name, dataset in [("전체", df), ("실제만", real)]:
        ranking = dataset.groupby('platform')['ROAS'].mean().sort_values(ascending=False)
        print(f"  [{dataset_name}]")
        for rank, (platform, roas) in enumerate(ranking.items(), 1):
            print(f"    {rank}. {platform}: ROAS {roas:.3f}")

    # 4-3. 산업 순위 비교
    print("\n[4-3] Industry Ranking Impact:")
    print("-" * 60)
    for dataset_name, dataset in [("전체", df), ("실제만", real)]:
        ranking = dataset.groupby('industry')['ROAS'].mean().sort_values(ascending=False)
        print(f"  [{dataset_name}]")
        for rank, (industry, roas) in enumerate(ranking.items(), 1):
            print(f"    {rank}. {industry}: ROAS {roas:.3f}")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Synthetic Data Impact Analysis', fontsize=16, fontweight='bold')

    # 4-1 시각화: 주요 지표 비교 (Real vs Synthetic)
    ax1 = axes[0, 0]
    comp_metrics = list(comparison.keys())[:5]
    x = np.arange(len(comp_metrics))
    width = 0.35
    real_vals_plot = [comparison[m]['real_mean'] for m in comp_metrics]
    synth_vals_plot = [comparison[m]['synth_mean'] for m in comp_metrics]
    ax1.bar(x - width/2, real_vals_plot, width, label='Real Data', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, synth_vals_plot, width, label='Synthetic Data', color='coral', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_metrics, rotation=30)
    ax1.set_title('Mean Comparison: Real vs Synthetic', fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 4-2 시각화: ROAS 분포 비교
    ax2 = axes[0, 1]
    real_roas = real['ROAS'].dropna().clip(0, real['ROAS'].quantile(0.99))
    synth_roas = synth['ROAS'].dropna().clip(0, synth['ROAS'].quantile(0.99))
    ax2.hist(real_roas, bins=50, density=True, alpha=0.6, color='steelblue', label='Real', edgecolor='white')
    ax2.hist(synth_roas, bins=50, density=True, alpha=0.6, color='coral', label='Synthetic', edgecolor='white')
    ax2.axvline(real_roas.mean(), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(synth_roas.mean(), color='red', linestyle='--', linewidth=2)
    ax2.set_title('ROAS Distribution: Real vs Synthetic', fontweight='bold')
    ax2.set_xlabel('ROAS')
    ax2.legend()

    # 4-3 시각화: 플랫폼 순위 변동
    ax3 = axes[1, 0]
    all_rank = df.groupby('platform')['ROAS'].mean().sort_values(ascending=True)
    real_rank = real.groupby('platform')['ROAS'].mean().reindex(all_rank.index)
    y_pos = np.arange(len(all_rank))
    ax3.barh(y_pos - 0.2, all_rank.values, 0.35, label='All Data', color='steelblue', alpha=0.8)
    ax3.barh(y_pos + 0.2, real_rank.values, 0.35, label='Real Only', color='forestgreen', alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(all_rank.index)
    ax3.set_xlabel('Mean ROAS')
    ax3.set_title('Platform ROAS: All Data vs Real Only', fontweight='bold')
    ax3.legend()

    # 4-4 시각화: 편차 비율
    ax4 = axes[1, 1]
    diff_pcts = [comparison[m]['diff_pct'] for m in comp_metrics]
    colors_diff = ['#2ecc71' if abs(d) < 5 else '#f39c12' if abs(d) < 15 else '#e74c3c' for d in diff_pcts]
    bars = ax4.bar(comp_metrics, diff_pcts, color=colors_diff, edgecolor='white', linewidth=2)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.axhline(5, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(-5, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Difference (%)')
    ax4.set_title('Synthetic vs Real: Metric Deviation %\n(Green < 5%, Yellow < 15%, Red >= 15%)', fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, diff_pcts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, '09_synthetic_impact.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [OK] 09_synthetic_impact.png saved")

    return results


# ============================================================================
# 5. 다중 요인 성과 분해
# ============================================================================
def factor_decomposition(df):
    """다중 회귀를 활용한 ROAS 성과 요인 분해"""
    print("\n" + "=" * 80)
    print("[5] FACTOR DECOMPOSITION (Multiple Regression)")
    print("=" * 80)

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # 수치형 피처 선택 (leakage 변수를 별도 표시)
    LEAKAGE_FEATURES = ['landing_page_load_time', 'bounce_rate', 'creative_impact_factor']

    feature_candidates = [
        'ad_spend', 'CTR', 'CPC',
        'cpi_yoy_pct', 'unemployment_rate', 'gdp_growth_pct', 'exchange_rate_usd',
        'trend_index', 'trend_momentum',
        'is_shopping_season', 'season_intensity', 'is_public_holiday',
        'competition_index', 'auction_density',
        'is_retargeting', 'is_video', 'is_lookalike',
        'landing_page_load_time', 'bounce_rate',
    ]

    available = [c for c in feature_candidates if c in df.columns]
    available_clean = [c for c in available if c not in LEAKAGE_FEATURES]

    if len(available) < 3:
        print("  [SKIP] 충분한 피처 없음")
        return None

    leakage_in_features = [c for c in available if c in LEAKAGE_FEATURES]
    if leakage_in_features:
        print(f"\n  ⚠ [WARNING] Leakage 변수 포함: {leakage_in_features}")
        print(f"    -> Leakage 포함/제외 두 가지 결과를 모두 보고합니다.")

    print(f"\n  분석 피처 (전체): {len(available)}개")
    print(f"  분석 피처 (leakage-free): {len(available_clean)}개")

    # 결측치 제거
    analysis_df = df[available + ['ROAS']].dropna()
    print(f"  분석 가능 데이터: {len(analysis_df)}건")

    # --- VIF (Variance Inflation Factor) 다중공선성 검사 ---
    print(f"\n[5-0] Multicollinearity Check (VIF):")
    print("-" * 60)
    X_for_vif = analysis_df[available_clean]  # leakage 제외 피처로 VIF 계산

    try:
        from numpy.linalg import LinAlgError
        vif_results = []
        for i, col in enumerate(X_for_vif.columns):
            other_cols = [c for c in X_for_vif.columns if c != col]
            if len(other_cols) == 0:
                continue
            X_other = X_for_vif[other_cols].values
            y_col = X_for_vif[col].values

            # R² 계산
            from sklearn.linear_model import LinearRegression as LR_vif
            lr_vif = LR_vif()
            lr_vif.fit(X_other, y_col)
            r2_i = lr_vif.score(X_other, y_col)
            vif = 1 / (1 - r2_i) if r2_i < 1 else float('inf')
            vif_results.append({'feature': col, 'VIF': round(vif, 2), 'R²': round(r2_i, 4)})

            status = "⚠ HIGH" if vif > 10 else "△ MODERATE" if vif > 5 else "OK"
            print(f"  {col:>25}: VIF={vif:>8.2f}  {status}")

        high_vif = [v for v in vif_results if v['VIF'] > 10]
        if high_vif:
            print(f"\n  [WARNING] VIF > 10 피처 {len(high_vif)}개 -- 다중공선성이 높아 계수 해석 시 주의:")
            for v in high_vif:
                print(f"    {v['feature']}: VIF={v['VIF']}")
        else:
            print(f"\n  [OK] 모든 피처 VIF ≤ 10 -- 심각한 다중공선성 없음")
    except Exception as e:
        vif_results = []
        print(f"  [WARN] VIF 계산 실패: {e}")

    X = analysis_df[available]
    y = analysis_df['ROAS']

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 선형 회귀 (표준화 계수로 상대적 기여도 파악)
    lr = LinearRegression()
    lr.fit(X_scaled, y)

    coef_df = pd.DataFrame({
        'feature': available,
        'std_coefficient': lr.coef_,
        'abs_coefficient': np.abs(lr.coef_),
        'is_leakage': [c in LEAKAGE_FEATURES for c in available],
    }).sort_values('abs_coefficient', ascending=False)

    r2_all = lr.score(X_scaled, y)
    print(f"\n  R² Score (전체 피처): {r2_all:.4f}")

    # Leakage-free 모델도 함께 보고
    r2_clean = None
    if available_clean and len(available_clean) >= 3:
        X_clean = analysis_df[available_clean]
        scaler_clean = StandardScaler()
        X_clean_scaled = scaler_clean.fit_transform(X_clean)
        lr_clean = LinearRegression()
        lr_clean.fit(X_clean_scaled, y)
        r2_clean = lr_clean.score(X_clean_scaled, y)
        print(f"  R² Score (leakage-free): {r2_clean:.4f}")
        print(f"  R² Drop (leakage 기여): {r2_all - r2_clean:.4f}")

    print(f"\n[5-1] Standardized Coefficients (Feature Importance):")
    print("-" * 60)
    for _, row in coef_df.iterrows():
        direction = "+" if row['std_coefficient'] > 0 else "-"
        bar = "█" * int(row['abs_coefficient'] * 10)
        leak_tag = " ⚠LEAKAGE" if row['is_leakage'] else ""
        print(f"  {row['feature']:>25}: {direction}{row['abs_coefficient']:.4f}  {bar}{leak_tag}")

    results = {
        'r2_all': round(r2_all, 4),
        'r2_clean': round(r2_clean, 4) if r2_clean is not None else None,
        'r2_leakage_contribution': round(r2_all - r2_clean, 4) if r2_clean is not None else None,
        'vif': vif_results,
        'coefficients': coef_df[['feature', 'std_coefficient', 'abs_coefficient', 'is_leakage']].to_dict('records'),
    }

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('ROAS Factor Decomposition', fontsize=16, fontweight='bold')

    # 5-1: 표준화 계수 (방향 포함)
    ax1 = axes[0]
    top_n = min(15, len(coef_df))
    top_features = coef_df.head(top_n)
    colors = ['#f39c12' if leak else ('#2ecc71' if c > 0 else '#e74c3c')
              for c, leak in zip(top_features['std_coefficient'], top_features['is_leakage'])]
    ax1.barh(range(top_n), top_features['std_coefficient'].values, color=colors, edgecolor='white')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('Standardized Coefficient')
    ax1.set_title(f'Top {top_n} Features by Impact\n(R²={lr.score(X_scaled, y):.3f})', fontweight='bold')
    ax1.invert_yaxis()

    # 5-2: 절대 기여도 파이 차트
    ax2 = axes[1]
    top_8 = coef_df.head(8)
    other_sum = coef_df.iloc[8:]['abs_coefficient'].sum()
    pie_labels = list(top_8['feature']) + ['Others']
    pie_values = list(top_8['abs_coefficient']) + [other_sum]
    colors_pie = sns.color_palette("Set3", len(pie_labels))
    wedges, texts, autotexts = ax2.pie(
        pie_values, labels=pie_labels, autopct='%1.1f%%',
        colors=colors_pie, startangle=90, pctdistance=0.85
    )
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(7)
    ax2.set_title('Relative Feature Contribution\n(Absolute Coefficients)', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(FIGURES_DIR, '10_factor_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [OK] 10_factor_decomposition.png saved")

    return results


# ============================================================================
# 메인 실행
# ============================================================================
def run_all():
    """전체 고급 분석 실행"""
    print("=" * 80)
    print("  ADVANCED STATISTICAL ANALYSIS")
    print("  통계적 유의성 검증 / 크로스 데이터 분석 / 합성 데이터 영향 분석")
    print("=" * 80)

    df, is_enriched = load_data()

    all_results = {}

    # 1. 데이터 품질 감사
    all_results['quality'] = data_quality_audit(df)

    # 2. 통계적 유의성 검증
    all_results['significance'] = statistical_significance_tests(df)

    # 3. 거시경제 크로스 분석
    if is_enriched:
        all_results['macro'] = macro_cross_analysis(df)

    # 4. 합성 데이터 영향
    if is_enriched:
        all_results['synthetic'] = synthetic_data_impact(df)

    # 5. 요인 분해
    if is_enriched:
        all_results['factors'] = factor_decomposition(df)

    # 종합 요약
    print("\n" + "=" * 80)
    print("  ANALYSIS SUMMARY")
    print("=" * 80)

    sig = all_results.get('significance', {})
    anova_p = sig.get('anova_platform_roas', {})
    print(f"""
  [통계적 유의성]
    - 플랫폼 간 ROAS 차이: {'유의' if anova_p.get('p', 1) < 0.05 else '비유의'} (p={anova_p.get('p', 'N/A')})
    - 산업 간 ROAS 차이: {'유의' if sig.get('anova_industry_roas', {}).get('p', 1) < 0.05 else '비유의'}
    - 국가 간 ROAS 차이: {'유의' if sig.get('anova_country_roas', {}).get('p', 1) < 0.05 else '비유의'}
    - 플랫폼-고성과 연관: Cramer's V = {sig.get('chi2_platform_highperf', {}).get('cramers_v', 'N/A')}
""")

    synth = all_results.get('synthetic', {})
    if synth:
        print(f"""  [합성 데이터 영향]
    - 실제 데이터 비율: {synth.get('real_pct', 'N/A')}%
    - 주요 지표 편차: """, end="")
        for m, v in synth.get('metric_comparison', {}).items():
            print(f"{m}({v['diff_pct']:+.1f}%) ", end="")
        print()

    factors = all_results.get('factors', {})
    if factors:
        print(f"\n  [요인 분해]")
        print(f"    - 다중 회귀 R²: {factors.get('r2', 'N/A')}")
        print(f"    - 상위 3 요인: ", end="")
        for c in factors.get('coefficients', [])[:3]:
            print(f"{c['feature']}({c['std_coefficient']:+.3f}) ", end="")
        print()

    print(f"""
  [생성된 시각화]
    - 06_data_quality_audit.png
    - 07_statistical_significance.png
    - 08_macro_cross_analysis.png
    - 09_synthetic_impact.png
    - 10_factor_decomposition.png
""")
    # JSON 결과 내보내기 (보고서 작성 지원)
    import json

    def convert_to_serializable(obj):
        """numpy/pandas 객체를 JSON 직렬화 가능 형태로 변환"""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return str(obj)

    try:
        json_path = 'analysis_results.json'
        json_results = {}

        # ANOVA 결과 추출
        sig = all_results.get('significance', {})
        json_results['anova'] = {
            'platform_roas': sig.get('anova_platform_roas', {}),
            'industry_roas': sig.get('anova_industry_roas', {}),
            'country_roas': sig.get('anova_country_roas', {}),
        }
        json_results['pairwise'] = sig.get('pairwise_platform', [])
        json_results['chi2'] = sig.get('chi2_platform_highperf', {})

        # 합성 데이터 영향
        synth_data = all_results.get('synthetic', {})
        if synth_data:
            json_results['synthetic_impact'] = {
                'real_count': synth_data.get('real_count'),
                'synthetic_count': synth_data.get('synthetic_count'),
                'real_pct': synth_data.get('real_pct'),
                'metric_comparison': synth_data.get('metric_comparison', {}),
            }

        # 요인 분해
        factors_data = all_results.get('factors', {})
        if factors_data:
            json_results['factor_decomposition'] = {
                'r2_all': factors_data.get('r2_all'),
                'r2_clean': factors_data.get('r2_clean'),
                'vif': factors_data.get('vif', []),
                'top_5_coefficients': factors_data.get('coefficients', [])[:5],
            }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2, default=convert_to_serializable)
        print(f"\n  [OK] 분석 결과 JSON 내보내기: {json_path}")
        print(f"    -> 보고서 작성 시 이 파일의 수치를 참조하세요.")
    except Exception as e:
        print(f"  [WARN] JSON 내보내기 실패: {e}")

    print("=" * 80)
    print("  COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    run_all()
