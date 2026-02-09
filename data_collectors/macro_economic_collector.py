# -*- coding: utf-8 -*-
"""
Phase 2-1: 거시경제 지표 수집 파이프라인
- World Bank API: GDP 성장률, 실업률
- FRED API: CPI (소비자물가지수)
- ExchangeRate API: 환율 (USD 기준)
- 모든 데이터를 country + month 단위로 조인 가능하게 정리

country 매핑: UAE, UK, USA, Germany, Canada, India, Australia
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# HTTP 요청
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# FRED API (선택적)
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


# ===========================================================================
# 국가 코드 매핑
# ===========================================================================
COUNTRY_CODES = {
    'USA': {'iso2': 'US', 'iso3': 'USA', 'wb': 'USA', 'currency': 'USD'},
    'UK': {'iso2': 'GB', 'iso3': 'GBR', 'wb': 'GBR', 'currency': 'GBP'},
    'Germany': {'iso2': 'DE', 'iso3': 'DEU', 'wb': 'DEU', 'currency': 'EUR'},
    'Canada': {'iso2': 'CA', 'iso3': 'CAN', 'wb': 'CAN', 'currency': 'CAD'},
    'India': {'iso2': 'IN', 'iso3': 'IND', 'wb': 'IND', 'currency': 'INR'},
    'UAE': {'iso2': 'AE', 'iso3': 'ARE', 'wb': 'ARE', 'currency': 'AED'},
    'Australia': {'iso2': 'AU', 'iso3': 'AUS', 'wb': 'AUS', 'currency': 'AUD'},
}

# FRED 시리즈 ID (국가별 CPI)
FRED_CPI_SERIES = {
    'USA': 'CPIAUCSL',          # US CPI (All Urban Consumers)
    'UK': 'GBRCPIALLMINMEI',    # UK CPI
    'Germany': 'DEUCPIALLMINMEI', # Germany CPI
    'Canada': 'CANCPIALLMINMEI',  # Canada CPI
    'India': 'INDCPIALLMINMEI',   # India CPI
    'Australia': 'AUSCPIALLMINMEI', # Australia CPI
    # UAE는 FRED에 CPI 시리즈가 제한적
}

# FRED 실업률 시리즈 ID
FRED_UNEMPLOYMENT_SERIES = {
    'USA': 'UNRATE',
    'UK': 'LMUNRRTTGBM156S',
    'Germany': 'LMUNRRTTDEM156S',
    'Canada': 'LMUNRRTTCAM156S',
    'India': 'LMUNRRTTINM156S',
    'Australia': 'LMUNRRTTAUM156S',
}


class MacroEconomicCollector:
    """거시경제 지표를 수집하여 광고 데이터와 조인 가능한 형태로 제공"""

    def __init__(self, fred_api_key=None, output_dir='./data/macro'):
        self.fred_api_key = fred_api_key or os.environ.get('FRED_API_KEY', '')
        self.output_dir = output_dir
        self.macro_data = pd.DataFrame()

        os.makedirs(output_dir, exist_ok=True)
        print("[Phase 2-1] Macro Economic Collector initialized")

    # -----------------------------------------------------------------------
    # World Bank API: GDP 성장률
    # -----------------------------------------------------------------------
    def fetch_gdp_growth(self, start_year=2020, end_year=2025):
        """
        World Bank API에서 국가별 GDP 성장률(연간) 수집
        indicator: NY.GDP.MKTP.KD.ZG
        """
        if not REQUESTS_AVAILABLE:
            print("[WARNING] requests 패키지 필요: pip install requests")
            return self._generate_fallback_gdp(start_year, end_year)

        print(f"\n[Phase 2-1] Fetching GDP growth rates ({start_year}-{end_year})...")
        indicator = 'NY.GDP.MKTP.KD.ZG'
        all_data = []

        for country_name, codes in COUNTRY_CODES.items():
            wb_code = codes['wb']
            url = (
                f"https://api.worldbank.org/v2/country/{wb_code}/indicator/{indicator}"
                f"?date={start_year}:{end_year}&format=json&per_page=100"
            )
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if len(data) > 1 and data[1]:
                        for entry in data[1]:
                            if entry.get('value') is not None:
                                all_data.append({
                                    'country': country_name,
                                    'year': int(entry['date']),
                                    'gdp_growth_pct': round(float(entry['value']), 2),
                                })
                        print(f"  {country_name}: {len([e for e in data[1] if e.get('value')])} years")
                    else:
                        print(f"  {country_name}: No data available")
                else:
                    print(f"  {country_name}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"  {country_name}: Error - {str(e)[:60]}")

        if not all_data:
            print("  [FALLBACK] Using estimated GDP data")
            return self._generate_fallback_gdp(start_year, end_year)

        df = pd.DataFrame(all_data)
        print(f"  Total GDP records: {len(df)}")
        return df

    def _generate_fallback_gdp(self, start_year, end_year):
        """World Bank API 실패 시 추정 GDP 데이터 생성"""
        # 2022-2024 대략적 GDP 성장률 (IMF 추정치 기반)
        estimates = {
            'USA':       {2020: -2.8, 2021: 5.9, 2022: 2.1, 2023: 2.5, 2024: 2.7, 2025: 2.0},
            'UK':        {2020: -11.0, 2021: 7.6, 2022: 4.3, 2023: 0.1, 2024: 0.5, 2025: 1.5},
            'Germany':   {2020: -3.7, 2021: 2.6, 2022: 1.8, 2023: -0.3, 2024: 0.2, 2025: 1.3},
            'Canada':    {2020: -5.2, 2021: 5.0, 2022: 3.4, 2023: 1.1, 2024: 1.2, 2025: 2.4},
            'India':     {2020: -5.8, 2021: 9.1, 2022: 7.2, 2023: 7.8, 2024: 6.8, 2025: 6.5},
            'UAE':       {2020: -4.8, 2021: 4.4, 2022: 7.9, 2023: 3.6, 2024: 3.9, 2025: 4.0},
            'Australia': {2020: -0.8, 2021: 5.6, 2022: 3.7, 2023: 2.0, 2024: 1.5, 2025: 2.2},
        }
        rows = []
        for country, yearly in estimates.items():
            for year in range(start_year, end_year + 1):
                if year in yearly:
                    rows.append({
                        'country': country,
                        'year': year,
                        'gdp_growth_pct': yearly[year],
                    })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # FRED API: CPI & 실업률
    # -----------------------------------------------------------------------
    def fetch_cpi_data(self, start_date='2020-01-01'):
        """FRED API에서 국가별 CPI(월간) 수집"""
        if FRED_AVAILABLE and self.fred_api_key:
            return self._fetch_cpi_from_fred(start_date)
        else:
            print("[Phase 2-1] FRED API unavailable, using fallback CPI data")
            return self._generate_fallback_cpi(start_date)

    def _fetch_cpi_from_fred(self, start_date):
        """FRED API 직접 호출"""
        fred = Fred(api_key=self.fred_api_key)
        all_data = []

        print(f"\n[Phase 2-1] Fetching CPI data from FRED...")
        for country, series_id in FRED_CPI_SERIES.items():
            try:
                series = fred.get_series(series_id, observation_start=start_date)
                for date, value in series.items():
                    if pd.notna(value):
                        all_data.append({
                            'country': country,
                            'date': date,
                            'year': date.year,
                            'month': date.month,
                            'cpi_index': round(float(value), 2),
                        })
                print(f"  {country}: {len(series.dropna())} months")
            except Exception as e:
                print(f"  {country}: Error - {str(e)[:60]}")

        if not all_data:
            return self._generate_fallback_cpi(start_date)

        df = pd.DataFrame(all_data)
        # CPI 변화율 계산 (YoY)
        df = df.sort_values(['country', 'date'])
        df['cpi_yoy_pct'] = df.groupby('country')['cpi_index'].pct_change(12) * 100
        return df

    def _generate_fallback_cpi(self, start_date='2020-01-01'):
        """FRED 불가 시 추정 CPI 데이터"""
        # 국가별 월간 인플레이션 추이 (대략적 추정)
        base_inflation = {
            'USA': 3.2, 'UK': 4.0, 'Germany': 3.5, 'Canada': 3.0,
            'India': 5.5, 'UAE': 2.5, 'Australia': 3.8,
        }

        rows = []
        dates = pd.date_range(start=start_date, end='2024-12-31', freq='MS')
        for country, base_rate in base_inflation.items():
            cpi = 100.0
            for i, date in enumerate(dates):
                # 시간에 따른 변동 (2022년 인플레 급등, 2023-2024 하락 추세)
                if date.year == 2022:
                    rate = base_rate * 1.8
                elif date.year == 2023:
                    rate = base_rate * 1.2
                else:
                    rate = base_rate

                monthly_rate = rate / 12 / 100
                noise = np.random.normal(0, 0.002)
                cpi *= (1 + monthly_rate + noise)

                rows.append({
                    'country': country,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'cpi_index': round(cpi, 2),
                })

        df = pd.DataFrame(rows)
        df = df.sort_values(['country', 'date'])
        df['cpi_yoy_pct'] = df.groupby('country')['cpi_index'].pct_change(12) * 100
        return df

    def fetch_unemployment(self, start_date='2020-01-01'):
        """FRED에서 국가별 실업률(월간) 수집"""
        if FRED_AVAILABLE and self.fred_api_key:
            return self._fetch_unemployment_from_fred(start_date)
        else:
            print("[Phase 2-1] FRED API unavailable, using fallback unemployment data")
            return self._generate_fallback_unemployment(start_date)

    def _fetch_unemployment_from_fred(self, start_date):
        """FRED에서 직접 실업률 수집"""
        fred = Fred(api_key=self.fred_api_key)
        all_data = []

        print(f"\n[Phase 2-1] Fetching unemployment rates from FRED...")
        for country, series_id in FRED_UNEMPLOYMENT_SERIES.items():
            try:
                series = fred.get_series(series_id, observation_start=start_date)
                for date, value in series.items():
                    if pd.notna(value):
                        all_data.append({
                            'country': country,
                            'date': date,
                            'year': date.year,
                            'month': date.month,
                            'unemployment_rate': round(float(value), 2),
                        })
                print(f"  {country}: {len(series.dropna())} months")
            except Exception as e:
                print(f"  {country}: Error - {str(e)[:60]}")

        if not all_data:
            return self._generate_fallback_unemployment(start_date)
        return pd.DataFrame(all_data)

    def _generate_fallback_unemployment(self, start_date='2020-01-01'):
        """추정 실업률 데이터"""
        base_rates = {
            'USA': 3.7, 'UK': 4.2, 'Germany': 3.0, 'Canada': 5.5,
            'India': 7.5, 'UAE': 2.6, 'Australia': 3.7,
        }
        rows = []
        dates = pd.date_range(start=start_date, end='2024-12-31', freq='MS')
        for country, base in base_rates.items():
            for date in dates:
                # COVID 효과
                if date.year == 2020 and date.month >= 3:
                    rate = base * 2.0
                elif date.year == 2021:
                    rate = base * 1.3
                else:
                    rate = base

                noise = np.random.normal(0, 0.3)
                rows.append({
                    'country': country,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'unemployment_rate': round(max(0.5, rate + noise), 2),
                })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # 환율 데이터
    # -----------------------------------------------------------------------
    def fetch_exchange_rates(self, start_date='2020-01-01'):
        """환율 데이터 수집 (USD 기준)"""
        if not REQUESTS_AVAILABLE:
            return self._generate_fallback_exchange_rates(start_date)

        print(f"\n[Phase 2-1] Fetching exchange rates...")
        # exchangerate.host 또는 fallback
        rows = []
        dates = pd.date_range(start=start_date, end='2024-12-31', freq='MS')

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            url = f"https://api.exchangerate.host/{date_str}?base=USD"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('success') and data.get('rates'):
                        rates = data['rates']
                        for country, codes in COUNTRY_CODES.items():
                            currency = codes['currency']
                            if currency in rates and currency != 'USD':
                                rows.append({
                                    'country': country,
                                    'date': date,
                                    'year': date.year,
                                    'month': date.month,
                                    'exchange_rate_usd': round(float(rates[currency]), 4),
                                    'currency': currency,
                                })
            except Exception:
                pass

        if not rows:
            print("  [FALLBACK] Using estimated exchange rates")
            return self._generate_fallback_exchange_rates(start_date)

        print(f"  Total exchange rate records: {len(rows)}")
        return pd.DataFrame(rows)

    def _generate_fallback_exchange_rates(self, start_date='2020-01-01'):
        """추정 환율 데이터"""
        base_rates = {
            'USA': 1.0, 'UK': 0.79, 'Germany': 0.92, 'Canada': 1.36,
            'India': 83.0, 'UAE': 3.67, 'Australia': 1.53,
        }
        rows = []
        dates = pd.date_range(start=start_date, end='2024-12-31', freq='MS')
        for country, base in base_rates.items():
            for date in dates:
                noise = np.random.normal(0, base * 0.02)
                rows.append({
                    'country': country,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'exchange_rate_usd': round(max(0.01, base + noise), 4),
                    'currency': COUNTRY_CODES[country]['currency'],
                })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # 통합 매크로 데이터 생성
    # -----------------------------------------------------------------------
    def collect_all(self, start_date='2020-01-01', end_year=2025):
        """모든 거시경제 지표를 수집하여 통합 데이터프레임 생성"""
        print("\n" + "=" * 70)
        print("[Phase 2-1] Collecting all macro economic indicators")
        print("=" * 70)

        start_year = int(start_date[:4])

        # 1. GDP 성장률 (연간)
        gdp_df = self.fetch_gdp_growth(start_year, end_year)

        # 2. CPI (월간)
        cpi_df = self.fetch_cpi_data(start_date)

        # 3. 실업률 (월간)
        unemp_df = self.fetch_unemployment(start_date)

        # 4. 환율 (월간)
        fx_df = self.fetch_exchange_rates(start_date)

        # --- 통합 ---
        # 월간 데이터를 기준으로 병합
        if len(cpi_df) > 0:
            macro = cpi_df[['country', 'year', 'month', 'cpi_index', 'cpi_yoy_pct']].copy()
        else:
            # 빈 프레임 생성
            dates = pd.date_range(start=start_date, end='2024-12-31', freq='MS')
            rows = []
            for country in COUNTRY_CODES.keys():
                for d in dates:
                    rows.append({'country': country, 'year': d.year, 'month': d.month})
            macro = pd.DataFrame(rows)

        # 실업률 조인
        if len(unemp_df) > 0:
            macro = macro.merge(
                unemp_df[['country', 'year', 'month', 'unemployment_rate']],
                on=['country', 'year', 'month'], how='left'
            )

        # GDP 조인 (연간 -> 월간으로 브로드캐스트)
        if len(gdp_df) > 0:
            macro = macro.merge(
                gdp_df[['country', 'year', 'gdp_growth_pct']],
                on=['country', 'year'], how='left'
            )

        # 환율 조인
        if len(fx_df) > 0:
            macro = macro.merge(
                fx_df[['country', 'year', 'month', 'exchange_rate_usd']],
                on=['country', 'year', 'month'], how='left'
            )

        # 결측치 보간
        for col in ['cpi_index', 'cpi_yoy_pct', 'unemployment_rate',
                     'gdp_growth_pct', 'exchange_rate_usd']:
            if col in macro.columns:
                macro[col] = macro.groupby('country')[col].transform(
                    lambda x: x.interpolate(method='linear').ffill().bfill()
                )

        self.macro_data = macro
        print(f"\n[Phase 2-1] Combined macro data: {len(macro)} rows")
        print(f"  Columns: {list(macro.columns)}")
        print(f"  Countries: {macro['country'].nunique()}")
        print(f"  Year range: {macro['year'].min()} ~ {macro['year'].max()}")

        return macro

    def save(self, output_path=None):
        """매크로 데이터 저장"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'macro_economic_data.csv')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.macro_data.to_csv(output_path, index=False)
        print(f"[Phase 2-1] Saved: {output_path} ({len(self.macro_data)} rows)")
        return output_path

    def enrich_ads_data(self, ads_df):
        """광고 데이터에 매크로 지표 조인"""
        if self.macro_data.empty:
            self.collect_all()

        ads = ads_df.copy()
        ads['date'] = pd.to_datetime(ads['date'])
        ads['year'] = ads['date'].dt.year
        ads['month'] = ads['date'].dt.month

        enriched = ads.merge(
            self.macro_data,
            on=['country', 'year', 'month'],
            how='left'
        )

        # 임시 컬럼 제거
        if 'year' not in ads_df.columns:
            enriched = enriched.drop(columns=['year'], errors='ignore')
        if 'month' not in ads_df.columns:
            enriched = enriched.drop(columns=['month'], errors='ignore')

        added_cols = [c for c in enriched.columns if c not in ads_df.columns]
        print(f"[Phase 2-1] Enriched ads data with {len(added_cols)} macro columns: {added_cols}")

        return enriched


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    collector = MacroEconomicCollector(
        fred_api_key=os.environ.get('FRED_API_KEY', ''),
        output_dir='./data/macro'
    )

    # 전체 수집
    macro_df = collector.collect_all(start_date='2020-01-01')

    # 저장
    collector.save()

    # 광고 데이터 보강 예시
    if os.path.exists('global_ads_performance_dataset.csv'):
        ads_df = pd.read_csv('global_ads_performance_dataset.csv')
        enriched = collector.enrich_ads_data(ads_df)
        print(f"\nEnriched dataset shape: {enriched.shape}")
        print(f"New columns: {[c for c in enriched.columns if c not in ads_df.columns]}")
        enriched.to_csv('./data/ads_with_macro.csv', index=False)
