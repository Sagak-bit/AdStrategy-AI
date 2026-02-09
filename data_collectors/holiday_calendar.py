# -*- coding: utf-8 -*-
"""
Phase 2-2: 시즌/공휴일 달력 데이터 매핑 테이블
- 국가별 공휴일 (holidays 패키지 또는 수동 매핑)
- 글로벌 쇼핑 시즌 플래그 (블랙프라이데이, 광군제, 연말 등)
- 주요 글로벌 이벤트 달력
- 피처: is_holiday, days_to_next_holiday, shopping_season, major_event
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# holidays 패키지 (선택적)
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False


# ===========================================================================
# 국가별 공휴일 데이터
# ===========================================================================
# holidays 패키지의 국가 코드 매핑
HOLIDAY_COUNTRY_MAP = {
    'USA': 'US',
    'UK': 'GB',
    'Germany': 'DE',
    'Canada': 'CA',
    'India': 'IN',
    'UAE': 'AE',
    'Australia': 'AU',
}

# ===========================================================================
# 글로벌 쇼핑 시즌 정의 (연도 독립적 월/일 범위)
# ===========================================================================
SHOPPING_SEASONS = [
    {
        'name': 'New Year Sales',
        'start_month': 1, 'start_day': 1,
        'end_month': 1, 'end_day': 15,
        'intensity': 'medium',
        'industries': ['E-commerce', 'SaaS', 'Fintech'],
    },
    {
        'name': 'Valentines Day',
        'start_month': 2, 'start_day': 7,
        'end_month': 2, 'end_day': 14,
        'intensity': 'medium',
        'industries': ['E-commerce'],
    },
    {
        'name': 'Back to School',
        'start_month': 8, 'start_day': 1,
        'end_month': 9, 'end_day': 15,
        'intensity': 'high',
        'industries': ['EdTech', 'E-commerce'],
    },
    {
        'name': 'Singles Day (11.11)',
        'start_month': 11, 'start_day': 8,
        'end_month': 11, 'end_day': 12,
        'intensity': 'high',
        'industries': ['E-commerce', 'SaaS'],
    },
    {
        'name': 'Black Friday / Cyber Monday',
        'start_month': 11, 'start_day': 20,
        'end_month': 12, 'end_day': 2,
        'intensity': 'very_high',
        'industries': ['E-commerce', 'SaaS', 'Fintech', 'EdTech', 'Healthcare'],
    },
    {
        'name': 'Holiday Season (Christmas)',
        'start_month': 12, 'start_day': 3,
        'end_month': 12, 'end_day': 31,
        'intensity': 'very_high',
        'industries': ['E-commerce', 'SaaS', 'Fintech', 'EdTech', 'Healthcare'],
    },
    {
        'name': 'Summer Sales',
        'start_month': 6, 'start_day': 15,
        'end_month': 7, 'end_day': 15,
        'intensity': 'medium',
        'industries': ['E-commerce'],
    },
    {
        'name': 'Tax Season (US)',
        'start_month': 3, 'start_day': 1,
        'end_month': 4, 'end_day': 15,
        'intensity': 'high',
        'industries': ['Fintech', 'SaaS'],
    },
    {
        'name': 'Open Enrollment (Healthcare)',
        'start_month': 10, 'start_day': 15,
        'end_month': 12, 'end_day': 15,
        'intensity': 'high',
        'industries': ['Healthcare'],
    },
]

# ===========================================================================
# 주요 글로벌 이벤트 (연도별)
# ===========================================================================
MAJOR_EVENTS = {
    2022: [
        {'name': 'FIFA World Cup', 'start': '2022-11-20', 'end': '2022-12-18', 'impact': 'global'},
        {'name': 'Winter Olympics', 'start': '2022-02-04', 'end': '2022-02-20', 'impact': 'global'},
    ],
    2023: [
        {'name': 'Cricket World Cup', 'start': '2023-10-05', 'end': '2023-11-19', 'impact': 'india'},
        {'name': 'Rugby World Cup', 'start': '2023-09-08', 'end': '2023-10-28', 'impact': 'global'},
    ],
    2024: [
        {'name': 'Paris Olympics', 'start': '2024-07-26', 'end': '2024-08-11', 'impact': 'global'},
        {'name': 'Euro 2024', 'start': '2024-06-14', 'end': '2024-07-14', 'impact': 'global'},
        {'name': 'US Presidential Election', 'start': '2024-10-01', 'end': '2024-11-05', 'impact': 'usa'},
        {'name': 'Super Bowl LVIII', 'start': '2024-02-11', 'end': '2024-02-11', 'impact': 'usa'},
    ],
}

# ===========================================================================
# 플랫폼 알고리즘/정책 변화 이벤트
# ===========================================================================
PLATFORM_EVENTS = [
    {'date': '2021-04-26', 'name': 'iOS 14.5 ATT Launch', 'platform': 'Meta Ads', 'impact': -0.15},
    {'date': '2022-02-01', 'name': 'Meta Reels Ads Launch', 'platform': 'Meta Ads', 'impact': 0.05},
    {'date': '2022-07-01', 'name': 'Google Performance Max GA', 'platform': 'Google Ads', 'impact': 0.10},
    {'date': '2023-01-01', 'name': 'TikTok Shop Launch (US)', 'platform': 'TikTok Ads', 'impact': 0.10},
    {'date': '2023-07-01', 'name': 'Google GA4 Mandatory', 'platform': 'Google Ads', 'impact': -0.05},
    {'date': '2024-01-04', 'name': 'Google 3P Cookie Deprecation Start', 'platform': 'Google Ads', 'impact': -0.10},
    {'date': '2024-02-01', 'name': 'Meta Advantage+ Shopping', 'platform': 'Meta Ads', 'impact': 0.08},
    {'date': '2024-06-01', 'name': 'TikTok Creative AI Tools', 'platform': 'TikTok Ads', 'impact': 0.05},
]

# 쇼핑 시즌 강도를 숫자로
INTENSITY_MAP = {
    'low': 1,
    'medium': 2,
    'high': 3,
    'very_high': 4,
}


class HolidayCalendar:
    """국가별 공휴일/시즌/이벤트 달력을 생성하는 클래스"""

    def __init__(self, years=None):
        self.years = years or [2022, 2023, 2024]
        self.calendar_df = None
        print(f"[Phase 2-2] Holiday Calendar initialized (years: {self.years})")

    # -----------------------------------------------------------------------
    # 공휴일 데이터 생성
    # -----------------------------------------------------------------------
    def generate_holidays(self):
        """국가별 공휴일 데이터 생성"""
        print(f"\n[Phase 2-2] Generating holiday data...")

        rows = []
        for country_name, country_code in HOLIDAY_COUNTRY_MAP.items():
            if HOLIDAYS_AVAILABLE:
                try:
                    country_holidays = holidays.country_holidays(
                        country_code, years=self.years
                    )
                    for date, name in sorted(country_holidays.items()):
                        rows.append({
                            'date': pd.to_datetime(date),
                            'country': country_name,
                            'holiday_name': name,
                            'is_public_holiday': True,
                        })
                    print(f"  {country_name}: {len(country_holidays)} holidays")
                except Exception as e:
                    print(f"  {country_name}: Error - {str(e)[:60]}")
                    rows.extend(self._get_manual_holidays(country_name))
            else:
                rows.extend(self._get_manual_holidays(country_name))

        df = pd.DataFrame(rows)
        print(f"  Total holiday records: {len(df)}")
        return df

    def _get_manual_holidays(self, country):
        """holidays 패키지 미설치 시 수동 공휴일 데이터"""
        # 주요 공휴일만 수동 매핑
        common_holidays = {
            'USA': [
                (1, 1, "New Year's Day"), (1, 15, "MLK Day"),
                (2, 19, "Presidents' Day"), (5, 27, "Memorial Day"),
                (7, 4, "Independence Day"), (9, 2, "Labor Day"),
                (11, 28, "Thanksgiving"), (12, 25, "Christmas"),
            ],
            'UK': [
                (1, 1, "New Year's Day"), (3, 29, "Good Friday"),
                (5, 6, "Early May Bank Holiday"), (5, 27, "Spring Bank Holiday"),
                (8, 26, "Summer Bank Holiday"), (12, 25, "Christmas"),
                (12, 26, "Boxing Day"),
            ],
            'Germany': [
                (1, 1, "Neujahr"), (5, 1, "Tag der Arbeit"),
                (10, 3, "Tag der Deutschen Einheit"), (12, 25, "Weihnachtstag"),
                (12, 26, "Zweiter Weihnachtstag"),
            ],
            'Canada': [
                (1, 1, "New Year's Day"), (7, 1, "Canada Day"),
                (9, 2, "Labour Day"), (10, 14, "Thanksgiving"),
                (12, 25, "Christmas"),
            ],
            'India': [
                (1, 26, "Republic Day"), (8, 15, "Independence Day"),
                (10, 2, "Gandhi Jayanti"), (11, 1, "Diwali"),
                (12, 25, "Christmas"),
            ],
            'UAE': [
                (1, 1, "New Year's Day"), (12, 2, "National Day"),
                (12, 3, "National Day Holiday"),
            ],
            'Australia': [
                (1, 1, "New Year's Day"), (1, 26, "Australia Day"),
                (4, 25, "Anzac Day"), (12, 25, "Christmas"),
                (12, 26, "Boxing Day"),
            ],
        }

        rows = []
        for year in self.years:
            for month, day, name in common_holidays.get(country, []):
                try:
                    date = datetime(year, month, day)
                    rows.append({
                        'date': pd.to_datetime(date),
                        'country': country,
                        'holiday_name': name,
                        'is_public_holiday': True,
                    })
                except ValueError:
                    pass

        if not HOLIDAYS_AVAILABLE:
            print(f"  {country}: {len(rows)} holidays (manual)")
        return rows

    # -----------------------------------------------------------------------
    # 쇼핑 시즌 데이터
    # -----------------------------------------------------------------------
    def generate_shopping_seasons(self):
        """쇼핑 시즌 플래그 데이터 생성"""
        print(f"\n[Phase 2-2] Generating shopping season data...")

        rows = []
        for year in self.years:
            for season in SHOPPING_SEASONS:
                start = datetime(year, season['start_month'], season['start_day'])
                end = datetime(year, season['end_month'], season['end_day'])

                current = start
                while current <= end:
                    rows.append({
                        'date': pd.to_datetime(current),
                        'shopping_season': season['name'],
                        'season_intensity': INTENSITY_MAP.get(season['intensity'], 1),
                        'season_industries': ','.join(season['industries']),
                    })
                    current += timedelta(days=1)

        df = pd.DataFrame(rows)
        # 같은 날짜에 여러 시즌이 겹칠 수 있음 -> 가장 높은 강도 선택
        df = df.sort_values('season_intensity', ascending=False).drop_duplicates(
            subset=['date'], keep='first'
        )
        print(f"  Total season records: {len(df)}")
        return df

    # -----------------------------------------------------------------------
    # 글로벌 이벤트 데이터
    # -----------------------------------------------------------------------
    def generate_major_events(self):
        """주요 글로벌 이벤트 플래그 생성"""
        print(f"\n[Phase 2-2] Generating major event flags...")

        rows = []
        for year in self.years:
            events = MAJOR_EVENTS.get(year, [])
            for event in events:
                start = pd.to_datetime(event['start'])
                end = pd.to_datetime(event['end'])
                current = start
                while current <= end:
                    rows.append({
                        'date': current,
                        'major_event': event['name'],
                        'event_impact_scope': event['impact'],
                    })
                    current += timedelta(days=1)

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.drop_duplicates(subset=['date'], keep='first')
        print(f"  Total event records: {len(df)}")
        return df

    # -----------------------------------------------------------------------
    # 플랫폼 이벤트 데이터
    # -----------------------------------------------------------------------
    def generate_platform_events(self):
        """플랫폼 알고리즘/정책 변화 이벤트 플래그"""
        print(f"\n[Phase 2-2] Generating platform event flags...")

        rows = []
        for event in PLATFORM_EVENTS:
            event_date = pd.to_datetime(event['date'])
            # 이벤트 전후 30일간 영향
            for delta in range(-7, 61):
                d = event_date + timedelta(days=delta)
                # 영향력 감쇠 (이벤트 날짜에서 멀어질수록 감소)
                if delta < 0:
                    decay = max(0, 1.0 + delta / 7)  # 7일 전부터 서서히
                else:
                    decay = max(0, 1.0 - delta / 60)  # 60일에 걸쳐 감소

                impact = event['impact'] * decay
                if abs(impact) > 0.01:
                    rows.append({
                        'date': d,
                        'platform': event['platform'],
                        'platform_event': event['name'],
                        'platform_event_impact': round(impact, 4),
                    })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            # 같은 날짜+플랫폼에 여러 이벤트 -> 합산
            df = df.groupby(['date', 'platform']).agg({
                'platform_event': 'first',
                'platform_event_impact': 'sum',
            }).reset_index()
        print(f"  Total platform event records: {len(df)}")
        return df

    # -----------------------------------------------------------------------
    # 통합 달력 데이터 생성
    # -----------------------------------------------------------------------
    def build_calendar(self):
        """모든 달력 데이터를 통합"""
        print("\n" + "=" * 70)
        print("[Phase 2-2] Building unified calendar dataset")
        print("=" * 70)

        # 모든 날짜 x 국가 조합의 기본 프레임
        all_dates = pd.date_range(
            start=f'{min(self.years)}-01-01',
            end=f'{max(self.years)}-12-31',
            freq='D'
        )
        countries = list(HOLIDAY_COUNTRY_MAP.keys())

        base = pd.DataFrame({
            'date': np.repeat(all_dates, len(countries)),
            'country': countries * len(all_dates),
        })

        # 1. 공휴일 조인
        holidays_df = self.generate_holidays()
        base = base.merge(
            holidays_df[['date', 'country', 'is_public_holiday']],
            on=['date', 'country'], how='left'
        )
        base['is_public_holiday'] = base['is_public_holiday'].fillna(False).astype(int)

        # 2. 쇼핑 시즌 조인 (국가 무관)
        seasons_df = self.generate_shopping_seasons()
        base = base.merge(
            seasons_df[['date', 'shopping_season', 'season_intensity']],
            on='date', how='left'
        )
        base['season_intensity'] = base['season_intensity'].fillna(0).astype(int)
        base['is_shopping_season'] = (base['season_intensity'] > 0).astype(int)

        # 3. 글로벌 이벤트 조인
        events_df = self.generate_major_events()
        if len(events_df) > 0:
            base = base.merge(
                events_df[['date', 'major_event']],
                on='date', how='left'
            )
        else:
            base['major_event'] = np.nan
        base['is_major_event'] = base['major_event'].notna().astype(int)

        # 4. 다음 공휴일까지 남은 일수 계산
        base = self._compute_days_to_holiday(base)

        # 5. 시간 피처 추가
        base['day_of_week'] = pd.to_datetime(base['date']).dt.dayofweek
        base['is_weekend'] = (base['day_of_week'] >= 5).astype(int)
        base['month'] = pd.to_datetime(base['date']).dt.month
        base['quarter'] = pd.to_datetime(base['date']).dt.quarter
        base['is_month_start'] = (pd.to_datetime(base['date']).dt.day <= 3).astype(int)
        base['is_month_end'] = (pd.to_datetime(base['date']).dt.day >= 28).astype(int)

        self.calendar_df = base
        print(f"\n[Phase 2-2] Calendar dataset: {len(base)} rows, {len(base.columns)} columns")
        print(f"  Columns: {list(base.columns)}")

        return base

    def _compute_days_to_holiday(self, df):
        """각 날짜에서 다음 공휴일까지 남은 일수 계산"""
        result = df.copy()
        result['days_to_next_holiday'] = np.nan

        for country in df['country'].unique():
            mask = df['country'] == country
            country_df = df[mask].sort_values('date')

            holiday_dates = country_df[country_df['is_public_holiday'] == 1]['date'].values

            if len(holiday_dates) == 0:
                continue

            for idx, row in country_df.iterrows():
                current_date = row['date']
                future_holidays = holiday_dates[holiday_dates > current_date]
                if len(future_holidays) > 0:
                    days_to = (future_holidays[0] - current_date) / np.timedelta64(1, 'D')
                    result.loc[idx, 'days_to_next_holiday'] = int(days_to)

        result['days_to_next_holiday'] = result['days_to_next_holiday'].fillna(30)
        return result

    # -----------------------------------------------------------------------
    # 저장 및 조인
    # -----------------------------------------------------------------------
    def save(self, output_path='./data/calendar/holiday_calendar.csv'):
        """달력 데이터 저장"""
        if self.calendar_df is None:
            self.build_calendar()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.calendar_df.to_csv(output_path, index=False)
        print(f"[Phase 2-2] Saved: {output_path}")
        return output_path

    def enrich_ads_data(self, ads_df):
        """광고 데이터에 달력 피처 조인"""
        if self.calendar_df is None:
            self.build_calendar()

        ads = ads_df.copy()
        ads['date'] = pd.to_datetime(ads['date'])

        calendar_cols = [
            'date', 'country', 'is_public_holiday', 'days_to_next_holiday',
            'is_shopping_season', 'season_intensity', 'is_major_event',
            'is_weekend', 'is_month_start', 'is_month_end',
        ]
        calendar_subset = self.calendar_df[
            [c for c in calendar_cols if c in self.calendar_df.columns]
        ].copy()

        enriched = ads.merge(calendar_subset, on=['date', 'country'], how='left')

        added_cols = [c for c in enriched.columns if c not in ads_df.columns]
        print(f"[Phase 2-2] Enriched ads data with {len(added_cols)} calendar columns: {added_cols}")

        return enriched

    def get_platform_events_for_ads(self, ads_df):
        """광고 데이터에 플랫폼 이벤트 조인"""
        platform_events = self.generate_platform_events()
        if len(platform_events) == 0:
            return ads_df

        ads = ads_df.copy()
        ads['date'] = pd.to_datetime(ads['date'])

        enriched = ads.merge(
            platform_events[['date', 'platform', 'platform_event_impact']],
            on=['date', 'platform'], how='left'
        )
        enriched['platform_event_impact'] = enriched['platform_event_impact'].fillna(0)

        print(f"[Phase 2-2] Added platform event impact column")
        return enriched


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    calendar = HolidayCalendar(years=[2022, 2023, 2024])

    # 통합 달력 생성
    cal_df = calendar.build_calendar()

    # 저장
    calendar.save('./data/calendar/holiday_calendar.csv')

    # 광고 데이터 보강 예시
    if os.path.exists('global_ads_performance_dataset.csv'):
        ads_df = pd.read_csv('global_ads_performance_dataset.csv')
        enriched = calendar.enrich_ads_data(ads_df)
        enriched = calendar.get_platform_events_for_ads(enriched)
        print(f"\nEnriched dataset shape: {enriched.shape}")
        enriched.to_csv('./data/ads_with_calendar.csv', index=False)
