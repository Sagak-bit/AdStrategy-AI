# -*- coding: utf-8 -*-
"""
광고 효과 예측 시뮬레이터
- 머신러닝 모델 기반 ROAS, CPC, 전환수, 매출 예측
- 입력: 산업, 플랫폼, 국가, 예산, 캠페인 유형, 월
- 출력: 예측값 + 신뢰 구간
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

class AdsPredictor:
    def __init__(self, data_path='global_ads_performance_dataset.csv'):
        print("=" * 70)
        print("[ADS EFFECT PREDICTOR] Initializing...")
        print("=" * 70)
        
        self.df = pd.read_csv(data_path)
        self.label_encoders = {}
        self.scalers = {}
        self.models = {}
        self.feature_columns = []
        self.target_columns = ['ROAS', 'CPC', 'CPA', 'conversions', 'revenue']
        
        self._preprocess_data()
        self._train_models()
        
    def _preprocess_data(self):
        """데이터 전처리 및 피처 엔지니어링"""
        print("\n[1/3] Preprocessing data...")
        
        # 날짜 처리
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
        # 카테고리 변수 목록
        categorical_cols = ['platform', 'campaign_type', 'industry', 'country']
        
        # Label Encoding
        self.df_encoded = self.df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_encoded[col + '_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # 피처 컬럼 정의
        self.feature_columns = [
            'platform_encoded', 'campaign_type_encoded', 
            'industry_encoded', 'country_encoded',
            'ad_spend', 'month', 'quarter'
        ]
        
        # 광고비 구간 피처 추가
        self.df_encoded['ad_spend_log'] = np.log1p(self.df_encoded['ad_spend'])
        self.feature_columns.append('ad_spend_log')
        
        print(f"  - Total samples: {len(self.df_encoded)}")
        print(f"  - Features: {len(self.feature_columns)}")
        print(f"  - Targets: {self.target_columns}")
        
        # 유니크 값 저장 (나중에 입력 검증용)
        self.unique_values = {
            'platform': self.df['platform'].unique().tolist(),
            'campaign_type': self.df['campaign_type'].unique().tolist(),
            'industry': self.df['industry'].unique().tolist(),
            'country': self.df['country'].unique().tolist(),
            'month': list(range(1, 13))
        }
        
    def _train_models(self):
        """각 타겟 변수에 대해 모델 학습"""
        print("\n[2/3] Training models...")
        
        X = self.df_encoded[self.feature_columns]
        
        # 스케일러 적용
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        for target in self.target_columns:
            print(f"\n  Training model for: {target}")
            y = self.df_encoded[target]
            
            # Train/Test 분리
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # 여러 모델 비교 (n_jobs=1로 설정하여 병렬처리 비활성화)
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=1
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=100, max_depth=5, random_state=42
                ),
                'Ridge': Ridge(alpha=1.0)
            }
            
            best_model = None
            best_score = -np.inf
            best_name = ""
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
            
            # 최종 모델로 전체 데이터 학습
            best_model.fit(X_scaled, y)
            self.models[target] = {
                'model': best_model,
                'model_name': best_name,
                'r2_score': best_score
            }
            
            # 예측 오차 계산 (신뢰구간용)
            y_pred = best_model.predict(X_scaled)
            residuals = y - y_pred
            self.models[target]['std'] = residuals.std()
            self.models[target]['mae'] = mean_absolute_error(y, y_pred)
            
            print(f"    Best: {best_name} (R2: {best_score:.3f}, MAE: {self.models[target]['mae']:.2f})")
        
        print("\n[3/3] Model training completed!")
        self._print_model_summary()
        
    def _print_model_summary(self):
        """모델 성능 요약 출력"""
        print("\n" + "=" * 70)
        print("[MODEL SUMMARY]")
        print("=" * 70)
        print(f"{'Target':<15} {'Model':<20} {'R2 Score':<12} {'MAE':<10}")
        print("-" * 70)
        for target, info in self.models.items():
            print(f"{target:<15} {info['model_name']:<20} {info['r2_score']:<12.3f} {info['mae']:<10.2f}")
        print("=" * 70)
        
    def predict(self, platform, industry, country, ad_spend, campaign_type, month):
        """
        광고 효과 예측
        
        Parameters:
        -----------
        platform : str - 'Google Ads', 'Meta Ads', 'TikTok Ads'
        industry : str - 'Fintech', 'EdTech', 'Healthcare', 'SaaS', 'E-commerce'
        country : str - 'UAE', 'UK', 'USA', 'Germany', 'Canada', 'India', 'Australia'
        ad_spend : float - 광고 예산 (USD)
        campaign_type : str - 'Search', 'Video', 'Shopping', 'Display'
        month : int - 월 (1-12)
        
        Returns:
        --------
        dict : 예측 결과
        """
        # 입력 검증
        self._validate_input(platform, industry, country, campaign_type, month)
        
        # 피처 인코딩
        features = {
            'platform_encoded': self.label_encoders['platform'].transform([platform])[0],
            'campaign_type_encoded': self.label_encoders['campaign_type'].transform([campaign_type])[0],
            'industry_encoded': self.label_encoders['industry'].transform([industry])[0],
            'country_encoded': self.label_encoders['country'].transform([country])[0],
            'ad_spend': ad_spend,
            'month': month,
            'quarter': (month - 1) // 3 + 1,
            'ad_spend_log': np.log1p(ad_spend)
        }
        
        X = pd.DataFrame([features])[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # 예측
        predictions = {}
        for target in self.target_columns:
            model_info = self.models[target]
            pred = model_info['model'].predict(X_scaled)[0]
            std = model_info['std']
            
            # 신뢰 구간 (약 68% 신뢰구간)
            lower = max(0, pred - std)
            upper = pred + std
            
            predictions[target] = {
                'predicted': round(pred, 2),
                'lower': round(lower, 2),
                'upper': round(upper, 2),
                'confidence': 'Medium' if model_info['r2_score'] > 0.3 else 'Low'
            }
        
        # 추가 계산
        predictions['estimated_clicks'] = {
            'predicted': round(ad_spend / max(predictions['CPC']['predicted'], 0.1), 0),
            'lower': round(ad_spend / max(predictions['CPC']['upper'], 0.1), 0),
            'upper': round(ad_spend / max(predictions['CPC']['lower'], 0.1), 0),
            'confidence': predictions['CPC']['confidence']
        }
        
        predictions['estimated_revenue'] = {
            'predicted': round(ad_spend * predictions['ROAS']['predicted'], 2),
            'lower': round(ad_spend * predictions['ROAS']['lower'], 2),
            'upper': round(ad_spend * predictions['ROAS']['upper'], 2),
            'confidence': predictions['ROAS']['confidence']
        }
        
        # 히스토리컬 데이터 참조
        historical = self._get_historical_stats(platform, industry, country, campaign_type, month)
        
        return {
            'input': {
                'platform': platform,
                'industry': industry,
                'country': country,
                'ad_spend': ad_spend,
                'campaign_type': campaign_type,
                'month': month
            },
            'predictions': predictions,
            'historical_reference': historical
        }
    
    def _validate_input(self, platform, industry, country, campaign_type, month):
        """입력값 검증"""
        if platform not in self.unique_values['platform']:
            raise ValueError(f"Invalid platform. Choose from: {self.unique_values['platform']}")
        if industry not in self.unique_values['industry']:
            raise ValueError(f"Invalid industry. Choose from: {self.unique_values['industry']}")
        if country not in self.unique_values['country']:
            raise ValueError(f"Invalid country. Choose from: {self.unique_values['country']}")
        if campaign_type not in self.unique_values['campaign_type']:
            raise ValueError(f"Invalid campaign_type. Choose from: {self.unique_values['campaign_type']}")
        if month not in self.unique_values['month']:
            raise ValueError(f"Invalid month. Choose from: 1-12")
            
    def _get_historical_stats(self, platform, industry, country, campaign_type, month):
        """유사 조건의 히스토리컬 데이터 통계"""
        # 정확히 일치하는 데이터
        mask = (
            (self.df['platform'] == platform) &
            (self.df['industry'] == industry) &
            (self.df['country'] == country) &
            (self.df['campaign_type'] == campaign_type)
        )
        exact_match = self.df[mask]
        
        # 플랫폼+산업만 일치
        mask_partial = (
            (self.df['platform'] == platform) &
            (self.df['industry'] == industry)
        )
        partial_match = self.df[mask_partial]
        
        stats = {
            'exact_match_count': len(exact_match),
            'partial_match_count': len(partial_match)
        }
        
        if len(exact_match) > 0:
            stats['exact_match_stats'] = {
                'avg_ROAS': round(exact_match['ROAS'].mean(), 2),
                'median_ROAS': round(exact_match['ROAS'].median(), 2),
                'min_ROAS': round(exact_match['ROAS'].min(), 2),
                'max_ROAS': round(exact_match['ROAS'].max(), 2)
            }
        
        if len(partial_match) > 0:
            stats['partial_match_stats'] = {
                'avg_ROAS': round(partial_match['ROAS'].mean(), 2),
                'median_ROAS': round(partial_match['ROAS'].median(), 2)
            }
            
        return stats
    
    def print_prediction(self, result):
        """예측 결과 출력"""
        inp = result['input']
        pred = result['predictions']
        hist = result['historical_reference']
        
        print("\n" + "=" * 70)
        print("[PREDICTION RESULT]")
        print("=" * 70)
        
        print("\n[INPUT PARAMETERS]")
        print(f"  Platform:      {inp['platform']}")
        print(f"  Industry:      {inp['industry']}")
        print(f"  Country:       {inp['country']}")
        print(f"  Campaign Type: {inp['campaign_type']}")
        print(f"  Ad Spend:      ${inp['ad_spend']:,.2f}")
        print(f"  Month:         {inp['month']}")
        
        print("\n[PREDICTED METRICS]")
        print("-" * 70)
        print(f"{'Metric':<20} {'Predicted':<15} {'Range (68% CI)':<25} {'Confidence':<10}")
        print("-" * 70)
        
        metrics_to_show = ['ROAS', 'CPC', 'CPA', 'conversions', 'estimated_clicks', 'estimated_revenue']
        labels = {
            'ROAS': 'ROAS',
            'CPC': 'CPC ($)',
            'CPA': 'CPA ($)',
            'conversions': 'Conversions',
            'estimated_clicks': 'Est. Clicks',
            'estimated_revenue': 'Est. Revenue ($)'
        }
        
        for metric in metrics_to_show:
            if metric in pred:
                p = pred[metric]
                predicted = f"{p['predicted']:,.2f}" if isinstance(p['predicted'], float) else f"{p['predicted']:,.0f}"
                range_str = f"{p['lower']:,.2f} ~ {p['upper']:,.2f}" if isinstance(p['lower'], float) else f"{p['lower']:,.0f} ~ {p['upper']:,.0f}"
                print(f"{labels.get(metric, metric):<20} {predicted:<15} {range_str:<25} {p['confidence']:<10}")
        
        print("-" * 70)
        
        # ROI 계산
        roi = ((pred['estimated_revenue']['predicted'] - inp['ad_spend']) / inp['ad_spend']) * 100
        print(f"\n[ROI ESTIMATE]")
        print(f"  Expected ROI: {roi:,.1f}%")
        print(f"  Break-even:   {'Yes' if roi > 0 else 'No'} (ROAS > 1.0)")
        
        # 히스토리컬 참조
        print(f"\n[HISTORICAL REFERENCE]")
        print(f"  Exact match samples:   {hist['exact_match_count']}")
        print(f"  Partial match samples: {hist['partial_match_count']}")
        
        if 'exact_match_stats' in hist:
            stats = hist['exact_match_stats']
            print(f"  Historical ROAS:       {stats['avg_ROAS']:.2f} (avg), {stats['median_ROAS']:.2f} (median)")
            print(f"  Historical Range:      {stats['min_ROAS']:.2f} ~ {stats['max_ROAS']:.2f}")
        
        # 신뢰도 경고
        if hist['exact_match_count'] < 5:
            print(f"\n[WARNING] Low sample count ({hist['exact_match_count']}). Predictions may be less reliable.")
        
        print("\n" + "=" * 70)
        
    def interactive_mode(self):
        """인터랙티브 모드"""
        print("\n" + "=" * 70)
        print("[INTERACTIVE MODE]")
        print("=" * 70)
        print("\nEnter 'q' to quit.\n")
        
        while True:
            try:
                print("\n--- New Prediction ---")
                
                # 플랫폼 선택
                print(f"\nAvailable Platforms: {self.unique_values['platform']}")
                platform = input("Platform: ").strip()
                if platform.lower() == 'q':
                    break
                    
                # 산업 선택
                print(f"Available Industries: {self.unique_values['industry']}")
                industry = input("Industry: ").strip()
                if industry.lower() == 'q':
                    break
                    
                # 국가 선택
                print(f"Available Countries: {self.unique_values['country']}")
                country = input("Country: ").strip()
                if country.lower() == 'q':
                    break
                    
                # 캠페인 유형
                print(f"Available Campaign Types: {self.unique_values['campaign_type']}")
                campaign_type = input("Campaign Type: ").strip()
                if campaign_type.lower() == 'q':
                    break
                    
                # 예산
                ad_spend = float(input("Ad Spend ($): ").strip())
                
                # 월
                month = int(input("Month (1-12): ").strip())
                
                # 예측 실행
                result = self.predict(platform, industry, country, ad_spend, campaign_type, month)
                self.print_prediction(result)
                
            except ValueError as e:
                print(f"\n[ERROR] {e}")
            except Exception as e:
                print(f"\n[ERROR] {e}")
                
        print("\nGoodbye!")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # 예측기 초기화 (모델 학습)
    predictor = AdsPredictor('global_ads_performance_dataset.csv')
    
    # 예시 예측
    print("\n" + "=" * 70)
    print("[EXAMPLE PREDICTIONS]")
    print("=" * 70)
    
    # 예시 1: TikTok + EdTech + India
    result1 = predictor.predict(
        platform='TikTok Ads',
        industry='EdTech',
        country='India',
        ad_spend=2000,
        campaign_type='Search',
        month=4
    )
    predictor.print_prediction(result1)
    
    # 예시 2: Google Ads + Healthcare + USA
    result2 = predictor.predict(
        platform='Google Ads',
        industry='Healthcare',
        country='USA',
        ad_spend=5000,
        campaign_type='Search',
        month=10
    )
    predictor.print_prediction(result2)
    
    # 예시 3: Meta Ads + E-commerce + Germany
    result3 = predictor.predict(
        platform='Meta Ads',
        industry='E-commerce',
        country='Germany',
        ad_spend=3000,
        campaign_type='Shopping',
        month=12
    )
    predictor.print_prediction(result3)
    
    # 인터랙티브 모드 실행 여부
    print("\n" + "=" * 70)
    run_interactive = input("\nRun interactive mode? (y/n): ").strip().lower()
    if run_interactive == 'y':
        predictor.interactive_mode()
