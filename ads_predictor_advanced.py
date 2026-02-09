# -*- coding: utf-8 -*-
"""
광고 효과 예측 시뮬레이터 (Advanced Version)
- 머신러닝 + 딥러닝 앙상블 모델
- XGBoost, Neural Network, RandomForest, GradientBoosting 비교
- 앙상블 예측으로 정확도 향상
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import sys
import io

# Windows 콘솔 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# XGBoost 설치 여부 확인
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not installed. Using alternative models.")

# LightGBM 설치 여부 확인
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class AdvancedAdsPredictor:
    def __init__(self, data_path='global_ads_performance_dataset.csv'):
        print("=" * 70)
        print("[ADVANCED ADS PREDICTOR] Initializing...")
        print("=" * 70)
        
        self.df = pd.read_csv(data_path)
        self.label_encoders = {}
        self.scalers = {}
        self.models = {}
        self.ensemble_models = {}
        self.feature_columns = []
        self.target_columns = ['ROAS', 'CPC', 'CPA', 'conversions', 'revenue']
        
        print(f"\n[INFO] XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
        print(f"[INFO] LightGBM: {'Available' if LIGHTGBM_AVAILABLE else 'Not Available'}")
        
        self._preprocess_data()
        self._engineer_features()
        self._train_models()
        
    def _preprocess_data(self):
        """데이터 전처리"""
        print("\n[1/4] Preprocessing data...")
        
        # 날짜 처리
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week.astype(int)
        
        # 카테고리 변수 목록
        self.categorical_cols = ['platform', 'campaign_type', 'industry', 'country']
        
        # Label Encoding
        self.df_encoded = self.df.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.df_encoded[col + '_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # 유니크 값 저장
        self.unique_values = {
            'platform': self.df['platform'].unique().tolist(),
            'campaign_type': self.df['campaign_type'].unique().tolist(),
            'industry': self.df['industry'].unique().tolist(),
            'country': self.df['country'].unique().tolist(),
            'month': list(range(1, 13))
        }
        
        print(f"  - Total samples: {len(self.df_encoded)}")
        
    def _engineer_features(self):
        """피처 엔지니어링"""
        print("\n[2/4] Engineering features...")
        
        # 기본 피처
        self.feature_columns = [
            'platform_encoded', 'campaign_type_encoded', 
            'industry_encoded', 'country_encoded',
            'ad_spend', 'month', 'quarter'
        ]
        
        # 로그 변환 피처
        self.df_encoded['ad_spend_log'] = np.log1p(self.df_encoded['ad_spend'])
        self.feature_columns.append('ad_spend_log')
        
        # 광고비 구간 피처
        self.df_encoded['ad_spend_bin'] = pd.cut(
            self.df_encoded['ad_spend'], 
            bins=[0, 1000, 3000, 5000, 10000, 20000, float('inf')],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
        self.feature_columns.append('ad_spend_bin')
        
        # 시즌 피처 (Q4 = 1, else = 0)
        self.df_encoded['is_q4'] = (self.df_encoded['quarter'] == 4).astype(int)
        self.feature_columns.append('is_q4')
        
        # 주말 여부
        self.df_encoded['is_weekend'] = (self.df_encoded['day_of_week'] >= 5).astype(int)
        self.feature_columns.append('is_weekend')
        
        # 플랫폼-산업 인터랙션 피처
        self.df_encoded['platform_industry'] = (
            self.df_encoded['platform_encoded'] * 10 + 
            self.df_encoded['industry_encoded']
        )
        self.feature_columns.append('platform_industry')
        
        # 플랫폼-국가 인터랙션 피처
        self.df_encoded['platform_country'] = (
            self.df_encoded['platform_encoded'] * 10 + 
            self.df_encoded['country_encoded']
        )
        self.feature_columns.append('platform_country')
        
        print(f"  - Total features: {len(self.feature_columns)}")
        print(f"  - Features: {self.feature_columns}")
        
    def _train_models(self):
        """각 타겟에 대해 여러 모델 학습 및 앙상블"""
        print("\n[3/4] Training models...")
        
        X = self.df_encoded[self.feature_columns].values
        
        # 스케일러 적용
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        for target in self.target_columns:
            print(f"\n  === Training for: {target} ===")
            y = self.df_encoded[target].values
            
            # Train/Test 분리
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # 모델 정의
            models = self._get_models()
            
            results = {}
            trained_models = {}
            
            for name, model in models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    
                    # 학습
                    model.fit(X_train, y_train)
                    
                    # 평가
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    results[name] = {
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    trained_models[name] = model
                    
                    print(f"    {name:<25} R2: {r2:.3f}, MAE: {mae:.2f}, CV: {cv_scores.mean():.3f}(+/-{cv_scores.std():.3f})")
                    
                except Exception as e:
                    print(f"    {name:<25} Failed: {str(e)[:50]}")
            
            # 최적 모델 선택
            best_name = max(results, key=lambda x: results[x]['r2'])
            best_model = trained_models[best_name]
            
            # 앙상블 모델 생성 (상위 3개 모델)
            top_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
            ensemble_estimators = [(name, trained_models[name]) for name, _ in top_models if name in trained_models]
            
            if len(ensemble_estimators) >= 2:
                ensemble = VotingRegressor(estimators=ensemble_estimators)
                ensemble.fit(X_train, y_train)
                y_pred_ensemble = ensemble.predict(X_test)
                ensemble_r2 = r2_score(y_test, y_pred_ensemble)
                ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
                
                print(f"    {'[ENSEMBLE]':<25} R2: {ensemble_r2:.3f}, MAE: {ensemble_mae:.2f}")
                
                # 앙상블이 더 좋으면 앙상블 사용
                if ensemble_r2 > results[best_name]['r2']:
                    best_model = ensemble
                    best_name = 'Ensemble'
                    results['Ensemble'] = {'r2': ensemble_r2, 'mae': ensemble_mae}
            
            # 최종 모델 전체 데이터로 재학습
            if best_name != 'Ensemble':
                best_model = trained_models[best_name]
            best_model.fit(X_scaled, y)
            
            # 잔차 계산
            y_pred_all = best_model.predict(X_scaled)
            residuals = y - y_pred_all
            
            self.models[target] = {
                'model': best_model,
                'model_name': best_name,
                'r2_score': results[best_name]['r2'],
                'mae': results[best_name]['mae'],
                'std': residuals.std(),
                'all_results': results
            }
            
            print(f"    >>> Best: {best_name} (R2: {results[best_name]['r2']:.3f})")
        
        print("\n[4/4] Model training completed!")
        self._print_model_summary()
        
    def _get_models(self):
        """사용 가능한 모델 목록 반환"""
        models = {
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_split=5,
                random_state=42, n_jobs=1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
            'NeuralNetwork_Small': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                random_state=42
            ),
            'NeuralNetwork_Large': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                random_state=42
            ),
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
            
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                verbosity=-1
            )
            
        return models
        
    def _print_model_summary(self):
        """모델 성능 요약"""
        print("\n" + "=" * 70)
        print("[MODEL SUMMARY]")
        print("=" * 70)
        print(f"{'Target':<15} {'Best Model':<25} {'R2 Score':<12} {'MAE':<12}")
        print("-" * 70)
        for target, info in self.models.items():
            print(f"{target:<15} {info['model_name']:<25} {info['r2_score']:<12.3f} {info['mae']:<12.2f}")
        print("=" * 70)
        
        # 전체 모델 비교 (ROAS 기준)
        print("\n[DETAILED COMPARISON - ROAS]")
        print("-" * 70)
        roas_results = self.models['ROAS']['all_results']
        sorted_results = sorted(roas_results.items(), key=lambda x: x[1]['r2'], reverse=True)
        for name, metrics in sorted_results:
            print(f"  {name:<25} R2: {metrics['r2']:.3f}, MAE: {metrics['mae']:.2f}")
        
    def predict(self, platform, industry, country, ad_spend, campaign_type, month):
        """광고 효과 예측"""
        # 입력 검증
        self._validate_input(platform, industry, country, campaign_type, month)
        
        # 피처 생성
        quarter = (month - 1) // 3 + 1
        features = {
            'platform_encoded': self.label_encoders['platform'].transform([platform])[0],
            'campaign_type_encoded': self.label_encoders['campaign_type'].transform([campaign_type])[0],
            'industry_encoded': self.label_encoders['industry'].transform([industry])[0],
            'country_encoded': self.label_encoders['country'].transform([country])[0],
            'ad_spend': ad_spend,
            'month': month,
            'quarter': quarter,
            'ad_spend_log': np.log1p(ad_spend),
            'ad_spend_bin': self._get_ad_spend_bin(ad_spend),
            'is_q4': 1 if quarter == 4 else 0,
            'is_weekend': 0,  # 기본값
            'platform_industry': self.label_encoders['platform'].transform([platform])[0] * 10 + 
                                self.label_encoders['industry'].transform([industry])[0],
            'platform_country': self.label_encoders['platform'].transform([platform])[0] * 10 + 
                               self.label_encoders['country'].transform([country])[0]
        }
        
        X = pd.DataFrame([features])[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # 예측
        predictions = {}
        for target in self.target_columns:
            model_info = self.models[target]
            pred = model_info['model'].predict(X_scaled)[0]
            std = model_info['std']
            
            # 95% 신뢰구간 (1.96 * std)
            ci_95_lower = max(0, pred - 1.96 * std)
            ci_95_upper = pred + 1.96 * std
            
            # 68% 신뢰구간 (1 * std)
            ci_68_lower = max(0, pred - std)
            ci_68_upper = pred + std
            
            confidence = 'High' if model_info['r2_score'] > 0.4 else ('Medium' if model_info['r2_score'] > 0.2 else 'Low')
            
            predictions[target] = {
                'predicted': round(pred, 2),
                'ci_68': (round(ci_68_lower, 2), round(ci_68_upper, 2)),
                'ci_95': (round(ci_95_lower, 2), round(ci_95_upper, 2)),
                'confidence': confidence,
                'model_used': model_info['model_name']
            }
        
        # 파생 지표 계산
        predictions['estimated_clicks'] = {
            'predicted': round(ad_spend / max(predictions['CPC']['predicted'], 0.1), 0),
            'ci_68': (
                round(ad_spend / max(predictions['CPC']['ci_68'][1], 0.1), 0),
                round(ad_spend / max(predictions['CPC']['ci_68'][0], 0.1), 0)
            ),
            'confidence': predictions['CPC']['confidence']
        }
        
        predictions['estimated_revenue'] = {
            'predicted': round(ad_spend * predictions['ROAS']['predicted'], 2),
            'ci_68': (
                round(ad_spend * predictions['ROAS']['ci_68'][0], 2),
                round(ad_spend * predictions['ROAS']['ci_68'][1], 2)
            ),
            'ci_95': (
                round(ad_spend * predictions['ROAS']['ci_95'][0], 2),
                round(ad_spend * predictions['ROAS']['ci_95'][1], 2)
            ),
            'confidence': predictions['ROAS']['confidence']
        }
        
        # 히스토리컬 참조
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
    
    def _get_ad_spend_bin(self, ad_spend):
        """광고비 구간 반환"""
        if ad_spend <= 1000:
            return 0
        elif ad_spend <= 3000:
            return 1
        elif ad_spend <= 5000:
            return 2
        elif ad_spend <= 10000:
            return 3
        elif ad_spend <= 20000:
            return 4
        else:
            return 5
            
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
        """히스토리컬 데이터 통계"""
        # 정확히 일치
        mask = (
            (self.df['platform'] == platform) &
            (self.df['industry'] == industry) &
            (self.df['country'] == country) &
            (self.df['campaign_type'] == campaign_type)
        )
        exact_match = self.df[mask]
        
        # 플랫폼+산업만
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
                'std_ROAS': round(exact_match['ROAS'].std(), 2),
                'min_ROAS': round(exact_match['ROAS'].min(), 2),
                'max_ROAS': round(exact_match['ROAS'].max(), 2),
                'avg_CPC': round(exact_match['CPC'].mean(), 2)
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
        print(f"{'Metric':<18} {'Predicted':<12} {'68% CI':<22} {'95% CI':<22} {'Model':<15}")
        print("-" * 70)
        
        metrics_to_show = [
            ('ROAS', 'ROAS'),
            ('CPC', 'CPC ($)'),
            ('CPA', 'CPA ($)'),
            ('conversions', 'Conversions'),
        ]
        
        for key, label in metrics_to_show:
            p = pred[key]
            predicted = f"{p['predicted']:,.2f}"
            ci_68 = f"{p['ci_68'][0]:,.2f} ~ {p['ci_68'][1]:,.2f}"
            ci_95 = f"{p['ci_95'][0]:,.2f} ~ {p['ci_95'][1]:,.2f}"
            model = p.get('model_used', '-')[:14]
            print(f"{label:<18} {predicted:<12} {ci_68:<22} {ci_95:<22} {model:<15}")
        
        print("-" * 70)
        
        # 예상 클릭/매출
        print(f"\n[DERIVED METRICS]")
        clicks = pred['estimated_clicks']
        revenue = pred['estimated_revenue']
        print(f"  Est. Clicks:  {clicks['predicted']:,.0f} ({clicks['ci_68'][0]:,.0f} ~ {clicks['ci_68'][1]:,.0f})")
        print(f"  Est. Revenue: ${revenue['predicted']:,.2f} (${revenue['ci_68'][0]:,.2f} ~ ${revenue['ci_68'][1]:,.2f})")
        
        # ROI
        roi = ((revenue['predicted'] - inp['ad_spend']) / inp['ad_spend']) * 100
        roi_low = ((revenue['ci_68'][0] - inp['ad_spend']) / inp['ad_spend']) * 100
        roi_high = ((revenue['ci_68'][1] - inp['ad_spend']) / inp['ad_spend']) * 100
        
        print(f"\n[ROI ESTIMATE]")
        print(f"  Expected ROI: {roi:,.1f}% ({roi_low:,.1f}% ~ {roi_high:,.1f}%)")
        print(f"  Break-even:   {'Yes' if roi_low > 0 else 'Uncertain' if roi > 0 else 'No'}")
        
        # 히스토리컬
        print(f"\n[HISTORICAL REFERENCE]")
        print(f"  Exact match samples:   {hist['exact_match_count']}")
        
        if 'exact_match_stats' in hist:
            stats = hist['exact_match_stats']
            print(f"  Historical ROAS:       {stats['avg_ROAS']:.2f} avg, {stats['median_ROAS']:.2f} median")
            print(f"  Historical Range:      {stats['min_ROAS']:.2f} ~ {stats['max_ROAS']:.2f}")
            print(f"  Historical CPC:        ${stats['avg_CPC']:.2f}")
            
            # 모델 vs 히스토리컬 비교
            diff = pred['ROAS']['predicted'] - stats['avg_ROAS']
            diff_pct = (diff / stats['avg_ROAS']) * 100
            print(f"\n  Model vs Historical:   {'+' if diff > 0 else ''}{diff:.2f} ({'+' if diff_pct > 0 else ''}{diff_pct:.1f}%)")
        
        # 경고
        if hist['exact_match_count'] < 5:
            print(f"\n[!] WARNING: Low sample count ({hist['exact_match_count']}). Use historical median as reference.")
            
        print("\n" + "=" * 70)
        
    def compare_scenarios(self, scenarios):
        """여러 시나리오 비교"""
        print("\n" + "=" * 70)
        print("[SCENARIO COMPARISON]")
        print("=" * 70)
        
        results = []
        for i, s in enumerate(scenarios):
            result = self.predict(**s)
            results.append(result)
            
        print(f"\n{'#':<3} {'Platform':<12} {'Industry':<12} {'Country':<10} {'Budget':<10} {'ROAS':<8} {'Revenue':<12} {'ROI':<10}")
        print("-" * 85)
        
        for i, (s, r) in enumerate(zip(scenarios, results)):
            roas = r['predictions']['ROAS']['predicted']
            revenue = r['predictions']['estimated_revenue']['predicted']
            roi = ((revenue - s['ad_spend']) / s['ad_spend']) * 100
            
            print(f"{i+1:<3} {s['platform']:<12} {s['industry']:<12} {s['country']:<10} ${s['ad_spend']:<9,.0f} {roas:<8.2f} ${revenue:<11,.0f} {roi:<9.1f}%")
        
        # 최적 시나리오
        best_idx = max(range(len(results)), key=lambda i: results[i]['predictions']['ROAS']['predicted'])
        print(f"\n>>> Best ROAS: Scenario #{best_idx + 1}")
        
        best_roi_idx = max(range(len(results)), key=lambda i: 
            (results[i]['predictions']['estimated_revenue']['predicted'] - scenarios[i]['ad_spend']) / scenarios[i]['ad_spend'])
        print(f">>> Best ROI:  Scenario #{best_roi_idx + 1}")
        
        return results
        
    def interactive_mode(self):
        """인터랙티브 모드"""
        print("\n" + "=" * 70)
        print("[INTERACTIVE MODE]")
        print("=" * 70)
        print("\nEnter 'q' to quit, 'c' to compare scenarios.\n")
        
        while True:
            try:
                print("\n--- New Prediction ---")
                
                # 플랫폼
                print(f"\nPlatforms: {self.unique_values['platform']}")
                platform = input("Platform: ").strip()
                if platform.lower() == 'q':
                    break
                if platform.lower() == 'c':
                    self._interactive_compare()
                    continue
                    
                # 산업
                print(f"Industries: {self.unique_values['industry']}")
                industry = input("Industry: ").strip()
                if industry.lower() == 'q':
                    break
                    
                # 국가
                print(f"Countries: {self.unique_values['country']}")
                country = input("Country: ").strip()
                if country.lower() == 'q':
                    break
                    
                # 캠페인 유형
                print(f"Campaign Types: {self.unique_values['campaign_type']}")
                campaign_type = input("Campaign Type: ").strip()
                if campaign_type.lower() == 'q':
                    break
                    
                # 예산
                ad_spend = float(input("Ad Spend ($): ").strip())
                
                # 월
                month = int(input("Month (1-12): ").strip())
                
                # 예측
                result = self.predict(platform, industry, country, ad_spend, campaign_type, month)
                self.print_prediction(result)
                
            except ValueError as e:
                print(f"\n[ERROR] {e}")
            except Exception as e:
                print(f"\n[ERROR] {e}")
                
        print("\nGoodbye!")
        
    def _interactive_compare(self):
        """인터랙티브 시나리오 비교"""
        print("\n--- Scenario Comparison Mode ---")
        print("Enter scenarios (empty line to finish):\n")
        
        scenarios = []
        while True:
            try:
                print(f"\nScenario #{len(scenarios) + 1} (or press Enter to compare)")
                platform = input("Platform: ").strip()
                if not platform:
                    break
                    
                industry = input("Industry: ").strip()
                country = input("Country: ").strip()
                campaign_type = input("Campaign Type: ").strip()
                ad_spend = float(input("Ad Spend ($): ").strip())
                month = int(input("Month (1-12): ").strip())
                
                scenarios.append({
                    'platform': platform,
                    'industry': industry,
                    'country': country,
                    'campaign_type': campaign_type,
                    'ad_spend': ad_spend,
                    'month': month
                })
                
            except Exception as e:
                print(f"[ERROR] {e}")
                
        if len(scenarios) >= 2:
            self.compare_scenarios(scenarios)
        else:
            print("Need at least 2 scenarios to compare.")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # 예측기 초기화
    predictor = AdvancedAdsPredictor('global_ads_performance_dataset.csv')
    
    # 예시 예측
    print("\n" + "=" * 70)
    print("[EXAMPLE PREDICTIONS]")
    print("=" * 70)
    
    # 예시 1
    result1 = predictor.predict(
        platform='TikTok Ads',
        industry='EdTech',
        country='India',
        ad_spend=2000,
        campaign_type='Search',
        month=4
    )
    predictor.print_prediction(result1)
    
    # 예시 2
    result2 = predictor.predict(
        platform='Google Ads',
        industry='Healthcare',
        country='USA',
        ad_spend=5000,
        campaign_type='Search',
        month=10
    )
    predictor.print_prediction(result2)
    
    # 시나리오 비교
    print("\n" + "=" * 70)
    print("[SCENARIO COMPARISON EXAMPLE]")
    print("=" * 70)
    
    scenarios = [
        {'platform': 'TikTok Ads', 'industry': 'E-commerce', 'country': 'Germany', 'ad_spend': 3000, 'campaign_type': 'Shopping', 'month': 12},
        {'platform': 'Meta Ads', 'industry': 'E-commerce', 'country': 'Germany', 'ad_spend': 3000, 'campaign_type': 'Shopping', 'month': 12},
        {'platform': 'Google Ads', 'industry': 'E-commerce', 'country': 'Germany', 'ad_spend': 3000, 'campaign_type': 'Shopping', 'month': 12},
        {'platform': 'TikTok Ads', 'industry': 'E-commerce', 'country': 'India', 'ad_spend': 3000, 'campaign_type': 'Shopping', 'month': 12},
    ]
    predictor.compare_scenarios(scenarios)
    
    # 인터랙티브 모드
    print("\n" + "=" * 70)
    run_interactive = input("\nRun interactive mode? (y/n): ").strip().lower()
    if run_interactive == 'y':
        predictor.interactive_mode()
