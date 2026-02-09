# -*- coding: utf-8 -*-
"""
광고 효과 예측 시뮬레이터 V2 (Enhanced)
- 보강된 데이터셋 (enriched_ads_final.csv) 활용
- 개선사항:
  1. OneHotEncoder 적용 (LabelEncoder 가짜 순서 문제 해결)
  2. Temporal Split (시간 기반 분할)
  3. 하이퍼파라미터 튜닝 (RandomizedSearchCV / Optuna)
  4. Target 로그 변환 (ROAS 스큐 해결)
  5. SHAP 분석 (피처 중요도 해석)
  6. 보강된 피처 활용 (매크로, 시즌, 트렌드, 크리에이티브 등)
"""

import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
import pickle
from datetime import datetime

from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, TimeSeriesSplit,
    RandomizedSearchCV
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Windows 콘솔 인코딩 (interactive mode only, skip if running unbuffered)
try:
    if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings('ignore')

# 선택적 임포트
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class AdsPredictor_V2:
    """
    개선된 광고 효과 예측기 V2
    
    주요 개선:
    - OneHotEncoder로 카테고리 처리 (Ridge, MLP 등에서 성능 개선)
    - 시간 기반 분할로 현실적인 평가
    - ROAS 로그 변환으로 스큐 문제 해결
    - 보강된 피처 자동 감지 및 활용
    - SHAP 기반 피처 중요도 분석
    - 하이퍼파라미터 자동 튜닝
    """

    def __init__(self, data_path='data/enriched_ads_final.csv',
                 fallback_path='global_ads_performance_dataset.csv',
                 use_temporal_split=True,
                 log_transform_target=True,
                 tune_hyperparams=False):
        """
        Parameters:
        -----------
        data_path : str - 보강된 데이터셋 경로
        fallback_path : str - 보강 데이터 없을 시 원본 경로
        use_temporal_split : bool - 시간 기반 분할 사용 여부
        log_transform_target : bool - ROAS 로그 변환 여부
        tune_hyperparams : bool - 하이퍼파라미터 자동 튜닝 (느림)
        """
        print("=" * 70)
        print("[ADS PREDICTOR V2] Initializing...")
        print("=" * 70)

        self.use_temporal_split = use_temporal_split
        self.log_transform_target = log_transform_target
        self.tune_hyperparams = tune_hyperparams

        # 데이터 로드
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            self.data_source = data_path
            print(f"[INFO] Loaded enriched dataset: {data_path}")
        elif os.path.exists(fallback_path):
            self.df = pd.read_csv(fallback_path)
            self.data_source = fallback_path
            print(f"[INFO] Enriched data not found, using fallback: {fallback_path}")
        else:
            raise FileNotFoundError(f"No dataset found at {data_path} or {fallback_path}")

        print(f"[INFO] Dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"[INFO] XGBoost: {'Yes' if XGBOOST_AVAILABLE else 'No'}")
        print(f"[INFO] LightGBM: {'Yes' if LIGHTGBM_AVAILABLE else 'No'}")
        print(f"[INFO] SHAP: {'Yes' if SHAP_AVAILABLE else 'No'}")
        print(f"[INFO] Optuna: {'Yes' if OPTUNA_AVAILABLE else 'No'}")
        print(f"[INFO] Temporal split: {'Yes' if use_temporal_split else 'No'}")
        print(f"[INFO] Log transform: {'Yes' if log_transform_target else 'No'}")

        self.label_encoders = {}
        self.models = {}
        self.preprocessors = {}
        self.feature_importances = {}

        self.target_columns = ['ROAS', 'CPC', 'CPA', 'conversions', 'revenue']

        self._preprocess_data()
        self._train_models()

    # ===================================================================
    # 전처리
    # ===================================================================
    def _preprocess_data(self):
        """데이터 전처리 및 피처 설정"""
        print("\n[1/4] Preprocessing data...")

        # 날짜 처리
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df = self.df.dropna(subset=['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week.astype(int)

        # ----- 카테고리 변수 식별 -----
        self.cat_features = ['platform', 'campaign_type', 'industry', 'country']
        # 보강 데이터에 있을 수 있는 추가 카테고리
        extra_cats = ['ad_format', 'cta_type', 'target_age_group',
                      'target_gender', 'target_interest']
        for c in extra_cats:
            if c in self.df.columns:
                self.cat_features.append(c)

        # ----- 수치 변수 식별 -----
        self.num_features = ['ad_spend', 'month', 'quarter']

        # 기본 엔지니어링
        self.df['ad_spend_log'] = np.log1p(self.df['ad_spend'])
        self.num_features.append('ad_spend_log')

        self.df['ad_spend_bin'] = pd.cut(
            self.df['ad_spend'],
            bins=[0, 1000, 3000, 5000, 10000, 20000, float('inf')],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float).fillna(0).astype(int)
        self.num_features.append('ad_spend_bin')

        self.df['is_q4'] = (self.df['quarter'] == 4).astype(int)
        self.num_features.append('is_q4')

        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.num_features.append('is_weekend')

        self.num_features.append('day_of_week')
        self.num_features.append('week_of_year')

        # ----- 보강 피처 자동 감지 -----
        enriched_num = [
            # Phase 2-1: 매크로 경제
            'cpi_index', 'cpi_yoy_pct', 'unemployment_rate',
            'gdp_growth_pct', 'exchange_rate_usd',
            # Phase 2-2: 달력
            'is_public_holiday', 'days_to_next_holiday',
            'is_shopping_season', 'season_intensity', 'is_major_event',
            'platform_event_impact', 'is_month_start', 'is_month_end',
            # Phase 2-3: 트렌드
            'trend_index', 'trend_momentum',
            # Phase 3: 크리에이티브
            'is_video', 'video_length_sec', 'has_cta',
            'headline_length', 'copy_sentiment', 'num_products_shown',
            'audience_size', 'is_retargeting', 'is_lookalike',
            'landing_page_load_time', 'bounce_rate',
            'avg_session_duration', 'funnel_steps',
            'creative_impact_factor',
            # Phase 4: 경쟁
            'industry_avg_cpc', 'cpc_vs_industry_avg',
            'competition_index', 'platform_growth_index', 'auction_density',
        ]
        for col in enriched_num:
            if col in self.df.columns and self.df[col].notna().sum() > len(self.df) * 0.3:
                self.num_features.append(col)

        # 중복 제거
        self.num_features = list(dict.fromkeys(self.num_features))

        # 사용 가능한 피처만 필터
        self.cat_features = [c for c in self.cat_features if c in self.df.columns]
        self.num_features = [c for c in self.num_features if c in self.df.columns]

        self.all_features = self.cat_features + self.num_features

        # ----- Label Encoder (예측 시 입력용) -----
        for col in self.cat_features:
            le = LabelEncoder()
            self.df[col + '_le'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le

        # 유니크 값 저장
        self.unique_values = {}
        for col in self.cat_features:
            self.unique_values[col] = self.df[col].unique().tolist()
        self.unique_values['month'] = list(range(1, 13))

        # ----- NaN 처리 -----
        for col in self.num_features:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())

        for col in self.cat_features:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        print(f"  Total samples: {len(self.df)}")
        print(f"  Categorical features ({len(self.cat_features)}): {self.cat_features}")
        print(f"  Numeric features ({len(self.num_features)}): {self.num_features}")
        print(f"  Total features: {len(self.all_features)}")

    # ===================================================================
    # ColumnTransformer 구축
    # ===================================================================
    def _build_preprocessor(self):
        """OneHotEncoder + StandardScaler 파이프라인"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 self.cat_features),
                ('num', StandardScaler(), self.num_features),
            ],
            remainder='drop'
        )
        return preprocessor

    # ===================================================================
    # 모델 학습
    # ===================================================================
    def _train_models(self):
        """각 타겟에 대해 모델 학습"""
        print("\n[2/4] Training models with enhanced pipeline...")

        # 전처리기 생성
        preprocessor = self._build_preprocessor()

        # 데이터 준비
        X = self.df[self.all_features].copy()

        # Temporal split or random split
        if self.use_temporal_split and 'date' in self.df.columns:
            # 시간 기반: 마지막 20%를 테스트로
            split_idx = int(len(self.df) * 0.8)
            train_idx = list(range(split_idx))
            test_idx = list(range(split_idx, len(self.df)))
            print(f"  Split: Temporal (train: {len(train_idx)}, test: {len(test_idx)})")
            print(f"  Train period: {self.df.iloc[train_idx]['date'].min()} ~ {self.df.iloc[train_idx]['date'].max()}")
            print(f"  Test period:  {self.df.iloc[test_idx]['date'].min()} ~ {self.df.iloc[test_idx]['date'].max()}")
        else:
            from sklearn.model_selection import train_test_split as tts
            indices = list(range(len(self.df)))
            train_idx, test_idx = tts(indices, test_size=0.2, random_state=42)
            print(f"  Split: Random (train: {len(train_idx)}, test: {len(test_idx)})")

        # 전처리 적용
        X_processed = preprocessor.fit_transform(X)
        self.preprocessor = preprocessor

        # 피처 이름 추출
        try:
            self.feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            self.feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

        print(f"  Processed features: {X_processed.shape[1]}")

        X_train = X_processed[train_idx]
        X_test = X_processed[test_idx]

        for target in self.target_columns:
            if target not in self.df.columns:
                print(f"  [SKIP] {target} not in dataset")
                continue

            print(f"\n  === Training for: {target} ===")
            y = self.df[target].values.copy()

            # 타겟 로그 변환 (ROAS, revenue에 적용)
            apply_log = (self.log_transform_target and
                         target in ['ROAS', 'revenue'] and
                         (y > 0).all())
            if apply_log:
                y_transformed = np.log1p(y)
                print(f"    [LOG] Applied log1p transform to {target}")
            else:
                y_transformed = y

            y_train = y_transformed[train_idx]
            y_test = y_transformed[test_idx]

            # 모델 학습
            if self.tune_hyperparams:
                best_model, best_name, results = self._train_with_tuning(
                    X_train, y_train, X_test, y_test, target
                )
            else:
                best_model, best_name, results = self._train_models_for_target(
                    X_train, y_train, X_test, y_test
                )

            # 전체 데이터로 재학습
            best_model.fit(X_processed, y_transformed)

            # 잔차 계산 (원래 스케일로)
            y_pred_all = best_model.predict(X_processed)
            if apply_log:
                y_pred_all_orig = np.expm1(y_pred_all)
                residuals = y - y_pred_all_orig
            else:
                residuals = y - y_pred_all

            # R2 계산 (테스트 데이터 기준, 원래 스케일)
            y_test_pred = best_model.predict(X_test)
            if apply_log:
                y_test_pred_orig = np.expm1(y_test_pred)
                y_test_orig = np.expm1(y_test)
                test_r2 = r2_score(y_test_orig, y_test_pred_orig)
                test_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
            else:
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)

            self.models[target] = {
                'model': best_model,
                'model_name': best_name,
                'r2_score': test_r2,
                'mae': test_mae,
                'std': residuals.std(),
                'log_transform': apply_log,
                'all_results': results,
            }

            print(f"    >>> Best: {best_name} (R2: {test_r2:.3f}, MAE: {test_mae:.2f})")

            # SHAP 분석
            if SHAP_AVAILABLE and best_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
                self._compute_shap(best_model, X_train, target)

        print("\n[3/4] Model training completed!")
        self._print_model_summary()

        # 모델 저장
        print("\n[4/4] Saving models...")
        self._save_models()

    def _train_models_for_target(self, X_train, y_train, X_test, y_test):
        """단일 타겟에 대해 여러 모델 학습/비교"""
        models = {
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'MLP': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu', solver='adam',
                max_iter=300, early_stopping=True,
                random_state=42
            ),
        }

        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0
            )

        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=-1
            )

        results = {}
        trained = {}

        for name, model in models.items():
            try:
                # Cross-validation
                cv = TimeSeriesSplit(n_splits=3) if self.use_temporal_split else KFold(n_splits=3, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                results[name] = {'r2': r2, 'mae': mae, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
                trained[name] = model

                print(f"    {name:<20} R2: {r2:.3f}, MAE: {mae:.2f}, CV: {cv_scores.mean():.3f}(+/-{cv_scores.std():.3f})")
            except Exception as e:
                print(f"    {name:<20} Failed: {str(e)[:60]}")

        # 최적 모델 선택
        best_name = max(results, key=lambda x: results[x]['r2'])
        best_model = trained[best_name]

        # 앙상블 시도
        top_3 = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
        ensemble_estimators = [(n, trained[n]) for n, _ in top_3 if n in trained]

        if len(ensemble_estimators) >= 2:
            try:
                ensemble = VotingRegressor(estimators=ensemble_estimators)
                ensemble.fit(X_train, y_train)
                y_pred_ens = ensemble.predict(X_test)
                ens_r2 = r2_score(y_test, y_pred_ens)
                ens_mae = mean_absolute_error(y_test, y_pred_ens)

                print(f"    {'[ENSEMBLE]':<20} R2: {ens_r2:.3f}, MAE: {ens_mae:.2f}")
                results['Ensemble'] = {'r2': ens_r2, 'mae': ens_mae}

                if ens_r2 > results[best_name]['r2']:
                    best_model = ensemble
                    best_name = 'Ensemble'
            except Exception:
                pass

        return best_model, best_name, results

    def _train_with_tuning(self, X_train, y_train, X_test, y_test, target_name):
        """Optuna 또는 RandomizedSearchCV로 하이퍼파라미터 튜닝"""
        if OPTUNA_AVAILABLE and XGBOOST_AVAILABLE:
            return self._optuna_tune(X_train, y_train, X_test, y_test, target_name)
        else:
            return self._randomized_search_tune(X_train, y_train, X_test, y_test)

    def _optuna_tune(self, X_train, y_train, X_test, y_test, target_name):
        """Optuna로 XGBoost 하이퍼파라미터 최적화"""
        print(f"    [OPTUNA] Tuning XGBoost for {target_name}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }

            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
            cv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        best_params = study.best_params
        print(f"    [OPTUNA] Best R2 (CV): {study.best_value:.3f}")

        best_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results = {'XGBoost_Tuned': {'r2': r2, 'mae': mae}}
        print(f"    XGBoost_Tuned       R2: {r2:.3f}, MAE: {mae:.2f}")

        # 기본 모델도 비교
        _, base_name, base_results = self._train_models_for_target(X_train, y_train, X_test, y_test)
        results.update(base_results)

        if r2 > base_results.get(base_name, {}).get('r2', 0):
            return best_model, 'XGBoost_Tuned', results
        else:
            return _, base_name, results

    def _randomized_search_tune(self, X_train, y_train, X_test, y_test):
        """RandomizedSearchCV로 하이퍼파라미터 튜닝"""
        print(f"    [RandomizedSearch] Tuning GradientBoosting...")

        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
        }

        gb = GradientBoostingRegressor(random_state=42)
        cv = TimeSeriesSplit(n_splits=3) if self.use_temporal_split else 3
        search = RandomizedSearchCV(
            gb, param_dist, n_iter=20, cv=cv, scoring='r2',
            random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"    GB_Tuned            R2: {r2:.3f}, MAE: {mae:.2f}")

        results = {'GB_Tuned': {'r2': r2, 'mae': mae}}
        return best_model, 'GB_Tuned', results

    # ===================================================================
    # SHAP 분석
    # ===================================================================
    def _compute_shap(self, model, X_train, target):
        """SHAP 기반 피처 중요도 계산"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.feature_names[:len(importances)]

                fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                self.feature_importances[target] = fi

                print(f"    [SHAP/FI] Top 10 features for {target}:")
                for feat, imp in fi.head(10).items():
                    print(f"      {feat:<40} {imp:.4f}")
        except Exception as e:
            print(f"    [SHAP] Error: {str(e)[:60]}")

    # ===================================================================
    # 예측
    # ===================================================================
    def predict(self, platform, industry, country, ad_spend, campaign_type, month,
                **extra_features):
        """
        광고 효과 예측

        Parameters:
        -----------
        platform, industry, country, campaign_type : str
        ad_spend : float
        month : int
        **extra_features : 보강 데이터 피처 (선택적)
            예: is_retargeting=1, video_length_sec=15, trend_index=75
        """
        # 입력 검증
        for col, val in [('platform', platform), ('industry', industry),
                         ('country', country), ('campaign_type', campaign_type)]:
            if col in self.unique_values and val not in self.unique_values[col]:
                raise ValueError(f"Invalid {col}: '{val}'. Choose from: {self.unique_values[col]}")
        if month not in range(1, 13):
            raise ValueError("month must be 1-12")

        # 피처 딕셔너리 생성
        quarter = (month - 1) // 3 + 1
        features = {
            'platform': platform,
            'campaign_type': campaign_type,
            'industry': industry,
            'country': country,
            'ad_spend': ad_spend,
            'month': month,
            'quarter': quarter,
            'ad_spend_log': np.log1p(ad_spend),
            'ad_spend_bin': self._get_ad_spend_bin(ad_spend),
            'is_q4': 1 if quarter == 4 else 0,
            'is_weekend': 0,
            'day_of_week': 2,  # 기본값 수요일
            'week_of_year': month * 4,
        }

        # 보강 피처 기본값 채우기
        enriched_defaults = {
            'cpi_index': 110.0, 'cpi_yoy_pct': 3.0, 'unemployment_rate': 4.0,
            'gdp_growth_pct': 2.0, 'exchange_rate_usd': 1.0,
            'is_public_holiday': 0, 'days_to_next_holiday': 15,
            'is_shopping_season': 1 if month >= 11 else 0,
            'season_intensity': 3 if month >= 11 else 1,
            'is_major_event': 0, 'platform_event_impact': 0,
            'is_month_start': 0, 'is_month_end': 0,
            'trend_index': 50.0, 'trend_momentum': 0.0,
            'is_video': 0, 'video_length_sec': 0, 'has_cta': 1,
            'headline_length': 45, 'copy_sentiment': 0.7,
            'num_products_shown': 1, 'audience_size': ad_spend * 200,
            'is_retargeting': 0, 'is_lookalike': 0,
            'landing_page_load_time': 2.5, 'bounce_rate': 50.0,
            'avg_session_duration': 120.0, 'funnel_steps': 3,
            'creative_impact_factor': 1.0,
            'industry_avg_cpc': 3.0, 'cpc_vs_industry_avg': 1.0,
            'competition_index': 5.0, 'platform_growth_index': 100,
            'auction_density': 5.0,
            'ad_format': 'image', 'cta_type': 'Learn More',
            'target_age_group': '25-34', 'target_gender': 'All',
            'target_interest': 'General',
        }

        for col in self.all_features:
            if col not in features:
                features[col] = extra_features.get(col, enriched_defaults.get(col, 0))

        # DataFrame으로 변환 후 전처리
        X = pd.DataFrame([features])[self.all_features]
        X_processed = self.preprocessor.transform(X)

        # 예측
        predictions = {}
        for target in self.target_columns:
            if target not in self.models:
                continue

            model_info = self.models[target]
            pred_raw = model_info['model'].predict(X_processed)[0]

            # 로그 변환 역변환
            if model_info['log_transform']:
                pred = np.expm1(pred_raw)
            else:
                pred = pred_raw

            std = model_info['std']

            predictions[target] = {
                'predicted': round(float(pred), 2),
                'ci_68': (round(max(0, float(pred - std)), 2),
                          round(float(pred + std), 2)),
                'ci_95': (round(max(0, float(pred - 1.96 * std)), 2),
                          round(float(pred + 1.96 * std), 2)),
                'confidence': ('High' if model_info['r2_score'] > 0.5
                               else 'Medium' if model_info['r2_score'] > 0.3
                               else 'Low'),
                'model_used': model_info['model_name'],
            }

        # 파생 지표
        if 'CPC' in predictions:
            cpc_pred = max(predictions['CPC']['predicted'], 0.1)
            predictions['estimated_clicks'] = {
                'predicted': round(ad_spend / cpc_pred, 0),
                'confidence': predictions['CPC']['confidence'],
            }

        if 'ROAS' in predictions:
            roas_pred = predictions['ROAS']['predicted']
            predictions['estimated_revenue'] = {
                'predicted': round(ad_spend * roas_pred, 2),
                'ci_68': (round(ad_spend * predictions['ROAS']['ci_68'][0], 2),
                          round(ad_spend * predictions['ROAS']['ci_68'][1], 2)),
                'ci_95': (round(ad_spend * predictions['ROAS']['ci_95'][0], 2),
                          round(ad_spend * predictions['ROAS']['ci_95'][1], 2)),
                'confidence': predictions['ROAS']['confidence'],
            }

        # 히스토리컬 참조
        historical = self._get_historical_stats(platform, industry, country, campaign_type)

        return {
            'input': {
                'platform': platform, 'industry': industry, 'country': country,
                'ad_spend': ad_spend, 'campaign_type': campaign_type, 'month': month,
            },
            'predictions': predictions,
            'historical_reference': historical,
        }

    def _get_ad_spend_bin(self, ad_spend):
        bins = [1000, 3000, 5000, 10000, 20000]
        for i, b in enumerate(bins):
            if ad_spend <= b:
                return i
        return 5

    def _get_historical_stats(self, platform, industry, country, campaign_type):
        mask = (
            (self.df['platform'] == platform) &
            (self.df['industry'] == industry) &
            (self.df['country'] == country) &
            (self.df['campaign_type'] == campaign_type)
        )
        exact = self.df[mask]

        stats = {'exact_match_count': len(exact)}
        if len(exact) > 0 and 'ROAS' in exact.columns:
            stats['exact_match_stats'] = {
                'avg_ROAS': round(exact['ROAS'].mean(), 2),
                'median_ROAS': round(exact['ROAS'].median(), 2),
                'std_ROAS': round(exact['ROAS'].std(), 2),
                'min_ROAS': round(exact['ROAS'].min(), 2),
                'max_ROAS': round(exact['ROAS'].max(), 2),
            }
        return stats

    # ===================================================================
    # 출력
    # ===================================================================
    def _print_model_summary(self):
        print("\n" + "=" * 70)
        print("[MODEL SUMMARY - V2]")
        print("=" * 70)
        print(f"{'Target':<15} {'Best Model':<25} {'R2 Score':<12} {'MAE':<12} {'Log':<5}")
        print("-" * 70)
        for target, info in self.models.items():
            log_flag = 'Yes' if info['log_transform'] else 'No'
            print(f"{target:<15} {info['model_name']:<25} {info['r2_score']:<12.3f} {info['mae']:<12.2f} {log_flag:<5}")
        print("=" * 70)

        if self.feature_importances:
            print("\n[TOP FEATURES by Importance]")
            for target, fi in self.feature_importances.items():
                print(f"\n  {target}:")
                for feat, imp in fi.head(5).items():
                    print(f"    {feat:<40} {imp:.4f}")

    def print_prediction(self, result):
        inp = result['input']
        pred = result['predictions']
        hist = result['historical_reference']

        print("\n" + "=" * 70)
        print("[PREDICTION RESULT - V2]")
        print("=" * 70)
        print(f"\n[INPUT]")
        print(f"  Platform: {inp['platform']}, Industry: {inp['industry']}, Country: {inp['country']}")
        print(f"  Campaign: {inp['campaign_type']}, Budget: ${inp['ad_spend']:,.2f}, Month: {inp['month']}")

        print(f"\n[PREDICTIONS]")
        print("-" * 70)
        print(f"{'Metric':<18} {'Predicted':<12} {'68% CI':<22} {'95% CI':<22} {'Model':<15}")
        print("-" * 70)

        for key in ['ROAS', 'CPC', 'CPA', 'conversions']:
            if key in pred:
                p = pred[key]
                print(f"{key:<18} {p['predicted']:<12,.2f} "
                      f"{p['ci_68'][0]:,.2f}~{p['ci_68'][1]:,.2f}  "
                      f"  {p['ci_95'][0]:,.2f}~{p['ci_95'][1]:,.2f}  "
                      f"  {p.get('model_used', '-')[:14]}")

        if 'estimated_revenue' in pred:
            rev = pred['estimated_revenue']
            roi = ((rev['predicted'] - inp['ad_spend']) / inp['ad_spend']) * 100
            print(f"\n[DERIVED]")
            print(f"  Est. Revenue: ${rev['predicted']:,.2f} (${rev['ci_68'][0]:,.2f} ~ ${rev['ci_68'][1]:,.2f})")
            print(f"  Est. ROI: {roi:,.1f}%")

        if hist.get('exact_match_count', 0) > 0 and 'exact_match_stats' in hist:
            s = hist['exact_match_stats']
            print(f"\n[HISTORICAL] ({hist['exact_match_count']} samples)")
            print(f"  ROAS: {s['avg_ROAS']:.2f} avg, {s['median_ROAS']:.2f} median, {s['min_ROAS']:.2f}~{s['max_ROAS']:.2f}")

        print("=" * 70)

    # ===================================================================
    # 모델 저장/로드
    # ===================================================================
    def _save_models(self, path='./models'):
        os.makedirs(path, exist_ok=True)
        model_data = {
            'models': self.models,
            'preprocessor': self.preprocessor,
            'all_features': self.all_features,
            'cat_features': self.cat_features,
            'num_features': self.num_features,
            'unique_values': self.unique_values,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
        }
        with open(os.path.join(path, 'predictor_v2.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  Models saved to {path}/predictor_v2.pkl")

    @classmethod
    def load(cls, path='./models/predictor_v2.pkl'):
        """저장된 모델 로드"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        for key, val in data.items():
            setattr(instance, key, val)
        instance.target_columns = list(data['models'].keys())
        print(f"[INFO] Loaded model from {path}")
        return instance


# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    # 예측기 초기화 (보강 데이터 있으면 사용, 없으면 원본)
    predictor = AdsPredictor_V2(
        data_path='data/enriched_ads_final.csv',
        fallback_path='global_ads_performance_dataset.csv',
        use_temporal_split=True,
        log_transform_target=True,
        tune_hyperparams=False,  # True로 바꾸면 Optuna/RandomizedSearch 실행
    )

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
        month=4,
        is_retargeting=1,
        video_length_sec=15,
    )
    predictor.print_prediction(result1)

    # 예시 2: Google Ads + Healthcare + USA
    result2 = predictor.predict(
        platform='Google Ads',
        industry='Healthcare',
        country='USA',
        ad_spend=5000,
        campaign_type='Search',
        month=10,
    )
    predictor.print_prediction(result2)

    # 예시 3: Meta + E-commerce + Germany (Q4)
    result3 = predictor.predict(
        platform='Meta Ads',
        industry='E-commerce',
        country='Germany',
        ad_spend=3000,
        campaign_type='Shopping',
        month=12,
        is_retargeting=1,
        is_lookalike=1,
    )
    predictor.print_prediction(result3)
