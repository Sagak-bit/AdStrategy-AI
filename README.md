# AdStrategy AI — AI가 자기 모델의 거짓말을 잡아낸 이야기

> **"R² 0.79는 모델의 성공이 아니라 데이터 파이프라인의 실패 증거였다."**

## 이 프로젝트가 특별한 이유

대부분의 ML 프로젝트는 R²를 높이는 것이 목표입니다. **이 프로젝트는 R²를 스스로 낮추었습니다.**

ROAS 예측 모델이 R² = 0.79를 기록했을 때, 우리는 축하 대신 **의심**했습니다. 마케팅 행동 데이터에서 이 수치는 비현실적이었습니다. 5단계 자체 감사(Ablation V1→V5)를 통해 데이터 생성 파이프라인 안에 숨어 있던 **target leakage**를 찾아냈고, `bounce_rate = 65 − ROAS × 2`라는 정답지 유출 공식을 실증했습니다.

Leakage를 제거하자 R²는 0.79 → **0.35**로 절반 이하로 떨어졌습니다. 이것이 운영 환경에서 무너지지 않는 **'정직한 성능'**이며, 이 0.35로도 예산 재할당 시뮬레이션에서 **ROAS +170% 개선**이라는 실질적 전략 가치를 끌어냈습니다.

**이 프로젝트의 산출물은 예측 모델이 아니라, 누수를 잡아내는 데이터 감사 프레임워크 그 자체입니다.**

---

## 핵심 지표

| 지표 | Before | After | 의미 |
|------|--------|-------|------|
| 데이터 규모 | 1,800건 | **10,030건** | 4단계 파이프라인으로 +457% 보강 |
| 피처 수 | 13개 | **42개** | 거시경제 + 시즌 + 크리에이티브 + 경쟁 |
| R² (leakage 포함) | 0.137 | ~~0.79~~ | **거짓 성능 — 정답지 유출** |
| R² (honest, 교정 후) | — | **0.35** | 정직한 성능, 전략 레버로 활용 가능 |
| Leakage 감사 | 없음 | **Ablation V1→V5** | 합성 데이터 프로젝트 재사용 가능 방법론 |
| 예산 재할당 효과 | — | **ROAS +170%** | honest 모델 기반 의사결정 가치 |
| 인터페이스 | CLI | **3탭 웹앱** | AI Agent + Leakage 감사 + 예산 시뮬레이터 |

---

## 데이터 흐름 (Data Flow)

```
[원본 데이터]                         [최종 출력]
global_ads_performance_dataset.csv    data/enriched_ads_final.csv
        (1,800건 × 13컬럼)                  (10,030건 × 42컬럼)

                    ┌──────────────────────────────┐
                    │    data_pipeline.py           │
                    │    run_pipeline() 함수        │
                    └──────────┬───────────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    ▼                          ▼                          ▼
Phase 1                    Phase 2                    Phase 3~4
kaggle_data_merger.py      macro_economic_collector    creative_data_enricher
  ↓ 시계열 확장             holiday_calendar            competition_and_synthetic
  ↓ 1,800 → 5,400건        google_trends_collector      ↓ ⚠ leakage 변수 생성
  ↓                          ↓ 외부 맥락 조인             ↓ 합성 데이터 증강
  └──────────────────────────┴──────────────────────────┘
                               │
                    data/phase1_expanded.csv (중간 산출물)
                               │
                    data/enriched_ads_final.csv (최종)
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ads_predictor_v2.py   ads_analysis.py    ablation_study*.py
    (ML 모델 학습)         (시각화 분석)      (모델 감사)
            │
    models/predictor_v2.pkl
            │
    ┌───────┴───────┐
    ▼               ▼
ad_agent.py    mcp_ad_server.py
(GPT-4o 에이전트)  (MCP 서버)
    │
streamlit_app.py
(웹 UI)
```

### 데이터 파일 관계

| 파일 | 생성 단계 | 행 수 | 컬럼 수 | 설명 |
|------|-----------|-------|---------|------|
| `global_ads_performance_dataset.csv` | 원본 | ~1,800 | 13 | Kaggle 원본 광고 캠페인 데이터 |
| `data/phase1_expanded.csv` | Phase 1 후 | ~5,400 | 13 | 시계열 확장 완료 |
| `data/enriched_ads_final.csv` | Phase 4 후 | ~10,030 | 42 | 모든 보강 데이터 통합 (최종) |
| `data/macro/macro_economic_data.csv` | Phase 2-1 | - | - | 거시경제 지표 (CPI, 실업률, GDP, 환율) |
| `data/calendar/holiday_calendar.csv` | Phase 2-2 | - | - | 국가별 공휴일/시즌 캘린더 |
| `data/trends/industry_trends.csv` | Phase 2-3 | - | - | 산업별 Google Trends 관심도 |
| `models/predictor_v2.pkl` | 모델 학습 | - | - | 학습된 AdsPredictor V2 모델 |

---

## 프로젝트 구조

```
AdStrategy-AI/
├── streamlit_app.py          # Streamlit 웹 UI (채팅 + 차트)
├── ad_agent.py               # AI 에이전트 (GPT-4o + Function Calling)
├── ads_predictor_v2.py       # ML 예측 모델 (AdsPredictor V2)
├── tool_schemas.py           # 공통 Tool 스키마 (DRY: Agent + MCP 공유)
├── ads_analysis.py           # 글로벌 광고 성과 분석 (시각화 → figures/)
├── tiktok_deep_analysis.py   # TikTok 심층 분석 (시각화 → figures/)
├── advanced_statistical_analysis.py  # 고급 통계 분석 (시각화 → figures/)
├── data_pipeline.py          # 4단계 데이터 수집/보강 파이프라인 (롤백 지원)
├── mcp_ad_server.py          # MCP 서버 (외부 LLM 연동)
├── data_collectors/          # Phase 1~4 데이터 수집 모듈
│   ├── kaggle_data_merger.py       # Phase 1: Kaggle 데이터 병합
│   ├── macro_economic_collector.py  # Phase 2-1: 거시경제 (FRED, World Bank)
│   ├── holiday_calendar.py          # Phase 2-2: 시즌/공휴일 캘린더
│   ├── google_trends_collector.py  # Phase 2-3: Google Trends
│   ├── creative_data_enricher.py   # Phase 3: 크리에이티브 메타데이터 (⚠ leakage 주의)
│   └── competition_and_synthetic.py # Phase 4: 경쟁 환경 + 합성 데이터
├── data/
│   ├── enriched_ads_final.csv      # 최종 보강 데이터셋 (10,030건 × 42컬럼)
│   ├── macro/                      # 거시경제 데이터
│   ├── calendar/                   # 공휴일/시즌 캘린더
│   └── trends/                     # Google Trends 데이터
├── figures/                   # 분석 시각화 산출물 (01~13번 PNG)
├── models/
│   └── predictor_v2.pkl            # 학습된 ML 모델
├── archive/                   # 보관용 (ablation 스크립트, 구 보고서, deprecated 코드)
│   ├── ablation_studies/      # Ablation V1~V5 스크립트 및 결과 PNG
│   ├── deprecated_code/       # ads_predictor.py, ads_predictor_advanced.py
│   ├── reports_old/            # LaTeX v1/v2 및 빌드 산출물
│   └── pdfs/                   # 과거 PDF 산출물
├── requirements.txt
├── ads_report_v3.tex / .pdf   # 최신 보고서
└── .env.example
```

---

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY=sk-proj-... 입력
```

| API 키 | 필수 여부 | 용도 | 발급 URL |
|--------|----------|------|----------|
| `OPENAI_API_KEY` | **필수** (Streamlit 앱) | GPT-4o 에이전트 | [platform.openai.com](https://platform.openai.com) |
| `FRED_API_KEY` | 선택 | 거시경제 데이터 수집 | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |

> API 키 없이도 분석 스크립트(`ads_analysis.py` 등)는 실행 가능합니다.
> OpenAI API 키는 Streamlit 앱의 대화형 에이전트 기능에만 필요합니다.

### 3. Streamlit 앱 실행

```bash
streamlit run streamlit_app.py
# 브라우저에서 http://localhost:8501 접속
```

### 4. 분석 스크립트 실행

```bash
# 기본 분석 (시각화 5장 → figures/)
python ads_analysis.py

# TikTok 심층 분석 (시각화 3장 → figures/)
python tiktok_deep_analysis.py

# 고급 통계 분석 (시각화 5장 → figures/ + analysis_results.json)
python advanced_statistical_analysis.py

# 데이터 파이프라인 실행 (Phase별 롤백 지원)
python data_pipeline.py
```

### 5. MCP 서버 실행

```bash
# MCP 프로토콜로 실행 (외부 LLM 에이전트 연동)
pip install mcp
python mcp_ad_server.py

# 테스트 모드 (MCP 없이 단독 실행)
python mcp_ad_server.py --test
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 데이터 수집 | FRED API, World Bank API, Google Trends, Holidays |
| 머신러닝 | scikit-learn, XGBoost, LightGBM, SHAP, Optuna |
| AI 에이전트 | OpenAI GPT-4o Function Calling |
| MCP 서버 | Model Context Protocol (mcp) |
| 웹 UI | Streamlit, Plotly |
| 통계 검증 | SciPy (ANOVA, t-test, Chi², Mann-Whitney U, VIF) |
| 배포 | Streamlit Community Cloud |

---

## 분석 파이프라인

### 데이터 보강 (4단계)

1. **Phase 1**: Kaggle 유사 데이터 병합 + 시계열 확장 (1,800 → 5,400건)
2. **Phase 2**: 외부 맥락 데이터 (거시경제 + 시즌/공휴일 + Google Trends)
3. **Phase 3**: 크리에이티브/타겟팅 메타데이터 시뮬레이션
4. **Phase 4**: 경쟁 환경 지표 + 합성 데이터 (5,400 → 10,030건)

### 분석 체계

| 분석 | 파일 | 시각화 |
|------|------|--------|
| 글로벌 성과 분석 | `ads_analysis.py` | `figures/` 01~05_*.png |
| 고급 통계 검증 | `advanced_statistical_analysis.py` | `figures/` 06~10_*.png |
| TikTok 심층 분석 | `tiktok_deep_analysis.py` | `figures/` 11~13_*.png |
| Ablation V1~V5 | `archive/ablation_studies/` | 동일 폴더 내 PNG |

### 통계 검증 (advanced_statistical_analysis.py)

- **데이터 품질 감사**: 결측치, 이상치(IQR), 정규성 검정(Shapiro-Wilk)
- **ANOVA**: 플랫폼/산업/국가 간 ROAS 차이 유의성 검증
- **사후 검정**: Bonferroni 보정 Pairwise t-test + Cohen's d 효과 크기
- **카이제곱 검정**: 플랫폼과 고성과 캠페인 연관성
- **VIF (다중공선성)**: 회귀 분석 피처 간 Variance Inflation Factor 검사
- **크로스 분석**: 거시경제 지표(CPI, 실업률, GDP) × 광고 성과 상관관계
- **합성 데이터 영향**: 실제 vs 합성 데이터 분리 분석, 순위 변동 검증
- **요인 분해**: 표준화 다중 회귀로 ROAS 기여 요인 분해 (leakage 포함/제외 비교)

> 분석 실행 시 `analysis_results.json`이 자동 생성됩니다.
> 보고서 작성 시 이 파일의 수치를 참조하세요.

---

## AI 에이전트

### 대화 흐름 (5단계)

1. **비즈니스 이해**: 산업, 제품, 광고 목표
2. **타겟 파악**: 국가, 연령, 성별
3. **예산/일정**: 월 예산, 시작 시기
4. **전략 설계**: ML 예측 기반 플랫폼/캠페인 비교
5. **최적화**: What-if 시나리오 비교

### Function Calling Tools (4개)

| Tool | 설명 |
|------|------|
| `predict_ad_performance` | 조건별 ROAS/CPC/CPA/전환수 예측 |
| `compare_scenarios` | 2~5개 시나리오 동시 비교 |
| `get_historical_benchmarks` | 과거 벤치마크 통계 조회 |
| `get_industry_trends` | 산업별 트렌드 지수 조회 |

> Tool 스키마는 `tool_schemas.py`에서 공통 관리됩니다 (Agent와 MCP 서버 공유).

### MCP 서버

`mcp_ad_server.py`는 Model Context Protocol을 통해 위 4개 Tool + `get_data_summary`를
외부 LLM 에이전트(Cursor, Claude Desktop 등)에 제공합니다.

---

## 데이터 투명성 및 한계

### 합성 데이터 정책

- `is_synthetic` 컬럼으로 실제/합성 데이터 구분 (합성 비율 약 46%)
- `advanced_statistical_analysis.py`에서 합성 데이터 영향도 정량 분석
- Phase 3(크리에이티브), Phase 4(경쟁 환경) 데이터는 통계 시뮬레이션 기반
- 주요 인사이트는 실제 데이터만으로도 검증 (플랫폼/산업 순위 변동 없음)

### Data Generation Audit (Target Leakage)

Ablation study 과정에서 Phase 3 데이터 생성 코드(`creative_data_enricher.py`)에
**target leakage**가 내장되어 있음을 발견:

- `bounce_rate = max(20, 65 - ROAS * 2) + noise` -- ROAS를 입력으로 사용
- `landing_page_load_time = max(0.5, 4.0 - ROAS * 0.1) + noise` -- ROAS를 입력으로 사용
- `creative_impact_factor`는 위 두 변수로부터 파생

이 3개 변수는 ROAS의 역함수이므로, 이를 ROAS 예측 피처로 사용하면 모델이 역함수를 복원하여
표면적 R²가 0.79~0.91로 부풀려짐. **Leakage 변수 제거 시 honest R²는 0.40~0.53.**

이 발견은 Shapley 분해 → sensitivity analysis → data generation audit → leakage-free 재검증의
체계적 분석 파이프라인을 통해 도출되었으며, 합성 데이터 기반 프로젝트에서 모델 성능을 자체 감사하는
방법론의 유효성을 보여줌.

### 데이터 소스 신뢰도

| Phase | 데이터 소스 | 신뢰도 | 비고 |
|-------|------------|--------|------|
| Phase 1 | Kaggle 원본 + 시계열 확장 | 높음 | 실제 광고 캠페인 기반 |
| Phase 2 (API) | FRED, World Bank, Google Trends | 높음 | 공공 데이터 |
| Phase 2 (Fallback) | 통계적 시뮬레이션 | 중간 | API 실패 시 사용, 실제와 차이 가능 |
| Phase 3 | 크리에이티브 메타데이터 | 낮음 | 통계 분포 기반 생성, **leakage 포함** |
| Phase 4 | 경쟁 환경 + 합성 데이터 | 중간 | 원본 분포 보존, 인과 구조 미보장 |

### 외부 API Fallback

| API | 용도 | Fallback |
|-----|------|----------|
| FRED | CPI, 실업률 | 통계적 시뮬레이션 |
| Google Trends | 산업 관심도 | 계절성 기반 추정 |
| World Bank | GDP 성장률 | 최근 공개 데이터 |
| Kaggle | 유사 데이터셋 | 시계열 확장만 수행 |

### 분석 인사이트 해석 가이드

| 인사이트 유형 | 신뢰 수준 | 해석 방법 |
|-------------|----------|----------|
| 플랫폼/산업/국가 순위 | 중~높음 | 실제 데이터만으로도 동일 순위 유지, ANOVA 유의 |
| 개별 캠페인 ROAS 예측값 | 낮~중 | Honest R²=0.40~0.53, 방향성 참고만 권장 |
| "리타겟팅이 효과적" 등 피처 수준 인사이트 | 낮음 | 합성 데이터 기반, 가설 생성 수준으로 해석 |
| 거시경제 교차 효과 | 낮음 | Shapley φ 음수, 현재 데이터 범위에서 신호 약함 |

---

## 트러블슈팅

### 자주 발생하는 문제

**Q: `ModuleNotFoundError: No module named 'xgboost'`**
```bash
pip install xgboost lightgbm
```

**Q: Streamlit 앱에서 "에이전트 초기화 실패"**
- `.env` 파일에 `OPENAI_API_KEY`가 올바르게 설정되어 있는지 확인
- API 키가 `sk-proj-`로 시작하는지 확인
- OpenAI 계정에 크레딧이 남아있는지 확인

**Q: `data/enriched_ads_final.csv`가 없음**
```bash
python data_pipeline.py  # 데이터 파이프라인 실행
```

**Q: 한글 폰트 깨짐 (시각화)**
- Windows: `Malgun Gothic` 폰트 자동 사용
- macOS: `plt.rcParams['font.family'] = 'AppleGothic'`으로 변경
- Linux: `sudo apt-get install fonts-nanum && fc-cache -fv`

**Q: MCP 서버 실행 오류**
```bash
pip install mcp  # MCP 패키지 설치
python mcp_ad_server.py --test  # 테스트 모드로 먼저 확인
```

**Q: 모델 학습을 다시 하고 싶음**
```bash
python ads_predictor_v2.py  # 새 모델 학습 → models/predictor_v2.pkl 저장
```

---

## 배포

- **서비스 URL**: https://adstrategy-ai.streamlit.app/
- **GitHub**: https://github.com/Sagak-bit/AdStrategy-AI
- **배포 플랫폼**: Streamlit Community Cloud
- **자동 배포**: GitHub push 시 자동 재배포 (CI/CD)
