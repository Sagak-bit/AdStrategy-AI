# AdStrategy AI — 합성 데이터 시대의 모델 감사 프레임워크

> **"R² 0.79는 모델의 성공이 아니라 데이터 파이프라인의 실패 증거였다."**
> 높은 성능 ≠ 좋은 모델이라는 교훈을 실증한, 재사용 가능한 데이터 감사 프로젝트.

---

## 1. Hero (한 줄 요약)

Kaggle 광고 데이터 1,800건을 4단계 파이프라인으로 10,030건×42피처로 보강한 뒤, 5단계 ablation으로 target leakage를 적발·제거하여 "정직한 R² 0.40"으로 교정하고, 이를 전략 의사결정 레버로 변환한 **데이터 감사 프레임워크**.

**핵심 deliverable은 예측 모델이 아니라, "결론이 틀렸을 때 그것을 잡아내는 분석 프레임워크".**

---

## 2. 문제 정의 (갈등)

초기 ROAS 예측 모델이 R² = 0.79를 기록했다. 마케팅 행동 데이터에서 이 수치는 **비현실적으로 높다**. 의심: Phase 3(크리에이티브 메타데이터 생성)에서 `bounce_rate = 65 − ROAS×2`처럼 타겟 변수의 역함수로 피처를 생성한 것이 원인일 수 있다.

> **핵심 질문**: "이 모델은 진짜로 광고 성과를 예측하는가, 아니면 정답지를 베끼고 있는가?"

---

## 3. 데이터 & 파이프라인 (구조)

| Phase | 내용 | 건수 변화 |
|-------|------|-----------|
| Phase 1 | Kaggle 원본 + 시계열 확장 | 1,800 → ~5,400 |
| Phase 2 | 거시경제(GDP, CPI) + 공휴일 + Google Trends | 피처 +15 |
| Phase 3 | 크리에이티브/타겟팅 메타데이터 시뮬레이션 | 피처 +18 (**leakage 3개 포함**) |
| Phase 4 | 경쟁 환경 + 합성 데이터(46%) + 한국 세그먼트(500건) | → 10,530건 × 42피처 |

**시각화**: `figures/waterfall_r2_leakage.png` — R² 하락 궤적 폭포수 차트

---

## 4. 분석 설계 (추적)

- **예측 모델**: XGBoost + GradientBoosting + Ridge + MLP → VotingRegressor 앙상블
- **타겟**: ROAS, CPC, CPA, conversions (로그 변환 적용)
- **평가**: 5-fold TimeSeriesSplit CV (시간 기반 분할로 미래 정보 유출 방지)
- **해석**: SHAP feature importance + 단변량 Ridge leakage 스코어보드
- **통계 검증**: ANOVA, t-test, Chi², Mann-Whitney U, VIF
- **시뮬레이션**: 플랫폼별 정책 반응 곡선 (Google S형, Meta 페널티, TikTok 변동)

---

## 5. Leakage 감사 로그 (증거)

### 5-1. 단변량 leakage 스코어보드

각 피처 하나만으로 Ridge → log(1+ROAS) R²를 측정한 결과, `bounce_rate`, `creative_impact_factor`, `landing_page_load_time`이 일반 피처 대비 비정상적으로 높은 단독 설명력을 보임.

**시각화**: `figures/leakage_risk_dashboard.png` — 피처별 단독 R² 막대 차트

### 5-2. Ablation V1→V5

| 단계 | 제거 피처 | R² (CV) | delta |
|------|-----------|---------|-------|
| V1 Full | — | 0.7850 | — |
| V2 | bounce_rate | 0.6867 | −0.0983 |
| V3 | + landing_page_load_time | 0.6215 | −0.0652 |
| V4 | + creative_impact_factor | 0.3473 | −0.2742 |

**시각화**: `figures/waterfall_r2_leakage.png` — R² 폭포수 차트

### 5-3. SHAP 대조

V1에서 `bounce_rate`가 feature importance의 49.8%를 차지했으나, V5(clean)에서는 `cpc_vs_industry_avg`(33.5%), `industry_avg_cpc`(12.6%) 등 **광고주가 통제 가능한 변수**가 상위를 차지.

**시각화**: `figures/shap_force_leakage_vs_clean.png` — SHAP 기여도 대조 차트

---

## 6. 누수 제거 전후 비교 (전환점)

### 가드레일 검증

셔플 기준선(R² = −0.095)과 난수 피처 기준선(R² = −0.100) 대비, clean 모델의 R² = 0.347은 명확히 "진짜 설명력" 영역에 위치.

**시각화**: `figures/ablation_guardrail.png` — Ablation 가드레일 플롯

### 합성 데이터 충실도

44개 수치 피처 K-S 검정 결과 전체가 p<0.05(분포 상이). Exact Match 0건으로 프라이버시 리스크 없음.

**시각화**: `figures/synthetic_fidelity_heatmap.png` — 합성 데이터 품질 히트맵

---

## 7. 전략 변환 (의사결정)

### 예산 재할당 시뮬레이션

예측 ROAS 상위 20%에 예산 70%를 집중 배분하면, 균등 배분 대비 **가중 ROAS +170.6% 개선**.

**시각화**: `figures/budget_reallocation_impact.png` — 예산 재할당 효과 차트

### 세그먼트 신뢰도 맵

플랫폼×산업별 예측 MAE 히트맵으로, 어떤 세그먼트에서 모델을 믿을 수 있는지 시각적으로 제시.

**시각화**: `figures/segment_confidence_heatmap.png` — 세그먼트 신뢰도 맵

### 플랫폼별 반응 곡선

Google(S형 포화), Meta(소예산 페널티), TikTok(선형+변동) 정책 반영 시뮬레이션.

**시각화**: `figures/platform_budget_response_curves.png` — 플랫폼 반응 곡선

### 글로벌 분석 시각화

| 시각화 파일 | Notion 섹션 | 설명 |
|------------|------------|------|
| `figures/01_overall_performance.png` | 부록 | 플랫폼별 전체 성과 |
| `figures/02_cross_analysis_heatmap.png` | 부록 | 교차 분석 히트맵 |
| `figures/03_monthly_trends.png` | 부록 | 월별 트렌드 |
| `figures/04_spend_efficiency.png` | 부록 | 예산 효율성 |
| `figures/05_top_bottom_comparison.png` | 부록 | 상위/하위 비교 |
| `figures/06_data_quality_audit.png` | 4. 분석 설계 | 데이터 품질 감사 |
| `figures/07_statistical_significance.png` | 4. 분석 설계 | 통계적 유의성 |
| `figures/08_macro_cross_analysis.png` | 부록 | 거시경제 교차 |
| `figures/09_synthetic_impact.png` | 6. 전후 비교 | 합성 데이터 영향 |
| `figures/10_factor_decomposition.png` | 5. 감사 로그 | 요인 분해 |
| `figures/11_tiktok_platform_comparison.png` | 부록 | TikTok 비교 |
| `figures/12_tiktok_industry_advantage.png` | 부록 | TikTok 산업별 |
| `figures/13_tiktok_optimal_strategy.png` | 부록 | TikTok 최적 전략 |

---

## 8. 부록

### 프로젝트 구조

```
Challengers/
├── data/                    # 4단계 파이프라인 데이터
├── data_collectors/         # Phase 1~4 수집기
├── figures/                 # 시각화 산출물 (20장)
├── docs/                    # 기획서, 스크립트, 방어 논리
├── scripts/                 # 분석·시뮬레이션 스크립트
├── ad_agent.py              # GPT-4o AI 에이전트
├── ads_predictor_v2.py      # ML 예측 모델
├── streamlit_app.py         # 웹 UI (3탭: 에이전트/감사/시뮬레이터)
├── mcp_ad_server.py         # MCP 서버 (외부 LLM 연동)
└── config.py                # 프로젝트 공유 설정
```

### 핵심 수치 요약

| 항목 | 값 |
|------|-----|
| 원본 데이터 | 1,800건 (Kaggle) |
| 보강 후 | 10,030건 × 42피처 (+ Korea 500건) |
| 합성 비율 | 46% |
| R² (leakage 포함) | 0.7850 |
| R² (leakage 제거) | 0.3473 ~ 0.40 |
| Leakage 변수 | bounce_rate, landing_page_load_time, creative_impact_factor |
| 검증 방법 | 5-fold TimeSeriesSplit CV |
| 예산 재할당 효과 | 상위 20% 집중 시 ROAS +170.6% |
| 플랫폼 정책 | Google S형, Meta 페널티, TikTok 변동 시뮬레이션 |

### Streamlit 앱

- **서비스 URL**: https://adstrategy-ai.streamlit.app/
- **3개 탭**: AI 에이전트 (대화형 전략 설계) / Leakage 감사 (대시보드) / 예산 시뮬레이터 (정책 반영)
- API 키 없이도 Leakage 감사·예산 시뮬레이터 탭 사용 가능
