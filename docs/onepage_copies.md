# 1page 기획서용 확정 문구 (30초 파악용)

> **문제** — 합성 데이터 시대, AI 모델의 화려한 R² 0.79를 어떻게 의심할 것인가? 데이터 파이프라인 어딘가에서 정답지가 새어 들어온 흔적을 추적한다.
>
> **결과** — 5단계 자체 감사(Shapley 분해 → sensitivity → data audit)로 bounce_rate 등 leakage 변수 3개를 식별하고 제거했다. R²는 0.79 → 0.40으로 교정되었지만, 이것이 운영 환경에서 무너지지 않는 **정직한 성능**이다.
>
> **의미** — 높은 성능 ≠ 좋은 모델이라는 교훈을 실증했다. 본 프로젝트의 핵심 산출물은 예측 모델이 아니라, **"결론이 틀렸을 때 그것을 잡아내는 재사용 가능한 분석 프레임워크"**이다.

---

## 1. 주제 (What)

AdStrategy AI는 디지털 광고 ROAS 예측을 넘어, **합성 데이터 기반 ML 프로젝트에서 '모델을 의심하는 방법'을 체계화**한 데이터 감사 프레임워크입니다. 4단계 데이터 보강 파이프라인(1,800건 → 10,030건 × 42피처)을 구축한 뒤, 5단계 ablation study로 leakage를 추적·제거하고, GPT-4o Function Calling 에이전트로 전략 의사결정까지 연결합니다.

## 2. 가설 (Why now / So what)

**H1.** Phase 3에서 생성된 bounce_rate, landing_page_load_time, creative_impact_factor는 `bounce_rate = 65 − ROAS×2`처럼 target 변수의 역함수로 만들어진 **사후 변수**이며, 이것이 R² 0.79의 약 49%를 차지하는 target leakage의 원인이다.

**H2.** 누수 변수를 제거하더라도 광고 집행 전·중에 확보 가능한 변수(예산, 플랫폼, 산업, 시즌, 경쟁 지표)만으로 R² ≈ 0.40의 설명력이 남으며, 이는 예산 배분·플랫폼 선택·캠페인 타이밍의 **전략 레버**로 변환 가능하다.

## 3. 데이터 출처 (Where)

Kaggle 공개 원본 1,800건을 4단계 파이프라인으로 **10,030건 × 42피처**로 보강(합성 약 46% 포함). Phase 2에서 FRED·World Bank·Google Trends 등 공공 API를 연동하고, Phase 3의 크리에이티브 메타데이터 생성 과정에서 leakage 3개 변수가 삽입됨. 한국 시장 세그먼트(500건)를 추가 합성하여 국내 광고주 관점의 분석도 포함.

## 4. 분석 방법 (How)

예측(XGBoost/GBR/Ridge/MLP 앙상블) + 통계 검정(ANOVA · t-test · Chi² · Mann-Whitney U) + 설명가능성(SHAP · Shapley 그룹 분해) + **ablation V1→V5**(Leave-One-Out → Shapley 분해 → Robustness → 개별 피처 분해 → Data Generation Audit)를 결합하여 **(1) 무엇이 유의미한가, (2) 왜 그렇게 예측하는가, (3) 어느 단계에서 성능이 부풀려졌는가**를 체계적으로 분해. 플랫폼별 정책 시뮬레이션(Google S형 포화, Meta 소예산 페널티, TikTok 경매 변동)도 설계·구현.

## 5. 핵심 인사이트 (What we learned)

Phase 3의 bounce_rate(단변량 R²=0.134), landing_page_load_time(R²=0.104), creative_impact_factor(R²=0.072)가 target leakage로 작동하여 R²를 0.79까지 끌어올렸음을 확인. 순차 제거 시 R²는 0.79 → 0.69 → 0.62 → 0.39로 하락하고, Creative 그룹 Shapley φ는 +0.397(86%)에서 −0.134(음수 반전)로 뒤집힘. TikTok의 통계적 우위도 재확인: Cohen's d=0.643(중), 모든 산업에서 유의 (p < 0.001).

**본 프로젝트의 3가지 산출물:**
1. **데이터 감사 파이프라인** — 합성 데이터 프로젝트에서 재사용 가능한 5단계 leakage 탐지 템플릿
2. **전략 의사결정 도구** — 예산 재할당 시뮬레이션(상위 20% 집중 시 ROAS +170%), 플랫폼별 반응 곡선
3. **AI 에이전트** — GPT-4o + Function Calling + Streamlit 기반 자연어 광고 전략 설계 인터페이스
