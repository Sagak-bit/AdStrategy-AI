# AdStrategy AI — 1주일 고도화 파이프라인

**마감**: 다음 주 화요일 17:00  
**제출물**: Notion 공개 URL · 1page 기획서 PDF · 1분 브리핑 영상 MP4 · (선택) 분석 코드  
**기준**: GPT + Gemini AgentReport 반영, Cursor 활용으로 일당 작업량 상향

---

## 일별 한눈에 보기

| 일차 | 핵심 산출물 | Cursor로 할 일 |
|------|-------------|----------------|
| **D-7** | Waterfall R² 차트, 누수 점수보드, 1page 문장 | 시각화 2개 구현 + `docs/onepage_copies.md` |
| **D-6** | 합성 데이터 3지표 표, R² 방어 문단 | `synthetic_quality_report.py`, `r2_philosophy_defense.md` |
| **D-5** | SHAP 대조·가드레일 그림, 시뮬레이션 설계 | XAI 시각화 2개 + strategy_simulation 설계 |
| **D-4** | 예산 재할당 그림, 세그먼트 신뢰도 맵 | 시뮬레이션 실행 + heatmap |
| **D-3** | Notion 8섹션, 1page PDF 초안 | Notion 본문 채우기 + PDF 레이아웃 |
| **D-2** | 1분 스크립트·스토리보드, (선택) Streamlit | 나레이션 확정 + 녹화 준비 |
| **D-1** | 1분 영상 MP4, 최종 점검 | 녹화 + Notion/PDF 최종 수정 |
| **D-day** | 제출 | URL · PDF · MP4 제출 |

---

## 창의 설계 (한국 데이터 + 플랫폼 정책)

시간이 허락하면 아래 두 가지를 넣으면 **상상력·영향력**이 크게 올라갑니다. 상세 설계는 `docs/CREATIVE_DESIGN_한국_플랫폼정책.md` 참고.

| 설계 | 요약 | 넣을 일차 | 산출물 |
|------|------|-----------|--------|
| **한국(·아시아) 데이터 확장** | 원본 7개국만 있는데 한국 없음 → 참조국(India 등) 기반 스케일링으로 `country=Korea` 합성 행 추가 | D-6 또는 D-5 | `data/enriched_ads_with_korea.csv`(또는 기존 파일 확장), `tool_schemas.py`에 Korea 추가 |
| **플랫폼별 정책 시뮬레이션** | 유튜브(예산↑→노출 포화), 메타(소예산 학습 구간 불리), 틱톡(경매 변동) 등 반영한 “예산→기대 ROAS” 곡선 | D-5 설계 + D-4 구현 | 플랫폼별 반응 곡선 1장, 예산 재할당 시 “플랫폼 유효 구간” 반영 |

- **한국**: “한국 시장 세그먼트를 반영해 국내 광고주도 참고 가능” 서사.
- **플랫폼 정책**: “실제 광고 플랫폼 정책(학습 구간·포화)을 가정한 가상 시뮬레이션” 서사.

---

## 전체 전략

| 목표 | 전략 |
|------|------|
| **30초 핵심** | "R² 0.79→0.40 = 실패가 아니라 정직해진 전환점" 한 줄로 고정 |
| **서사** | 갈등(의심) → 추적(감사) → 증거(전후 그래프) → 결론(전략) |
| **평가** | 상상력: 서사·시각화 / 실행력: 감사·검증 / 영향력: 전략·의사결정 |

---

## D-7 (오늘): 기반 시각화 + 1page 초안

**목표**: 제출물에 들어갈 **핵심 그림 2개** 완성 + 1page 문장 확정.

| 순서 | 작업 | 산출물 | 참고 (리포트) | Cursor 프롬프트 예시 |
|------|------|--------|----------------|----------------------|
| 1 | **R² 폭포수(Waterfall) 차트** | `figures/waterfall_r2_leakage.png` | Gemini #1, GPT 서사 | "ablation V1~V5 단계별 R²를 waterfall로: 0.79 → bounce_rate 제거 → load_time 제거 → 0.40. plotly 또는 matplotlib." |
| 2 | **누수 위험 점수보드** (피처 단독 예측력 or 사전/사후 태깅) | `figures/leakage_risk_dashboard.png` | GPT 분석 고도화 #1 | "각 피처 단독으로 간단 회귀 R² 구해서 막대차트, bounce_rate/load_time이 비정상 높게 나오는지 시각화" |
| 3 | **1page 기획서 문장 확정** | `docs/onepage_copies.md` | GPT 원페이지 예시 A/B | "GPT 리포트의 주제·가설·데이터·방법·인사이트 문장을 우리 수치에 맞게 한 줄씩 수정해 docs/onepage_copies.md에 정리" |

**완료 체크**: `figures/`에 waterfall + leakage 대시보드 PNG 2개, `docs/onepage_copies.md` 5개 항목 채움.

---

## D-6: 합성 데이터 검증 + R² 방어 논리

**목표**: 약점(46% 합성, R² 0.40)을 **수치로 방어**하고, Notion/기획서용 문단 확보.

| 순서 | 작업 | 산출물 | 참고 | Cursor 프롬프트 예시 |
|------|------|--------|------|----------------------|
| 1 | **합성 데이터 3대 지표** (Fidelity K-S, Utility 비교, Exact Match) | `scripts/synthetic_quality_report.py` → 표 + 짧은 해석 | Gemini #2, GPT #2 | "enriched_ads_final에서 is_synthetic로 나눠, 연속형은 ks_2samp, 원본만 vs 합성포함 모델 R², exact duplicate 개수. 결과를 표로 출력하고 3줄 해석." |
| 2 | **Real vs Synthetic 분포 비교** (피처별 K-S 또는 히트맵 1장) | `figures/synthetic_fidelity_heatmap.png` | GPT #2 | "주요 연속 변수별 real vs synthetic 분포 차이를 K-S 통계로 하고, 히트맵 또는 막대 1장으로 저장" |
| 3 | **"왜 R² 0.40이 충분한가" 문단** (마케팅/행동 데이터 문헌 근거) | `docs/r2_philosophy_defense.md` | Gemini #3, GPT | "마케팅·행동 데이터에서 R² 0.2~0.5가 실무적 타당성 있다는 식으로 1page용 2~3문장 + 참고문헌 1~2개" |
| 4 | **(선택) 한국 세그먼트 확장** | `scripts/add_korea_segment.py` → `data/enriched_ads_with_korea.csv` | `docs/CREATIVE_DESIGN_한국_플랫폼정책.md` | "enriched_ads_final에서 India 참조로 Korea 행 합성(스케일링), is_synthetic=True. tool_schemas COUNTRIES에 Korea 추가." |

**완료 체크**: 합성 품질 표·해석 1세트, `figures/synthetic_fidelity_heatmap.png`, R² 방어 문단 파일. (선택) 한국 포함 데이터셋.

---

## D-5: XAI 고도화 + 전략 시뮬레이션 설계

**목표**: SHAP 대조 시각화 + **의사결정 시뮬레이션** 로직/데이터 준비.

| 순서 | 작업 | 산출물 | 참고 | Cursor 프롬프트 예시 |
|------|------|--------|------|----------------------|
| 1 | **SHAP Force Plot 대조** (누수 모델 vs 정직 모델, 동일 관측치) | `figures/shap_force_leakage_vs_clean.png` | GPT #4, Gemini #4 | "V1(leakage 포함)과 leakage 제거 모델에 대해 같은 행으로 SHAP force plot 두 개 나란히. bounce_rate가 V1에서 압도적 기여하는 걸 보여줘." |
| 2 | **가드레일 플롯** (무작위 셔플/난수 피처 R² 기준선 + ablation 곡선) | `figures/ablation_guardrail.png` | Gemini #4 | "ablation으로 피처 제거 시 R² 곡선 그리고, 셔플 라벨/난수 피처 R² 수평선을 가드레일로 겹쳐서 저장" |
| 3 | **전략 시뮬레이션 설계** (예산 상위 k% vs 균등 배분, 백테스트 구조) | `scripts/budget_simulation_spec.md` 또는 `scripts/strategy_simulation.py` 골격 | GPT #3, Gemini #5 | "예측 ROAS 상위 캠페인에 예산 가중 vs 균등 배분 시, holdout 또는 시계열 분할로 기대 ROAS 차이 계산하는 함수 설계. 입력: df, 모델, k%; 출력: 기대 ROAS, 개선율." |
| 4 | **(창의) 플랫폼별 정책 가정 정리** (Google S형 포화, Meta 최소 유효 예산, TikTok 선형+변동) | `scripts/platform_policy_params.py` 또는 명세 | `docs/CREATIVE_DESIGN_한국_플랫폼정책.md` | "플랫폼별 budget→effective_multiplier 또는 min_effective_budget 파라미터를 코드/딕셔너리로 정의. 주석에 실무 근거 1줄씩." |

**완료 체크**: SHAP 대조 1장, 가드레일 1장, 시뮬레이션 스크립트 또는 명세서. (선택) 플랫폼 정책 파라미터.

---

## D-4: 전략 시뮬레이션 구현 + 세그먼트 신뢰도

**목표**: "결정이 바뀌는 그림" 1장 + 세그먼트별 신뢰도 맵.

| 순서 | 작업 | 산출물 | 참고 | Cursor 프롬프트 예시 |
|------|------|--------|------|----------------------|
| 1 | **예산 재할당 시뮬레이션 실행** (상위 k% vs 균등 → 기대 ROAS/개선율) | `scripts/strategy_simulation.py` 결과 표 + `figures/budget_reallocation_impact.png` | GPT #3, Gemini #5 | "strategy_simulation 실행해서 결과 표 저장하고, 예산 이동 전후 기대 ROAS 차이를 막대 또는 폭포수 1장으로 시각화" |
| 2 | **세그먼트 신뢰도 맵** (플랫폼×산업 등 조합별 RMSE/MAE 히트맵) | `figures/segment_confidence_heatmap.png` | GPT #4 | "플랫폼·산업(또는 국가) 조합별로 holdout RMSE/MAE 계산해서 히트맵, 신뢰할 수 있는 세그먼트 강조" |
| 3 | **(선택) 반응 곡선 S/C형** (채널별 예산–ROAS 곡선, 한계 효용 구간 표시) | `figures/response_curve_by_channel.png` | Gemini Case 1, #5 | "채널별 ad_spend 구간과 예측 ROAS로 스무스 곡선 그려서 포화 구간 표시. 없으면 D-3으로 미룸." |
| 4 | **(창의) 플랫폼별 예산–ROAS 반응 곡선** (정책 반영 시뮬레이션) | `figures/platform_budget_response_curves.png` | `docs/CREATIVE_DESIGN_한국_플랫폼정책.md` | "Google S형, Meta 소예산 구간 페널티, TikTok 선형 가정으로 예산 구간별 기대 ROAS 곡선 3개 한 그림에. 플랫폼 정책 시뮬레이션 결과 1장." |

**완료 체크**: 예산 재할당 그림 1장, 세그먼트 히트맵 1장. (선택) 플랫폼별 반응 곡선 1장.

---

## D-3: Notion 골격 + 기획서 PDF 초안

**목표**: Notion **8개 섹션 채우기**(텍스트+이미지) + 1page PDF 초안.

| 순서 | 작업 | 산출물 | 참고 | Cursor 프롬프트 예시 |
|------|------|--------|------|----------------------|
| 1 | **Notion 페이지 생성** (웹에 게시 설정) | Notion URL | GPT Notion 구성안 | "README와 INSIGHTS 기반으로 Notion용 마크다운 초안 생성: Hero 한줄, 문제정의, 데이터·파이프라인, 분석설계, Leakage 감사 로그, 누수 제거 전후 비교, 전략 변환, 부록. 각 섹션 2~3문장 + 이미지 경로 placeholder." |
| 2 | **Notion 본문 채우기** (D-7~D-4 그림 삽입, 문구 붙여넣기) | Notion 업데이트 | - | "figures/ 목록 주고, 어떤 그래프를 어떤 섹션에 넣을지 매핑 표 만들어줘. 그 다음 각 섹션에 들어갈 2~3문장 초안." |
| 3 | **1page 기획서 PDF 초안** (상단 3줄 + 5항목 + ROAS/CPC/CPA 정의 박스) | `docs/onepage_proposal.pdf` 또는 Google Docs/Word | GPT 원페이지 + Gemini | "onepage_copies.md와 r2_philosophy_defense.md 사용해서, 상단 문제–결과–의미 3줄, 아래 주제·가설·데이터·방법·인사이트, 우측 작은 박스에 지표 정의. 1페이지 레이아웃으로 구성해줘." |

**완료 체크**: Notion URL 공개 가능, 1page PDF 1차 버전.

---

## D-2: 1분 영상 스크립트 확정 + Streamlit(선택)

**목표**: 나레이션 최종본 + (가능하면) 인터랙티브 대시보드 1페이지.

| 순서 | 작업 | 산출물 | 참고 | Cursor 프롬프트 예시 |
|------|------|--------|------|----------------------|
| 1 | **1분 영상 스크립트 최종화** (0–10 / 10–25 / 25–50 / 50–60초, 타이밍 표시) | `docs/script_1min_briefing.md` | GPT 나레이션 초안, Gemini #7 | "GPT 나레이션 초안을 우리 수치(1,800→10,030, 46%, bounce_rate, 0.79→0.40)에 맞게 수정하고, 구간별 초 단위 표시. 읽었을 때 55~60초 나오게." |
| 2 | **영상용 슬라이드/캡처 순서** (보여줄 그래프 3장 고정) | `docs/video_storyboard.md` | GPT "그래프 3장: 파이프라인 → 누수 전후 → SHAP 전후" | "0–10초: 어떤 화면, 10–25초: 코드/파이프라인, 25–50초: 그래프 3장 순서와 각 1문장, 50–60초: 결론 화면. 스토리보드 표로 정리." |
| 3 | **(선택) Streamlit 예산 슬라이더 미니 대시** (예산 조절 → 예측 ROAS/차트) | Streamlit 앱 1페이지 + 배포 링크 | Gemini #6 | "기존 predictor로 플랫폼·산업 선택 + 예산 슬라이더 넣고, 예측 ROAS와 간단 차트 보여주는 streamlit 페이지 하나. 기존 streamlit_app과 별도 탭 또는 별 파일." |

**완료 체크**: `script_1min_briefing.md`, `video_storyboard.md`, (선택) 대시 URL.

---

## D-1: 녹화 + 최종 점검

**목표**: 1분 영상 녹화, Notion/기획서/영상 최종 수정.

| 순서 | 작업 | 산출물 | 참고 |
|------|------|--------|------|
| 1 | **PC 화면 녹화** (Notion 스크롤 + 스크립트 구간별 설명, 55~60초) | `deliverables/briefing_1min.mp4` | 제출 시나리오 0–10 / 10–25 / 25–50 / 50–60초 |
| 2 | **Notion 최종** (이미지 로딩, 권한, 링크 테스트) | Notion URL | "웹에 게시" 확인 |
| 3 | **1page PDF 최종** (오타, 수치, 30초 파악 테스트) | `deliverables/onepage_proposal.pdf` | 상단 3줄 + 5항목 + 체크리스트 |
| 4 | **제출 체크리스트** | - | Notion URL · 1page PDF · 1min MP4 · (선택) .py |

---

## D-day (화 17:00): 제출

- Notion 공개 페이지 URL 제출
- 1 Page 기획서 PDF 제출
- 1분 브리핑 영상 MP4 제출
- (선택) 분석 코드 압축 또는 repo 링크

---

## Cursor 활용 팁

1. **한 번에 한 블록씩**: "D-7 작업 1번 waterfall 차트 구현해줘"처럼 **산출물 파일명·참고 리포트 번호**까지 적어 주면 빠름.
2. **이미 있는 코드 재사용**: `archive/ablation_studies/`, `ads_predictor_v2.py`, `advanced_statistical_analysis.py`를 참조하라고 명시하면 일관된 데이터 경로·모델 사용.
3. **문서 생성**: "AgentReport의 GPT 원페이지 예시 A를 우리 프로젝트 수치로 바꿔서 docs/onepage_copies.md에 넣어줘"처럼 **출처 + 출력 경로** 지정.
4. **Notion**: Cursor로 마크다운 초안 생성 → Notion에 붙여넣기 후 이미지 업로드.

---

## 우선순위 요약 (리포트 통합)

| 우선순위 | 작업 | 평가 기여 | 난이도 |
|----------|------|-----------|--------|
| 1 | R² Waterfall 차트 | 영향력, 실행력 | 하 |
| 2 | 합성 데이터 3대 지표 + 표 | 실행력 | 중 |
| 3 | R² 0.40 방어 문단 | 상상력, 영향력 | 하 |
| 4 | SHAP Force 대조 + 가드레일 | 실행력 | 상 |
| 5 | 예산 재할당 시뮬레이션 + 그림 | 영향력, 상상력 | 중 |
| 6 | 세그먼트 신뢰도 맵 | 실행력 | 중 |
| 7 | 1분 스크립트·스토리보드 | 상상력 | 하 |
| (선택) | Streamlit 미니 대시 | 상상력, 실행력 | 상 |

이 파이프라인대로 진행하면 **기능 나열이 아닌 문제 해결 과정·결과**와 **30초 핵심 전달**을 동시에 잡을 수 있습니다.
