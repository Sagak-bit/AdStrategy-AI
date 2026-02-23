# Day 1 (D-7) Cursor 프롬프트 — 실행 가이드

**대상**: 새로운 Agent(다른 채팅/에이전트)에게 복사해 넣을 프롬프트.  
**목표**: R² Waterfall 차트, 누수 위험 점수보드, 1page 기획서 문장 확정.  
**완료 기준**: `figures/waterfall_r2_leakage.png`, `figures/leakage_risk_dashboard.png`, `docs/onepage_copies.md` 5개 항목.

---

## 모드 선택 가이드

| 작업 | 권장 모드 | 이유 |
|------|-----------|------|
| **1. R² Waterfall** | **Agent** | 코드 작성 + matplotlib/plotly 실행이 한 번에 필요. Plan은 단계 나누기 좋지만, “한 번에 스크립트 하나 생성”이면 Agent가 빠름. |
| **2. 누수 점수보드** | **Agent** | 데이터 로드·회귀·시각화가 연결된 단일 플로우. Agent로 “스크립트 생성 후 실행”까지 요청. |
| **3. 1page 문장** | **Agent** (또는 Plan) | 텍스트 산출물만 필요. Agent로 “참고 문서 읽고 우리 수치로 수정해 md 작성” 한 번에 요청 가능. Plan 쓰면 “문단별 초안 → 통합”으로 나눌 수 있음. |
| **비판 검토** | **Agent** (별 채팅) | “산출물만 검토해라”는 단일 역할이므로 Agent가 적합. Plan은 불필요. |

**전략 요약**: 세 작업 모두 **Agent 모드**로 진행하고, 세 개 산출물이 나온 뒤 **비판용 Agent**를 **별도 채팅**에서 한 번 돌린다.

---

## 사전 컨텍스트 (필요 시 프롬프트 앞에 붙여 넣기)

아래 블록을 **각 작업 프롬프트 맨 위**에 붙이면, 새 Agent가 프로젝트를 빠르게 파악할 수 있다.

```
[프로젝트 컨텍스트]
- 프로젝트명: AdStrategy AI (광고 ROAS 예측 + 데이터 감사)
- 데이터: data/enriched_ads_final.csv (10,030행×42컬럼). 없으면 global_ads_performance_dataset.csv 사용.
- Ablation: archive/ablation_studies/ 에 V1~V5 스크립트 있음. Phase 3에서 bounce_rate, landing_page_load_time, creative_impact_factor 가 target leakage로 확인됨. Leakage 포함 시 R²≈0.79, 제거 시 R²≈0.40.
- 산출물 저장: figures/ (PNG), docs/ (마크다운). 한글 폰트: Windows Malgun Gothic, macOS AppleGothic.
- 루트: d:\dev\Challengers (또는 현재 워크스페이스 루트).
```

---

## 프롬프트 1: R² 폭포수(Waterfall) 차트

**모드**: Agent

**복사용 전문**:

```
[프로젝트 컨텍스트]
- 프로젝트명: AdStrategy AI (광고 ROAS 예측 + 데이터 감사)
- 데이터: data/enriched_ads_final.csv (10,030행×42컬럼). 없으면 global_ads_performance_dataset.csv 사용.
- Ablation: archive/ablation_studies/ 에 V1~V5 스크립트 있음. Phase 3에서 bounce_rate, landing_page_load_time, creative_impact_factor 가 target leakage로 확인됨. Leakage 포함 시 R²≈0.79, 제거 시 R²≈0.40.
- 산출물 저장: figures/ (PNG), docs/ (마크다운). 한글 폰트: Windows Malgun Gothic, macOS AppleGothic.
- 루트: d:\dev\Challengers (또는 현재 워크스페이스 루트).

다음 작업을 수행해줘.

1) R²가 "과대평가(0.79)"에서 "정직한 성능(0.40)"으로 바뀌는 과정을 보여주는 **폭포수(Waterfall) 차트**를 만들어줘.
   - 단계 예시: [시작] Full model (leakage 포함) R²=0.79 → [감소] bounce_rate 제거 → [감소] landing_page_load_time 제거 → [감소] creative_impact_factor 제거 → [끝] Clean model R²=0.40
   - archive/ablation_studies/ 의 ablation_study_v5_clean.py 또는 v2, v4 결과를 참고해서, 실제 수치가 있으면 그걸 쓰고 없으면 위 수치(0.79 → 0.40)로 단계별 델타를 계산해줘.
2) matplotlib 또는 plotly로 그려서 **figures/waterfall_r2_leakage.png** 에 저장해줘. (plotly 사용 시 pio.write_image 또는 fig.to_image로 PNG 저장)
3) 한글 라벨이 들어가면 한글 폰트가 깨지지 않게 설정해줘 (Windows: Malgun Gothic).
4) 제목은 "R² 하락 궤적: Leakage 제거에 따른 성능 교정" 같은 서사가 드러나게 해줘.
```

---

## 프롬프트 2: 누수 위험 점수보드

**모드**: Agent

**복사용 전문**:

```
[프로젝트 컨텍스트]
- 프로젝트명: AdStrategy AI (광고 ROAS 예측 + 데이터 감사)
- 데이터: data/enriched_ads_final.csv (10,030행×42컬럼). 없으면 global_ads_performance_dataset.csv 사용.
- Ablation: archive/ablation_studies/ 에 V1~V5 스크립트 있음. Phase 3에서 bounce_rate, landing_page_load_time, creative_impact_factor 가 target leakage로 확인됨. Leakage 포함 시 R²≈0.79, 제거 시 R²≈0.40.
- 산출물 저장: figures/ (PNG), docs/ (마크다운). 한글 폰트: Windows Malgun Gothic, macOS AppleGothic.
- 루트: d:\dev\Challengers (또는 현재 워크스페이스 루트).

다음 작업을 수행해줘.

1) **누수 위험 점수보드**: 예측 타깃을 ROAS로 두고, 데이터의 **숫자형 피처 각각**에 대해 "그 피처 하나만으로 간단한 회귀(예: Ridge 또는 LinearRegression)"를 학습시켜 R²를 구해줘.
2) 피처별 R²를 **막대 차트**(가로 막대 권장)로 그려서, 비정상적으로 높은 피처(bounce_rate, landing_page_load_time, creative_impact_factor 등)가 눈에 띄게 보이도록 해줘. 이들이 leakage 후보임을 시각적으로 보여주는 게 목적이야.
3) 데이터는 data/enriched_ads_final.csv 를 쓰고, ROAS와 결측이 많은 컬럼은 제외한 뒤 숫자형만 사용. 범주형은 제외하거나 OneHot 후 일부만 포함해도 됨.
4) 결과를 **figures/leakage_risk_dashboard.png** 로 저장하고, 한글 폰트 설정(Windows Malgun Gothic) 적용해줘.
5) (선택) 같은 결과를 표로 출력하는 코드나 CSV로 저장하는 코드를 추가해도 좋아.
```

---

## 프롬프트 3: 1page 기획서 문장 확정 (onepage_copies.md)

**모드**: Agent

**복사용 전문**:

```
[프로젝트 컨텍스트]
- 프로젝트명: AdStrategy AI (광고 ROAS 예측 + 데이터 감사)
- 데이터: data/enriched_ads_final.csv (10,030행×42컬럼). 없으면 global_ads_performance_dataset.csv 사용.
- Ablation: archive/ablation_studies/ 에 V1~V5 스크립트 있음. Phase 3에서 bounce_rate, landing_page_load_time, creative_impact_factor 가 target leakage로 확인됨. Leakage 포함 시 R²≈0.79, 제거 시 R²≈0.40.
- 산출물 저장: figures/ (PNG), docs/ (마크다운). 한글 폰트: Windows Malgun Gothic, macOS AppleGothic.
- 루트: d:\dev\Challengers (또는 현재 워크스페이스 루트).

다음 작업을 수행해줘.

1) AgentReport/gpt-deep-research-report.md 의 "원페이지 기획서 문장 예시" 섹션을 읽어줘. (1) 주제, (2) 가설, (3) 데이터 출처, (4) 분석 방법, (5) 핵심 인사이트 항목별 예시 A/B가 있음.
2) 그 예시 문장들을 **우리 프로젝트 실제 수치**에 맞게 수정해서, **docs/onepage_copies.md** 파일 하나로 정리해줘.
   - 반드시 반영할 수치: 원본 1,800건 → 10,030건×42피처, 합성 약 46%, Phase 3, bounce_rate·load_time·creative_impact_factor, R² 0.79→0.40, ANOVA·t-test·Chi²·SHAP·ablation V1~V5.
3) 파일 형식: 마크다운. 각 항목을 ## 1. 주제(What) / ## 2. 가설 / ## 3. 데이터 출처 / ## 4. 분석 방법 / ## 5. 핵심 인사이트 로 나누고, 그 아래에 **최종 채택 문장 1~2개**만 넣어줘. (예시 A/B 중 하나를 골라 수정하거나, 둘을 합쳐서 한 문장으로 만들어도 됨.)
4) 상단에 "1page 기획서용 확정 문구 (30초 파악용)" 같은 제목과, "상단 3줄: 문제 1줄 / 결과 1줄(0.79→0.40) / 배운 점 1줄" 요약 블록을 추가해줘.
```

---

## 비판용 Agent 전략 (Critique Agent)

**시점**: 위 세 작업(Waterfall PNG, Leakage 대시보드 PNG, onepage_copies.md)이 모두 생성된 **후**.

**방식**: **새 채팅**을 열고, 아래 프롬프트를 **그대로** 붙여 넣어서 **검토만** 시키는 에이전트로 활용한다. 이 에이전트는 코드를 수정하지 않고, “체크리스트·기준 충족 여부”와 “개선 제안 1~2줄”만 출력하게 한다.

**비판용 프롬프트 (복사용)**:

```
당신은 데이터 분석 트랙 제출물 품질을 검토하는 **비판 전용** 역할입니다. 코드 수정이나 파일 생성은 하지 말고, 아래 체크만 하고 결과를 요약해주세요.

[검토 대상]
- figures/waterfall_r2_leakage.png (R² 폭포수 차트)
- figures/leakage_risk_dashboard.png (누수 위험 점수보드)
- docs/onepage_copies.md (1page 기획서용 문장 5항목)

[검토 기준]
1. Waterfall: R² 0.79 → leakage 변수 제거 단계 → 0.40 이라는 서사가 한눈에 들어오는가? "성능 하락"이 "교정"으로 읽히는가?
2. Leakage 대시보드: bounce_rate, landing_page_load_time, creative_impact_factor 가 다른 피처보다 비정상적으로 높게(또는 눈에 띄게) 나오는가?
3. onepage_copies.md: (1)~(5) 항목이 모두 채워져 있는가? 수치(1,800, 10,030, 42, 46%, 0.79, 0.40, Phase 3, ablation V1~V5)가 실제 프로젝트와 일치하는가? "30초 만에 핵심 파악"이 가능한 문장인가?

[출력 형식]
- 각 항목별로: 충족 / 미충족 (이유 1줄)
- 전체: 수정이 필요하면 구체적 개선 제안 1~2줄만 제시. 수정 불필요하면 "제출 준비 완료"로 결론.
```

---

## 실행 순서 요약

1. **Agent 모드**로 새 채팅 열기.
2. **프롬프트 1** 붙여 넣기 → `figures/waterfall_r2_leakage.png` 확인.
3. 같은 채팅 또는 새 메시지에 **프롬프트 2** 붙여 넣기 → `figures/leakage_risk_dashboard.png` 확인.
4. **프롬프트 3** 붙여 넣기 → `docs/onepage_copies.md` 확인.
5. **새 채팅** 열고 **비판용 프롬프트** 붙여 넣기 → 충족/미충족·개선 제안 확인 후, 필요 시 1~3 중 해당 작업만 다시 요청.

---

## Day 1 완료 체크리스트

- [ ] `figures/waterfall_r2_leakage.png` 존재, R² 0.79→0.40 서사 명확
- [ ] `figures/leakage_risk_dashboard.png` 존재, leakage 후보 피처 시각적 강조
- [ ] `docs/onepage_copies.md` 존재, 5개 항목(주제·가설·데이터·방법·인사이트) + 상단 3줄 요약
- [ ] 비판 Agent 실행 후 미충족 항목 보완 여부 결정
