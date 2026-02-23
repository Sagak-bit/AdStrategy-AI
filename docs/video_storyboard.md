# 1분 브리핑 영상 — 스토리보드

## 그래프 3장 + 전환 화면 구성

| 시간 | 화면 | 파일/소스 | 나레이션 핵심 | 전환 효과 |
|------|------|-----------|-------------|-----------|
| 0:00–0:03 | **타이틀 카드**: "AdStrategy AI — 데이터 감사로 찾은 정직한 성능" | 텍스트 슬라이드 | (무음 또는 배경음) | 페이드 인 |
| 0:03–0:10 | **R² = 0.79** 큰 숫자 + 물음표 | 텍스트 애니메이션 | "R² 0.79 — 너무 좋은 숫자" | 줌 인 → 물음표 |
| 0:10–0:18 | **파이프라인 다이어그램** (Phase 1→2→3→4, Phase 3 빨간 강조) | 직접 제작 또는 Notion 캡처 | "4단계 파이프라인에서 Phase 3이 문제" | 좌→우 슬라이드 |
| 0:18–0:25 | **코드 줌인**: `bounce_rate = 65 − ROAS × 2` | `creative_data_enricher.py` 248행 캡처 | "정답이 피처에 새어 들어간 것" | 코드 하이라이트 |
| 0:25–0:33 | **그래프 1**: `waterfall_r2_leakage.png` | figures/ | "0.79 → 0.35로 절반 이하" | 컷 전환 |
| 0:33–0:41 | **그래프 2**: `shap_force_leakage_vs_clean.png` | figures/ | "bounce_rate 50% → 통제 가능 변수로 전환" | 컷 전환 |
| 0:41–0:50 | **그래프 3**: `budget_reallocation_impact.png` | figures/ | "상위 20% 집중 시 ROAS +170%" | 컷 전환 |
| 0:50–0:57 | **핵심 문장 카드**: "높은 R²는 좋은 모델의 증거가 아니라 감사의 출발점이다" | 텍스트 슬라이드 | 결론 나레이션 | 페이드 인 |
| 0:57–1:00 | **엔딩 카드**: 프로젝트명 + 감사 인사 | 텍스트 슬라이드 | "감사합니다" | 페이드 아웃 |

---

## 녹화 체크리스트

- [ ] 화면 녹화 도구 준비 (OBS / Loom / PPT 녹화)
- [ ] 나레이션 연습 3회 (55~60초 이내)
- [ ] 그래프 3장 고해상도 준비 (dpi=200, figures/ 폴더)
- [ ] 배경음악 (선택: 조용한 테크 BGM, 볼륨 10~15%)
- [ ] 자막 (선택: 한국어 + 핵심 수치 강조)
- [ ] 최종 MP4 확인: 1분 이내, 해상도 1080p 이상

---

## 그래프 ↔ 섹션 매핑

| 그래프 파일 | Notion 섹션 | 영상 등장 시간 |
|------------|-------------|--------------|
| `waterfall_r2_leakage.png` | 5-2. Ablation | 0:25–0:33 |
| `shap_force_leakage_vs_clean.png` | 5-3. SHAP 대조 | 0:33–0:41 |
| `budget_reallocation_impact.png` | 7. 전략 변환 | 0:41–0:50 |
| `leakage_risk_dashboard.png` | 5-1. 점수보드 | Notion 전용 |
| `ablation_guardrail.png` | 6. 가드레일 | Notion 전용 |
| `segment_confidence_heatmap.png` | 7. 세그먼트 | Notion 전용 |
| `synthetic_fidelity_heatmap.png` | 6. 합성 품질 | Notion 전용 |
| `platform_budget_response_curves.png` | 7. 반응 곡선 | Notion 전용 |
