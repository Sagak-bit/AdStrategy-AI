# -*- coding: utf-8 -*-
"""
AdStrategy AI -- Streamlit 웹 앱
광고주와 대화하며 데이터 기반 광고 전략을 설계하는 AI 에이전트 UI.

실행: streamlit run streamlit_app.py
"""

import os
import sys

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import MAX_INPUT_LENGTH, PROJECT_ROOT, FIGURES_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from ad_agent import AdStrategyAgent

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="AdStrategy AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 비밀번호 표시 버튼 숨기기 (API 키 보호)
# ============================================================================
st.markdown("""
<style>
    [data-testid="stSidebar"] button[kind="icon"][aria-label*="password"],
    [data-testid="stSidebar"] button[aria-label="Show password text"],
    [data-testid="stSidebar"] button[aria-label="Hide password text"] {
        display: none !important;
    }
    input[type="password"]::-ms-reveal,
    input[type="password"]::-ms-clear {
        display: none !important;
    }
    /* 전체 앱 브랜딩 */
    .stMetric label { font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    h1 { color: #1a1a2e !important; }
    h2 { color: #16213e !important; }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        color: #0f3460;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 세션 초기화
# ============================================================================
_DEFAULTS = {
    "agent": None,
    "chat_history": [],
    "tool_results": [],
    "api_key_set": False,
    "api_key_from_env": False,
    "current_model": "gpt-4o",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================================
# 사이드바
# ============================================================================
with st.sidebar:
    st.title("AdStrategy AI")
    st.caption("데이터 기반 광고 전략 설계 에이전트")

    st.divider()

    # --- API 키 입력 ---
    st.subheader("OpenAI API Key")

    # 1순위: Streamlit Cloud secrets, 2순위: .env, 3순위: 직접 입력
    server_key = ""
    key_source = ""
    try:
        server_key = st.secrets.get("OPENAI_API_KEY", "")
        if server_key:
            key_source = "Streamlit Secrets"
    except Exception:
        pass

    if not server_key:
        server_key = os.environ.get("OPENAI_API_KEY", "")
        if server_key:
            key_source = ".env"

    if server_key and not st.session_state.api_key_set:
        st.session_state.api_key = server_key
        st.session_state.api_key_set = True
        st.session_state.api_key_from_env = True
        st.session_state.api_key_source = key_source

    # 서버에서 로드된 경우 키를 노출하지 않음
    if st.session_state.get("api_key_from_env"):
        masked = "•" * 24
        st.text_input("API Key", value=masked, disabled=True,
                      help=f"{st.session_state.get('api_key_source', '서버')}에서 로드됨")
        st.caption(f"`{st.session_state.get('api_key_source', '서버')}`에서 키를 불러왔습니다.")
    else:
        api_key = st.text_input(
            "API Key",
            value="",
            type="password",
            placeholder="sk-proj-...",
            help="platform.openai.com 에서 발급",
        )
        if api_key:
            st.session_state.api_key = api_key
            st.session_state.api_key_set = True

    # 모델 선택
    model_choice = st.selectbox(
        "LLM 모델",
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="gpt-4o: 고품질, gpt-4o-mini: 빠르고 저렴",
    )

    # 모델 변경 감지 → 에이전트 재초기화
    if st.session_state.get("current_model") != model_choice:
        st.session_state.current_model = model_choice
        if st.session_state.agent is not None:
            st.session_state.agent = None

    st.divider()

    # --- 에이전트 초기화 ---
    if st.session_state.api_key_set and st.session_state.get("api_key"):
        if st.session_state.agent is None:
            with st.spinner("모델 로딩 중..."):
                try:
                    st.session_state.agent = AdStrategyAgent(
                        openai_api_key=st.session_state.api_key,
                        model=model_choice,
                    )
                    st.success("에이전트 준비 완료")
                except Exception as e:
                    st.error(f"초기화 실패: {e}")

        # --- 모델 성능 표시 ---
        if st.session_state.agent:
            st.subheader("ML 모델 성능")
            model_info = st.session_state.agent.get_model_info()
            for target, info in model_info.items():
                col1, col2 = st.columns(2)
                col1.metric(
                    f"{target}",
                    f"R2: {info['r2_score']:.3f}",
                    help=f"Model: {info['model_name']}, MAE: {info['mae']:.2f}"
                )

    st.divider()

    # --- 빠른 시작 ---
    st.subheader("Quick Start")
    st.caption("아래 버튼으로 예시 질문을 시작할 수 있습니다")

    quick_options = [
        "핀테크 앱 광고를 미국에서 시작하려고 해요",
        "이커머스 쇼핑몰의 Q4 광고 전략을 짜주세요",
        "에드테크 스타트업인데, 인도 시장 진출을 고려 중이에요",
        "헬스케어 SaaS 제품, 영국/독일 동시 론칭 광고",
    ]

    for opt in quick_options:
        if st.button(opt, use_container_width=True):
            st.session_state.quick_start_msg = opt

    st.divider()

    # --- 대화 초기화 ---
    if st.button("대화 초기화", type="secondary", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.tool_results = []
        if st.session_state.agent:
            st.session_state.agent.reset_conversation()
        st.rerun()

    st.divider()
    st.caption("Powered by GPT-4o + Custom ML Models")


# ============================================================================
# 차트 렌더링 함수
# ============================================================================

_HONEST_R2_DISCLAIMER = (
    "이 예측은 leakage 제거 후 honest R² ≈ 0.35 모델 기반입니다. "
    "개별 수치보다 **플랫폼·캠페인 간 상대 비교**에 활용하세요."
)


def render_prediction_chart(result_data):
    """단일 예측 결과 차트"""
    predictions = result_data.get("predictions", {})
    inp = result_data.get("input", {})

    # ROAS, CPC, CPA 바 차트
    metrics = []
    values = []
    for m in ["ROAS", "CPC", "CPA"]:
        if m in predictions:
            p = predictions[m]
            val = p.get("predicted", 0) if isinstance(p, dict) else p
            metrics.append(m)
            values.append(val)

    if metrics:
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition="auto",
                marker_color=["#2ecc71", "#3498db", "#e74c3c"],
            )
        ])
        fig.update_layout(
            title=f"예측 결과: {inp.get('platform', '')} / {inp.get('industry', '')} / {inp.get('country', '')}",
            yaxis_title="값",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"⚠️ {_HONEST_R2_DISCLAIMER}")


def render_comparison_chart(comparison_data):
    """시나리오 비교 차트"""
    comparison = comparison_data.get("comparison", [])
    if not comparison:
        return

    # 데이터프레임 변환
    df = pd.DataFrame(comparison)

    if "ROAS" in df.columns:
        roas_valid = df[df["ROAS"] != "N/A"].copy()
        if len(roas_valid) > 0:
            roas_valid["ROAS"] = pd.to_numeric(roas_valid["ROAS"], errors="coerce")
            roas_valid = roas_valid.dropna(subset=["ROAS"])

            fig = go.Figure(data=[
                go.Bar(
                    x=roas_valid["scenario"],
                    y=roas_valid["ROAS"],
                    text=[f"{v:.2f}" for v in roas_valid["ROAS"]],
                    textposition="auto",
                    marker_color=px.colors.qualitative.Set2[:len(roas_valid)],
                )
            ])
            fig.update_layout(
                title="시나리오별 ROAS 비교",
                yaxis_title="ROAS",
                height=350,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption(f"⚠️ {_HONEST_R2_DISCLAIMER}")

    display_cols = [c for c in ["scenario", "ROAS", "CPC", "CPA", "estimated_revenue", "estimated_ROI_percent", "ROAS_confidence"] if c in df.columns]
    if display_cols:
        st.dataframe(
            df[display_cols].rename(columns={
                "scenario": "시나리오",
                "ROAS": "ROAS",
                "CPC": "CPC ($)",
                "CPA": "CPA ($)",
                "estimated_revenue": "예상 매출 ($)",
                "estimated_ROI_percent": "예상 ROI (%)",
                "ROAS_confidence": "신뢰도",
            }),
            use_container_width=True,
            hide_index=True,
        )


def render_trend_chart(trend_data):
    """트렌드 차트"""
    monthly = trend_data.get("monthly_trend_index") or trend_data.get("monthly_trend", {})
    if not monthly:
        return

    try:
        months = sorted(monthly.keys(), key=lambda x: int(x))
    except (ValueError, TypeError):
        months = sorted(monthly.keys())
    values = [monthly[m] for m in months]
    month_labels = [f"{int(m)}월" for m in months]

    fig = go.Figure(data=[
        go.Scatter(
            x=month_labels,
            y=values,
            mode="lines+markers",
            name=trend_data.get("industry", ""),
            line=dict(width=3),
            marker=dict(size=8),
        )
    ])

    peak = trend_data.get("peak_month")
    if peak:
        peak_label = f"{peak}월"
        fig.add_vline(x=peak_label, line_dash="dash", line_color="red")
        if peak_label in month_labels:
            peak_idx = month_labels.index(peak_label)
            fig.add_annotation(
                x=peak_label, y=values[peak_idx],
                text=f"Peak: {peak_label}",
                showarrow=True, arrowhead=2, yshift=15,
            )

    fig.update_layout(
        title=f"{trend_data.get('industry', '')} 산업 월별 관심도 트렌드 ({trend_data.get('country', '전체')})",
        yaxis_title="트렌드 지수",
        height=350,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tool_results(tool_results):
    """Tool 결과에 따라 적절한 차트 렌더링"""
    for tr in tool_results:
        tool_name = tr["tool"]
        result = tr["result"]

        if "error" in result:
            continue

        if tool_name == "predict_ad_performance":
            render_prediction_chart(result)

        elif tool_name == "compare_scenarios":
            render_comparison_chart(result)

        elif tool_name == "get_industry_trends":
            render_trend_chart(result)

        elif tool_name == "get_historical_benchmarks":
            # 벤치마크는 플랫폼 비교가 있으면 차트
            plat_comp = result.get("platform_comparison")
            if plat_comp:
                platforms = list(plat_comp.keys())
                roas_vals = [plat_comp[p].get("avg_ROAS", 0) for p in platforms]

                fig = go.Figure(data=[
                    go.Bar(x=platforms, y=roas_vals,
                           text=[f"{v:.2f}" for v in roas_vals],
                           textposition="auto",
                           marker_color=px.colors.qualitative.Pastel[:len(platforms)])
                ])
                fig.update_layout(
                    title=f"{result.get('filters', {}).get('industry', '')} 산업 플랫폼별 평균 ROAS",
                    yaxis_title="평균 ROAS",
                    height=350,
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Leakage 감사 대시보드 (Tab 2)
# ============================================================================

def render_leakage_audit_tab():
    """Leakage 감사 대시보드 탭 렌더링"""
    st.header("Leakage 감사 대시보드")
    st.caption("R² 0.79 → 0.40: 높은 성능의 함정을 추적한 기록")

    st.markdown("""
> **핵심 발견**: 모델의 R² 0.79 중 약 49%가 `bounce_rate = 65 - ROAS × 2`라는
> 데이터 생성 공식의 **역함수 복원**에 의존하고 있었습니다.
> 5단계 자체 감사를 통해 이를 식별하고, honest R² 0.40으로 교정했습니다.
""")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Leakage 포함 R²", "0.79", delta=None)
    col2.metric("Honest R² (교정 후)", "0.40", delta="-0.39", delta_color="off")
    col3.metric("Leakage 변수", "3개", help="bounce_rate, landing_page_load_time, creative_impact_factor")
    col4.metric("Ablation 단계", "V1→V5", help="Leave-One-Out → Shapley → Robustness → Individual → Data Audit")

    st.divider()

    # Waterfall 차트
    st.subheader("1. R² 하락 궤적 (Waterfall)")
    waterfall_path = os.path.join(FIGURES_DIR, "waterfall_r2_leakage.png")
    if os.path.exists(waterfall_path):
        st.image(waterfall_path, use_container_width=True)
    else:
        st.warning("waterfall_r2_leakage.png 파일이 없습니다. `python waterfall_r2_leakage.py`를 실행하세요.")
    st.caption("Full model(R² 0.79)에서 leakage 변수를 하나씩 제거하며 R²가 0.40으로 교정되는 과정")

    st.divider()

    # Leakage 리스크 대시보드
    st.subheader("2. 피처별 Leakage 위험도")
    leakage_path = os.path.join(FIGURES_DIR, "leakage_risk_dashboard.png")
    if os.path.exists(leakage_path):
        st.image(leakage_path, use_container_width=True)
    else:
        st.warning("leakage_risk_dashboard.png 파일이 없습니다.")

    risk_csv = os.path.join(FIGURES_DIR, "leakage_risk_scores.csv")
    if os.path.exists(risk_csv):
        risk_df = pd.read_csv(risk_csv)
        with st.expander("피처별 단독 예측 R² (상세 데이터)"):
            st.dataframe(risk_df.sort_values("r2_mean", ascending=False),
                         use_container_width=True, hide_index=True)

    st.divider()

    # SHAP 대조
    st.subheader("3. SHAP 기여도 대조 (Leakage 포함 vs 제거)")
    shap_path = os.path.join(FIGURES_DIR, "shap_force_leakage_vs_clean.png")
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=True)
    st.caption("Leakage 포함 모델에서 bounce_rate가 압도적 기여 → 제거 후 competition_index 등이 주도")

    # Guardrail
    st.subheader("4. Ablation 가드레일")
    guard_path = os.path.join(FIGURES_DIR, "ablation_guardrail.png")
    if os.path.exists(guard_path):
        st.image(guard_path, use_container_width=True)

    st.divider()

    # Ablation V1-V5 인터랙티브 차트
    st.subheader("5. Ablation Study 진화 (V1 → V5)")
    ablation_data = {
        "단계": ["V1 Leave-One-Out", "V2 Shapley 분해", "V3 Robustness",
                 "V4 개별 피처 분해", "V5 Data Audit (Clean)"],
        "R²": [0.79, 0.79, 0.78, 0.65, 0.40],
        "핵심 발견": [
            "Creative 그룹 26.5% 기여 (의심 시작)",
            "Creative 86% 기여 (Shapley φ=0.397)",
            "audience_size 이동해도 Creative 지배 유지",
            "bounce_rate 단독 40.8%, load_time 18.3%",
            "ROAS 역함수 발견 → Leakage 제거 후 φ 음수 반전",
        ],
    }
    abl_df = pd.DataFrame(ablation_data)

    fig_abl = go.Figure()
    fig_abl.add_trace(go.Scatter(
        x=abl_df["단계"], y=abl_df["R²"],
        mode="lines+markers+text",
        text=[f"{r:.2f}" for r in abl_df["R²"]],
        textposition="top center",
        line=dict(width=3, color="#E74C3C"),
        marker=dict(size=12),
    ))
    fig_abl.update_layout(
        title="Ablation V1→V5: R² 변화 궤적",
        yaxis_title="R² Score", yaxis_range=[0, 1],
        height=350, template="plotly_white",
    )
    st.plotly_chart(fig_abl, use_container_width=True)
    st.dataframe(abl_df, use_container_width=True, hide_index=True)

    st.divider()

    # R² 0.40 방어 논리
    st.subheader("6. 왜 R² 0.40이 충분한가")
    defense_path = os.path.join(PROJECT_ROOT, "docs", "r2_philosophy_defense.md")
    if os.path.exists(defense_path):
        with open(defense_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.markdown("""
- 마케팅/행동 데이터에서 R² 0.20~0.50은 실무적으로 타당한 범위 (Leeflang et al., 2009)
- Meta Robyn MMM 벤치마크: R² 0.50~0.70
- Honest R² 0.40은 예산 배분의 **방향성 가이드**로 충분히 활용 가능
""")


# ============================================================================
# 예산 시뮬레이터 (Tab 3)
# ============================================================================

def render_budget_simulator_tab():
    """예산-ROAS 시뮬레이터 탭 렌더링"""
    from scripts.platform_policy_params import (
        PLATFORM_POLICY, BASE_ROAS_BY_PLATFORM_INDUSTRY, compute_roas_multiplier,
    )

    st.header("예산-ROAS 시뮬레이터")
    st.caption("플랫폼별 정책 반영 예산 시뮬레이션 (실무 가정 기반)")

    col1, col2, col3 = st.columns(3)
    with col1:
        platform = st.selectbox("플랫폼", ["Google Ads", "Meta Ads", "TikTok Ads"], key="sim_platform")
    with col2:
        industry = st.selectbox("산업", ["Fintech", "EdTech", "Healthcare", "SaaS", "E-commerce"], key="sim_industry")
    with col3:
        country = st.selectbox("국가",
                               ["USA", "UK", "Germany", "Canada", "India", "UAE", "Australia", "Korea"],
                               key="sim_country")

    budget = st.slider("월 광고 예산 (USD)", min_value=100, max_value=20000, value=5000, step=100, key="sim_budget")

    st.divider()

    policy = PLATFORM_POLICY[platform]
    base_roas = BASE_ROAS_BY_PLATFORM_INDUSTRY.get(platform, {}).get(industry, 4.0)

    budgets = np.arange(100, 20001, 100)
    roas_curve = []
    for b in budgets:
        mult = compute_roas_multiplier(float(b), policy)
        roas_curve.append(max(0, base_roas * mult))

    current_idx = (budget - 100) // 100
    current_roas = roas_curve[min(current_idx, len(roas_curve) - 1)]
    est_revenue = budget * current_roas
    est_roi = ((est_revenue - budget) / budget * 100) if budget > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예측 ROAS", f"{current_roas:.2f}")
    m2.metric("예상 매출", f"${est_revenue:,.0f}")
    m3.metric("예상 ROI", f"{est_roi:.1f}%")
    m4.metric("최소 유효 예산", f"${policy['min_effective_budget']:,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(budgets), y=roas_curve, mode="lines",
        name=f"{platform} 반응 곡선",
        line=dict(width=3, color="#3498db"),
    ))
    fig.add_trace(go.Scatter(
        x=[budget], y=[current_roas], mode="markers",
        name=f"현재: ${budget:,} → ROAS {current_roas:.2f}",
        marker=dict(size=15, color="#E74C3C", symbol="star"),
    ))
    fig.add_vline(x=policy["min_effective_budget"], line_dash="dot",
                  annotation_text=f"최소 유효: ${policy['min_effective_budget']:,}")
    fig.update_layout(
        title=f"{platform} x {industry} — 예산-ROAS 반응 곡선",
        xaxis_title="월 광고 예산 (USD)",
        yaxis_title="기대 ROAS",
        height=450, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**{platform} 정책**: {policy['note']}")

    st.divider()

    # 플랫폼 비교
    st.subheader("플랫폼 간 반응 곡선 비교")
    fig_cmp = go.Figure()
    colors_map = {"Google Ads": "#4A90D9", "Meta Ads": "#F39C12", "TikTok Ads": "#E74C3C"}
    for plat, pol in PLATFORM_POLICY.items():
        br = BASE_ROAS_BY_PLATFORM_INDUSTRY.get(plat, {}).get(industry, 4.0)
        curve = [max(0, br * compute_roas_multiplier(float(b), pol)) for b in budgets]
        fig_cmp.add_trace(go.Scatter(
            x=list(budgets), y=curve, mode="lines",
            name=plat, line=dict(width=2.5, color=colors_map.get(plat, "#999")),
        ))
    fig_cmp.update_layout(
        title=f"{industry} 산업 — 플랫폼별 예산-ROAS 비교",
        xaxis_title="월 광고 예산 (USD)", yaxis_title="기대 ROAS",
        height=400, template="plotly_white",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # 세그먼트 신뢰도 히트맵
    seg_path = os.path.join(FIGURES_DIR, "segment_confidence_heatmap.png")
    if os.path.exists(seg_path):
        st.subheader("세그먼트 신뢰도 맵")
        st.image(seg_path, use_container_width=True)
        st.caption("MAE가 낮을수록 해당 플랫폼 x 산업 조합의 예측 신뢰도가 높음")

    # --- 원클릭 추천 카드 ---
    st.divider()
    st.subheader("이번 달 추천 예산 배분")

    all_roas = {}
    for plat, pol in PLATFORM_POLICY.items():
        br = BASE_ROAS_BY_PLATFORM_INDUSTRY.get(plat, {}).get(industry, 4.0)
        mult = compute_roas_multiplier(float(budget), pol)
        all_roas[plat] = max(0, br * mult)

    sorted_plats = sorted(all_roas.items(), key=lambda x: x[1], reverse=True)
    best_plat, best_roas_val = sorted_plats[0]
    second_plat, second_roas_val = sorted_plats[1]

    primary_budget = int(budget * 0.7)
    secondary_budget = budget - primary_budget

    rec_cols = st.columns(2)
    with rec_cols[0]:
        st.success(f"""
**주력 채널: {best_plat}** — 예산 ${primary_budget:,} (70%)
- 기대 ROAS: {best_roas_val:.2f}
- 예상 매출: ${primary_budget * best_roas_val:,.0f}
""")
    with rec_cols[1]:
        st.info(f"""
**탐색 채널: {second_plat}** — 예산 ${secondary_budget:,} (30%)
- 기대 ROAS: {second_roas_val:.2f}
- 예상 매출: ${secondary_budget * second_roas_val:,.0f}
""")

    total_expected = primary_budget * best_roas_val + secondary_budget * second_roas_val
    st.metric("합산 기대 매출", f"${total_expected:,.0f}",
              delta=f"ROI {((total_expected - budget) / budget * 100):.0f}%" if budget > 0 else "")

    # --- 보고서 다운로드 ---
    st.divider()
    report_md = f"""# AdStrategy AI — 예산 배분 추천 보고서

**생성일**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## 입력 조건
- 산업: {industry}
- 타겟 국가: {country}
- 월 예산: ${budget:,}
- 선택 플랫폼: {platform}

## 추천 예산 배분

| 채널 | 배분 비율 | 예산 | 기대 ROAS | 예상 매출 |
|------|----------|------|----------|----------|
| {best_plat} (주력) | 70% | ${primary_budget:,} | {best_roas_val:.2f} | ${primary_budget * best_roas_val:,.0f} |
| {second_plat} (탐색) | 30% | ${secondary_budget:,} | {second_roas_val:.2f} | ${secondary_budget * second_roas_val:,.0f} |
| **합계** | 100% | **${budget:,}** | — | **${total_expected:,.0f}** |

## 예상 ROI: {((total_expected - budget) / budget * 100):.1f}%

## A/B 테스트 제안
첫 2주간 두 플랫폼에 50:50 배분 후, ROAS 비교하여 3주차부터 승리 플랫폼에 80% 집중.

## 리스크
- 이 예측은 honest R² ≈ 0.35 모델 기반이며, 개별 수치보다 플랫폼 간 상대 비교에 활용하세요.
- 합성 데이터 46% 포함 — 실제 캠페인 결과와 차이가 있을 수 있습니다.

---
*AdStrategy AI — 데이터 기반 광고 전략 설계*
"""
    st.download_button(
        "추천 보고서 다운로드 (Markdown)",
        data=report_md,
        file_name=f"adstrategy_report_{industry}_{budget}.md",
        mime="text/markdown",
    )

    st.caption("⚠️ 이 시뮬레이션은 실무 가정 기반이며 실측 데이터가 아닙니다.")


# ============================================================================
# 대시보드 (Tab 0) — 첫 화면에서 10초 안에 핵심 전달
# ============================================================================

def render_dashboard_tab():
    """프로젝트 핵심 요약 대시보드"""

    # --- 엔드투엔드 스토리라인 ---
    st.header("문제 → 발견 → 해결 → 임팩트")

    story_cols = st.columns(4)
    with story_cols[0]:
        st.markdown("### 1. 문제")
        st.error("R² 0.79 — 너무 좋은 성능")
        st.caption("마케팅 데이터에서 비현실적. 데이터 파이프라인 어딘가에 정답지가 새어 들어간 것 아닌가?")
    with story_cols[1]:
        st.markdown("### 2. 발견")
        st.warning("Target Leakage 3개 변수")
        st.caption("`bounce_rate = 65 - ROAS×2`\n\n정답(ROAS)의 역함수가 피처로 사용되고 있었음")
    with story_cols[2]:
        st.markdown("### 3. 해결")
        st.success("Honest R² = 0.35")
        st.caption("Leakage 3개 변수 제거 → 5-fold TimeSeriesSplit CV로 검증된 정직한 성능")
    with story_cols[3]:
        st.markdown("### 4. 임팩트")
        st.info("예산 최적화 ROAS +170%")
        st.caption("정직한 모델로도 상위 20% 캠페인에 예산 집중 시 ROAS 170% 개선 가능")

    st.divider()

    # --- 추천 액션 (마케터가 바로 쓸 수 있는) ---
    st.header("지금 바로 할 수 있는 3가지 액션")

    act_cols = st.columns(3)
    with act_cols[0]:
        st.markdown("""
**예산 재배분**

예측 ROAS 상위 20% 캠페인에 예산의 70%를 집중 배분하세요.
균등 배분 대비 **가중 ROAS가 +170% 개선**됩니다.

👉 **예산 시뮬레이터** 탭에서 직접 테스트
""")
    with act_cols[1]:
        st.markdown("""
**플랫폼 선택**

- **Google Ads**: 가장 안정적 (MAE 1.9~2.6)
- **Meta Ads**: 소예산 시 학습 구간 주의
- **TikTok Ads**: 높은 기대수익, 높은 변동성

👉 **AI 에이전트** 탭에서 맞춤 전략 상담
""")
    with act_cols[2]:
        st.markdown("""
**데이터 감사**

자체 데이터에서도 leakage가 있을 수 있습니다.
피처 단독 R²가 비정상적으로 높으면 의심하세요.

👉 **Leakage 감사** 탭에서 방법론 확인
""")

    st.divider()

    # --- 핵심 시각화 미리보기 ---
    st.header("핵심 결과 미리보기")

    prev_cols = st.columns(2)
    with prev_cols[0]:
        waterfall_path = os.path.join(FIGURES_DIR, "waterfall_r2_leakage.png")
        if os.path.exists(waterfall_path):
            st.image(waterfall_path, caption="R² 하락 궤적: Leakage 제거 과정", use_container_width=True)

    with prev_cols[1]:
        shap_path = os.path.join(FIGURES_DIR, "shap_force_leakage_vs_clean.png")
        if os.path.exists(shap_path):
            st.image(shap_path, caption="SHAP 대조: bounce_rate 압도 → 정직한 피처 분포", use_container_width=True)

    prev_cols2 = st.columns(2)
    with prev_cols2[0]:
        budget_path = os.path.join(FIGURES_DIR, "budget_reallocation_impact.png")
        if os.path.exists(budget_path):
            st.image(budget_path, caption="예산 재할당 효과: 상위 20% 집중 시 +170%", use_container_width=True)

    with prev_cols2[1]:
        seg_path = os.path.join(FIGURES_DIR, "segment_confidence_heatmap.png")
        if os.path.exists(seg_path):
            st.image(seg_path, caption="세그먼트 신뢰도: 플랫폼×산업별 예측 MAE", use_container_width=True)

    st.divider()

    # --- 데이터 투명성 공지 ---
    with st.expander("데이터 투명성 및 한계"):
        st.markdown("""
| 항목 | 내용 |
|------|------|
| 원본 데이터 | Kaggle 공개 광고 캠페인 1,800건 |
| 보강 후 | 10,030건 × 42피처 (4단계 파이프라인) |
| 합성 비율 | 약 46% — 볼륨 확보 목적, 분포 충실도에 한계 있음 |
| Honest R² | 0.35 (leakage 제거, 5-fold TimeSeriesSplit CV) |
| 모델 용도 | **개별 수치 예측이 아닌, 플랫폼·캠페인 간 상대 비교 가이드** |
| Leakage 변수 | bounce_rate, landing_page_load_time, creative_impact_factor |
""")
        st.caption("이 프로젝트의 핵심 산출물은 예측 모델이 아니라, **데이터 감사 프레임워크**입니다.")


# ============================================================================
# 내 데이터 분석 (Tab 4)
# ============================================================================

def render_my_data_tab():
    """사용자 CSV 업로드 → 자동 EDA 탭"""
    st.header("내 데이터 분석")
    st.caption("CSV 파일을 업로드하면 자동으로 기초 탐색 분석(EDA)을 수행합니다")

    uploaded = st.file_uploader("광고 데이터 CSV 업로드", type=["csv"], key="user_csv")
    if uploaded is None:
        st.info(
            "CSV 파일에 다음 컬럼이 포함되면 최적입니다:\n\n"
            "`platform`, `industry`, `country`, `ad_spend`, `ROAS`, `CPC`, `CPA`, `impressions`, `clicks`, `conversions`\n\n"
            "컬럼이 일부 없어도 가능한 분석은 자동 수행됩니다."
        )
        return

    try:
        user_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"CSV 파싱 실패: {e}")
        return

    st.success(f"데이터 로드 완료: **{len(user_df):,}행 × {len(user_df.columns)}열**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("행 수", f"{len(user_df):,}")
    col_b.metric("컬럼 수", f"{len(user_df.columns)}")
    null_pct = user_df.isnull().mean().mean() * 100
    col_c.metric("평균 결측률", f"{null_pct:.1f}%")

    st.divider()

    st.subheader("1. 데이터 미리보기")
    st.dataframe(user_df.head(20), use_container_width=True)

    st.subheader("2. 기초 통계")
    st.dataframe(user_df.describe().round(2), use_container_width=True)

    st.subheader("3. 결측치 현황")
    null_df = user_df.isnull().sum().reset_index()
    null_df.columns = ["컬럼", "결측 수"]
    null_df["결측률(%)"] = (null_df["결측 수"] / len(user_df) * 100).round(1)
    null_df = null_df[null_df["결측 수"] > 0].sort_values("결측 수", ascending=False)
    if len(null_df) > 0:
        st.dataframe(null_df, use_container_width=True, hide_index=True)
    else:
        st.success("결측치 없음")

    num_cols = user_df.select_dtypes("number").columns.tolist()
    cat_cols = user_df.select_dtypes(["object", "category"]).columns.tolist()

    if num_cols:
        st.subheader("4. 수치형 변수 분포")
        sel_num = st.selectbox("변수 선택", num_cols, key="eda_num_col")
        fig_hist = go.Figure(data=[go.Histogram(x=user_df[sel_num].dropna(), nbinsx=30,
                                                marker_color="#3498db")])
        fig_hist.update_layout(title=f"{sel_num} 분포", height=350, template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    if cat_cols:
        st.subheader("5. 범주형 변수 분포")
        sel_cat = st.selectbox("변수 선택", cat_cols, key="eda_cat_col")
        vc = user_df[sel_cat].value_counts().head(15)
        fig_bar = go.Figure(data=[go.Bar(x=vc.index.tolist(), y=vc.values,
                                         marker_color="#2ecc71")])
        fig_bar.update_layout(title=f"{sel_cat} 분포 (상위 15)", height=350,
                              template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    if "ROAS" in user_df.columns and "platform" in user_df.columns:
        st.subheader("6. 플랫폼별 ROAS 비교")
        plat_roas = user_df.groupby("platform")["ROAS"].agg(["mean", "median", "count"])
        plat_roas.columns = ["평균 ROAS", "중앙값 ROAS", "건수"]
        st.dataframe(plat_roas.round(2), use_container_width=True)

        fig_box = go.Figure()
        for plat in user_df["platform"].unique():
            fig_box.add_trace(go.Box(y=user_df[user_df["platform"] == plat]["ROAS"],
                                     name=plat))
        fig_box.update_layout(title="플랫폼별 ROAS 분포", height=400, template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True)

    if "ad_spend" in user_df.columns and "ROAS" in user_df.columns:
        st.subheader("7. 광고비 vs ROAS 산점도")
        fig_sc = go.Figure(data=[go.Scatter(
            x=user_df["ad_spend"], y=user_df["ROAS"], mode="markers",
            marker=dict(size=5, opacity=0.5, color="#e74c3c"),
        )])
        fig_sc.update_layout(xaxis_title="Ad Spend ($)", yaxis_title="ROAS",
                             height=400, template="plotly_white")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.caption("이 분석은 업로드된 데이터의 기초 탐색 결과이며, 모델 예측은 포함되지 않습니다.")


# ============================================================================
# 메인 영역 — Hero + 5탭 구조
# ============================================================================

st.title("AdStrategy AI")

# --- Hero Section: 10초 안에 전달하는 스토리 ---
hero_cols = st.columns([2, 3])
with hero_cols[0]:
    st.metric("R² Before (Leakage)", "0.79", delta="거짓 성능", delta_color="inverse")
    st.metric("R² After (Honest)", "0.35", delta="정직한 성능", delta_color="off")
    st.metric("예산 최적화 효과", "+170%", delta="ROAS 개선")
with hero_cols[1]:
    st.markdown("""
**AI가 자기 모델의 거짓말을 잡아낸 이야기**

광고 ROAS 예측 모델이 R² 0.79를 기록했습니다.
하지만 이 숫자는 **데이터 파이프라인의 실수**가 만든 환상이었습니다.

5단계 자체 감사로 leakage를 찾아내고, **정직한 R² 0.35**로 교정한 뒤에도
예산 재할당만으로 **ROAS +170% 개선**이 가능하다는 것을 증명했습니다.

> 아래 탭에서 감사 과정, 예산 시뮬레이션, AI 전략 상담을 직접 체험하세요.
""")

st.divider()

tab_overview, tab_agent, tab_leakage, tab_simulator, tab_mydata = st.tabs([
    "대시보드", "AI 에이전트", "Leakage 감사", "예산 시뮬레이터", "내 데이터 분석"
])

# --- Tab 0: 대시보드 (API 키 불필요) ---
with tab_overview:
    render_dashboard_tab()

# --- Tab 2: Leakage 감사 (API 키 불필요) ---
with tab_leakage:
    render_leakage_audit_tab()

# --- Tab 3: 예산 시뮬레이터 (API 키 불필요) ---
with tab_simulator:
    render_budget_simulator_tab()

# --- Tab 4: 내 데이터 분석 (API 키 불필요) ---
with tab_mydata:
    render_my_data_tab()

# --- Tab 1: AI 에이전트 (API 키 필요, 데모 모드 제공) ---
with tab_agent:
    if not st.session_state.api_key_set or not st.session_state.get("api_key"):
        st.info(
            "왼쪽 사이드바에 **OpenAI API Key**를 입력하면 실시간 AI 상담이 가능합니다.\n\n"
            "키가 없어도 아래 **데모 대화**에서 에이전트의 작동 방식을 확인할 수 있습니다."
        )

        st.divider()
        st.subheader("데모: E-commerce $5,000 예산 전략 상담")
        st.caption("실제 AI 에이전트가 생성한 대화 예시입니다")

        _demo_conversation = [
            {"role": "user", "content": "이커머스 쇼핑몰인데, 미국 시장에서 월 $5,000으로 광고 시작하려고 해요."},
            {"role": "assistant", "content": """안녕하세요! 이커머스 미국 시장 진출, 좋은 계획이시네요. 몇 가지 여쭤볼게요.

1. **어떤 제품**을 주로 판매하시나요? (패션, 전자기기, 생활용품 등)
2. **광고 목표**가 구매 전환인가요, 아니면 브랜드 인지도 확보가 먼저인가요?"""},
            {"role": "user", "content": "패션 액세서리고, 구매 전환이 목표입니다. 10월에 시작하려고요."},
            {"role": "assistant", "content": """좋습니다! Q4 시즌(10~12월)은 이커머스에 최적의 시기입니다. 3개 플랫폼을 비교해볼게요.

**플랫폼별 예측 결과** (E-commerce / USA / Shopping&Video / $5,000 / 10월):

| 플랫폼 | 예측 ROAS | CPC | 예상 매출 | 신뢰도 |
|--------|----------|-----|----------|--------|
| Google Ads (Shopping) | 4.2 | $1.85 | $21,000 | High (n=89) |
| Meta Ads (Display) | 3.8 | $1.42 | $19,000 | Medium (n=65) |
| TikTok Ads (Video) | 5.1 | $2.30 | $25,500 | Low (n=31) |

**추천 전략: 70:30 분할**
- **$3,500 (70%) → Google Ads Shopping**: ROAS가 가장 안정적이고, Q4 Shopping 시즌과 맞물려 전환율 최고
- **$1,500 (30%) → TikTok Ads Video**: 잠재적 ROAS가 높지만 변동성도 커서 탐색적 배분

⚠️ 이 예측은 honest R²≈0.35 모델 기반입니다. 개별 수치보다 플랫폼 간 상대 비교에 활용하세요.

**리스크 분석**:
- TikTok E-commerce의 MAE가 6.8로, 실제 ROAS가 예측 대비 크게 벗어날 수 있습니다
- Q4 경쟁 격화로 CPC가 평소 대비 20~40% 상승 가능

**A/B 테스트 제안**: 첫 2주간 50:50 배분 후, ROAS 비교하여 3주차부터 승리 플랫폼에 80% 집중하세요."""},
        ]

        for msg in _demo_conversation:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.divider()
        st.markdown("**API 키를 입력하면 위와 같은 대화를 실시간으로 진행할 수 있습니다.**")
        st.stop()

    if st.session_state.agent is None:
        st.info("에이전트를 초기화하는 중입니다... 사이드바를 확인해주세요.")
        st.stop()

    chat_container = st.container()
    quick_msg = st.session_state.pop("quick_start_msg", None)
    user_input = st.chat_input("광고에 대해 무엇이든 물어보세요...")
    message_to_send = quick_msg or user_input

    with chat_container:
        for entry in st.session_state.chat_history:
            role = entry["role"]
            content = entry["content"]
            with st.chat_message(role):
                st.markdown(content)
                if "tool_results" in entry and entry["tool_results"]:
                    render_tool_results(entry["tool_results"])

        if message_to_send:
            message_to_send = message_to_send.strip()
            if not message_to_send:
                st.warning("메시지를 입력해주세요.")
                st.stop()

            if len(message_to_send) > MAX_INPUT_LENGTH:
                st.warning(f"메시지가 너무 깁니다 (최대 {MAX_INPUT_LENGTH}자). 줄여서 입력해주세요.")
                message_to_send = message_to_send[:MAX_INPUT_LENGTH]

            with st.chat_message("user"):
                st.markdown(message_to_send)

            st.session_state.chat_history.append({
                "role": "user",
                "content": message_to_send,
            })

            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    try:
                        response_text, tool_results = st.session_state.agent.chat(message_to_send)
                    except Exception as e:
                        error_msg = str(e)
                        if "rate_limit" in error_msg.lower() or "429" in error_msg:
                            response_text = "API 호출 한도에 도달했습니다. 잠시 후 다시 시도해주세요."
                        elif "authentication" in error_msg.lower() or "401" in error_msg:
                            response_text = "API 키가 유효하지 않습니다. 사이드바에서 올바른 키를 입력해주세요."
                        elif "timeout" in error_msg.lower():
                            response_text = "응답 시간이 초과되었습니다. 질문을 간단하게 줄여서 다시 시도해주세요."
                        else:
                            response_text = f"오류가 발생했습니다: {error_msg}"
                        tool_results = []

                st.markdown(response_text)
                if tool_results:
                    render_tool_results(tool_results)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "tool_results": tool_results,
            })
            st.session_state.tool_results = tool_results
