# -*- coding: utf-8 -*-
"""
AdStrategy AI -- Streamlit 웹 앱
광고주와 대화하며 데이터 기반 광고 전략을 설계하는 AI 에이전트 UI.

실행: streamlit run streamlit_app.py
"""

import os
import sys
import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# 프로젝트 루트 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    /* 사이드바 password input의 show/hide 토글 버튼 숨기기 */
    [data-testid="stSidebar"] button[kind="icon"][aria-label*="password"],
    [data-testid="stSidebar"] button[aria-label="Show password text"],
    [data-testid="stSidebar"] button[aria-label="Hide password text"] {
        display: none !important;
    }
    /* input type password의 reveal 버튼 (Edge/Chrome 내장) 숨기기 */
    input[type="password"]::-ms-reveal,
    input[type="password"]::-ms-clear {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 세션 초기화
# ============================================================================
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tool_results" not in st.session_state:
    st.session_state.tool_results = []
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False
if "api_key_from_env" not in st.session_state:
    st.session_state.api_key_from_env = False


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


def render_comparison_chart(comparison_data):
    """시나리오 비교 차트"""
    comparison = comparison_data.get("comparison", [])
    if not comparison:
        return

    # 데이터프레임 변환
    df = pd.DataFrame(comparison)

    # ROAS 비교 바 차트
    if "ROAS" in df.columns:
        roas_valid = df[df["ROAS"] != "N/A"].copy()
        if len(roas_valid) > 0:
            roas_valid["ROAS"] = pd.to_numeric(roas_valid["ROAS"])

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

    # 비교 테이블
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
    monthly = trend_data.get("monthly_trend_index", {})
    if not monthly:
        return

    months = sorted(monthly.keys(), key=lambda x: int(x))
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
        fig.add_vline(x=f"{peak}월", line_dash="dash", line_color="red",
                       annotation_text=f"Peak: {peak}월")

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
# 메인 채팅 영역
# ============================================================================

# 타이틀
st.title("AdStrategy AI")
st.caption("데이터 기반 디지털 광고 전략 설계 에이전트")

# API 키 미입력 시 안내
if not st.session_state.api_key_set or not st.session_state.get("api_key"):
    st.info(
        "왼쪽 사이드바에 **OpenAI API Key**를 입력하세요.\n\n"
        "키가 없다면 [platform.openai.com](https://platform.openai.com) 에서 발급받을 수 있습니다."
    )
    st.stop()

# 에이전트 미초기화 시 로딩
if st.session_state.agent is None:
    st.info("에이전트를 초기화하는 중입니다... 사이드바를 확인해주세요.")
    st.stop()

# --- 대화 이력 렌더링 ---
for entry in st.session_state.chat_history:
    role = entry["role"]
    content = entry["content"]

    with st.chat_message(role):
        st.markdown(content)

        # 해당 메시지에 연결된 Tool 결과가 있으면 차트 표시
        if "tool_results" in entry and entry["tool_results"]:
            render_tool_results(entry["tool_results"])

# --- Quick Start 메시지 처리 ---
quick_msg = st.session_state.pop("quick_start_msg", None)

# --- 사용자 입력 ---
user_input = st.chat_input("광고에 대해 무엇이든 물어보세요...")

# Quick Start 또는 직접 입력
message_to_send = quick_msg or user_input

if message_to_send:
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(message_to_send)

    st.session_state.chat_history.append({
        "role": "user",
        "content": message_to_send,
    })

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                response_text, tool_results = st.session_state.agent.chat(message_to_send)
            except Exception as e:
                response_text = f"오류가 발생했습니다: {str(e)}"
                tool_results = []

        st.markdown(response_text)

        # Tool 결과 시각화
        if tool_results:
            render_tool_results(tool_results)

    # 대화 이력에 저장
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text,
        "tool_results": tool_results,
    })

    st.session_state.tool_results = tool_results
