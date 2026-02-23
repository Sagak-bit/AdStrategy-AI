# -*- coding: utf-8 -*-
"""
예산 시뮬레이터 미니 대시보드
=============================
플랫폼·산업 선택 + 예산 슬라이더 → 예측 ROAS·CPC·CPA 차트

실행: streamlit run streamlit_budget_simulator.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.platform_policy_params import PLATFORM_POLICY, BASE_ROAS_BY_PLATFORM_INDUSTRY

st.set_page_config(page_title="예산 시뮬레이터", page_icon="💰", layout="wide")

st.title("💰 예산-ROAS 시뮬레이터")
st.caption("플랫폼별 정책 반영 예산 시뮬레이션 (실무 가정 기반)")

col1, col2, col3 = st.columns(3)

with col1:
    platform = st.selectbox("플랫폼", ["Google Ads", "Meta Ads", "TikTok Ads"])
with col2:
    industry = st.selectbox("산업", ["Fintech", "EdTech", "Healthcare", "SaaS", "E-commerce"])
with col3:
    country = st.selectbox("국가", ["USA", "UK", "Germany", "Canada", "India", "UAE", "Australia"])

budget = st.slider("월 광고 예산 (USD)", min_value=100, max_value=20000, value=5000, step=100)

st.divider()

# 반응 곡선 시뮬레이션
policy = PLATFORM_POLICY[platform]
base_roas = BASE_ROAS_BY_PLATFORM_INDUSTRY.get(platform, {}).get(industry, 4.0)

budgets = np.arange(100, 20001, 100)
roas_curve = []
for b in budgets:
    if policy["curve"] == "sigmoid":
        x = (b - policy["min_effective_budget"]) / (policy["saturation_budget"] - policy["min_effective_budget"])
        x = np.clip(x, -2, 5)
        mult = policy["max_multiplier"] / (1 + np.exp(-5 * (x - 0.5)))
    elif policy["curve"] == "log_penalty":
        if b < policy["min_effective_budget"]:
            mult = policy["penalty_below_min"] * (b / policy["min_effective_budget"])
        else:
            mult = policy["max_multiplier"] * np.log1p(b / 1000) / np.log1p(20)
    else:
        mult = policy["max_multiplier"] * np.log1p(b / 500) / np.log1p(40)
    roas_curve.append(max(0, base_roas * mult))

current_idx = (budget - 100) // 100
current_roas = roas_curve[min(current_idx, len(roas_curve) - 1)]
est_revenue = budget * current_roas
est_roi = ((est_revenue - budget) / budget * 100) if budget > 0 else 0

# 메트릭
m1, m2, m3, m4 = st.columns(4)
m1.metric("예측 ROAS", f"{current_roas:.2f}")
m2.metric("예상 매출", f"${est_revenue:,.0f}")
m3.metric("예상 ROI", f"{est_roi:.1f}%")
m4.metric("플랫폼 정책", policy["note"][:20])

# 반응 곡선 차트
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
    title=f"{platform} × {industry} — 예산-ROAS 반응 곡선",
    xaxis_title="월 광고 예산 (USD)",
    yaxis_title="기대 ROAS",
    height=450,
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

st.info(f"**{platform} 정책**: {policy['note']}")
st.caption("⚠ 실무 가정 기반 시뮬레이션이며 실측 데이터가 아닙니다.")
