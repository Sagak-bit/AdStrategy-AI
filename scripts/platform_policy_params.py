# -*- coding: utf-8 -*-
"""
플랫폼별 광고 정책 파라미터
============================
실제 광고 플랫폼의 예산-성과 관계를 모사하는 파라미터 정의.
strategy_simulation.py와 streamlit_budget_simulator.py에서 공유합니다.

각 플랫폼의 반응 곡선 유형:
- Google Ads: S형 포화 (sigmoid) — 초기 학습 후 노출이 포화되는 구조
  (근거: Google Ads 스마트 비딩은 충분한 전환 데이터 축적 후 최적화)
- Meta Ads: 소예산 페널티 (log_penalty) — 최소 유효 예산 미달 시 학습 부족
  (근거: Meta 공식 문서 "최소 일예산 = CPA x 50" 권장)
- TikTok Ads: 선형 + 변동 (linear_volatile) — 경매 변동성이 높은 구조
  (근거: TikTok Ads Manager의 예산 변동 알림 빈도가 타 플랫폼 대비 높음)
"""
from __future__ import annotations

from typing import Any

import numpy as np

PLATFORM_POLICY: dict[str, dict[str, Any]] = {
    "Google Ads": {
        "curve": "sigmoid",
        "min_effective_budget": 500,
        "saturation_budget": 15000,
        "max_multiplier": 1.0,
        "note": "S형 포화: 초기 학습 구간 후 노출 포화",
    },
    "Meta Ads": {
        "curve": "log_penalty",
        "min_effective_budget": 1000,
        "penalty_below_min": 0.6,
        "max_multiplier": 1.1,
        "note": "소예산 학습 기간 불리, 충분 예산 시 효율적",
    },
    "TikTok Ads": {
        "curve": "linear_volatile",
        "min_effective_budget": 300,
        "volatility": 0.15,
        "max_multiplier": 1.2,
        "note": "경매 변동 높음, 선형+변동 구조",
    },
}

BASE_ROAS_BY_PLATFORM_INDUSTRY: dict[str, dict[str, float]] = {
    "Google Ads": {
        "Fintech": 4.2, "EdTech": 3.8, "Healthcare": 5.1,
        "SaaS": 4.5, "E-commerce": 3.5,
    },
    "Meta Ads": {
        "Fintech": 3.5, "EdTech": 4.1, "Healthcare": 3.2,
        "SaaS": 3.8, "E-commerce": 4.8,
    },
    "TikTok Ads": {
        "Fintech": 5.0, "EdTech": 6.2, "Healthcare": 3.0,
        "SaaS": 4.0, "E-commerce": 7.5,
    },
}


def compute_roas_multiplier(
    budget: float,
    policy: dict[str, Any],
    random_state: int | None = None,
) -> float:
    """예산과 정책 파라미터로 ROAS 배수를 계산."""
    if policy["curve"] == "sigmoid":
        x = (budget - policy["min_effective_budget"]) / (
            policy["saturation_budget"] - policy["min_effective_budget"]
        )
        x = np.clip(x, -2, 5)
        return float(policy["max_multiplier"] / (1 + np.exp(-5 * (x - 0.5))))

    if policy["curve"] == "log_penalty":
        if budget < policy["min_effective_budget"]:
            return float(
                policy["penalty_below_min"] * (budget / policy["min_effective_budget"])
            )
        return float(
            policy["max_multiplier"] * np.log1p(budget / 1000) / np.log1p(20)
        )

    mult = float(policy["max_multiplier"] * np.log1p(budget / 500) / np.log1p(40))
    if random_state is not None:
        rng = np.random.RandomState(int(budget) if random_state == -1 else random_state)
        mult *= 1 + rng.normal(0, policy.get("volatility", 0.15))
    return mult


def estimate_roas(
    platform: str,
    industry: str,
    budget: float,
    random_state: int | None = None,
) -> float:
    """플랫폼·산업·예산 조합의 기대 ROAS를 계산."""
    policy = PLATFORM_POLICY.get(platform)
    if policy is None:
        return 0.0
    base = BASE_ROAS_BY_PLATFORM_INDUSTRY.get(platform, {}).get(industry, 4.0)
    mult = compute_roas_multiplier(budget, policy, random_state)
    return max(0.0, base * mult)
