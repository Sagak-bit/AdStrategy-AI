# -*- coding: utf-8 -*-
"""
공통 Tool 스키마 정의
=====================
ad_agent.py (OpenAI Function Calling)와 mcp_ad_server.py (MCP Server)에서
공유하는 Tool 파라미터 스키마를 한 곳에서 관리합니다.

DRY 원칙: 스키마 변경 시 이 파일만 수정하면 양쪽에 자동 반영됩니다.
"""
from __future__ import annotations

from typing import Any

# ============================================================================
# 공통 Enum 값 (플랫폼, 산업, 국가, 캠페인 유형, 연령대)
# ============================================================================

PLATFORMS = ["Google Ads", "Meta Ads", "TikTok Ads"]
INDUSTRIES = ["Fintech", "EdTech", "Healthcare", "SaaS", "E-commerce"]
COUNTRIES = ["USA", "UK", "Germany", "Canada", "India", "UAE", "Australia", "Korea"]
CAMPAIGN_TYPES = ["Search", "Video", "Shopping", "Display"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["Male", "Female", "All"]

# ============================================================================
# 공통 파라미터 스키마 (JSON Schema 형식)
# ============================================================================

PREDICT_PARAMS = {
    "type": "object",
    "properties": {
        "platform": {
            "type": "string",
            "enum": PLATFORMS,
            "description": "광고 플랫폼"
        },
        "industry": {
            "type": "string",
            "enum": INDUSTRIES,
            "description": "광고주의 산업 분야"
        },
        "country": {
            "type": "string",
            "enum": COUNTRIES,
            "description": "타겟 국가"
        },
        "campaign_type": {
            "type": "string",
            "enum": CAMPAIGN_TYPES,
            "description": "캠페인 유형"
        },
        "ad_spend": {
            "type": "number",
            "description": "월 광고 예산 (USD)"
        },
        "month": {
            "type": "integer",
            "minimum": 1,
            "maximum": 12,
            "description": "광고 집행 월 (1-12)"
        },
        "is_retargeting": {
            "type": "integer",
            "enum": [0, 1],
            "description": "리타겟팅 캠페인 여부 (0: 아니오, 1: 예)"
        },
        "is_video": {
            "type": "integer",
            "enum": [0, 1],
            "description": "비디오 광고 여부 (0: 아니오, 1: 예)"
        },
        "target_age_group": {
            "type": "string",
            "enum": AGE_GROUPS,
            "description": "타겟 연령대"
        },
        "target_gender": {
            "type": "string",
            "enum": GENDERS,
            "description": "타겟 성별"
        },
    },
    "required": ["platform", "industry", "country", "campaign_type", "ad_spend", "month"]
}

COMPARE_PARAMS = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "description": "비교할 시나리오 목록 (각각 predict_ad_performance와 동일한 파라미터)",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "시나리오 이름 (예: 'Google Search')"},
                    "platform": {"type": "string", "enum": PLATFORMS},
                    "industry": {"type": "string", "enum": INDUSTRIES},
                    "country": {"type": "string", "enum": COUNTRIES},
                    "campaign_type": {"type": "string", "enum": CAMPAIGN_TYPES},
                    "ad_spend": {"type": "number"},
                    "month": {"type": "integer", "minimum": 1, "maximum": 12},
                    "is_retargeting": {"type": "integer", "enum": [0, 1]},
                    "is_video": {"type": "integer", "enum": [0, 1]},
                    "target_age_group": {"type": "string", "enum": AGE_GROUPS},
                },
                "required": ["label", "platform", "industry", "country", "campaign_type", "ad_spend", "month"]
            },
            "minItems": 2,
            "maxItems": 5,
        }
    },
    "required": ["scenarios"]
}

BENCHMARKS_PARAMS = {
    "type": "object",
    "properties": {
        "industry": {
            "type": "string",
            "enum": INDUSTRIES,
            "description": "산업 분야"
        },
        "platform": {
            "type": "string",
            "enum": PLATFORMS,
            "description": "광고 플랫폼 (선택)"
        },
        "country": {
            "type": "string",
            "enum": COUNTRIES,
            "description": "국가 (선택)"
        },
    },
    "required": ["industry"]
}

TRENDS_PARAMS = {
    "type": "object",
    "properties": {
        "industry": {
            "type": "string",
            "enum": INDUSTRIES,
            "description": "산업 분야"
        },
        "country": {
            "type": "string",
            "enum": COUNTRIES,
            "description": "국가 (선택, 미지정 시 전체 평균)"
        },
    },
    "required": ["industry"]
}

# ============================================================================
# OpenAI Function Calling 포맷으로 변환
# ============================================================================

def get_openai_tools() -> list[dict[str, Any]]:
    """OpenAI Function Calling 형식의 Tool 정의 목록 반환"""
    return [
        {
            "type": "function",
            "function": {
                "name": "predict_ad_performance",
                "description": "주어진 조건에서의 광고 성과를 예측합니다. ROAS, CPC, CPA, 전환수, 예상 매출을 반환합니다.",
                "parameters": PREDICT_PARAMS,
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_scenarios",
                "description": "여러 광고 시나리오를 한꺼번에 비교합니다. 최대 5개 시나리오를 동시에 예측하여 비교 테이블을 반환합니다.",
                "parameters": COMPARE_PARAMS,
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_historical_benchmarks",
                "description": "특정 산업/플랫폼/국가 조합의 과거 광고 성과 벤치마크(평균, 중앙값, 범위)를 조회합니다.",
                "parameters": BENCHMARKS_PARAMS,
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_industry_trends",
                "description": "산업별 검색 관심도 트렌드(월별 트렌드 지수)를 조회합니다. 어떤 산업이 상승/하락 추세인지 파악할 수 있습니다.",
                "parameters": TRENDS_PARAMS,
            }
        },
    ]


# ============================================================================
# MCP Tool 스키마로 변환
# ============================================================================

def get_mcp_tool_schemas() -> dict[str, dict[str, Any]]:
    """MCP 서버용 Tool inputSchema 딕셔너리 반환"""
    return {
        "predict_ad_performance": PREDICT_PARAMS,
        "compare_scenarios": COMPARE_PARAMS,
        "get_benchmarks": BENCHMARKS_PARAMS,
        "get_trends": TRENDS_PARAMS,
        "get_data_summary": {"type": "object", "properties": {}},
    }
