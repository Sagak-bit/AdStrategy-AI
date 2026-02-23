# -*- coding: utf-8 -*-
"""
Smoke Test — 프로젝트 핵심 기능 검증
=====================================
데이터 로드, 모델 예측, 에이전트 도구, 시각화 산출물의
기본 동작을 PASS/FAIL 형식으로 검증합니다.

실행: python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import pickle
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, func):
    try:
        msg = func()
        results.append((PASS, name, msg or "OK"))
    except Exception as e:
        results.append((FAIL, name, f"{e.__class__.__name__}: {e}"))


# =========================================================================
# 1. 데이터 로드 검증
# =========================================================================
def test_data_load():
    from config import ENRICHED_DATA_PATH
    import pandas as pd
    df = pd.read_csv(ENRICHED_DATA_PATH)
    assert len(df) >= 10000, f"행 수 부족: {len(df)}"
    assert len(df.columns) >= 40, f"컬럼 부족: {len(df.columns)}"
    return f"{len(df):,}행 x {len(df.columns)}컬럼"


def test_korea_data():
    from config import DATA_DIR
    import pandas as pd
    path = os.path.join(DATA_DIR, "enriched_ads_with_korea.csv")
    df = pd.read_csv(path)
    korea = df[df["country"] == "Korea"]
    assert len(korea) >= 100, f"Korea 행 부족: {len(korea)}"
    return f"Korea {len(korea)}건 / 전체 {len(df):,}건"


# =========================================================================
# 2. 모델 로드 + 예측 검증
# =========================================================================
def test_model_load():
    from config import PREDICTOR_PKL_PATH
    assert os.path.exists(PREDICTOR_PKL_PATH), "모델 파일 미존재"
    with open(PREDICTOR_PKL_PATH, "rb") as f:
        model = pickle.load(f)
    return f"모델 로드 성공: {type(model).__name__}"


def test_single_prediction():
    from mcp_ad_server import _load_predictor
    predictor = _load_predictor()
    assert predictor is not None, "Predictor 로드 실패"
    result = predictor.predict(
        platform="Google Ads", industry="Fintech", country="USA",
        ad_spend=5000, campaign_type="Search", month=6,
    )
    preds = result.get("predictions", result)
    roas_data = preds.get("ROAS", preds)
    val = roas_data.get("predicted", roas_data) if isinstance(roas_data, dict) else roas_data
    assert val is not None and float(val) > 0, f"ROAS 예측값 비정상: {val}"
    return f"ROAS={float(val):.2f}"


# =========================================================================
# 3. 에이전트 도구 검증
# =========================================================================
def test_ad_tools():
    from mcp_ad_server import AdTools
    tools = AdTools()
    pred = tools.predict(
        platform="Meta Ads", industry="E-commerce", country="UK",
        campaign_type="Video", ad_spend=3000, month=11,
    )
    assert "error" not in pred, f"예측 에러: {pred.get('error')}"
    return f"predict OK, ROAS={pred['predictions']['ROAS']['predicted']:.2f}"


def test_compare_scenarios():
    from mcp_ad_server import AdTools
    tools = AdTools()
    scenarios = [
        {"label": "A", "platform": "Google Ads", "industry": "SaaS",
         "country": "USA", "campaign_type": "Search", "ad_spend": 5000, "month": 3},
        {"label": "B", "platform": "TikTok Ads", "industry": "SaaS",
         "country": "USA", "campaign_type": "Video", "ad_spend": 5000, "month": 3},
    ]
    cmp = tools.compare(scenarios)
    assert isinstance(cmp, dict), f"compare 반환 타입 이상: {type(cmp)}"
    assert "comparison" in cmp or "error" not in cmp, f"compare 에러: {cmp}"
    comp_list = cmp.get("comparison", [])
    assert len(comp_list) >= 1, f"비교 결과 부족: {len(comp_list)}"
    rec = cmp.get("recommendation", {})
    best = rec.get("best_scenario", "N/A") if isinstance(rec, dict) else str(rec)[:30]
    return f"compare OK, {len(comp_list)}개 시나리오, best={best}"


def test_benchmarks():
    from mcp_ad_server import AdTools
    tools = AdTools()
    b = tools.benchmarks(industry="Fintech")
    assert "overall" in b or "summary" in b or "filters" in b, f"벤치마크 응답 이상"
    return "benchmarks OK"


def test_trends():
    from mcp_ad_server import AdTools
    tools = AdTools()
    t = tools.trends(industry="E-commerce")
    assert "industry" in t, f"trends 응답 이상"
    return "trends OK"


# =========================================================================
# 4. 시각화 검증
# =========================================================================
def test_figures_exist():
    from config import FIGURES_DIR
    pngs = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
    assert len(pngs) >= 15, f"PNG {len(pngs)}개 (최소 15개 필요)"
    return f"{len(pngs)}개 PNG 존재"


REQUIRED_FIGURES = [
    "waterfall_r2_leakage.png",
    "leakage_risk_dashboard.png",
    "shap_force_leakage_vs_clean.png",
    "ablation_guardrail.png",
    "budget_reallocation_impact.png",
    "segment_confidence_heatmap.png",
    "platform_budget_response_curves.png",
    "synthetic_fidelity_heatmap.png",
]


def test_required_figures():
    from config import FIGURES_DIR
    missing = [f for f in REQUIRED_FIGURES if not os.path.exists(os.path.join(FIGURES_DIR, f))]
    assert not missing, f"누락: {missing}"
    return f"필수 {len(REQUIRED_FIGURES)}개 모두 존재"


# =========================================================================
# 5. 설정 / 스키마 검증
# =========================================================================
def test_tool_schemas():
    from tool_schemas import COUNTRIES, PLATFORMS, get_openai_tools
    assert "Korea" in COUNTRIES, "Korea 미포함"
    assert len(PLATFORMS) == 3
    tools = get_openai_tools()
    assert len(tools) == 4, f"Tool 수: {len(tools)}"
    return f"COUNTRIES={len(COUNTRIES)}, Tools={len(tools)}"


def test_platform_policy():
    from scripts.platform_policy_params import PLATFORM_POLICY, estimate_roas
    assert len(PLATFORM_POLICY) == 3
    r = estimate_roas("Google Ads", "Fintech", 5000)
    assert r > 0, f"estimate_roas 비정상: {r}"
    return f"정책 3개, Google/Fintech/$5K → ROAS={r:.2f}"


# =========================================================================
# 실행
# =========================================================================
def main():
    print("=" * 60)
    print("  AdStrategy AI — Smoke Test")
    print("=" * 60)

    tests = [
        ("데이터 로드", test_data_load),
        ("한국 데이터", test_korea_data),
        ("모델 로드", test_model_load),
        ("단일 예측", test_single_prediction),
        ("AdTools predict", test_ad_tools),
        ("AdTools compare", test_compare_scenarios),
        ("AdTools benchmarks", test_benchmarks),
        ("AdTools trends", test_trends),
        ("시각화 개수", test_figures_exist),
        ("필수 시각화", test_required_figures),
        ("Tool 스키마", test_tool_schemas),
        ("플랫폼 정책", test_platform_policy),
    ]

    for name, fn in tests:
        check(name, fn)

    print()
    passed = sum(1 for s, _, _ in results if s == PASS)
    failed = sum(1 for s, _, _ in results if s == FAIL)

    for status, name, msg in results:
        icon = "[OK]" if status == PASS else "[!!]"
        print(f"  {icon} {name}: {msg}")

    print()
    print(f"  결과: {passed} PASS / {failed} FAIL (총 {len(results)})")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
