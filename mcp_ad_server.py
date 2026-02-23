# -*- coding: utf-8 -*-
"""
AdStrategy MCP Server
=====================
Model Context Protocol (MCP) 서버로 광고 전략 AI 기능을 외부 LLM/에이전트에 제공합니다.

MCP Tools:
  1. predict_ad_performance - 광고 성과 예측
  2. compare_scenarios - 시나리오 비교
  3. get_benchmarks - 벤치마크 조회
  4. get_trends - 산업 트렌드 조회
  5. get_data_summary - 데이터 요약 통계

MCP Resources:
  - ads://data/summary - 데이터셋 요약
  - ads://model/info - 모델 성능 정보

실행:
  pip install mcp
  python mcp_ad_server.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import pandas as pd

from config import (
    BASE_DATA_PATH,
    ENRICHED_DATA_PATH,
    MAX_SCENARIOS,
    MIN_AD_SPEND,
    PREDICTOR_PKL_PATH,
    PROJECT_ROOT,
    TRENDS_DATA_PATH,
)
from utils import compute_roi_pct

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tool_schemas import get_mcp_tool_schemas

logger = logging.getLogger(__name__)

# ============================================================================
# 데이터/모델 로딩
# ============================================================================

def _load_predictor() -> Any | None:
    """ML 예측 모델 로드 (pickle 우선, 없으면 재학습)."""
    try:
        from ads_predictor_v2 import AdsPredictor_V2

        if os.path.exists(PREDICTOR_PKL_PATH):
            predictor = AdsPredictor_V2.load(PREDICTOR_PKL_PATH)
        else:
            logger.info("Pickle not found, training fresh model")
            predictor = AdsPredictor_V2(
                data_path=ENRICHED_DATA_PATH,
                fallback_path=BASE_DATA_PATH,
                exclude_leakage=True,
            )

        if not hasattr(predictor, "df") or predictor.df is None:
            for path in (ENRICHED_DATA_PATH, BASE_DATA_PATH):
                if os.path.exists(path):
                    predictor.df = pd.read_csv(path)
                    predictor.df["date"] = pd.to_datetime(
                        predictor.df["date"], errors="coerce",
                    )
                    break

        return predictor
    except Exception as e:
        logger.warning("Predictor load failed: %s", e)
    return None


def _load_data() -> dict[str, pd.DataFrame]:
    """벤치마크/트렌드 데이터 로드."""
    data: dict[str, pd.DataFrame] = {}
    for key, primary, fallback in [
        ("benchmark", ENRICHED_DATA_PATH, BASE_DATA_PATH),
        ("trends", TRENDS_DATA_PATH, None),
    ]:
        for path in (primary, fallback):
            if path and os.path.exists(path):
                data[key] = pd.read_csv(path)
                break
    return data


# ============================================================================
# Tool 구현 함수 (ad_agent 와 MCP 서버 공통 사용)
# ============================================================================

class AdTools:
    """광고 분석 Tool 구현체.

    ad_agent.py 와 MCP 서버 양쪽에서 이 클래스를 사용하여
    predict/compare/benchmarks/trends 로직을 단일 소스로 유지한다.
    """

    def __init__(self) -> None:
        self.predictor = _load_predictor()
        self.data = _load_data()

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, platform: str, industry: str, country: str,
                campaign_type: str, ad_spend: float, month: int,
                **kwargs: Any) -> dict[str, Any]:
        """광고 성과 예측 (에이전트/MCP 공용)."""
        if self.predictor is None:
            return {"error": "예측 모델이 로드되지 않았습니다. models/predictor_v2.pkl 을 확인하세요."}

        try:
            ad_spend = float(ad_spend)
            month = int(month)
        except (TypeError, ValueError) as e:
            return {"error": f"ad_spend/month 변환 실패: {e}"}

        if ad_spend < MIN_AD_SPEND:
            return {"error": f"ad_spend는 {MIN_AD_SPEND} 이상이어야 합니다."}
        if not 1 <= month <= 12:
            return {"error": "month는 1~12 사이여야 합니다."}

        try:
            result = self.predictor.predict(
                platform=platform, industry=industry, country=country,
                ad_spend=ad_spend, campaign_type=campaign_type,
                month=month, **kwargs,
            )
        except (ValueError, KeyError) as e:
            return {"error": str(e)}

        pred = result["predictions"]
        summary: dict[str, Any] = {
            "input": result["input"],
            "predictions": {},
            "historical": result.get("historical_reference", {}),
        }

        for metric in ("ROAS", "CPC", "CPA", "conversions"):
            if metric in pred:
                p = pred[metric]
                summary["predictions"][metric] = {
                    "predicted": p["predicted"],
                    "confidence_interval_68": f"{p['ci_68'][0]} ~ {p['ci_68'][1]}",
                    "confidence_interval_95": f"{p['ci_95'][0]} ~ {p['ci_95'][1]}",
                    "confidence_level": p["confidence"],
                    "model_used": p.get("model_used", "unknown"),
                }

        if "estimated_revenue" in pred:
            rev = pred["estimated_revenue"]
            summary["predictions"]["estimated_revenue"] = {
                "predicted": rev["predicted"],
                "confidence_interval_68": f"{rev['ci_68'][0]} ~ {rev['ci_68'][1]}",
            }

        if "estimated_clicks" in pred:
            summary["predictions"]["estimated_clicks"] = pred["estimated_clicks"]["predicted"]

        if ad_spend > 0 and "estimated_revenue" in pred:
            rev_val = pred["estimated_revenue"]["predicted"]
            summary["predictions"]["estimated_ROI_percent"] = round(
                compute_roi_pct(rev_val, ad_spend), 1,
            )

        return summary

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    def compare(self, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
        """여러 시나리오 비교."""
        if len(scenarios) < 2:
            return {"error": "최소 2개 시나리오가 필요합니다."}

        results: list[dict[str, Any]] = []
        for s in scenarios[:MAX_SCENARIOS]:
            params = {k: v for k, v in s.items() if k != "label"}
            label = s.get("label", f"Scenario {len(results) + 1}")
            pred = self.predict(**params)
            row: dict[str, Any] = {"scenario": label}
            if "error" not in pred:
                preds = pred.get("predictions", {})
                row["ROAS"] = preds.get("ROAS", {}).get("predicted", "N/A")
                row["CPC"] = preds.get("CPC", {}).get("predicted", "N/A")
                row["CPA"] = preds.get("CPA", {}).get("predicted", "N/A")
                row["conversions"] = preds.get("conversions", {}).get("predicted", "N/A")
                est_rev = preds.get("estimated_revenue", {})
                row["estimated_revenue"] = est_rev.get("predicted", "N/A") if isinstance(est_rev, dict) else est_rev
                row["estimated_ROI_percent"] = preds.get("estimated_ROI_percent", "N/A")
                row["ROAS_confidence"] = preds.get("ROAS", {}).get("confidence_level", "N/A")
                row["historical_samples"] = pred.get("historical", {}).get("exact_match_count", 0)
            else:
                row["error"] = pred["error"]
            results.append(row)

        best_roas: float | None = None
        best_label: str | None = None
        for r in results:
            roas = r.get("ROAS")
            if isinstance(roas, (int, float)) and (best_roas is None or roas > best_roas):
                best_roas = roas
                best_label = r["scenario"]

        return {
            "comparison": results,
            "best_roas_scenario": best_label,
            "recommendation": (
                f"ROAS 기준 최적 시나리오는 '{best_label}' (ROAS: {best_roas:.2f})입니다."
                if best_label and best_roas is not None
                else None
            ),
        }

    def benchmarks(self, industry: str, platform: str | None = None, country: str | None = None) -> dict[str, Any]:
        """과거 벤치마크 통계 조회"""
        if 'benchmark' not in self.data:
            return {"error": "벤치마크 데이터가 없습니다."}

        df = self.data['benchmark']
        mask = df["industry"] == industry
        if platform:
            mask &= df["platform"] == platform
        if country:
            mask &= df["country"] == country

        filtered = df[mask]
        if len(filtered) == 0:
            return {"message": "해당 조건의 데이터 없음", "sample_count": 0}

        result = {"sample_count": len(filtered), "filters": {"industry": industry, "platform": platform, "country": country}}
        for m in ["ROAS", "CPC", "CPA", "CTR"]:
            if m in filtered.columns:
                col = filtered[m].dropna()
                if len(col) > 0:
                    result[m] = {
                        "mean": round(float(col.mean()), 3),
                        "median": round(float(col.median()), 3),
                        "std": round(float(col.std()), 3),
                        "q25": round(float(col.quantile(0.25)), 3),
                        "q75": round(float(col.quantile(0.75)), 3),
                    }
        return result

    def trends(self, industry: str, country: str | None = None) -> dict[str, Any]:
        """산업별 트렌드 조회"""
        if 'trends' not in self.data:
            return {"error": "트렌드 데이터가 없습니다."}

        df = self.data['trends']
        mask = df["industry"] == industry
        if country:
            mask &= df["country"] == country

        filtered = df[mask]
        if len(filtered) == 0:
            return {"message": "해당 산업의 트렌드 데이터 없음"}

        if "month" in filtered.columns and "trend_index" in filtered.columns:
            monthly = filtered.groupby("month")["trend_index"].mean().round(1)
            return {
                "industry": industry,
                "country": country or "전체",
                "monthly_trend": {str(k): v for k, v in monthly.to_dict().items()},
                "peak_month": int(monthly.idxmax()),
                "low_month": int(monthly.idxmin()),
                "avg_index": round(float(filtered["trend_index"].mean()), 1),
            }
        return {"message": "트렌드 인덱스 컬럼 없음"}

    def data_summary(self) -> dict[str, Any]:
        """데이터셋 요약 통계"""
        if 'benchmark' not in self.data:
            return {"error": "데이터가 없습니다."}

        df = self.data['benchmark']
        summary = {
            "total_records": len(df),
            "columns": len(df.columns),
            "platforms": sorted(df['platform'].unique().tolist()) if 'platform' in df.columns else [],
            "industries": sorted(df['industry'].unique().tolist()) if 'industry' in df.columns else [],
            "countries": sorted(df['country'].unique().tolist()) if 'country' in df.columns else [],
            "date_range": f"{df['date'].min()} ~ {df['date'].max()}" if 'date' in df.columns else "N/A",
        }

        for m in ["ROAS", "CPC", "CPA", "CTR"]:
            if m in df.columns:
                summary[f"{m}_stats"] = {
                    "mean": round(float(df[m].mean()), 3),
                    "median": round(float(df[m].median()), 3),
                    "std": round(float(df[m].std()), 3),
                }

        if 'is_synthetic' in df.columns:
            real = (df['is_synthetic'] == 0).sum()
            synth = (df['is_synthetic'] == 1).sum()
            summary["real_data_count"] = int(real)
            summary["synthetic_data_count"] = int(synth)
            summary["real_data_pct"] = round(real / len(df) * 100, 1)

        return summary


# ============================================================================
# MCP 서버 정의
# ============================================================================

def create_mcp_server():
    """MCP 서버 생성"""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent, Resource
    except ImportError:
        print("[ERROR] MCP 패키지가 설치되지 않았습니다.")
        print("  pip install mcp")
        print("\n대신 Standalone 모드로 실행합니다...\n")
        return None

    app = Server("adstrategy-mcp")
    tools = AdTools()
    mcp_schemas = get_mcp_tool_schemas()  # 공통 스키마 사용 (DRY)

    # --- Tool 등록 ---
    @app.list_tools()
    async def list_tools():
        return [
            Tool(
                name="predict_ad_performance",
                description="특정 조건에서의 광고 성과(ROAS, CPC, CPA 등)를 ML 모델로 예측합니다.",
                inputSchema=mcp_schemas["predict_ad_performance"],
            ),
            Tool(
                name="compare_scenarios",
                description="여러 광고 시나리오를 동시에 비교하여 최적 조합을 추천합니다. (최대 5개)",
                inputSchema=mcp_schemas["compare_scenarios"],
            ),
            Tool(
                name="get_benchmarks",
                description="특정 산업/플랫폼/국가의 과거 광고 성과 벤치마크(평균, 중앙값 등)를 조회합니다.",
                inputSchema=mcp_schemas["get_benchmarks"],
            ),
            Tool(
                name="get_trends",
                description="산업별 월간 검색 관심도 트렌드를 조회합니다.",
                inputSchema=mcp_schemas["get_trends"],
            ),
            Tool(
                name="get_data_summary",
                description="전체 광고 데이터셋의 요약 통계(건수, 플랫폼, 산업, 주요 지표)를 조회합니다.",
                inputSchema=mcp_schemas["get_data_summary"],
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "predict_ad_performance":
                result = tools.predict(**arguments)
            elif name == "compare_scenarios":
                result = tools.compare(arguments.get("scenarios", []))
            elif name == "get_benchmarks":
                result = tools.benchmarks(**arguments)
            elif name == "get_trends":
                result = tools.trends(**arguments)
            elif name == "get_data_summary":
                result = tools.data_summary()
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    # --- Resource 등록 ---
    @app.list_resources()
    async def list_resources():
        return [
            Resource(uri="ads://data/summary", name="광고 데이터 요약",
                     description="전체 광고 데이터셋 요약 통계", mimeType="application/json"),
            Resource(uri="ads://model/info", name="ML 모델 정보",
                     description="AdsPredictor V2 모델 성능 정보", mimeType="application/json"),
        ]

    @app.read_resource()
    async def read_resource(uri: str):
        if uri == "ads://data/summary":
            return json.dumps(tools.data_summary(), ensure_ascii=False, default=str)
        elif uri == "ads://model/info":
            if tools.predictor:
                info = {}
                for target, model_data in tools.predictor.models.items():
                    info[target] = {
                        "model_name": model_data["model_name"],
                        "r2_score": round(model_data["r2_score"], 3),
                        "mae": round(model_data["mae"], 2),
                    }
                return json.dumps(info, ensure_ascii=False, default=str)
            return json.dumps({"error": "모델 로드 안 됨"})
        return json.dumps({"error": f"Unknown resource: {uri}"})

    return app


# ============================================================================
# Standalone 모드 (MCP 없이 테스트)
# ============================================================================

def standalone_test():
    """MCP 없이 Tool 직접 테스트"""
    print("=" * 60)
    print("  AdStrategy MCP Server - Standalone Test Mode")
    print("=" * 60)

    tools = AdTools()

    # 데이터 요약
    print("\n[1] Data Summary:")
    summary = tools.data_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # 예측 테스트
    print("\n[2] Prediction Test (TikTok / Fintech / USA / Video / $5000 / March):")
    pred = tools.predict(
        platform="TikTok Ads", industry="Fintech", country="USA",
        campaign_type="Video", ad_spend=5000, month=3
    )
    print(json.dumps(pred, indent=2, ensure_ascii=False, default=str))

    # 벤치마크
    print("\n[3] Benchmarks (Fintech):")
    bench = tools.benchmarks(industry="Fintech")
    print(json.dumps(bench, indent=2, ensure_ascii=False, default=str))

    # 트렌드
    print("\n[4] Trends (E-commerce):")
    trend = tools.trends(industry="E-commerce")
    print(json.dumps(trend, indent=2, ensure_ascii=False, default=str))


# ============================================================================
# 메인
# ============================================================================

async def main():
    """MCP 서버 실행"""
    server = create_mcp_server()
    if server is None:
        standalone_test()
        return

    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    if "--test" in sys.argv:
        standalone_test()
    else:
        asyncio.run(main())
