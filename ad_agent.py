# -*- coding: utf-8 -*-
"""
AdStrategy AI -- 광고 전략 설계 에이전트 코어
- OpenAI GPT-4o + Function Calling
- AdsPredictor_V2 ML 모델 연동
- 광고주와 대화하며 최적의 광고 전략을 설계
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from openai import OpenAI

# 프로젝트 루트를 path에 추가 (ads_predictor_v2 임포트용)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ads_predictor_v2 import AdsPredictor_V2

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """당신은 **AdStrategy AI**입니다. 디지털 광고 전략을 설계하는 전문 AI 컨설턴트입니다.
광고주와 자연스럽게 대화하면서, 데이터에 기반한 최적의 광고 집행 전략을 함께 만들어갑니다.

## 당신이 사용할 수 있는 도구(Tool)
1. `predict_ad_performance` — 특정 조건(플랫폼, 산업, 국가, 예산, 캠페인 유형, 월)에서의 광고 성과를 예측합니다.
2. `compare_scenarios` — 여러 시나리오를 한꺼번에 비교합니다 (최대 5개).
3. `get_historical_benchmarks` — 특정 산업/플랫폼/국가의 과거 벤치마크 통계(평균 ROAS, CPC 등)를 조회합니다.
4. `get_industry_trends` — 산업별 검색 관심도 트렌드를 조회합니다.

## 대화 흐름 (5단계)
자연스러운 대화 속에서 아래 정보를 순서대로 파악하세요. 한 번에 질문은 **1~2개**만 하세요.

**1단계 — 비즈니스 이해**
- 어떤 산업/분야인지 (Fintech, EdTech, Healthcare, SaaS, E-commerce)
- 제품/서비스가 무엇인지
- 광고의 목표 (브랜딩, 전환, 매출, 앱 설치 등)

**2단계 — 타겟 파악**
- 타겟 국가/시장 (USA, UK, Germany, Canada, India, UAE, Australia)
- 타겟 연령대, 성별, 관심사

**3단계 — 예산과 일정**
- 월 광고 예산 (USD)
- 광고 집행 시작 월, 기간

**4단계 — 전략 설계 (Tool 활용)**
정보가 모이면, 반드시 Tool을 호출하여 데이터에 근거한 전략을 제시하세요.
- 플랫폼별 비교 (Google Ads vs Meta Ads vs TikTok Ads)
- 캠페인 유형별 비교 (Search, Video, Shopping, Display)
- 예상 ROAS, CPC, CPA, 전환수, 매출을 구체적으로 제시

**5단계 — 최적화 & What-if**
- 광고주가 "다른 플랫폼은?", "예산을 늘리면?" 같은 질문을 하면 추가 시나리오를 비교
- 리타겟팅, 비디오 광고 등 세부 옵션 효과도 분석

## 응답 규칙
- **한국어**로 대화하세요 (광고주가 영어를 쓰면 영어로 전환).
- 숫자와 데이터는 **구체적으로** 제시하세요 (예: "ROAS 4.9 예상, 월 $5,000 투자 시 약 $24,500 매출 기대").
- 예측 결과를 전달할 때는 **신뢰도(High/Medium/Low)**와 **근거 샘플 수**를 반드시 언급하세요.
- 추천을 할 때는 **왜 그 플랫폼/전략이 좋은지 데이터 근거**를 설명하세요.
- 불확실할 때는 솔직하게 말하고, 추가 정보를 요청하세요.
- 대화 처음에는 친근한 인사와 함께 "어떤 광고를 계획하고 계신지" 물어보세요.

## 사용 가능한 값
- 플랫폼: Google Ads, Meta Ads, TikTok Ads
- 산업: Fintech, EdTech, Healthcare, SaaS, E-commerce
- 국가: USA, UK, Germany, Canada, India, UAE, Australia
- 캠페인 유형: Search, Video, Shopping, Display
- 타겟 연령: 18-24, 25-34, 35-44, 45-54, 55+
"""

# ============================================================================
# Tool 정의 (OpenAI Function Calling Schema)
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_ad_performance",
            "description": "주어진 조건에서의 광고 성과를 예측합니다. ROAS, CPC, CPA, 전환수, 예상 매출을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "enum": ["Google Ads", "Meta Ads", "TikTok Ads"],
                        "description": "광고 플랫폼"
                    },
                    "industry": {
                        "type": "string",
                        "enum": ["Fintech", "EdTech", "Healthcare", "SaaS", "E-commerce"],
                        "description": "광고주의 산업 분야"
                    },
                    "country": {
                        "type": "string",
                        "enum": ["USA", "UK", "Germany", "Canada", "India", "UAE", "Australia"],
                        "description": "타겟 국가"
                    },
                    "campaign_type": {
                        "type": "string",
                        "enum": ["Search", "Video", "Shopping", "Display"],
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
                        "enum": ["18-24", "25-34", "35-44", "45-54", "55+"],
                        "description": "타겟 연령대"
                    },
                    "target_gender": {
                        "type": "string",
                        "enum": ["Male", "Female", "All"],
                        "description": "타겟 성별"
                    },
                },
                "required": ["platform", "industry", "country", "campaign_type", "ad_spend", "month"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_scenarios",
            "description": "여러 광고 시나리오를 한꺼번에 비교합니다. 최대 5개 시나리오를 동시에 예측하여 비교 테이블을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scenarios": {
                        "type": "array",
                        "description": "비교할 시나리오 목록 (각각 predict_ad_performance와 동일한 파라미터)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "시나리오 이름 (예: 'Google Search')"},
                                "platform": {"type": "string", "enum": ["Google Ads", "Meta Ads", "TikTok Ads"]},
                                "industry": {"type": "string", "enum": ["Fintech", "EdTech", "Healthcare", "SaaS", "E-commerce"]},
                                "country": {"type": "string", "enum": ["USA", "UK", "Germany", "Canada", "India", "UAE", "Australia"]},
                                "campaign_type": {"type": "string", "enum": ["Search", "Video", "Shopping", "Display"]},
                                "ad_spend": {"type": "number"},
                                "month": {"type": "integer", "minimum": 1, "maximum": 12},
                                "is_retargeting": {"type": "integer", "enum": [0, 1]},
                                "is_video": {"type": "integer", "enum": [0, 1]},
                                "target_age_group": {"type": "string", "enum": ["18-24", "25-34", "35-44", "45-54", "55+"]},
                            },
                            "required": ["label", "platform", "industry", "country", "campaign_type", "ad_spend", "month"]
                        },
                        "minItems": 2,
                        "maxItems": 5,
                    }
                },
                "required": ["scenarios"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_benchmarks",
            "description": "특정 산업/플랫폼/국가 조합의 과거 광고 성과 벤치마크(평균, 중앙값, 범위)를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "enum": ["Fintech", "EdTech", "Healthcare", "SaaS", "E-commerce"],
                        "description": "산업 분야"
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["Google Ads", "Meta Ads", "TikTok Ads"],
                        "description": "광고 플랫폼 (선택)"
                    },
                    "country": {
                        "type": "string",
                        "enum": ["USA", "UK", "Germany", "Canada", "India", "UAE", "Australia"],
                        "description": "국가 (선택)"
                    },
                },
                "required": ["industry"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_industry_trends",
            "description": "산업별 검색 관심도 트렌드(월별 트렌드 지수)를 조회합니다. 어떤 산업이 상승/하락 추세인지 파악할 수 있습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "enum": ["Fintech", "EdTech", "Healthcare", "SaaS", "E-commerce"],
                        "description": "산업 분야"
                    },
                    "country": {
                        "type": "string",
                        "enum": ["USA", "UK", "Germany", "Canada", "India", "UAE", "Australia"],
                        "description": "국가 (선택, 미지정 시 전체 평균)"
                    },
                },
                "required": ["industry"]
            }
        }
    },
]

# ============================================================================
# 에이전트 클래스
# ============================================================================

class AdStrategyAgent:
    """
    광고 전략 설계 AI 에이전트
    - OpenAI GPT-4o + Function Calling
    - AdsPredictor_V2 ML 모델 연동
    """

    def __init__(self, openai_api_key, model="gpt-4o",
                 predictor_path="models/predictor_v2.pkl",
                 data_path="data/enriched_ads_final.csv",
                 trends_path="data/trends/industry_trends.csv"):
        # OpenAI 클라이언트
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        # ML 예측 모델 로드
        self.predictor = self._load_predictor(predictor_path)

        # 벤치마크용 데이터 로드
        self.benchmark_df = self._load_benchmark_data(data_path)

        # 트렌드 데이터 로드
        self.trends_df = self._load_trends_data(trends_path)

        # 대화 이력
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 대화 중 수집된 광고주 정보
        self.collected_info = {}

        # Tool 호출 결과 기록 (시각화용)
        self.tool_results_log = []

    # ------------------------------------------------------------------
    # 모델/데이터 로딩
    # ------------------------------------------------------------------
    def _load_predictor(self, path):
        """AdsPredictor_V2 pickle 로드"""
        abs_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(abs_path):
            predictor = AdsPredictor_V2.load(abs_path)
            # pickle에 df가 저장되지 않으므로 수동으로 로드
            if not hasattr(predictor, 'df') or predictor.df is None:
                data_path = os.path.join(PROJECT_ROOT, "data", "enriched_ads_final.csv")
                fallback_path = os.path.join(PROJECT_ROOT, "global_ads_performance_dataset.csv")
                if os.path.exists(data_path):
                    predictor.df = pd.read_csv(data_path)
                elif os.path.exists(fallback_path):
                    predictor.df = pd.read_csv(fallback_path)
                else:
                    predictor.df = pd.DataFrame()
            return predictor
        else:
            print(f"[WARNING] Predictor not found at {abs_path}")
            return None

    def _load_benchmark_data(self, path):
        """벤치마크용 광고 데이터 로드"""
        abs_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(abs_path):
            df = pd.read_csv(abs_path)
            return df
        return None

    def _load_trends_data(self, path):
        """트렌드 데이터 로드"""
        abs_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(abs_path):
            df = pd.read_csv(abs_path)
            return df
        return None

    # ------------------------------------------------------------------
    # 메인 대화 루프
    # ------------------------------------------------------------------
    def chat(self, user_message):
        """
        사용자 메시지를 처리하고 응답을 반환합니다.

        Returns:
            tuple: (assistant_text, tool_results_list)
                - assistant_text: str - LLM의 최종 텍스트 응답
                - tool_results_list: list[dict] - 호출된 Tool 결과 목록
                  각 dict = {"tool": tool_name, "args": {}, "result": {}}
        """
        # 사용자 메시지 추가
        self.messages.append({"role": "user", "content": user_message})
        tool_results = []

        # Tool Calling 루프 (최대 5회 반복)
        for _ in range(5):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2000,
            )

            msg = response.choices[0].message

            # Tool 호출이 없으면 최종 응답
            if not msg.tool_calls:
                assistant_text = msg.content or ""
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.tool_results_log = tool_results
                return assistant_text, tool_results

            # Tool 호출 처리
            self.messages.append(msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                # Tool 실행
                result = self._execute_tool(fn_name, fn_args)
                tool_results.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result": result,
                })

                # 결과를 메시지에 추가
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        # 루프 종료 (드문 경우)
        self.tool_results_log = tool_results
        return "분석을 완료했습니다. 추가 질문이 있으시면 말씀해주세요.", tool_results

    # ------------------------------------------------------------------
    # Tool 실행기
    # ------------------------------------------------------------------
    def _execute_tool(self, tool_name, arguments):
        """Tool 이름에 따라 실제 함수 실행"""
        try:
            if tool_name == "predict_ad_performance":
                return self._tool_predict(arguments)
            elif tool_name == "compare_scenarios":
                return self._tool_compare(arguments)
            elif tool_name == "get_historical_benchmarks":
                return self._tool_benchmarks(arguments)
            elif tool_name == "get_industry_trends":
                return self._tool_trends(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Tool 구현: predict_ad_performance
    # ------------------------------------------------------------------
    def _tool_predict(self, args):
        """광고 성과 예측"""
        if self.predictor is None:
            return {"error": "예측 모델이 로드되지 않았습니다."}

        # 필수 파라미터
        required = ["platform", "industry", "country", "campaign_type", "ad_spend", "month"]
        for key in required:
            if key not in args:
                return {"error": f"필수 파라미터 누락: {key}"}

        # 선택적 extra features 분리
        extra = {}
        optional_keys = ["is_retargeting", "is_video", "target_age_group", "target_gender",
                         "video_length_sec", "is_lookalike"]
        for key in optional_keys:
            if key in args:
                extra[key] = args[key]

        result = self.predictor.predict(
            platform=args["platform"],
            industry=args["industry"],
            country=args["country"],
            ad_spend=float(args["ad_spend"]),
            campaign_type=args["campaign_type"],
            month=int(args["month"]),
            **extra,
        )

        # 결과를 깔끔하게 정리
        pred = result["predictions"]
        hist = result["historical_reference"]

        summary = {
            "input": result["input"],
            "predictions": {},
            "historical": hist,
        }

        for metric in ["ROAS", "CPC", "CPA", "conversions"]:
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

        # ROI 계산
        ad_spend = float(args["ad_spend"])
        if "estimated_revenue" in pred:
            roi = ((pred["estimated_revenue"]["predicted"] - ad_spend) / ad_spend) * 100
            summary["predictions"]["estimated_ROI_percent"] = round(roi, 1)

        return summary

    # ------------------------------------------------------------------
    # Tool 구현: compare_scenarios
    # ------------------------------------------------------------------
    def _tool_compare(self, args):
        """여러 시나리오 비교"""
        scenarios = args.get("scenarios", [])
        if len(scenarios) < 2:
            return {"error": "최소 2개 시나리오가 필요합니다."}

        results = []
        for s in scenarios[:5]:
            label = s.pop("label", f"Scenario {len(results)+1}")
            pred_result = self._tool_predict(s)

            row = {"scenario": label}
            if "error" in pred_result:
                row["error"] = pred_result["error"]
            else:
                preds = pred_result.get("predictions", {})
                row["ROAS"] = preds.get("ROAS", {}).get("predicted", "N/A")
                row["CPC"] = preds.get("CPC", {}).get("predicted", "N/A")
                row["CPA"] = preds.get("CPA", {}).get("predicted", "N/A")
                row["conversions"] = preds.get("conversions", {}).get("predicted", "N/A")
                row["estimated_revenue"] = preds.get("estimated_revenue", {}).get("predicted", "N/A")
                row["estimated_ROI_percent"] = preds.get("estimated_ROI_percent", "N/A")
                row["ROAS_confidence"] = preds.get("ROAS", {}).get("confidence_level", "N/A")

                hist = pred_result.get("historical", {})
                row["historical_samples"] = hist.get("exact_match_count", 0)

            results.append(row)

        # 최고 ROAS 시나리오 찾기
        best_roas = None
        best_label = None
        for r in results:
            roas = r.get("ROAS")
            if isinstance(roas, (int, float)) and (best_roas is None or roas > best_roas):
                best_roas = roas
                best_label = r["scenario"]

        return {
            "comparison": results,
            "best_roas_scenario": best_label,
            "recommendation": f"ROAS 기준 최적 시나리오는 '{best_label}' (ROAS: {best_roas:.2f})입니다." if best_label else None,
        }

    # ------------------------------------------------------------------
    # Tool 구현: get_historical_benchmarks
    # ------------------------------------------------------------------
    def _tool_benchmarks(self, args):
        """과거 벤치마크 통계"""
        if self.benchmark_df is None:
            return {"error": "벤치마크 데이터가 로드되지 않았습니다."}

        df = self.benchmark_df.copy()
        industry = args.get("industry")
        platform = args.get("platform")
        country = args.get("country")

        # 필터링
        mask = df["industry"] == industry
        if platform:
            mask = mask & (df["platform"] == platform)
        if country:
            mask = mask & (df["country"] == country)

        filtered = df[mask]

        if len(filtered) == 0:
            return {"message": f"해당 조건의 데이터가 없습니다.", "sample_count": 0}

        metrics = ["ROAS", "CPC", "CPA", "CTR", "ad_spend", "revenue"]
        benchmarks = {"sample_count": len(filtered), "filters": args}

        for m in metrics:
            if m in filtered.columns:
                col = filtered[m].dropna()
                if len(col) > 0:
                    benchmarks[m] = {
                        "mean": round(float(col.mean()), 2),
                        "median": round(float(col.median()), 2),
                        "std": round(float(col.std()), 2),
                        "min": round(float(col.min()), 2),
                        "max": round(float(col.max()), 2),
                        "q25": round(float(col.quantile(0.25)), 2),
                        "q75": round(float(col.quantile(0.75)), 2),
                    }

        # 플랫폼별 비교 (플랫폼 미지정 시)
        if not platform and "platform" in df.columns:
            plat_stats = {}
            for p in df[mask if not platform else df["industry"] == industry]["platform"].unique():
                p_data = filtered[filtered["platform"] == p] if platform is None else df[(df["industry"] == industry) & (df["platform"] == p)]
                if len(p_data) > 0 and "ROAS" in p_data.columns:
                    plat_stats[p] = {
                        "avg_ROAS": round(float(p_data["ROAS"].mean()), 2),
                        "avg_CPC": round(float(p_data["CPC"].mean()), 2) if "CPC" in p_data.columns else None,
                        "sample_count": len(p_data),
                    }
            benchmarks["platform_comparison"] = plat_stats

        return benchmarks

    # ------------------------------------------------------------------
    # Tool 구현: get_industry_trends
    # ------------------------------------------------------------------
    def _tool_trends(self, args):
        """산업별 트렌드 조회"""
        if self.trends_df is None:
            return {"error": "트렌드 데이터가 로드되지 않았습니다."}

        df = self.trends_df.copy()
        industry = args.get("industry")
        country = args.get("country")

        mask = df["industry"] == industry
        if country:
            mask = mask & (df["country"] == country)

        filtered = df[mask].copy()

        if len(filtered) == 0:
            return {"message": f"해당 산업의 트렌드 데이터가 없습니다.", "sample_count": 0}

        # 월별 평균 트렌드
        if "month" in filtered.columns and "trend_index" in filtered.columns:
            monthly = filtered.groupby("month")["trend_index"].mean().round(1)

            # 최고/최저 월
            peak_month = int(monthly.idxmax())
            low_month = int(monthly.idxmin())

            # 최근 3개월 트렌드 방향
            recent = monthly.tail(3)
            if len(recent) >= 2:
                direction = "상승" if recent.iloc[-1] > recent.iloc[0] else "하락"
            else:
                direction = "불확실"

            return {
                "industry": industry,
                "country": country or "전체",
                "monthly_trend_index": {str(k): v for k, v in monthly.to_dict().items()},
                "peak_month": peak_month,
                "low_month": low_month,
                "recent_direction": direction,
                "overall_avg": round(float(filtered["trend_index"].mean()), 1),
                "sample_count": len(filtered),
                "insight": (
                    f"{industry} 산업은 {peak_month}월에 관심도가 가장 높고, "
                    f"{low_month}월에 가장 낮습니다. "
                    f"최근 추세는 {direction} 방향입니다."
                ),
            }

        return {"message": "트렌드 인덱스 컬럼을 찾을 수 없습니다."}

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------
    def reset_conversation(self):
        """대화 이력 초기화"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.collected_info = {}
        self.tool_results_log = []

    def get_model_info(self):
        """모델 성능 정보 반환 (UI 표시용)"""
        if self.predictor is None:
            return {}

        info = {}
        for target, model_data in self.predictor.models.items():
            info[target] = {
                "model_name": model_data["model_name"],
                "r2_score": round(model_data["r2_score"], 3),
                "mae": round(model_data["mae"], 2),
            }
        return info


# ============================================================================
# CLI 모드 (디버그/테스트용)
# ============================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        print("  .env 파일에 OPENAI_API_KEY=sk-... 를 설정하세요.")
        sys.exit(1)

    agent = AdStrategyAgent(openai_api_key=api_key)
    print("\n=== AdStrategy AI (CLI Mode) ===")
    print("'quit' 입력 시 종료\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        response, tools = agent.chat(user_input)
        print(f"\nAI: {response}\n")

        if tools:
            for t in tools:
                print(f"  [Tool: {t['tool']}] args={json.dumps(t['args'], ensure_ascii=False)[:80]}")
            print()
