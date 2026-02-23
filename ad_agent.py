# -*- coding: utf-8 -*-
"""
AdStrategy AI -- 광고 전략 설계 에이전트 코어
- OpenAI GPT-4o + Function Calling
- AdsPredictor_V2 ML 모델 연동
- 광고주와 대화하며 최적의 광고 전략을 설계
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from openai import OpenAI

from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    PROJECT_ROOT,
    TOOL_CALL_MAX_ITERATIONS,
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp_ad_server import AdTools
from tool_schemas import get_openai_tools

logger = logging.getLogger(__name__)

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
- 타겟 국가/시장 (USA, UK, Germany, Canada, India, UAE, Australia, Korea)
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

**6단계 — 리스크 검증 (비판적 사고)**
전략 추천 후, 반드시 **반대 의견을 스스로 제시**하세요:
- "이 전략의 리스크는 무엇인가?"
- "예산이 부족할 때 어떤 문제가 생기는가?"
- "경쟁사가 같은 전략을 쓰면?"
광고주의 선택이 위험하다고 판단되면, 구체적 데이터를 들어 **반론**하세요.
예: "TikTok에 올인하시겠다면, Google Ads 대비 CPA가 1.8배 높고 예측 신뢰도가 낮아(MAE 6.8) 리스크가 큽니다."

**7단계 — A/B 테스트 설계 제안**
전략 추천 마지막에, 가설 검증을 위한 A/B 테스트 설계를 1~2문장으로 제안하세요.
예: "이 가설을 검증하려면, 2주간 예산의 30%를 TikTok Video로, 70%를 Google Search로 분배한 뒤 ROAS를 비교하세요."

## 응답 규칙
- **한국어**로 대화하세요 (광고주가 영어를 쓰면 영어로 전환).
- 숫자와 데이터는 **구체적으로** 제시하세요 (예: "ROAS 4.9 예상, 월 $5,000 투자 시 약 $24,500 매출 기대").
- 예측 결과를 전달할 때는 **신뢰도(High/Medium/Low)**와 **근거 샘플 수**를 반드시 언급하세요. 신뢰도는 각 지표별 모델의 R² 기준(R²>0.5=High, >0.3=Medium)이며, 실제 성과를 보장하지 않습니다.
- 추천을 할 때는 **왜 그 플랫폼/전략이 좋은지 데이터 근거**를 설명하세요.
- 불확실할 때는 솔직하게 말하고, 추가 정보를 요청하세요.
- 대화 처음에는 친근한 인사와 함께 "어떤 광고를 계획하고 계신지" 물어보세요.
- **투명성**: 모든 예측에는 "이 예측은 honest R²≈0.35 모델 기반이며, 개별 수치보다 상대 비교에 활용하세요"라는 단서를 포함하세요. 플랫폼 추천이 학습 데이터 상 TikTok 우위 패턴을 반영할 수 있음을 알고, "실제 성과는 A/B 테스트로 검증해야 한다"는 점을 강조하세요.
- **비판적 사고**: 전략 추천 후 반드시 리스크 1~2가지를 스스로 언급하세요.
- **A/B 테스트**: 최종 추천에는 "이 가설을 검증하려면 이렇게 A/B 테스트하세요"를 1문장으로 추가하세요.

## 사용 가능한 값
- 플랫폼: Google Ads, Meta Ads, TikTok Ads
- 산업: Fintech, EdTech, Healthcare, SaaS, E-commerce
- 국가: USA, UK, Germany, Canada, India, UAE, Australia, Korea
- 캠페인 유형: Search, Video, Shopping, Display
- 타겟 연령: 18-24, 25-34, 35-44, 45-54, 55+
"""

# ============================================================================
# Tool 정의 (공통 스키마에서 생성 -- DRY)
# ============================================================================

TOOLS = get_openai_tools()

# ============================================================================
# 에이전트 클래스
# ============================================================================

class AdStrategyAgent:
    """광고 전략 설계 AI 에이전트 (GPT-4o + Function Calling + AdTools)."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o") -> None:
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        self.ad_tools = AdTools()
        self.predictor = self.ad_tools.predictor

        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_results_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 메인 대화 루프
    # ------------------------------------------------------------------
    def chat(self, user_message: str) -> tuple[str, list[dict[str, Any]]]:
        """사용자 메시지를 처리하고 (응답 텍스트, Tool 호출 결과 목록)을 반환."""
        self.messages.append({"role": "user", "content": user_message})
        tool_results: list[dict[str, Any]] = []

        for _ in range(TOOL_CALL_MAX_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )

            if not response.choices:
                logger.error("OpenAI returned empty choices")
                break
            msg = response.choices[0].message

            if not msg.tool_calls:
                assistant_text = msg.content or ""
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.tool_results_log = tool_results
                return assistant_text, tool_results

            self.messages.append(msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("JSON parse failed for tool %s: %s", fn_name, exc)
                    fn_args = {}

                result = self._execute_tool(fn_name, fn_args)
                tool_results.append({"tool": fn_name, "args": fn_args, "result": result})
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        self.tool_results_log = tool_results
        return "분석을 완료했습니다. 추가 질문이 있으시면 말씀해주세요.", tool_results

    # ------------------------------------------------------------------
    # Tool 실행기 — 모든 로직은 AdTools 에 위임 (DRY)
    # ------------------------------------------------------------------
    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Tool 이름에 따라 AdTools 메서드를 호출."""
        try:
            dispatch: dict[str, Any] = {
                "predict_ad_performance": lambda: self.ad_tools.predict(**arguments),
                "compare_scenarios": lambda: self.ad_tools.compare(arguments.get("scenarios", [])),
                "get_historical_benchmarks": lambda: self.ad_tools.benchmarks(**arguments),
                "get_industry_trends": lambda: self.ad_tools.trends(**arguments),
            }
            handler = dispatch.get(tool_name)
            if handler is None:
                return {"error": f"Unknown tool: {tool_name}"}
            return handler()
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool_name)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------
    def reset_conversation(self) -> None:
        """대화 이력 초기화"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_results_log = []

    def get_model_info(self) -> dict[str, dict[str, Any]]:
        """모델 성능 정보 반환 (UI 표시용)"""
        if self.predictor is None:
            return {}

        info: dict[str, dict[str, Any]] = {}
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
