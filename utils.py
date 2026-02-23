# -*- coding: utf-8 -*-
"""
공유 유틸리티 함수
==================
데이터 로딩, ROI 계산, 인코딩 설정 등
여러 모듈에서 중복되던 로직을 중앙 집중.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from typing import Optional

import pandas as pd

from config import ENRICHED_DATA_PATH, BASE_DATA_PATH

logger = logging.getLogger(__name__)


def configure_windows_encoding() -> None:
    """Windows 콘솔 UTF-8 인코딩 설정.

    모든 모듈에서 이 함수를 호출하여 인코딩 설정을 통일한다.
    non-Windows 환경에서는 no-op.
    """
    try:
        if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_ads_data(
    primary_path: str = ENRICHED_DATA_PATH,
    fallback_path: str = BASE_DATA_PATH,
    parse_dates: bool = True,
) -> Optional[pd.DataFrame]:
    """보강 데이터 우선, 없으면 원본 폴백, 둘 다 없으면 None."""
    for path in (primary_path, fallback_path):
        if path and os.path.exists(path):
            logger.info("데이터 로드: %s", path)
            df = pd.read_csv(path)
            if parse_dates and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
    logger.warning("데이터 파일 없음: %s, %s", primary_path, fallback_path)
    return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """0, NaN, inf 등 비정상 산술 방지. 피연산자가 유한하지 않으면 default 반환."""
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def compute_roi_pct(revenue: float, spend: float) -> float:
    """ROI(%) = (revenue - spend) / spend * 100. spend=0이면 0.0 반환."""
    return safe_divide(revenue - spend, spend) * 100
