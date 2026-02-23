# -*- coding: utf-8 -*-
"""
한국 합성 데이터 포함 데이터로 예측 모델 재학습 후 pkl 저장.
실행: 프로젝트 루트에서
  python scripts/train_predictor_with_korea.py
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

os.chdir(root)

from config import (
    BASE_DATA_PATH,
    ENRICHED_DATA_WITH_KOREA_PATH,
    MODELS_DIR,
)

from ads_predictor_v2 import AdsPredictor_V2

def main():
    if not os.path.exists(ENRICHED_DATA_WITH_KOREA_PATH):
        print(f"[ERROR] 한국 포함 데이터 없음: {ENRICHED_DATA_WITH_KOREA_PATH}")
        print("  먼저 scripts/add_korea_segment.py 를 실행하세요.")
        sys.exit(1)

    print("한국 포함 데이터로 예측기 학습 및 저장")
    print("=" * 60)
    predictor = AdsPredictor_V2(
        data_path=ENRICHED_DATA_WITH_KOREA_PATH,
        fallback_path=BASE_DATA_PATH,
        use_temporal_split=True,
        log_transform_target=True,
        tune_hyperparams=False,
        exclude_leakage=True,
        use_gpu=True,  # RTX 등 CUDA GPU 있으면 학습 가속. 오류 나면 use_gpu=False 로 변경
    )
    predictor._save_models(MODELS_DIR)

    # Korea 예측 한 번 실행해서 동작 확인
    print("\n[Korea 예측 검증]")
    result = predictor.predict(
        platform="Meta Ads",
        industry="E-commerce",
        country="Korea",
        ad_spend=5000,
        campaign_type="Video",
        month=6,
    )
    pred = result["predictions"].get("ROAS", {})
    print(f"  Korea / Meta / E-commerce / $5000 / Video / 6월 → ROAS 예측: {pred.get('predicted', 'N/A')}")
    print("\n완료. models/predictor_v2.pkl 이 갱신되었습니다.")

if __name__ == "__main__":
    main()
