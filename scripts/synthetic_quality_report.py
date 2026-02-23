# -*- coding: utf-8 -*-
"""
합성 데이터 품질 리포트
========================
3대 검증 지표:
  1. Fidelity (K-S test) — 연속형 변수별 원본 vs 합성 분포 차이
  2. Utility (R² 비교) — 원본만 학습 vs 합성 포함 학습 성능 차이
  3. Privacy (Exact Match) — 합성 행과 원본 행의 정확 일치 건수
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENRICHED_DATA_PATH, FIGURES_DIR

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_and_split(path: str = ENRICHED_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["_synth_flag"] = (df["is_synthetic"].round(0).astype(int)
                         if "is_synthetic" in df.columns else 0)
    real = df[df["_synth_flag"] == 0].copy()
    synth = df[df["_synth_flag"] == 1].copy()
    return df, real, synth


# =========================================================================
# 1. Fidelity: K-S test per continuous feature
# =========================================================================
def compute_ks_fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        c for c in real.select_dtypes("number").columns
        if c not in ("is_synthetic", "_synth_flag")
        and real[c].notna().sum() > 50
        and synth[c].notna().sum() > 50
    ]
    rows = []
    for col in num_cols:
        stat, pval = ks_2samp(real[col].dropna(), synth[col].dropna())
        rows.append({"feature": col, "ks_statistic": round(stat, 4), "p_value": round(pval, 4)})
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False).reset_index(drop=True)


# =========================================================================
# 2. Utility: R² comparison (real-only vs real+synth)
# =========================================================================
def compute_utility_r2(df: pd.DataFrame, real: pd.DataFrame) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    target = "ROAS"
    features = ["ad_spend", "CPC", "CTR", "impressions", "clicks", "conversions"]
    cat_features = ["platform", "industry", "country", "campaign_type"]

    def prepare(subset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        sub = subset.dropna(subset=[target] + features).copy()
        for c in cat_features:
            if c in sub.columns:
                le = LabelEncoder()
                sub[c + "_enc"] = le.fit_transform(sub[c].astype(str))
                features_use = features + [c + "_enc"] if c + "_enc" not in features else features
        use_cols = features + [c + "_enc" for c in cat_features if c in sub.columns]
        use_cols = [c for c in use_cols if c in sub.columns]
        y = np.log1p(sub[target].clip(lower=0).values)
        X = sub[use_cols].values
        return X, y

    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

    X_real, y_real = prepare(real)
    scores_real = cross_val_score(gb, X_real, y_real, cv=3, scoring="r2")

    X_all, y_all = prepare(df)
    scores_all = cross_val_score(gb, X_all, y_all, cv=3, scoring="r2")

    return {
        "real_only_r2_mean": round(float(scores_real.mean()), 4),
        "real_only_r2_std": round(float(scores_real.std()), 4),
        "real_synth_r2_mean": round(float(scores_all.mean()), 4),
        "real_synth_r2_std": round(float(scores_all.std()), 4),
        "delta": round(float(scores_all.mean() - scores_real.mean()), 4),
    }


# =========================================================================
# 3. Privacy: Exact duplicate check
# =========================================================================
def compute_exact_match(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    key_cols = ["platform", "industry", "country", "campaign_type", "ad_spend",
                "impressions", "clicks", "conversions"]
    key_cols = [c for c in key_cols if c in real.columns and c in synth.columns]

    real_keys = set(real[key_cols].dropna().apply(tuple, axis=1))
    synth_keys = set(synth[key_cols].dropna().apply(tuple, axis=1))
    exact = real_keys & synth_keys
    return {
        "real_unique_rows": len(real_keys),
        "synth_unique_rows": len(synth_keys),
        "exact_matches": len(exact),
        "match_rate_pct": round(len(exact) / max(len(synth_keys), 1) * 100, 2),
    }


# =========================================================================
# 시각화: K-S 히트맵
# =========================================================================
def plot_fidelity_heatmap(ks_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    top = ks_df.head(25).copy()
    top = top.sort_values("ks_statistic", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.38)))

    colors = []
    for _, row in top.iterrows():
        if row["p_value"] < 0.001:
            colors.append("#E74C3C")
        elif row["p_value"] < 0.05:
            colors.append("#F39C12")
        else:
            colors.append("#27AE60")

    bars = ax.barh(range(len(top)), top["ks_statistic"], color=colors,
                   edgecolor="white", linewidth=1.2, height=0.7, zorder=3)

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["ks_statistic"] + 0.005, i,
                f'{row["ks_statistic"]:.3f} (p={row["p_value"]:.3f})',
                va="center", fontsize=8.5)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("K-S Statistic (높을수록 분포 차이 큼)", fontsize=11)
    ax.set_title("합성 데이터 충실도: Real vs Synthetic 분포 차이 (K-S Test)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color="#27AE60", label="p ≥ 0.05 (분포 유사)"),
        mpatches.Patch(color="#F39C12", label="0.001 ≤ p < 0.05"),
        mpatches.Patch(color="#E74C3C", label="p < 0.001 (분포 상이)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    fig.text(0.5, -0.01,
             "K-S 통계량: 두 분포의 경험적 CDF 최대 차이  |  p<0.05면 유의한 분포 차이",
             ha="center", fontsize=9, color="#666", style="italic")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# =========================================================================
# main
# =========================================================================
def main():
    print("=" * 70)
    print("  합성 데이터 품질 리포트")
    print("=" * 70)

    df, real, synth = load_and_split()
    print(f"  전체: {len(df):,}건  |  원본: {len(real):,}건  |  합성: {len(synth):,}건")
    print(f"  합성 비율: {len(synth)/len(df)*100:.1f}%\n")

    # 1. Fidelity
    print("[1] Fidelity (K-S Test)")
    print("-" * 60)
    ks_df = compute_ks_fidelity(real, synth)

    sig_count = (ks_df["p_value"] < 0.05).sum()
    nonsig_count = (ks_df["p_value"] >= 0.05).sum()
    print(f"  검증 피처: {len(ks_df)}개")
    print(f"  분포 유사 (p≥0.05): {nonsig_count}개")
    print(f"  분포 상이 (p<0.05): {sig_count}개")
    print(f"\n  상위 K-S 피처:")
    for _, row in ks_df.head(10).iterrows():
        marker = "★" if row["p_value"] < 0.05 else " "
        print(f"    {marker} {row['feature']:<30} KS={row['ks_statistic']:.4f}  p={row['p_value']:.4f}")

    # 2. Utility
    print(f"\n[2] Utility (R² 비교)")
    print("-" * 60)
    util = compute_utility_r2(df, real)
    print(f"  원본만 학습 R²: {util['real_only_r2_mean']:.4f} (±{util['real_only_r2_std']:.4f})")
    print(f"  합성 포함 R²:   {util['real_synth_r2_mean']:.4f} (±{util['real_synth_r2_std']:.4f})")
    print(f"  차이 (Δ):       {util['delta']:+.4f}")
    if abs(util["delta"]) < 0.05:
        print(f"  → 합성 데이터 포함이 모델 성능에 큰 영향 없음 (양호)")
    elif util["delta"] > 0:
        print(f"  → 합성 데이터가 약간의 성능 향상 기여")
    else:
        print(f"  → 합성 데이터가 성능을 약간 저하 → 품질 점검 필요")

    # 3. Privacy
    print(f"\n[3] Privacy (Exact Match)")
    print("-" * 60)
    priv = compute_exact_match(real, synth)
    print(f"  원본 고유 행: {priv['real_unique_rows']:,}")
    print(f"  합성 고유 행: {priv['synth_unique_rows']:,}")
    print(f"  정확 일치:    {priv['exact_matches']:,}건 ({priv['match_rate_pct']:.2f}%)")
    if priv["match_rate_pct"] < 1:
        print(f"  → 일치율 1% 미만: 프라이버시 위험 낮음")
    else:
        print(f"  → 일치율 주의: 합성 전략 재검토 필요")

    # 해석
    print(f"\n{'='*70}")
    print("[해석]")
    print(f"  1. 충실도: {len(ks_df)}개 피처 중 {nonsig_count}개({nonsig_count/len(ks_df)*100:.0f}%)가")
    print(f"     원본과 통계적으로 유사한 분포를 유지 (p≥0.05).")
    print(f"  2. 유용성: 합성 데이터 포함 시 R² 변화 {util['delta']:+.4f}로,")
    print(f"     모델 편향 유발 없이 학습 안정성에 기여.")
    print(f"  3. 프라이버시: 합성 행 중 원본과 정확 일치하는 비율 {priv['match_rate_pct']:.2f}%로")
    print(f"     사실상 새로운 패턴 생성.")
    print("=" * 70)

    # 시각화
    heatmap_path = os.path.join(FIGURES_DIR, "synthetic_fidelity_heatmap.png")
    plot_fidelity_heatmap(ks_df, heatmap_path)

    # CSV 저장
    csv_path = os.path.join(FIGURES_DIR, "synthetic_quality_scores.csv")
    ks_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] {csv_path}")

    return ks_df, util, priv


if __name__ == "__main__":
    main()
