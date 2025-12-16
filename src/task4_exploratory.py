from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plots_dir() -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform EDA: descriptive stats, outlier detection (boxplot), correlation analysis."""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")

    out_dir = _plots_dir()

    # 1) Descriptive statistics
    desc = df.describe(include="all").T
    try:
        modes = df.mode(numeric_only=False).iloc[0]
        desc["mode"] = modes
    except Exception:
        desc["mode"] = np.nan
    desc.to_csv(out_dir / "task4_descriptive_statistics.csv")

    # 2) Boxplot for outliers (numeric only)
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([num_df[col].dropna().values for col in num_df.columns], labels=list(num_df.columns))
        ax.set_title("Box Plot for Outlier Detection")
        ax.set_xlabel("Variables")
        ax.set_ylabel("Values")
        fig.tight_layout()
        fig.savefig(out_dir / "task4_boxplot.png", dpi=150)
        plt.close(fig)

    # 3) Correlation + heatmap with annotations
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        _save_correlation_heatmap(corr, out_dir / "task4_correlation_heatmap.png")
    else:
        corr = pd.DataFrame()

    explanation = []
    explanation.append("Descriptive statistics saved to task4_descriptive_statistics.csv.")
    if not num_df.empty:
        explanation.append("Box plot saved to task4_boxplot.png; values beyond whiskers suggest potential outliers.")
    if not corr.empty:
        explanation.append("Correlation heatmap saved to task4_correlation_heatmap.png; coefficients near 1/-1 indicate strong linear relationships.")
    (out_dir / "task4_findings.txt").write_text("\n".join(explanation), encoding="utf-8")

    return {"descriptive_stats": desc, "correlation": corr}
