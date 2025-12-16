from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_scatter_plot(df: pd.DataFrame):
    """Create a clear, simple Seaborn scatter plot for columns 'x' and 'y'."""
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'x' and 'y' columns.")

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="x", y="y", ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Scatter Plot of x vs y")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "task2_scatter_seaborn.png", dpi=150)

    return fig
