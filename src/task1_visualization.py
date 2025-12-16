from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(data: Iterable[str]):
    arr = np.asarray(list(data))

    # Compute frequencies
    unique, counts = np.unique(arr, return_counts=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(unique, counts)
    ax.set_xlabel("Category")
    ax.set_ylabel("Frequency")
    ax.set_title("Category Frequency")

    fig.tight_layout()

    # Save result to plots/
    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "task1_bar_chart.png", dpi=150)

    return fig
