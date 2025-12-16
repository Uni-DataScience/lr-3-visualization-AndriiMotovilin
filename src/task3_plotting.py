from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _plots_dir() -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_1d(values: np.ndarray):
    """1D line plot for a sequence of values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("1D Line Plot")
    fig.tight_layout()
    fig.savefig(_plots_dir() / "task3_1d_line.png", dpi=150)
    plt.close(fig)


def plot_2d(x: np.ndarray, y: np.ndarray):
    """2D scatter plot showing relationship between x and y."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Scatter Plot")
    fig.tight_layout()
    fig.savefig(_plots_dir() / "task3_2d_scatter.png", dpi=150)
    plt.close(fig)


def plot_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """3D scatter plot using Matplotlib Axes3D."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Scatter Plot")
    fig.tight_layout()
    fig.savefig(_plots_dir() / "task3_3d_scatter.png", dpi=150)
    plt.close(fig)
