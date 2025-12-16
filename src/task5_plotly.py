from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


def create_interactive_plotly(df: pd.DataFrame):
    """Create an interactive Plotly scatter plot for columns 'x' and 'y'."""
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'x' and 'y' columns.")

    fig = px.scatter(
        df,
        x="x",
        y="y",
        title="Interactive Scatter Plot (Plotly)",
        labels={"x": "x", "y": "y"},
    )

    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_dir / "task5_plotly.html", include_plotlyjs="cdn")

    return fig
