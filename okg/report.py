"""Notebook charts. Categorical slots follow a validated colorblind-safe order (blue, orange,
aqua, yellow; contrast relief = legend + the metrics table printed next to every chart)."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def shap_bar(imp: pd.DataFrame, top: int = 15):
    import matplotlib.pyplot as plt
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 0.8), facecolor=SURFACE)
    ax.barh(d["feature"], d["mean_abs_shap"], color=SERIES[0], height=0.6)
    _style(ax)
    ax.set_title("LightGBM feature importance (mean |SHAP|, log-odds)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig


def metric_by_group(metrics: pd.DataFrame, metric: str, prefix: str = "source=", systems=None):
    """Grouped horizontal bars: one bar per system for each dataset (<= 4 systems)."""
    import matplotlib.pyplot as plt
    m = metrics[metrics["group"].str.startswith(prefix)].copy()
    m["dataset"] = m["group"].str[len(prefix):]
    systems = systems or list(m["system"].unique())[:4]
    piv = m.pivot_table(index="dataset", columns="system", values=metric).reindex(columns=systems)
    piv = piv.dropna(how="all")
    n, k = len(piv), len(systems)
    fig, ax = plt.subplots(figsize=(8, 0.42 * n * max(1, k / 2) + 1.2), facecolor=SURFACE)
    h = 0.8 / k
    y = np.arange(n)
    for i, s in enumerate(systems):
        ax.barh(y + (k - 1 - i) * h - 0.4 + h / 2, piv[s].fillna(0).to_numpy(), height=h * 0.85,
                color=SERIES[i], label=s)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index)
    ax.set_xlim(0, 1)
    _style(ax)
    ax.set_title(f"{metric} per dataset (test role)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    fig.tight_layout()
    return fig, piv
