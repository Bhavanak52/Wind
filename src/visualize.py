from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


MODEL_NAMES = ["LSTM", "CNN-LSTM", "CNN-GRU"]
MODEL_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
}


def plot_decomposition(df: pd.DataFrame, output_path: str | Path, max_points: int = 288) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.iloc[:max_points].copy()
    ewt_mode_cols = sorted([c for c in plot_df.columns if c.startswith("ewt_mode_")])
    if ewt_mode_cols:
        n_panels = 1 + min(5, len(ewt_mode_cols))
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.2 * n_panels + 2), sharex=True)
        if n_panels == 1:
            axes = [axes]

        axes[0].plot(plot_df["timestamp"], plot_df["wind_speed_40m"], color="black", linewidth=1.2)
        axes[0].set_title("Original 40m Wind Speed")
        axes[0].set_ylabel("m/s")

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for idx, mode_col in enumerate(ewt_mode_cols[: n_panels - 1], start=1):
            axes[idx].plot(plot_df["timestamp"], plot_df[mode_col], color=palette[(idx - 1) % len(palette)])
            axes[idx].set_title(mode_col.replace("_", " ").upper())

        axes[-1].set_xlabel("Time")
    else:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(plot_df["timestamp"], plot_df["wind_speed_40m"], color="black", linewidth=1.2)
        axes[0].set_title("Original 40m Wind Speed")
        axes[0].set_ylabel("m/s")

        axes[1].plot(plot_df["timestamp"], plot_df["imf_trend"], color="#1f77b4")
        axes[1].set_title("IMF Trend")

        axes[2].plot(plot_df["timestamp"], plot_df["imf_osc"], color="#ff7f0e")
        axes[2].set_title("IMF Oscillation")

        axes[3].plot(plot_df["timestamp"], plot_df["imf_noise"], color="#d62728")
        axes[3].set_title("IMF Noise")
        axes[3].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_decision_timeline(timestamps: Iterable, actions: Iterable[int], output_path: str | Path, max_points: int = 288) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timeline_df = pd.DataFrame({
        "timestamp": pd.to_datetime(list(timestamps)),
        "action": list(actions),
    }).iloc[:max_points]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    colors = [MODEL_COLORS[int(action)] for action in timeline_df["action"]]
    ax.bar(timeline_df["timestamp"], [1] * len(timeline_df), color=colors, width=0.0032)
    ax.set_title("DRL Master Selector Decision Timeline (24h sample)")
    ax.set_ylabel("Chosen expert")
    ax.set_yticks([])
    ax.set_xlabel("Time")

    legend_handles = [Patch(color=MODEL_COLORS[idx], label=name) for idx, name in enumerate(MODEL_NAMES)]
    ax.legend(handles=legend_handles, loc="upper right", ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(
    results_df: pd.DataFrame,
    output_path: str | Path,
    max_points: int = 288,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = results_df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    plot_df = plot_df.iloc[:max_points]

    actual = plot_df["actual_wind_speed_40m"].to_numpy(dtype=float)
    predicted = plot_df["pred_drl_hybrid"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(np.square(actual - predicted)))) if len(plot_df) else float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(plot_df["timestamp"], actual, label="Actual", color="black", linewidth=1.4)
    axes[0].plot(plot_df["timestamp"], predicted, label="DRL-Hybrid Predicted", color="#d62728", linewidth=1.2)
    axes[0].set_title(f"Actual vs Predicted 40m Wind Speed (RMSE={rmse:.3f})")
    axes[0].set_ylabel("m/s")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(actual, predicted, s=18, alpha=0.65, color="#1f77b4")
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    axes[1].set_title("Prediction Scatter Check")
    axes[1].set_xlabel("Actual wind speed (m/s)")
    axes[1].set_ylabel("Predicted wind speed (m/s)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metrics_table(metrics_records: List[dict], output_csv: str | Path) -> pd.DataFrame:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(output_csv, index=False)
    return metrics_df
