from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_soh_series(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    soh_cols = [col for col in df.columns if col.startswith("soh_")]
    if not soh_cols:
        raise ValueError(f"No soh_* columns found in {csv_path}")
    soh = df[soh_cols].to_numpy(dtype=float)
    return soh


def weekly_statistics(soh: np.ndarray, dt_seconds: int = 3600) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps_per_week = int((7 * 24 * 3600) / dt_seconds)
    if steps_per_week <= 0:
        raise ValueError("dt_seconds must be positive")

    sample_indices = list(range(0, len(soh), steps_per_week))
    medians = []
    q1 = []
    q3 = []
    years = []

    for idx in sample_indices:
        current = soh[idx]
        medians.append(float(np.percentile(current, 50)))
        q1.append(float(np.percentile(current, 25)))
        q3.append(float(np.percentile(current, 75)))
        years.append((idx * dt_seconds) / (365.0 * 24 * 3600))

    return np.array(years), np.array(medians), np.array(q1), np.array(q3)


def limit_to_years(soh: np.ndarray, dt_seconds: int = 3600, max_years: float = 5.0) -> np.ndarray:
    max_steps = int((max_years * 365.0 * 24 * 3600) / dt_seconds)
    if max_steps <= 0:
        raise ValueError("max_years must be positive")
    return soh[:max_steps]


def plot_soh_three(
    baseline_csv: str,
    ppo_csv: str,
    ppo_ttm_csv: str,
    output_png: str = "soh_three_strategies.png",
) -> None:
    series = {
        "Baseline": load_soh_series(baseline_csv),
        "PPO-RL": load_soh_series(ppo_csv),
        "PPO+TTM": load_soh_series(ppo_ttm_csv),
    }

    colors = {
        "Baseline": "#1f77b4",
        "PPO-RL": "#ff7f0e",
        "PPO+TTM": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    for label, soh in series.items():
        soh = limit_to_years(soh, dt_seconds=3600, max_years=5.0)
        years, median, lower, upper = weekly_statistics(soh)
        ax.fill_between(years, lower, upper, color=colors[label], alpha=0.25, linewidth=0)
        ax.plot(years, median, label=label, color=colors[label], linewidth=2)

    ax.set_xlabel("Years")
    ax.set_ylabel(r"State of Health $\bar{Q}_v(t)/\bar{Q}_v(0)$")
    ax.set_title("SoH across Years for Simulated Fleets")
    ax.set_xlim(0, 5)
    ax.set_ylim(0.68, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)

    print(f"saved {output_png}")
    for label, soh in series.items():
        soh = limit_to_years(soh, dt_seconds=3600, max_years=5.0)
        print(
            label,
            "median_final=", float(np.percentile(soh[-1], 50)),
            "iqr_final=", (
                float(np.percentile(soh[-1], 75)),
                float(np.percentile(soh[-1], 25)),
            ),
        )


if __name__ == "__main__":
    plot_soh_three(
        baseline_csv="output_baseline_fix1.csv",
        ppo_csv="output_ppo_fix1.csv",
        ppo_ttm_csv="output_ppo_ttm_fix1.csv",
    )