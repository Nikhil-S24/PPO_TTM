import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def main():
    ppo = pd.read_csv("output_ppo_fix1.csv")["total_revenue"]
    ppo_ttm = pd.read_csv("output_ppo_ttm_fix1.csv")["total_revenue"]

    years = np.linspace(0, 5, len(ppo))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(years, ppo, label="PPO-RL", linewidth=2, color="#1f77b4")
    ax.plot(years, ppo_ttm, label="PPO+TTM", linewidth=2, color="#ff7f0e")

    ax.fill_between(
        years,
        ppo,
        ppo_ttm,
        where=(ppo_ttm >= ppo),
        color="#ff7f0e",
        alpha=0.15,
        interpolate=True,
        label="TTM gain",
    )

    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.set_title("Cumulative Revenue over 5 Years")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(7, 7))
    ax.grid(alpha=0.35)
    ax.legend(loc="upper left")

    final_gain = float(ppo_ttm.iloc[-1] - ppo.iloc[-1])

    # Add a zoomed inset so the PPO+TTM improvement is visibly clear.
    axins = inset_axes(ax, width="43%", height="48%", loc="lower right", borderpad=1.0)
    axins.plot(years, ppo, linewidth=1.6, color="#1f77b4")
    axins.plot(years, ppo_ttm, linewidth=1.6, color="#ff7f0e")

    x1, x2 = 4.0, 5.0
    tail_ppo = ppo.iloc[-200:]
    tail_ttm = ppo_ttm.iloc[-200:]
    y_min = min(tail_ppo.min(), tail_ttm.min())
    y_max = max(tail_ppo.max(), tail_ttm.max())
    pad = max(50.0, (y_max - y_min) * 0.2)

    axins.set_xlim(x1, x2)
    axins.set_ylim(y_min - pad, y_max + pad)
    axins.grid(alpha=0.25)
    axins.set_title("Zoom (Years 4-5)", fontsize=8)
    axins.tick_params(labelsize=7)

    ax.annotate(
        f"Final gain: ${final_gain:,.2f}",
        xy=(years[-1], ppo_ttm.iloc[-1]),
        xytext=(2.8, ppo_ttm.iloc[-1] - 0.085 * (ppo_ttm.max() - ppo_ttm.min())),
        arrowprops={"arrowstyle": "->", "lw": 1},
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig("revenue_ppo_vs_ttm_highlight.png", dpi=220)

    print("saved revenue_ppo_vs_ttm_highlight.png")
    print("final PPO", float(ppo.iloc[-1]))
    print("final PPO+TTM", float(ppo_ttm.iloc[-1]))
    print("final gain", final_gain)


if __name__ == "__main__":
    main()
