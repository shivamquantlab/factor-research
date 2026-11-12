"""Fixed-seed synthetic panel illustrating cross-sectional factor returns."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cross_section import factor_slope, long_short_return


def run(output_dir: Path, seed: int = 31) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    periods, assets = 120, 40
    cumulative = [1.0]
    factor_returns = []
    slopes = []
    for _ in range(periods):
        scores = rng.normal(size=assets)  # observed at t before forward returns
        next_returns = 0.0002 + 0.003 * scores + rng.normal(0, 0.012, size=assets)
        slopes.append(factor_slope(scores, next_returns))
        factor_returns.append(long_short_return(scores, next_returns, quantile=0.2))
        cumulative.append(cumulative[-1] * (1.0 + factor_returns[-1]))
    summary = {
        "seed": seed,
        "periods": periods,
        "assets_per_period": assets,
        "mean_cross_sectional_slope": float(np.mean(slopes)),
        "mean_long_short_return": float(np.mean(factor_returns)),
        "synthetic_terminal_wealth": float(cumulative[-1]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(cumulative, color="#b45309")
    ax.set(title="Synthetic equal-weight factor spread", xlabel="Synthetic period", ylabel="Growth of 1 unit")
    ax.text(0.01, 0.98, "SYNTHETIC • gross of costs • not investment performance",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "factor_spread.png", dpi=150)
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    run(args.output_dir, seed=args.seed)
    print(f"Wrote synthetic factor report to {args.output_dir}")


if __name__ == "__main__":
    main()

