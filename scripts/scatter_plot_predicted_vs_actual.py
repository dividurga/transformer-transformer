"""Generate the paper's predicted-vs-actual reward scatter plots.

Example usage (the paper's figures used --predicted_scale 176.697 for the
wheeled bimanual space and 40.3875 for ViperX/quadruped, which convert the
model's normalized per-step value predictions back to episode reward sums):

    python scripts/scatter_plot_predicted_vs_actual.py \\
        --summaries viperx runs/a/hardware_opt_summary.zarr runs/b/hardware_opt_summary.zarr \\
        --predicted_scale viperx 40.3875 \\
        --summaries quadruped runs/c/hardware_opt.zarr runs/d/hardware_opt.zarr \\
        --predicted_scale quadruped 40.3875 \\
        --output_root hardware_opt_plots/
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns

from t2.eval.utils import summarize_rollout

# =============================================================================
# Plot Configuration (adapted from scatter_plot_control_performance.py)
# =============================================================================

PLOT_CONFIG = {
    "font": {
        "family": "MonoLisa Nerd Font Mono",
        "tick_size": 9,
        "label_size": 12,
        "title_size": 14,
    },
    "scatter": {
        "linewidths": 0.5,
        "size": 40,
        "alpha": 0.7,
        "marker": "o",
        "edgecolor": "black",
    },
    "figure": {
        "dpi": 300,
    },
}


def setup_plot_style():
    """Configure matplotlib/seaborn style for publication-quality plots."""
    sns.set_style("whitegrid", {"font.family": PLOT_CONFIG["font"]["family"]})
    plt.rcParams.update(
        {
            "font.family": PLOT_CONFIG["font"]["family"],
            "font.size": PLOT_CONFIG["font"]["tick_size"],
            "axes.labelsize": PLOT_CONFIG["font"]["label_size"],
            "axes.titlesize": PLOT_CONFIG["font"]["title_size"],
            "xtick.labelsize": PLOT_CONFIG["font"]["tick_size"],
            "ytick.labelsize": PLOT_CONFIG["font"]["tick_size"],
        }
    )


def plot_predicted_vs_actual(design_space_zarr_paths, predicted_scales, output_root):
    os.makedirs(output_root, exist_ok=True)
    setup_plot_style()

    for design_space, zarr_path_list in design_space_zarr_paths.items():
        per_zarr_dfs = []
        predicted_scale = predicted_scales.get(design_space, 1.0)

        for zarr_path in zarr_path_list:
            stats = summarize_rollout(zarr_path, reduce=False, use_cache=True)
            run_df = pd.DataFrame(
                {
                    "predicted_value": stats["hardware_meta/predicted_value"]
                    * predicted_scale,
                    "actual_value": stats["hardware_meta/actual_value"],
                    "hardware_seed": stats["hardware_meta/seed"],
                }
            )
            # Average over rollouts sharing the same hardware seed within this zarr
            run_df = (
                run_df.groupby("hardware_seed")[["predicted_value", "actual_value"]]
                .mean()
                .reset_index()
            )
            per_zarr_dfs.append(run_df)

        df = pd.concat(per_zarr_dfs, ignore_index=True)
        df.to_csv(
            os.path.join(output_root, f"predicted_vs_actual_{design_space}.csv"),
            index=False,
        )

        print(f"\n{'=' * 60}")
        print(f"Design space: {design_space}")
        print(f"N = {len(df)}")
        print(
            f"Predicted value: mean={df['predicted_value'].mean():.4f}, std={df['predicted_value'].std():.4f}"
        )
        print(
            f"Actual value:    mean={df['actual_value'].mean():.4f}, std={df['actual_value'].std():.4f}"
        )

        # Pearson correlation
        pearson_r, pearson_p = st.pearsonr(df["predicted_value"], df["actual_value"])
        print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4e}")

        # Linear regression
        slope, intercept, r_value, p_value, std_err = st.linregress(
            df["predicted_value"], df["actual_value"]
        )
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-value: {r_value:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"Std error: {std_err:.4f}")

        fig = plt.figure(figsize=(3, 2.5))
        ax = fig.add_subplot(1, 1, 1)

        # Scatter with regression
        g = sns.regplot(
            x="predicted_value",
            y="actual_value",
            data=df,
            scatter_kws={"alpha": 0.1, "marker": "o", "s": 4, "linewidths": 0.5},
            ax=ax,
        )

        g.set_xlabel(
            "Predicted Value",
            fontsize=PLOT_CONFIG["font"]["label_size"],
        )
        g.set_ylabel(
            "Actual Value",
            fontsize=PLOT_CONFIG["font"]["label_size"],
        )
        g.tick_params(
            axis="both",
            which="major",
            labelsize=PLOT_CONFIG["font"]["tick_size"],
        )

        plt.tight_layout()

        for ext in ["pdf", "svg"]:
            plt.savefig(
                os.path.join(output_root, f"predicted_vs_actual_{design_space}.{ext}"),
                dpi=PLOT_CONFIG["figure"]["dpi"],
                bbox_inches="tight",
            )
        plt.close("all")

    print(f"\nSaved plots to {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--summaries",
        action="append",
        nargs="+",
        metavar=("DESIGN_SPACE", "ZARR_PATH"),
        required=True,
        help="Design space name followed by one or more rollout summary zarr paths; repeat per design space",
    )
    parser.add_argument(
        "--predicted_scale",
        action="append",
        nargs=2,
        metavar=("DESIGN_SPACE", "SCALE"),
        default=[],
        help="Multiply a design space's predicted values by SCALE to undo value "
        "normalization (see the module docstring for the paper's values)",
    )
    parser.add_argument("--output_root", default="hardware_opt_plots/")
    args = parser.parse_args()

    design_space_zarr_paths = {}
    for entry in args.summaries:
        if len(entry) < 2:
            parser.error(
                "--summaries requires a design space name followed by at least one zarr path"
            )
        design_space_zarr_paths[entry[0]] = entry[1:]
    predicted_scales = {name: float(scale) for name, scale in args.predicted_scale}
    plot_predicted_vs_actual(design_space_zarr_paths, predicted_scales, args.output_root)


if __name__ == "__main__":
    main()
