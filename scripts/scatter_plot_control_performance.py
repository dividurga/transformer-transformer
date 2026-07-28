"""Generate the paper's control-performance scatter plots.

Example usage:

    python scripts/scatter_plot_control_performance.py quadruped \\
        --rl_validation runs/a/hardware_opt.zarr runs/b/hardware_opt.zarr \\
        --self_validation runs/c/hardware_opt_summary.zarr runs/d/hardware_opt_summary.zarr \\
        --labels seeds=128 seeds=64

    python scripts/scatter_plot_control_performance.py wheeled_bimanual \\
        --mink runs/choice00-ctrl-eval.zarr runs/choice01-ctrl-eval.zarr \\
        --small runs/a/ctrl_eval_summary.zarr runs/b/ctrl_eval_summary.zarr \\
        --large runs/c/ctrl_eval_summary.zarr runs/d/ctrl_eval_summary.zarr \\
        --choices 00 01 \\
        --output_root wheeled_bimanual_control_plots/
"""

import argparse
import os
from t2.eval.utils import summarize_rollout
import pandas as pd
import zarr
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

# =============================================================================
# Plot Configuration (adapted from plot_hardware_opt.py)
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
        "dpi": 200,
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


def plot_quadruped_control_performance(num_seeds_approach_zarr_paths):
    # build a shared dataframe, which stores for each hardware seed
    # the performance in one domain (rl_validation/reward) along with
    # performance in the other domain (self_validation/reward) in one row
    # and the hardware seed in one column
    data_dict = {
        "num_seeds": [],
        "rl_validation_reward": [],
        "self_validation_reward": [],
        "rl_validation_pos_err": [],
        "rl_validation_orn_err": [],
        "self_validation_pos_err": [],
        "self_validation_orn_err": [],
        "hardware_seed": [],
    }
    for num_seeds, approach_zarr_paths in num_seeds_approach_zarr_paths.items():
        added_hardware_seeds = False
        for approach, zarr_path in approach_zarr_paths.items():
            summary_stats = summarize_rollout(zarr_path, reduce=False, use_cache=True)
            data_dict[f"{approach}_reward"].extend(summary_stats["metric/reward/sum"])
            data_dict[f"{approach}_pos_err"].extend(
                summary_stats["metric/pos_err/mean"]
            )
            data_dict[f"{approach}_orn_err"].extend(
                summary_stats["metric/orn_err/mean"]
            )
            if not added_hardware_seeds:
                data_dict["hardware_seed"].extend(summary_stats["hardware_meta/seed"])
                data_dict["num_seeds"].extend(
                    [num_seeds] * len(summary_stats["hardware_meta/seed"])
                )
                added_hardware_seeds = True
    for k, v in data_dict.items():
        print(k, len(v))
    df = pd.DataFrame(data_dict)
    df.to_csv("scatter_plot_control_performance.csv", index=False)

    # Setup publication-quality plot style
    setup_plot_style()
    # mean reward for rl_validation v.s. ours
    rl_validation_reward = df["rl_validation_reward"].mean()
    self_validation_reward = df["self_validation_reward"].mean()
    print(f"Mean reward for rl_validation: {rl_validation_reward:.4f}")
    print(f"Mean reward for self_validation: {self_validation_reward:.4f}")

    # mean pos err for rl_validation v.s. ours
    rl_validation_pos_err = df["rl_validation_pos_err"].mean()
    self_validation_pos_err = df["self_validation_pos_err"].mean()
    print(f"Mean pos err for rl_validation: {rl_validation_pos_err:.4f}")
    print(f"Mean pos err for self_validation: {self_validation_pos_err:.4f}")

    # mean orn err for rl_validation v.s. ours
    rl_validation_orn_err = df["rl_validation_orn_err"].mean()
    self_validation_orn_err = df["self_validation_orn_err"].mean()
    print(f"Mean orn err for rl_validation: {rl_validation_orn_err:.4f}")
    print(f"Mean orn err for self_validation: {self_validation_orn_err:.4f}")

    for x_key, y_key, x_label, y_label, metric_key in [
        (
            "rl_validation_reward",
            "self_validation_reward",
            "RL Validation Reward",
            "Self Validation Reward",
            "reward",
        ),
        (
            "rl_validation_pos_err",
            "self_validation_pos_err",
            "RL Validation Pos Err",
            "Self Validation Pos Err",
            "pos_err",
        ),
        (
            "rl_validation_orn_err",
            "self_validation_orn_err",
            "RL Validation Orn Err",
            "Self Validation Orn Err",
            "orn_err",
        ),
    ]:
        # Create jointplot with styled scatter
        g = sns.jointplot(
            x=x_key,
            y=y_key,
            # hue="num_seeds",
            data=df,
            kind="reg",
            scatter_kws={"alpha": 0.5, "marker": "o", "s": 6, "linewidths": 0.5},
        )

        # Style axis labels
        g.ax_joint.set_xlabel(
            x_label,
            fontsize=PLOT_CONFIG["font"]["label_size"],
        )
        g.ax_joint.set_ylabel(
            y_label,
            fontsize=PLOT_CONFIG["font"]["label_size"],
        )

        # Style tick labels
        g.ax_joint.tick_params(
            axis="both",
            which="major",
            labelsize=PLOT_CONFIG["font"]["tick_size"],
        )

        # Adjust layout
        plt.tight_layout()

        # Save figure with high DPI
        g.savefig(
            f"scatter_plot_control_performance_{x_key}_{y_key}.pdf",
            dpi=PLOT_CONFIG["figure"]["dpi"],
            bbox_inches="tight",
        )
        g.savefig(
            f"scatter_plot_control_performance_{x_key}_{y_key}.svg",
            dpi=PLOT_CONFIG["figure"]["dpi"],
            bbox_inches="tight",
        )
        plt.savefig(
            f"scatter_plot_control_performance_{x_key}_{y_key}.svg",
            dpi=PLOT_CONFIG["figure"]["dpi"],
            bbox_inches="tight",
        )

    # Compute Pearson's correlation coefficient
    pearson_r, pearson_p = st.pearsonr(
        df["rl_validation_reward"], df["self_validation_reward"]
    )
    print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4e}")
    # compute slope of the regression line
    slope, intercept, r_value, p_value, std_err = st.linregress(
        df["rl_validation_reward"], df["self_validation_reward"]
    )
    print(f"Slope of regression line: {slope:.4f}")
    print(f"Intercept of regression line: {intercept:.4f}")
    print(f"R-value of regression line: {r_value:.4f}")
    print(f"P-value of regression line: {p_value:.4e}")
    print(f"Standard error of regression line: {std_err:.4f}")


def plot_wheeled_bimanual_control_performance(
    choice_approach_zarr_paths, pickle_path, output_root
):
    # build a shared dataframe, which stores for each hardware seed
    # the performance in one domain (rl_validation/reward) along with
    # performance in the other domain (self_validation/reward) in one row
    # and the hardware seed in one column
    data_dict = {
        "hardware_seed": [],
        "traj_idx": [],
        "choice": [],
        "size": [],
    }
    approach_keys = ["mink", "small", "large"]
    for prefix in ["mink", "learned"]:
        data_dict[f"{prefix}_reward"] = []
        data_dict[f"{prefix}_pos_err"] = []
        data_dict[f"{prefix}_orn_err"] = []
        data_dict[f"{prefix}_done"] = []

    for choice, approach_zarr_paths in choice_approach_zarr_paths.items():
        assert (
            dict(zarr.open(approach_zarr_paths["mink"], mode="r").attrs["runner"])[
                "env"
            ]["pickle_path"]
            == pickle_path
        ), f"Mink controller pickle path is not correct for choice {choice}"
        data_frames = {}
        for approach in approach_keys:
            stats = summarize_rollout(
                approach_zarr_paths[approach], reduce=False, use_cache=False
            )
            reward = stats["metric/reward/sum"]
            pos_err = stats["metric/pos_err/mean"]
            orn_err = stats["metric/orn_err/mean"]
            done = stats["done/timeout/any"]
            hardware_seed = stats["hardware_meta/seed"]
            traj_idx = stats["rollout_meta/seed"]
            prefix = "mink" if approach == "mink" else "learned"
            data_frames[approach] = pd.DataFrame(
                {
                    "hardware_seed": hardware_seed,
                    "traj_idx": traj_idx,
                    f"{prefix}_reward": reward,
                    f"{prefix}_pos_err": pos_err,
                    f"{prefix}_orn_err": orn_err,
                    f"{prefix}_done": done,
                }
            )
        # Create DataFrames for each source and merge on (hardware_seed, traj_idx)
        assert len(approach_keys) == 3, "Only 3 approaches are supported"
        merged_small = data_frames["mink"].merge(
            data_frames["small"], on=["hardware_seed", "traj_idx"], how="inner"
        )
        merged_small["size"] = "small"
        merged_small["choice"] = choice
        merged_large = data_frames["mink"].merge(
            data_frames["large"], on=["hardware_seed", "traj_idx"], how="inner"
        )
        merged_large["size"] = "large"
        merged_large["choice"] = choice

        for col in data_dict:
            data_dict[col].extend(merged_small[col].tolist())
            data_dict[col].extend(merged_large[col].tolist())
    df = pd.DataFrame(data_dict)
    os.makedirs(output_root, exist_ok=True)
    df.to_csv(
        os.path.join(output_root, "scatter_plot_control_performance.csv"), index=False
    )

    # Setup publication-quality plot style
    setup_plot_style()

    reward = df["mink_reward"].mean()
    pos_err = df["mink_pos_err"].mean()
    orn_err = df["mink_orn_err"].mean()
    done = df["mink_done"].mean()
    print(f"Mean reward for mink: {reward:.4f}")
    print(f"Mean pos err for mink: {pos_err:.4f}")
    print(f"Mean orn err for mink: {orn_err:.4f}")
    print(f"Mean done for mink: {done:.4f}")
    for size in ["small", "large"]:
        size_df = df[df["size"] == size]
        reward = size_df["learned_reward"].mean()
        pos_err = size_df["learned_pos_err"].mean()
        orn_err = size_df["learned_orn_err"].mean()
        done = size_df["learned_done"].mean()
        print(f"Mean reward for {size}: {reward:.4f}")
        print(f"Mean pos err for {size}: {pos_err:.4f}")
        print(f"Mean orn err for {size}: {orn_err:.4f}")
        print(f"Mean done for {size}: {done:.4f}")

    reward_ticks = [0, 500, 1000, 1500, 2000]

    # Create jointplot with styled scatter
    g = sns.jointplot(
        x="mink_reward",
        y="learned_reward",
        hue="size",
        data=df,
        alpha=0.5,
        s=6,
        linewidths=0.5,
        # kind="kde",
        # kind="hist",
        # kind="reg",
        # scatter_kws={"alpha": 0.5, "marker": "o", "s": 6, "linewidths": 0.5},
        # marginal_kws=dict(bins=10, fill=False),
    )

    # Style axis labels
    g.ax_joint.set_xlabel(
        "Oracle Reward",
        fontsize=PLOT_CONFIG["font"]["label_size"],
    )
    g.ax_joint.set_ylabel(
        "Learned Controller Reward",
        fontsize=PLOT_CONFIG["font"]["label_size"],
    )

    # Style tick labels
    g.ax_joint.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_CONFIG["font"]["tick_size"],
    )
    g.ax_joint.set_xticks(reward_ticks)
    g.ax_joint.set_yticks(reward_ticks)
    g.ax_joint.set_xticklabels(reward_ticks)
    g.ax_joint.set_yticklabels(reward_ticks)
    g.ax_joint.set_xlim(min(reward_ticks), max(reward_ticks))
    g.ax_joint.set_ylim(min(reward_ticks), max(reward_ticks))
    # Adjust layout
    plt.tight_layout()

    # Save figure with high DPI
    g.savefig(
        os.path.join(output_root, "reward.svg"),
        dpi=PLOT_CONFIG["figure"]["dpi"],
        bbox_inches="tight",
    )
    for size in ["small", "large"]:
        size_df = df[df["size"] == size]
        print("=" * 100)
        print(f"Size: {size}")

        # Compute Pearson's correlation coefficient
        pearson_r, pearson_p = st.pearsonr(
            size_df["mink_reward"], size_df["learned_reward"]
        )
        print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4e}")
        # compute slope of the regression line
        slope, intercept, r_value, p_value, std_err = st.linregress(
            size_df["mink_reward"], size_df["learned_reward"]
        )
        print(f"Slope of regression line: {slope:.4f}")
        print(f"Intercept of regression line: {intercept:.4f}")
        print(f"R-value of regression line: {r_value:.4f}")
        print(f"P-value of regression line: {p_value:.4e}")
        print(f"Standard error of regression line: {std_err:.4f}")

        # make individual scatter plots for each size
        g = sns.jointplot(
            x="mink_reward",
            y="learned_reward",
            kind="reg",
            data=size_df,
            scatter_kws={"alpha": 0.5, "marker": "o", "s": 6, "linewidths": 0.5},
        )
        g.ax_joint.set_xlabel(
            "Oracle Reward",
            fontsize=PLOT_CONFIG["font"]["label_size"],
        )
        g.ax_joint.set_ylabel(
            f"Learned Controller Reward ({size})",
            fontsize=PLOT_CONFIG["font"]["label_size"],
        )

        # Style tick labels
        g.ax_joint.tick_params(
            axis="both",
            which="major",
            labelsize=PLOT_CONFIG["font"]["tick_size"],
        )
        g.ax_joint.set_xticks(reward_ticks)
        g.ax_joint.set_yticks(reward_ticks)
        g.ax_joint.set_xticklabels(reward_ticks)
        g.ax_joint.set_yticklabels(reward_ticks)
        g.ax_joint.set_xlim(min(reward_ticks), max(reward_ticks))
        g.ax_joint.set_ylim(min(reward_ticks), max(reward_ticks))
        plt.tight_layout()
        g.savefig(
            os.path.join(output_root, f"reward_{size}.svg"),
            dpi=PLOT_CONFIG["figure"]["dpi"],
            bbox_inches="tight",
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    quadruped_parser = subparsers.add_parser("quadruped")
    quadruped_parser.add_argument(
        "--rl_validation",
        nargs="+",
        required=True,
        help="RL-validation rollout summary zarr path per setting",
    )
    quadruped_parser.add_argument(
        "--self_validation",
        nargs="+",
        required=True,
        help="Self-validation rollout summary zarr path per setting",
    )
    quadruped_parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help='Label per summary pair, e.g. "seeds=128" (defaults to indices)',
    )

    wheeled_parser = subparsers.add_parser("wheeled_bimanual")
    wheeled_parser.add_argument(
        "--mink",
        nargs="+",
        required=True,
        help="Mink (oracle) controller eval zarr path per choice",
    )
    wheeled_parser.add_argument(
        "--small",
        nargs="+",
        required=True,
        help="Small learned controller eval summary zarr path per choice",
    )
    wheeled_parser.add_argument(
        "--large",
        nargs="+",
        required=True,
        help="Large learned controller eval summary zarr path per choice",
    )
    wheeled_parser.add_argument(
        "--choices",
        nargs="+",
        default=None,
        help="Choice label per summary triple (defaults to indices)",
    )
    wheeled_parser.add_argument(
        "--pickle_path",
        default="data/bimanual_dish_washing_test.pkl",
        help="Expected env pickle path recorded in the mink eval zarrs",
    )
    wheeled_parser.add_argument(
        "--output_root", default="wheeled_bimanual_control_plots/"
    )

    args = parser.parse_args()
    if args.mode == "quadruped":
        if len(args.rl_validation) != len(args.self_validation):
            parser.error(
                "--rl_validation and --self_validation must have the same number of paths"
            )
        labels = args.labels or [str(i) for i in range(len(args.rl_validation))]
        if len(labels) != len(args.rl_validation):
            parser.error("--labels must match the number of summary paths")
        num_seeds_approach_zarr_paths = {
            label: {"rl_validation": rl_path, "self_validation": self_path}
            for label, rl_path, self_path in zip(
                labels, args.rl_validation, args.self_validation
            )
        }
        plot_quadruped_control_performance(num_seeds_approach_zarr_paths)
    else:
        if not (len(args.mink) == len(args.small) == len(args.large)):
            parser.error(
                "--mink, --small, and --large must have the same number of paths"
            )
        choices = args.choices or [str(i) for i in range(len(args.mink))]
        if len(choices) != len(args.mink):
            parser.error("--choices must match the number of summary paths")
        choice_approach_zarr_paths = {
            choice: {"mink": mink_path, "small": small_path, "large": large_path}
            for choice, mink_path, small_path, large_path in zip(
                choices, args.mink, args.small, args.large
            )
        }
        plot_wheeled_bimanual_control_performance(
            choice_approach_zarr_paths, args.pickle_path, args.output_root
        )


if __name__ == "__main__":
    main()
