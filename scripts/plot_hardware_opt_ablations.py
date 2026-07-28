from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from plot_hardware_opt import HardwareOptPlotter
from t2.eval.utils import summarize_rollout
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_compute_budget(config: dict) -> int:
    return config["eval_fn/hardware_optimizer_fn/num_seeds"]["value"]


def get_ckpt_path(config: dict) -> str:
    return config["ckpt_path"]["value"]


# Map the ckpt_path recorded in each eval run's config.yaml to a model name,
# e.g. "wandb/run-<timestamp>-<id>/files/<step>.pt": "large"
CKPT_TO_NAME = {
    "path/to/large_model_checkpoint.pt": "large",
    "path/to/small_model_checkpoint.pt": "small",
}

NAME_COMPUTE_TO_TIME = {
    "large": {
        1: 2.3,
        2: 4.0,
        4: 7.7,
        8: 14.8,
        16: 29.4,
        32: 58.3,
        64: 116.4,
        128: 233.1,
    },
    "small": {
        1: 0.8,
        2: 1.2,
        4: 2.1,
        8: 4.0,
        16: 7.7,
        32: 15.0,
        64: 29.8,
        128: 59.5,
    },
}


@hydra.main(
    config_path="../config",
    config_name="plot_hardware_opt",
    version_base="1.3",
)
def main(cfg: DictConfig):
    summary_paths = Path(cfg.paths.results_root).rglob("hardware_opt_summary.zarr")

    data_dict = {
        "name": [],
        "reward": [],
        "reward_ci_low": [],
        "reward_ci_high": [],
        "optimize_time": [],
        "compute_budget": [],
    }

    for summary_path in summary_paths:
        summary = summarize_rollout(str(summary_path), use_cache=True, reduce=False)
        config_path = summary_path.parent / "config.yaml"
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        ckpt_path = get_ckpt_path(config)
        compute_budget = get_compute_budget(config)
        ckpt_name = CKPT_TO_NAME.get(ckpt_path, "unknown")

        reward = summary["hardware_meta/actual_value"]
        optimize_time = summary["hardware_meta/optimize_time"]
        avg_reward = np.mean(reward)
        reward_ci = st.t.interval(
            confidence=0.95,
            df=len(reward) - 1,
            loc=avg_reward,
            scale=st.sem(reward),
        )
        try:
            avg_optimize_time = NAME_COMPUTE_TO_TIME[ckpt_name][compute_budget]
        except KeyError:
            print(
                f"No optimize time for {ckpt_name} with compute budget {compute_budget}"
            )
            avg_optimize_time = np.mean(optimize_time)
        print(
            f"CKPT: {ckpt_name}, Compute Budget: {compute_budget}, Reward: {avg_reward:.2f}, Optimize Time: {avg_optimize_time:.2f}"
        )
        data_dict["name"].append(ckpt_name)
        data_dict["reward"].append(avg_reward)
        data_dict["reward_ci_low"].append(reward_ci[0])
        data_dict["reward_ci_high"].append(reward_ci[1])
        data_dict["optimize_time"].append(avg_optimize_time)
        data_dict["compute_budget"].append(compute_budget)

    # plot
    # convert data_dict to experiments
    df = pd.DataFrame(data_dict)
    hardware_opt_plotter = HardwareOptPlotter(cfg, {})

    line_colors = {
        "large": plt.get_cmap("Greens")(0.5),
        "small": plt.get_cmap("Blues")(0.5),
    }
    scatter_colors = {
        "large": plt.get_cmap("Greens")(0.5),
        "small": plt.get_cmap("Blues")(0.5),
    }

    cmaps = {
        "large": "Greens",
        "small": "Blues",
    }

    scatter_kwargs = {
        "linewidths": 1,
        "s": 40,
        "zorder": 5,
        "alpha": 0.9,
        "marker": "o",
    }

    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot confidence intervals
    for name in df.name.unique():
        name_df = df[df["name"] == name]
        name_df = name_df.sort_values(by="compute_budget")
        if len(name_df) > 0:
            ax.fill_between(
                x=name_df["optimize_time"],
                y1=name_df["reward_ci_low"],
                y2=name_df["reward_ci_high"],
                color=line_colors[name],
                alpha=0.5,
                zorder=2,
            )

        # Plot lines
        sns.lineplot(
            data=name_df,
            x="optimize_time",
            y="reward",
            hue="name",
            ax=ax,
            legend=True,
            linestyle="solid",
            alpha=1.0,
            palette=line_colors,
            zorder=3,
        )

        # Plot scatter points for each approach
        cmap = plt.get_cmap(cmaps[name])
        colors = [cmap(np.log2(b) / np.log2(128)) for b in name_df["compute_budget"]]
        ax.scatter(
            x=name_df["optimize_time"],
            y=name_df["reward"],
            c=colors,
            edgecolors=scatter_colors[name],
            **scatter_kwargs,
        )

    # Configure axis (title is set by caller)
    ax.set_xlabel("Optimize Time (s)")
    ax.set_ylabel("Average Reward")
    ax.set_xscale("log")

    # Set tick marks
    plot_times = [0.25, 1, 4, 16, 64, 256]
    ax.set_xticks(plot_times)
    ax.set_xticklabels([f"{int(t)}" for t in plot_times])
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.tight_layout()
    plt.savefig("bimanual_opt_ablation.pdf", dpi=300)


if __name__ == "__main__":
    main()
