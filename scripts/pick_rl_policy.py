"""Build the design-choice -> RL-policy-checkpoint map used by RL data generation.

Scans wandb run directories produced by scripts/train_rl_procedural.py, keeps
the newest sufficiently-trained checkpoint per discrete design choice, and
writes a JSON map from the 8-bit choice string (e.g. "00010110") to the
checkpoint path. scripts/datagen_rl.py consumes this map to roll out the
matching expert policy for each sampled design.

Usage:
    python scripts/pick_rl_policy.py \\
        --roots wandb_rl_policies_root1 wandb_rl_policies_root2 \\
        --output choice_to_ckpt_path.json
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the best RL policy checkpoint per design choice."
    )
    parser.add_argument(
        "--roots",
        type=str,
        nargs="+",
        required=True,
        help="Root directories containing wandb RL training runs to scan.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="choice_to_ckpt_path.json",
        help="Output JSON path for the choice -> checkpoint map.",
    )
    parser.add_argument(
        "--min_training_iteration",
        type=int,
        default=3_560_964_096,
        help="Skip runs whose latest checkpoint has fewer environment steps.",
    )
    parser.add_argument(
        "--max_pos_err",
        type=float,
        default=0.1,
        help="Median eval position error (m) above which a run is rejected.",
    )
    parser.add_argument(
        "--num_choices",
        type=int,
        default=128,
        help="Total number of discrete design choices (for missing-choice report).",
    )
    args = parser.parse_args()

    choices = {}
    for root_path in args.roots:
        root = Path(root_path)
        for config_path in root.rglob("config.pkl"):
            config = pickle.load(open(config_path, "rb"))
            choice = config.mj_model_data_path.split("/")[-1].split(".")[0]
            # sort checkpoints by largest training iteration
            ckpt_paths = list(
                sorted(
                    Path(os.path.dirname(config_path) + "/checkpoints").glob("*"),
                    key=lambda x: "{:30d}".format(int(str(x).split("/")[-1])),
                )
            )
            if len(ckpt_paths) == 0:
                continue
            wandb_metadata_path = os.path.join(
                os.path.dirname(config_path), "wandb-metadata.json"
            )
            wandb_metadata = json.load(open(wandb_metadata_path, "r"))
            start_date = wandb_metadata["startedAt"]
            start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            training_iteration = int(str(ckpt_paths[-1]).split("/")[-1])
            if training_iteration < args.min_training_iteration:
                continue
            successfully_trained = False

            wandb_summary_path = os.path.join(
                os.path.dirname(config_path), "wandb-summary.json"
            )
            output_log = os.path.join(os.path.dirname(config_path), "output.log")
            if os.path.exists(wandb_summary_path):
                wandb_summary = json.load(open(wandb_summary_path, "r"))
                successfully_trained = (
                    wandb_summary["eval/summary/pos_err/q50"] < args.max_pos_err
                )
                if not successfully_trained:
                    print(
                        "Not successfully trained",
                        config_path,
                        wandb_summary["eval/summary/pos_err/q50"],
                        f"based on {wandb_summary_path}",
                    )
            elif os.path.exists(output_log):
                with open(output_log, "r") as f:
                    lines = "\n".join(f.readlines())
                    pos_err_q50 = lines.split("eval/summary/pos_err/q50: ")[-1].split(
                        "\n"
                    )[0]
                    successfully_trained = float(pos_err_q50.strip()) < args.max_pos_err
                    if not successfully_trained:
                        print(
                            "Not successfully trained",
                            config_path,
                            pos_err_q50,
                            f"based on {output_log}",
                        )
            else:
                print("No wandb summary or output log found", config_path)
                continue
            if not successfully_trained:
                continue
            if choice not in choices:
                choices[choice] = []
            choices[choice].append((start_date, ckpt_paths[-1]))

    choice_to_ckpt_path = {}
    for choice, ckpt_paths in sorted(choices.items(), key=lambda x: x[0]):
        # for hardware design choices that have multiple checkpoints, pick the latest training run
        latest_ckpt_path = sorted(ckpt_paths, key=lambda x: x[0], reverse=True)[0][1]
        choice_to_ckpt_path[choice] = str(latest_ckpt_path)

    num_bits = max(1, (args.num_choices - 1).bit_length())
    all_possible_choices = {
        format(i, f"0{num_bits}b") for i in range(args.num_choices)
    }
    missing_choices = all_possible_choices - set(choice_to_ckpt_path.keys())
    print("missing choices:", missing_choices)
    json.dump(choice_to_ckpt_path, open(args.output, "w"), indent=4)
    print(f"Wrote {len(choice_to_ckpt_path)} entries to {args.output}")
