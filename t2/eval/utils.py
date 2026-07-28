import hashlib
import logging
import os
import pickle
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import tqdm
import zarr
from numpy.typing import NDArray
from omegaconf import DictConfig

from t2.data.dataset import TimestepSampler


def iterate_traj_seeds(
    traj_indices: list[int] | list[list[int]],
    num_seeds_per_datapoint: int,
) -> Iterator[tuple[int | list[int], int, int]]:
    """Iterate over trajectory/seed combinations for hardware optimization.

    This function consolidates the iteration logic used across hardware optimization
    evaluation scripts. It generates consistent (traj_items, hardware_iter, hardware_seed)
    tuples using a deterministic hash-based approach.

    Args:
        traj_indices: Either a list of trajectory indices (single-traj optimization)
            or a list of lists of trajectory indices (multi-traj composed optimization).
        num_seeds_per_datapoint: Number of random seeds to try per trajectory item.

    Yields:
        Tuple of (traj_items, hardware_iter, hardware_seed) where:
            - traj_items: The trajectory index or list of indices
            - hardware_iter: The iteration number (0 to num_seeds_per_datapoint-1)
            - hardware_seed: Deterministically generated seed for this combination
    """
    for traj_items in traj_indices:
        traj_str = str(traj_items)
        for hardware_iter in range(num_seeds_per_datapoint):
            hashobj = hashlib.sha256(f"{hardware_iter}{traj_str}".encode())
            rs = np.random.RandomState(
                int(hashobj.hexdigest(), 16) % (np.iinfo(np.int32).max)
            )
            hardware_seed = rs.randint(0, np.iinfo(np.int32).max)
            yield traj_items, hardware_iter, hardware_seed


def summarize_rollout(
    data_path: str, use_pbar: bool = False, reduce: bool = True, use_cache: bool = True
) -> dict[str, float]:
    cache_path = data_path.replace(".zarr", "-summary-cache.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            episode_metrics = pickle.load(f)
        if reduce:
            return {k: float(np.mean(v)) for k, v in episode_metrics.items()}
        else:
            return episode_metrics

    store = zarr.storage.LocalStore(data_path, read_only=True)
    root = zarr.open(store, mode="r")
    episode_metrics = {}
    for k in list(root["hardware_meta"].keys()):
        if k in {
            "link",
            "fixed_joint",
            "actuator",
            "dyna_joint",
            "robot_generator",
        }:
            continue
        data = root[f"hardware_meta/{k}"][:]
        episode_metrics["hardware_meta/" + k] = data[
            root["rollout_meta/hardware_id"][:]
        ]
    episode_metrics["rollout_meta/seed"] = root["rollout_meta/seed"][:]
    rollout_meta_ends = root["rollout_meta/ends"][:]
    rollout_data = {
        "metric": root["rollout/metric"][:],
        "done": root["rollout/done"][:].astype(np.bool_),
    }
    rollout_data_attrs = {
        "metric": dict(root["rollout/metric"].attrs),
        "done": dict(root["rollout/done"].attrs),
    }
    pbar = tqdm.tqdm(
        range(rollout_meta_ends.shape[0]),
        dynamic_ncols=True,
        desc="Summarizing rollout",
        disable=not use_pbar,
    )
    for episode_idx in pbar:
        episode_ends = int(rollout_meta_ends[episode_idx])
        episode_starts = (
            0 if episode_idx == 0 else rollout_meta_ends[episode_idx - 1]  # type: ignore
        )
        for key in rollout_data.keys():
            data = rollout_data[key][episode_starts:episode_ends]
            data_attrs = rollout_data_attrs[key]
            data_terms = {}
            dim_idx = 0
            for attr_name, attr_dim in data_attrs.items():
                attr_dim = int(attr_dim)
                data_terms[attr_name] = data[..., dim_idx : dim_idx + attr_dim]
                dim_idx += attr_dim
            for term in data_terms.keys():
                if term.endswith("/id"):
                    continue
                data_term = data_terms[f"{term}"]
                if data_term.dtype == np.bool_:
                    prefix = f"{key}/{term}"
                    if f"{prefix}/any" not in episode_metrics:
                        episode_metrics[f"{prefix}/any"] = []
                    episode_metrics[f"{prefix}/any"].append(np.any(data_term))
                    continue
                metric = np.abs(data_term)
                prefix = f"{key}/{term}"
                if f"{prefix}/mean" not in episode_metrics:
                    episode_metrics[f"{prefix}/mean"] = []
                if f"{prefix}/sum" not in episode_metrics:
                    episode_metrics[f"{prefix}/sum"] = []
                episode_metrics[f"{prefix}/mean"].append(np.mean(metric))
                episode_metrics[f"{prefix}/sum"].append(np.sum(metric))

    with open(cache_path, "wb") as f:
        pickle.dump(episode_metrics, f)
    if reduce:
        return {k: float(np.mean(v)) for k, v in episode_metrics.items()}
    else:
        return episode_metrics


def traj2batch(
    traj: NDArray[np.float32],
    seq_lens: DictConfig | dict[str, int],
    sampler: TimestepSampler,
    center_traj: bool,
):
    n_steps = int(seq_lens["rollout_steps"])
    if traj.ndim == 3:
        traj = traj[:, None, :, :]
    n_target_links = traj.shape[1]
    assert int(seq_lens["track_link"]) == n_target_links, (
        f"Number of target links {n_target_links} does not match sequence length {int(seq_lens['track_link'])}"
    )
    assert sampler.num_rollout_steps == n_steps, (
        f"Sampler num rollout steps {sampler.num_rollout_steps} does not match sequence length {n_steps}"
    )
    episode_starts = 0
    episode_ends = traj.shape[0]
    anchor_idx = 0  # always include the first timestep
    rollout_idx, rollout_idxs = sampler(
        rollout_idx=anchor_idx,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
    )
    rollout_mask = torch.zeros((1, n_steps, 1), dtype=torch.bool)
    poses = traj[rollout_idxs.astype(int).tolist()]  # (n_steps, n_target_links, 4, 4)
    if center_traj:
        init_pos_xy = poses[0, :, :2, 3].mean(axis=0)
        poses[:, :, :2, 3] = poses[:, :, :2, 3] - init_pos_xy
    batch = {}
    # prepare target pose
    assert poses[:, :, :3, 3].shape == (n_steps, n_target_links, 3)
    batch["target_pose/pos"] = (
        torch.from_numpy(poses[:, :, :3, 3]).float().reshape(1, -1, 3)
    )
    assert poses[:, :, :3, :3].shape == (n_steps, n_target_links, 3, 3)
    batch["target_pose/rotmat"] = (
        torch.from_numpy(poses[:, :, :3, :3]).float().reshape(1, -1, 9)
    )
    # prepare masks and times
    for prefix, element_len in [
        ("actuator_obs", int(seq_lens["actuator_obs"])),
        ("ctrl", int(seq_lens["actuator"])),
        ("target_pose", int(seq_lens["target_pose"])),
        ("track_link_obs", int(seq_lens["track_link_obs"])),
        ("dyna_joint_obs", int(seq_lens["dyna_joint_obs"])),
        ("free_link_obs", int(seq_lens["free_link_obs"])),
    ]:
        batch[f"{prefix}/mask"] = torch.zeros((1, n_steps, element_len, 1)).bool()
        batch[f"{prefix}/time/id"] = torch.zeros((1, n_steps, element_len, 1))
        batch[f"{prefix}/time/id"][0, :, :, 0] = torch.from_numpy(
            rollout_idxs - rollout_idx
        )[:, None]
        batch[f"{prefix}/time/id"] = batch[f"{prefix}/time/id"].reshape(1, -1, 1)
        batch[f"{prefix}/mask"][0, :, :, 0] = rollout_mask
        if prefix == "target_pose" or prefix == "track_link_obs":
            batch[f"{prefix}/mask"][0, :, n_target_links:] = True
        batch[f"{prefix}/mask"] = batch[f"{prefix}/mask"].reshape(1, -1, 1).bool()

    for prefix in ["dyna_joint", "fixed_joint"]:
        # add dyna_joint/order/id evenly
        batch[f"{prefix}/order/id"] = torch.zeros((1, seq_lens[prefix], 1)).long()
        batch[f"{prefix}/order/id"][
            :,
            ::2,
        ] = 1
    return batch


def extract_masked_hardware(
    robot_dict: dict[str, Any],
) -> tuple[dict[str, NDArray[np.float32 | np.int_]], int, int, int, int]:
    """Extract masked hardware from a robot dict and return counts.

    Takes a robot dict with mask arrays and returns a filtered dict with only
    unmasked values, along with the counts of each component type.

    Args:
        robot_dict: Dictionary with hardware arrays and corresponding mask arrays.
            Expected to have keys like "link/...", "actuator/...", etc. with
            corresponding "link/mask", "actuator/mask", etc.

    Returns:
        A tuple of (masked_hardware_dict, num_links, num_actuators, num_dyna_joints, num_fixed_joints)
        where masked_hardware_dict contains only the unmasked values.
    """
    masked_hardware_dict: dict[str, NDArray[np.float32 | np.int_]] = {}
    num_links = -1
    num_actuators = -1
    num_dyna_joints = -1
    num_fixed_joints = -1

    for k, v in robot_dict.items():
        if k.endswith("/mask"):
            continue
        group = k.split("/")[0]
        mask_key = group + "/mask"
        if mask_key not in robot_dict:
            logging.warning(f"Missing mask for {k}")
            continue
        mask = robot_dict[mask_key].reshape(-1)
        masked_hardware_dict[k] = v[~mask]
        seq_len = len(masked_hardware_dict[k])
        if group == "link":
            num_links = seq_len
        elif group == "actuator":
            num_actuators = seq_len
        elif group == "dyna_joint":
            num_dyna_joints = seq_len
        elif group == "fixed_joint":
            num_fixed_joints = seq_len

    return (
        masked_hardware_dict,
        num_links,
        num_actuators,
        num_dyna_joints,
        num_fixed_joints,
    )


def run_model_upfront_phase(
    optimizer: Any,  # HardwareOptimizer - avoid circular import
    trajs: list[Any],
    traj_indices: list[int] | list[list[int]],
    num_seeds_per_datapoint: int,
    optimize_hardware: bool = True,
) -> None:
    """Pre-compute all model outputs and unload model from GPU.

    This function handles the "run model upfront" optimization strategy where
    all model inferences are done first, cached, and then the model is unloaded
    to free GPU memory for subsequent evaluation.

    Args:
        optimizer: HardwareOptimizer instance with optimize() and unload_model() methods.
        trajs: List of trajectory arrays loaded from pickle.
        traj_indices: Either a list of trajectory indices (single-traj optimization)
            or a list of lists of trajectory indices (multi-traj composed optimization).
        num_seeds_per_datapoint: Number of random seeds to try per trajectory item.
        optimize_hardware: Whether hardware optimization is enabled. If False, this
            function does nothing.
    """
    if not optimize_hardware:
        return

    total_combinations = len(list(traj_indices)) * num_seeds_per_datapoint
    logging.info(
        f"Running model upfront for {len(list(traj_indices))} trajectory items "
        f"x {num_seeds_per_datapoint} seeds = {total_combinations} combinations..."
    )

    completed = 0
    for traj_items, hardware_iter, hardware_seed in iterate_traj_seeds(
        traj_indices, num_seeds_per_datapoint
    ):
        # Prepare trajectory list (handles both single and multi-trajectory cases)
        is_multi = (
            hasattr(traj_items, "__iter__")
            and hasattr(traj_items, "__len__")
            and not isinstance(traj_items, (str, int))
        )
        if is_multi:
            traj_list = [trajs[int(idx)] for idx in traj_items]
            traj_label = f"trajs {list(traj_items)}"
        else:
            traj_list = [trajs[int(traj_items)]]
            traj_label = f"traj {traj_items}"

        try:
            _, predicted_value, optimize_time = optimizer.optimize(
                trajs=traj_list, seed=hardware_seed
            )
            completed += 1
            logging.info(
                f"[{completed}/{total_combinations}] Pre-computed {traj_label}, "
                f"seed={hardware_seed}: value={predicted_value:.2f}, time={optimize_time:.2f}s"
            )
        except Exception as e:
            completed += 1
            logging.warning(
                f"[{completed}/{total_combinations}] Pre-compute failed for {traj_label}, "
                f"seed={hardware_seed}: {e}"
            )

    logging.info("Model upfront computation complete. Unloading model from GPU...")
    optimizer.unload_model()
    logging.info(
        "Model unloaded. GPU memory should now be at baseline. "
        "Continuing with evaluation using cached results..."
    )


def log_summary_stats(summary_stats: dict[str, float]) -> None:
    """Log summary statistics in a consistent format.

    Logs metrics with appropriate formatting based on their type:
    - Quantile/mean metrics: formatted as floats with 2 decimal places
    - Boolean metrics (ending in /any): formatted as percentages
    - Skips actuator metrics for brevity

    Args:
        summary_stats: Dictionary of metric names to values.
    """
    for k, v in summary_stats.items():
        if k.startswith("metric/actuator"):
            continue
        if (
            any(k.endswith(suffix) for suffix in ["/q95", "/q50", "/mean"])
            or k
            in {
                "metric/reward/sum",
                "predicted_value",
                "optimize_time",
            }
            or k.startswith("actual_value")
        ):
            logging.info(f"{k}: {v:.2f}")
        elif any(k.endswith(suffix) for suffix in ["/any"]):
            logging.info(f"{k}: {v * 100:.1f}%")
