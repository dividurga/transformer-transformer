from typing import Callable
import numpy as np
import os
import pickle
import zarr
import scipy.stats as st
import tqdm
from numpy.typing import NDArray
import sys

from t2.eval.diffusion_guidance import RegularizedExpTrackingError
from t2.eval.hardware_optimization import (
    label_episode_with_actual_value,
)


def actual_metrics_fn(
    episode_data: dict[str, NDArray[np.float32]],
    hardware: dict[str, NDArray[np.float32]],
) -> dict[str, float]:
    reward_fn = RegularizedExpTrackingError(
        pose_weight=1.0,
        pos_sigma=0.01,
        orn_sigma=0.5,
        power=2,
        torque_weight=0.0,
        velocity_weight=0.0,
        leaky_clip_slope=0.0,
    )
    _, labeled_hardware = label_episode_with_actual_value(
        _env=None,
        episode_data={k.split("rollout/")[1]: v for k, v in episode_data.items()},
        hardware={k.split("hardware/")[1]: v for k, v in hardware.items()},
        reward_fns={
            "regularized_tracking_error": reward_fn,
        },
    )
    tracking_only_actual_value = labeled_hardware["metadata/actual_value"]
    dimension = (
        hardware["hardware/link/geom_size"] * 2  # (num_geoms, 3)
    ).max(  # all geom size are half sizes (radii/half-lengths)
        axis=1
    )  # (num_geoms,)
    total_size = dimension.sum(axis=0)
    mass = float(hardware["hardware/link/mass"].sum())

    avg_pos_err = float(np.mean(episode_data["rollout/metric/pos_err"]))
    max_pos_err = float(np.max(episode_data["rollout/metric/pos_err"]))
    median_pos_err = float(np.median(episode_data["rollout/metric/pos_err"]))

    orn_err_deg = np.rad2deg(episode_data["rollout/metric/orn_err"])

    avg_orn_err = float(np.mean(orn_err_deg))
    max_orn_err = float(np.max(orn_err_deg))
    median_orn_err = float(np.median(orn_err_deg))

    torque = np.abs(episode_data["rollout/actuator_obs/force"])

    avg_torque = float(np.mean(torque))
    max_torque = float(np.max(torque))
    median_torque = float(np.median(torque))

    velocity = np.abs(episode_data["rollout/actuator_obs/velocity"])

    avg_velocity = float(np.mean(velocity))
    max_velocity = float(np.max(velocity))
    median_velocity = float(np.median(velocity))

    survived = float(np.any(episode_data["rollout/done/timeout"]))
    # reward_sum = float(np.sum(episode_data["rollout/metric/reward"]))

    return {
        "size (m)": total_size,
        "mass (kg)": mass,
        "avg_pos_err (cm)": avg_pos_err * 100,
        "avg_orn_err (deg)": avg_orn_err,
        "max_pos_err (cm)": max_pos_err * 100,
        "max_orn_err (deg)": max_orn_err,
        "median_pos_err (cm)": median_pos_err * 100,
        "median_orn_err (deg)": median_orn_err,
        "avg_torque (Nm)": avg_torque,
        "max_torque (Nm)": max_torque,
        "median_torque (Nm)": median_torque,
        "avg_velocity (rad/s)": avg_velocity,
        "max_velocity (rad/s)": max_velocity,
        "median_velocity (rad/s)": median_velocity,
        "survived (%)": survived * 100,
        "actual_value/tracking_only": tracking_only_actual_value,
    }


def compute_actual_metrics(
    data_path: str,
    preload_rollout_keys: list[str],
    preload_hardware_keys: list[str],
    episode_metric_fn: Callable[
        [dict[str, NDArray[np.float32]]], dict[str, NDArray[np.float32]]
    ],
    use_pbar: bool = False,
    use_cache: bool = True,
) -> dict[str, NDArray[np.float32]]:
    cache_path = data_path.replace(".zarr", "-actual-metrics-cache.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            episode_metrics = pickle.load(f)
        return episode_metrics

    store = zarr.storage.LocalStore(data_path, read_only=True)
    root = zarr.open(store, mode="r")
    episode_metrics = {}
    rollout_ends = np.array(root["rollout_meta/ends"][:])
    rollout_hardware_ids = np.array(root["rollout_meta/hardware_id"][:])
    link_ends = np.array(root["hardware_meta/link/ends"][:])
    actuator_ends = np.array(root["hardware_meta/actuator/ends"][:])
    dyna_joint_ends = np.array(root["hardware_meta/dyna_joint/ends"][:])
    fixed_joint_ends = np.array(root["hardware_meta/fixed_joint/ends"][:])
    hardware_optimize_times = np.array(root["hardware_meta/optimize_time"][:])
    preload_rollout_data = {k: np.array(root[k][:]) for k in preload_rollout_keys}
    preload_rollout_attrs = {k: dict(root[k].attrs) for k in preload_rollout_keys}

    preload_hardware_data = {k: np.array(root[k][:]) for k in preload_hardware_keys}
    preload_hardware_attrs = {k: dict(root[k].attrs) for k in preload_hardware_keys}

    pbar = tqdm.tqdm(
        range(rollout_ends.shape[0]),
        dynamic_ncols=True,
        desc="Summarizing rollout",
        disable=not use_pbar,
    )
    for episode_idx in pbar:
        episode_ends = int(rollout_ends[episode_idx])
        episode_starts = (
            0 if episode_idx == 0 else rollout_ends[episode_idx - 1]  # type: ignore
        )
        episode_data_dict = {}
        for key in preload_rollout_keys:
            data = preload_rollout_data[key][episode_starts:episode_ends]
            data_attrs = preload_rollout_attrs[key]
            dim_idx = 0
            for attr_name, attr_dim in data_attrs.items():
                attr_dim = int(attr_dim)
                episode_data_dict[key + "/" + attr_name] = data[
                    ..., dim_idx : dim_idx + attr_dim
                ]
                dim_idx += attr_dim
        hardware_id = rollout_hardware_ids[episode_idx]
        link_start = 0 if hardware_id == 0 else link_ends[hardware_id - 1]
        link_end = link_ends[hardware_id]
        actuator_start = 0 if hardware_id == 0 else actuator_ends[hardware_id - 1]
        actuator_end = actuator_ends[hardware_id]
        dyna_joint_start = 0 if hardware_id == 0 else dyna_joint_ends[hardware_id - 1]
        dyna_joint_end = dyna_joint_ends[hardware_id]
        fixed_joint_start = 0 if hardware_id == 0 else fixed_joint_ends[hardware_id - 1]
        fixed_joint_end = fixed_joint_ends[hardware_id]
        hardware_dict = {}
        for key in preload_hardware_keys:
            hardware_group = key.split("/")[1]
            if hardware_group == "link":
                data = preload_hardware_data[key][link_start:link_end]
            elif hardware_group == "actuator":
                data = preload_hardware_data[key][actuator_start:actuator_end]
            elif hardware_group == "dyna_joint":
                data = preload_hardware_data[key][dyna_joint_start:dyna_joint_end]
            elif hardware_group == "fixed_joint":
                data = preload_hardware_data[key][fixed_joint_start:fixed_joint_end]
            else:
                raise ValueError(f"Unknown hardware key: {key}")
            data_attrs = preload_hardware_attrs[key]
            dim_idx = 0
            for attr_name, attr_dim in data_attrs.items():
                attr_dim = int(attr_dim)
                hardware_dict[key + "/" + attr_name] = data[
                    ..., dim_idx : dim_idx + attr_dim
                ]
                dim_idx += attr_dim
        metrics = episode_metric_fn(episode_data_dict, hardware_dict)
        metrics["optimize_time"] = hardware_optimize_times[hardware_id]
        for key, value in metrics.items():
            if key not in episode_metrics:
                episode_metrics[key] = []
            episode_metrics[key].append(value)

    return {k: np.array(v) for k, v in episode_metrics.items()}


if __name__ == "__main__":
    data_path = sys.argv[1]
    actual_metrics = compute_actual_metrics(
        data_path,
        preload_rollout_keys=[
            "rollout/metric",
            "rollout/actuator_obs",
            "rollout/done",
            "rollout/track_link_obs",
            "rollout/target_pose",
        ],
        preload_hardware_keys=[
            "hardware/link",
            "hardware/actuator",
            "hardware/dyna_joint",
            "hardware/fixed_joint",
        ],
        episode_metric_fn=actual_metrics_fn,
        use_pbar=True,
        use_cache=True,
    )
    for k, v in actual_metrics.items():
        avg_v = v.mean()
        ci = st.t.interval(
            confidence=0.95,
            df=len(v) - 1,
            loc=avg_v,
            scale=st.sem(v),
        )
        if np.isnan(ci[0]) and np.isnan(ci[1]):
            ci = (avg_v, avg_v)
        print(f"{k}: {avg_v:.3f}, ({ci[0]:.3f}, {ci[1]:.3f})")
