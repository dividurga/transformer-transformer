import functools
import time
from typing import Any, Callable
import logging
import hashlib
import os
import shutil
import pickle
import typing

import cma
import numpy as np
from dm_control import mjcf
import torch
import mujoco

from t2.data.pad import pad
from t2.env.base_env import BaseEnv, HardwareDict
from t2.env.ik_runner import IKEnvRunner
from t2.env.mink_runner import MinkEnvRunner
from t2.env.mj_utils import TARGET_SITE_GROUP, add_mocap_body_with_site
from t2.env.rl_runner import RLEnvRunner
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv
from t2.eval.diffusion_guidance import RewardFn, HardwareOptimizer
from t2.eval.utils import iterate_traj_seeds, run_model_upfront_phase, summarize_rollout
from t2.model.t2 import DecoderBundle
from t2.io.schema import concat_zarr_stores
from t2.model.t2 import T2Decoder
from t2.robotok.io import deserialize
from t2.robotok.tokenizer import DetokenizeError, detokenize, preprocess_mjcf, tokenize
from t2.train.augment import AddPositionId

TrackEnvRunner = IKEnvRunner | MinkEnvRunner | RLEnvRunner
VERBOSE = False


def key_to_dtype(key: str) -> torch.dtype:
    if key.endswith("/id"):
        return torch.long
    elif key.endswith("/mask"):
        return torch.bool
    return torch.float32


def create_seq_len_cfg_from_episode_data(
    episode_data: dict[str, Any], hardware: HardwareDict
) -> dict[str, int]:
    episode_len = len(episode_data["metric/reward"])
    n_track_links = len(list(filter(lambda x: x != -1, hardware["link/track_link/id"])))
    assert n_track_links >= 1, (
        f"There should be at least one track link, got {n_track_links}"
    )
    # Compute number of track links/targets from the data shape
    # Data has shape (time, n_links, features) for 3D arrays
    track_link_obs_pos = episode_data["track_link_obs/pos"]
    target_pose_pos = episode_data["target_pose/pos"]
    n_track_links = (
        track_link_obs_pos.shape[1]
        if track_link_obs_pos.ndim == 3
        else (len(track_link_obs_pos) // episode_len)
    )
    n_target_poses = (
        target_pose_pos.shape[1]
        if target_pose_pos.ndim == 3
        else (len(target_pose_pos) // episode_len)
    )
    seq_len_cfg = {
        "dyna_joint": len(hardware["dyna_joint/link/id"]),
        "fixed_joint": len(hardware["fixed_joint/link/id"]),
        "link": len(hardware["link/geom_type"]),
        "actuator": len(hardware["actuator/dyna_joint/id"]),
        "dyna_joint_obs": len(hardware["dyna_joint/link/id"]) // 2,
        "actuator_obs": len(hardware["actuator/dyna_joint/id"]),
        "ctrl": len(hardware["actuator/dyna_joint/id"]),
        "track_link": n_track_links,
        "track_link_obs": n_track_links,
        "target_pose": n_target_poses,
    }
    if "free_link_obs/pos" in episode_data:
        num_free_link_obs = len(episode_data["free_link_obs/pos"])
        num_free_link = num_free_link_obs // episode_len
        seq_len_cfg["free_link"] = num_free_link
        seq_len_cfg["free_link_obs"] = num_free_link
    return seq_len_cfg


def create_add_pos_id(seq_len_cfg: dict[str, int], episode_len: int) -> AddPositionId:
    add_pos_id_cfg = {}
    # hardware keys with non repeating dimensions
    for k in ["link", "actuator"]:
        add_pos_id_cfg[k] = (seq_len_cfg[k], 0, 1, 1)
    # hardware keys with repeating dimensions
    for k in ["dyna_joint", "fixed_joint"]:
        add_pos_id_cfg[k] = (seq_len_cfg[k] // 2, 0, 1, 2)
    # episode data that's scale by episode_len
    for k in [
        "track_link_obs/track_link",
        "dyna_joint_obs/dyna_joint",
        "free_link_obs/free_link",
        "actuator_obs/actuator",
        "ctrl/actuator",
        "target_pose/track_link",
    ]:
        prefix = k.split("/")[0]
        if prefix not in seq_len_cfg:
            continue
        add_pos_id_cfg[k] = (seq_len_cfg[prefix], 0, episode_len, 1)
    # add time groups
    # TODO double check this when doing bimanual trajectory tracking
    for k in ["target_pose", "track_link_obs", "actuator_obs"]:
        add_pos_id_cfg[k + "/time"] = (episode_len, 0, 1, seq_len_cfg[k])
    add_pos_id = AddPositionId(add_pos_id_cfg)
    return add_pos_id


def label_episode_with_actual_value(
    _env: BaseEnv,
    episode_data: dict[str, Any],
    hardware: HardwareDict,
    reward_fns: dict[str, Callable[[dict[str, torch.Tensor]], torch.Tensor]],
):
    seq_len_cfg = create_seq_len_cfg_from_episode_data(episode_data, hardware)
    episode_len = len(episode_data["metric/reward"])
    add_pos_id = create_add_pos_id(seq_len_cfg, episode_len)

    reward_terms = {}
    filtered_episode_data = dict(
        filter(
            lambda x: not (x[0].startswith("metric/") or x[0].startswith("done/")),
            episode_data.items(),
        )
    )
    filtered_hardware = dict(
        filter(
            lambda x: not (x[0].startswith("metadata/")),
            hardware.items(),
        )
    )
    padded_episode_data = pad(
        data_dict={**filtered_episode_data, **filtered_hardware},
        group_seq_lens={
            **seq_len_cfg,
            "rollout_steps": episode_len,
        },
    )
    processed_torch_episode_data = add_pos_id(
        {
            k: torch.from_numpy(v)[None, :].to(dtype=key_to_dtype(k))
            for k, v in padded_episode_data.items()
        }
    )
    for reward_term, reward_fn in reward_fns.items():
        reward = reward_fn(processed_torch_episode_data)
        reward_terms["metadata/actual_value_" + reward_term] = reward.mean().item()
    reward_sum = sum(reward_terms.values())
    reward_terms["metadata/actual_value"] = reward_sum
    return episode_data, {**hardware, **reward_terms}


def hardware_reset_from_params(
    _env: BaseEnv,
    _seed: int,
    choices: list[int],
    uniforms: list[float],
    robogen_from_params_fn: Callable[[list[int], list[float]], mjcf.RootElement],
):
    mjcf_model = robogen_from_params_fn(choices=choices, uniforms=uniforms)  # pyright: ignore[reportCallIssue]
    mjcf_model = preprocess_mjcf(mjcf_model)
    (
        tokenized_robot,
        mjjntid_to_dyna_jntid,
        mjgeomid_to_linkid,
        mjsiteid_to_tracklinkid,
    ) = tokenize(
        mj_robot=mjcf_model,
    )  # type: ignore
    _env.hardware_seed = int(_seed)
    # Add target sites for each end effector (bimanual robots have multiple)
    n_end_effectors = getattr(_env, "n_end_effectors", 1)
    for i in range(n_end_effectors):
        mjcf_model = add_mocap_body_with_site(
            mjcf_model, f"target_{i}", site_group=TARGET_SITE_GROUP
        )
    _env.post_hardware_reset(
        mjcf_model=mjcf_model,
        tokenized_robot=tokenized_robot,
        mjjntid_to_dyna_jntid=mjjntid_to_dyna_jntid,
        mjgeomid_to_linkid=mjgeomid_to_linkid,
        mjsiteid_to_tracklinkid=mjsiteid_to_tracklinkid,
    )


def label_optimize_time(
    _env: BaseEnv,
    episode_data: dict[str, Any],
    hardware: HardwareDict,
    _start_monotonic_time: float,
):
    optimize_time = float(time.monotonic() - _start_monotonic_time)
    hardware["metadata/optimize_time"] = np.array(optimize_time)
    return episode_data, hardware


def cmaes_objective(
    x: np.ndarray,
    choices: list[int],
    root_dir: str,
    _traj_indices: list[int],
    _episode_seed_base: int,
    _n_target_trajs: int,
    _runner: EnvRunner,
    reward_fns: dict[str, RewardFn],
    robogen_from_params_fn: Callable[[list[int], list[float]], mjcf.RootElement],
    start_monotonic_time: float,
) -> float:
    """CMA-ES objective function for trajectory optimization.

    Evaluates a candidate robot design across one or more trajectories.
    For single-trajectory optimization, pass _traj_indices=[idx].

    Args:
        x: Candidate uniforms vector from CMA-ES
        choices: Fixed discrete choices for robot morphology
        root_dir: Directory for temporary data storage
        _traj_indices: List of trajectory indices to evaluate (use [idx] for single trajectory)
        _episode_seed_base: Base seed for episode computation
        _n_target_trajs: Total number of target trajectories
        _runner: EnvRunner
        reward_fns: Reward functions for evaluation
        robogen_from_params_fn: Function to generate robot MJCF from params
        start_monotonic_time: Start time for optimization

    Returns:
        Negative reward (cost to minimize)
    """
    # Compute episode seeds for all trajectory indices
    episode_seeds = [
        traj_idx + _n_target_trajs * _episode_seed_base for traj_idx in _traj_indices
    ]

    hash_str = hashlib.sha256(
        f"{choices}_{x.tolist()}_{_traj_indices}_{_episode_seed_base}".encode()
    ).hexdigest()

    data_path = os.path.join(root_dir, f"{hash_str}.zarr")
    assert not os.path.exists(data_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    _runner.env.robot_generator = lambda _: robogen_from_params_fn(
        choices=choices,
        uniforms=x.tolist(),
    )

    _label_optimization_time = functools.partial(
        label_optimize_time,
        _start_monotonic_time=start_monotonic_time,
    )
    _label_episode_with_actual_value = functools.partial(
        label_episode_with_actual_value,
        reward_fns=reward_fns,
    )

    def post_process_episode_data_fn(
        _env: BaseEnv,
        episode_data: dict[str, Any],
        hardware: HardwareDict,
    ):
        episode_data, hardware = _label_optimization_time(_env, episode_data, hardware)
        episode_data, hardware = _label_episode_with_actual_value(
            _env, episode_data, hardware
        )
        return episode_data, hardware

    _runner.run_episodes(
        hardware_seed=0,
        episode_seeds=episode_seeds,
        policy_fn=None,
        hardware_reset_fn=functools.partial(
            hardware_reset_from_params,
            robogen_from_params_fn=robogen_from_params_fn,
            uniforms=x.tolist(),
            choices=choices,
        ),
        post_process_episode_data_fn=post_process_episode_data_fn,
        data_path=data_path,
    )
    _runner.close()

    summary_stats = summarize_rollout(data_path)
    value = summary_stats["hardware_meta/actual_value"]
    cost = -value
    return cost


def run_cmaes_on_traj_and_hardware_seed(
    runner_fn: Callable[[], TrackEnvRunner],
    optimize_runner_fn: Callable[[], TrackEnvRunner],
    reward_fns: dict[str, RewardFn],
    traj_indices: list[int],
    hardware_seed: int,
    data_path: str,
    cmaes_dir: str,
    choices: list[int],
    robogen_from_params_fn: Callable[[list[int], list[float]], mjcf.RootElement],
    episode_seed_base: int,
    # cmaes params
    num_params: int,
    init_sigma: float,
    pop_size: int,
    max_fun: int,
    n_jobs: int,
):
    """Run CMA-ES optimization over one or more trajectories.

    Args:
        runner_fn: Factory function to create the final evaluation runner
        optimize_runner_fn: Factory function to create the optimization runner
        reward_fns: Reward functions for evaluation
        traj_indices: List of trajectory indices to optimize over (use [idx] for single trajectory)
        hardware_seed: Seed for hardware initialization
        data_path: Path to save final optimized hardware data
        cmaes_dir: Directory for intermediate CMA-ES data
        choices: Discrete choices for robot morphology
        robogen_from_params_fn: Function to generate robot from parameters
        episode_seed_base: Base seed for computing episode seeds
        num_params: Number of continuous parameters to optimize
        init_sigma: Initial sigma for CMA-ES
        pop_size: Population size for CMA-ES
        max_fun: Maximum function evaluations for CMA-ES
        n_jobs: Number of parallel jobs for CMA-ES
    """
    runner = runner_fn()
    optimize_runner = optimize_runner_fn()
    optimize_runner.render = False
    optimize_runner.use_gui = False
    # Enable dump_hardware_every_episode since hardware metadata (actual_value)
    # changes between episodes (harmless for single-trajectory case)
    optimize_runner.dump_hardware_every_episode = True
    runner.dump_hardware_every_episode = True

    init_x = np.random.rand(num_params)
    # Compute episode seeds for all trajectory indices
    n_target_trajs = runner.tracking_env.n_target_trajs
    episode_seeds = [
        traj_idx + n_target_trajs * episode_seed_base for traj_idx in traj_indices
    ]
    start_monotonic_time = time.monotonic()

    def optimize_hardware_for_traj(
        _env: BaseEnv,
        _seed: int,
        _traj_indices: list[int] = traj_indices,
        _episode_seed_base: int = episode_seed_base,
        _init_x: np.ndarray = init_x,
    ):
        if max_fun == 0:
            # random baseline
            rs = np.random.RandomState(_seed)
            best_x = rs.rand(num_params)
        else:
            obj_fn_partial = functools.partial(
                cmaes_objective,
                reward_fns=reward_fns,
                root_dir=cmaes_dir,
                _traj_indices=_traj_indices,
                _episode_seed_base=_episode_seed_base,
                _n_target_trajs=n_target_trajs,
                _runner=optimize_runner,
                choices=choices,
                robogen_from_params_fn=robogen_from_params_fn,
                start_monotonic_time=start_monotonic_time,
            )
            es = cma.CMAEvolutionStrategy(
                _init_x,
                init_sigma,
                inopts={
                    "bounds": [0, 1],
                    "seed": _seed,
                    "popsize": pop_size,
                    "verbose": -2 if VERBOSE else -9,
                },
            )
            es.optimize(obj_fn_partial, maxfun=max_fun, n_jobs=n_jobs)
            best_x = es.result.xbest
        total_optimization_time = float(time.monotonic() - start_monotonic_time)
        hardware_reset_from_params(
            _env=_env,
            _seed=_seed,
            choices=choices,
            uniforms=best_x.tolist(),
            robogen_from_params_fn=robogen_from_params_fn,
        )
        _env.robot_generator = lambda _: robogen_from_params_fn(
            choices=choices,
            uniforms=best_x.tolist(),
        )
        _env.hardware_dict["metadata/optimize_time"] = np.array(total_optimization_time)

    _label_episode_with_actual_value = functools.partial(
        label_episode_with_actual_value,
        reward_fns=reward_fns,
    )

    runner.run_episodes(
        hardware_seed=hardware_seed,
        episode_seeds=episode_seeds,
        policy_fn=None,
        hardware_reset_fn=optimize_hardware_for_traj,
        data_path=data_path,
        post_process_episode_data_fn=_label_episode_with_actual_value,
    )
    runner.close()


def post_process_mjcf_solver_parameters(
    mjcf_model: mjcf.RootElement,
) -> mjcf.RootElement:
    logging.warning("HACK: post_process_mjcf_solver_parameters")
    #########################################################
    ##### HACK: Inorder to match solver parameters
    # used in assets/mjcf/umi_on_legs_plus_plus/defaults.xml
    # not required for validity of physics, but required to
    # more closely match the physics of the data generation
    #########################################################
    for joint in mjcf_model.find_all("joint"):
        if "free_joint" in joint.name:
            continue
        joint.solreflimit = "0.005 1"
    for equality in mjcf_model.find_all("equality"):
        equality.solref = "0.005 1"
    return mjcf_model


def post_process_sim_collision_pairs(mj_model: mujoco.MjModel) -> mujoco.MjModel:
    logging.warning("HACK: post_process_sim_collision_pairs")
    #########################################################
    ##### HACK: Inorder to match contact pair exclusions
    # used in assets/mjcf/umi_on_legs_plus_plus/defaults.xml
    # not required for validity of physics, but required to
    # more closely match the physics of the data generation
    #########################################################
    for geom_id in range(mj_model.ngeom):
        is_sphere = mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
        radius = mj_model.geom_size[geom_id][0]
        if is_sphere and radius < 0.03 and radius > 0.01:
            continue
        is_plane = mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
        if is_plane:
            continue
        mj_model.geom_conaffinity[geom_id] = 0
        mj_model.geom_contype[geom_id] = 0
    return mj_model


def evaluate_hardware_opt(
    decoder: T2Decoder | DecoderBundle,
    log_dir: str,
    runner: IKEnvRunner | MinkEnvRunner,
    hardware_optimizer_fn: Callable[[T2Decoder | DecoderBundle], HardwareOptimizer],
    gravcomp: bool,
    num_seeds_per_datapoint: int,
    traj_indices: list[int] | list[list[int]],
    optimize_hardware: bool,
    post_process_mjcf_fns: (
        list[Callable[[mjcf.RootElement], mjcf.RootElement]] | None
    ) = None,
    post_process_mj_model_fns: (
        list[Callable[[mujoco.MjModel], mujoco.MjModel]] | None
    ) = None,
    run_model_upfront: bool = False,
):
    """
    Evaluate hardware optimization.

    Args:
        decoder: The T2 decoder for hardware generation. Should be a DecoderBundle
            if run_model_upfront=True to enable model unloading.
        log_dir: Directory to save evaluation results.
        runner: Environment runner for evaluation.
        hardware_optimizer_fn: Function to create hardware optimizer from decoder.
        gravcomp: Whether to use gravity compensation.
        num_seeds_per_datapoint: Number of seeds per datapoint.
        traj_indices: Either a list of trajectory indices (single-traj optimization)
            or a list of lists of trajectory indices (multi-traj composed optimization).
            For composed optimization, each inner list contains indices to compose together.
        optimize_hardware: Whether to optimize hardware or use default.
        post_process_mjcf_fns: Optional list of functions to post-process MJCF model.
        post_process_mj_model_fns: Optional list of functions to post-process MuJoCo model.
        run_model_upfront: If True, pre-compute all model outputs for all trajectory/seed
            combinations at the start, then unload the model from GPU to save memory.
            The optimizer will be put in cache-only mode after pre-computation.
    """
    optimizer = hardware_optimizer_fn(decoder)
    reward_fns = optimizer.reward_fns
    env: TrackEnv = typing.cast(TrackEnv, runner.env)
    post_process_episode_data_fn = functools.partial(
        label_episode_with_actual_value,
        reward_fns=reward_fns,
    )

    runner.log_dir = log_dir
    pickle_path = env.pickle_path
    trajs = pickle.load(open(pickle_path, "rb"))

    # Phase 1: Pre-compute all model outputs if run_model_upfront is enabled
    if run_model_upfront:
        run_model_upfront_phase(
            optimizer=optimizer,
            trajs=trajs,
            traj_indices=traj_indices,
            num_seeds_per_datapoint=num_seeds_per_datapoint,
            optimize_hardware=optimize_hardware,
        )

    def hardware_reset_fn(_env: BaseEnv, _seed: int, _traj_idx: int | list[int]):
        """
        Reset hardware with optimized design.

        Args:
            _env: Environment to reset.
            _seed: Random seed for optimization.
            _traj_idx: Either a single trajectory index or a list of indices
                for composed multi-trajectory optimization.
        """
        # Handle both single and multi-trajectory cases
        # Check for list-like (handles both Python lists and OmegaConf ListConfig)
        is_multi = (
            hasattr(_traj_idx, "__iter__")
            and hasattr(_traj_idx, "__len__")
            and not isinstance(_traj_idx, (str, int))
        )
        if is_multi:
            traj_list = [trajs[int(idx)] for idx in _traj_idx]
            traj_label = f"trajs {list(_traj_idx)}"
        else:
            traj_list = [trajs[int(_traj_idx)]]
            traj_label = f"traj {_traj_idx}"

        # Warmup is handled internally by the optimizer
        robot_dict, value, optimize_time = optimizer.optimize(
            trajs=traj_list, seed=_seed
        )
        logging.info(f"value: {value:.02f} for {traj_label}, took {optimize_time:.2f}s")
        robot_dict = {k: v.cpu().numpy() for k, v in robot_dict.items()}
        masked_hardware_dict = {}
        for k, v in robot_dict.items():
            if k.endswith("/mask"):
                continue
            group = k.split("/")[0]
            mask_key = group + "/mask"
            if mask_key not in robot_dict:
                logging.warning(f"Missing mask for {k}")
            mask = robot_dict[mask_key].reshape(-1)
            masked_hardware_dict[k] = v[~mask]

        tokenized_robot = deserialize(masked_hardware_dict, include_states=True)
        (
            mjcf_model,
            linkid_to_mjgeomid,
            dyna_jntid_to_mjjntid,
            tracklinkid_to_mjsiteid,
        ) = detokenize(
            tokenized_robot,
            gravcomp=gravcomp,
        )

        if post_process_mjcf_fns is not None:
            for post_process_mjcf_fn in post_process_mjcf_fns:
                mjcf_model = post_process_mjcf_fn(mjcf_model)

        mjgeomid_to_linkid = {
            mjgeomid: linkid for linkid, mjgeomid in linkid_to_mjgeomid.items()
        }
        mjjntid_to_dyna_jntid = {
            mjjntid: dyna_jntid for dyna_jntid, mjjntid in dyna_jntid_to_mjjntid.items()
        }
        mjsiteid_to_tracklinkid = {
            mjsiteid: tracklinkid
            for tracklinkid, mjsiteid in tracklinkid_to_mjsiteid.items()
        }

        _env.hardware_seed = int(_seed)
        _env.post_hardware_reset(
            mjcf_model=_env.post_process_mjcf(mjcf_model),
            tokenized_robot=tokenized_robot,
            mjjntid_to_dyna_jntid=mjjntid_to_dyna_jntid,
            mjgeomid_to_linkid=mjgeomid_to_linkid,
            mjsiteid_to_tracklinkid=mjsiteid_to_tracklinkid,
        )

        if post_process_mj_model_fns is not None:
            for post_process_mj_model_fn in post_process_mj_model_fns:
                _env.m = post_process_mj_model_fn(_env.m)

        _env.hardware_dict["metadata/seed"] = np.array(int(_seed), dtype=np.uint64)
        _env.hardware_dict["metadata/predicted_value"] = np.array(value)
        _env.hardware_dict["metadata/optimize_time"] = np.array(optimize_time)

    temp_data_paths = []

    validity = []

    # Use iterate_traj_seeds() for consistent iteration logic
    for traj_items, hardware_iter, hardware_seed in iterate_traj_seeds(
        traj_indices, num_seeds_per_datapoint
    ):
        hardware_reset_for_traj_fn = functools.partial(
            hardware_reset_fn, _traj_idx=traj_items
        )
        # Create a string representation for path naming (works for both int and list)
        traj_str = str(traj_items)
        # Determine episode_seeds based on whether traj_items is iterable
        is_multi = (
            hasattr(traj_items, "__iter__")
            and hasattr(traj_items, "__len__")
            and not isinstance(traj_items, (str, int))
        )
        episode_seeds = list(traj_items) if is_multi else [traj_items]
        temp_data_path = f"{log_dir}/hardware_opt_{hardware_iter}_{traj_str}.zarr"
        try:
            runner.run_episodes(
                hardware_seed=hardware_seed,
                episode_seeds=episode_seeds,  # since traj_idx is episode_seed % num_trajs, this ensures that the selected trajectory idx is traj_idx
                policy_fn=None,
                hardware_reset_fn=(
                    hardware_reset_for_traj_fn if optimize_hardware else None
                ),
                post_process_episode_data_fn=post_process_episode_data_fn,
                data_path=temp_data_path,
            )
            validity.append(True)
            temp_data_paths.append(temp_data_path)
        except DetokenizeError:
            validity.append(False)
        except np.linalg.LinAlgError:
            # IK didn't converge, most likely because
            # joint transforms was bad
            validity.append(False)
        except Exception as e:
            logging.error(f"seed: {hardware_seed}, traj: {traj_str}, error: {e}")
            validity.append(False)
    avg_validity = float(np.mean(validity))
    if avg_validity == 0:
        # Nothing simulable means there is no rollout to summarize, so no
        # summary zarr gets written and this run contributes no point to the
        # results figure. Say so loudly: the caller exits 0 either way, and a
        # silently missing cell is very hard to trace back to here.
        logging.error(
            f"All {len(validity)} generated designs failed to detokenize or "
            f"simulate, so no {log_dir}/hardware_opt_summary.zarr was written "
            f"and this run produces no results. When using guided diffusion "
            f"this usually means guidance pushed samples off the training "
            f"manifold - lower eval_fn.hardware_optimizer_fn.guidance.scale, "
            f"or lower the guidance reward's embodiment term (see the guidance "
            f"table in docs/visualization.md)."
        )
        return {"validity": torch.tensor(avg_validity)}
    output_path = os.path.join(log_dir, "hardware_opt_summary.zarr")
    concat_zarr_stores(from_paths=temp_data_paths, to_path=output_path)
    for temp_data_path in temp_data_paths:
        shutil.rmtree(temp_data_path)
        lock_path = temp_data_path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)
    runner.close()
    summary_stats = summarize_rollout(output_path)
    summary_stats["validity"] = avg_validity
    return {k: torch.tensor(v) for k, v in summary_stats.items()}
