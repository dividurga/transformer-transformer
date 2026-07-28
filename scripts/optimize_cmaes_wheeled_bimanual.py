import glob
import logging
import os
import shutil
import zarr
import time
import hashlib
import numpy as np

import hydra
import ray
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

import wandb
from t2.eval.hardware_optimization import run_cmaes_on_traj_and_hardware_seed
from t2.eval.utils import summarize_rollout
from t2.io.schema import concat_zarr_stores, get_flattened_leaf_keys
from t2.robogen.components import enumerate_choices
from t2.robogen.wheeled_bimanual import COMPONENTS
from t2.utils.misc import (
    ensure_wandb_run_metadata,
    flatten_dict,
    seed_everything,
)
from t2.utils.ray import wait_with_pbar

from itertools import product


def is_multi_traj_mode(traj_indices_raw: list) -> bool:
    """Check if traj_indices contains lists (multi-traj mode) or ints (single-traj mode).

    Args:
        traj_indices_raw: List from config, either [0, 1, 2] or [[0, 1], [2, 3]]

    Returns:
        True if multi-traj mode (contains lists), False if single-traj mode (contains ints)
    """
    return len(traj_indices_raw) > 0 and isinstance(traj_indices_raw[0], (list, tuple))


def format_traj_str(traj_items: int | list[int], is_multi: bool) -> str:
    """Format trajectory items for path naming.

    Args:
        traj_items: Single trajectory index or list of indices
        is_multi: Whether we're in multi-trajectory mode

    Returns:
        Formatted string for use in paths, e.g., "traj00" or "trajs_0_1"
    """
    if is_multi:
        indices_str = "_".join(str(idx) for idx in traj_items)
        return f"trajs_{indices_str}"
    else:
        return f"traj{traj_items:02d}"


@hydra.main(
    config_path="../config",
    config_name="optimize_cmaes_wheeled_bimanual",
    version_base="1.3",
)
def main(cfg):
    seed_everything(cfg.seed)
    flattened_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True), sep="/")  # type: ignore
    wandb.init(project="transformer-transformer", config=flattened_cfg, tags=cfg.tags)
    assert wandb.run is not None
    ensure_wandb_run_metadata(wandb.run)
    cfg.runner.log_dir = wandb.run.dir

    def runner_fn(_runner_cfg: DictConfig = cfg.runner, _log_dir: str = wandb.run.dir):
        return hydra.utils.instantiate(_runner_cfg, log_dir=_log_dir)

    optimized_hardware_path = f"{wandb.run.dir}/optimized_hardwares.zarr"
    optimized_hardware_paths = []
    logging.info(f"Optimized hardware path: {optimized_hardware_path}")
    intermediate_hardware_path = f"{wandb.run.dir}/intermediate_hardwares.zarr"
    intermediate_temp_dir = f"{wandb.run.dir}/cmaes_intermediate"
    os.makedirs(intermediate_temp_dir, exist_ok=True)
    tasks = []
    run_cmaes_on_traj_and_hardware_seed_ray = ray.remote(
        run_cmaes_on_traj_and_hardware_seed
    )
    reward_fns = hydra.utils.instantiate(cfg.reward_fns)
    try:
        choice_options = [(cfg.choices, cfg.num_params)]
        task_desc = f"optimize {cfg.choices} with {cfg.num_params} params"
    except MissingMandatoryValue:
        # Enumerate all choice combinations for wheeled_bimanual
        choice_options = enumerate_choices(COMPONENTS, resolve_before_mount=True)
        task_desc = "randomizing over choices"

    # Detect multi-trajectory mode: [[0, 1], [2, 3]] vs [0, 1, 2, 3]
    traj_indices_raw = OmegaConf.to_container(cfg.traj_indices, resolve=True)
    is_multi = is_multi_traj_mode(traj_indices_raw)
    if is_multi:
        logging.info(
            "Multi-trajectory mode enabled: optimizing across trajectory groups"
        )

    for traj_items, episode_seed_base, hardware_iter in product(
        cfg.traj_indices,
        range(cfg.num_episodes_per_traj),
        range(cfg.num_hardware_seeds_per_traj),
    ):
        # Convert traj_items to a consistent string representation for hashing
        traj_str = str(traj_items)
        hashobj = hashlib.sha256(f"{hardware_iter}{traj_str}{cfg.seed}".encode())
        rs = np.random.RandomState(
            int(hashobj.hexdigest(), 16) % (np.iinfo(np.int32).max)
        )
        hardware_seed = int(rs.randint(0, np.iinfo(np.int32).max))
        choice, num_params = choice_options[rs.randint(0, len(choice_options))]

        # Convert to list format expected by run_cmaes_on_traj_and_hardware_seed
        traj_indices_list = list(traj_items) if is_multi else [traj_items]

        # Format path name based on mode
        traj_path_str = format_traj_str(traj_items, is_multi)
        _optimized_hardware_path = (
            f"{wandb.run.dir}/{traj_path_str}_seed{hardware_seed:06d}_optimized.zarr"
        )
        optimized_hardware_paths.append(_optimized_hardware_path)
        tasks.append(
            run_cmaes_on_traj_and_hardware_seed_ray.remote(
                runner_fn=runner_fn,
                optimize_runner_fn=runner_fn,
                reward_fns=reward_fns,
                traj_indices=traj_indices_list,
                hardware_seed=hardware_seed,
                data_path=_optimized_hardware_path,
                cmaes_dir=intermediate_temp_dir,
                choices=choice,
                robogen_from_params_fn=hydra.utils.instantiate(
                    cfg.robogen_from_params_fn
                ),
                episode_seed_base=episode_seed_base,
                # cmaes params
                num_params=num_params,
                init_sigma=cfg.init_sigma,
                pop_size=cfg.pop_size,
                max_fun=cfg.max_fun,
                n_jobs=cfg.n_jobs,
            )
        )
        if len(tasks) > 20:
            wait_with_pbar({task_desc: tasks})[task_desc]
            tasks = []
    wait_with_pbar({task_desc: tasks})[task_desc]

    start_time = time.time()
    fn = ray.remote(concat_zarr_stores)
    flattened_keys = get_flattened_leaf_keys(
        zarr.group(zarr.storage.LocalStore(optimized_hardware_paths[0]))
    )
    tasks = [
        fn.remote(
            from_paths=optimized_hardware_paths,
            to_path=optimized_hardware_path,
            root_metadata={"from_paths": optimized_hardware_paths},
            overwrite=True,
            use_pbar=False,
            flattened_keys=[flattened_key],
        )
        for flattened_key in flattened_keys
    ]
    wait_with_pbar({"concatenating optimized": tasks})["concatenating optimized"]
    end_time = time.time()
    logging.info(f"Concatenation Time: {end_time - start_time} seconds")

    summary_stats = summarize_rollout(optimized_hardware_path)
    wandb.log(data=summary_stats)

    for optimized_hardware_path in optimized_hardware_paths:
        shutil.rmtree(optimized_hardware_path)
        lock_path = optimized_hardware_path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)

    for k, v in summary_stats.items():
        if k == "metric/pos_err/mean":
            logging.info(f"{k}: {v * 100:.1f}cm")
        elif k == "metric/orn_err/mean":
            logging.info(f"{k}: {v * 180 / np.pi:.1f}deg")
        elif k == "optimize_time":
            logging.info(f"{k}: {v:.1f}s")
        elif "actual_value" in k:
            logging.info(f"{k}: {v:.1f}")
        elif any(k.endswith(suffix) for suffix in ["/any"]):
            logging.info(f"{k}: {v * 100:.1f}%")

    intermediate_data_paths = [
        str(path) for path in glob.glob(os.path.join(intermediate_temp_dir, "*.zarr"))
    ]
    start_time = time.time()
    fn = ray.remote(concat_zarr_stores)
    tasks = [
        fn.remote(
            from_paths=intermediate_data_paths,
            to_path=intermediate_hardware_path,
            root_metadata={"from_paths": intermediate_data_paths},
            flattened_keys=[flattened_key],
            overwrite=True,
            use_pbar=False,
        )
        for flattened_key in flattened_keys
    ]
    wait_with_pbar({"concatenating intermediate": tasks})["concatenating intermediate"]
    end_time = time.time()
    logging.info(f"Consolidation Time: {end_time - start_time} seconds")
    start_time = time.time()
    shutil.rmtree(intermediate_temp_dir)
    end_time = time.time()
    logging.info(f"Cleanup Time: {end_time - start_time} seconds")
    exit(0)


if __name__ == "__main__":
    main()
