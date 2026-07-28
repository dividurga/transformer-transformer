import copy
import functools
import os
import pickle
import shutil
import time
import warnings
from typing import Callable

# Suppress Zarr v3 warnings early
warnings.filterwarnings("ignore", message=r".*does not have a Zarr V3 specification.*")
warnings.filterwarnings("ignore", message=r".*is currently not part in the Zarr format 3 specification.*")

import hydra
import numpy as np
import ray
import zarr
from dm_control import mjcf
from omegaconf import OmegaConf

from t2.eval.utils import summarize_rollout
from t2.io.schema import concat_zarr_stores
from t2.utils.ray import wait_with_pbar


def robogen_with_choice(
    seed: int,
    robogen_from_params: Callable[[list[int], list[float]], mjcf.RootElement],
    choices: list[int],
    num_uniforms: int,
):
    rs = np.random.RandomState(seed)
    return robogen_from_params(
        choices=copy.deepcopy(choices),  # pyright: ignore[reportCallIssue]
        uniforms=rs.uniform(0, 1, num_uniforms).tolist(),
    )


@hydra.main(config_path="../config", config_name="datagen", version_base="1.3")
def main(cfg):
    if os.path.exists(cfg.data_path):
        print(cfg.data_path, "exists")
        exit()
    hardware_seeds = cfg.hardware_seeds
    if hardware_seeds is None:
        hardware_seeds = list(range(cfg.num_hardware))
    start_time = time.time()

    ray.init(num_cpus=cfg.num_processes)

    ckpt_path = os.path.abspath(cfg.runner.ckpt_path)
    config_path = ckpt_path.split("/checkpoints/")[0] + "/config.pkl"
    config = pickle.load(open(config_path, "rb"))
    mj_model_data_path = config.mj_model_data_path
    root = zarr.open(mj_model_data_path, mode="r")
    choices: list[int] = root.attrs["choices"]  # pyright: ignore[reportArgumentType]
    robogen_fn = hydra.utils.instantiate(root.attrs["robogen_fn"])
    num_uniforms = int(root.attrs["num_uniforms"])  # pyright: ignore[reportArgumentType]

    # Create remote runner class
    @ray.remote(num_cpus=1)
    class RemoteRunner:
        def __init__(self, config, robogen: Callable[[int], mjcf.RootElement]):
            env = hydra.utils.instantiate(config.runner.env, robot_generator=robogen)
            self.__runner = hydra.utils.instantiate(config.runner, env=env)

        def run_episodes(self, **kwargs):
            return self.__runner.run_episodes(**kwargs)

    robot_generator = functools.partial(
        robogen_with_choice,
        robogen_from_params=robogen_fn,
        choices=choices,
        num_uniforms=num_uniforms,
    )

    # Create remote runners
    runners = [
        RemoteRunner.remote(cfg, robot_generator) for _ in range(cfg.num_processes)
    ]
    runner_paths = [
        cfg.data_path + f"_{i:02d}"
        for i, _ in zip(range(cfg.num_processes), hardware_seeds)
    ]
    rs = np.random.RandomState(seed=cfg.episode_seed)
    wait_with_pbar(
        {
            "inference": [
                runners[runner_id % len(runners)].run_episodes.remote(
                    episode_seeds=(
                        rs.randint(
                            0,
                            np.iinfo(np.int32).max,
                            cfg.num_episodes_per_hardware,
                        )
                        if cfg.randomized_episode_seeds
                        else list(range(cfg.num_episodes_per_hardware))
                    ),
                    hardware_seed=hardware_seed,
                    # allow each process to write to a different dataset
                    # to avoid race conditions
                    data_path=runner_paths[runner_id % len(runners)],
                )
                for runner_id, hardware_seed in enumerate(hardware_seeds)
            ]
        }
    )
    end_time = time.time()
    print(f"Data Gen Time: {end_time - start_time} seconds")
    start_time = time.time()
    concat_zarr_stores(
        from_paths=runner_paths,
        to_path=cfg.data_path,
        root_metadata=OmegaConf.to_container(cfg),
    )
    end_time = time.time()
    print(f"Consolidation Time: {end_time - start_time} seconds")
    for runner_path in runner_paths:
        shutil.rmtree(runner_path)
        lock_path = runner_path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)
    summary_stats = summarize_rollout(cfg.data_path)
    for k, v in summary_stats.items():
        if k.startswith("metric/actuator"):
            continue
        if (
            any(k.endswith(suffix) for suffix in ["/q95", "/q50", "/mean"])
            or k == "metric/reward/sum"
        ):
            print(f"{k}: {v:.3f}")
        elif any(k.endswith(suffix) for suffix in ["/any"]):
            print(f"{k}: {v * 100:.1f}%")


if __name__ == "__main__":
    main()
