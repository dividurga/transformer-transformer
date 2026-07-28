import os

# Set before any JAX import so XLA picks them up in both main process and
# Ray workers (which inherit os.environ via spawn).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")
# Force non-interactive matplotlib backend in workers — the default Tk backend
# is not thread-safe and SIGSEGVs Ray workers during figure GC.
os.environ.setdefault("MPLBACKEND", "Agg")

import shutil
import time
import warnings

# Suppress Zarr v3 warnings early (before any zarr imports in workers)
warnings.filterwarnings("ignore", message=r".*does not have a Zarr V3 specification.*")
warnings.filterwarnings("ignore", message=r".*is currently not part in the Zarr format 3 specification.*")

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf

from t2.eval.utils import summarize_rollout
from t2.io.schema import concat_zarr_stores
from t2.utils.ray import wait_with_pbar


@hydra.main(
    config_path="../config",
    config_name="datagen",
    version_base="1.3",
)
def main(cfg):
    hardware_seeds = cfg.hardware_seeds
    if hardware_seeds is None:
        hardware_seeds = list(range(cfg.num_hardware))
    start_time = time.time()

    ray.init(num_cpus=cfg.num_processes)

    # Create remote runner class
    @ray.remote(num_cpus=1)
    class RemoteRunner:
        def __init__(self, config):
            self.__runner = hydra.utils.instantiate(config)

        def run_episodes(self, **kwargs):
            return self.__runner.run_episodes(**kwargs)

    # Create remote runners
    RemoteRunnerWithGPU = RemoteRunner.options(num_gpus=cfg.num_gpus_per_worker)
    runners = [RemoteRunnerWithGPU.remote(cfg.runner) for _ in range(cfg.num_processes)]
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
