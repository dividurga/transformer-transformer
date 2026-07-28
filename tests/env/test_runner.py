import os
import pickle
import tempfile

import hydra
import numpy as np

from t2.env.ik_runner import IKEnvRunner
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv


def get_env():
    with hydra.initialize(config_path="../../config/env", version_base=None):
        cfg = hydra.compose(config_name="viperx")
    # synthesize a short single-arm end-effector trajectory so the test
    # needs no downloaded data: (T, n_end_effectors, 4, 4) SE(3) poses
    traj = np.tile(np.eye(4), (50, 1, 1, 1))
    traj[..., :3, 3] = [0.3, 0.0, 0.3]
    pickle_path = os.path.join(tempfile.mkdtemp(), "trajs.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump([traj], f)
    cfg.pickle_path = pickle_path
    env: TrackEnv = hydra.utils.instantiate(cfg)
    return env


def test_runner():
    env = get_env()
    runner = EnvRunner(env=env, log_dir="/tmp/", render=True)
    runner.run_episodes(episode_seeds=[0], hardware_seed=1)


def test_ik_runner():
    env = get_env()
    runner = IKEnvRunner(
        env=env,
        log_dir="/tmp/",
        render=False,
        look_ahead_steps=0,
        ik_gamma=1.0,
        qpos_noise=0.0,
        pos_noise=0.0,
        orn_noise=0.0,
        ik_max_steps=100,
        ik_max_attempts=10,
        local_frame_perturb=False,
    )
    runner.run_episodes(
        episode_seeds=[0],
        hardware_seed=0,
        data_path="/tmp/test.zarr",
    )


if __name__ == "__main__":
    test_ik_runner()
