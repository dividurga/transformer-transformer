import argparse

from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv
from t2.robogen.wheeled_bimanual import wheeled_bimanual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal example: tokenize one procedurally sampled robot into a zarr."
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="Motion trajectory pickle, e.g. data/july25th2025-huy-20skills-train.pkl",
    )
    parser.add_argument("--output_path", type=str, default="wheeled_bimanual.zarr")
    args = parser.parse_args()

    output_path = args.output_path
    robot_generator = wheeled_bimanual
    env = TrackEnv(
        episode_len=1,
        sim_dt=0.005,
        ctrl_dt=0.02,
        robot_generator=robot_generator,
        pos_noise=0.00,
        orn_noise=0.00,
        scale_noise=0.00,
        noise_sample_prob=0.00,
        pos_err_sigma=0.1,
        orn_err_sigma=0.25,
        termination_pos_err_threshold=0.00,
        center_traj=True,
        obs_time_indices=[0],
        include_obs_time_indices="none",
        pickle_path=args.pickle_path,
        n_end_effectors=2,
    )
    runner = EnvRunner(
        env=env,
        log_dir=output_path,
        render=False,
    )
    runner.run_episodes(
        hardware_seed=0,
        episode_seeds=[0],
        data_path=output_path,
    )
    runner.close()
