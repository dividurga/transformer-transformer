import argparse
import os
import shutil

from dm_control import mjcf
from tqdm import tqdm

from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv
from t2.io.schema import concat_zarr_stores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize the MuJoCo Menagerie robots into one zarr dataset."
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="Motion trajectory pickle, e.g. data/july25th2025-huy-20skills-train.pkl",
    )
    parser.add_argument("--output_path", type=str, default="mj_menagerie.zarr")
    args = parser.parse_args()

    output_path = args.output_path
    paths = [
        "assets/mjcf/wonik_allegro/left_hand.xml",
        "assets/mjcf/anybotics_anymal_c/anymal_c.xml",
        "assets/mjcf/anybotics_anymal_b/anymal_b.xml",
        "assets/mjcf/unitree_h1/h1.xml",
        "assets/mjcf/unitree_go1/go1.xml",
        "assets/mjcf/unitree_go2/go2.xml",
        "assets/mjcf/unitree_a1/a1.xml",
        "assets/mjcf/agility_cassie/cassie_collision_enabled.xml",
        "assets/mjcf/aloha/partial_aloha.xml",
        "assets/mjcf/universal_robots_ur5e/ur5e.xml",
        "assets/mjcf/universal_robots_ur10e/ur10e.xml",
        "assets/mjcf/umi_on_legs/umi_on_legs.xml",
    ]
    temp_output_paths = []
    for path in tqdm(paths, desc="Generating MJCF dataset", dynamic_ncols=True):
        tmp_output_path = output_path + f"_{path.split('/')[-1].split('.')[0]}"
        temp_output_paths.append(tmp_output_path)
        env = TrackEnv(
            episode_len=1,
            sim_dt=0.005,
            ctrl_dt=0.02,
            robot_generator=lambda seed: mjcf.from_path(path),
            remove_visual_asset=False,
            obs_time_indices=[0],
            include_obs_time_indices="none",
            pickle_path=args.pickle_path,
            pos_noise=0.00,
            orn_noise=0.00,
            scale_noise=0.00,
            noise_sample_prob=0.00,
            pos_err_sigma=0.1,
            orn_err_sigma=0.25,
            termination_pos_err_threshold=0.00,
            center_traj=True,
        )
        runner = EnvRunner(
            env=env,
            log_dir=tmp_output_path,
            render=False,
        )
        runner.run_episodes(
            hardware_seed=0,
            episode_seeds=[0],
            data_path=tmp_output_path,
        )
        runner.close()

    concat_zarr_stores(
        from_paths=temp_output_paths,
        to_path=output_path,
    )
    for temp_output_path in temp_output_paths:
        shutil.rmtree(temp_output_path)
        lock_path = temp_output_path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)
