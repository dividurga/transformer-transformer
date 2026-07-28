import subprocess

from t2.robogen.umi_on_legs_plus_plus import COMPONENTS
from t2.robogen.components import enumerate_choices
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population_size", type=int, default=5)
    parser.add_argument("--max_fun", type=int, default=15)
    parser.add_argument("--n_jobs", type=int, default=0)
    parser.add_argument("--qpos_noise", type=float, default=0.0)
    parser.add_argument("--pos_noise", type=float, default=0.2)
    parser.add_argument("--orn_noise", type=float, default=0.1)
    parser.add_argument("--scale_noise", type=float, default=0.1)
    parser.add_argument("--num_episodes_per_traj", type=int, default=5)
    parser.add_argument("--num_hardware_seeds_per_traj", type=int, default=3)
    parser.add_argument("--start_choice_idx", type=int, default=0)
    parser.add_argument("--end_choice_idx", type=int, default=-1)
    parser.add_argument("--traj_indices", type=int, nargs="+", default=list(range(56)))
    parser.add_argument(
        "--pickle_path", type=str, required=True
    )
    args = parser.parse_args()
    root_path = f"varviper_cmaes_pop{args.population_size}_maxfun{args.max_fun}/"
    choices = enumerate_choices(COMPONENTS)
    choices = list(
        filter(
            lambda x: x[0][0] == 0,
            choices,
        )
    )
    assert len(choices) == 128
    print("found", len(choices), "choices")
    choices = choices[args.start_choice_idx : args.end_choice_idx]
    print(f"running for {len(choices)} choices")
    for choice, num_uniforms in sorted(choices, key=lambda x: str(x[0])):
        choice_str = "[" + ",".join(map(str, choice)) + "]"
        cmd = [
            "python",
            "scripts/optimize_cmaes_quadruped.py",
            "reward_fns=tracking_only",
            f"choices={choice_str}",
            f"num_params={num_uniforms}",
            f"max_fun={args.max_fun}",
            f"n_jobs={args.n_jobs}",
            f"pop_size={args.population_size}",
            "runner.render=false",
            f"runner.env.pickle_path={args.pickle_path}",
            "runner.env.termination_pos_err_threshold=1.0",
            "runner.env.pos_err_sigma=0.01",
            "runner.env.orn_err_sigma=0.5",
            f"runner.qpos_noise={args.qpos_noise}",
            f"runner.env.pos_noise={args.pos_noise}",
            f"runner.env.orn_noise={args.orn_noise}",
            f"runner.env.scale_noise={args.scale_noise}",
            f"traj_indices={args.traj_indices}",
            "tags=[cmaes,quadruped,datagen]",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Output: {e.output}")
            print(f"Return code: {e.returncode}")
            continue
