# Data Generation

This page documents the full data generation pipeline behind Transformer Transformer, from robot tokenization to oracle controllers to the on-disk dataset format.
The pipeline is the same for every design space: procedurally sample an embodiment, tokenize it into RoboTokens, roll out an oracle controller on human motion trajectories, and append everything to a zarr store.

You do not need to run any of this to train or evaluate — [pre-generated data and checkpoints are downloadable](starter.md#checkpoints--data), including [the paper's full training datasets](training.md#hydra-configs).
This page is for regenerating the paper's datasets from scratch, or for generating data in your own design space.

Download the motion trajectory pickles first, since every command below consumes them

```sh
wget -qO- https://real.stanford.edu/transformer-transformer/data.zip | bsdtar -xvf- -C ./
```

> 📘 **Info**
>
> Long-running scripts log to the `transformer-transformer` Weights & Biases project, and some write their outputs into the run's `wandb/run-*/files/` directory.
> Run `wandb login` once during [setup](starter.md#setup), or prefix any command with `WANDB_MODE=offline` if you don't want cloud logging.

## RoboTokens

RoboToken is the representation that makes one model work across embodiments.
Any rigid articulated robot — same scope as an MJCF file — is decomposed into a sequence of continuous-valued tokens: link tokens, fixed joint tokens, dynamic joint tokens (sliding, rotating, and ball joints), and motor tokens describe the embodiment, while state and action tokens describe its time-varying dynamics.
Tokens reference each other through ID attributes (a joint points at its two links, a motor at its joint, a state at its embodiment token and timestep), which become learned positional embeddings in the model.
This handles variable connectivity, passive joints, and heterogeneous state/action spaces without any per-robot code.

The encoding is also compact: 27-110x fewer tokens than feeding the raw MJCF XML to a text tokenizer.
The 11 MuJoCo Menagerie robots we tested — spanning the 0.65 kg Allegro hand to the 67.5 kg ANYmal C, and 6 DoF (UR5e) to 35 DoF (Cassie) — tokenize into sequences of just 28 to 101 tokens.
Crucially, tokenization is invertible: `tokenize()` and `detokenize()` in `t2/robotok/tokenizer.py` round-trip between MJCF and RoboTokens, which is what lets us diffuse tokens and then simulate the result.

Check that tokenization round-trips on hundreds of randomized robots

```sh
pytest tests/robotok
```

Tokenize a single procedurally sampled robot into a minimal one-robot dataset

```sh
python scripts/generate_robotoken_dataset.py --pickle_path data/july25th2025-huy-20skills-train.pkl
```

This writes `wheeled_bimanual.zarr`, containing the tokenized robot under `hardware/` and one environment reset under `rollout/`.
It is the smallest end-to-end example of the design space sampler, the tracking environment, and the zarr schema, so I recommend reading this ~50-line script first.

Tokenize the MuJoCo Menagerie robots from the paper's unification experiment

```sh
python scripts/generate_mj_menagerie_dataset.py --pickle_path data/july25th2025-huy-20skills-train.pkl
```

This writes `mj_menagerie.zarr` with all 11 Menagerie robots plus UMI on Legs, one hardware entry each.

For procedural design spaces, `scripts/enumerate_robogen_robotokens.py` (config `config/generate_robogen_tokens.yaml`) samples `num_hardware` continuous variations of one discrete design choice and tokenizes them all in parallel.

Tokenize 100 continuous variations of one quadruped manipulator design choice

```sh
python scripts/enumerate_robogen_robotokens.py 'choices=[0,0,0,0,0,0,0,0]' num_uniforms=13 num_hardware=100 num_processes=32 output_path=umi_on_legs_plus_plus_robotokens/00000000.zarr
```

Sweep every discrete choice of the quadruped manipulator design space

```sh
python scripts/enumerate_robogen.py
```

This driver calls `enumerate_robogen_robotokens.py` once per discrete choice returned by `enumerate_choices()`, writing one `<choice_bits>.zarr` per choice under `umi_on_legs_plus_plus_robotokens/`.
The paper's own bank — all 138 quadruped choices, pre-tokenized — ships inside `rl_policies.zip`, so you only run this sweep for a design space of your own.
To tokenize a different design space, override `robogen_from_params._target_` to `t2.robogen.viperx.variable_dof_viperx_full_variation_from_params` or `t2.robogen.wheeled_bimanual.wheeled_bimanual_from_params`, and pass the matching `choices` and `num_uniforms` (each design space's `enumerate_choices()` reports the correct `num_uniforms` per choice).

> 📘 **Info**
>
> The model pads each token type to a per-design-space maximum sequence length.
> These token budgets live in the `seq_len` block of the `config/addon_*.yaml` files — for instance, ViperX (`config/addon_hardware_viperx.yaml`) budgets 18 link, 22 fixed joint, 14 dynamic joint, and 7 actuator tokens.
> `config/train.yaml` derives everything else from them, so this is the only place to change when your robots grow.

## Controllers: Mink v.s. RL

To supervise cross-embodiment control (`ctrl`) and give hardware generation (`hardware_gen`) realistic dynamics tokens, every sampled embodiment needs an expert controller.
We use two oracles, split by whether the robot can fall over:

- **Mink (differential IK)** for fixed-base and statically stable robots — ViperX, the ALOHA bimanual space, and the wheeled bimanual space. [Mink](https://github.com/kevinzakka/mink) solves DiffIK with tracking, posture regularization, and damping costs, looking ahead `look_ahead_steps=4` control steps (80 ms at 50 Hz).
- **RL whole-body experts** for the quadruped manipulator, where tracking requires dynamic whole-body coordination. We extend the [UMI on Legs](https://umi-on-legs.github.io/) task formulation and train one PPO expert per discrete design choice — 128 experts, covering the RL-trainable subset of the 138-choice quadruped space (the other 10 choices ship with pre-tokenized robotoken/options stores but no expert) — with continuous design variations handled by domain randomization within each expert.

Both oracles support [PDP](https://arxiv.org/abs/2406.00966)-style noise injection: set `runner.qpos_noise` to execute noisy joint position commands during the rollout while logging the clean commands as `target_qpos`.
Noisy rollouts, clean supervision — the controller you distill becomes robust to its own tracking errors.

### Mink data generation

All Mink data generation runs through `scripts/inference.py`, which fans episodes out over Ray workers and concatenates their shards into one zarr.
The runner is selected by the config's `runner` group: `mink` (single arm), `bimanual_mink` (two fixed-base arms), or `wheeled_bimanual_mink` (two arms on a wheeled base, with inferred per-episode controller weights).
Despite the name, `bimanual_mink` is fixed-base only — it refuses mobile-base robots at hardware reset, so the wheeled design space always goes through `wheeled_bimanual_mink`.

Generate ViperX kinematic-design data with the single-arm Mink oracle

```sh
python scripts/inference.py --config-name=datagen runner=mink 'env@runner.env=variable_viperx' runner.env.pickle_path=data/july25th2025-huy-20skills-train.pkl data_path=data/viperx_mink.zarr num_hardware=1000 num_episodes_per_hardware=10 num_processes=16
```

Generate wheeled bimanual data on the dishwashing demonstrations

```sh
python scripts/inference.py --config-name=datagen_wheeled_bimanual data_path=data/wheeled_bimanual_mink.zarr num_hardware=5000 num_episodes_per_hardware=10 num_processes=16
```

Generate ALOHA bimanual data for the real-world design space

```sh
python scripts/inference.py --config-name=datagen_mink_viperx_bimanual_opposing_70cm runner.env.pickle_path=data/huy-unfold-04292026.pkl data_path=data/huy-unfold-04292026_mink.zarr
```

Each run prints a progress bar, then `Data Gen Time` and `Consolidation Time`, then rollout summary statistics.
Expect lines like `metric/pos_err/q50` (median tracking position error in meters), `metric/reward/sum`, and `done/bad_termination/any` (percentage of episodes that exceeded the termination threshold).
The knobs you will actually touch:

- `num_hardware`: how many embodiments to sample. Hardware seeds default to `0..num_hardware-1`; pass `hardware_seeds=[3,7]` to pin specific ones.
- `num_episodes_per_hardware`: episodes per embodiment. The paper's datasets use 10.
- `runner.env.pickle_path`: the motion trajectory pickle (see [UMI Data Collection & Processing](#umi-data-collection--processing)).
- `data_path`: output zarr path.
- `runner.render=true`: dump mp4s of every episode to `runner.log_dir`. Slow — I only turn this on to spot-check a handful of episodes.

For reference, the paper's dataset scales are 3.8M episodes / 380K embodiments (ViperX), 1.3M / 130K (quadruped manipulator), and 50K / 5K (wheeled bimanual).
At those scales you will want to shard across machines — see [Parallelization](#parallelization).

### RL data generation

The quadruped manipulator pipeline has more moving parts: enumerate the design space, train experts, pick checkpoints, then roll them out.

First, dump the MuJoCo model attributes for the continuous variations of one discrete design choice.
`scripts/train_rl_procedural.py` domain-randomizes over these attributes so a single policy covers the whole continuous slice.

```sh
python scripts/enumerate_robogen_mjcf.py 'choices=[0,0,0,0,0,0,0,0]' num_uniforms=13 num_hardware=20000 num_processes=32 output_path=umi_on_legs_plus_plus_options/00000000.zarr
```

Train the PPO expert for that design choice

```sh
python scripts/train_rl_procedural.py mj_model_data_path=umi_on_legs_plus_plus_options/00000000.zarr env.trajectory_file=data/july25th2025-huy-20skills-train.pkl eval_env.trajectory_file=data/july25th2025-huy-20skills-test.pkl
```

This trains with brax PPO on MJX for 400M environment steps (config `config/run_rl_procedural.yaml`), logging to wandb and saving orbax checkpoints under `wandb/run-*/files/checkpoints/<env_steps>`.
The randomized hardware attributes are appended to the policy observation, so the expert is conditioned on which continuous variation it is controlling.

> ❗**Caution**
>
> Each expert takes roughly 16 A100-hours, and the full expert pool is 128 of them.
> Unless you are changing the design space or the RL task, download our pretrained experts instead:
>
> ```sh
> wget -qO- https://real.stanford.edu/transformer-transformer/rl_policies.zip | bsdtar -xvf- -C ./
> ```
>
> This unpacks everything the quadruped RL paths need: the per-design-choice expert checkpoints, a ready-made `choice_to_ckpt_path.json`, the per-choice MuJoCo model options stores (`umi_on_legs_plus_plus_options/`), and the pre-tokenized RoboToken bank (`umi_on_legs_plus_plus_robotokens/`) that the RL co-design evaluation decodes against.

There is also `scripts/train_rl.py` (config `config/run_rl.yaml`), which trains on the single fixed UMI on Legs embodiment with a randomized arm mount — useful for iterating on the RL task itself without the procedural machinery. It takes the same `env.trajectory_file`/`eval_env.trajectory_file` overrides.

Sanity-check the MJX environment by rendering a random-action rollout

```sh
python scripts/rollout_env.py
```

This writes `video.mp4` of a procedurally generated quadruped manipulator twitching under Gaussian actions.
If the robot spawns intersecting the floor or the target site is missing, fix that before spending GPU-days on PPO.

Once experts are trained, build the map from discrete design choice to best checkpoint

```sh
python scripts/pick_rl_policy.py --roots wandb --output choice_to_ckpt_path.json
```

This scans the wandb run directories, keeps the newest sufficiently trained checkpoint per choice (at least `--min_training_iteration` environment steps, median eval position error below `--max_pos_err` of 0.1 m), and prints any design choices that still lack a usable expert.

Finally, roll out an expert into a training dataset

```sh
python scripts/datagen_rl.py runner.ckpt_path=$(jq -r '."00000000"' choice_to_ckpt_path.json) runner.env.pickle_path=data/july25th2025-huy-20skills-train.pkl data_path=data/quadruped_00000000.zarr num_hardware=1000 num_episodes_per_hardware=10 num_processes=16
```

`datagen_rl.py` reads the `config.pkl` saved next to the checkpoint to recover which design choice the expert was trained on, samples `num_hardware` continuous variations of that choice, and rolls the expert out on each — this time in regular MuJoCo (not MJX) through the same `TrackEnv` used by the Mink runners, so the output schema is identical.

> 🪲 **Troubleshooting missing `mj_model_data_path`**
>
> `datagen_rl.py` resolves the design space through the training run's `config.pkl`, whose `mj_model_data_path` is a path relative to the repository root (e.g. `umi_on_legs_plus_plus_options/00000000.zarr`).
> `rl_policies.zip` ships these stores, so downloaders are covered out of the box; the `enumerate_robogen_mjcf.py` command above is only needed when you build your own design space or experts.

> 📘 **Info**
>
> The oracles above generate training data, and are also the default controllers when evaluating generated designs.
> You can instead roll generated designs out with the model's *own* cross-embodiment controller by selecting `runner=t2_ctrl` (see `config/runner/t2_ctrl.yaml`) — the fully self-contained designer-plus-controller setting we analyze in the paper.

## CMA-ES

[CMA-ES](https://github.com/CMA-ES/pycma) plays two roles in this codebase.

First, it is the classical co-design baseline: sample the discrete design choices, then let CMA-ES optimize the continuous parameters, where every function evaluation is an actual simulated rollout scored by the reward functions in `config/reward_fns/`.
This is what the paper compares Zeroth Order and Dynamics Self-Guidance (DSG) against.

Second, it is a data-quality biasing tool.
In kinematically constrained fixed-base spaces (ViperX and the ALOHA bimanual space), most randomly sampled designs simply cannot reach the demonstrated trajectories, and a dataset of failures teaches the model little.
So we run a deliberately tiny CMA-ES — population 5, `max_fun=15`, i.e. 3 generations — on the *training* trajectories, and keep **every** sampled embodiment, not just the winner.
The point is to bias the sampling distribution toward feasible designs without collapsing its diversity.

Run CMA-ES on the ViperX design space

```sh
python scripts/optimize_cmaes_viperx.py pickle_path=data/july25th2025-huy-20skills-test.pkl
```

Run CMA-ES on the quadruped manipulator design space

```sh
python scripts/optimize_cmaes_quadruped.py pickle_path=data/july25th2025-huy-20skills-test.pkl
```

The quadruped version needs `choice_to_ckpt_path.json` in the repository root (`choice_to_ckpt_path=` to point elsewhere), because each CMA-ES candidate is rolled out by the RL expert matching its discrete choice.

Run CMA-ES on the wheeled bimanual design space

```sh
python scripts/optimize_cmaes_wheeled_bimanual.py pickle_path=data/bimanual_dish_washing_test.pkl
```

Run CMA-ES on the real-world ALOHA bimanual space

```sh
python scripts/optimize_cmaes_viperx.py --config-name=optimize_cmaes_viperx_bimanual
```

All variants share `config/optimize_cmaes_base.yaml`.
If `choices` and `num_params` are left unset (the default), each run item randomizes over the enumerated discrete choices; pin one with `'choices=[0,0,0,0,0]' num_params=6`.
Other knobs worth knowing: `reward_fns=` selects the objective (`tracking_only`, `tracking_velocity`, `tracking_velocity_quadruped`, `tracking_torque`, `tracking_size`, `tracking_size_bimanual`, `tracking_size_quadruped`, `tracking_weight`, `tracking_joint_vel` — the `_quadruped`/`_bimanual` variants carry design-space-tuned constants, the unsuffixed ones are ViperX-tuned), `traj_indices=` selects which trajectories to optimize for (a list of lists triggers multi-trajectory optimization), and `pop_size`/`max_fun`/`init_sigma` control CMA-ES itself.

Each run writes two stores into its wandb run directory: `optimized_hardwares.zarr` (the best design per trajectory, for baseline numbers) and `intermediate_hardwares.zarr` (rollouts of *every* embodiment CMA-ES evaluated).
For data-quality biasing, the intermediate store is the product — concatenate them across runs with `scripts/concat_datasets.py` and train on that.

Sweep CMA-ES over every discrete design choice

```sh
python scripts/optimize_cmaes_all_viperx_choices.py --pickle_path data/july25th2025-huy-20skills-train.pkl
```

```sh
python scripts/optimize_cmaes_all_quadruped_choices.py --pickle_path data/july25th2025-huy-20skills-train.pkl
```

These drivers enumerate the discrete choices (468 for ViperX, 128 for the quadruped) and launch one CMA-ES run per choice over all 56 training trajectories, with the biasing defaults (population 5, `max_fun` 15) baked in.
Use `--start_choice_idx`/`--end_choice_idx` to split the sweep across machines.

## Parallelization

Everything here parallelizes with [Ray](https://www.ray.io/) on a single machine — `ray.init(num_cpus=num_processes)`, no cluster setup.

- `num_processes` controls the worker count for `scripts/inference.py`, `scripts/datagen_rl.py`, and the tokenization scripts.
- `num_gpus_per_worker` (for `scripts/inference.py`) reserves GPU fractions per worker; the Mink runners are CPU-only, so it defaults to 0.
- Each worker writes its own shard (`<data_path>_00`, `<data_path>_01`, ...), which the script concatenates into `data_path` at the end and deletes — no locking during generation.
- Progress bars come from `wait_with_pbar` in `t2/utils/ray.py`: one rich bar per task group with tasks/s and ETA. Ctrl-C aborts the remaining tasks but keeps completed results.

## UMI Data Collection & Processing

All tasks are specified as gripper pose trajectories, collected without any robot.
For gripper-based collection, use the original [UMI](https://github.com/real-stanford/universal_manipulation_interface) codebase and hardware.
For iPhone-based collection — strap an iPhone to the UMI gripper and use ARKit odometry instead of SLAM — use [Austin Patel](https://austinapatel.github.io/)'s excellent [iPhUMI](https://github.com/real-stanford/iPhUMI) codebase.
The paper's 76 single-arm trajectories (56 train / 20 test, spanning 20 manipulation skills) were collected with iPhUMI; the bimanual dishwashing trajectories came from UMI's SLAM pipeline.

Convert iPhUMI recordings into a motion trajectory pickle

```sh
python scripts/process_iphumi.py --dir_path iphumi_recordings/ --output_path data/my_motions.pkl --zero_min_z
```

This recursively finds pose JSONs under `--dir_path`, pairs each directory's `left.json` and `right.json` into one bimanual episode, and slerps both arms onto a shared 50 Hz clock (`--output_dt 0.02`) over the time window where both recordings overlap.
`--zero_min_z` shifts each episode so its lowest end-effector height is z=0, which I use to pin table-top demonstrations to the ground plane.
Expect a printout of episode length statistics and the total episode count before the pickle is saved.

Convert a UMI zarr dataset into train/test motion trajectory pickles

```sh
python scripts/process_umi_data.py --zarr_path data/bimanual_dish_washing.zarr --output_path data/bimanual_dish_washing.pkl --seed 0
```

This reads the UMI end-effector poses (`data/robot{0,1}_eef_pos`, `..._rot_axis_angle`, `meta/episode_ends`), resamples from `--input_hz` (59.94, the GoPro frame rate) to `--output_dt` (0.02 s = 50 Hz), and writes `<stem>_train.pkl` / `<stem>_test.pkl` split by `--train_split` (0.9).
`--center_init_xy_pos` re-centers each episode's initial mean xy position at the origin, and `--global_z_rotation` / `--global_z_pos_offset` rotate and lift the world frame into the robot's convention (the 0.4 m default puts trajectories at table height).

> ❗**Caution**
>
> Pass `--seed 0` for the dishwashing dataset.
> The script asserts it: with that split, test trajectory 21 has its left/right arms swapped in the recording, and the fix is keyed to that exact seed.

Both scripts output the same format: a Python list of float arrays of shape `(T, n_arms, 4, 4)` — per-timestep SE(3) end-effector poses at 50 Hz, with `n_arms` 1 for single-arm and 2 for bimanual.
This is what every `runner.env.pickle_path` consumes.
At reset, the environment selects trajectory `episode_seed % len(trajs)`, so episode seeds double as trajectory indices.

## Data Format

Every dataset in this codebase — Mink, RL, CMA-ES, or evaluation output — is a zarr store with the same four top-level groups:

```
<dataset>.zarr/
├─ rollout/          # time-indexed arrays; first axis is all episodes concatenated (N total steps)
│  ├─ ctrl                (N, n_actuators, 2)        commanded + observed joint positions
│  ├─ dyna_joint_obs      (N, n_joints, 2)           qpos, qvel per joint
│  ├─ actuator_obs        (N, n_actuators, 2)        force, velocity per actuator
│  ├─ track_link_obs      (N, n_end_effectors, 12)   achieved EE pose (pos + rotmat)
│  ├─ target_pose         (N, n_end_effectors, 12)   commanded EE pose
│  ├─ free_link_obs       (N, n_free_links, 12)      floating-base spaces only
│  ├─ metric              (N, 1, n_metrics)          per-step scalars (pos_err, reward, ...)
│  └─ done                (N, 1, 3)                  termination flags
├─ rollout_meta/
│  ├─ ends                (E,)  cumulative episode end row
│  ├─ seed                (E,)  episode seed (doubles as trajectory index)
│  └─ hardware_id         (E,)  row into hardware_meta
├─ hardware/         # RoboTokens; rows concatenated across embodiments
│  └─ link, dyna_joint, fixed_joint, actuator
└─ hardware_meta/
   ├─ <group>/ends        (H,)  cumulative token-row end per embodiment
   ├─ seed                (H,)  hardware seed
   └─ robot_generator     (H,)  string repr of the generator that built the MJCF
```

Three rules make any array in the store self-describing:

**1. The last axis is alphabetical sub-key slabs.**
Each array's zarr attributes map sub-key names to their widths, and slabs are concatenated along the last axis in alphabetical order of the sub-key.

```python
import zarr
root = zarr.open("data/huy-unfold-04292026_mink.zarr", mode="r")
root["rollout/ctrl"].attrs.asdict()            # {"observed_qpos": 1, "target_qpos": 1}
root["rollout/track_link_obs"].attrs.asdict()  # {"pos": 3, "rotmat": 9}
```

So `ctrl[..., 0]` is `observed_qpos` and `ctrl[..., 1]` is `target_qpos`; `track_link_obs[..., :3]` is position and `track_link_obs[..., 3:]` is the row-major 3x3 rotation matrix.
The same rule decodes `metric` (alphabetical: `actuator/energy/*`, `actuator/force/*`, `orn_err`, `pos_err`, `reward`) and `done` (`bad_termination`, `timeout`, `unstable`).

**2. `ends` arrays are cumulative.**
Episode `i` occupies rows `[starts[i], ends[i])` in every `rollout/*` array:

```python
import numpy as np
ends = root["rollout_meta/ends"][:]
starts = np.concatenate([[0], ends[:-1]])
```

`hardware_meta/<group>/ends` works identically for slicing embodiment `j`'s token rows out of `hardware/<group>`.
`rollout_meta/hardware_id[i]` joins the two: it is the row into `hardware_meta` (and thus the `ends`-slice into `hardware/*`) for the embodiment that episode `i` ran on.

**3. The root attributes carry full provenance.**
`root.attrs.asdict()` contains the complete Hydra config of the generating run (so `root.attrs["runner"]["env"]["ctrl_dt"]` tells you the control rate, and `runner.env.pickle_path` the source motion data) plus a `metadata` dict with the git commit, date, and hostname.
(The released training sets are lightly scrubbed: their `metadata` keeps only `date`, and concatenated sets record their inputs as relative `from_paths` — datasets you generate yourself carry the full record.)
Episodes shorter than 100 steps (2 s) are filtered at train time by `dataset.min_episode_length`, not at generation time — the store keeps everything.

### Replaying actions on a real robot

The `target_qpos` slab of `rollout/ctrl` is a directly replayable joint position command stream: radians, 50 Hz (`ctrl_dt=0.02`), in MuJoCo actuator order — which is the actuator declaration order of the design space's MJCF.
This is exactly how we replayed the optimized ALOHA design's trajectories on real hardware (dataset generated by the `datagen_mink_viperx_bimanual_opposing_70cm` command [above](#controllers-mink-vs-rl)).

> ❗**Caution**
>
> The runner logs each command *before* checking termination, so the last command of every episode was produced by the oracle but never executed in simulation.
> For exact replay, drop the final row of each episode (`starts[i] .. ends[i]-2`).

```python
import time
import numpy as np
import zarr

root = zarr.open("data/huy-unfold-04292026_mink.zarr", mode="r")
ends = root["rollout_meta/ends"][:]
starts = np.concatenate([[0], ends[:-1]])
ctrl_dt = root.attrs["runner"]["env"]["ctrl_dt"]  # 0.02 s -> 50 Hz
target_idx = sorted(root["rollout/ctrl"].attrs.asdict()).index("target_qpos")

def replay_episode(robot, episode_idx: int):
    s, e = int(starts[episode_idx]), int(ends[episode_idx])
    for qpos in root["rollout/ctrl"][s : e - 1, :, target_idx]:
        robot.send_position_command(qpos)  # radians, MJCF actuator order
        time.sleep(ctrl_dt)
```

A real driver should additionally move the robot slowly to the first commanded pose before playback (trajectories do not start at home), clamp commands to its own joint limits, and hold the grippers fixed — the ALOHA data-generation MJCF does not actuate fingers.

To eyeball any dataset before committing GPU-hours to it, `scripts/visualize_dataset.py` detokenizes the stored RoboTokens back into MJCF and steps through episodes in the MuJoCo viewer

```sh
python scripts/visualize_dataset.py dataset.path=data/viperx_mink.zarr
```

Seeing the reconstructed robot track its target poses is also the quickest end-to-end check that tokenization, the alphabetical-slab decoding, and your episode bookkeeping all agree.
