# Getting Started

This page takes you from a fresh clone to evaluating a pretrained Transformer Transformer on both of its tasks — cross-embodiment control (`ctrl`) and motion-to-robot co-design, called hardware generation (`hardware_gen`) in the configs.

## Setup

I've tested this codebase on Ubuntu with NVIDIA GPUs.
The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for environment management — uv will fetch the right Python for you, so the only real prerequisite is an NVIDIA driver recent enough for CUDA 13 (older drivers make the CUDA wheels silently fall back to CPU).

Install uv if you don't already have it.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository.

```sh
git clone https://github.com/real-stanford/transformer-transformer.git
cd transformer-transformer
```

Install all dependencies into a project-local virtual environment, then activate it.

```sh
uv sync --extra dev
source .venv/bin/activate
```

On Linux, uv resolves CUDA-enabled builds of both PyTorch (cu130 index, with kernels for everything up to Blackwell-generation GPUs) and JAX automatically.
On macOS you'll get CPU-only wheels, which are enough for the smoke test below but not for evaluation or training.

Log into Weights & Biases.

```sh
wandb login
```

> 📘 **Info**
>
> Every script in this repo logs to the `transformer-transformer` W&B project unconditionally.
> If you'd rather keep everything local, prefix any command with `WANDB_MODE=offline`.

Check your install by round-tripping hundreds of randomized robots and MuJoCo Menagerie models through the RoboToken tokenizer.
This runs entirely on CPU and needs no downloads.

```sh
pytest tests/robotok
```

Run this from the repo root — the Menagerie test cases load MJCFs from `assets/mjcf/` via relative paths.
It takes a minute or two, and you should see `811 passed` at the end on every platform.
If tokenize → detokenize → simulate reproduces the same link poses, joint states, and contacts for every robot, your MuJoCo stack is healthy.

> 📘 **Info**
>
> `ffmpeg` and Blender (>= 4.0) are only needed for the rendering pipelines in the [visualization docs](visualization.md).
> You can skip both for everything on this page.

## Checkpoints & Data

All release artifacts are hosted at [real.stanford.edu/transformer-transformer](https://real.stanford.edu/transformer-transformer/).
If `bsdtar` is missing on Ubuntu, `sudo apt install libarchive-tools`.

The commands below stream each archive straight into extraction without keeping the zip around.
If you'd rather verify checksums first (or keep the archives), download them to disk instead — a `SHA256SUMS` file covering all four zips lives at the same URL.

```sh
wget https://real.stanford.edu/transformer-transformer/{SHA256SUMS,checkpoints.zip,data.zip}
sha256sum -c SHA256SUMS --ignore-missing   # then: bsdtar -xf checkpoints.zip -C ./
```

Download the pretrained Transformer Transformer checkpoints.

```sh
wget -qO- https://real.stanford.edu/transformer-transformer/checkpoints.zip | bsdtar -xvf- -C ./
```

This unpacks one directory per design space, with one subdirectory per training run.

```
checkpoints/
  viperx/                 # ViperX kinematic design space
    hw3n9bux/{032.pt, cfg.pkl}      # hardware generation (all paper viperx results)
  umi_on_legs_plus_plus/  # quadruped manipulator
    xrh4wk4l/{040.pt, cfg.pkl}      # hardware generation (all paper quadruped results)
    b9n2rc6s/{024.pt, cfg.pkl}      # hardware generation + control multi-task
    ...
  wheeled_bimanual/       # wheeled bimanual (dish washing task complexity space)
    mgoc83ra/{035.pt, cfg.pkl}      # hardware generation
    z4454nxj/{045.pt, cfg.pkl}      # cross-embodiment control
    ...
```

Each run directory holds the model weights next to that run's training config (`cfg.pkl`).
The evaluation scripts load `cfg.pkl` from the directory containing whatever `ckpt_path` you give them, so keep the two files together.

Every released run, what it was trained for, and its capacity (the backbone yamls live under `config/model/backbone/`):

| Run | Design space | Tasks | Backbone | Notes |
| --- | --- | --- | --- | --- |
| `hw3n9bux` | ViperX | hardware gen | `small` (256 × 8 blocks) | all paper ViperX results |
| `xrh4wk4l` | quadruped | hardware gen | `small` | all paper quadruped co-design results |
| `d7lra7j0` | quadruped | hardware gen + control | `medium` (512 × 12, the paper's "Large") | multi-task run behind the paper's qualitative quadruped figure |
| `b9n2rc6s` | quadruped | hardware gen + control | `large` (1024 × 8) | multi-task |
| `2xdws25b` | quadruped | hardware gen + control | `large` + `model.backbone.depth=18` | multi-task |
| `mgoc83ra` | wheeled bimanual | hardware gen | `small` | main bimanual co-design |
| `ac028tq7` | wheeled bimanual | hardware gen | `small` | additional hardware-generation run |
| `z4454nxj` | wheeled bimanual | hardware gen + control | `small` | cross-embodiment control evaluation |
| `52cbmwin` | wheeled bimanual | hardware gen + control | `small` | capacity ablation, small |
| `u5iyxc4d` | wheeled bimanual | hardware gen + control | `medium` | capacity ablation, large |

These capacities were read back from the shipped `cfg.pkl` files, so the table is authoritative — `medium.yaml` is what the paper calls "Large" (~63.6M params), and the two 1024-wide multi-task runs use `large.yaml`.

Download the motion trajectory data.

```sh
wget -qO- https://real.stanford.edu/transformer-transformer/data.zip | bsdtar -xvf- -C ./
```

This unpacks the human demonstration pickles into `data/` with the exact filenames the configs and documented commands expect — the 20-skill UMI collection used by the ViperX and quadruped spaces (`data/july25th2025-huy-20skills-train.pkl`, `...-test.pkl`) and the bimanual dish washing set used by the wheeled bimanual space (`data/bimanual_dish_washing_test.pkl`), along with the held-out evaluation pickles.

> 📘 **Info**
>
> Two more archives live at the same URL, but you don't need them yet.
> `rl_policies.zip` contains everything the quadruped RL paths need with zero training: the 128 pretrained per-design-choice expert policies, `choice_to_ckpt_path.json`, the per-choice MuJoCo model options stores the RL runner requires at rollout time, and the pre-tokenized RoboToken bank the RL co-design evaluation decodes against — see the [data generation docs](data_generation.md).
> `blender_templates.zip` unpacks to `blender_templates/*.blend`, the pre-lit scenes used in the [visualization docs](visualization.md).
> Grab them with the same `wget -qO- <url> | bsdtar -xvf- -C ./` pattern when you get there.

## Evaluation

Both quick-start evaluations below run on the wheeled bimanual design space and the same dish washing test pickle — the control eval uses the cross-embodiment controller checkpoint (`z4454nxj`), and the co-design eval uses the hardware generation checkpoint (`mgoc83ra`), whose own dynamics predictions also serve as its critic.

> ❗**Caution**
>
> You need an NVIDIA GPU here.
> The control evaluation schedules policy servers through Ray with a fractional GPU reservation each, and raises an error if Ray can't see a GPU.
> The co-design evaluation will silently fall back to CPU diffusion sampling, which is unusably slow.

### Cross-embodiment Control

Evaluate the pretrained model as a cross-embodiment controller on 5 held-out wheeled bimanual designs, each tracking all 26 dish washing test trajectories.

```sh
python scripts/evaluate_ctrl.py evals/ctrl@eval_fn=wheeled_bimanual ckpt_path=checkpoints/wheeled_bimanual/z4454nxj/045.pt
```

Ray spins up `eval_fn.num_processes=50` CPU simulation workers and one policy server per GPU, batching observations from all environments through the model.
You'll see a `running policy` progress bar, then the summary metrics printed at the end.

- `metric/pos_err/mean` is the position tracking error in meters, summed over both end effectors at each step, averaged over episodes. This is the headline control number.
- `metric/orn_err/mean` is the orientation tracking error in radians.
- `done/bad_termination/any` is the percentage of episodes that were cut short because tracking error exceeded 0.5 meters. Survival rate is 100% minus this.

Some knobs I find myself overriding often:

- `eval_fn.num_processes=16` if you have fewer CPU cores — Ray reserves this many workers.
- `eval_fn.num_policies_per_gpu=2` to squeeze more policy servers onto a large GPU.
- `'eval_fn.hardware_traj_pairs_generator.list1=[10000000]' 'eval_fn.hardware_traj_pairs_generator.list2=[0,1,2]'` for a quick 3-episode sanity run (design seeds × trajectory indices).
- `evals/ctrl@eval_fn=quadruped ckpt_path=checkpoints/umi_on_legs_plus_plus/b9n2rc6s/024.pt eval_fn.runner.env.pickle_path=data/july25th2025-huy-20skills-test.pkl` evaluates the released multi-task quadruped model on held-out arm-mount variations of the UMI-on-Legs embodiment — 5 hardware seeds × the 20 test trajectories (`quadruped` is the config default). The `viperx` eval config works the same way for controllers you [train yourself](training.md#control-only).

All metrics also land in your W&B run, and the raw per-step rollouts are concatenated into a summary zarr in the run directory.

Inspect any rollout summary again without re-running the evaluation.

```sh
python scripts/summarize_rollout.py wandb/latest-run/files/ctrl_eval_summary.zarr
```

### Motion-to-robot Co-design

Generate a robot for each of the 26 dish washing test trajectories and evaluate the designs in simulation.

```sh
python scripts/evaluate_hardware_opt.py --config-name=evaluate_hardware_opt_bimanual ckpt_path=checkpoints/wheeled_bimanual/mgoc83ra/035.pt
```

For every trajectory × 9 random seeds (the paper's protocol, the config default), this samples 64 candidate robots from the hardware diffusion model, scores each candidate with the reward functions evaluated on the model's *own* predicted dynamics — the generator is its own critic — and keeps the best one.
That's the Zeroth Order optimizer, the config default.
Dynamics Self-Guidance is the other optimizer: it additionally backpropagates those reward gradients into every denoising step, and you select it with `hardware_optimizer@eval_fn.hardware_optimizer_fn=guided_diffusion` (see the switches below).
Each winning design is then detokenized from RoboTokens into an MJCF, and a Mink differential IK (DiffIK) controller tracks the target trajectory with it so the predicted reward can be checked against reality.
Expect a `value: ... for traj N, took ...s` log line per optimization, followed by a simulation rollout.

The switches worth knowing:

- `eval_fn.runner.infer_controller_weights=false` reproduces the paper's exact rollout setting for the bimanual tables. The shipped default (`true`) tracks measurably better; note also that the paper reported per-end-effector *mean* tracking errors while this codebase logs per-end-effector *sums*, so halve bimanual `pos_err`/`orn_err` when comparing against the paper.
- `hardware_optimizer@eval_fn.hardware_optimizer_fn=guided_diffusion` swaps in the gradient-guided optimizer, which injects reward gradients into every denoising step (over the link, joint, and actuator tokens) instead of only ranking finished samples. It needs settings the defaults don't give you: **a guidance scale matched to your design space and protocol** (it does not transfer, and too small a value leaves guidance inert, silently degrading to plain zeroth-order ranking), **`eta` near 1.0** (1.0 everywhere except the ViperX multi-trajectory row, which used 0.9; the `eta: 0.0` default is what the *unguided* results used), **`clip_samples_in_guidance=false` on every ViperX cell**, and — on ViperX `tracking_size` and `tracking_weight` only — **a weaker guidance-only reward coefficient** (`minimize_size.weight=0.005` / `minimize_weight.weight=0.005`, against reported values of 0.1 and 10), without which those runs generate designs that no longer simulate. The paper used scale `50.0` for ViperX single-trajectory (the config default), `500.0` for ViperX multi-trajectory, `100.0` for the quadruped, and `0.2` for the wheeled bimanual space — so for the bimanual command above, add `eval_fn.hardware_optimizer_fn.guidance.scale=0.2 eta=1.0` (bimanual needs no coefficient override). The full per-panel table, with the exact override syntax, is in the [figure reproduction docs](visualization.md#reproducing-the-co-design-results-figure).
- `reward_fns@eval_fn.hardware_optimizer_fn.reward_fns=tracking_size_bimanual` changes the design objective — see `config/reward_fns/` for the tracking + torque / velocity / weight / size variants. Because scoring runs on the model's predicted dynamics, you can steer generation with rewards the model was never trained on. If you are also running `guided_diffusion`, check the guidance table before switching to `tracking_size`/`tracking_weight`: those need a second, weaker coefficient on the guidance copy of the reward.
- `num_seeds_per_datapoint=1 'eval_fn.traj_indices=[0,1]'` for a quick sanity run.
- `--config-name=evaluate_hardware_opt_bimanual_multitraj` reproduces the paper's bimanual co-design protocol: instead of one robot per trajectory, composed diffusion generates a *single* design for all 26 dish washing trajectories (noise predictions from every trajectory are averaged at each denoising step), then rolls that one design out on each of them. This is what the paper's bimanual figure row reports.
- `--config-name=evaluate_hardware_opt ckpt_path=checkpoints/viperx/hw3n9bux/032.pt pickle_path=data/july25th2025-huy-20skills-test.pkl` runs the same co-design evaluation on the ViperX design space (the base config leaves `pickle_path` unset; the ViperX and quadruped spaces evaluate on the 20-skill UMI test pickle).

Everything ends up in the W&B run directory.

- `hardware_cache/<sha256>.pt` — the output cache, one file per (trajectory, seed): the best design's decoded RoboTokens, the model's `predicted_value`, and `optimize_time`. Keys are hashed from the trajectory and seed, so re-running the same config hits the cache instead of re-running diffusion.
- `hardware_opt_summary.zarr` — the concatenated evaluation rollouts of every generated design, in the same format as the control eval (so `scripts/summarize_rollout.py` works on it too).

The summary metrics tell you how good the designs are — and how well the model knows itself:

- `validity` is the fraction of generated designs that detokenized into a simulable robot.
- `hardware_meta/predicted_value` is the reward the model predicted for its own design; `hardware_meta/actual_value` is the same reward recomputed from the simulated rollout. Don't compare them raw: both are *sums over timesteps*, but the prediction covers only the model's rollout horizon (8 steps for the released checkpoints) while the actual value covers the whole episode (hundreds of steps). Normalize each by its horizon before comparing — per-step, the released critics track their rollouts within a few percent. This horizon mismatch is also why `scripts/scatter_plot_predicted_vs_actual.py` has a `--predicted_scale` flag.
- `metric/pos_err/mean` and friends are the same tracking metrics as the control evaluation, now achieved by robots that didn't exist a few seconds earlier.

From here, the [data generation docs](data_generation.md) show how the training data was made (RoboTokens, Mink and RL controllers, CMA-ES baselines), the [training docs](training.md) show how to train your own Transformer Transformer, the [extending docs](extending.md) cover adding new design spaces and RoboToken fields, and the [visualization docs](visualization.md) cover the Blender pipeline behind all our renders.
