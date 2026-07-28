# Model Training

This page covers training Transformer Transformer on the Zarr datasets produced by [data generation](data_generation.md).
One model class, one training script, two tasks: cross-embodiment control (`ctrl`) and motion-to-robot hardware generation (`hardware_gen`).
If you just want to run pretrained weights, download `checkpoints.zip` from [getting started](starter.md#checkpoints--data) instead.

## Hydra Configs

All training runs go through a single entry point

```sh
python scripts/train.py
```

which composes `config/train.yaml` with Hydra.
You pick a design space and a task by listing *addon* configs in the `defaults` list of `config/train.yaml`

```yaml
defaults:
  - model: t2
  - addon_hardware_bimanual_diffusion # 👈 swap out addons here
  - override hydra/job_logging: custom
  - _self_
```

where the available addons are

- `addon_ctrl_viperx`: `ctrl` on the ViperX design space.
- `addon_ctrl_quadruped`: `ctrl` on the quadruped manipulator design space.
- `addon_ctrl_bimanual`: `ctrl` on the wheeled bimanual design space — the task behind the released bimanual control checkpoints, fed by the `9012be`/`a019e7` training sets. Those checkpoints composed this addon *together with* `addon_hardware_bimanual_diffusion` (multi-task), and their controllers are memoryless single-step predictors (distilled from Mink) — add `ctrl_seq_len.rollout_steps=1` on the command line to match them.
- `addon_hardware_viperx_diffusion`: `hardware_gen` on the ViperX design space.
- `addon_hardware_quadruped_diffusion`: `hardware_gen` on the quadruped manipulator design space.
- `addon_hardware_bimanual_diffusion`: `hardware_gen` on the wheeled bimanual design space (the repo ships with this one enabled).

Each addon carries its design space's RoboToken counts in a `seq_len` block.
For instance, `config/addon_ctrl_viperx.yaml` ends with

```yaml
# ViperX design-space token counts
seq_len:
  dyna_joint: 14
  free_link: 1
  fixed_joint: 22
  link: 18
  actuator: 7
  track_link: 1
```

while the quadruped manipulator addons declare 62 dynamic joints, 100 fixed joints, 78 links, and 19 actuators, and the wheeled bimanual addon declares 40 dynamic joints, 74 fixed joints, 55 links, 20 actuators, and 2 track links.
`config/train.yaml` only derives the remaining counts from these (observation token counts, plus `rollout_steps: 8` future timesteps per training sample), so the addon is the single place where a design space's sequence layout is defined.

Each addon also declares a mandatory dataset path with Hydra's `???` marker.
Control addons require `ctrl_dataset_path`, hardware generation addons require `hardware_dataset_path`, and Hydra will refuse to launch with a `Missing mandatory value` error until you provide them on the command line.
Both point at Zarr datasets — either your own from [data generation](data_generation.md), or the paper's actual training sets, which are hosted as plain tars of the Zarr stores.

The training sets are hosted on the [Hugging Face dataset hub](https://huggingface.co/datasets/hqhuy/transformer-transformer) (better bandwidth and resumable downloads for files this size than our lab server, which hosts the smaller archives).

Download a training set and untar it into `data/`

```sh
wget -qO- https://huggingface.co/datasets/hqhuy/transformer-transformer/resolve/main/6c5628-bimanual-dishwashing-train-10-1000-perchoice.zarr.tar | tar -xf- -C data/
```

If you prefer parallel, resumable downloads, `pip install huggingface_hub` and `hf download hqhuy/transformer-transformer --repo-type dataset --local-dir training_data/` fetches everything at once.

| Training zarr | Size | Trained the released checkpoint |
| --- | --- | --- |
| `5059e4-varviper-pop5-maxfun15-q0.0-intermediate-transformaug` | 561 GB download (gzipped parts), 646 GB unpacked | `viperx/hw3n9bux/032.pt` (`hardware_gen`) |
| `36f120-plusplus-q0.2-10-1000` | 393 GB (parts) | `umi_on_legs_plus_plus/xrh4wk4l/040.pt` (`hardware_gen`) and `d7lra7j0` (multi-task: `hardware_gen` + `ctrl`) |
| `6c5628-bimanual-dishwashing-train-10-1000-perchoice` | 46 GB | `wheeled_bimanual/mgoc83ra/035.pt` and `ac028tq7` (`hardware_gen`) |
| `9012be-bimanual-dishwashing-train-10-1000-q0.05` | 37 GB | `wheeled_bimanual/z4454nxj/045.pt` (multi-task: `hardware_gen` + `ctrl`) |
| `a019e7-bimanual-dishwashing-train-10-1000-q0.05` | 48 GB | `wheeled_bimanual/52cbmwin` + `u5iyxc4d` (capacity ablation) |

The two large sets ship in 45 GB parts — download all parts of a set, then pipe them straight into `tar`: `cat 36f120-*.zarr.tar.part-* | tar -xf- -C data/` for the quadruped set, and `cat 5059e4-*.zarr.tar.gz.part-* | tar -xzf- -C data/` for the ViperX set (its chunks compress well, so it ships gzipped — note the `z`).
To verify downloads, fetch the checksum manifest from the dataset repo: `curl -LO https://huggingface.co/datasets/hqhuy/transformer-transformer/resolve/main/SHA256SUMS`.
This is a different manifest from the same-named `SHA256SUMS` on `real.stanford.edu` that covers the four zips — fetch them into different directories so one doesn't overwrite the other.
Its paths carry a `training_data/` prefix, so place the downloaded files in a `training_data/` directory and run `sha256sum -c --ignore-missing SHA256SUMS` from the directory above it (from anywhere else, `--ignore-missing` silently checks nothing); the split sets additionally ship a `.streamsha256` holding the hash of the concatenated part stream.
Expect multi-hour transfers for the large sets — `curl -L -C -` resumes an interrupted download, and `hf download` (above) parallelizes.
Each tar bundles the zarr together with its `.zarr.idx` index and, where one was precomputed, its `.norm.pt` normalization cache — so training skips the first-epoch index build; anything missing is computed and cached automatically on the first run.

> 📘 **Info**
>
> Every released checkpoint trains on one of the five sets above, except the two multi-task
> quadruped models `b9n2rc6s` and `2xdws25b`. Their configs name
> `9ebb55-plusplus-128arch-q0.0-10-10k` (hardware gen) and
> `f3d555-plusplus-128arch-q0.2-50-1k` (control), and neither survived our archives.
> Those checkpoints still ship and evaluate fine — but to retrain something like them, regenerate the data with the [data generation docs](data_generation.md).

The backbone is a DiT, configured under `config/model/backbone/`.
The default is `small.yaml` (hidden 256, 8 blocks, 4 heads, ~11.6M params) — the capacity behind every released main-result checkpoint and the ablation's small model, verified against the shipped checkpoints' own configs.
`medium.yaml` (hidden 512, 12 blocks, 8 heads, ~63.6M) is the paper's "Large" ablation capacity, and `large.yaml` (hidden 1024) is the configuration of the big multi-task control models (one of which additionally overrides `model.backbone.depth=18`).
The run-by-run capacity mapping is tabulated in [Getting Started](starter.md).
`scripts/train.py` logs a `number of parameters` line right after building whatever you composed, so you never have to guess.
Swap capacities from the command line

```sh
python scripts/train.py model/backbone=medium hardware_dataset_path=$hardware_dataset
```

The rest of `config/train.yaml` is the usual diffusion training recipe: AdamW at learning rate `1e-4` with no weight decay, a cosine schedule with 500 warmup steps, gradient clipping at `1.0`, and an EMA of the weights (`diffusers`' `EMAModel` with power `0.75`) which is what gets used at evaluation and inference time.
The `accelerator` block configures 🤗 Accelerate — `gradient_accumulation_steps`, `mixed_precision`, and wandb logging.
For multi-GPU training, launch the same script with Accelerate

```sh
accelerate launch scripts/train.py hardware_dataset_path=$hardware_dataset
```

and the script will log the `effective batch size` (number of processes × `train.batch_size` × gradient accumulation steps) at start up.

The `train` block has the knobs I reach for most often

- `train.batch_size`: defaults to `1024`, which assumes a beefy GPU. If you're running out of memory, I recommend lowering this and compensating with `accelerator.gradient_accumulation_steps` to keep the effective batch size fixed.
- `train.num_epochs` and `train.num_batches_per_epoch`: total optimization steps is their product (`20` × `1024` by default), which is what the cosine schedule is stretched over.
- `train.load_ckpt_path`: path to a checkpoint to resume from. This restores the model, EMA, optimizer, scheduler, and epoch counter, so interrupted runs pick up where they left off.

Everything a run produces lands in the wandb run directory (`wandb/run-*/files/`): the composed config as `cfg.pkl`, a checkpoint every `train.ckpt_every_n_epochs` epochs as `000.pt`, `001.pt`, ..., and `final.pt` at the end.
Each checkpoint bundles the model weights, EMA weights, and the dataset normalization statistics, and the downstream evaluation and inference scripts read `cfg.pkl` from the checkpoint's directory to rebuild the model — so keep checkpoints where they were saved, next to their `cfg.pkl`.
On the first run against a new dataset, the script computes normalization statistics with one pass over the data and caches them next to the Zarr as `<dataset>.norm.pt` — expect a one-time delay before the first epoch starts.

> 📘 **Info**
>
> `scripts/train.py` logs to the `transformer-transformer` wandb project unconditionally, so make sure you've run `wandb login` once during [setup](starter.md#setup).
> If you'd rather not sync anything, prefix your command with `WANDB_MODE=offline` — checkpoints still land in the local `wandb/` directory.

> 🪲 **Troubleshooting flash attention**
>
> `config/train.yaml` defaults to `use_flash_attn: false` and bf16 mixed precision — the settings every released checkpoint trained with. (The inference-side configs default to `true`, paired with bf16.)
> One deliberate exception: the `hardware_gen` task config opts back in with `use_flash_attn: true`, because that task uses no key-padding masks (flash attention silently drops them) and all released hardware-generation checkpoints trained with it.
> Flash attention requires half precision: keep `accelerator.mixed_precision=bf16` (with fp32 it fails at the first step with `RuntimeError: No available kernel` on any GPU) — this applies to the default training config too, since it includes the `hardware_gen` task.

## Control Only

The `ctrl` task learns embodiment-conditioned control by behavior cloning oracle rollouts: given a robot's embodiment RoboTokens, its current state, and the target motion, the model predicts the action sequence the oracle took on that robot.
Since embodiments are randomized in the dataset, one set of weights learns to control the entire design space — this is the controller (and, with [Dynamics Self-Guidance (DSG)](https://transformer-transformer.github.io), the critic) used at co-design time.
Under the hood, `ctrl` is also a diffusion task, just a fast one — 5 DDIM steps (`config/model/tasks/noise_scheduler/ddim.yaml`) instead of the 100 used for hardware generation.

To train a ViperX controller, set the addon in `config/train.yaml`

```yaml
defaults:
  - model: t2
  - addon_ctrl_viperx
  - override hydra/job_logging: custom
  - _self_
```

then launch

```sh
python scripts/train.py ctrl_dataset_path=$ctrl_dataset
```

where `$ctrl_dataset` is the `data_path` of a ViperX Mink (differential IK) datagen run from [data generation](data_generation.md).
The ViperX addon uses the `memoryless_nofreelink_ctrl` task config: it only supervises the action tokens and doesn't condition on past actions, which is the right shape for distilling a Mink DiffIK oracle.

For the quadruped manipulator, swap the addon

```yaml
defaults:
  - model: t2
  - addon_ctrl_quadruped
  - override hydra/job_logging: custom
  - _self_
```

and launch the same command with `$ctrl_dataset` pointing at RL expert rollouts from `scripts/datagen_rl.py` (pretrained per-design-choice experts ship in `rl_policies.zip`, see [getting started](starter.md#checkpoints--data)).
The quadruped uses the full `ctrl` task config, which additionally diffuses future states — including the floating base's `free_link_obs` — alongside actions, and adds pose augmentations for the mobile base.

After launching, you should see the `number of parameters` log line, the one-time normalization pass, then a tqdm bar per epoch.
In wandb, watch the step-level `loss` and the per-token-type breakdown (e.g. `ctrl/ctrl` for the action tokens); epoch averages appear under the `train/` prefix.
Expect a steep drop over the first epoch followed by a long slow decay that tracks the cosine learning rate (`lr`).
In-training environment rollout evals are disabled in the release configs (`evals: {}`) — evaluate checkpoints offline instead, as described in [evaluation](starter.md#evaluation).

## Hardware Generation Only

The `hardware_gen` task is the motion-to-robot direction: conditioned on the target motion tokens (`target_pose`, the only adapter with zero loss weight), the model jointly diffuses everything else — the embodiment RoboTokens (`link`, `dyna_joint`, `fixed_joint`, `actuator`), the state tokens (`track_link_obs`, `dyna_joint_obs`, `actuator_obs`, plus `free_link_obs` on the quadruped), and the action tokens (`ctrl`).
In other words, it dreams up a robot *and* the trajectory of that robot performing the motion, together.
It uses a DDIM noise scheduler with 100 train timesteps (`config/model/tasks/hardware_gen.yaml`), and the same 100 steps at inference.
The `*_log_diffuse` adapter overrides in the addons put wide-dynamic-range attributes — link masses and inertias, joint damping and armature, actuator gains and force ranges — into log space, which I found critical for the diffusion model to learn properly.

Set one hardware addon in `config/train.yaml`

```yaml
defaults:
  - model: t2
  - addon_hardware_viperx_diffusion # or addon_hardware_quadruped_diffusion
  - override hydra/job_logging: custom
  - _self_
```

then launch

```sh
python scripts/train.py hardware_dataset_path=$hardware_dataset
```

where `$hardware_dataset` is a datagen Zarr for the matching design space.
For the wheeled bimanual design space, the repo already ships with `addon_hardware_bimanual_diffusion` enabled, so no edit is needed

```sh
python scripts/train.py hardware_dataset_path=$hardware_dataset
```

In wandb, the loss splits per token type (`hardware_gen/link`, `hardware_gen/dyna_joint`, `hardware_gen/actuator`, `hardware_gen/ctrl`, ...), which I find handy for spotting which part of the robot the model is struggling to denoise.
To visualize what the model is generating as it trains, render intermediate checkpoints with the [diffusion visualization pipeline](visualization.md#diffusion-processes).

## Control + Hardware Generation

To get the unified model from the paper — designer, critic, and controller in one set of weights — list both addons for the *same* design space

```yaml
defaults:
  - model: t2
  - addon_ctrl_quadruped
  - addon_hardware_quadruped_diffusion
  - override hydra/job_logging: custom
  - _self_
```

then provide both dataset paths

```sh
python scripts/train.py ctrl_dataset_path=$ctrl_dataset hardware_dataset_path=$hardware_dataset
```

This composes cleanly (I've verified `addon_ctrl_viperx` + `addon_hardware_viperx_diffusion` and `addon_ctrl_quadruped` + `addon_hardware_quadruped_diffusion`): the model gets both task heads over a shared backbone, and each training step draws one batch from each dataset (`ctrl_rand` and `clean`) before a single optimizer step.
The two datasets can be different Zarrs — for instance, clean oracle rollouts for `hardware_gen` and broader embodiment-randomized rollouts for `ctrl`.

Since a run has exactly one `seq_len` block, a run trains on exactly one design space.
To cover another design space, train another model.

> ❗**Caution**
>
> Mixing addons from different design spaces in one run is unsupported.
> Their `seq_len` token counts conflict, and Hydra won't error out — the last addon in the defaults list silently wins, leaving the other design space's dataset with the wrong token counts.
> Keep every addon in a run from the same design space.
