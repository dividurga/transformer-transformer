# Aim 0: Morphology-Intrinsic Robustness to Unmodeled Physics Mismatch

## Context

This is the first concrete experiment for an undergraduate thesis on morphology-controller co-design, targeting robustness to sim-to-real mismatch, eventually validated on real hardware (later phase, out of scope here).

**The research question:** if two robot morphologies perform equally well under nominal/trained physics, does one hold up better than the other under physics effects *not* covered by the training-time domain randomization (DR)? I.e., is there a morphology-intrinsic robustness property, independent of how well the controller was optimized?

**Why this isn't just re-running the repo's existing DR:** the repo (`scripts/train_rl.py:68-204`, `t2/mjx_env/wrapper.py:214-273`) already randomizes floor friction, joint friction/armature, link mass, motor gain/bias, CoM, and mount pose during PPO training — but only to make *one* controller for *one fixed morphology* robust within that sampled distribution. It never compares morphologies to each other, and it's never evaluated outside the family of parameters it was trained on. That's also the general critique of DR you raised: testing within the same perturbation family the model was trained against doesn't tell you anything about genuinely unmodeled mismatch.

**Prior art check (done — see conversation):** this general question ("does morphology affect robustness independent of the controller") is not new — Gupta et al. ("Embodied Intelligence via Learning and Evolution," Nat. Comms 2021), Kriegman & Bongard's work on morphological robustness, and a flapping-wing paper (non-monotonic sim2real gap vs. morphological complexity) all study related territory. **Aim 0 is therefore scoped as a small, time-boxed sanity check / motivating case study — not the thesis's core contribution.** The actual contribution is whether the finding can be acted on: making the paper's `hardware_gen` generative co-design objective robustness-aware (sketched as Aim 1 below, not implemented in this pass). Read the three prior-art papers above before writing the thesis intro, so the framing is precise about what's known vs. new.

**Major feasibility finding:** pretrained frozen controllers already exist for all 128 discrete quadruped morphology variants in this repo's `umi_on_legs_plus_plus` design space, downloadable via:
```
wget -qO- https://real.stanford.edu/transformer-transformer/rl_policies.zip | bsdtar -xvf- -C ./
```
(`docs/data_generation.md:152-155`, `docs/starter.md:126`). This ships `choice_to_ckpt_path.json`, per-choice MuJoCo model option stores, and a pre-tokenized RoboToken bank. **No new PPO training is required for Aim 0** — this is purely an evaluation/analysis project, which is what makes it tractable in a few weeks on modest compute.

## Design decisions (confirmed)

- **Mismatch source:** unmodeled effects on the *same* MuJoCo engine — contact stiffness/restitution, actuator torque-saturation nonlinearity, actuator latency, extra sensor noise, and a simple backlash proxy. None of these are in the existing DR list, so they're a genuine held-out generalization test, not a re-sample of the training distribution.
- **Morphology pool:** the quadruped (`umi_on_legs_plus_plus`) design space, using the 128 choices that ship with pretrained experts.
- **Controller protocol:** freeze each morphology's pretrained controller; vary only the physics between the nominal and perturbed rollout. This isolates morphology's contribution from controller adaptability.
- **Hardware:** out of scope for this plan; sketched as a later phase only.

## Verified technical foundations

- **Rollout architecture** (`t2/env/rl_runner.py:79-296`, read in full): `RLEnvRunner` steps real physics on a plain `mujoco.MjModel`/`MjData` (`self.env`), and separately maintains a JAX/MJX copy (`self.mjx_env`) used *only* to recompute observations and run the frozen PPO policy each step (`get_action`/`update_obs_and_get_action`, lines 172-214). Both models are rebuilt from scratch on every `post_hardware_reset` (lines 262-296: `self.env.m` first, then `self.mjx_env.mj_model = copy.deepcopy(...)` + `mjx.put_model(...)`). **Any physics perturbation must patch both copies, every reset.**
- **No shape changes allowed:** the frozen checkpoint's action/observation sizes are baked into the PPO network (`load_policy`, `rl_runner.py:48-76`, reads `action_size`/`obs_shape` straight from the checkpoint's params). Perturbations must not change `nq`/`nu`/`nv` — rules out adding new joints/bodies (e.g. for backlash).
- **No fall-over signal exists on the eval path.** `RLEnvRunner.__init__` explicitly disables it for eval (`rl_runner.py:99-101`: `terminate_on_bad_contact = False`, `termination_fellover_threshold = -1.0`, `termination_pos_err_threshold = 100`), and `TrackEnv.get_bad_termination` (`t2/env/track_env.py:262-264`) only ever checked position-tracking error, not orientation/fall state, even before eval disabled it. **The mismatch module needs its own fall/instability detector** — reuse the up-vector dot-product check from `BaseEnv.get_bad_termination` (`t2/env/base_env.py:346-361`) as a logged metric, not a termination condition (since perturbed episodes should keep running to see how badly they degrade, not cut off early).
- **Reward/metric path for frozen-checkpoint eval is `t2/env/track_env.py`, not the MJX training-time reward.** `TrackEnv.step` (`track_env.py:248-260`) computes `reward = exp(-(pos_err²/σ_pos + orn_err²/σ_orn))` from `metric/pos_err`, `metric/orn_err` (`get_info`, lines 151-193). Use `metric/pos_err` (lower = better) as the primary performance/robustness metric — it's also the metric `scripts/pick_rl_policy.py` already uses to gate "successfully trained" (`pos_err_q50 < 0.1`).
- **128-choice structure confirmed:** `t2/robogen/components.py:803-818` (`enumerate_choices`) enumerates all 138 discrete combinations in the full `umi_on_legs_plus_plus` space (3 mutually exclusive top-level mounts: legs-forward / flipped-legs / wheeled, `assets/mjcf/umi_on_legs_plus_plus/base.xml:6`); `scripts/optimize_cmaes_quadruped.py:64-72` filters to the 128 in the plain `quadruped_manipulator` branch (the only branch with pretrained experts — the other 10 have RoboToken/options data but no PPO checkpoint). **Restrict the morphology pool strictly to keys present in `choice_to_ckpt_path.json`.**
- **Continuous shape is separate from discrete `choices`:** thigh/calf length, leg yaw/splay, battery position are `uniforms`, resolved per-episode from a `hardware_seed` (`scripts/datagen_rl.py:27-34`), not part of the 128-way discrete choice. A "choice" is a body-plan family, not one fixed morphology — average over a few `hardware_seed`s per choice rather than trusting one draw.
- **Observation noise mechanism already exists and is reusable as-is:** `RLEnvRunner` sets `self.mjx_env.obs_noise = {"gravity":0, "qpos":0, ...}` at construction (`rl_runner.py:129-135`); mutating this dict after construction is enough to add sensor noise, no new code needed.
- **`"ctrl/target_qpos"` vs `"ctrl/observed_qpos"`** (`run_rl_policy`, `rl_runner.py:298-360`, and `qpos_noise` handling at lines 343-345) is the existing pattern for injecting an actuation-side perturbation between "what the policy wants" and "what physically happens" — latency and backlash should follow this exact pattern via a wrapped `policy_fn`, not by editing `rl_runner.py`.

## Implementation plan

### Step 0 — verify assumptions before writing analysis code
Download `rl_policies.zip`, confirm `len(choice_to_ckpt_path.json) == 128` and inspect its key format. Call `enumerate_choices(COMPONENTS)` from `t2/robogen/components.py` and empirically confirm the bit-index → physical-axis mapping (front/rear leg type, knee-flip, arm mount) by diffing single-bit flips against `assets/mjcf/umi_on_legs_plus_plus/quadruped_manipulator.xml`'s `<numeric ... choice ...>` declarations — don't assume the mapping, print it.

### Step 1 — baseline nominal-performance sweep (no new code beyond a loop)
Run `scripts/datagen_rl.py` for each of the 128 choice keys with a small `num_hardware`/`num_episodes_per_hardware` (enough for ranking, not production volume), then `t2.eval.utils.summarize_rollout(...)["metric/pos_err/mean"]` to rank all 128 choices by nominal tracking performance.

### Step 2 — morphology selection (small new script)
From the 128 ranked choices: pick N=5-8 that (a) fall within a tight nominal-performance band (e.g. within the top quartile, or within ~0.03m of the best `pos_err/mean`, consistent with the repo's own `<0.1` "successfully trained" bar) and (b) within that band, maximize diversity across the decoded body-plan axes from Step 0 (aim to include a short-leg vs. long-leg pair, a linkage-vs-simple-leg pair, and different knee-flip combos).

### Step 3 — perturbation module: new file `t2/eval/mismatch.py`
A small, isolated module (same convention as `t2/eval/utils.py`, `t2/eval/diffusion_guidance.py`):
- `MismatchConfig` dataclass: `contact_stiffness_scale`, `contact_restitution_delta`, `forcerange_scale`, `latency_steps`, `obs_noise_scale`, `backlash_deadband_rad`.
- **`MISMATCH_SCENARIOS: dict[str, MismatchConfig]`** — a small hand-picked battery of named presets, not a single config and not a random sweep over configs. A single arbitrary `MismatchConfig` makes the whole Aim 0 result hostage to whether that one knob setting happened to flatter a given morphology; sampling a distribution over configs just recreates the DR family the experiment is supposed to be held out from (the exact critique motivating Aim 0 in the first place). The middle ground: 4-6 scenarios, each perturbing a *different subset* of axes at a clearly-large-but-plausible severity, so each is a distinct "unmodeled physics" hypothesis rather than a point sample from one distribution:
  - `stiff_terrain`: `contact_stiffness_scale`, `contact_restitution_delta` only
  - `degraded_actuators`: `forcerange_scale`, `latency_steps` only
  - `noisy_sensors`: `obs_noise_scale` only
  - `worn_joints`: `backlash_deadband_rad` only
  - `compound`: all axes at moderate (not maximal) severity
  - Pick severities per axis via the Step 3 sanity-check (large enough to visibly matter, not so large the robot fails outright regardless of morphology).
- `apply_model_perturbations(mj_model, cfg)`: in-place numpy edits to `geom_solref`/`geom_solimp` (contact stiffness/restitution) and `actuator_forcerange` (torque saturation) — ordinary MuJoCo model fields, no JAX shape constraints on the physics side.
- `install_mismatch(runner, cfg) -> policy_fn`: wraps `runner.post_hardware_reset` so perturbations are re-applied to **both** `runner.env.m` and `runner.mjx_env.mj_model`/`mjx_model` after every reset (per the verified consistency requirement above); sets `runner.mjx_env.obs_noise.update(...)` once; returns a `policy_fn` closure wrapping `runner.run_rl_policy` that adds actuator latency (a small `deque` ring buffer over `"ctrl/observed_qpos"`) and a backlash deadband (only update the physically-committed ctrl for a joint once the policy's target moves more than `backlash_deadband_rad` from the last committed value).
- **Backlash note:** implement the deadband proxy, don't skip it — a real compliant-joint model would change `nq`/`nu`, which is structurally incompatible with reusing frozen checkpoints. The deadband costs little given the latency wrapper is already needed.
- Add a fall/instability metric (ported from `BaseEnv.get_bad_termination`'s up-vector check) logged as `done/fellover`, since perturbed rollouts have no such signal today (see verified foundations above) — log-only, don't terminate on it.

### Step 4 — comparison script: new file `scripts/run_mismatch_benchmark.py`
For each selected choice: instantiate env + `RLEnvRunner` from its checkpoint (reuses `config.mj_model_data_path`'s stored choice/robogen metadata automatically, no manual bookkeeping), run once with no perturbation (nominal) and once per scenario in `MISMATCH_SCENARIOS` with `install_mismatch` applied, via the existing `run_episode`/`run_episodes` + `summarize_rollout` machinery. Output a flat table keyed by `(choice, scenario)`: `nominal_pos_err`, `perturbed_pos_err`, `degradation`, `nominal_fellover_rate`, `perturbed_fellover_rate`.

### Step 5 — analysis
Nominal performance across N=5-8 morphologies will never be exactly equal, so don't rely solely on hand-matched pairs:
1. Run the same nominal/perturbed comparison across a larger reference set (ideally all 128 choices, or 20-30 if time-boxed), for every scenario in `MISMATCH_SCENARIOS`.
2. Per scenario, fit `perturbed_pos_err ~ f(nominal_pos_err)` over the reference set (start with OLS).
3. For the N morphologies of interest, compute the **residual** from each scenario's fit — a large negative residual means "more robust than its nominal skill predicts," isolating morphology's contribution from "some designs are just nominally better."
4. **Cross-scenario consistency is the actual robustness claim, not any single scenario's residual.** Check whether a morphology's residual sign/rank holds across most or all scenarios (e.g. Spearman rank-correlation of residuals between scenario pairs, or just eyeballing the sign across 4-6 scenarios given small N). A morphology that looks robust under one scenario but not others is evidence of a lucky knob setting, not a morphology-intrinsic property — the whole point of the scenario battery over a single config.
5. Report the tight-band matched comparison (legible headline result), the per-scenario residual analysis, and the cross-scenario consistency check together. Frame Aim 0's output as a motivating case study, not a significance-tested claim.

## Aim 1 sketch (not implemented now — for thesis framing only)

`t2/eval/diffusion_guidance.py`'s zeroth-order path (`HardwareOptimizer._run_optimization_core`, ~lines 848-889) scores each sampled candidate morphology with `sum(reward_fn(decoded) for reward_fn in self.reward_fns.values())` and keeps the best — no differentiability required, since it's already an argmax over a finite batch. A new `RobustnessScore(RewardFn)` could plug directly into this path (same Hydra `reward_fns@...` override mechanism used elsewhere) with zero changes to the optimizer itself. The cheapest version: reuse `t2.eval.mismatch.apply_model_perturbations`-style perturbations against the model's own predicted-dynamics fitness estimate (the same substitution `hardware_gen` already uses instead of a real rollout) — i.e. `robustness = -(predicted_error_under_perturbed_dynamics − predicted_error_under_nominal_dynamics)`. This would make morphology *generation* prefer robust designs directly, turning Aim 0's empirical finding into an actionable design objective. Revisit this only after Aim 0's result is in hand and worth acting on.

## Critical files
- `t2/env/rl_runner.py` — frozen-policy rollout, perturbation injection point
- `t2/env/runner.py` — `EnvRunner.run_episode`/`run_episodes`, `policy_fn` override hook
- `t2/env/track_env.py` — reward/metric computation, `get_bad_termination`
- `t2/robogen/components.py` — `enumerate_choices`, choice/uniform structure
- `scripts/datagen_rl.py` — reference pattern for driving rollouts from a checkpoint
- `t2/eval/utils.py` — `summarize_rollout`
- `t2/eval/diffusion_guidance.py` — Aim 1 extension point only, not touched in Aim 0

## Verification
- Step 0's assumption check (choice count, bit-mapping) is itself a quick executable script — run it first and confirm against the plan before writing anything else.
- After Step 3, sanity-check the perturbation module in isolation: instantiate one `RLEnvRunner`, apply a large/obvious perturbation (e.g. `forcerange_scale=0.1`), and visually confirm (via existing repo visualization scripts, e.g. `scripts/rollout_env.py` or `visualize*.py`) that the robot visibly struggles more — a qualitative check before trusting any quantitative table.
- Confirm the nominal-condition numbers from Step 4 roughly match Step 1's baseline sweep for the same choices (they should, since nominal = no perturbation installed) as an internal consistency check on the new eval script.

## Progress log

- **Step 0, in progress:** downloaded `rl_policies.zip` on the `dd6849` Princeton scratch cluster (10GB uncompressed: `umi_on_legs_plus_plus_options` 6.1GB, `umi_on_legs_plus_plus_robotokens` 3.7GB, two `rl_policies_*` checkpoint snapshots ~0.3GB combined) and confirmed the archive itself is small — the scratch filesystem's 100GB per-user quota was already nearly full from unrelated prior usage, not this project. Extraction was interrupted partway by that quota and the partial extract was removed again (this project only needs ~10GB, well within a fresh quota on a new system). Re-run the download/unzip on whichever system this continues on:
  ```sh
  wget -qO- https://real.stanford.edu/transformer-transformer/rl_policies.zip | bsdtar -xvf- -C ./
  ```
  and confirm `len(choice_to_ckpt_path.json) == 128` before trusting it — that's the actual first checkpoint of Step 0, not yet done.
