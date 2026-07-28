import copy
import functools
import os
import pickle
import time

import jax
import jax.numpy as jnp
from absl import logging
from mujoco import mjx

from t2.env.mj_utils import (
    TARGET_SITE_GROUP,
    add_mocap_body_with_site,
    set_up_default_scene,
)
from t2.mjx_env.base_env import MjxEnv, State

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)
from t2.mjx_env.utils import render_policy
from t2.mjx_env.wrapper import wrap_for_brax_training

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import hydra
import jax
import zarr
from dm_control import mjcf
from flax.training import orbax_utils
from omegaconf import OmegaConf
from orbax import checkpoint as ocp

import wandb


@hydra.main(
    config_path="../config", config_name="run_rl_procedural", version_base="1.2"
)
def main(cfg):
    # Initialize Weights & Biases if required
    OmegaConf.resolve(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="transformer-transformer",
        tags=cfg.wandb.tags,
        config=config_dict,
    )
    pickle.dump(cfg, open(f"{wandb.run.dir}/config.pkl", "wb"))

    root = zarr.open(cfg.mj_model_data_path, mode="r")
    hardware_dict = {}
    num_hardware = -1
    for k in root.keys():
        hardware_dict[k] = jnp.array(root[k][:])
        num_hardware = root[k].shape[0]
    assert num_hardware != -1
    # hardware discrete choices
    choices = root.attrs["choices"]
    robogen_fn = hydra.utils.instantiate(root.attrs["robogen_fn"])
    num_uniforms = int(root.attrs["num_uniforms"])

    # Set up checkpoint directory
    ckpt_path = wandb.run.dir + "/checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    def mj_model_fn(
        num_uniforms: int = num_uniforms,
        choices: list[int] = choices,  # pyright: ignore[reportArgumentType]
    ):
        mjcf_model = robogen_fn(
            choices=copy.deepcopy(choices), uniforms=[0.5] * num_uniforms
        )
        mjcf_model = add_mocap_body_with_site(
            set_up_default_scene(mjcf_model, add_plane=True, plane_z_pos=0.0),
            name="target",
            site_group=TARGET_SITE_GROUP,
        )
        mj_model = mjcf.Physics.from_mjcf_model(mjcf_model).model.ptr
        return mj_model

    def randomization_fn(
        model: mjx.Model,
        rng: jax.Array,
        _hardware_dict: dict[str, jax.Array] = hardware_dict,
        _num_hardware: int = num_hardware,
    ):
        @jax.vmap
        def rand(rng):
            rng, key = jax.random.split(rng)
            hardware_idx = jax.random.randint(key, (), 0, _num_hardware)
            return {k: _hardware_dict[k][hardware_idx] for k in _hardware_dict.keys()}

        hardware_data = rand(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({k: 0 for k in _hardware_dict.keys()})
        model = model.tree_replace(hardware_data)  # type: ignore
        return model, in_axes

    env: MjxEnv = hydra.utils.instantiate(cfg.env, mj_model_fn=mj_model_fn)
    # Load evaluation environment
    eval_env: MjxEnv = hydra.utils.instantiate(cfg.eval_env, mj_model_fn=mj_model_fn)

    def randomization_obs_fn(model: mjx.Model):
        extra_obs = []
        for attr in cfg.extra_obs_attrs:
            extra_obs.append(getattr(model, attr).flatten())
        return jnp.hstack(extra_obs)

    jit_randomization_obs_fn = jax.jit(randomization_obs_fn)

    def reset(rng: jax.Array):
        state = eval_env.reset(rng)
        if cfg.observe_domain_randomized_attrs:
            extra_obs = jit_randomization_obs_fn(eval_env.mjx_model)
            state = state.replace(
                obs=jax.tree.map(
                    lambda x, y: jnp.hstack([x, y]),
                    state.obs,
                    extra_obs,
                )
            )
        return state

    jit_reset = jax.jit(reset)

    def step(state: State, action: jax.Array):
        state = eval_env.step(state, action)
        if cfg.observe_domain_randomized_attrs:
            extra_obs = jit_randomization_obs_fn(eval_env.mjx_model)
            state = state.replace(
                obs=jax.tree.map(
                    lambda x, y: jnp.hstack([x, y]),
                    state.obs,
                    extra_obs,
                )
            )
        return state

    jit_step = jax.jit(step)

    def render_rollout(
        make_policy,
        params,
        video_path: str,
        seed: int = 0,
    ):
        # Create inference function
        # params should be a tuple of (normalizer_params, policy_params, ....) (only first two are used)
        inference_fn = make_policy(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        return render_policy(
            reset_fn=jit_reset,
            step_fn=jit_step,
            policy=jit_inference_fn,
            env=eval_env,
            video_path=video_path,
            camera="body_cam",
            seed=seed,
        )

    # Define policy parameters function for saving checkpoints
    def policy_params_fn(current_step, make_policy, params):
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path + f"/{current_step}"
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        if cfg.render_on_checkpoint:
            render_path = path + f"/{current_step:06d}.mp4"
            start_time = time.time()
            render_rollout(
                make_policy,
                (params[0], params[1], params[2]),
                str(render_path),
            )
            end_time = time.time()
            print(f"Rendered rollout in {end_time - start_time:.2f} seconds")
            wandb.log(
                {"rollout_video": wandb.Video(str(render_path), format="mp4")},
                step=current_step,
            )

    # Progress function for logging
    def progress(num_steps, metrics):
        has_nan_metrics = any(jnp.isnan(v) for v in jax.tree_util.tree_leaves(metrics))
        wandb.log(metrics, step=num_steps)
        if has_nan_metrics:
            print(f"[{num_steps:06d}] NaN metrics detected")
            # terminate the training
            raise ValueError("NaN metrics detected")
        print("-" * 40 + f"{num_steps}" + "-" * 40)
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("-" * 80)

    # Train or load the model
    make_inference_fn, params, _ = hydra.utils.instantiate(
        cfg.train,
        policy_params_fn=policy_params_fn,
        wrap_env_fn=functools.partial(
            wrap_for_brax_training,
            full_reset=cfg.full_reset,
            randomization_obs_fn=(
                jit_randomization_obs_fn
                if cfg.observe_domain_randomized_attrs
                else None
            ),
        ),
        randomization_fn=randomization_fn,
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    render_path = wandb.run.dir + "/final-rollout.mp4"
    render_rollout(make_inference_fn, params, str(render_path))
    wandb.log({"rollout_video": wandb.Video(str(render_path))})


if __name__ == "__main__":
    main()
