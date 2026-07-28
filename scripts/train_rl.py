import functools
import os
import pickle
import time

import jax
import jax.numpy as jnp
import mujoco
from absl import logging
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)
from t2.mjx_env.base_env import MjxEnv, State
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
from flax.training import orbax_utils
from omegaconf import OmegaConf
from orbax import checkpoint as ocp

import wandb


@hydra.main(config_path="../config", config_name="run_rl", version_base="1.2")
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

    # Set up checkpoint directory
    ckpt_path = wandb.run.dir + "/checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    env: MjxEnv = hydra.utils.instantiate(cfg.env)
    # Load evaluation environment
    eval_env: MjxEnv = hydra.utils.instantiate(cfg.eval_env)

    FLOOR_GEOM_ID = 0

    arm_body_name = "arx5_base_link"
    arm_body_id = mujoco.mj_name2id(
        env.mj_model, mujoco.mjtObj.mjOBJ_BODY, arm_body_name
    )
    arm_mount_pos_range = jnp.array(cfg.arm_mount_pos)
    arm_mount_euler_range = jnp.array(cfg.arm_mount_euler)

    def randomization_fn(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand(rng):
            # Floor friction: =U(0.4, 1.0).
            rng, key = jax.random.split(rng)
            geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
                jax.random.uniform(key, minval=0.4, maxval=1.0)
            )

            # Scale static friction: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
                key, shape=(18,), minval=0.8, maxval=1.2
            )
            dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

            # Scale armature: *U(1.0, 1.05).
            rng, key = jax.random.split(rng)
            armature = model.dof_armature[6:] * jax.random.uniform(
                key, shape=(18,), minval=1.0, maxval=1.05
            )
            dof_armature = model.dof_armature.at[6:].set(armature)

            # Jitter center of mass positiion: +U(-0.05, 0.05).
            rng, key = jax.random.split(rng)
            dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
            body_ipos = model.body_ipos.at[1:].set(model.body_ipos[1:] + dpos)

            # Scale all link masses: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, shape=(model.nbody,), minval=0.9, maxval=1.1
            )
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

            rng, key = jax.random.split(rng)
            delta_kp = jax.random.uniform(key, shape=(18,), minval=0.8, maxval=1.2)
            gain_param = model.actuator_gainprm.at[:, 0].set(
                model.actuator_gainprm[:, 0] * delta_kp
            )
            bias_param = model.actuator_biasprm.at[:, 1].set(
                model.actuator_biasprm[:, 1] * delta_kp
            )

            rng, key = jax.random.split(rng)
            delta_kv = jax.random.uniform(key, shape=(18,), minval=0.8, maxval=1.2)
            bias_param = bias_param.at[:, 2].set(bias_param[:, 2] * delta_kv)

            rng, key = jax.random.split(rng)
            arm_mount_pos = jax.random.uniform(
                key,
                shape=(3,),
                minval=arm_mount_pos_range[0],
                maxval=arm_mount_pos_range[1],
            )
            body_pos = model.body_pos.at[arm_body_id, :].set(arm_mount_pos)

            rng, key = jax.random.split(rng)
            arm_mount_euler = jax.random.uniform(
                key,
                shape=(3,),
                minval=arm_mount_euler_range[0],
                maxval=arm_mount_euler_range[1],
            )
            euler_rot = Rotation.from_euler("xyz", arm_mount_euler)
            quat_rot = euler_rot.as_quat(scalar_first=True)
            body_quat = model.body_quat.at[arm_body_id, :].set(quat_rot)
            return (
                geom_friction,
                body_ipos,
                body_mass,
                dof_frictionloss,
                dof_armature,
                gain_param,
                bias_param,
                body_pos,
                body_quat,
            )

        (
            friction,
            body_ipos,
            body_mass,
            dof_frictionloss,
            dof_armature,
            gain_param,
            bias_param,
            body_pos,
            body_quat,
        ) = rand(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace(
            {
                "geom_friction": 0,
                "body_ipos": 0,
                "body_mass": 0,
                "dof_frictionloss": 0,
                "dof_armature": 0,
                "actuator_gainprm": 0,
                "actuator_biasprm": 0,
                "body_pos": 0,
                "body_quat": 0,
            }
        )

        model = model.tree_replace(
            {
                "geom_friction": friction,
                "body_ipos": body_ipos,
                "body_mass": body_mass,
                "dof_frictionloss": dof_frictionloss,
                "dof_armature": dof_armature,
                "actuator_gainprm": gain_param,
                "actuator_biasprm": bias_param,
                "body_pos": body_pos,
                "body_quat": body_quat,
            }
        )  # type: ignore

        return model, in_axes

    def randomization_obs_fn(model: mjx.Model):
        extra_obs = [
            model.geom_friction[FLOOR_GEOM_ID, 0],
            model.body_ipos[1:],
            model.body_mass[1:],
            model.dof_frictionloss[6:],
            model.dof_armature[6:],
            model.actuator_gainprm[:, 0],
            model.actuator_biasprm[:, 1],
            model.actuator_biasprm[:, 2],
            model.body_pos[arm_body_id, :],
            model.body_quat[arm_body_id, :],
        ]
        return jnp.hstack(jax.tree.map(lambda x: x.flatten(), extra_obs))

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
