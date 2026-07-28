import copy
import functools
import gc
import logging
import os
import pickle
from typing import Callable
import typing

import hydra
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import zarr
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.checkpoint import config_dict
from dm_control import mjcf
from etils import epath

from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.mj_utils import (
    TARGET_SITE_GROUP,
    TRACKING_SITE_GROUP,
    add_mocap_body_with_site,
    set_up_default_scene,
)
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv
from t2.mjx_env.base_env import State, make_data
from t2.mjx_env.track_env import TrackMjxEnv

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_logging_level", "CRITICAL")


def load_policy(
    path: str | epath.Path,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks],
    deterministic: bool = True,
) -> tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], int, int]:
    """Loads policy inference function from PPO checkpoint."""
    path = epath.Path(path)
    # silence all logging
    logging.getLogger().setLevel(logging.ERROR)
    params = ppo_checkpoint.load(path)
    logging.getLogger().setLevel(logging.INFO)
    policy_params = params[1]["params"]

    action_size = len(policy_params["std_param"]["value"])
    obs_shape = policy_params["MLP_0"]["hidden_0"]["kernel"].shape[0]

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        obs_shape, action_size, preprocess_observations_fn=normalize
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    return (
        make_policy(params, deterministic=deterministic),  # type: ignore
        obs_shape,
        action_size,
    )


class RLEnvRunner(EnvRunner):
    def __init__(
        self,
        ckpt_path: str,
        qpos_noise: float,
        **kwargs,
    ):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.qpos_noise = qpos_noise
        config_path = self.ckpt_path.split("/checkpoints/")[0] + "/config.pkl"
        config = pickle.load(open(config_path, "rb"))
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        jax.config.update("jax_platforms", "cpu")

        network_factory = config.train.network_factory
        config.env.randomize_target_z_pos = (0.0, 0.0)
        config.env.reset_z_angle = 0.0
        config.env.reset_xy_pos = 0.0
        config.env.reset_qpos = 0.0
        config.env.reset_qvel = 0.0
        config.env.terminate_on_bad_contact = False
        config.env.termination_fellover_threshold = -1.0
        config.env.termination_pos_err_threshold = 100
        config.env.termination_orn_err_threshold = 100
        config.env.velocity_kick = (0.0, 0.0)
        config.env.transport_interval = -1
        config.env.penalty_bad_contact_pairs = []

        root = zarr.open(config.mj_model_data_path, mode="r")
        choices = root.attrs["choices"]
        robogen_fn = hydra.utils.instantiate(root.attrs["robogen_fn"])
        num_uniforms = int(root.attrs["num_uniforms"])

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

        self.mjx_env: TrackMjxEnv = hydra.utils.instantiate(
            config.env,
            obs_noise={
                "gravity": 0.0,
                "qpos": 0.0,
                "qvel": 0.0,
                "lin_vel": 0.0,
                "ang_vel": 0.0,
            },
            mj_model_fn=mj_model_fn,
            penalty_joint_acc=0.0,  # new feature added after many rl checkpoints were trained
        )

        mjx_reset_fn = jax.jit(self.mjx_env.reset)
        mjx_forward = jax.jit(mjx.forward)

        ppo_config = config_dict.ConfigDict(config.train)
        policy, obs_shape, self.action_size = load_policy(
            path=self.ckpt_path,
            normalize_observations=ppo_config.normalize_observations,
            network_factory=hydra.utils.call(network_factory),
            deterministic=True,
        )
        policy = jax.jit(policy)
        self.rng = jax.random.PRNGKey(0)
        key, self.rng = jax.random.split(self.rng)
        # test run
        policy(jnp.zeros((obs_shape)), key)
        self.env_config = config.env
        self.state = mjx_reset_fn(self.rng)
        self.extra_obs = None

        def extra_obs_fn(
            model: mujoco.MjModel,
            extra_obs_attrs: list[str] = config.extra_obs_attrs,
        ):
            extra_obs = []
            for attr in extra_obs_attrs:
                extra_obs.append(jnp.array(getattr(model, attr)).flatten())
            return jnp.hstack(extra_obs)

        self.extra_obs_fn = (
            extra_obs_fn if config.observe_domain_randomized_attrs else None
        )

        @functools.partial(jax.jit, static_argnums=(5,))
        def update_obs_and_get_action(
            mjx_data: mjx.Data,
            state: State,
            key: jax.Array,
            target_obs: jax.Array,
            extra_obs: jax.Array | None = None,
            mjx_env: TrackMjxEnv = self.mjx_env,
        ):
            mjx_data = mjx_forward(mjx_env.mjx_model, mjx_data)
            state, _ = mjx_env.update_state_info(state, mjx_data)
            obs = mjx_env._get_obs(data=mjx_data, info=state.info)
            obs = obs.at[-len(target_obs) :].set(target_obs)
            if extra_obs is not None:
                obs = jnp.hstack([obs, extra_obs])
            action, _ = policy(obs, key)
            action = jnp.clip(action, mjx_env.clip_actions[0], mjx_env.clip_actions[1])
            ctrl = mjx_env.action_offset + action * mjx_env.action_scale
            return state, ctrl

        def get_action(
            data: mujoco.MjData,
            mjx_data: mjx.Data,
            state: State,
            key: jax.Array,
            target_obs: jax.Array,
            extra_obs: jax.Array | None = None,
        ) -> tuple[State, np.ndarray]:
            mjx_data = mjx_data.replace(
                qpos=jnp.array(data.qpos),
                qvel=jnp.array(data.qvel),
                ctrl=jnp.array(data.ctrl),
                actuator_force=jnp.array(data.actuator_force),
            )
            return update_obs_and_get_action(
                mjx_data=mjx_data,
                state=state,
                key=key,
                target_obs=target_obs,
                extra_obs=extra_obs,
            )

        self.get_action = get_action

        self.time_offset = -2  # TODO look into this, RL policy seems to be delayed
        obs_time_indices = [
            t + self.time_offset for t in self.env_config.obs_time_indices
        ]
        self.obs_time_indices = np.array(sorted(obs_time_indices))
        reset_fn = (
            kwargs["default_reset_fn"]
            if kwargs.get("default_reset_fn", None) is not None
            else lambda _env, _seed: _env.reset(seed=_seed)
        )

        def augmented_reset_fn(
            _env: BaseEnv, _seed: int
        ) -> tuple[ObsDict, InfoDict, HardwareDict]:
            retval = reset_fn(_env, _seed)
            data = _env.d
            mjx_data = make_data(
                self.mjx_env.mj_model,
                qpos=data.qpos,
                qvel=data.qvel,
                impl=self.mjx_env.impl,
                nconmax=self.mjx_env._nconmax,
                njmax=self.mjx_env._njmax,
                mocap_pos=jnp.zeros((len(self.mjx_env._site_names), 3)),
                mocap_quat=jnp.zeros((len(self.mjx_env._site_names), 4)),
            )
            self.state = self.state.replace(
                data=mjx_data,
                info={
                    **self.state.info,
                    "steps": 0,
                    "prev_action": jnp.zeros((self.action_size,)),
                },
            )
            gc.collect()
            return retval

        super().__init__(
            default_policy=self.run_rl_policy,
            **{**kwargs, "default_reset_fn": augmented_reset_fn},
        )

    @property
    def tracking_env(self) -> TrackEnv:
        return typing.cast(TrackEnv, self.env)

    def post_hardware_reset(self, hardware_seed: int):
        super().post_hardware_reset(hardware_seed=hardware_seed)
        target_site_id = -1
        for site_id in range(self.env.m.nsite):
            site_group = self.env.m.site_group[site_id]
            if site_group == TRACKING_SITE_GROUP:
                target_site_id = site_id
                break
        assert target_site_id != -1
        self.target_site_id = target_site_id

        mjcf_model = self.env.robot_generator(
            hardware_seed,
        )
        mjcf_model = add_mocap_body_with_site(
            set_up_default_scene(mjcf_model, add_plane=True, plane_z_pos=0.0),
            name="target",
            site_group=TARGET_SITE_GROUP,
        )
        physics = mjcf.Physics.from_mjcf_model(mjcf_model)

        self.extra_obs = (
            self.extra_obs_fn(physics.model.ptr)
            if self.extra_obs_fn is not None
            else None
        )

        # when the CPU environment resets,
        # we also need to reset the mjx state
        self.mjx_env.mj_model = copy.deepcopy(physics.model.ptr)
        self.mjx_env.preprocess_mj_model(self.mjx_env.mj_model)
        self.mjx_env.mjx_model = mjx.put_model(
            self.mjx_env.mj_model, impl=self.mjx_env.impl
        )
        gc.collect()

    def run_rl_policy(self, _: ObsDict) -> dict[str, np.ndarray]:
        obs = self.env.sample_target_poses(
            self.env.episode_step + self.obs_time_indices
        )
        key, self.rng = jax.random.split(self.rng)
        target_obs = []
        target_pos = obs["target_pose/pos"]
        target_rot = obs["target_pose/rotmat"]
        actual_pose = np.identity(4)
        actual_pose[:3, :3] = self.env.d.site_xmat[self.target_site_id].reshape(3, 3)
        actual_pose[:3, 3] = self.env.d.site_xpos[self.target_site_id]
        inv_actual_pose = np.linalg.inv(actual_pose)

        for target_pos, target_rot in zip(
            target_pos.reshape(len(self.env_config.obs_time_indices), 3),
            target_rot.reshape(len(self.env_config.obs_time_indices), 3, 3),
        ):
            target_pose = np.identity(4)
            target_pose[:3, :3] = target_rot
            target_pose[:3, 3] = target_pos
            delta_pose = inv_actual_pose @ target_pose
            delta_pos = delta_pose[:3, 3]
            delta_rot_mat = delta_pose[:3, :3]
            if self.mjx_env.pos_obs_enc == "log-direction":
                distance = jnp.linalg.norm(delta_pos, axis=-1) + 1e-8
                direction = delta_pos / distance[..., None]
                pos_obs = jnp.concatenate(
                    [jnp.log(distance)[..., None], direction], axis=-1
                ).reshape(-1)
            elif self.mjx_env.pos_obs_enc == "linear":
                pos_obs = delta_pos.reshape(-1)
            else:
                raise ValueError(f"Unknown pos_obs_enc: {self.mjx_env.pos_obs_enc}")
            orn_obs = delta_rot_mat.reshape(-1)
            target_obs.append(pos_obs)
            target_obs.append(orn_obs)

        self.state, ctrl = self.get_action(
            data=self.env.d,
            mjx_data=self.state.data,
            state=self.state,
            target_obs=jnp.array(np.concatenate(target_obs, axis=-1)),
            key=key,
            extra_obs=self.extra_obs,
        )
        noisy_ctrl = ctrl
        if self.qpos_noise > 0:
            noisy_ctrl += self.env.rs.randn(*noisy_ctrl.shape) * self.qpos_noise
        action = (
            jnp.array(noisy_ctrl) - self.mjx_env.action_offset
        ) / self.mjx_env.action_scale
        self.state = self.state.replace(
            info={
                **self.state.info,
                "steps": self.state.info["steps"] + 1,
                "prev_action": action,
            },
        )

        return {
            "ctrl/target_qpos": ctrl,
            "ctrl/observed_qpos": copy.deepcopy(noisy_ctrl),
        }
