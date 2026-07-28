import copy
import typing

import numpy as np
from transforms3d import affines, euler, quaternions

from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.mj_utils import repeated_ik
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv


class IKEnvRunner(EnvRunner):
    def __init__(
        self,
        look_ahead_steps: int,
        ik_gamma: float,
        qpos_noise: float,
        pos_noise: float | np.ndarray,
        orn_noise: float | np.ndarray,
        ik_max_attempts: int,
        ik_max_steps: int,
        local_frame_perturb: bool,
        **kwargs,
    ):
        super().__init__(
            default_policy=self.ik_policy,
            default_reset_fn=self.ik_reset_fn,
            **kwargs,
        )
        self.look_ahead_steps = look_ahead_steps
        self.ik_gamma = ik_gamma
        self.qpos_noise = qpos_noise
        self.pos_noise = (
            pos_noise if type(pos_noise) == np.ndarray else np.array([pos_noise] * 3)
        )
        self.orn_noise = (
            orn_noise if type(orn_noise) == np.ndarray else np.array([orn_noise] * 3)
        )
        self.ik_max_attempts = ik_max_attempts
        self.ik_max_steps = ik_max_steps
        self.local_frame_perturb = local_frame_perturb

    def post_hardware_reset(self, hardware_seed: int):
        super().post_hardware_reset(hardware_seed=hardware_seed)
        self.ik_data = copy.deepcopy(self.env.d)

    @property
    def tracking_env(self) -> TrackEnv:
        return typing.cast(TrackEnv, self.env)

    def ik_reset_fn(
        self, env: BaseEnv, seed: int
    ) -> tuple[ObsDict, InfoDict, HardwareDict]:
        target_traj_idx = seed % self.tracking_env.n_target_trajs
        self.tracking_env.traj_idx = target_traj_idx
        env.reset_no_obs(seed=seed)
        ik_result = self.ik_policy({})
        env.d.qpos[:] = ik_result["ctrl/target_qpos"]
        obs = self.tracking_env.get_obs()
        info = self.tracking_env.get_info()
        hardware = self.tracking_env.hardware_dict
        return obs, info, hardware

    def ik_policy(self, obs: ObsDict) -> dict[str, np.ndarray]:
        self.ik_data.qpos[:] = self.tracking_env.d.qpos[:]

        target_step = min(
            self.tracking_env.episode_step + self.look_ahead_steps,
            len(self.tracking_env.pose_traj) - 1,
        )
        target_pose = (
            self.tracking_env.augmentation @ self.tracking_env.pose_traj[target_step]
        )  # (n_end_effectors, 4, 4)
        target_pose = target_pose[0]  # (4, 4)
        target_mat = target_pose[:3, :3]
        target_pos = target_pose[:3, 3]
        assert target_mat.shape == (3, 3)
        assert target_pos.shape == (3,)
        target_quat = quaternions.mat2quat(target_mat)
        clean_ctrl = repeated_ik(
            model=self.env.m,
            data=self.ik_data,
            site_name=self.tracking_env.mj_site_names[0],
            dof_indices=np.arange(self.env.num_actuators),
            target_quat=target_quat,
            target_pos=target_pos,
            inplace=True,
            rs=self.tracking_env.rs,
            gamma=self.ik_gamma,
            max_attempts=self.ik_max_attempts,
            max_steps=self.ik_max_steps,
        )
        clean_ctrl = copy.deepcopy(clean_ctrl)
        noisy_ctrl = copy.deepcopy(clean_ctrl)
        if (self.pos_noise > 0).any() or (self.orn_noise > 0).any():
            target_delta_pose = affines.compose(
                T=self.tracking_env.rs.randn(3) * self.pos_noise,
                R=euler.euler2mat(
                    *self.tracking_env.rs.randn(3) * self.orn_noise,
                ),
                Z=np.ones(3),
            )
            target_pose = affines.compose(
                T=target_pos,
                R=target_mat.reshape(3, 3),
                Z=np.ones(3),
            )
            if self.local_frame_perturb:
                # local frame perturbation
                target_pose = target_pose @ target_delta_pose
            else:
                # global frame perturbation
                target_pose = target_delta_pose @ target_pose
            target_quat = quaternions.mat2quat(target_pose[:3, :3])
            target_pos = target_pose[:3, 3]
            self.ik_data.qpos[:] = self.tracking_env.d.qpos[:]
            noisy_ctrl = repeated_ik(
                model=self.env.m,
                data=self.ik_data,
                site_name=self.tracking_env.mj_site_names[0],
                dof_indices=np.arange(self.env.num_actuators),
                target_quat=target_quat,
                target_pos=target_pos,
                inplace=True,
                rs=self.tracking_env.rs,
                gamma=self.ik_gamma,
                max_attempts=self.ik_max_attempts,
                max_steps=self.ik_max_steps,
            )
        if self.qpos_noise > 0:
            noisy_ctrl += (
                self.tracking_env.rs.randn(*noisy_ctrl.shape) * self.qpos_noise
            )
        return {
            "ctrl/target_qpos": clean_ctrl,
            "ctrl/observed_qpos": copy.deepcopy(noisy_ctrl),
        }
