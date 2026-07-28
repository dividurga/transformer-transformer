import pickle
from typing import Any

import mujoco
import numpy as np
from dm_control import mjcf
from transforms3d import affines, euler, quaternions

from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.mj_utils import (
    TARGET_SITE_GROUP,
    add_mocap_body_with_site,
)
from t2.env.transforms import mat_norm
from t2.robotok.token import Robot


class TrackEnv(BaseEnv):
    def __init__(
        self,
        obs_time_indices: list[int],
        include_obs_time_indices: str | None,
        pickle_path: str,
        pos_noise: float,
        orn_noise: float,
        scale_noise: float,
        center_traj: bool,
        # probabiliy that pose augmentation changes per timestep
        noise_sample_prob: float,
        pos_err_sigma: float,
        orn_err_sigma: float,
        termination_pos_err_threshold: float,
        n_end_effectors: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_end_effectors = n_end_effectors
        self.obs_time_indices = np.array(sorted(obs_time_indices))
        assert include_obs_time_indices in {"none", "relative", "absolute"}
        self.include_obs_time_indices = include_obs_time_indices
        self.pickle_path = pickle_path
        self.traj_idx = -1
        self.n_target_trajs = len(pickle.load(open(self.pickle_path, "rb")))
        self.pos_noise = pos_noise
        self.orn_noise = orn_noise
        self.scale_noise = scale_noise
        self.noise_sample_prob = noise_sample_prob
        self.pos_err_sigma = pos_err_sigma
        self.orn_err_sigma = orn_err_sigma
        self.termination_pos_err_threshold = termination_pos_err_threshold
        self.augmentation = np.identity(4)
        self.center_traj = center_traj
        self.mj_site_names: list[str]
        self.curr_pos_err: np.ndarray
        self.curr_orn_err: np.ndarray
        self.pose_traj: np.ndarray

    def post_process_mjcf(self, mjcf_model: mjcf.RootElement) -> mjcf.RootElement:
        for i in range(self.n_end_effectors):
            mjcf_model = add_mocap_body_with_site(
                mjcf_model, f"target_{i}", site_group=TARGET_SITE_GROUP
            )
        return mjcf_model

    def post_hardware_reset(
        self,
        mjcf_model: mjcf.RootElement,
        tokenized_robot: Robot,
        mjjntid_to_dyna_jntid: dict[int, int],
        mjgeomid_to_linkid: dict[int, int],
        mjsiteid_to_tracklinkid: dict[int, int],
    ):
        super().post_hardware_reset(
            mjcf_model=mjcf_model,
            tokenized_robot=tokenized_robot,
            mjjntid_to_dyna_jntid=mjjntid_to_dyna_jntid,
            mjgeomid_to_linkid=mjgeomid_to_linkid,
            mjsiteid_to_tracklinkid=mjsiteid_to_tracklinkid,
        )

        self.curr_pos_err = np.zeros((len(self.track_link_ids)))
        self.curr_orn_err = np.zeros((len(self.track_link_ids)))

        # Use track_mj_site_ids from base class which is sorted by track_link_id
        # This ensures consistent ordering after tokenize/detokenize
        if len(self.track_mj_site_ids) != self.n_end_effectors:
            raise ValueError(
                f"Expected {self.n_end_effectors} tracking site(s), "
                f"got {len(self.track_mj_site_ids)}"
            )

        self.mj_site_names = [
            mujoco.mj_id2name(
                self.m,
                mujoco.mjtObj.mjOBJ_SITE,
                i,
            )
            for i in self.track_mj_site_ids
        ]

    def reset(self, seed: int) -> tuple[ObsDict, InfoDict, HardwareDict]:
        target_traj_idx = seed % self.n_target_trajs
        return self.reset_sim_and_target_traj(seed, target_traj_idx)

    def reset_no_obs(self, seed: int):
        super().reset_no_obs(seed=seed)
        if self.traj_idx == -1:
            raise ValueError("`traj_idx` is not set")
        self.pose_traj = pickle.load(open(self.pickle_path, "rb"))[self.traj_idx]
        # pose_traj shape: (T, 4, 4) for single end effector
        #                  (T, n_end_effectors, 4, 4) for multiple end effectors
        if self.pose_traj.ndim == 3:
            # Single end effector case: (T, 4, 4) -> (T, 1, 4, 4)
            self.pose_traj = self.pose_traj[:, None, :, :]
        if self.pose_traj.shape[1] != self.n_end_effectors:
            raise ValueError(
                f"pose_traj has {self.pose_traj.shape[1]} end effectors, "
                f"expected {self.n_end_effectors}"
            )
        # Camera looks at mean position across all end effectors and timesteps
        self.camera_lookat = np.mean(self.pose_traj[:, :, :3, 3], axis=(0, 1))
        self.augmentation = affines.compose(
            T=self.rs.uniform(-self.pos_noise, self.pos_noise, 3),
            R=euler.euler2mat(
                *self.rs.uniform(-self.orn_noise, self.orn_noise, 3),
            ),
            Z=self.rs.uniform(-self.scale_noise, self.scale_noise, 3) + 1,
        )
        if self.center_traj:
            # Center based on mean of first timestep across all end effectors
            init_pos_xy = np.mean(
                (self.augmentation @ self.pose_traj[0])[:, :2, 3], axis=0
            )
            self.pose_traj[:, :, :2, 3] = self.pose_traj[:, :, :2, 3] - init_pos_xy
            # Compute camera lookat after centering
            augmented_traj = np.einsum(
                "ij,tkjl->tkil", self.augmentation, self.pose_traj
            )
            self.camera_lookat = np.mean(augmented_traj[:, :, :3, 3], axis=(0, 1))

    def reset_sim_and_target_traj(
        self, seed: int, target_traj_idx: int
    ) -> tuple[ObsDict, InfoDict, HardwareDict]:
        self.traj_idx = target_traj_idx
        self.reset_no_obs(seed=seed)
        self.p.forward()  # might be redundant here
        obs = self.get_obs()
        info = self.get_info()
        return obs, info, self.hardware_dict

    def get_info(self) -> InfoDict:
        info = super().get_info()
        d = self.d
        target_step = min(self.episode_step, len(self.pose_traj) - 1)
        if self.rs.uniform() < self.noise_sample_prob:
            self.augmentation = affines.compose(
                T=self.rs.uniform(-self.pos_noise, self.pos_noise, 3),
                R=euler.euler2mat(
                    *self.rs.uniform(-self.orn_noise, self.orn_noise, 3),
                ),
                Z=self.rs.uniform(-self.scale_noise, self.scale_noise, 3) + 1,
            )
        # target_poses shape: (n_end_effectors, 4, 4)
        target_poses = self.augmentation @ self.pose_traj[target_step]
        assert target_poses.shape == (self.n_end_effectors, 4, 4)
        target_pos = np.zeros((self.n_end_effectors, 3))
        target_quat = np.zeros((self.n_end_effectors, 4))
        for track_link_id in range(self.n_end_effectors):
            pos, rotmat = affines.decompose(target_poses[track_link_id])[:2]
            target_pos[track_link_id] = pos
            target_quat[track_link_id] = quaternions.mat2quat(rotmat)
            target_body_name = "target_" + str(track_link_id)
            target_site_body = self.p.model.body(target_body_name)
            target_mocap_id = target_site_body.mocapid[0]
            self.d.mocap_pos[target_mocap_id] = target_pos[track_link_id]
            self.d.mocap_quat[target_mocap_id] = target_quat[track_link_id]
        self.p.forward()
        curr_pos = d.site_xpos[self.track_mj_site_ids].reshape(self.n_end_effectors, 3)
        curr_rotmat = d.site_xmat[self.track_mj_site_ids].reshape(
            self.n_end_effectors, 3, 3
        )
        self.curr_pos_err = np.linalg.norm(self.d.mocap_pos[:] - curr_pos, axis=1)
        self.curr_orn_err = np.array(
            [
                mat_norm(quaternions.quat2mat(q).T @ m)
                for q, m in zip(self.d.mocap_quat[:], curr_rotmat)
            ]
        )
        return {
            **info,
            "metric/pos_err": self.curr_pos_err.sum(),
            "metric/orn_err": self.curr_orn_err.sum(),
        }

    def get_timeout(self) -> bool:
        timeout = super().get_timeout()
        end_of_traj = self.episode_step >= len(self.pose_traj)
        return timeout or end_of_traj

    def sample_target_poses(
        self, task_obs_indices: np.ndarray, include_obs_time_indices: str | None = None
    ) -> dict[str, np.ndarray]:
        if include_obs_time_indices is None:
            include_obs_time_indices = self.include_obs_time_indices
        # pose_traj shape: (T, n_end_effectors, 4, 4)
        # target_poses shape: (len(task_obs_indices), n_end_effectors, 4, 4)
        clipped_indices = np.clip(task_obs_indices, 0, len(self.pose_traj) - 1)
        target_poses = np.einsum(
            "ij,tkjl->tkil", self.augmentation, self.pose_traj[clipped_indices]
        )
        # Transpose to (n_end_effectors, len(task_obs_indices), 4, 4)
        target_poses = target_poses.transpose(1, 0, 2, 3)
        target_pos = target_poses[..., :3, 3]
        target_rotmats = target_poses[..., :3, :3]
        obs_dict = {
            "target_pose/pos": target_pos,
            "target_pose/rotmat": target_rotmats,
        }
        if include_obs_time_indices == "relative":
            obs_dict["target_pose/time/id"] = (
                np.repeat(task_obs_indices[:, None], self.n_end_effectors, axis=1)
                - self.episode_step
            )
        elif include_obs_time_indices == "absolute":
            obs_dict["target_pose/time/id"] = np.repeat(
                task_obs_indices[:, None], self.n_end_effectors, axis=1
            )
        elif include_obs_time_indices == "none":
            assert len(self.obs_time_indices) == 1 and self.obs_time_indices[0] == 0

        return {
            k: v.reshape(len(task_obs_indices) * self.n_end_effectors, -1)
            for k, v in obs_dict.items()
        }

    def get_task_obs(self) -> ObsDict:
        self.get_info()
        task_obs = self.sample_target_poses(self.obs_time_indices + self.episode_step)
        return task_obs

    def get_render_text_rows(self) -> list[str]:
        text_rows = super().get_render_text_rows()
        return text_rows + [
            f"pos_err: {self.curr_pos_err.mean() * 100:.1f}cm",
            f"orn_err: {self.curr_orn_err.mean():.3f}",
        ]

    def step(self, ctrl: np.ndarray) -> tuple[ObsDict, float, bool, dict[str, Any]]:
        """Override step to provide a reward based on tracking error."""
        obs, _, done, info = super().step(ctrl)

        # Calculate reward based on position and orientation errors
        pos_err = info["metric/pos_err"]
        orn_err = info["metric/orn_err"]

        # Simple reward function that penalizes errors
        reward = np.exp(
            -(pos_err**2 / self.pos_err_sigma + orn_err**2 / self.orn_err_sigma)
        )
        return obs, reward, done, info

    def get_bad_termination(self) -> bool:
        pos_err_done = np.any(self.curr_pos_err > self.termination_pos_err_threshold)
        return bool(pos_err_done)

    def get_pose_obs(self) -> ObsDict:
        d = self.d
        pose_obs = super().get_pose_obs()
        if len(self.track_link_ids) == 0:
            return pose_obs
        pose_obs["track_link_obs/pos"] = np.zeros(
            (len(self.track_link_ids), 3),
            dtype=self.dtype,
        )
        pose_obs["track_link_obs/rotmat"] = np.zeros(
            (len(self.track_link_ids), 9),
            dtype=self.dtype,
        )
        for track_link_id, mj_geom_id in enumerate(self.track_mj_geom_ids):
            pose_obs["track_link_obs/pos"][track_link_id] = d.geom_xpos[mj_geom_id]
            pose_obs["track_link_obs/rotmat"][track_link_id] = d.geom_xmat[mj_geom_id]

        return pose_obs


class FreeBaseTrackEnv(TrackEnv):
    """
    TrackEnv for robots with a free joint
    """

    def __init__(
        self,
        center_robot: bool,
        reset_xy_pos: float,
        reset_z_angle: float,
        reset_qvel: float,
        reset_qpos: float,
        push_prob: float,
        push_velocity: tuple[float, float],
        transport_prob: float,
        transport_pos: tuple[tuple[float, float, float], tuple[float, float, float]],
        transport_euler: tuple[tuple[float, float, float], tuple[float, float, float]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.center_robot = center_robot
        self.qpos_free_xy: list[int]
        self.reset_xy_pos = reset_xy_pos
        self.reset_z_angle = reset_z_angle
        self.reset_qvel = reset_qvel
        self.reset_qpos = reset_qpos
        self.push_prob = push_prob
        self.push_velocity = push_velocity
        self.transport_prob = transport_prob
        self.transport_pos = np.array(transport_pos)
        self.transport_euler = np.array(transport_euler)

    def post_hardware_reset(
        self,
        mjcf_model: mjcf.RootElement,
        tokenized_robot: Robot,
        mjjntid_to_dyna_jntid: dict[int, int],
        mjgeomid_to_linkid: dict[int, int],
        mjsiteid_to_tracklinkid: dict[int, int],
    ):
        super().post_hardware_reset(
            mjcf_model=mjcf_model,
            tokenized_robot=tokenized_robot,
            mjjntid_to_dyna_jntid=mjjntid_to_dyna_jntid,
            mjgeomid_to_linkid=mjgeomid_to_linkid,
            mjsiteid_to_tracklinkid=mjsiteid_to_tracklinkid,
        )
        self.qpos_free_xy = []
        self.qpos_free_quat = []
        for jnt_id in range(self.m.njnt):
            if self.m.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_start_idx = self.m.jnt_qposadr[jnt_id]
                self.qpos_free_xy.extend([qpos_start_idx, qpos_start_idx + 1])
                self.qpos_free_quat.extend(
                    list(range(qpos_start_idx + 3, qpos_start_idx + 7))
                )
                break
        assert len(self.qpos_free_xy) > 0, "No free joints found"
        assert len(self.qpos_free_xy) == 2, "Only 1 free joints supported"

    def reset_no_obs(self, seed: int):
        super().reset_no_obs(seed=seed)
        if self.center_robot:
            d = self.d
            self.d.qvel[:] = self.rs.uniform(
                -self.reset_qvel, self.reset_qvel, len(self.d.qvel)
            )
            self.d.qpos[:] += self.rs.uniform(
                -self.reset_qpos, self.reset_qpos, len(self.d.qpos)
            )
            self.p.forward()
            curr_pos = d.site_xpos[self.track_mj_site_ids].reshape(
                len(self.track_mj_site_ids), 3
            )
            planar_pos = np.mean(curr_pos[:, :2], axis=0)
            self.d.qpos[self.qpos_free_xy] -= planar_pos
            self.d.qpos[self.qpos_free_xy] += self.rs.uniform(
                -self.reset_xy_pos, self.reset_xy_pos, len(self.qpos_free_xy)
            )
            z_angle = self.rs.uniform(-self.reset_z_angle, self.reset_z_angle)
            delta_quat = euler.euler2quat(0, 0, z_angle)
            base_quat = self.d.qpos[self.qpos_free_quat]
            new_quat = quaternions.qmult(base_quat, delta_quat)
            self.d.qpos[self.qpos_free_quat] = new_quat
            self.p.forward()

    def step(self, ctrl: np.ndarray) -> tuple[ObsDict, float, bool, dict[str, Any]]:
        if self.rs.uniform() < self.push_prob:
            self.d.qvel[:] += self.rs.uniform(
                -self.push_velocity[0],
                self.push_velocity[1],
                self.d.qvel.shape[0],
            )
        if self.rs.uniform() < self.transport_prob:
            dxy = self.rs.uniform(
                -self.transport_pos[0],
                self.transport_pos[1],
                2,
            )
            base_quat = self.d.qpos[self.qpos_free_quat]
            delta_quat = euler.euler2quat(
                *self.rs.uniform(-self.transport_euler[0], self.transport_euler[1], 3),
            )
            quat = quaternions.qmult(base_quat, delta_quat)
            self.d.qpos[self.qpos_free_xy] += dxy
            self.d.qpos[self.qpos_free_quat] = quat
        return super().step(ctrl)
