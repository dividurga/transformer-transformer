import copy
import typing
from typing import Optional

import mink
import numpy as np

from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.mj_utils import TARGET_SITE_GROUP, TRACKING_SITE_GROUP
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv


class MinkEnvRunner(EnvRunner):
    def __init__(
        self,
        look_ahead_steps: int,
        qpos_noise: float,
        mink_solver: str,
        mink_max_iters: int,
        mink_reset_max_iters: int,
        mink_pos_threshold: float,
        mink_ori_threshold: float,
        mink_pos_cost: float,
        mink_ori_cost: float,
        mink_lm_damping: float,
        mink_posture_cost: float,
        randomize_default_posture: bool,
        **kwargs,
    ):
        super().__init__(
            default_policy=self.mink_policy,
            default_reset_fn=self.mink_reset_fn,
            **kwargs,
        )
        self.look_ahead_steps = look_ahead_steps
        self.qpos_noise = qpos_noise
        self.mink_solver = mink_solver
        self.mink_max_iters = mink_max_iters
        self.mink_reset_max_iters = mink_reset_max_iters
        self.mink_pos_threshold = mink_pos_threshold
        self.mink_ori_threshold = mink_ori_threshold
        self.mink_pos_cost = mink_pos_cost
        self.mink_ori_cost = mink_ori_cost
        self.mink_lm_damping = mink_lm_damping
        self.mink_posture_cost = mink_posture_cost
        self.randomize_default_posture = randomize_default_posture
        self.mink_configuration: mink.Configuration
        self.mink_limits: list[mink.Limit] = []

    @property
    def tracking_env(self) -> TrackEnv:
        return typing.cast(TrackEnv, self.env)

    def post_hardware_reset(self, hardware_seed: int):
        super().post_hardware_reset(hardware_seed=hardware_seed)
        assert self.env.p is not None
        self.mink_configuration = mink.Configuration(self.env.m)

        # parse out tracking sites and their target sites
        site_groups = self.env.m.site_group
        assert (site_groups == TARGET_SITE_GROUP).sum() <= 1, (
            "only one target site allowed"
        )
        assert (site_groups == TRACKING_SITE_GROUP).sum() == 1, (
            "only one tracking site allowed"
        )
        tracking_site_id = np.where(site_groups == TRACKING_SITE_GROUP)[0][0]
        tracking_site = self.env.p.model.site(tracking_site_id)
        self.mink_frame_task = mink.FrameTask(
            frame_name=tracking_site.name,
            frame_type="site",
            position_cost=self.mink_pos_cost,
            orientation_cost=self.mink_ori_cost,
            lm_damping=self.mink_lm_damping,
        )
        target_site_id = np.where(site_groups == TARGET_SITE_GROUP)[0][0]
        self.mj_target_site_name = self.env.p.model.site(target_site_id).name
        self.posture_task = mink.PostureTask(self.env.m, cost=self.mink_posture_cost)

        root_id = self.env.m.body_rootid[self.env.m.site_bodyid[tracking_site_id]]
        geom_ids = mink.get_subtree_geom_ids(model=self.env.m, body_id=root_id)
        is_contact = np.logical_and(
            self.env.m.geom_contype[geom_ids] != 0,
            self.env.m.geom_conaffinity[geom_ids] != 0,
        )
        geom_ids_is_contact = is_contact[geom_ids]
        contact_geom_ids = np.array(geom_ids)[geom_ids_is_contact].astype(int).tolist()
        self.mink_limits.clear()
        self.mink_limits.append(mink.ConfigurationLimit(model=self.env.m))
        self.mink_limits.append(
            mink.CollisionAvoidanceLimit(
                model=self.env.m,
                geom_pairs=[(contact_geom_ids, contact_geom_ids)],  # type: ignore
                minimum_distance_from_collisions=0.05,
                collision_detection_distance=0.1,
            )
        )
        assert self.tracking_env.n_end_effectors == 1, "only one end effector allowed"

    def mink_policy(
        self, obs: ObsDict, mink_max_iters: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        if mink_max_iters is None:
            mink_max_iters = self.mink_max_iters
        self.mink_configuration.update(self.env.d.qpos)
        target_step = min(
            self.tracking_env.episode_step + self.look_ahead_steps,
            len(self.tracking_env.pose_traj) - 1,
        )
        target_pose = (
            self.tracking_env.augmentation @ self.tracking_env.pose_traj[target_step]
        )  # (n_end_effectors, 4, 4)
        target_pose = target_pose[0]  # (4, 4)

        self.mink_frame_task.set_target(mink.SE3.from_matrix(target_pose))

        configuration = self.mink_configuration
        for _ in range(mink_max_iters):
            try:
                vel = mink.solve_ik(
                    configuration=configuration,
                    tasks=[self.posture_task] + [self.mink_frame_task],
                    dt=self.env.m.opt.timestep,
                    solver=self.mink_solver,
                    limits=self.mink_limits,
                    damping=1e-5,
                )
            except mink.NoSolutionFound:
                break
            configuration.integrate_inplace(vel, self.env.m.opt.timestep)

            err = self.mink_frame_task.compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.mink_pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.mink_ori_threshold
            if pos_achieved and ori_achieved:
                break

        ctrl = copy.deepcopy(configuration.q)
        noisy_ctrl = copy.deepcopy(ctrl)
        if self.qpos_noise > 0:
            noisy_ctrl += self.env.rs.randn(*noisy_ctrl.shape) * self.qpos_noise
        return {
            "ctrl/target_qpos": ctrl,
            "ctrl/observed_qpos": noisy_ctrl,
        }

    def mink_reset_fn(
        self, env: BaseEnv, seed: int
    ) -> tuple[ObsDict, InfoDict, HardwareDict]:
        target_traj_idx = seed % self.tracking_env.n_target_trajs
        self.tracking_env.traj_idx = target_traj_idx
        env.reset_no_obs(seed=seed)
        if self.randomize_default_posture:
            target_q = env.rs.uniform(env.m.jnt_range[:, 0], env.m.jnt_range[:, 1])
        else:
            target_q = env.d.qpos[:]
        self.posture_task.set_target(target_q)
        mink_result = self.mink_policy(obs={}, mink_max_iters=self.mink_reset_max_iters)
        env.d.qpos[:] = mink_result["ctrl/target_qpos"]
        env.d.ctrl[:] = mink_result["ctrl/target_qpos"]
        env.p.forward()
        obs = self.tracking_env.get_obs()
        info = self.tracking_env.get_info()
        hardware = self.tracking_env.hardware_dict
        return obs, info, hardware
