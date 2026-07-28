import copy
import logging
import typing
from typing import Optional
import mujoco
import mink
import numpy as np

from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.mj_utils import TARGET_SITE_GROUP, TRACKING_SITE_GROUP
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv


class WheeledBimanualMinkEnvRunner(EnvRunner):
    """
    MinkEnvRunner specialised for the wheeled bimanual robot.

    Handles IK for two tracking sites simultaneously while applying
    wheeled-base-specific posture/damping cost weighting and a
    qpos-vs-ctrl mapping for the free wheeled base.
    """

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
        mink_damping_cost: float,
        mink_posture_cost: float,
        randomize_default_posture: bool,
        # HACK
        infer_controller_weights: bool,
        reset_base_posture_cost_multiplier: float,
        reset_body_posture_cost_multiplier: float,
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
        self.mink_damping_cost = mink_damping_cost
        self.mink_posture_cost = mink_posture_cost
        self.randomize_default_posture = randomize_default_posture
        self.mink_configuration: mink.Configuration
        self.mink_limits: dict[str, mink.Limit] = {}
        # List of frame tasks, one per end effector
        self.mink_tasks: dict[str, mink.Task] = {}
        self.qpos_to_ctrl_map: dict[int, int] = {}
        self.infer_controller_weights = infer_controller_weights
        self.reset_base_posture_cost_multiplier = reset_base_posture_cost_multiplier
        self.reset_body_posture_cost_multiplier = reset_body_posture_cost_multiplier
        if self.infer_controller_weights:
            logging.warning(
                "Inferring controller weights. This will only work for the wheeled bimanual design space."
            )

    @property
    def tracking_env(self) -> TrackEnv:
        return typing.cast(TrackEnv, self.env)

    @property
    def n_end_effectors(self) -> int:
        return self.tracking_env.n_end_effectors

    def post_hardware_reset(self, hardware_seed: int):
        super().post_hardware_reset(hardware_seed=hardware_seed)
        assert self.env.p is not None
        self.qpos_to_ctrl_map.clear()
        for mj_actid in range(self.env.m.nu):
            actuator = self.env.p.model.actuator(mj_actid)
            mj_jntid = actuator.trnid[0]
            mj_qposid = int(self.env.p.model.joint(mj_jntid).qposadr[0])
            self.qpos_to_ctrl_map[mj_qposid] = mj_actid
        assert set(self.qpos_to_ctrl_map.keys()) == set(
            self.qpos_to_ctrl_map.values()
        ), "qpos_to_ctrl_map is not bijective"

        self.mink_configuration = mink.Configuration(self.env.m)

        # Parse out tracking sites and their target sites
        site_groups = self.env.m.site_group
        tracking_site_ids = np.where(site_groups == TRACKING_SITE_GROUP)[0]
        target_site_ids = np.where(site_groups == TARGET_SITE_GROUP)[0]

        if len(tracking_site_ids) != self.n_end_effectors:
            raise ValueError(
                f"Expected {self.n_end_effectors} tracking site(s), "
                f"got {len(tracking_site_ids)}"
            )
        if len(target_site_ids) != self.n_end_effectors:
            raise ValueError(
                f"Expected {self.n_end_effectors} target site(s), "
                f"got {len(target_site_ids)}"
            )

        # Create frame tasks for each tracking site
        self.mink_tasks.clear()

        tracking_sites = map(self.env.p.model.site, tracking_site_ids)

        for site in sorted(tracking_sites, key=lambda x: x.name):
            frame_task = mink.FrameTask(
                frame_name=site.name,
                frame_type="site",
                position_cost=self.mink_pos_cost,
                orientation_cost=self.mink_ori_cost,
                lm_damping=self.mink_lm_damping,
            )
            self.mink_tasks[f"tracking_{site.name}"] = frame_task
        try:
            mink_posture_scale = self.env.p.model.keyframe(
                "penalty_default_qpos_weight"
            ).qpos
        except KeyError:
            mink_posture_scale = np.ones(self.env.m.nq)
        try:
            mink_damping_scale = self.env.p.model.keyframe("mink_damping_scale").qpos
        except KeyError:
            mink_damping_scale = np.ones(self.env.m.nq)

        if self.infer_controller_weights:
            mink_posture_scale[:] = 1
            mink_damping_scale[:] = 1
            # up to this point, it matches torso and two arms.
            # we need to handle the separate costs of the spine and base

            # all sliding joints in the robot has high qpos weight (10)
            # and high damping scale (1000)
            sliding_dof_ids = []
            sliding_qpos_ids = []
            for jntid in range(self.env.m.njnt):
                joint = self.env.p.model.joint(jntid)
                if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                    qpos_id = int(joint.qposadr[0])
                    sliding_qpos_ids.append(qpos_id)
                    dof_id = joint.dofadr[0]
                    sliding_dof_ids.append(int(dof_id))
            mink_posture_scale[sliding_qpos_ids] = 10
            mink_damping_scale[sliding_dof_ids] = 100
            # also, the base is parameterized with three joints,
            # one is a rotating joints. They will always be one of the
            # first three joints, and all have high weights
            mink_posture_scale[:3] = 10
            mink_damping_scale[:3] = 1000

            # up to this point, it matches spine.xml
            # the base joint whose axes points forward has qpos weight 100
            jnt_idx = np.array(
                np.where(
                    np.isclose(
                        self.env.m.jnt_axis[:3], [[1, 0, 0]], atol=1e-1, rtol=1e-1
                    ).all(axis=1)
                )
            ).reshape(-1)
            assert len(jnt_idx) == 1, "expected only one base joint"
            mink_posture_scale[jnt_idx[0]] = 100

            # find the rotating joints from spine_3dof.xml
            # it's damping is 200, its armature is 10.0
            spine_rotating_dof_ids = []
            spine_rotating_qpos_ids = []
            for jntid in range(self.env.m.njnt):
                joint = self.env.p.model.joint(jntid)
                is_rotating = joint.type == mujoco.mjtJoint.mjJNT_HINGE
                is_damping_200 = np.isclose(joint.damping, 200, atol=5, rtol=1e-1)
                is_armature_10 = np.isclose(joint.armature, 10.0, atol=1, rtol=1e-1)
                if is_rotating and is_damping_200 and is_armature_10:
                    spine_rotating_dof_ids.append(int(joint.dofadr[0]))
                    spine_rotating_qpos_ids.append(int(joint.qposadr[0]))
            if len(spine_rotating_qpos_ids) > 0:
                mink_damping_scale[spine_rotating_dof_ids] = 100
                # only first joint in the rotating joints has qpos weight 20
                mink_posture_scale[min(spine_rotating_qpos_ids)] = 20

            logging.info(f"inferred mink_posture_scale: {mink_posture_scale}")
            logging.info(f"inferred mink_damping_scale: {mink_damping_scale}")

        mink_posture_cost = mink_posture_scale * self.mink_posture_cost
        mink_damping_cost = mink_damping_scale * self.mink_damping_cost

        # Create posture task
        self.mink_tasks["posture"] = mink.PostureTask(
            model=self.env.m, cost=mink_posture_cost
        )
        self.mink_tasks["damping"] = mink.DampingTask(
            model=self.env.m, cost=mink_damping_cost
        )

        self.mink_limits = {}
        self.mink_limits["configuration"] = mink.ConfigurationLimit(model=self.env.m)

    def mink_policy(
        self,
        obs: ObsDict,
        mink_max_iters: Optional[int] = None,
        tasks: Optional[dict[str, mink.Task]] = None,
    ) -> dict[str, np.ndarray]:
        if mink_max_iters is None:
            mink_max_iters = self.mink_max_iters
        if tasks is None:
            tasks = self.mink_tasks
        self.mink_configuration.update(self.env.d.qpos)

        target_step = min(
            self.tracking_env.episode_step + self.look_ahead_steps,
            len(self.tracking_env.pose_traj) - 1,
        )

        # Set targets for all end effectors
        # pose_traj shape: (T, n_end_effectors, 4, 4)
        for track_link_id, (_, frame_task) in enumerate(
            sorted(
                filter(lambda x: x[0].startswith("tracking_"), tasks.items()),
                key=lambda x: x[0],
            )
        ):
            frame_task = typing.cast(mink.FrameTask, frame_task)
            target_pose = (
                self.tracking_env.augmentation
                @ self.tracking_env.pose_traj[target_step, track_link_id]
                # pose_traj is stored in increasing track_link_id order
            )
            frame_task.set_target(mink.SE3.from_matrix(target_pose))

        configuration = self.mink_configuration
        for _ in range(mink_max_iters):
            try:
                vel = mink.solve_ik(
                    configuration=configuration,
                    tasks=list(tasks.values()),
                    # tasks=self.mink_frame_tasks,
                    dt=self.env.m.opt.timestep,
                    solver=self.mink_solver,
                    limits=list(self.mink_limits.values()),
                    damping=1e-5,
                )
            except mink.NoSolutionFound:
                break
            configuration.integrate_inplace(vel, self.env.m.opt.timestep)

            # Check convergence for all end effectors
            all_converged = True
            for _, frame_task in filter(
                lambda x: x[0].startswith("tracking_"), tasks.items()
            ):
                frame_task = typing.cast(mink.FrameTask, frame_task)
                err = frame_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= self.mink_pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= self.mink_ori_threshold
                if not (pos_achieved and ori_achieved):
                    all_converged = False
                    break

            if all_converged:
                break

        ctrl = np.zeros(self.env.m.nu)
        ctrl[list(self.qpos_to_ctrl_map.values())] = configuration.q[
            list(self.qpos_to_ctrl_map.keys())
        ]
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
            target_qpos = env.rs.uniform(env.m.jnt_range[:, 0], env.m.jnt_range[:, 1])
        else:
            target_qpos = env.d.qpos[:]
        self.mink_tasks["posture"].set_target(target_qpos)
        original_posture_cost = self.mink_tasks["posture"].cost.copy()
        reset_posture_cost = original_posture_cost.copy()
        reset_posture_cost[:3] *= self.reset_base_posture_cost_multiplier
        reset_posture_cost[3:] *= self.reset_body_posture_cost_multiplier
        self.mink_tasks["posture"].set_cost(reset_posture_cost)
        mink_result = self.mink_policy(
            obs={},
            mink_max_iters=self.mink_reset_max_iters,
        )
        self.mink_tasks["posture"].set_cost(original_posture_cost)
        env.d.qpos[list(self.qpos_to_ctrl_map.keys())] = mink_result[
            "ctrl/target_qpos"
        ][list(self.qpos_to_ctrl_map.values())]
        env.d.ctrl[:] = mink_result["ctrl/target_qpos"]
        env.p.forward()
        obs = self.tracking_env.get_obs()
        info = self.tracking_env.get_info()
        hardware = self.tracking_env.hardware_dict
        return obs, info, hardware
