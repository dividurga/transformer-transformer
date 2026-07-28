import pickle
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax.scipy.spatial.transform import Rotation, Slerp
from mujoco import mjx
from mujoco.mjx._src import math

from t2.mjx_env.base_env import MjxEnv, State, make_data, step


def pos_mat_to_4x4(pos: jax.Array, mat: jax.Array) -> jax.Array:
    pose = jnp.eye(4)
    pose = pose.at[0:3, 3].set(pos)
    pose = pose.at[0:3, 0:3].set(mat)
    return pose


class TrackMjxEnv(MjxEnv):
    def __init__(
        self,
        trajectory_file: str,
        max_episode_length: int,
        tracking_links: list[str],
        mj_model_fn: Callable[[], mujoco.MjModel],
        action_scale: float,
        obs_time_indices: list[int],
        pos_sigma: float,
        orn_sigma: float,
        reward_exp_pose_err: float,
        reward_pos_err: float,
        reward_orn_err: float,
        penalty_orientation: float,
        penalty_action_rate: float,
        penalty_torque: float,
        penalty_joint_vel: float,
        penalty_joint_acc: float,
        penalty_joint_pos_limit: float,
        penalty_default_qpos: Union[float, jax.Array],
        reward_clip_positive: bool,
        reward_nan_to_num: Optional[float],
        penalty_bad_contact: float,
        penalty_bad_contact_pairs: list[tuple[str, str]],
        penalty_termination: float,
        contact_threshold: float,
        ctrl_dt: float,
        sim_dt: float,
        solver_iterations: int,
        solver_ls_iterations: int,
        impratio: int,
        impl: Literal["jax", "warp"],
        kick_wait_times: tuple[float, float],
        kick_durations: tuple[float, float],
        velocity_kick: tuple[float, float],
        transport_interval: int,
        transport_magnitude: jax.Array,
        soft_joint_pos_limit_factor: float,
        reset_xy_pos: float,
        reset_z_angle: float,
        reset_qvel: float,
        reset_qpos: float,
        pos_obs_enc: Literal["log-direction", "linear"],
        terminate_on_bad_contact: bool,
        randomize_target_z_pos: tuple[float, float],
        termination_fellover_threshold: float,
        termination_pos_err_threshold: float,
        termination_orn_err_threshold: float,
        termination_pose_consecutive_steps: int,
        obs_noise: dict[str, float],
        clip_actions: tuple[float, float] = (-10, 10),
    ):
        super().__init__(ctrl_dt, sim_dt)
        assert reward_exp_pose_err >= 0.0, "reward_exp_pose_err must be non-negative"
        assert reward_pos_err <= 0.0, "reward_pos_err must be non-positive"
        assert reward_orn_err <= 0.0, "reward_orn_err must be non-positive"
        assert penalty_action_rate <= 0.0, "penalty_action_rate must be non-positive"
        assert penalty_orientation <= 0.0, "penalty_orientation must be non-positive"

        self.max_episode_length = max_episode_length
        self.tracking_links = tracking_links
        self.action_scale = action_scale
        self.obs_time_indices = sorted(obs_time_indices)
        self.clip_actions = clip_actions
        self.pos_sigma = pos_sigma
        self.orn_sigma = orn_sigma
        self.reward_exp_pose_err = reward_exp_pose_err
        self.reward_pos_err = reward_pos_err
        self.reward_orn_err = reward_orn_err
        self.penalty_orientation = penalty_orientation
        self.penalty_action_rate = penalty_action_rate
        self.penalty_torque = penalty_torque
        self.penalty_joint_vel = penalty_joint_vel
        self.penalty_joint_acc = penalty_joint_acc
        self.penalty_joint_pos_limit = penalty_joint_pos_limit

        self.impratio = impratio
        self.solver_iterations = solver_iterations
        self.solver_ls_iterations = solver_ls_iterations
        self.penalty_bad_contact_pairs = penalty_bad_contact_pairs
        self.tracking_links = tracking_links

        self.reward_clip_positive = reward_clip_positive
        self.reward_nan_to_num = reward_nan_to_num
        self.penalty_bad_contact = penalty_bad_contact
        self.penalty_termination = penalty_termination
        self.contact_threshold = contact_threshold
        self.penalty_default_qpos_arg = penalty_default_qpos
        self.impl = impl
        self.kick_wait_times = kick_wait_times
        self.kick_durations = kick_durations
        self.velocity_kick = velocity_kick
        self.transport_interval = transport_interval
        self.transport_magnitude = transport_magnitude
        self.soft_joint_pos_limit_factor = soft_joint_pos_limit_factor
        self.reset_xy_pos = reset_xy_pos
        self.reset_z_angle = reset_z_angle
        self.reset_qvel = reset_qvel
        self.reset_qpos = reset_qpos
        self.pos_obs_enc = pos_obs_enc
        self.terminate_on_bad_contact = terminate_on_bad_contact
        self.randomize_target_z_pos = randomize_target_z_pos
        self.termination_fellover_threshold = termination_fellover_threshold
        self.termination_pos_err_threshold = termination_pos_err_threshold
        self.termination_orn_err_threshold = termination_orn_err_threshold
        self.termination_pose_consecutive_steps = termination_pose_consecutive_steps
        self.obs_noise = obs_noise

        self.mj_model = mj_model_fn()
        self.preprocess_mj_model(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model, impl=impl)

        traj_dt = 0.02  # HARDCODED
        with open(trajectory_file, "rb") as f:
            trajs = jnp.concatenate(pickle.load(f), axis=0)
            frame_times = jnp.arange(trajs.shape[0]) * traj_dt
        self.max_traj_time = frame_times[-1]
        self.trajs = {}
        for site_name in self._site_names:
            target_pos = trajs[:, :3, 3]
            target_rotmat = jnp.array(trajs[:, :3, :3])

            def lerp(
                t: jax.Array,
                xp: jax.Array = frame_times,
                fp: jax.Array = target_pos,
            ) -> jax.Array:
                x = jnp.interp(x=t, xp=xp, fp=fp[:, 0])
                y = jnp.interp(x=t, xp=xp, fp=fp[:, 1])
                z = jnp.interp(x=t, xp=xp, fp=fp[:, 2])
                return jnp.stack([x, y, z], axis=-1)

            slerp = Slerp.init(
                times=frame_times, rotations=Rotation.from_matrix(target_rotmat)
            )

            self.trajs[site_name] = {
                "pos": lerp,
                "rotmat": slerp,
            }

    def preprocess_mj_model(self, mj_model: mujoco.MjModel):
        mj_model.opt.impratio = self.impratio
        mj_model.opt.timestep = self.sim_dt
        mj_model.opt.iterations = self.solver_iterations
        mj_model.opt.ls_iterations = self.solver_ls_iterations
        geom_is_sphere = mj_model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
        mj_model.geom_conaffinity[geom_is_sphere] = 1
        mj_model.geom_contype[geom_is_sphere] = 1
        geom_not_plane = mj_model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
        should_disable_contact = (~geom_is_sphere) & geom_not_plane
        mj_model.geom_conaffinity[should_disable_contact] = 0
        mj_model.geom_contype[should_disable_contact] = 0

        geom_name_to_id = {
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx): geom_idx
            for geom_idx in range(mj_model.ngeom)
        }

        penalty_bad_contact_indices = []
        for geom_names_1, geom_name_2 in self.penalty_bad_contact_pairs:
            geom_idx_1 = geom_idx_2 = -1
            for geom_name_1 in geom_names_1.split("|"):
                geom_idx_1 = geom_name_to_id.get(geom_name_1, -1)
                if geom_idx_1 != -1:
                    break
            for geom_name_2 in geom_name_2.split("|"):
                geom_idx_2 = geom_name_to_id.get(geom_name_2, -1)
                if geom_idx_2 != -1:
                    break
            assert geom_idx_1 != -1 and geom_idx_2 != -1, (
                f"{geom_names_1} {geom_name_2}"
            )
            penalty_bad_contact_indices.append(tuple(sorted((geom_idx_1, geom_idx_2))))
            mj_model.geom_conaffinity[[geom_idx_1, geom_idx_2]] = 1
            mj_model.geom_contype[[geom_idx_1, geom_idx_2]] = 1

        self.penalty_bad_contact_indices = jnp.array(penalty_bad_contact_indices)

        self.nv = mj_model.nv
        self.nq = mj_model.nq

        jnt_is_free = []
        qpos_is_free = []
        qvel_is_free = []
        qpos_free_xy = []
        qpos_free_quat = []
        for jnt_id in range(mj_model.njnt):
            jnt_type = mj_model.jnt_type[jnt_id]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                jnt_is_free.append(True)
                qpos_is_free.extend([True] * 7)
                qvel_is_free.extend([True] * 6)
                qpos_free_xy.extend([True] * 2 + [False] * 5)
                qpos_free_quat.extend([False] * 3 + [True] * 4)
            else:
                jnt_is_free.append(False)
                assert jnt_type in {
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                }
                qpos_is_free.append(False)
                qvel_is_free.append(False)
                qpos_free_xy.append(False)
                qpos_free_quat.append(False)

        self.jnt_is_free = np.array(jnt_is_free)
        self.qpos_is_free = np.array(qpos_is_free)
        self.qvel_is_free = np.array(qvel_is_free)
        self.qpos_free_xy = np.array(qpos_free_xy)
        self.qpos_free_quat = np.array(qpos_free_quat)
        assert self.jnt_is_free.sum() < 2, "at most one free joint"

        site_ids = []

        for site_names in self.tracking_links:
            for site_name in site_names.split("|"):
                site_id = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site_name
                )
                if site_id != -1:
                    site_ids.append(site_id)
                    break

        self._site_ids = jnp.array(site_ids)
        self._site_names = self.tracking_links

        home_keyframe = mj_model.keyframe("home")

        self.init_q = jnp.array(home_keyframe.qpos)
        # action offset will assume there are no other objects in the scene
        # and there may be a non-fixed base
        self.action_offset = jnp.array(home_keyframe.ctrl)
        try:
            weight = mj_model.keyframe("penalty_default_qpos_weight").qpos
            if (weight[:7] == np.array([0, 0, 0, 1, 0, 0, 0])).all():
                weight[:7] = 0
            self.penalty_default_qpos = (
                jnp.array(weight) * self.penalty_default_qpos_arg
            )
        except KeyError:
            self.penalty_default_qpos = jnp.array(
                [0] * 7
                + [self.penalty_default_qpos_arg] * (len(home_keyframe.qpos) - 7)
            )
        assert (self.penalty_default_qpos <= 0).all()

        # Note: First joint is freejoint.
        self._lowers, self._uppers = mj_model.jnt_range[~self.jnt_is_free].T
        self._soft_lowers = self._lowers * self.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self.soft_joint_pos_limit_factor

        self.action_size = mj_model.nu
        self._base_idx = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )

        self.base_mass = mj_model.body_subtreemass[self._base_idx]

    def reset(self, rng: jax.Array) -> State:
        rng, time_offset_rng = jax.random.split(rng)
        time_offset = jax.random.uniform(
            time_offset_rng,
            minval=0.0,
            maxval=self.max_traj_time,
        )
        return self.reset_with_traj(rng, time_offset)

    def reset_with_traj(self, rng: jax.Array, time_offset: jax.Array) -> State:
        qpos = self.init_q
        qvel = jnp.zeros(self.mj_model.nv)

        rng, key = jax.random.split(rng)
        # reset z angle
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(
            key, (1,), minval=-self.reset_z_angle, maxval=self.reset_z_angle
        )
        quat = math.axis_angle_to_quat(jnp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[self.qpos_free_quat], quat)
        qpos = qpos.at[self.qpos_free_quat].set(new_quat)
        # reset joint qpos
        rng, key = jax.random.split(rng)
        qpos = qpos.at[~self.qpos_is_free].set(
            qpos[~self.qpos_is_free]
            + jax.random.uniform(
                key,
                ((~self.qpos_is_free).sum(),),
                minval=-self.reset_qpos,
                maxval=self.reset_qpos,
            )
        )

        rng, key = jax.random.split(rng)
        qvel = qvel.at[~self.qvel_is_free].set(
            jax.random.uniform(
                key,
                ((~self.qvel_is_free).sum(),),
                minval=-self.reset_qvel,
                maxval=self.reset_qvel,
            )
        )

        mocap_quat = jnp.zeros((len(self._site_names), 4))
        mocap_quat = mocap_quat.at[:, 0].set(1.0)

        data = make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.impl,
            nconmax=self._nconmax,
            njmax=self._njmax,
            mocap_pos=jnp.zeros((len(self._site_names), 3)),
            mocap_quat=mocap_quat,
        )
        data = mjx.forward(self.mjx_model, data)
        planar_pos = jnp.mean(data.site_xpos[self._site_ids][..., :2], axis=0)
        # adjust the robot's init pose so that it's at zero
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(
            key, (2,), minval=-self.reset_xy_pos, maxval=self.reset_xy_pos
        )
        data = data.replace(
            qpos=data.qpos.at[self.qpos_free_xy].set(
                data.qpos[self.qpos_free_xy] - planar_pos + dxy
            )
        )
        data = mjx.forward(self.mjx_model, data)

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        time_until_next_kick = jax.random.uniform(
            key1,
            minval=self.kick_wait_times[0],
            maxval=self.kick_wait_times[1],
        )
        steps_until_next_kick = jnp.round(time_until_next_kick / self.dt).astype(
            jnp.int32
        )
        kick_duration_seconds = jax.random.uniform(
            key2,
            minval=self.kick_durations[0],
            maxval=self.kick_durations[1],
        )
        kick_duration_steps = jnp.round(kick_duration_seconds / self.dt).astype(
            jnp.int32
        )
        kick_mag = jax.random.uniform(
            key3,
            minval=self.velocity_kick[0],
            maxval=self.velocity_kick[1],
        )

        up_vec = math.rotate(jnp.array([0, 0, 1]), data.xquat[self._base_idx])

        info: dict[str, Union[jax.Array, float]] = {
            **self.get_init_state_info(
                rng=rng,
                data=data,
                time_offset=time_offset,
            ),
            "rng": rng,
            "steps_until_next_kick": steps_until_next_kick,
            "kick_duration_seconds": kick_duration_seconds,
            "kick_duration": kick_duration_steps,
            "steps_since_last_kick": 0,
            "kick_steps": 0,
            "kick_dir": jnp.zeros(3),
            "kick_mag": kick_mag,
            "up_vec": up_vec,
            "bad_contact": jnp.zeros(()),
        }

        data = data.replace(
            mocap_pos=info["target_pos"],
            mocap_quat=Rotation.from_matrix(info["target_rot"]).as_quat(
                scalar_first=True
            ),
        )
        obs = self._get_obs(data, info)
        rewards = self.compute_reward(
            data=data, action=jnp.zeros(self.action_size), info=info
        )
        reward = sum(rewards.values()) * self.dt

        if self.reward_nan_to_num is not None:
            reward = jnp.nan_to_num(
                reward,
                nan=self.reward_nan_to_num,
                posinf=self.reward_nan_to_num,
                neginf=self.reward_nan_to_num,
            )

        if self.reward_clip_positive:
            reward = jnp.maximum(reward, 0.0)
        done = jnp.zeros(())
        metrics = {**self.get_init_task_metrics(), **rewards, "reward": reward}
        state = State(data, obs, reward, done, metrics, info)
        return state

    def get_init_state_info(
        self, rng: jax.Array, data: mjx.Data, time_offset: jax.Array
    ) -> dict[str, Union[jax.Array, float]]:
        rng, key = jax.random.split(rng)
        z_pos_offset = jax.random.uniform(
            key,
            minval=self.randomize_target_z_pos[0],
            maxval=self.randomize_target_z_pos[1],
        )
        init_target_pos = jnp.stack(
            [
                self.trajs[link_name]["pos"](time_offset)
                for link_name in self._site_names
            ]
        )
        xy_offset = -init_target_pos[:, :2].mean(axis=0)
        pos_offset = jnp.concatenate([xy_offset, jnp.array([z_pos_offset])])
        init_target_rot = jnp.stack(
            [
                self.trajs[link_name]["rotmat"](time_offset).as_matrix()
                for link_name in self._site_names
            ]
        )
        state_info = {
            "state_is_nan": jnp.zeros(()),
            "prev_action": jnp.zeros(len(self.action_offset)),
            "steps": 0,
            "target_pos": init_target_pos + pos_offset,
            "target_rot": init_target_rot,
            "pos_offset": pos_offset,
            "time_offset": time_offset,
            "actual_pos": data.site_xpos[self._site_ids],
            "actual_rot": data.site_xmat[self._site_ids],
            "pos_err": jnp.zeros(len(self._site_names)),
            "orn_err": jnp.zeros(len(self._site_names)),
            "done/fellover": jnp.zeros(()),
            "done/badcontact": jnp.zeros(()),
            "done/pos_err": jnp.zeros(()),
            "done/orn_err": jnp.zeros(()),
            "pose_steps_violated": jnp.zeros(()),
        }
        return state_info

    def get_target(
        self,
        step: jax.Array,
        pos_offset: jax.Array,
        time_offset: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        t = step * self.dt + time_offset
        t = t % self.max_traj_time
        target_pos = []
        target_rotmat = []
        for site_name in self._site_names:
            target_pos.append(self.trajs[site_name]["pos"](t) + pos_offset)
            target_rotmat.append(self.trajs[site_name]["rotmat"](t).as_matrix())

        return jnp.stack(target_pos), jnp.stack(target_rotmat)

    def update_state_info(
        self, state: State, data: mjx.Data
    ) -> tuple[State, jax.Array]:
        data_is_nan = jnp.any(
            jnp.array([jnp.isnan(x).any() for x in jax.tree_util.tree_leaves(data)])
        )
        state.info["state_is_nan"] = data_is_nan.astype(jnp.float32)
        # for nan values, replace with previous value
        data = jax.tree.map(
            lambda x, y: jnp.where(jnp.isnan(x), y, x), data, state.data
        )

        target_pos, target_rot = self.get_target(
            step=state.info["steps"],
            pos_offset=state.info["pos_offset"],
            time_offset=state.info["time_offset"],
        )

        state.info["target_pos"] = target_pos
        state.info["target_rot"] = target_rot

        data = data.replace(
            mocap_pos=target_pos,
            mocap_quat=Rotation.from_matrix(target_rot).as_quat(scalar_first=True),
        )
        state = state.replace(data=data)

        actual_pos = data.site_xpos[self._site_ids]
        actual_rot = data.site_xmat[self._site_ids]
        state.info["actual_pos"] = actual_pos
        state.info["actual_rot"] = actual_rot
        pos_err = jnp.linalg.norm(
            state.info["target_pos"] - state.info["actual_pos"], axis=-1
        )
        state.info["pos_err"] = pos_err
        rot_err_mat = state.info["target_rot"] @ jnp.transpose(
            state.info["actual_rot"], (0, 2, 1)
        )
        state.info["up_vec"] = math.rotate(
            jnp.array([0, 0, 1]), data.xquat[self._base_idx]
        )

        trace = jnp.trace(rot_err_mat, axis1=1, axis2=2)
        trace = jnp.clip(trace, -1.0 + 1e-8, 3.0 - 1e-8)
        orn_err = jnp.arccos((trace - 1.0) / 2.0)
        state.info["orn_err"] = orn_err

        state.info["bad_contact"] = (
            contact_penalty_jit(
                penalized_pairs=self.penalty_bad_contact_indices,
                contact_pairs=jnp.sort(data.contact.geom, axis=1),
                contact_dists=data.contact.dist,
                contact_threshold=self.contact_threshold,
            )
            if self.penalty_bad_contact_indices.shape[0] > 0
            else 0.0
        )

        return state, data_is_nan

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, self.clip_actions[0], self.clip_actions[1])
        state = self._maybe_apply_kick(state)
        rng = state.info["rng"]
        joint_targets = self.action_offset + action * self.action_scale
        data = step(
            model=self.mjx_model,
            data=state.data,
            action=joint_targets,
            n_substeps=self.n_substeps,
        )
        state, data_is_nan = self.update_state_info(state, data)
        # done conditions
        state = self.get_done(state)
        state = state.replace(
            done=(state.done.astype(jnp.bool_) | data_is_nan).astype(jnp.float32)
        )

        rewards = self.compute_reward(data=data, action=action, info=state.info)
        reward = sum(rewards.values()) * self.dt

        if self.reward_nan_to_num:
            reward = jnp.nan_to_num(
                reward,
                nan=self.reward_nan_to_num,
                posinf=self.reward_nan_to_num,
                neginf=self.reward_nan_to_num,
            )

        if self.reward_clip_positive:
            reward = jnp.maximum(reward, 0.0)

        # state management
        state = state.replace(
            reward=reward,
            metrics={
                **self.get_task_metrics(state.info),
                **rewards,
                "reward": reward,
            },
            info={
                **state.info,
                "prev_action": action,
            },
        )

        state.info["steps"] += 1

        if self.transport_interval != -1:
            data = state.data
            key, rng = jax.random.split(rng)
            transport_dxyz = jax.random.uniform(
                key,
                shape=(3,),
                minval=self.transport_magnitude[0],
                maxval=self.transport_magnitude[1],
            ) * (state.info["steps"] % self.transport_interval == 0).astype(jnp.float32)
            transport_dxyzquat = jnp.concatenate([transport_dxyz, jnp.zeros(4)])
            data = data.replace(
                qpos=data.qpos.at[self.qpos_is_free].set(
                    data.qpos[self.qpos_is_free] + transport_dxyzquat
                )
            )
            state = state.replace(data=data)
            mjx.forward(self.mjx_model, data)

        state.info["rng"] = rng
        obs = self._get_obs(data, state.info)

        state = state.replace(
            obs=obs,
        )
        return state

    def compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Union[jax.Array, float]],
    ) -> dict[str, jax.Array]:
        rewards = {}
        rewards["action_rate"] = (
            jnp.sum((action - info["prev_action"]) ** 2) * self.penalty_action_rate
        )
        rewards["torque"] = jnp.sum(data.actuator_force**2) * self.penalty_torque
        joint_vel = data.qvel[~self.qvel_is_free]
        rewards["joint_vel"] = jnp.sum(joint_vel**2) * self.penalty_joint_vel
        joint_acc = data.qacc[~self.qvel_is_free]
        rewards["joint_acc"] = jnp.sum(joint_acc**2) * self.penalty_joint_acc
        pos_err = info["pos_err"]
        orn_err = info["orn_err"]
        pos_err_squared = pos_err**2
        orn_err_squared = orn_err**2

        rewards["exp_pose_err"] = (
            jnp.exp(
                -orn_err_squared / self.orn_sigma - pos_err_squared / self.pos_sigma
            )
            * self.reward_exp_pose_err
        ).sum()
        rewards["pos_err"] = pos_err.sum() * self.reward_pos_err
        rewards["orn_err"] = orn_err.sum() * self.reward_orn_err
        rewards["bad_contact"] = info["bad_contact"] * self.penalty_bad_contact

        up_vec = info["up_vec"]

        rewards["upright"] = jnp.sum(jnp.square(up_vec[:2])) * self.penalty_orientation

        joint_pos = data.qpos[~self.qpos_is_free]

        out_of_limits = -jnp.clip(joint_pos - self._soft_lowers, None, 0.0)
        out_of_limits += jnp.clip(joint_pos - self._soft_uppers, 0.0, None)
        rewards["joint_pos_limit"] = (
            jnp.sum(out_of_limits) * self.penalty_joint_pos_limit
        )
        rewards["default_qpos"] = jnp.sum(
            jnp.square(data.qpos - self.init_q) * self.penalty_default_qpos
        )
        rewards["termination"] = (
            info["done/fellover"]
            + info["done/badcontact"]
            + info["done/pos_err"]
            + info["done/orn_err"]
            + info["state_is_nan"]
        ) * self.penalty_termination
        return rewards

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Union[jax.Array, float]],
    ) -> jax.Array:
        # local frame gravity
        rng = info["rng"]
        gravity = data.xmat[self._base_idx].T @ jnp.array([0, 0, -1])
        key, rng = jax.random.split(rng)
        gravity = (
            gravity + jax.random.normal(key, gravity.shape) * self.obs_noise["gravity"]
        )

        # joint relative to default
        key, rng = jax.random.split(rng)
        joint_angles = (data.qpos - self.init_q)[~self.qpos_is_free]
        joint_angles = (
            joint_angles
            + jax.random.normal(key, joint_angles.shape) * self.obs_noise["qpos"]
        )
        joint_vel = data.qvel[~self.qvel_is_free]
        key, rng = jax.random.split(rng)
        joint_vel = (
            joint_vel + jax.random.normal(key, joint_vel.shape) * self.obs_noise["qvel"]
        )
        # last action so the policy can account for action rate
        last_action = info["prev_action"]
        obs_list = [
            gravity,
            joint_angles,
            joint_vel,
            last_action,
        ]

        if self.jnt_is_free.sum() == 1:
            # local frame velocities
            base_vel = data.qvel[self.qvel_is_free]
            assert base_vel.shape == (6,)
            lin_vel = data.xmat[self._base_idx].T @ base_vel[:3]
            key, rng = jax.random.split(rng)
            lin_vel = (
                lin_vel
                + jax.random.normal(key, lin_vel.shape) * self.obs_noise["lin_vel"]
            )
            ang_vel = data.xmat[self._base_idx].T @ base_vel[3:6]
            key, rng = jax.random.split(rng)
            ang_vel = (
                ang_vel
                + jax.random.normal(key, ang_vel.shape) * self.obs_noise["ang_vel"]
            )
            obs_list.extend([lin_vel, ang_vel])

        actual_pos = info["actual_pos"]
        actual_rot = info["actual_rot"]
        actual_pose = jnp.array(
            [pos_mat_to_4x4(pos, mat) for pos, mat in zip(actual_pos, actual_rot)]
        )
        inv_actual_pose = jnp.linalg.inv(actual_pose)
        for step_offset in self.obs_time_indices:
            obs_step = info["steps"] + step_offset
            target_pos, target_rot = self.get_target(
                step=obs_step,
                pos_offset=info["pos_offset"],
                time_offset=info["time_offset"],
            )
            target_pose = jnp.array(
                [pos_mat_to_4x4(pos, mat) for pos, mat in zip(target_pos, target_rot)]
            )
            delta_pose = inv_actual_pose @ target_pose
            delta_pos = delta_pose[..., :3, 3]
            delta_rot_mat = delta_pose[..., :3, :3]

            if self.pos_obs_enc == "log-direction":
                distance = jnp.linalg.norm(delta_pos, axis=-1) + 1e-8
                direction = delta_pos / distance[..., None]
                pos_obs = jnp.concatenate(
                    [jnp.log(distance)[..., None], direction], axis=-1
                ).reshape(-1)
            elif self.pos_obs_enc == "linear":
                pos_obs = delta_pos.reshape(-1)
            else:
                raise ValueError(f"Unknown pos_obs_enc: {self.pos_obs_enc}")
            orn_obs = delta_rot_mat.reshape(-1)
            obs_list.append(pos_obs)
            obs_list.append(orn_obs)
        obs = jnp.concatenate(obs_list)
        return obs

    def get_task_metrics(self, info: dict[str, jax.Array]) -> dict[str, jax.Array]:
        metrics = {}
        pos_err = info["pos_err"]
        orn_err = info["orn_err"]
        for i, link_name in enumerate(self._site_names):
            metrics[f"pos_err/{link_name}"] = pos_err[i]
            metrics[f"orn_err/{link_name}"] = orn_err[i]
        metrics["pos_err/all"] = pos_err.mean()
        metrics["orn_err/all"] = orn_err.mean()
        metrics["done/fellover"] = info["done/fellover"]
        metrics["done/badcontact"] = info["done/badcontact"]
        metrics["done/pos_err"] = info["done/pos_err"]
        metrics["done/orn_err"] = info["done/orn_err"]
        return metrics

    def get_init_task_metrics(self) -> dict[str, jax.Array]:
        return self.get_task_metrics(
            info={
                "pos_err": jnp.zeros(len(self._site_names)),
                "orn_err": jnp.zeros(len(self._site_names)),
                "up_vec": jnp.array([0, 0, 1]),
                "done/fellover": jnp.zeros(()),
                "done/badcontact": jnp.zeros(()),
                "done/pos_err": jnp.zeros(()),
                "done/orn_err": jnp.zeros(()),
            },
        )

    def get_done(self, state: State) -> State:
        end_of_traj = state.info["steps"] >= self.max_episode_length
        fell_over = (
            jnp.dot(state.info["up_vec"], jnp.array([0, 0, 1]))
            < self.termination_fellover_threshold
        )
        bad_contact = jnp.logical_and(
            state.info["bad_contact"] > 0.0, self.terminate_on_bad_contact
        )
        pos_err = state.info["pos_err"]
        orn_err = state.info["orn_err"]
        pos_err_violation = jnp.any(pos_err > self.termination_pos_err_threshold)
        orn_err_violation = jnp.any(orn_err > self.termination_orn_err_threshold)
        state.info["pose_steps_violated"] = jnp.where(
            pos_err_violation | orn_err_violation,
            state.info["pose_steps_violated"] + 1,
            jnp.zeros(()),
        )
        start_pose_termination = (
            state.info["pose_steps_violated"] >= self.termination_pose_consecutive_steps
        )
        pos_err_done = pos_err_violation & start_pose_termination
        orn_err_done = orn_err_violation & start_pose_termination
        done = bad_contact | end_of_traj | fell_over | pos_err_done | orn_err_done
        state.info["done/fellover"] = fell_over.astype(jnp.float32)
        state.info["done/badcontact"] = bad_contact.astype(jnp.float32)
        state.info["done/pos_err"] = pos_err_done.astype(jnp.float32)
        state.info["done/orn_err"] = orn_err_done.astype(jnp.float32)

        return state.replace(done=done.astype(jnp.float32))

    def _maybe_apply_kick(self, state: State) -> State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jnp.pi * 2)
            return jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])

        def apply_kick(state: State) -> State:
            t = state.info["kick_steps"] * self.dt
            u_t = 0.5 * jnp.sin(jnp.pi * t / state.info["kick_duration_seconds"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            force = (
                u_t  # (unitless)
                * self.base_mass  # kg
                * state.info["kick_mag"]  # m/s
                / state.info["kick_duration_seconds"]  # 1/s
            )
            xfrc_applied = jnp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self._base_idx, :3].set(
                force * state.info["kick_dir"]
            )
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state = state.replace(data=data)
            state.info["steps_since_last_kick"] = jnp.where(
                state.info["kick_steps"] >= state.info["kick_duration"],
                0,
                state.info["steps_since_last_kick"],
            )
            state.info["kick_steps"] += 1
            return state

        def wait(state: State) -> State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_kick"] += 1
            xfrc_applied = jnp.zeros((self.mjx_model.nbody, 6))
            data = state.data.replace(xfrc_applied=xfrc_applied)
            state.info["kick_steps"] = jnp.where(
                state.info["steps_since_last_kick"]
                >= state.info["steps_until_next_kick"],
                0,
                state.info["kick_steps"],
            )
            state.info["kick_dir"] = jnp.where(
                state.info["steps_since_last_kick"]
                >= state.info["steps_until_next_kick"],
                gen_dir(rng),
                state.info["kick_dir"],
            )
            return state.replace(data=data)

        return jax.lax.cond(
            state.info["steps_since_last_kick"] >= state.info["steps_until_next_kick"],
            apply_kick,
            wait,
            state,
        )


def contact_penalty(
    penalized_pairs: jnp.ndarray,  # (N, 2), each row sorted (gid_i, gid_j)
    contact_pairs: jnp.ndarray,  # (M, 2), each row sorted (gid_i, gid_j)
    contact_dists: jnp.ndarray,  # (M,)
    *,
    contact_threshold: float = 0.0,
    mode: str = "binary",  # "hinge", "quadratic", or "binary"
):
    # Hash each pair (i, j) into a unique integer key = i * base + j
    max_id = jnp.maximum(
        penalized_pairs.max(initial=0),
        contact_pairs.max(initial=0),
    )
    base = max_id + 1

    def to_key(pairs):  # (K,2) -> (K,)
        return pairs[:, 0] * base + pairs[:, 1]

    penalized_keys = jnp.sort(to_key(penalized_pairs))  # (N,)
    contact_keys = to_key(contact_pairs)  # (M,)

    # Membership test via binary search
    idx = jnp.searchsorted(penalized_keys, contact_keys, side="left")
    in_bounds = idx < penalized_keys.size
    is_penalized = jnp.where(in_bounds, penalized_keys[idx] == contact_keys, False)

    # Penetration depth (positive if colliding)
    depth = jnp.clip(contact_threshold - contact_dists, a_min=0.0)

    if mode == "hinge":
        # linear penalty on depth
        per_contact = depth
    elif mode == "quadratic":
        # quadratic penalty on depth (smooth, common)
        per_contact = depth * depth
    elif mode == "binary":
        # binary penalty on depth
        per_contact = jnp.where(depth > 0, 1.0, 0.0)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Only keep contacts that are (a) in penalized set and (b) actually colliding
    mask = is_penalized & (depth > 0)
    return jnp.sum(jnp.where(mask, per_contact, 0.0))


# JIT-compiled version
contact_penalty_jit = jax.jit(contact_penalty)
