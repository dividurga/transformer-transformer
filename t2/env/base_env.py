import logging
from collections import deque
from io import BytesIO
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")

import mujoco
import numpy as np
import seaborn as sns
from dm_control import mjcf
from dm_control.mujoco.engine import MovableCamera
from dm_control.rl.control import PhysicsError
from gym import spaces
from matplotlib import pyplot as plt
from PIL import Image

from t2.env.mj_utils import (
    default_root_element,
    find_rigidly_attached_descendant_with_contact_geom,
    render_opt,
    set_up_default_scene,
)
from t2.robotok.io import serialize
from t2.robotok.token import Robot
from t2.robotok.tokenizer import preprocess_mjcf, tokenize
from t2.utils.misc import add_text_to_image

ObsDict = dict[str, np.ndarray]
InfoDict = dict[str, int | float | bool]
HardwareDict = dict[str, np.ndarray | str]

NUM_VIS_HISTORY_STEPS = 100


class BaseEnv:
    def __init__(
        self,
        episode_len: int,
        sim_dt: float,
        ctrl_dt: float,
        robot_generator: Callable[[int], mjcf.RootElement],
        remove_visual_asset: bool = False,  # enable to save some RAM
        impratio: float = 1.0,
        add_plane: bool = False,
        termination_fellover_threshold: Optional[float] = 0.2,
        max_vis_actuator_force: float = 50.0,
        gravcomp: bool = False,
    ):
        self.hardware_seed = -1
        self.dtype = np.float32
        self.episode_len = episode_len
        self.sim_dt = sim_dt
        self.ctrl_dt = ctrl_dt
        self.impratio = impratio
        self.num_repeat_actions = round(self.ctrl_dt / self.sim_dt)

        self.episode_step = 0
        self.physics_camera = None
        self.render_opt = None
        self.__past_actuator_forces = deque(maxlen=NUM_VIS_HISTORY_STEPS)
        self.max_vis_actuator_force = max_vis_actuator_force
        self.p: mjcf.Physics
        self.m: mujoco.MjModel
        self.d: mujoco.MjData
        self.robot_generator = robot_generator
        self.camera_lookat = np.array([0, 0, 0])
        self.termination_fellover_threshold = termination_fellover_threshold

        # RoboTok configuration
        self.remove_visual_asset = remove_visual_asset
        self.add_plane = add_plane
        self.gravcomp = gravcomp

    def get_render_text_rows(self) -> list[str]:
        max_actuator_force = np.max(np.abs(self.d.actuator_force))
        max_actuator_speed = np.max(np.abs(self.d.actuator_velocity))
        return [
            f"step: {self.episode_step}",
            f"max force: {max_actuator_force:.1f}Nm",
            f"max speed: {max_actuator_speed:.1f}rad/s",
        ]

    def get_info(self) -> dict[str, int | float | bool]:
        act_force = np.abs(self.d.actuator_force)
        act_vel = np.abs(self.d.actuator_velocity)
        return {
            "metric/actuator/force/sum": act_force.sum(),
            "metric/actuator/force/mean": act_force.mean(),
            "metric/actuator/force/max": act_force.max(),
            "metric/actuator/force/std": act_force.std(),
            "metric/actuator/energy/sum": (act_force * act_vel).sum(),
            "metric/actuator/energy/mean": (act_force * act_vel).mean(),
            "metric/actuator/energy/max": (act_force * act_vel).max(),
            "metric/actuator/energy/std": (act_force * act_vel).std(),
            "done/timeout": self.get_timeout(),
            "done/bad_termination": self.get_bad_termination(),
            "done/unstable": False,
        }

    def post_process_mjcf(self, mjcf_model: mjcf.RootElement) -> mjcf.RootElement:
        return mjcf_model

    def reset_hardware(self, hardware_seed: int):
        self.hardware_seed = int(hardware_seed)
        mjcf_model = self.robot_generator(
            hardware_seed,
        )
        mjcf_model = preprocess_mjcf(
            mjcf_model,
            remove_visual=self.remove_visual_asset,
        )
        (
            tokenized_robot,
            mjjntid_to_dyna_jntid,
            mjgeomid_to_linkid,
            mjsiteid_to_tracklinkid,
        ) = tokenize(
            mj_robot=mjcf_model,
        )  # type: ignore
        mjcf_model = self.post_process_mjcf(mjcf_model)
        self.post_hardware_reset(
            mjcf_model=mjcf_model,
            tokenized_robot=tokenized_robot,
            mjjntid_to_dyna_jntid=mjjntid_to_dyna_jntid,
            mjgeomid_to_linkid=mjgeomid_to_linkid,
            mjsiteid_to_tracklinkid=mjsiteid_to_tracklinkid,
        )

    def post_hardware_reset(
        self,
        mjcf_model: mjcf.RootElement,
        tokenized_robot: Robot,
        mjjntid_to_dyna_jntid: dict[int, int],
        mjgeomid_to_linkid: dict[int, int],
        mjsiteid_to_tracklinkid: dict[int, int],
    ):
        if self.add_plane:
            plane = set_up_default_scene(default_root_element(), plane_z_pos=0.0)
            mjcf_model.attach(plane)
        physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        assert physics is not None
        self.p = physics

        self.m = self.p.model.ptr
        self.d = self.p.data.ptr

        self.m.opt.timestep = self.sim_dt
        self.m.opt.impratio = self.impratio

        if self.gravcomp:
            self.m.body_gravcomp[:] = 1

        self.mjjntid_to_dyna_jntid = mjjntid_to_dyna_jntid
        # make sure joints are either slide or hinge
        for mjjntid in self.mjjntid_to_dyna_jntid.keys():
            joint_type = self.m.jnt_type[mjjntid]
            if (
                joint_type != mujoco.mjtJoint.mjJNT_HINGE
                and joint_type != mujoco.mjtJoint.mjJNT_SLIDE
            ):
                raise ValueError("Dynamic foints can only be slide or hinge")
        self.mjgeomid_to_linkid = mjgeomid_to_linkid
        self.mjsiteid_to_tracklinkid = mjsiteid_to_tracklinkid

        self.num_dyna_joints = len(tokenized_robot.dynamic_joints)
        self.num_free_links = sum(
            1 for link in tokenized_robot.links if link.free_link_idx != -1
        )
        assert self.num_free_links <= 1, "at most one free link is supported"
        self.num_links = len(tokenized_robot.links)
        self.num_actuators = len(tokenized_robot.actuators)
        link_to_tracklink = {
            link.idx: link.track_link_idx
            for link in sorted(tokenized_robot.links, key=lambda l: l.idx)
            if link.track_link_idx != -1
        }
        self.track_mj_geom_ids = []  # this must be in order sorted by track_link/id
        self.track_link_ids = []

        for link_id, track_link_id in sorted(
            link_to_tracklink.items(), key=lambda x: x[1]
        ):
            self.track_link_ids.append(track_link_id)
            for mj_geom_id, other_link_id in self.mjgeomid_to_linkid.items():
                if link_id == other_link_id:
                    self.track_mj_geom_ids.append(mj_geom_id)
        assert self.track_link_ids == list(range(len(self.track_link_ids)))

        # Build track_mj_site_ids sorted by track_link_id
        # This maps track_link_idx -> mj_site_id (inverts mjsiteid_to_tracklinkid)
        self.tracklinkid_to_mjsiteid = {
            track_link_id: mj_site_id
            for mj_site_id, track_link_id in mjsiteid_to_tracklinkid.items()
        }
        self.track_mj_site_ids = [
            self.tracklinkid_to_mjsiteid[track_link_id]
            for track_link_id in self.track_link_ids
        ]

        # TODO haven't done anything to anything to ensure `free_mj_geom_ids`
        # is in sorted order by free_link/id, since this is trivially satisfied
        # if there is at most one free joint (assertion below).
        # Need to do some book keeping to support multiple free joints.
        free_link_ids = [
            link.idx for link in tokenized_robot.links if link.free_link_idx != -1
        ]
        assert len(free_link_ids) == self.num_free_links, "mismatched free link ids"

        mj_free_body_ids = []
        for mjgeomid, linkid in mjgeomid_to_linkid.items():
            if linkid in free_link_ids:
                mj_body_id = self.m.geom_bodyid[mjgeomid]
                mj_free_body_ids.append(mj_body_id)
        self.free_mj_geom_ids = []
        for mj_body_id in set(mj_free_body_ids):
            descendant_body_id = find_rigidly_attached_descendant_with_contact_geom(
                self.m, mj_body_id
            )
            geom_ids = np.arange(
                self.m.body_geomadr[descendant_body_id],
                self.m.body_geomadr[descendant_body_id]
                + self.m.body_geomnum[descendant_body_id],
            )
            is_contact = np.logical_and(
                self.m.geom_contype[geom_ids] != 0,
                self.m.geom_conaffinity[geom_ids] != 0,
            )
            contact_geom_ids = geom_ids[is_contact]
            self.free_mj_geom_ids.extend(contact_geom_ids)
        self.hardware_dict = serialize(tokenized_robot, include_states=False)
        self.hardware_dict["metadata/seed"] = np.array(
            self.hardware_seed, dtype=np.uint64
        )
        self.hardware_dict["metadata/robot_generator"] = repr(self.robot_generator)

        self.action_space = spaces.Box(
            low=self.m.actuator_ctrlrange[:, 0].astype(self.dtype),
            high=self.m.actuator_ctrlrange[:, 1].astype(self.dtype),
            dtype=self.dtype,
        )

    def reset_no_obs(self, seed: int):
        self.rs = np.random.RandomState(seed)
        self.episode_step = 0
        self.close()
        self.__past_actuator_forces.clear()
        self.p.reset(0 if self.m.nkey > 0 else None)
        self.p.forward()

    def reset(self, seed: int) -> tuple[ObsDict, InfoDict, HardwareDict]:
        self.reset_no_obs(seed=seed)
        obs = self.get_obs()
        info = self.get_info()
        return obs, info, self.hardware_dict

    def get_joint_obs(self) -> ObsDict:
        m = self.m
        d = self.d

        qpos = d.qpos[:]
        qvel = d.qvel[:]

        dyna_jnt_qpos = np.zeros(
            (self.num_dyna_joints, 1),
            dtype=self.dtype,
        )
        dyna_jnt_qvel = np.zeros(
            (self.num_dyna_joints, 1),
            dtype=self.dtype,
        )

        for mj_jnt_id, jnt_id in sorted(
            self.mjjntid_to_dyna_jntid.items(), key=lambda x: x[1]
        ):
            if m.jnt_type[mj_jnt_id] not in {
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            }:
                # leave ball joint state as zero
                continue
            qpos_idx = m.jnt_qposadr[mj_jnt_id]
            qvel_idx = m.jnt_dofadr[mj_jnt_id]
            dyna_jnt_qpos[jnt_id] = qpos[qpos_idx]
            dyna_jnt_qvel[jnt_id] = qvel[qvel_idx]

        return {
            "dyna_joint_obs/qpos": dyna_jnt_qpos,
            "dyna_joint_obs/qvel": dyna_jnt_qvel,
        }

    def get_pose_obs(self) -> ObsDict:
        d = self.d
        pose_obs = {}
        if len(self.free_mj_geom_ids) == 0:
            return pose_obs

        pose_obs["free_link_obs/pos"] = np.zeros(
            (len(self.free_mj_geom_ids), 3),
            dtype=self.dtype,
        )
        pose_obs["free_link_obs/rotmat"] = np.zeros(
            (len(self.free_mj_geom_ids), 9),
            dtype=self.dtype,
        )

        for idx, mj_geom_id in enumerate(self.free_mj_geom_ids):
            pose_obs["free_link_obs/pos"][idx] = d.geom_xpos[mj_geom_id]
            pose_obs["free_link_obs/rotmat"][idx] = d.geom_xmat[mj_geom_id]

        return pose_obs

    def get_actuator_obs(self) -> ObsDict:
        """Return actuator state observations."""
        d = self.d
        vel = np.array(d.actuator_velocity, dtype=self.dtype).reshape(
            self.num_actuators, 1
        )
        force = np.array(d.actuator_force, dtype=self.dtype).reshape(
            self.num_actuators, 1
        )
        self.__past_actuator_forces.append(np.abs(d.actuator_force))
        return {
            "actuator_obs/velocity": vel,
            "actuator_obs/force": force,
        }

    def get_task_obs(self) -> ObsDict:
        return {}

    def get_obs(self) -> ObsDict:
        self.p.forward()
        return {
            **self.get_joint_obs(),
            **self.get_pose_obs(),
            **self.get_actuator_obs(),
            **self.get_task_obs(),
        }

    def get_timeout(self) -> bool:
        end_of_episode = self.episode_step >= self.episode_len
        return end_of_episode

    def get_bad_termination(self) -> bool:
        # TODO the target orientation needs to depend on the geometry
        # (it might not be upright in global frame)
        fellover = (
            any(
                np.dot(
                    self.d.geom_xmat[mj_geom_id].reshape(3, 3) @ np.array([0, 0, 1]),
                    np.array([0, 0, 1]),
                )
                < self.termination_fellover_threshold
                for mj_geom_id in self.free_mj_geom_ids
            )
            if self.termination_fellover_threshold
            else False
        )
        return fellover

    def step(self, ctrl: np.ndarray) -> tuple[ObsDict, float, bool, dict[str, Any]]:
        assert self.episode_step < self.episode_len, (
            "episode is done, call `collect_rollout()`"
        )

        reward = 0.0  # Default reward, should be overridden in subclasses

        self.d.ctrl[:] = ctrl
        unstable = False
        for _ in range(self.num_repeat_actions):
            try:
                self.p.step()
            except PhysicsError as e:
                logging.error(e)
                unstable = True

        self.episode_step += 1
        timeout = self.get_timeout()
        bad_termination = self.get_bad_termination()
        done = unstable or timeout or bad_termination
        obs = self.get_obs()
        info = self.get_info()
        return obs, reward, done, info

    def render(self, mode="human") -> Optional[np.ndarray]:
        """
        Render the environment.
        Creates the camera on first call.
        """
        # Create camera on first render call
        if self.physics_camera is None and self.p is not None:
            self.physics_camera = MovableCamera(
                physics=self.p,
                height=400,
                width=400,
                scene_callback=None,
            )
            self.render_opt = render_opt()
            self.render_opt.geomgroup[3] = 1
            self.render_opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        if self.physics_camera is not None:
            lookat = self.camera_lookat
            frequency = 0.005
            magnitude = 0.1
            distance = 2.0
            azimuth = (
                magnitude * np.sin(self.episode_step * 2 * np.pi * frequency)
                + np.pi * 3 / 4
            )
            elevation = (
                magnitude * np.cos(self.episode_step * 2 * np.pi * frequency)
                - np.pi / 5
            )
            azimuth = np.rad2deg(azimuth)
            elevation = np.rad2deg(elevation)
            self.physics_camera.set_pose(
                lookat=lookat,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
            )
            rgb = self.physics_camera.render(
                overlays=(),
                depth=False,
                segmentation=False,
                scene_option=self.render_opt,
                render_flag_overrides=None,
            )
            text_rows = self.get_render_text_rows()
            rgb = add_text_to_image(
                rgb,
                text_rows,
                [(10, 20 * i + 10) for i in range(len(text_rows))],
                fontsize=18,
                color="rgb(255, 255, 255)",
            )

            # plot actuator force
            sns.set_style("darkgrid")
            fig = plt.figure(figsize=(4, 1.12), dpi=100)
            plt.plot(np.array(self.__past_actuator_forces))
            plt.ylim(0, self.max_vis_actuator_force)
            plt.xlim(0, NUM_VIS_HISTORY_STEPS)
            buf = BytesIO()
            plt.tight_layout(pad=0)
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_img = np.array(Image.open(buf))[..., :3]
            buf.close()
            plt.close(fig)
            return np.concatenate([rgb, plot_img], axis=0)
        return None

    def close(self):
        """Clean up resources."""
        if self.physics_camera is not None:
            self.physics_camera._scene.free()  # pylint: disable=protected-access
            self.physics_camera = None
            self.render_opt = None
