#!/usr/bin/env python3
"""
Dataset visualizer script for T2.

This script loads a zarr dataset, creates robot instances from batch data,
and visualizes them using MuJoCo viewer with target poses and joint configurations.

Optionally exports physics states to pickle files for later analysis or rendering.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from dm_control import mjcf
from mujoco import viewer
from omegaconf import DictConfig
from transforms3d import quaternions

from t2.env.mj_utils import (
    TARGET_SITE_GROUP,
    TRACKING_SITE_GROUP,
    add_mocap_body_with_site,
    render_opt,
    set_up_default_scene,
)
from t2.eval.hardware_optimization import post_process_mjcf_solver_parameters
from t2.robotok.io import deserialize
from t2.robotok.tokenizer import detokenize

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION: Physics State Parsing (from visualize_robotoken_diffusion.py)
# =============================================================================


def parse_physics_state(
    physics,
    anchor_positions: Optional[np.ndarray] = None,
    max_distance: float = 2.0,
    colormap_name: str = "jet",
    prev_data_dict: Optional[dict[str, list]] = None,
    use_original_rgba: bool = True,
) -> dict[str, list]:
    """Parse physics state into a dictionary of geom properties.

    Extracts geom types, sizes, colors, positions, and orientations from
    a MuJoCo physics state. Optionally colors geoms based on distance from
    anchor positions.

    Args:
        physics: dm_control Physics object.
        anchor_positions: Reference positions for distance-based coloring.
            If None, uses current positions.
        max_distance: Maximum distance for colormap normalization.
        colormap_name: Matplotlib colormap name for distance coloring.
        prev_data_dict: Previous frame's data for quaternion sign consistency.
        use_original_rgba: If True, use model's original colors. If False,
            use distance-based colormap.

    Returns:
        Dictionary with keys:
        - link/geom_type: List of geom type integers
        - link/geom_size: List of (3,) size tuples
        - link/rgba: List of (4,) RGBA tuples
        - link/pos: List of (3,) position tuples
        - link/quat_wxyz: List of (4,) quaternion tuples (wxyz order)
    """
    data_dict: dict[str, list] = {
        "link/geom_type": [],
        "link/geom_size": [],
        "link/rgba": [],
        "link/pos": [],
        "link/quat_wxyz": [],
    }

    # Get current geom positions
    positions = np.array(
        [
            np.array(physics.data.geom_xpos[geom_id])
            for geom_id in range(physics.model.ngeom)
        ]
    )

    # Handle anchor positions for distance coloring
    if anchor_positions is None:
        anchor_positions = positions.copy()
    else:
        anchor_positions = np.asarray(anchor_positions)
        if anchor_positions.shape[0] != positions.shape[0]:
            # Align shapes by copying available anchors
            aligned = positions.copy()
            overlap = min(anchor_positions.shape[0], positions.shape[0])
            aligned[:overlap] = anchor_positions[:overlap]
            anchor_positions = aligned

    # Compute distance-based colors
    distances = np.linalg.norm(positions - anchor_positions, axis=1)
    normalized = np.clip(distances / max_distance, 0.0, 1.0) ** 0.5
    colormap = plt.get_cmap(colormap_name)
    colors = colormap(normalized)
    colors[:, 3] = 1.0  # Full opacity

    geom_xmat = physics.data.geom_xmat[:]

    # Extract per-geom data
    for geom_id in range(physics.model.ngeom):
        data_dict["link/geom_type"].append(int(physics.model.geom_type[geom_id]))
        data_dict["link/geom_size"].append(
            [float(x) for x in physics.model.geom_size[geom_id]]
        )

        if use_original_rgba:
            data_dict["link/rgba"].append(
                tuple(float(x) for x in physics.model.geom_rgba[geom_id])
            )
        else:
            data_dict["link/rgba"].append(tuple(float(x) for x in colors[geom_id]))

        data_dict["link/pos"].append([float(x) for x in positions[geom_id]])

        # Convert rotation matrix to quaternion
        quat = quaternions.mat2quat(geom_xmat[geom_id])

        # Maintain quaternion sign consistency with previous frame
        if prev_data_dict is not None:
            prev_quat = prev_data_dict["link/quat_wxyz"][geom_id]
            if np.linalg.norm(quat - prev_quat, ord=1) > np.linalg.norm(
                -quat - prev_quat, ord=1
            ):
                quat = -quat

        data_dict["link/quat_wxyz"].append(tuple(float(x) for x in quat))

    return data_dict


class DatasetVisualizer:
    """Dataset visualizer that displays robots and target poses from batch data."""

    def __init__(
        self,
        dt: float = 0.1,
        gravcomp: bool = False,
        step_physics: bool = True,
        pickle_output_root: Optional[str] = None,
        disable_gui: bool = False,
    ):
        """
        Initialize the visualizer.

        Args:
            dt: Sleep time per simulation step in seconds
            gravcomp: Whether to enable gravity compensation
            step_physics: Whether to step physics simulation
            pickle_output_root: Optional base path for saving physics states.
                If set, states are saved to {pickle_output_root}/ep{episode_seed:02d}_hw{hardware_seed}.pkl
        """
        self.dt = dt
        self.current_physics = None
        self.current_viewer = None
        self.current_robot_hash = None
        self.gravcomp = gravcomp
        self.update_qpos_fn = None
        self.update_ctrl_fn = None
        self.step_physics = step_physics
        self.pickle_output_root = pickle_output_root
        self.n_end_effectors = 0
        self.disable_gui = disable_gui
        # Track link mappings (populated after robot creation)
        self.tracklinkid_to_mjgeomid: dict[int, int] = {}
        self.track_link_ids: list[int] = []
        # Collected physics states for pickle export
        self.collected_states: list[dict[str, list]] = []
        self.current_hardware_seed: Optional[int] = None
        self.current_episode_seed: Optional[int] = None

    def _extract_robot_data(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, np.ndarray]:
        """Extract robot hardware data from batch."""
        robot_data = {}

        # Extract hardware groups (actuator, joint, link)
        hardware_groups = ["actuator", "joint", "link"]

        for group in hardware_groups:
            for key, value in batch.items():
                if key.startswith(f"{group}/") and not key.endswith("/id"):
                    # Convert tensor to numpy and squeeze batch dimension
                    robot_data[key] = value.squeeze(0).numpy()

        return robot_data

    def _compute_robot_hash(self, robot_data: dict[str, np.ndarray]) -> str:
        """Compute a hash for robot data to detect changes."""
        # Concatenate all robot data arrays and compute hash
        data_str = ""
        for key in sorted(robot_data.keys()):
            data_str += f"{key}:{robot_data[key].tobytes().hex()}"
        return str(hash(data_str))

    def _create_robot_from_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> Optional[mjcf.Physics]:
        """Create a MuJoCo physics instance from batch robot data."""
        try:
            # Deserialize robot tokens
            masks = {k: v.squeeze(-1) for k, v in batch.items() if k.endswith("/mask")}
            data_dict = {}
            for k, v in batch.items():
                if k.endswith("/mask"):
                    continue
                data = v
                group = k.split("/")[0]
                mask_key = group + "/mask"
                if mask_key in batch:
                    data = data[~masks[mask_key]]
                data_dict[k] = data.numpy().reshape(-1, data.shape[-1])
            robot = deserialize(data_dict, include_states=True)

            # Detokenize to get MJCF model
            mjcf_model = detokenize(robot, gravcomp=self.gravcomp)[0]
            mjcf_model = post_process_mjcf_solver_parameters(mjcf_model)

            # Set up scene
            mjcf_model = set_up_default_scene(
                mjcf_model, add_plane=True, plane_z_pos=0.0
            )
            # Determine number of end effectors from target_pose/track_link/id
            if "target_pose/track_link/id" in batch:
                track_link_ids_in_batch = batch["target_pose/track_link/id"].squeeze(0)
                mask = batch.get("target_pose/mask")
                if mask is not None:
                    mask = mask.squeeze(0).squeeze(-1)
                    track_link_ids_in_batch = track_link_ids_in_batch[~mask]
                unique_track_link_ids = track_link_ids_in_batch.unique()
                self.n_end_effectors = len(unique_track_link_ids)
            else:
                # Fallback: try to infer from track_link_obs
                if "track_link_obs/track_link/id" in batch:
                    track_link_ids_in_batch = batch[
                        "track_link_obs/track_link/id"
                    ].squeeze(0)
                    mask = batch.get("track_link_obs/mask")
                    if mask is not None:
                        mask = mask.squeeze(0).squeeze(-1)
                        track_link_ids_in_batch = track_link_ids_in_batch[~mask]
                    unique_track_link_ids = track_link_ids_in_batch.unique()
                    self.n_end_effectors = len(unique_track_link_ids)
                else:
                    self.n_end_effectors = 1  # Default fallback

            logger.info(f"Detected {self.n_end_effectors} end effector(s)")

            # Create mocap bodies for each end effector (matching track_env.py naming)
            for i in range(self.n_end_effectors):
                mjcf_model = add_mocap_body_with_site(
                    mjcf_model, f"target_{i}", site_group=TARGET_SITE_GROUP
                )

            # Create physics
            physics = mjcf.Physics.from_mjcf_model(mjcf_model)
            if physics is not None:
                physics.reset()

            # Build track link to geom mapping (same logic as base_env.py and tokenizer.py)
            # The track_link_obs data is collected from geom positions, not site positions
            model = physics.model.ptr

            # Build link_id to geom_id mapping from geom names (geom_{link_id})
            linkid_to_mjgeomid = {}
            for mj_geomid in range(model.ngeom):
                geom_name = physics.model.id2name(mj_geomid, "geom")
                if geom_name is not None and geom_name.startswith("geom_"):
                    link_id = int(geom_name.split("geom_")[1])
                    linkid_to_mjgeomid[link_id] = mj_geomid

            # Build track_link_id to link_id mapping from the robot's link data
            # The robot has links with track_link_idx field - we need to find
            # which link_id corresponds to which track_link_id
            # We can get this from the batch data: link/track_link/id
            link_track_link_ids = batch["link/track_link/id"].squeeze(0).numpy()
            link_mask = batch.get("link/mask")
            if link_mask is not None:
                link_mask = link_mask.squeeze(0).squeeze(-1).numpy()
                link_track_link_ids = link_track_link_ids[~link_mask]

            # Build track_link_id to geom_id mapping
            self.tracklinkid_to_mjgeomid = {}
            for link_id, track_link_id in enumerate(link_track_link_ids.squeeze(-1)):
                if track_link_id != -1:
                    if link_id in linkid_to_mjgeomid:
                        self.tracklinkid_to_mjgeomid[int(track_link_id)] = (
                            linkid_to_mjgeomid[link_id]
                        )

            # Store sorted list of track link IDs
            self.track_link_ids = sorted(self.tracklinkid_to_mjgeomid.keys())

            def update_qpos_fn(
                data: mujoco.MjData,
                batch: dict[str, np.ndarray],
                time_idx: int,
            ):
                qpos_time_mask = np.logical_and(
                    (batch["dyna_joint_obs/time/id"] == time_idx).squeeze(-1),
                    ~batch["dyna_joint_obs/mask"].squeeze(-1),
                )
                qpos = batch["dyna_joint_obs/qpos"][qpos_time_mask].squeeze(-1)
                dyna_joint_ids = range(len(qpos))
                dyna_joint_mask = []
                joint_names = []
                for joint_id in dyna_joint_ids:
                    joint_name = f"dyna_joint_{joint_id}"
                    try:
                        jnt = physics.model.joint(joint_name)
                        joint_names.append(jnt.name)
                        dyna_joint_mask.append(True)
                    except (KeyError, ValueError):
                        dyna_joint_mask.append(False)

                mj_joints = [physics.model.joint(name) for name in joint_names]
                mj_joint_qpos_adr = [mj_joint.qposadr[0] for mj_joint in mj_joints]
                data.qpos[mj_joint_qpos_adr] = qpos[dyna_joint_mask]

                qvel = batch["dyna_joint_obs/qvel"][qpos_time_mask].squeeze(-1)
                mj_joint_qvel_adr = [mj_joint.dofadr[0] for mj_joint in mj_joints]
                data.qvel[mj_joint_qvel_adr] = qvel[dyna_joint_mask]
                if len(mj_joint_qpos_adr) < len(data.qpos):
                    other_qpos_adr = set(range(len(data.qpos))) - set(mj_joint_qpos_adr)
                    assert len(other_qpos_adr) == 7, (
                        "only expected at most one more free joint"
                    )
                    assert other_qpos_adr == set(range(7))
                    free_link_obs_time_mask = np.logical_and(
                        (batch["free_link_obs/time/id"] == time_idx).squeeze(-1),
                        ~batch["free_link_obs/mask"].squeeze(-1),
                    )

                    link_pos = batch["free_link_obs/pos"][
                        free_link_obs_time_mask
                    ].reshape(3)
                    link_rotmat = batch["free_link_obs/rotmat"][
                        free_link_obs_time_mask
                    ].reshape(3, 3)
                    link_quat = quaternions.mat2quat(link_rotmat)
                    data.qpos[0:3] = link_pos
                    data.qpos[3:7] = link_quat

            self.update_qpos_fn = update_qpos_fn

            return physics

        except Exception as e:
            raise e
            logger.error(f"Failed to create robot from batch: {e}")
            return None

    def _apply_state(
        self,
        physics: mjcf.Physics,
        batch: dict[str, torch.Tensor],
        time_idx: int,
    ):
        """Apply target pose and joint configurations from batch."""
        if physics is None:
            return

        try:
            data = physics.data.ptr
            batch_np = {k: v.squeeze(0).numpy() for k, v in batch.items()}

            # Set target poses for all end effectors
            # The batch has target_pose indexed by (time * n_end_effectors + track_link_id)
            # We need to filter by time_idx using target_pose/time/id
            target_time_ids = batch_np["target_pose/time/id"].squeeze(-1)
            target_track_link_ids = batch_np["target_pose/track_link/id"].squeeze(-1)

            # Filter entries for this time_idx
            # time_mask = target_time_ids == time_idx + 4 # HACK to visualize control results for policy, which looks ahead 4 steps
            time_mask = target_time_ids == time_idx

            for ee_idx in range(self.n_end_effectors):
                # Find the entry matching both time_idx and track_link_id=ee_idx
                entry_mask = np.logical_and(time_mask, target_track_link_ids == ee_idx)
                if not entry_mask.any():
                    logger.warning(
                        f"No target_pose entry found for time_idx={time_idx}, ee_idx={ee_idx}"
                    )
                    continue

                entry_idx = np.where(entry_mask)[0][0]
                target_pos = batch_np["target_pose/pos"][entry_idx]
                target_rotmat = batch_np["target_pose/rotmat"][entry_idx]
                target_quat = quaternions.mat2quat(target_rotmat.reshape(3, 3))

                # Set mocap position and orientation for this end effector
                data.mocap_pos[ee_idx] = target_pos
                data.mocap_quat[ee_idx] = target_quat

            # Set joint positions
            if self.update_qpos_fn is not None and (
                not self.step_physics or time_idx == 0
            ):
                self.update_qpos_fn(
                    data=data,
                    batch={k: v.squeeze(0).numpy() for k, v in batch.items()},
                    time_idx=time_idx,
                )

            # Set actuator controls
            ctrl_time_mask = torch.logical_and(
                (batch["ctrl/time/id"] == time_idx).squeeze(-1),
                ~batch["ctrl/mask"].squeeze(-1),
            )
            ctrl = batch["ctrl/target_qpos"][ctrl_time_mask].squeeze(-1).numpy()
            actuators = [
                physics.model.actuator(f"actuator_{act_id}")
                for act_id in range(len(ctrl))
            ]
            mj_actuator_id = [actuator.id for actuator in actuators]
            data.ctrl[mj_actuator_id] = ctrl

        except Exception as e:
            raise e
            logger.warning(f"Failed to apply target pose: {e}")

    def _verify_state(
        self, physics: mjcf.Physics, batch: dict[str, torch.Tensor], time_idx: int
    ):
        """Verify the state of the physics instance for all track links."""
        if physics is None:
            return

        if len(self.track_link_ids) == 0:
            return

        data = physics.data.ptr
        batch_np = {k: v.squeeze(0).numpy() for k, v in batch.items()}

        # Get track_link_obs data for this time_idx
        # The batch has track_link_obs indexed by (time * n_track_links + track_link_id)
        # We need to filter by time_idx using track_link_obs/time/id
        track_time_ids = batch_np["track_link_obs/time/id"].squeeze(-1)
        track_link_ids = batch_np["track_link_obs/track_link/id"].squeeze(-1)

        # Filter entries for this time_idx
        time_mask = track_time_ids == time_idx

        for track_link_id in self.track_link_ids:
            # Find the entry matching both time_idx and track_link_id
            entry_mask = np.logical_and(time_mask, track_link_ids == track_link_id)
            if not entry_mask.any():
                logger.warning(
                    f"No track_link_obs entry found for time_idx={time_idx}, "
                    f"track_link_id={track_link_id}"
                )
                continue

            entry_idx = np.where(entry_mask)[0][0]
            track_link_pos = batch_np["track_link_obs/pos"][entry_idx]
            track_link_rotmat = batch_np["track_link_obs/rotmat"][entry_idx]

            # Get the corresponding geom (track_link_obs is collected from geom positions)
            mj_geom_id = self.tracklinkid_to_mjgeomid[track_link_id]

            pos_err = np.linalg.norm(track_link_pos - data.geom_xpos[mj_geom_id])
            rotmat_err = np.linalg.norm(track_link_rotmat - data.geom_xmat[mj_geom_id])

            # assert pos_err < 1e-3, (
            #     f"Position mismatch for track_link_id={track_link_id}: "
            #     f"expected {track_link_pos}, got {data.geom_xpos[mj_geom_id]}, err={pos_err}"
            # )
            # assert rotmat_err < 1e-3, (
            #     f"Rotmat mismatch for track_link_id={track_link_id}: "
            #     f"expected {track_link_rotmat}, got {data.geom_xmat[mj_geom_id]}, err={rotmat_err}"
            # )

    def _setup_viewer(self, physics: mjcf.Physics):
        """Set up the MuJoCo viewer."""
        if self.disable_gui:
            return None
        if self.current_viewer is not None:
            self.current_viewer.close()

        self.current_viewer = viewer.launch_passive(
            model=physics.model.ptr, data=physics.data.ptr
        )

        # Configure viewer options
        render_opt(self.current_viewer.opt)
        self.current_viewer.opt.geomgroup[3] = 1
        self.current_viewer.opt.sitegroup[:] = 0

        self.current_viewer.opt.sitegroup[TARGET_SITE_GROUP] = 1
        self.current_viewer.opt.sitegroup[TRACKING_SITE_GROUP] = 1
        return self.current_viewer

    def visualize_batch(
        self, batch: dict[str, torch.Tensor], step_through_time: bool = True
    ):
        """
        Visualize a single batch.

        Args:
            batch: Batch data containing robot and rollout information
            step_through_time: Whether to step through time dimension in the batch
        """
        # Extract robot data and compute hash
        robot_data = self._extract_robot_data(batch)
        robot_hash = self._compute_robot_hash(robot_data)

        # Track seeds for pickle filename
        hardware_seed = int(batch["hardware_seed"].squeeze(0).item())
        episode_seed = int(batch["episode_seed"].squeeze(0).item())

        # Check if we need to create a new robot or save previous batch states
        if robot_hash != self.current_robot_hash:
            # Save states from previous batch before creating new robot
            if self.pickle_output_root and self.collected_states:
                self.save_states_to_pickle()

            logger.info("Updating robot visualization:")
            logging.info(f"Robot seed: {hardware_seed}")
            logging.info(f"Episode seed: {episode_seed}")

            # Reset state collection for new robot
            self._reset_state_collection()
            self.current_hardware_seed = hardware_seed
            self.current_episode_seed = episode_seed

            # Create new physics instance
            physics = self._create_robot_from_batch(batch)
            if physics is None:
                logger.error("Failed to create robot, skipping batch")
                return

            # Setup new viewer
            self.current_physics = physics
            self.current_robot_hash = robot_hash
            self._setup_viewer(physics)
        else:
            # Same robot, but potentially new episode - save previous states
            if (
                self.pickle_output_root
                and self.collected_states
                and episode_seed != self.current_episode_seed
            ):
                self.save_states_to_pickle()
                self._reset_state_collection()
                self.current_episode_seed = episode_seed

        physics = self.current_physics

        if physics is None:
            logger.error("No physics instance available")
            return

        # Get time dimension
        time_steps = 1
        for key, value in batch.items():
            if key.startswith("target_pose/"):
                if len(value.shape) >= 2:
                    time_steps = value.shape[1]
                    break

        if step_through_time and time_steps > 1:
            # Step through each time step
            for t in range(time_steps):
                self._apply_state(physics, batch, time_idx=t)
                physics.forward()
                if self.step_physics:
                    physics.step()

                # Collect physics state for pickle export
                self._collect_physics_state(physics)

                if (
                    self.current_viewer
                    and self.current_viewer.is_running()
                    and not self.disable_gui
                ):
                    self.current_viewer.sync()
                    time.sleep(self.dt)
                else:
                    logger.info("Viewer closed, stopping visualization")
                    return
                self._verify_state(physics, batch, time_idx=t)
        else:
            # Just show the first/only time step
            self._apply_state(physics, batch, time_idx=0)
            physics.forward()
            self._verify_state(physics, batch, time_idx=0)

            # Collect physics state for pickle export
            self._collect_physics_state(physics)

            if (
                self.current_viewer
                and self.current_viewer.is_running()
                and not self.disable_gui
            ):
                self.current_viewer.sync()
                time.sleep(self.dt)

    def _collect_physics_state(self, physics: mjcf.Physics) -> None:
        """Collect current physics state for pickle export.

        Args:
            physics: dm_control Physics object to extract state from.
        """
        if self.pickle_output_root is None:
            return

        prev_data_dict = self.collected_states[-1] if self.collected_states else None
        state = parse_physics_state(
            physics,
            anchor_positions=None,  # Use current positions as anchors
            max_distance=2.0,
            colormap_name="jet",
            prev_data_dict=prev_data_dict,
            use_original_rgba=True,  # Preserve original model colors
        )
        state["target_pose/pos"] = [
            (float(pos[0]), float(pos[1]), float(pos[2]))
            for pos in physics.data.mocap_pos
        ]
        state["target_pose/quat"] = [
            (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            for quat in physics.data.mocap_quat
        ]
        state["track_link_obs/pos"] = []
        state["track_link_obs/quat"] = []
        for track_link_id in self.track_link_ids:
            mj_geom_id = self.tracklinkid_to_mjgeomid[track_link_id]
            link_pos = physics.data.geom_xpos[mj_geom_id]
            state["track_link_obs/pos"].append(
                (float(link_pos[0]), float(link_pos[1]), float(link_pos[2]))
            )
            link_quat = quaternions.mat2quat(
                physics.data.geom_xmat[mj_geom_id].reshape(3, 3)
            )
            state["track_link_obs/quat"].append(
                (
                    float(link_quat[0]),
                    float(link_quat[1]),
                    float(link_quat[2]),
                    float(link_quat[3]),
                )
            )
        self.collected_states.append(state)

    def save_states_to_pickle(self) -> Optional[Path]:
        """Save collected physics states to a pickle file.

        Returns:
            Path to the saved pickle file, or None if no states were collected
            or pickle_output_root is not set.
        """
        if self.pickle_output_root is None:
            return None

        if not self.collected_states:
            logger.warning("No physics states collected, skipping pickle save.")
            return None

        if self.current_hardware_seed is None or self.current_episode_seed is None:
            logger.warning("Hardware or episode seed not set, skipping pickle save.")
            return None

        # Build output path with seeds
        output_path = Path(
            f"{self.pickle_output_root}/ep{self.current_episode_seed:02d}_hw{self.current_hardware_seed}.pkl"
        )

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(self.collected_states, f)

        logger.info(
            f"Saved {len(self.collected_states)} physics states to {output_path}"
        )
        return output_path

    def _reset_state_collection(self) -> None:
        """Reset the collected states for a new batch."""
        self.collected_states = []

    def close(self):
        """Close the visualizer and clean up resources."""
        if self.current_viewer is not None and not self.disable_gui:
            self.current_viewer.close()
            self.current_viewer = None
        self.current_physics = None
        self.current_robot_hash = None
        self.tracklinkid_to_mjgeomid = {}
        self.track_link_ids = []
        self.collected_states = []
        self.current_hardware_seed = None
        self.current_episode_seed = None


@hydra.main(
    config_path="../config",
    config_name="visualize",
    version_base="1.3",
)
def main(cfg: DictConfig):
    """Main function to run the dataset visualizer."""
    logger.info("Starting dataset visualizer...")

    # Create dataset
    logger.info(f"Loading dataset from: {cfg.dataset.path}")
    dataset = hydra.utils.instantiate(
        cfg.dataset,
        group_seq_lens=cfg.seq_len,
    )

    # Create data loader
    dataloader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=dataset,
    )

    # Create visualizer
    pickle_output_root = cfg.get("pickle_output_root", None)
    visualizer = DatasetVisualizer(
        dt=cfg.visualization.dt,
        gravcomp=cfg.gravcomp,
        step_physics=cfg.visualization.step_physics,
        pickle_output_root=pickle_output_root,
        disable_gui=cfg.visualization.disable_gui,
    )

    try:
        logger.info(f"Dataset contains {len(dataset)} samples")

        max_batches = cfg.visualization.get("max_batches", None)

        batch_process_fn = hydra.utils.instantiate(cfg.batch_process_fn)

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                logger.info(f"Reached maximum batch limit ({max_batches})")
                break
            torch.manual_seed(0)
            batch = batch_process_fn(batch)
            # Visualize the batch
            visualizer.visualize_batch(
                batch, step_through_time=cfg.visualization.step_through_time
            )

            # Check if viewer is still running
            if visualizer.current_viewer and not visualizer.current_viewer.is_running():
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(str(e))
    finally:
        # Save any remaining collected states
        if visualizer.pickle_output_root and visualizer.collected_states:
            visualizer.save_states_to_pickle()

        # Clean up
        visualizer.close()
        logger.info("Visualization complete")


if __name__ == "__main__":
    main()
