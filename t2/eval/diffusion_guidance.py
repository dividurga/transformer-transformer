"""Utilities for defining objective-based diffusion guidance."""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, ListConfig, OmegaConf

from t2.data.dataset import TimestepSampler
from t2.eval.utils import traj2batch
from t2.train.augment import AddPositionId

if TYPE_CHECKING:
    from t2.model.t2 import DecoderBundle


class RewardFn:
    """Base class for rewards that operate on decoded diffusion samples."""

    def __call__(
        self, decoded: dict[str, torch.Tensor]
    ) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class MinimizeNumActuators(RewardFn):
    """Encourages solutions with low number of actuators."""

    def __init__(
        self,
        weight: float,
        target_num_actuators: int,
        power: int,
        multiply_by_timesteps: bool,
    ):
        self.weight = weight
        self.target_num_actuators = target_num_actuators
        self.power = power
        self.multiply_by_timesteps = multiply_by_timesteps

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        actuator_mask = batch["actuator/mask"]  # (num_seeds, max_num_actuators)
        num_actuators = (~actuator_mask).sum(dim=1)
        actuator_cost = (
            torch.clip(
                (num_actuators - self.target_num_actuators),
                min=0,
            )
            ** self.power
        ).float()  # penalize if num_actuators is more than target_num_actuators
        if self.multiply_by_timesteps:
            timesteps = (
                batch["actuator_obs/time/id"]
                .reshape(actuator_mask.shape[0], -1, actuator_mask.shape[1])
                .shape[1]
            )
            actuator_cost = actuator_cost * timesteps
        return -actuator_cost * self.weight


class MinimizeWeight(RewardFn):
    """Encourages solutions with low weight."""

    def __init__(
        self,
        weight: float,
        power: int,
        multiply_by_timesteps: bool,
        target_weight: float,
    ):
        self.weight = weight
        self.power = power
        self.multiply_by_timesteps = multiply_by_timesteps
        self.target_weight = target_weight

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        link_mask = batch["link/mask"].reshape(batch["link/mask"].shape[:2])
        link_mass = batch["link/mass"].reshape(batch["link/mass"].shape[:2])
        total_weight = torch.where(link_mask, 0.0, link_mass).sum(dim=1)
        weight_cost = (
            torch.clip(total_weight - self.target_weight, min=0.0) ** self.power
        )
        if self.multiply_by_timesteps:
            actuator_mask = batch["actuator/mask"]  # (num_seeds, max_num_actuators)
            timesteps = (
                batch["actuator_obs/time/id"]
                .reshape(actuator_mask.shape[0], -1, actuator_mask.shape[1])
                .shape[1]
            )
            weight_cost = weight_cost * timesteps
        return -weight_cost * self.weight


class MinimizeSize(RewardFn):
    """Encourages solutions with low size."""

    def __init__(
        self,
        weight: float,
        target_dimension: float,
        power: int,
        multiply_by_timesteps: bool,
    ):
        self.weight = weight
        self.target_dimension = target_dimension
        self.power = power
        self.multiply_by_timesteps = multiply_by_timesteps

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # heuristically define size of a link as the maximum dimension of its size
        link_mask = batch["link/mask"].reshape(batch["link/mask"].shape[:2])
        dimension = (
            (
                batch["link/geom_size"] * 2
            )  # all geom size are half sizes (radii/half-lengths)
            .reshape(*batch["link/geom_size"].shape[:2], 3)
            .max(dim=2)
            .values.reshape(batch["link/geom_size"].shape[:2])
        )
        total_size = torch.where(link_mask, 0.0, dimension).sum(dim=1).float()
        size_cost = (
            torch.clip(total_size - self.target_dimension, min=0.0) ** self.power
        )
        if self.multiply_by_timesteps:
            actuator_mask = batch["actuator/mask"]  # (num_seeds, max_num_actuators)
            timesteps = (
                batch["actuator_obs/time/id"]
                .reshape(actuator_mask.shape[0], -1, actuator_mask.shape[1])
                .shape[1]
            )
            size_cost = size_cost * timesteps
        return -size_cost * self.weight


class MinimizeTrackingError(RewardFn):
    """Encourages solutions with low tracking error.

    Supports both single and multiple tracking sites (e.g., bimanual robots).
    When multiple tracking sites are present, errors are computed per-site
    and aggregated (typically summed or averaged in subclasses).
    """

    def compute_tracking_err(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute position and orientation errors for all tracking sites.

        Args:
            batch: Dictionary containing:
                - track_link_obs/track_link/id: (b, t, 1) link IDs
                - track_link_obs/pos: (b, t, 3) current positions
                - track_link_obs/rotmat: (b, t, 9) current rotation matrices
                - track_link_obs/time/id: (b, t, 1) time indices
                - target_pose/pos: (b, t, 3) target positions
                - target_pose/rotmat: (b, t, 9) target rotation matrices
                - target_pose/track_link/id: (b, t, 1) target link IDs
                - target_pose/time/id: (b, t, 1) target time indices

        Returns:
            pos_err: (b, n_timesteps) position error per timestep (aggregated across links)
            orn_err: (b, n_timesteps) orientation error per timestep (aggregated across links)
        """
        link_id = batch["track_link_obs/track_link/id"]
        target_link_id = batch["target_pose/track_link/id"]
        target_link_time_id = batch["target_pose/time/id"]

        b = link_id.shape[0]

        # Get unique link IDs (sorted)
        unique_link_ids = torch.unique(link_id)

        # Get positions and rotations for tracked links
        pos = batch["track_link_obs/pos"].reshape(b, -1, 3)
        orn = batch["track_link_obs/rotmat"].reshape(b, -1, 9)
        link_time_id = batch["track_link_obs/time/id"].reshape(b, -1, 1)
        link_id_flat = link_id.reshape(b, -1, 1)

        # Get target positions and rotations
        target_pos = batch["target_pose/pos"]
        target_orn = batch["target_pose/rotmat"]

        # For each unique link, compute errors against corresponding targets
        # Group observations and targets by link ID and time
        all_pos_errs = []
        all_orn_errs = []

        for lid in unique_link_ids:
            # Get observations for this link
            obs_mask = (link_id_flat == lid).squeeze(-1)  # (b, n_obs)
            target_mask = (target_link_id == lid).squeeze(-1)  # (b, n_targets)

            # Extract positions and orientations for this link
            # We need to be careful about ordering by time
            link_pos = pos[obs_mask].reshape(b, -1, 3)  # (b, t_link, 3)
            link_orn = orn[obs_mask].reshape(b, -1, 9)  # (b, t_link, 9)
            link_time = link_time_id[obs_mask].reshape(b, -1, 1)  # (b, t_link, 1)

            link_target_pos = target_pos[target_mask].reshape(b, -1, 3)
            link_target_orn = target_orn[target_mask].reshape(b, -1, 9)
            link_target_time = target_link_time_id[target_mask].reshape(b, -1, 1)

            # Verify time alignment
            assert (link_time == link_target_time).all(), (
                f"Time mismatch for link {lid}: obs times {link_time} vs target times {link_target_time}"
            )

            # Compute position error
            pos_err = torch.linalg.norm(link_pos - link_target_pos, dim=-1)

            # Compute orientation error
            delta_orn = torch.transpose(
                link_orn.reshape(b, -1, 3, 3), -1, -2
            ) @ link_target_orn.reshape(b, -1, 3, 3)
            trace = delta_orn.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
            trace = torch.clamp(trace, min=-1 + 1e-8, max=3 - 1e-8)
            orn_err = torch.arccos((trace - 1) / 2)
            orn_err = orn_err % (2 * torch.pi)
            orn_err = torch.minimum(orn_err, 2 * torch.pi - orn_err)

            all_pos_errs.append(pos_err)
            all_orn_errs.append(orn_err)

        # Stack errors for all links: (n_links, b, t_per_link) -> aggregate
        # For backwards compatibility, sum across links (effectively treating
        # multiple end effectors as independent tracking tasks)
        pos_err = torch.stack(all_pos_errs, dim=2).sum(dim=2)  # (b, t)
        orn_err = torch.stack(all_orn_errs, dim=2).sum(dim=2)  # (b, t)

        return pos_err, orn_err


class ExpTrackingError(MinimizeTrackingError):
    """Encourages solutions with low tracking error and low actuator effort."""

    def __init__(
        self,
        pose_weight: float,
        pos_sigma: float,
        orn_sigma: float,
        power: int,
        torque_weight: float,
        velocity_weight: float,
    ):
        self.pose_weight = pose_weight
        self.pos_sigma = pos_sigma
        self.orn_sigma = orn_sigma
        self.power = power
        self.torque_weight = torque_weight
        self.velocity_weight = velocity_weight

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pos_err, orn_err = self.compute_tracking_err(batch)
        reward = (
            torch.exp(
                -(pos_err**self.power) / self.pos_sigma
                - (orn_err**self.power) / self.orn_sigma
            )
            * self.pose_weight
        )
        return reward.sum(dim=1)  # reduce along time


class RegularizedExpTrackingError(ExpTrackingError):
    def __init__(
        self,
        pose_weight: float,
        pos_sigma: float,
        orn_sigma: float,
        power: int,
        torque_weight: float,
        velocity_weight: float,
        leaky_clip_slope: float = 0.0,
    ):
        self.pose_weight = pose_weight
        self.pos_sigma = pos_sigma
        self.orn_sigma = orn_sigma
        self.power = power
        self.torque_weight = torque_weight
        self.velocity_weight = velocity_weight
        self.leaky_clip_slope = leaky_clip_slope

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pos_err, orn_err = self.compute_tracking_err(batch)
        tracking_reward = (
            torch.exp(
                -(pos_err**self.power) / self.pos_sigma
                - (orn_err**self.power) / self.orn_sigma
            )
            * self.pose_weight
        )  # (b, t)

        actuator_mask = batch["actuator/mask"]  # (num_seeds, max_num_actuators)
        num_seeds, max_num_actuators = actuator_mask.shape[:2]

        actuator_force = batch["actuator_obs/force"].reshape(
            num_seeds, -1, max_num_actuators
        )
        num_timesteps = actuator_force.shape[1]

        actuator_mask = actuator_mask.reshape(num_seeds, 1, max_num_actuators).repeat(
            1, num_timesteps, 1
        )

        actuator_velocity = batch["actuator_obs/velocity"].reshape(
            num_seeds,
            -1,
            max_num_actuators,
        )
        actuator_effort = torch.where(
            actuator_mask, 0.0, torch.square(actuator_force.abs())
        ).sum(dim=2)
        actuator_speed = torch.where(
            actuator_mask, 0.0, torch.square(actuator_velocity.abs())
        ).sum(dim=2)

        assert actuator_effort.shape == tracking_reward.shape
        assert actuator_speed.shape == tracking_reward.shape

        reward = (
            tracking_reward
            - self.torque_weight * actuator_effort
            - self.velocity_weight * actuator_speed
        )
        reward = torch.nn.functional.leaky_relu(
            input=reward, negative_slope=self.leaky_clip_slope
        )  # this allows gradients to pass through when the reward is negative

        return reward.sum(dim=1)


class JointDiff(RewardFn):
    """Penalizes the amount of joint movement between consecutive timesteps.

    Computes the delta of qpos between timesteps, averages over the number of
    (non-masked) joints, and sums across timesteps.
    """

    def __init__(
        self,
        weight: float,
        joint_indices: Sequence[int] | None = None,
    ):
        self.weight = weight
        self.joint_indices = list(joint_indices) if joint_indices is not None else None

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Get the mask and qpos
        obs_mask = batch["dyna_joint_obs/mask"]  # (batch, time * num_joints, 1)
        qpos = batch["dyna_joint_obs/qpos"]  # (batch, time * num_joints, 1)

        num_seeds = obs_mask.shape[0]

        # Determine number of joints from dyna_joint_obs/dyna_joint/id
        joint_ids = batch[
            "dyna_joint_obs/dyna_joint/id"
        ]  # (batch, time * num_joints, 1)
        max_num_joints = int(joint_ids.max().item()) + 1

        # Reshape to (batch, time, num_joints)
        qpos = qpos.reshape(num_seeds, -1, max_num_joints)
        obs_mask = obs_mask.reshape(num_seeds, -1, max_num_joints)

        # Optionally filter to specific joint indices
        if self.joint_indices is not None:
            joint_idx_tensor = torch.tensor(
                self.joint_indices, device=qpos.device, dtype=torch.long
            )
            qpos = qpos[:, :, joint_idx_tensor]
            obs_mask = obs_mask[:, :, joint_idx_tensor]

        # Compute delta between consecutive timesteps
        # qpos_diff has shape (batch, time-1, num_joints)
        qpos_diff = qpos[:, 1:, :] - qpos[:, :-1, :]

        # Create mask for valid diffs (both timesteps must be valid)
        # mask is True for padding, so valid when False
        valid_mask = ~obs_mask[:, 1:, :] & ~obs_mask[:, :-1, :]

        # Compute absolute diff and mask invalid entries
        abs_diff = torch.where(valid_mask, qpos_diff.abs(), torch.zeros_like(qpos_diff))

        # Count valid joints per timestep for averaging
        valid_count = valid_mask.float().sum(dim=2)  # (batch, time-1)
        valid_count = torch.clamp(valid_count, min=1.0)  # avoid division by zero

        # Sum over joints, divide by valid count (average per timestep)
        diff_per_timestep = abs_diff.sum(dim=2) / valid_count  # (batch, time-1)

        # Sum across timesteps
        total_diff = diff_per_timestep.sum(dim=1)  # (batch,)

        return -total_diff * self.weight  # negate to minimize


class JointVel(RewardFn):
    """Penalizes the norm of joint velocities.

    Computes the norm of qvel, averages over the number of (non-masked) joints,
    and sums across timesteps.
    """

    def __init__(
        self,
        weight: float,
        max_joint_vel: float,
        joint_indices: Sequence[int] | None = None,
    ):
        self.weight = weight
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        # clipping at this value side steps the occasional simulation instability
        self.max_joint_vel = max_joint_vel

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Get the mask and qvel
        obs_mask = batch["dyna_joint_obs/mask"]  # (batch, time * num_joints, 1)
        qvel = batch["dyna_joint_obs/qvel"]  # (batch, time * num_joints, 1)

        num_seeds = obs_mask.shape[0]

        # Determine number of joints from dyna_joint_obs/dyna_joint/id
        joint_ids = batch[
            "dyna_joint_obs/dyna_joint/id"
        ]  # (batch, time * num_joints, 1)
        max_num_joints = int(joint_ids.max().item()) + 1

        # Reshape to (batch, time, num_joints)
        qvel = qvel.reshape(num_seeds, -1, max_num_joints)
        obs_mask = obs_mask.reshape(num_seeds, -1, max_num_joints)

        # Optionally filter to specific joint indices
        if self.joint_indices is not None:
            joint_idx_tensor = torch.tensor(
                self.joint_indices, device=qvel.device, dtype=torch.long
            )
            qvel = qvel[:, :, joint_idx_tensor]
            obs_mask = obs_mask[:, :, joint_idx_tensor]

        # valid_mask is True where data is valid (mask is True for padding)
        valid_mask = ~obs_mask

        # Compute squared velocity and mask invalid entries
        vel_squared = torch.where(
            valid_mask,
            torch.square(
                qvel.clip(
                    min=-self.max_joint_vel,
                    max=self.max_joint_vel,
                )
            ),
            torch.zeros_like(qvel),
        )

        # Count valid joints per timestep for averaging
        valid_count = valid_mask.float().sum(dim=2)  # (batch, time)
        valid_count = torch.clamp(valid_count, min=1.0)  # avoid division by zero

        # Sum over joints, divide by valid count (average per timestep)
        vel_per_timestep = vel_squared.sum(dim=2) / valid_count  # (batch, time)

        # Sum across timesteps
        total_vel = vel_per_timestep.sum(dim=1)  # (batch,)

        return -total_vel * self.weight  # negate to minimize


class JointRangeCenter(RewardFn):
    """Penalizes joints for deviating from the middle of their range.

    For each joint, computes the deviation from the center of [range_min, range_max],
    averages over the number of (non-masked) joints, and sums across timesteps.
    """

    def __init__(
        self,
        weight: float,
        joint_indices: Sequence[int] | None = None,
    ):
        self.weight = weight
        self.joint_indices = list(joint_indices) if joint_indices is not None else None

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Get the mask and qpos
        obs_mask = batch["dyna_joint_obs/mask"]  # (batch, time * num_joints, 1)
        qpos = batch["dyna_joint_obs/qpos"]  # (batch, time * num_joints, 1)

        # Get joint range - shape is (batch, num_joints * 2, 2) since stored twice per joint
        joint_range = batch["dyna_joint/range"]  # (batch, num_joints * 2, 2)

        num_seeds = obs_mask.shape[0]

        # Determine number of joints from dyna_joint_obs/dyna_joint/id
        joint_ids = batch[
            "dyna_joint_obs/dyna_joint/id"
        ]  # (batch, time * num_joints, 1)
        max_num_joints = int(joint_ids.max().item()) + 1

        # Extract range for each joint (take every other entry since range is doubled)
        # joint_range has shape (batch, num_joints * 2, 2), we want (batch, num_joints, 2)
        joint_range = joint_range[:, ::2, :]  # (batch, num_joints, 2)
        # Only use the first max_num_joints (in case dyna_joint has more due to padding)
        joint_range = joint_range[:, :max_num_joints, :]

        # Compute the center and half-range for each joint
        range_min = joint_range[:, :, 0]  # (batch, num_joints)
        range_max = joint_range[:, :, 1]  # (batch, num_joints)
        range_center = (range_min + range_max) / 2  # (batch, num_joints)
        half_range = (range_max - range_min) / 2  # (batch, num_joints)

        # Reshape qpos to (batch, time, num_joints)
        qpos = qpos.reshape(num_seeds, -1, max_num_joints)
        obs_mask = obs_mask.reshape(num_seeds, -1, max_num_joints)

        # Optionally filter to specific joint indices
        if self.joint_indices is not None:
            joint_idx_tensor = torch.tensor(
                self.joint_indices, device=qpos.device, dtype=torch.long
            )
            qpos = qpos[:, :, joint_idx_tensor]
            obs_mask = obs_mask[:, :, joint_idx_tensor]
            range_center = range_center[:, joint_idx_tensor]
            half_range = half_range[:, joint_idx_tensor]

        # Expand range_center and half_range to match qpos shape (batch, time, num_joints)
        range_center = range_center[:, None, :]  # (batch, 1, num_joints)
        half_range = half_range[:, None, :]  # (batch, 1, num_joints)

        # Identify joints with meaningful range (skip joints with zero/near-zero range)
        # These are likely fixed joints or joints that can't move
        min_half_range = 0.01  # ~0.6 degrees for revolute joints
        has_range = half_range > min_half_range  # (batch, 1, num_joints)

        # Compute normalized deviation from center (0 at center, 1 at limits)
        # Use safe division - set to 0 for joints without meaningful range
        safe_half_range = torch.where(
            has_range, half_range, torch.ones_like(half_range)
        )
        deviation = torch.abs(qpos - range_center) / safe_half_range
        deviation = torch.where(has_range, deviation, torch.zeros_like(deviation))

        # valid_mask is True where data is valid (mask is True for padding)
        # Also exclude joints without meaningful range
        valid_mask = ~obs_mask & has_range

        # Mask invalid entries
        deviation = torch.where(valid_mask, deviation, torch.zeros_like(deviation))

        # Count valid joints per timestep for averaging
        valid_count = valid_mask.float().sum(dim=2)  # (batch, time)
        valid_count = torch.clamp(valid_count, min=1.0)  # avoid division by zero

        # Sum over joints, divide by valid count (average per timestep)
        dev_per_timestep = deviation.sum(dim=2) / valid_count  # (batch, time)

        # Sum across timesteps
        total_dev = dev_per_timestep.sum(dim=1)  # (batch,)

        return -total_dev * self.weight  # negate to minimize


class ObjectiveGuidedDiffusion:
    """Callable that produces gradients for diffusion guidance."""

    def __init__(
        self,
        reward_fns: dict[str, RewardFn],
        guidance_terms: Sequence[str],
        clip_nan: bool,
        scale: Any,
    ):
        self.reward_fns = reward_fns
        self.guidance_terms = list(guidance_terms)
        self.scale = self._resolve_scale(scale)
        self.clip_nan = clip_nan

    @staticmethod
    def _resolve_scale(scale: Any) -> Any:
        if isinstance(scale, (DictConfig, ListConfig)):
            return OmegaConf.to_container(scale, resolve=True)
        return scale

    def _scale_tensor(self, term: str, grad: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        if isinstance(scale, Mapping):
            value = scale.get(term, 1.0)
        else:
            value = scale
        if value is None:
            value = 1.0
        if isinstance(value, torch.Tensor):
            return value.to(device=grad.device, dtype=grad.dtype)
        return grad.new_tensor(value)

    def __call__(
        self,
        x_t_plus_one: dict[str, torch.Tensor],
        x_t: dict[str, torch.nn.Parameter | torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for param in x_t.values():
            if param.grad is not None:
                param.grad.zero_()

        reward_terms = {
            k: reward_fn(x_t_plus_one).mean()
            for k, reward_fn in self.reward_fns.items()
        }
        # NOTE we're maximizing the reward, so we need to negate the value to get the loss
        loss = -sum(reward_terms.values())
        loss.backward()

        guidance = {}
        for term in self.guidance_terms:
            if term not in x_t:
                continue
            grad = x_t[term].grad
            assert grad is not None
            if self.clip_nan:
                grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
            guidance[term] = grad * self._scale_tensor(term, grad)
        return guidance


class HardwareOptimizer:
    """
    A zeroth order optimizer that generates hardware designs for given trajectories.
    """

    def __init__(
        self,
        hardware_generator: (
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
            | "DecoderBundle"
        ),
        reward_fns: dict[str, RewardFn],
        num_seeds: int,
        batch_size: int,
        # Parameters for internal data processing
        seq_len_cfg: dict[str, int],
        timestep_sampler: TimestepSampler,
        center_traj: bool,
        add_pos_id: AddPositionId,
        device: torch.device,
        max_traj_len: int,
        output_cache_dir: str | None = None,
    ):
        self.hardware_generator = hardware_generator
        self.num_seeds = num_seeds
        self.batch_size = batch_size
        self.reward_fns = reward_fns
        self.seq_len_cfg = seq_len_cfg
        self.timestep_sampler = timestep_sampler
        self.center_traj = center_traj
        self.add_pos_id = add_pos_id
        self.device = device
        self.output_cache_dir = output_cache_dir
        self._ran_warmup = False
        self._cache_only_mode = False
        self._model_unloaded = False
        self.max_traj_len = max_traj_len
        if output_cache_dir is not None:
            os.makedirs(output_cache_dir, exist_ok=True)
            logging.info(
                f"HardwareOptimizer cache enabled at {output_cache_dir}. "
                "Model may not run if all results are cached."
            )

    def unload_model(self) -> None:
        """Unload the model from GPU to free memory.

        This method should be called after all model inference is complete
        (e.g., after pre-computing all results in run_model_upfront mode).
        The hardware_generator must be a DecoderBundle for this to work.

        After calling this method, the optimizer enters cache-only mode
        and will error if called on inputs that aren't cached.
        """
        from t2.model.t2 import DecoderBundle

        if not isinstance(self.hardware_generator, DecoderBundle):
            logging.warning(
                "Cannot unload model: hardware_generator is not a DecoderBundle. "
                "The model will remain on GPU."
            )
            return

        logging.info("Unloading model from GPU...")
        self.hardware_generator.unload_model()
        self._model_unloaded = True
        self._cache_only_mode = True
        # Additional gc.collect() to clean up any references held by the optimizer
        gc.collect()
        logging.info(
            "Model unloaded. Optimizer is now in cache-only mode - "
            "any cache miss will raise an error."
        )

    def set_cache_only_mode(self, enabled: bool) -> None:
        """Enable or disable cache-only mode.

        In cache-only mode, the optimizer will raise an error if called on
        inputs that are not already cached, rather than running the model.

        Args:
            enabled: Whether to enable cache-only mode.
        """
        self._cache_only_mode = enabled
        if enabled:
            logging.info("Cache-only mode enabled. Any cache miss will raise an error.")
        else:
            if self._model_unloaded:
                logging.warning(
                    "Disabling cache-only mode but model is unloaded. "
                    "Model inference will fail if cache miss occurs."
                )
            logging.info("Cache-only mode disabled.")

    def _prepare_batch(
        self, trajs: list[NDArray[np.float32]]
    ) -> dict[str, torch.Tensor]:
        """Convert trajectories to a batched tensor dict ready for the hardware generator.

        Args:
            trajs: List of trajectory arrays to optimize for.

        Returns:
            Batched tensor dict with position IDs added, moved to device.
        """
        batches = []
        for traj in trajs:
            if traj.shape[0] > self.max_traj_len:
                logging.warning(
                    f"Truncating trajectory from {traj.shape[0]} to {self.max_traj_len} timesteps"
                )
                traj = traj[: self.max_traj_len]
            batch = traj2batch(
                traj=traj,
                seq_lens=self.seq_len_cfg,
                sampler=self.timestep_sampler,
                center_traj=self.center_traj,
            )
            batches.append(batch)

        # Concatenate along batch dimension (each traj2batch returns batch_size=1)
        combined_batch = {}
        for key in batches[0].keys():
            combined_batch[key] = torch.cat([b[key] for b in batches], dim=0)

        # Move to device and add position IDs
        combined_batch = {k: v.to(self.device) for k, v in combined_batch.items()}
        combined_batch = self.add_pos_id(combined_batch)
        return combined_batch

    def _compute_cache_key(self, trajs: list[NDArray[np.float32]], seed: int) -> str:
        """Compute a unique cache key from trajectories and seed.

        Args:
            trajs: List of trajectory arrays.
            seed: Random seed.

        Returns:
            SHA256 hash string as cache key.
        """
        hasher = hashlib.sha256()
        for traj in trajs:
            hasher.update(traj.tobytes())
        hasher.update(seed.to_bytes(8, byteorder="little", signed=True))
        return hasher.hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file.

        Args:
            cache_key: The cache key (hash).

        Returns:
            Full path to the cache file.
        """
        assert self.output_cache_dir is not None
        return os.path.join(self.output_cache_dir, f"{cache_key}.pt")

    def _load_from_cache(
        self, cache_key: str
    ) -> tuple[dict[str, torch.Tensor], float, float] | None:
        """Try to load results from cache.

        Args:
            cache_key: The cache key to look up.

        Returns:
            Tuple of (decoded dict, predicted_value, optimize_time) if found, None otherwise.
        """
        if self.output_cache_dir is None:
            return None
        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None
        try:
            cached = torch.load(cache_path, map_location=self.device, weights_only=True)
            logging.info(f"Cache hit: {cache_key[:16]}...")
            return (
                cached["decoded"],
                cached["predicted_value"],
                cached["optimize_time"],
            )
        except Exception as e:
            logging.warning(f"Failed to load cache {cache_path}: {e}")
            return None

    def _save_to_cache(
        self,
        cache_key: str,
        decoded: dict[str, torch.Tensor],
        predicted_value: float,
        optimize_time: float,
    ) -> None:
        """Save results to cache.

        Args:
            cache_key: The cache key.
            decoded: The decoded hardware dict.
            predicted_value: The predicted reward value.
            optimize_time: Time taken for optimization.
        """
        if self.output_cache_dir is None:
            return
        cache_path = self._get_cache_path(cache_key)
        try:
            torch.save(
                {
                    "decoded": decoded,
                    "predicted_value": predicted_value,
                    "optimize_time": optimize_time,
                },
                cache_path,
            )
            logging.debug(f"Saved to cache: {cache_key[:16]}...")
        except Exception as e:
            logging.warning(f"Failed to save cache {cache_path}: {e}")

    def _run_optimization_core(
        self, trajs: list[NDArray[np.float32]], seed: int
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Core optimization logic without caching or timing.

        Args:
            trajs: List of trajectory arrays to optimize for.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (best hardware dict, highest reward value).
        """
        assert self.num_seeds % self.batch_size == 0
        num_iters = int(self.num_seeds // self.batch_size)
        highest_value = -float("inf")
        best_decoded = None

        # Prepare batch from trajectories
        batch = self._prepare_batch(trajs)
        num_trajs = len(trajs)

        # Repeat batch for batch_size samples per trajectory
        batch = {
            k: v.repeat(self.batch_size, *([1] * (v.ndim - 1)))
            for k, v in batch.items()
        }

        rs = np.random.RandomState(seed)
        for _ in range(num_iters):
            decoded = self.run_hardware_generator(
                batch, seed=rs.randint(0, np.iinfo(np.int32).max)
            )
            value = sum(reward_fn(decoded) for reward_fn in self.reward_fns.values())
            assert value.ndim == 1 and value.shape[0] == self.batch_size * num_trajs
            value = value.reshape(self.batch_size, num_trajs)
            # set the average value over all trajectories to each seed
            value[:] = value.mean(dim=1, keepdim=True)
            value = value.reshape(self.batch_size * num_trajs)
            best_seed = value.argmax(dim=0)
            curr_highest_value = value[best_seed].item()
            if curr_highest_value > highest_value:
                highest_value = curr_highest_value
                best_decoded = {k: v[best_seed] for k, v in decoded.items()}
        assert best_decoded is not None
        return best_decoded, highest_value

    def optimize(
        self, trajs: list[NDArray[np.float32]], seed: int = 0
    ) -> tuple[dict[str, torch.Tensor], float, float]:
        """Optimize hardware for the given trajectories.

        Args:
            trajs: List of trajectory arrays to optimize for.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (best hardware dict, highest reward value, optimization time in seconds).

        Raises:
            RuntimeError: If in cache-only mode and the result is not cached.
        """
        # Check cache first
        cache_key = self._compute_cache_key(trajs, seed)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # If in cache-only mode and we get here, it's a cache miss - raise error
        if self._cache_only_mode:
            raise RuntimeError(
                f"Cache miss in cache-only mode for cache_key={cache_key[:16]}... "
                f"(seed={seed}). The model has been unloaded and cannot run inference. "
                "Ensure all inputs were pre-computed before enabling cache-only mode."
            )

        # Run warmup if not already done (for torch.compile to finish)
        # Warmup is not timed and not cached
        if not self._ran_warmup:
            logging.info("Running warmup optimization (not timed, not cached)...")
            self._run_optimization_core(trajs, seed)
            self._ran_warmup = True
            logging.info("Warmup complete.")

        # Run the actual optimization with timing
        start_time = time.time()
        best_decoded, highest_value = self._run_optimization_core(trajs, seed)
        optimize_time = float(time.time() - start_time)

        # Save to cache
        self._save_to_cache(cache_key, best_decoded, highest_value, optimize_time)

        return best_decoded, highest_value, optimize_time

    def run_hardware_generator(
        self, batch: dict[str, torch.Tensor], seed: int
    ) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            decoded = self.hardware_generator(batch, seed=seed)  # type: ignore
        return {**batch, **decoded}


class ObjectiveGuidedDiffusionOptimizer(HardwareOptimizer):
    """
    A zeroth order optimizer that uses a first-order objective-guided diffusion process
    to optimize hardware parameters.
    """

    def __init__(
        self,
        guidance: ObjectiveGuidedDiffusion,
        output_cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(output_cache_dir=output_cache_dir, **kwargs)
        self.guidance = guidance
        guidance_rewards = set(self.guidance.reward_fns.keys())
        actual_rewards = set(self.reward_fns.keys())
        assert guidance_rewards == actual_rewards, (
            f"Guidance rewards {guidance_rewards} do not match actual rewards {actual_rewards}"
        )

    def run_hardware_generator(
        self, batch: dict[str, torch.Tensor], seed: int
    ) -> dict[str, torch.Tensor]:
        decoded = self.hardware_generator(  # type: ignore
            batch, seed=seed, guidance_fn=self.guidance
        )
        return {**batch, **decoded}
