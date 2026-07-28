import torch

from t2.robotok.token import GROUND_LINK_ID
from t2.utils.misc import get_batch_device, get_batch_size

# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# `_axis_angle_rotation` and `euler_angles_to_matrix` are copied directly from
# PyTorch3D (https://github.com/facebookresearch/pytorch3d), specifically
# pytorch3d/transforms/rotation_conversions.py, because installing pytorch3d
# takes so long :p . They are used under the BSD-3-Clause license reproduced in
# LICENSES/BSD-3-Clause-pytorch3d.txt


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def compose(pos: torch.Tensor, rotmat: torch.Tensor) -> torch.Tensor:
    shape = pos.shape
    T = torch.eye(4, device=pos.device).expand(*shape[:-1], 4, 4).clone()
    T[..., :3, :3] = rotmat.reshape(*shape[:-1], 3, 3)
    T[..., :3, 3] = pos.reshape(*shape[:-1], 3)
    return T


def decompose(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos = T[..., :3, 3]
    rotmat = T[..., :3, :3]
    return pos, rotmat


class TransformAugmentation:
    def __init__(
        self,
        pos_aug_magnitude: tuple[float, float, float],
        orn_aug_magnitude: tuple[float, float, float],
        affected_pose_fields: list[str],
    ):
        self.pos_aug_magnitude = torch.tensor(pos_aug_magnitude)
        self.orn_aug_magnitude = torch.tensor(orn_aug_magnitude)
        # these fields are stored here so the normalization range
        # can be updated according to the pos augmentation magnitude
        # meanwhile, orn magnitude doesn't need to be updated since
        # rotation matrices are always between [-1, 1]
        self.affected_pose_fields = affected_pose_fields

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if (self.pos_aug_magnitude == 0.0).all() and (
            self.orn_aug_magnitude == 0.0
        ).all():
            return batch
        device = get_batch_device(batch)
        b = get_batch_size(batch)

        self.pos_aug_magnitude = self.pos_aug_magnitude.to(device)
        self.orn_aug_magnitude = self.orn_aug_magnitude.to(device)

        pos_aug = (torch.rand(b, 3, device=device) - 0.5) * self.pos_aug_magnitude * 2
        euler_aug = (torch.rand(b, 3, device=device) - 0.5) * self.orn_aug_magnitude * 2
        T = torch.eye(4, device=device).expand(b, 4, 4).clone()
        T[:, :3, :3] = euler_angles_to_matrix(euler_aug, "XYZ")
        T[:, :3, 3] = pos_aug

        # step 1: augment pose trajectories
        for group in self.affected_pose_fields:
            pose = compose(batch[f"{group}/pos"], batch[f"{group}/rotmat"])
            pose = T[:, None] @ pose
            mask = batch[f"{group}/mask"].squeeze(dim=-1)
            pos_shape = batch[f"{group}/pos"].shape
            rot_shape = batch[f"{group}/rotmat"].shape
            dtype = batch[f"{group}/pos"].dtype
            augmented_pos, augmented_rotmat = decompose(pose)
            batch[f"{group}/pos"][~mask], batch[f"{group}/rotmat"][~mask] = (
                augmented_pos.reshape(pos_shape)[~mask].to(dtype),
                augmented_rotmat.reshape(rot_shape)[~mask].to(dtype),
            )

        # step 2: augment hardware design for fixed based robots
        # where the joint is fixed and one of the links is the ground
        link_is_ground = batch["fixed_joint/link/id"] == GROUND_LINK_ID
        mask = batch["fixed_joint/mask"].squeeze(dim=-1)
        joint_should_augment = link_is_ground.squeeze(dim=-1) & ~mask

        pose = compose(batch["fixed_joint/pos"], batch["fixed_joint/rotmat"])
        T_inv = torch.linalg.inv(T)
        pose = pose @ T_inv[:, None]
        augmented_joint_pos, augmented_joint_rotmat = decompose(pose)
        pos_shape = batch["fixed_joint/pos"].shape
        rot_shape = batch["fixed_joint/rotmat"].shape
        dtype = batch["fixed_joint/pos"].dtype
        batch["fixed_joint/pos"][joint_should_augment] = augmented_joint_pos.reshape(
            pos_shape
        )[joint_should_augment].to(dtype)
        batch["fixed_joint/rotmat"][joint_should_augment] = (
            augmented_joint_rotmat.reshape(rot_shape)[joint_should_augment].to(dtype)
        )
        return batch


class AddPositionId:
    def __init__(self, pos_seq_lens: dict[str, tuple[int, int, int, int]]):
        # key, seq_len, seq_repeat, element_repeat
        self.pos_seq_lens = pos_seq_lens

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = get_batch_size(batch)
        device = get_batch_device(batch)
        for group, (
            seq_len,
            offset,
            seq_repeat,
            element_repeat,
        ) in self.pos_seq_lens.items():
            assert f"{group}/id" not in batch, f"{group}/id already exists"
            batch[f"{group}/id"] = (
                torch.arange(seq_len, device=device)[None, None, :, None]
                .repeat(batch_size, seq_repeat, 1, element_repeat)
                .reshape(batch_size, -1, 1)
            ).long() + offset
        return batch


class AddJointOrderAugmentation:
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = get_batch_size(batch)
        for prefix in ["dyna_joint", "fixed_joint"]:
            joint_link_ids = batch[f"{prefix}/link/id"]
            joint_link_ids = joint_link_ids.view(batch_size, -1, 2)
            delta_joint_link_ids = joint_link_ids[:, :, 0] - joint_link_ids[:, :, 1]
            higher_joint_link_ids = delta_joint_link_ids > 0  # (batch, joint_count)
            batch[f"{prefix}/order/id"] = (
                torch.stack(
                    [
                        higher_joint_link_ids,
                        ~higher_joint_link_ids,
                    ],
                    dim=2,
                )
                .reshape(batch_size, -1, 1)
                .long()
            )
        return batch


class ComposeAugmentation:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for aug in self.augmentations:
            batch = aug(batch)
        return batch
