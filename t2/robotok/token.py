from enum import Enum
from typing import FrozenSet

import mujoco
import numpy as np
import pydantic
from transforms3d import quaternions

GROUND_LINK_ID = -1


@pydantic.dataclasses.dataclass(frozen=True, repr=True)
class Pose:
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]

    @property
    def matrix(self):
        pose = np.eye(4)
        pose[:3, 3] = self.pos
        pose[:3, :3] = quaternions.quat2mat(self.quat_wxyz)
        return pose

    @staticmethod
    def from_matrix(matrix: np.ndarray):
        pos = matrix[:3, 3]
        quat_wxyz = quaternions.mat2quat(matrix[:3, :3])
        return Pose(pos=tuple(pos), quat_wxyz=tuple(quat_wxyz))

    # check validity of quat
    @pydantic.field_validator("quat_wxyz")
    def check_quat_validity(cls, v):
        if not quaternions.qisunit(v):
            raise ValueError("Quaternion must be a unit quaternion")
        try:
            quaternions.quat2mat(v)
        except Exception as e:
            raise ValueError(f"Invalid quaternion: {e}")
        return v

    @property
    def inverse(self):
        return Pose.from_matrix(np.linalg.inv(self.matrix))

    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix)


class GeomType(Enum):
    CAPSULE = "capsule"
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    EMPTY = "empty"

    @staticmethod
    def from_mujoco(geom_type: int):
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return GeomType.CAPSULE
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return GeomType.BOX
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return GeomType.SPHERE
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return GeomType.CYLINDER
        else:
            raise ValueError(f"Unknown geom type: {geom_type}")

    @staticmethod
    def to_int(geom_type: "GeomType"):
        if geom_type == GeomType.CAPSULE:
            return 0
        elif geom_type == GeomType.BOX:
            return 1
        elif geom_type == GeomType.SPHERE:
            return 2
        elif geom_type == GeomType.CYLINDER:
            return 3
        elif geom_type == GeomType.EMPTY:
            return 4
        else:
            raise ValueError(f"Unknown geom type: {geom_type}")

    @staticmethod
    def from_int(geom_type: int):
        if geom_type == 0:
            return GeomType.CAPSULE
        elif geom_type == 1:
            return GeomType.BOX
        elif geom_type == 2:
            return GeomType.SPHERE
        elif geom_type == 3:
            return GeomType.CYLINDER
        elif geom_type == 4:
            return GeomType.EMPTY
        else:
            raise ValueError(f"Unknown geom type: {geom_type}")


@pydantic.dataclasses.dataclass(frozen=True, repr=True, eq=True)
class Link:
    idx: int
    track_link_idx: int  # -1 if not a track link
    free_link_idx: int  # -1 if not a free link
    geom_type: GeomType
    size: tuple[float, float, float]

    mass: float
    ipos: tuple[float, float, float]
    iquat: tuple[float, float, float, float]
    diaginertia: tuple[float, float, float]

    friction: tuple[float, float, float]  # slide, spin, roll
    contact_dim: int

    rgba: tuple[float, float, float, float]

    @property
    def inertia(self):
        imat = quaternions.quat2mat(self.iquat)
        inertia = np.diag(self.diaginertia)
        inertia = imat @ inertia @ imat.T
        return inertia


class DynamicJointType(Enum):
    HINGE = "hinge"
    SLIDE = "slide"
    BALL = "ball"

    @staticmethod
    def from_mujoco(joint_type: int):
        if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            return DynamicJointType.HINGE
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            return DynamicJointType.SLIDE
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return DynamicJointType.BALL
        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

    @staticmethod
    def to_int(joint_type: "DynamicJointType"):
        if joint_type == DynamicJointType.HINGE:
            return 0
        elif joint_type == DynamicJointType.SLIDE:
            return 1
        elif joint_type == DynamicJointType.BALL:
            return 2
        else:
            raise ValueError(f"Unknown joint type: {joint_type}")

    @staticmethod
    def from_int(joint_type: int):
        if joint_type == 0:
            return DynamicJointType.HINGE
        elif joint_type == 1:
            return DynamicJointType.SLIDE
        elif joint_type == 2:
            return DynamicJointType.BALL
        else:
            raise ValueError(f"Unknown joint type: {joint_type}")


@pydantic.dataclasses.dataclass(frozen=True, repr=True)
class JointLinkConnection:
    # in the joint's local frame, its axis should always be 0, 0, 1
    # for hinge joints with limits, a joint value of 0 should be along the +x axis
    # for slide joints with limits, a joint value of 0 should be at the x-y plane

    # link_idx is the index of the link in the robot's links list
    link_idx: int
    # pose is the pose of the link in the joint's local frame
    pose: Pose

    def __eq__(self, other):
        return self.link_idx == other.link_idx and self.pose == other.pose


@pydantic.dataclasses.dataclass(frozen=True, repr=True)
class FixedJoint:
    # joint's axis is always 0, 0, 1 in it's local frame
    idx: int
    connections: FrozenSet[JointLinkConnection]

    # len(connections) must be exactly 2
    @pydantic.field_validator("connections")
    def check_connections_length(cls, v):
        if len(v) != 2:
            raise ValueError("Joint must have exactly 2 connections")
        return v


@pydantic.dataclasses.dataclass(frozen=True, repr=True)
class DynamicJoint(FixedJoint):
    joint_type: DynamicJointType
    # this means if link1 is parent and link2 is child, then link2's child pose in link1's frame
    # inverse of link1_pose * link2_pose
    joint_range: tuple[float, float]
    armature: float
    damping: float
    frictionloss: float
    stiffness: float
    spring_ref: float
    qpos0_ref: float


class ActuatorType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"

    @staticmethod
    def to_int(actuator_type: "ActuatorType"):
        if actuator_type == ActuatorType.POSITION:
            return 0
        elif actuator_type == ActuatorType.VELOCITY:
            return 1
        else:
            raise ValueError(f"Unknown actuator type: {actuator_type}")

    @staticmethod
    def from_int(actuator_type: int):
        if actuator_type == 0:
            return ActuatorType.POSITION
        elif actuator_type == 1:
            return ActuatorType.VELOCITY
        else:
            raise ValueError(f"Unknown actuator type: {actuator_type}")


@pydantic.dataclasses.dataclass(frozen=True, repr=True, eq=True)
class Actuator:
    idx: int
    dyna_joint_idx: int  # can't actuate free joints or fixed joints
    # Only position and velocity control is supported
    ctrl_range: tuple[float, float]
    force_range: tuple[float, float]
    kp: float  # 0 when is velocity controlled
    kv: float
    gear_ratio: float
    actuator_type: ActuatorType

    @pydantic.field_validator("force_range")
    def check_nonzero_force_range(cls, v):
        if v[0] > v[1]:
            raise ValueError("Force range min must be <= max")
        return v


@pydantic.dataclasses.dataclass(frozen=True, repr=True, eq=True)
class JointState:  # only for non-fixed joints
    jnt_idx: int
    qpos: tuple[
        float, float, float, float, float, float, float
    ]  # up to 7 for free joints


@pydantic.dataclasses.dataclass(frozen=True, repr=True, eq=True)
class DynamicJointState:
    dyna_jnt_idx: int
    time_idx: int
    qpos: float
    qvel: float


@pydantic.dataclasses.dataclass(frozen=True, repr=True, eq=True)
class FreeLinkObs(Pose):
    """World pose of a free link's GEOM frame at a given time index.

    Convention: throughout the tokenized representation, every link's body
    frame is equal to its (single) collision geom's frame — `detokenize`
    places the geom at the body origin for both free and non-free links.
    Consequently, the `pos` / `quat_wxyz` fields here record
    `data.geom_xpos[geom_id]` / `data.geom_xmat[geom_id]` from the source
    model, and `detokenize` writes them into the freejoint qpos slots so
    that the new body — which coincides with the new geom — lands at the
    original geom's world pose. The original body's world pose is *not*
    preserved on round-trip (only its geom's world pose is).
    """

    free_link_idx: int
    time_idx: int


@pydantic.dataclasses.dataclass(frozen=True, repr=True, eq=True)
class Robot:
    # a canonical robot representation, which has the following properties
    # 1. each link has only one geometry, centered at the origin
    # 2. each joint has axis 0, 0, 1 in it's local frame
    links: FrozenSet[Link]
    dynamic_joints: FrozenSet[DynamicJoint]
    fixed_joints: FrozenSet[FixedJoint]
    actuators: FrozenSet[Actuator]

    free_link_obs: FrozenSet[FreeLinkObs]
    dyna_jnt_obs: FrozenSet[DynamicJointState]

    def __eq__(self, other):
        return (
            sorted(self.links, key=lambda x: x.idx)
            == sorted(other.links, key=lambda x: x.idx)
            and sorted(self.dynamic_joints, key=lambda x: x.idx)
            == sorted(other.dynamic_joints, key=lambda x: x.idx)
            and sorted(self.fixed_joints, key=lambda x: x.idx)
            == sorted(other.fixed_joints, key=lambda x: x.idx)
        )
