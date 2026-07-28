import warnings
from typing import List

import numpy as np
from numpy.typing import NDArray
from transforms3d import quaternions

from t2.robotok.token import (
    Actuator,
    ActuatorType,
    DynamicJoint,
    DynamicJointState,
    DynamicJointType,
    FixedJoint,
    FreeLinkObs,
    GeomType,
    JointLinkConnection,
    Link,
    Pose,
    Robot,
)


def get_robot_serialization_schema(
    include_token_ids: bool = False, include_states: bool = False
) -> dict[str, List]:
    data_dict = {
        # fixed joints
        "fixed_joint/id": [],
        "fixed_joint/link/id": [],
        "fixed_joint/pos": [],
        "fixed_joint/rotmat": [],
        # dynamic joints
        "dyna_joint/id": [],
        "dyna_joint/link/id": [],
        "dyna_joint/pos": [],
        "dyna_joint/rotmat": [],
        "dyna_joint/type": [],
        "dyna_joint/range": [],
        "dyna_joint/armature": [],
        "dyna_joint/damping": [],
        "dyna_joint/frictionloss": [],
        "dyna_joint/stiffness": [],
        "dyna_joint/spring_ref": [],
        "dyna_joint/qpos0_ref": [],
        # links
        "link/geom_type": [],
        "link/track_link/id": [],
        "link/free_link/id": [],
        "link/geom_size": [],
        "link/mass": [],
        "link/ipos": [],
        "link/iquat": [],
        "link/imat": [],
        # NOTE: "link/inertia" is intentionally NOT included here.
        # Eigendecomposition of inertia tensors has sign ambiguity that causes
        # incorrect iquat reconstruction. Use iquat + diaginertia instead.
        "link/diaginertia": [],
        "link/friction": [],
        "link/contact_dim": [],
        "link/rgba": [],
        "link/id": [],
        # actuators
        "actuator/dyna_joint/id": [],
        "actuator/ctrl_range": [],
        "actuator/force_range": [],
        "actuator/kp": [],
        "actuator/kv": [],
        "actuator/gear_ratio": [],
        "actuator/type": [],
        "actuator/id": [],
    }
    if include_states:
        data_dict.update(
            {  # init free link state
                "free_link_obs/pos": [],
                "free_link_obs/rotmat": [],
                "free_link_obs/free_link/id": [],
                "free_link_obs/time/id": [],
                # init dynamic joint states
                "dyna_joint_obs/dyna_joint/id": [],
                "dyna_joint_obs/time/id": [],
                "dyna_joint_obs/qpos": [],
                "dyna_joint_obs/qvel": [],
            }
        )
    if not include_token_ids:
        data_dict.pop("fixed_joint/id")
        data_dict.pop("dyna_joint/id")
        data_dict.pop("link/id")
        data_dict.pop("actuator/id")
    return data_dict


def assert_contiguous_id_seq(tokens) -> List:
    sorted_tokens = sorted(tokens, key=lambda x: x.idx)
    if len(sorted_tokens) == 0:
        return []
    assert np.all(np.diff(np.array([token.idx for token in sorted_tokens])) == 1), (
        "Tokens must be contiguous"
    )
    assert sorted_tokens[0].idx == 0, "First token must be 0"
    return sorted_tokens


def serialize(
    robot: Robot, include_token_ids: bool = False, include_states: bool = False
):
    data_dict = get_robot_serialization_schema(
        include_token_ids=include_token_ids, include_states=include_states
    )
    sorted_links = assert_contiguous_id_seq(robot.links)
    sorted_dyna_joints = assert_contiguous_id_seq(robot.dynamic_joints)
    sorted_fixed_joints = assert_contiguous_id_seq(robot.fixed_joints)
    sorted_actuators = assert_contiguous_id_seq(robot.actuators)
    # at this point, all tokens have valid, contiguous IDs from 0 to seq_len - 1
    # this means training on such token sequences will no longer require
    # loading the IDs from the dataset (since they are presumed to have a fixed order)
    # and thus we also don't need to save them (which is why `include_token_ids`
    # defaults to False)
    for jnt in sorted_dyna_joints + sorted_fixed_joints:
        # TODO collapse fixed joint to only one token
        prefix = "dyna_joint" if type(jnt) is DynamicJoint else "fixed_joint"
        jnt_idx = jnt.idx
        # shared parameters for both of the joint's connections
        if include_token_ids:
            data_dict[f"{prefix}/id"].extend([[jnt_idx]] * 2)
        connections = list(sorted(jnt.connections, key=lambda x: x.link_idx))
        # link-specific parameters
        data_dict[f"{prefix}/link/id"].append([connections[0].link_idx])
        data_dict[f"{prefix}/link/id"].append([connections[1].link_idx])
        data_dict[f"{prefix}/pos"].append(connections[0].pose.pos)
        data_dict[f"{prefix}/pos"].append(connections[1].pose.pos)
        data_dict[f"{prefix}/rotmat"].append(
            quaternions.quat2mat(connections[0].pose.quat_wxyz).reshape(-1)
        )
        data_dict[f"{prefix}/rotmat"].append(
            quaternions.quat2mat(connections[1].pose.quat_wxyz).reshape(-1)
        )

        if type(jnt) is DynamicJoint:
            data_dict[f"{prefix}/type"].extend(
                [[DynamicJointType.to_int(jnt.joint_type)]] * 2
            )
            data_dict[f"{prefix}/range"].extend([jnt.joint_range] * 2)
            data_dict[f"{prefix}/armature"].extend([[jnt.armature]] * 2)
            data_dict[f"{prefix}/damping"].extend([[jnt.damping]] * 2)
            data_dict[f"{prefix}/frictionloss"].extend([[jnt.frictionloss]] * 2)
            data_dict[f"{prefix}/stiffness"].extend([[jnt.stiffness]] * 2)
            data_dict[f"{prefix}/spring_ref"].extend([[jnt.spring_ref]] * 2)
            data_dict[f"{prefix}/qpos0_ref"].extend([[jnt.qpos0_ref]] * 2)
        elif type(jnt) is FixedJoint:
            continue
        else:
            raise ValueError(f"Invalid joint type: {type(jnt)}")

    for link_idx, link in enumerate(sorted_links):
        if include_token_ids:
            data_dict["link/id"].append([link_idx])
        data_dict["link/geom_type"].append([GeomType.to_int(link.geom_type)])
        data_dict["link/track_link/id"].append([link.track_link_idx])
        data_dict["link/free_link/id"].append([link.free_link_idx])
        data_dict["link/geom_size"].append(link.size)
        data_dict["link/mass"].append([link.mass])
        data_dict["link/ipos"].append(link.ipos)
        data_dict["link/iquat"].append(link.iquat)
        R_body2ibody = quaternions.quat2mat(link.iquat)
        data_dict["link/imat"].append(R_body2ibody.reshape(-1))
        data_dict["link/diaginertia"].append(link.diaginertia)
        # NOTE: We intentionally do NOT store link/inertia (full 3x3 inertia tensor).
        # Eigendecomposition to recover iquat has sign ambiguity issues.
        data_dict["link/friction"].append(link.friction)
        data_dict["link/contact_dim"].append([link.contact_dim])
        data_dict["link/rgba"].append(link.rgba)

    if len(sorted_actuators) > 0:
        for actuator_idx, actuator in enumerate(sorted_actuators):
            if include_token_ids:
                data_dict["actuator/id"].append([actuator_idx])
            data_dict["actuator/dyna_joint/id"].append([actuator.dyna_joint_idx])
            data_dict["actuator/ctrl_range"].append(actuator.ctrl_range)
            data_dict["actuator/force_range"].append(actuator.force_range)
            data_dict["actuator/gear_ratio"].append([actuator.gear_ratio])
            data_dict["actuator/kp"].append([actuator.kp])
            data_dict["actuator/kv"].append([actuator.kv])
            data_dict["actuator/type"].append(
                [ActuatorType.to_int(actuator.actuator_type)]
            )
    # TODO handle include_token_ids = False for free link and dynamic joint obs
    if include_states:
        for robot_obs in robot.free_link_obs:
            data_dict["free_link_obs/pos"].append(robot_obs.pos)
            data_dict["free_link_obs/rotmat"].append(
                quaternions.quat2mat(robot_obs.quat_wxyz).reshape(-1)
            )
            data_dict["free_link_obs/free_link/id"].append([robot_obs.free_link_idx])
            data_dict["free_link_obs/time/id"].append([robot_obs.time_idx])
        for robot_obs in robot.dyna_jnt_obs:
            data_dict["dyna_joint_obs/dyna_joint/id"].append([robot_obs.dyna_jnt_idx])
            data_dict["dyna_joint_obs/time/id"].append([robot_obs.time_idx])
            data_dict["dyna_joint_obs/qpos"].append([robot_obs.qpos])
            data_dict["dyna_joint_obs/qvel"].append([robot_obs.qvel])
    return {k: np.array(v) for k, v in data_dict.items()}


def deserialize(
    data_dict: dict[str, NDArray[np.float32 | np.int_ | np.bool_]],
    include_states: bool = False,
) -> Robot:
    links: list[Link] = []
    if "link/id" in data_dict:
        link_ids = data_dict["link/id"].reshape(-1)
    else:
        link_ids = np.arange(len(data_dict["link/geom_type"]))
    for i, link_idx in enumerate(link_ids):
        # Prefer using iquat directly if available, since eigenvalue decomposition
        # from inertia tensor can give different rotations due to sign ambiguity
        if "link/iquat" in data_dict:
            iquat = tuple(data_dict["link/iquat"][i])
            diaginertia = data_dict["link/diaginertia"][i]
        elif "link/imat" in data_dict:
            iquat = tuple(quaternions.mat2quat(data_dict["link/imat"][i].reshape(3, 3)))
            diaginertia = data_dict["link/diaginertia"][i]
        elif "link/inertia" in data_dict:
            # DEPRECATED: Fall back to eigendecomposition only for legacy data.
            # This path has sign ambiguity issues and should not be used for new data.
            warnings.warn(
                "Deserializing from 'link/inertia' is deprecated and may produce "
                "incorrect inertia orientation due to eigendecomposition sign ambiguity. "
                "Re-serialize data with 'link/iquat' to avoid this issue.",
                DeprecationWarning,
                stacklevel=2,
            )
            inertia = data_dict["link/inertia"][i].reshape(3, 3)
            diaginertia, imat = np.linalg.eigh(inertia)
            iquat = quaternions.mat2quat(imat)
        else:
            raise ValueError("Missing inertia information in data_dict")

        links.append(
            Link(
                idx=int(link_idx),
                geom_type=GeomType.from_int(int(data_dict["link/geom_type"][i].item())),
                track_link_idx=int(data_dict["link/track_link/id"][i].item()),
                free_link_idx=int(
                    data_dict["link/free_link/id"][i].item()
                    if "link/free_link/id" in data_dict
                    else -1
                ),
                size=tuple(data_dict["link/geom_size"][i].tolist()),
                mass=float(data_dict["link/mass"][i].item()),
                ipos=tuple(data_dict["link/ipos"][i].tolist()),
                iquat=tuple(float(x) for x in iquat),
                diaginertia=tuple(float(x) for x in diaginertia),
                friction=tuple(data_dict["link/friction"][i].tolist()),
                contact_dim=int(data_dict["link/contact_dim"][i].item()),
                rgba=tuple(data_dict["link/rgba"][i].tolist()),
            )
        )
    dyna_joints: list[DynamicJoint] = []
    fixed_joints: list[FixedJoint] = []
    if "dyna_joint/id" in data_dict:
        dyna_joint_ids = data_dict["dyna_joint/id"]
    else:
        assert len(data_dict["dyna_joint/type"]) % 2 == 0, (
            "Dynamic joints must have even number of types"
        )
        dyna_joint_ids = np.array(
            [np.arange(int(len(data_dict["dyna_joint/type"]) // 2))] * 2
        ).T.reshape(-1)
    for joint_idx in np.unique(dyna_joint_ids):
        is_this_joint = dyna_joint_ids == joint_idx
        assert is_this_joint.sum() == 2, "Joint must have two connections"
        indices = np.where(is_this_joint)[0]
        dyna_joints.append(
            DynamicJoint(
                idx=int(joint_idx),
                joint_type=DynamicJointType.from_int(
                    int(data_dict["dyna_joint/type"][indices[0]].item())
                ),
                connections=frozenset(
                    {
                        JointLinkConnection(
                            link_idx=int(data_dict["dyna_joint/link/id"][indices[0]].item()),
                            pose=Pose(
                                pos=tuple(data_dict["dyna_joint/pos"][indices[0]].tolist()),
                                quat_wxyz=tuple(
                                    float(x) for x in quaternions.mat2quat(
                                        data_dict["dyna_joint/rotmat"][indices[0]]
                                    )
                                ),
                            ),
                        ),
                        JointLinkConnection(
                            link_idx=int(data_dict["dyna_joint/link/id"][indices[1]].item()),
                            pose=Pose(
                                pos=tuple(data_dict["dyna_joint/pos"][indices[1]].tolist()),
                                quat_wxyz=tuple(
                                    float(x) for x in quaternions.mat2quat(
                                        data_dict["dyna_joint/rotmat"][indices[1]]
                                    )
                                ),
                            ),
                        ),
                    }
                ),
                joint_range=tuple(data_dict["dyna_joint/range"][indices[0]].tolist()),
                armature=float(data_dict["dyna_joint/armature"][indices[0]].item()),
                damping=float(data_dict["dyna_joint/damping"][indices[0]].item()),
                frictionloss=float(data_dict["dyna_joint/frictionloss"][indices[0]].item()),
                stiffness=float(data_dict["dyna_joint/stiffness"][indices[0]].item()),
                spring_ref=float(data_dict["dyna_joint/spring_ref"][indices[0]].item()),
                qpos0_ref=float(data_dict["dyna_joint/qpos0_ref"][indices[0]].item()),
            )
        )

    if "fixed_joint/id" in data_dict:
        fixed_joint_ids = data_dict["fixed_joint/id"]
    else:
        assert len(data_dict["fixed_joint/pos"]) % 2 == 0, (
            "Fixed joints must have even number of positions"
        )
        fixed_joint_ids = np.array(
            [np.arange(int(len(data_dict["fixed_joint/pos"]) // 2))] * 2
        ).T.reshape(-1)
    for joint_idx in np.unique(fixed_joint_ids):
        is_this_joint = fixed_joint_ids == joint_idx
        assert is_this_joint.sum() == 2, "Joint must have two connections"
        indices = np.where(is_this_joint)[0]
        fixed_joints.append(
            FixedJoint(
                idx=int(joint_idx),
                connections=frozenset(
                    {
                        JointLinkConnection(
                            link_idx=int(data_dict["fixed_joint/link/id"][indices[0]].item()),
                            pose=Pose(
                                pos=tuple(data_dict["fixed_joint/pos"][indices[0]].tolist()),
                                quat_wxyz=tuple(
                                    float(x) for x in quaternions.mat2quat(
                                        data_dict["fixed_joint/rotmat"][indices[0]]
                                    )
                                ),
                            ),
                        ),
                        JointLinkConnection(
                            link_idx=int(data_dict["fixed_joint/link/id"][indices[1]].item()),
                            pose=Pose(
                                pos=tuple(data_dict["fixed_joint/pos"][indices[1]].tolist()),
                                quat_wxyz=tuple(
                                    float(x) for x in quaternions.mat2quat(
                                        data_dict["fixed_joint/rotmat"][indices[1]]
                                    )
                                ),
                            ),
                        ),
                    }
                ),
            )
        )

    if "actuator/id" in data_dict:
        actuator_ids = data_dict["actuator/id"]
    else:
        actuator_ids = np.arange(len(data_dict["actuator/dyna_joint/id"]))

    actuators: list[Actuator] = []
    for actuator_idx in np.unique(actuator_ids):
        idx = int(actuator_idx)
        actuators.append(
            Actuator(
                idx=idx,
                dyna_joint_idx=int(data_dict["actuator/dyna_joint/id"][idx].item()),
                ctrl_range=tuple(sorted(data_dict["actuator/ctrl_range"][idx].tolist())),
                force_range=tuple(sorted(data_dict["actuator/force_range"][idx].tolist())),
                kp=float(data_dict["actuator/kp"][idx].item()),
                kv=float(data_dict["actuator/kv"][idx].item()),
                actuator_type=ActuatorType.from_int(
                    int(data_dict["actuator/type"][idx].item())
                ),
                gear_ratio=float(data_dict["actuator/gear_ratio"][idx].item()),
            )
        )

    free_link_obs: list[FreeLinkObs] = []
    dyna_jnt_obs: list[DynamicJointState] = []
    if include_states and "free_link_obs/pos" in data_dict:
        for i in range(len(data_dict["free_link_obs/pos"])):
            free_link_obs.append(
                FreeLinkObs(
                    pos=tuple(data_dict["free_link_obs/pos"][i].tolist()),
                    quat_wxyz=tuple(
                        float(x) for x in quaternions.mat2quat(data_dict["free_link_obs/rotmat"][i])
                    ),
                    free_link_idx=int(data_dict["free_link_obs/free_link/id"][i][0]),
                    time_idx=int(data_dict["free_link_obs/time/id"][i][0]),
                )
            )

    if include_states and "dyna_joint_obs/dyna_joint/id" in data_dict:
        for i in range(len(data_dict["dyna_joint_obs/dyna_joint/id"])):
            dyna_jnt_obs.append(
                DynamicJointState(
                    dyna_jnt_idx=int(data_dict["dyna_joint_obs/dyna_joint/id"][i][0]),
                    time_idx=int(data_dict["dyna_joint_obs/time/id"][i][0]),
                    qpos=float(data_dict["dyna_joint_obs/qpos"][i][0]),
                    qvel=float(data_dict["dyna_joint_obs/qvel"][i][0]),
                )
            )

    return Robot(
        links=frozenset(links),
        dynamic_joints=frozenset(dyna_joints),
        fixed_joints=frozenset(fixed_joints),
        actuators=frozenset(actuators),
        free_link_obs=frozenset(free_link_obs),
        dyna_jnt_obs=frozenset(dyna_jnt_obs),
    )
