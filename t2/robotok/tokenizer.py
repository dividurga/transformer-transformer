"""Utilities for converting between MuJoCo XML and tokenized robots.

Conventions used throughout this module
--------------------------------------
- Transformations use the form ``T_ab`` to denote the pose of frame ``b``
  expressed in frame ``a``. The same notation applies to positions ``p_ab`` and
  rotations ``R_ab``.
- Quaternions are always in ``wxyz`` order and all angles are in radians.
- ``link_idx`` and ``jnt_idx`` refer to the indices of links and joints in the
  tokenized representation.
"""

import logging
from graphlib import CycleError, TopologicalSorter

import mujoco
import numpy as np
from dm_control import mjcf
from transforms3d import affines, euler, quaternions

from t2.env.mj_utils import (
    ALLOWED_GEOM_PRIMITIVES,
    TRACKING_SITE_GROUP,
    default_root_element,
)
from t2.env.transforms import mjuu_z2quat, rotation_matrix_from_vectors
from t2.robotok.token import (
    GROUND_LINK_ID,
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

WARNED_ABOUT_DYNA_JOINT_ERRORS = False


class TokenizeError(Exception):
    pass


class DetokenizeError(Exception):
    pass


# TODO refactor `_tokenize_joint` and `_tokenize_fixed_joint` to share
# transform utilities


def _tokenize_joint(
    T_b2g2: Pose,
    T_b1g1: Pose,
    T_b2b1: Pose,
    jnt_pos: np.ndarray,
    jnt_axis: np.ndarray,
    jnt_type: DynamicJointType,
    prev_link_idx: int,
    new_link_idx: int,
    new_dyna_jnt_idx: int,
    joint_range: tuple[float, float],
    armature: float,
    damping: float,
    frictionloss: float,
    stiffness: float,
    spring_ref: float,
    qpos0_ref: float,
) -> DynamicJoint:
    """
    T_b2g2: pose of parent (2)'s geometry in parent's frame
    T_b1g1: pose of child (1)'s geometry in child's frame
    T_b2b1: pose of child (1)'s frame in parent's frame
    jnt_pos: position of joint in child's frame
    jnt_axis: axis of joint in child's frame
    jnt_type: type of joint
    prev_link_idx: index of parent 2
    new_link_idx: index of child 1
    new_jnt_idx: index of joint 1
    """
    T_g1g2 = Pose.from_matrix(
        T_b1g1.inverse.matrix @ T_b2b1.inverse.matrix @ T_b2g2.matrix
    )
    jnt_rot_mat = rotation_matrix_from_vectors(np.array([0, 0, 1]), jnt_axis)
    jnt_quat = quaternions.mat2quat(jnt_rot_mat)
    # j is the joint
    T_b1j = Pose(
        pos=tuple(jnt_pos),
        quat_wxyz=tuple(jnt_quat),
    )  # pose of B's joint in B's frame
    T_jg1 = Pose.from_matrix(
        T_b1j.inverse.matrix @ T_b1g1.matrix
    )  # pose of B's joint in B's geometry frame
    T_jg2 = Pose.from_matrix(
        T_jg1.matrix @ T_g1g2.matrix
    )  # pose of parent A's geometry in B's joint frame

    jnt = DynamicJoint(
        idx=new_dyna_jnt_idx,
        joint_type=jnt_type,
        connections=frozenset(
            {
                JointLinkConnection(link_idx=prev_link_idx, pose=T_jg2),
                JointLinkConnection(link_idx=new_link_idx, pose=T_jg1),
            }
        ),
        joint_range=joint_range,
        armature=armature,
        damping=damping,
        frictionloss=frictionloss,
        stiffness=stiffness,
        spring_ref=spring_ref,
        qpos0_ref=qpos0_ref,
    )
    return jnt


def _tokenize_fixed_joint(
    T_b2g2: Pose,
    T_b1g1: Pose,
    T_b2b1: Pose,
    jnt_pos: np.ndarray,
    jnt_axis: np.ndarray,
    prev_link_idx: int,
    new_link_idx: int,
    new_fixed_jnt_idx: int,
) -> FixedJoint:
    """
    T_b2g2: pose of parent (2)'s geometry in parent's frame
    T_b1g1: pose of child (1)'s geometry in child's frame
    T_b2b1: pose of child (1)'s frame in parent's frame
    jnt_pos: position of joint in child's frame
    jnt_axis: axis of joint in child's frame
    jnt_type: type of joint
    prev_link_idx: index of parent 2
    new_link_idx: index of child 1
    new_fixed_jnt_idx: index of joint 1
    """
    T_g1g2 = Pose.from_matrix(
        T_b1g1.inverse.matrix @ T_b2b1.inverse.matrix @ T_b2g2.matrix
    )
    jnt_rot_mat = rotation_matrix_from_vectors(np.array([0, 0, 1]), jnt_axis)
    jnt_quat = quaternions.mat2quat(jnt_rot_mat)
    # j is the joint
    T_b1j = Pose(
        pos=tuple(jnt_pos),
        quat_wxyz=tuple(jnt_quat),
    )  # pose of B's joint in B's frame
    T_jg1 = Pose.from_matrix(
        T_b1j.inverse.matrix @ T_b1g1.matrix
    )  # pose of B's joint in B's geometry frame
    T_jg2 = Pose.from_matrix(
        T_jg1.matrix @ T_g1g2.matrix
    )  # pose of parent A's geometry in B's joint frame

    # in mujoco, fixed joints are defined implicitly, so joint-child
    # transforms can't be stored in the non-existent joint. Therefore,
    # we need to move all the pose offsets to the parent-child transform
    T_jg2 = Pose.from_matrix(T_jg1.inverse.matrix @ T_jg2.matrix)
    T_jg1 = Pose.from_matrix(np.identity(4))
    jnt = FixedJoint(
        idx=new_fixed_jnt_idx,
        connections=frozenset(
            {
                JointLinkConnection(link_idx=prev_link_idx, pose=T_jg2),
                JointLinkConnection(link_idx=new_link_idx, pose=T_jg1),
            }
        ),
    )
    return jnt


def _detokenize_joint(
    jnt: DynamicJoint | FixedJoint,
    child_idx: int,
) -> tuple[Pose, np.ndarray, np.ndarray]:
    """
    Returns:
        Tuple of (T_g2g1, jnt_pos, jnt_axis) where:
        - T_g2g1: pose of child's frame in parent's frame
        - jnt_pos: position of joint in child's frame
        - jnt_axis: axis of joint in child's frame
    """
    # Find parent and child connections
    child_conn = next(conn for conn in jnt.connections if conn.link_idx == child_idx)
    parent_conn = next(conn for conn in jnt.connections if conn.link_idx != child_idx)

    # T_jBgB is the child's connection pose
    T_jg1 = child_conn.pose
    # T_jBgA is the parent's connection pose
    T_jg2 = parent_conn.pose

    # Get T_gBgA from T_jBgA and T_jBgB
    T_g2g1 = Pose.from_matrix(T_jg2.inverse.matrix @ T_jg1.matrix)

    # Since joint axis is always [0, 0, 1] in joint frame,
    # we can recover jnt_axis from T_BjB's rotation
    T_g1j = T_jg1.inverse
    jnt_rot_mat = quaternions.quat2mat(T_g1j.quat_wxyz)
    jnt_axis = jnt_rot_mat @ np.array([0, 0, 1])
    jnt_pos = np.array(T_g1j.pos)

    return T_g2g1, jnt_pos, jnt_axis


def _is_ground(
    model: mujoco.MjModel,
    body_idx: int,
) -> bool:
    geom_ids = np.arange(
        model.body_geomadr[body_idx],
        model.body_geomadr[body_idx] + model.body_geomnum[body_idx],
    )
    if (geom_ids == -1).any():
        return False
    geom_types = model.geom_type[geom_ids]
    has_plane = any(geom_types == mujoco.mjtGeom.mjGEOM_PLANE)
    return has_plane


def _geom_to_volume(
    geom_type: GeomType,
    size: tuple[float, float, float],
) -> float:
    """
    From MuJoCo's documentation
    Type, Size, Interpretation
    sphere, 1, Radius of the sphere.
    capsule, 2, Radius of the capsule; half-length of the cylinder part.
    cylinder, 2, Radius of the cylinder; half-length of the cylinder.
    box, 3, X half-size; Y half-size; Z half-size.
    """
    if geom_type == GeomType.SPHERE:
        assert size[0] != 0 and size[1] == 0 and size[2] == 0
        radius = size[0]
        return 4 / 3 * np.pi * radius**3
    elif geom_type == GeomType.CAPSULE:
        assert size[0] != 0 and size[1] != 0 and size[2] == 0
        radius = size[0]
        length = size[1] * 2
        volume_of_sphere = 4 / 3 * np.pi * radius**3
        volume_of_cylinder = np.pi * radius**2 * length
        return volume_of_sphere + volume_of_cylinder
    elif geom_type == GeomType.CYLINDER:
        assert size[0] != 0 and size[1] != 0 and size[2] == 0
        radius = size[0]
        length = size[1] * 2
        volume_of_cylinder = np.pi * radius**2 * length
        return volume_of_cylinder
    elif geom_type == GeomType.BOX:
        assert size[0] != 0 and size[1] != 0 and size[2] != 0
        h = size[2] * 2
        w = size[0] * 2
        l = size[1] * 2
        return h * w * l
    else:
        raise ValueError(f"Unknown geom type: {geom_type}")


def _geom_to_diag_inertia(
    geom_type: GeomType,
    m: float,
    size: tuple[float, float, float],
) -> np.ndarray:
    if geom_type == GeomType.BOX:
        h = size[2] * 2
        w = size[0] * 2
        l = size[1] * 2
        Ixx = m * (w**2 + l**2) / 12
        Iyy = m * (h**2 + l**2) / 12
        Izz = m * (h**2 + w**2) / 12
        return np.array([Ixx, Iyy, Izz])
    elif geom_type == GeomType.SPHERE:
        radius = size[0]
        return 2 / 5 * m * radius**2 * np.ones(3)
    elif geom_type == GeomType.CYLINDER:
        radius = size[0]
        length = size[1] * 2

        Izz = m * radius**2 / 2
        Ixx = Iyy = m * (3 * radius**2 + length**2) / 12
        return np.array([Ixx, Iyy, Izz])
    elif geom_type == GeomType.CAPSULE:
        # Capsule parameters:
        radius = size[0]
        length = size[1] * 2  # Cylinder length

        # Volumes
        vol_cyl = np.pi * radius**2 * length
        vol_hemi = (2 / 3) * np.pi * radius**3  # volume of one hemisphere
        vol_total = vol_cyl + 2 * vol_hemi

        # Mass distribution
        m_cyl = m * vol_cyl / vol_total
        m_hemi = m * vol_hemi / vol_total  # each hemisphere

        # Cylinder inertias (about its own center)
        Izz_cyl = 0.5 * m_cyl * radius**2
        Ixx_cyl = (m_cyl * (3 * radius**2 + length**2)) / 12

        # Hemisphere inertias (about their own center-of-mass)
        Izz_hemi_cm = (2 / 5) * m_hemi * radius**2
        Ixx_hemi_cm = (83 / 320) * m_hemi * radius**2  # approximate value

        # Distance from capsule center to hemisphere center-of-mass along z
        d = (length / 2) + (3 * radius / 8)

        # Use parallel axis theorem for hemispheres (for x and y axes)
        Ixx_hemi = Ixx_hemi_cm + m_hemi * d**2

        # Total inertia: sum contributions from cylinder and both hemispheres
        Izz = Izz_cyl + 2 * Izz_hemi_cm
        Ixx = Ixx_cyl + 2 * Ixx_hemi

        return np.array([Ixx, Ixx, Izz])
    else:
        raise ValueError("Unsupported geometry type.")


def _shift_inertia(full_inertia: np.ndarray, m: float, displacement: np.ndarray):
    """
    Based on Steiner's theorem
    full_inertia: inertia of the object at its COM position
    m: mass of the object
    displacement: displacement from COM to new point
    """
    return full_inertia + m * (
        np.inner(displacement, displacement) * np.eye(3)
        - np.outer(displacement, displacement)
    )


def _convert_tracking_sites(mj_robot: mjcf.RootElement) -> int:
    """Convert tracking sites to tiny spherical geoms and return their count."""
    num_sites = 0
    connect_constraints = mj_robot.find_all("equality")
    for node_id, node in enumerate(mj_robot.worldbody.find_all("site")):
        if node.group not in {TRACKING_SITE_GROUP}:
            continue
        num_sites += 1
        body = node.parent
        transform_attrs = {
            attr: getattr(node, attr)
            for attr in ["pos", "quat", "axisangle", "xyaxes", "zaxis", "euler"]
        }
        site_body = body.add("body", name=node.name + "_site_body", **transform_attrs)
        site_body.add(
            "geom",
            name=node.name + "_site_geom",
            type="sphere",
            size=[0.0001],
            mass=0.0001,
            contype=1,
            conaffinity=1,
            group=3,
        )
        node_name = node.name
        node.name = node.name + "_old"
        new_site = site_body.add("site", name=node_name, group=node.group)
        for connect in connect_constraints:
            if connect.site1 == node:
                connect.site1 = new_site
            elif connect.site2 == node:
                connect.site2 = new_site
        node.remove()
    return num_sites


def _normalize_geoms(mj_robot: mjcf.RootElement, angle: str) -> None:
    """Normalize geom orientation attributes and ensure names."""
    for node_id, node in enumerate(mj_robot.worldbody.find_all("geom")):
        mjcf.commit_defaults(node)
        node.dclass = None
        if node.euler is not None:
            node_euler = np.array(node.euler)
            if angle != "radian":
                logging.debug("interpreting angle as degree")
                if mj_robot.compiler.eulerseq == "zyx":
                    node_euler = node_euler[::-1]
                else:
                    assert mj_robot.compiler.eulerseq == "xyz"
                node_euler = np.deg2rad(node_euler)
            node.quat = euler.euler2quat(*node_euler)
            node.euler = None
        if node.axisangle is not None:
            assert angle == "radian"
            vector = node.axisangle
            theta = np.linalg.norm(vector)
            vector = vector / theta
            node.quat = quaternions.axangle2quat(vector, theta, is_normalized=True)
            node.axisangle = None
        if node.name is None:
            node.name = f"geom_{node_id}"
        if node.fromto is not None:
            p1 = node.fromto[:3]
            p2 = node.fromto[3:]
            if node.type == "capsule" or node.type == "cylinder":
                radius = node.size[0]
                vec = p1 - p2
                length = np.linalg.norm(vec)
                vec = vec / length
                node.fromto = None
                node.size = [radius, length / 2, 0]
                node.quat = mjuu_z2quat(vec)
                node.pos = (p1 + p2) / 2
            elif node.type == "box":
                # For box with fromto: size[0], size[1] are x/y half-sizes,
                # z half-size is computed from the fromto distance
                x_half = node.size[0]
                y_half = node.size[1] if len(node.size) > 1 else node.size[0]
                vec = p1 - p2
                length = np.linalg.norm(vec)
                vec = vec / length
                node.fromto = None
                node.size = [x_half, y_half, length / 2]
                node.quat = mjuu_z2quat(vec)
                node.pos = (p1 + p2) / 2
            else:
                raise NotImplementedError(
                    f"Unsupported geom type for `fromto` conversion: {node.type}"
                )


def _cleanup_transform_defaults(mj_robot: mjcf.RootElement) -> None:
    """Remove transform attributes from defaults."""
    for dclass in mj_robot.find_all("default"):
        for child in dclass.all_children():
            for attr in [
                "pos",
                "quat",
                "axisangle",
                "xyaxes",
                "zaxis",
                "euler",
            ]:
                if hasattr(child, attr):
                    delattr(child, attr)


def _add_collision_boxes(
    mj_robot: mjcf.RootElement, model: mujoco.MjModel, has_collision_class: bool
) -> None:
    """Add a default collision box for bodies that only define inertials."""
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name == "world":
            continue
        node = mj_robot.worldbody.find("body", body_name)
        if node is None:
            continue
        has_inertial = node.inertial is not None
        has_no_collision_geom = not any(
            model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0
            for geom_id in range(
                model.body_geomadr[body_id],
                model.body_geomadr[body_id] + model.body_geomnum[body_id],
            )
        )
        should_add_geom = has_inertial and has_no_collision_geom
        if not should_add_geom:
            continue
        assert has_collision_class
        Ixx, Iyy, Izz = model.body_inertia[body_id]
        mass = model.body_mass[body_id]
        scale_inertia = np.sqrt(3)
        size = np.zeros(3)
        size[0] = np.sqrt((Iyy + Izz - Ixx) / (2 * mass)) * scale_inertia
        size[1] = np.sqrt((Ixx + Izz - Iyy) / (2 * mass)) * scale_inertia
        size[2] = np.sqrt((Ixx + Iyy - Izz) / (2 * mass)) * scale_inertia
        quat = model.body_iquat[body_id]
        pos = model.body_ipos[body_id]
        node.add(
            "geom",
            name=node.name + "_collision",
            type="box",
            size=size,
            pos=pos,
            quat=quat,
            dclass="collision",
        )


def _remove_visual_geoms(mj_robot: mjcf.RootElement, model: mujoco.MjModel) -> None:
    """Remove all visual geoms from the model."""
    visual_geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).split("/")[-1]
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] == 0
    ]
    for geom in mj_robot.find_all("geom"):
        if geom.name in visual_geom_names:
            # delete from assets
            if geom.mesh is not None:
                for child in mj_robot.asset.all_children():
                    if child.file == geom.mesh.file:
                        child.remove()
                        break
            geom.remove()


def _split_multi_collision_geoms(
    mj_robot: mjcf.RootElement, model: mujoco.MjModel
) -> None:
    """Split bodies that contain multiple collision geoms into separate bodies."""
    multigeom_bodyids = np.arange(model.nbody)[(model.body_geomnum[:] > 1)]
    multi_collision_geom_bodies: dict[str, dict[str, np.ndarray]] = {}
    for body_id in multigeom_bodyids:
        body_geom_ids = np.arange(
            model.body_geomadr[body_id],
            model.body_geomadr[body_id] + model.body_geomnum[body_id],
        )
        body_geom_contype = model.geom_contype[body_geom_ids]
        body_geom_conaffinity = model.geom_conaffinity[body_geom_ids]
        is_contact_geom = np.logical_and(
            body_geom_contype != 0, body_geom_conaffinity != 0
        )
        if is_contact_geom.sum() <= 1:
            continue
        body_geom_contact_ids = body_geom_ids[is_contact_geom]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        multi_collision_geom_bodies[body_name] = {}
        for geom_id in sorted(body_geom_contact_ids):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            multi_collision_geom_bodies[body_name][geom_name] = affines.compose(
                T=tuple(model.geom_pos[geom_id]),
                R=tuple(quaternions.quat2mat(model.geom_quat[geom_id])),
                Z=np.ones(3),
            )

    bodies_to_inertia_split = {}
    for body_name in multi_collision_geom_bodies.keys():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_inertia = model.body_inertia[body_id]
        body_mass = model.body_mass[body_id]
        body_ipos = model.body_ipos[body_id]
        body_iquat = model.body_iquat[body_id]

        geom_infos = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id): {
                "id": geom_id,
                "volume": float(
                    _geom_to_volume(
                        GeomType.from_mujoco(model.geom_type[geom_id]),
                        tuple(model.geom_size[geom_id]),
                    )
                ),
                "pos_body": model.geom_pos[geom_id],
                "quat_body": model.geom_quat[geom_id],
                "ipos_body": model.geom_pos[geom_id],
                "iquat_body": model.geom_quat[geom_id],
            }
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == body_id
            and model.geom_contype[geom_id] != 0
        }
        total_volume = sum(v["volume"] for v in geom_infos.values())
        density = body_mass / total_volume
        for geom_id, geom_info in geom_infos.items():
            geom_info["mass"] = geom_info["volume"] * density
        main_geom_name = max(geom_infos, key=lambda x: geom_infos[x]["volume"])
        main_geom_mass = geom_infos[main_geom_name]["mass"]

        remaining_com = np.zeros(3)
        for geom_name, geom_info in geom_infos.items():
            if geom_name == main_geom_name:
                continue
            remaining_com += geom_info["ipos_body"] * geom_info["mass"]
        main_geom_ipos = (body_ipos * body_mass - remaining_com) / main_geom_mass
        geom_infos[main_geom_name]["ipos_body"] = main_geom_ipos
        geom_infos[main_geom_name]["ipos"] = main_geom_ipos

        R_body2ibody = quaternions.quat2mat(body_iquat)
        for geom_name, geom_info in geom_infos.items():
            if geom_name == main_geom_name:
                continue
            geom_diag_inertia = _geom_to_diag_inertia(
                GeomType.from_mujoco(model.geom_type[geom_info["id"]]),
                geom_info["mass"],
                tuple(model.geom_size[geom_info["id"]]),
            )
            geom_inertia_igeom = np.diag(geom_diag_inertia)
            geom_ipos = geom_info["ipos_body"]
            displacement_body = body_ipos - geom_ipos
            R_body2igeom = quaternions.quat2mat(geom_info["iquat_body"])
            T_body2igeom = affines.compose(T=geom_ipos, R=R_body2igeom, Z=np.ones(3))
            displacement_igeom = T_body2igeom @ np.concatenate((displacement_body, [1]))
            displacement_igeom = displacement_igeom[:3]
            geom_inertia_igeom = _shift_inertia(
                geom_inertia_igeom, geom_info["mass"], displacement_igeom
            )
            R_igeom2body = R_body2igeom.T
            R_igeom2ibody = R_body2ibody @ R_igeom2body
            geom_inertia_ibody = R_igeom2ibody.T @ geom_inertia_igeom @ R_igeom2ibody
            geom_infos[geom_name]["inertia_ibody"] = geom_inertia_ibody

        main_geom_fullinertia_ibody = np.diag(body_inertia) - sum(
            geom_volume["inertia_ibody"]
            for geom_name, geom_volume in geom_infos.items()
            if geom_name != main_geom_name
        )
        R_ibody2body = R_body2ibody.T
        main_geom_inertia_body = (
            R_ibody2body.T @ main_geom_fullinertia_ibody @ R_ibody2body
        )
        displacement_body = -body_ipos
        main_geom_inertia_body = _shift_inertia(
            main_geom_inertia_body,
            m=main_geom_mass,
            displacement=displacement_body,
        )
        geom_infos[main_geom_name]["inertia_body"] = main_geom_inertia_body
        bodies_to_inertia_split[body_name] = geom_infos

    for body_name, body_geoms in multi_collision_geom_bodies.items():
        for local_geom_idx, (geom_name, geom_inertia) in enumerate(
            sorted(
                bodies_to_inertia_split[body_name].items(),
                key=lambda v: v[1]["volume"],
                reverse=True,
            )
        ):
            geom_transform = body_geoms[geom_name]
            should_split = local_geom_idx != 0

            old_geom = mj_robot.worldbody.find("geom", geom_name)
            old_body = old_geom.parent
            new_geom = old_geom
            new_body = old_body
            if should_split:
                new_body = old_body.add(
                    "body",
                    pos=geom_transform[:3, 3],
                    quat=quaternions.mat2quat(geom_transform[:3, :3]),
                )
                new_geom = new_body.add("geom")

            geom_mass = None
            geom_density = None
            geom_shellinertia = None
            if (
                body_name in bodies_to_inertia_split
                and "diaginertia" not in bodies_to_inertia_split[body_name][geom_name]
            ):
                geom_mass = bodies_to_inertia_split[body_name][geom_name]["mass"]
            else:
                geom_mass = old_geom.mass
                geom_density = old_geom.density
                geom_shellinertia = old_geom.shellinertia

            new_geom.size = old_geom.size
            new_geom.type = old_geom.type
            new_geom.contype = old_geom.contype
            new_geom.condim = old_geom.condim
            new_geom.conaffinity = old_geom.conaffinity
            new_geom.priority = old_geom.priority
            new_geom.friction = old_geom.friction
            if old_geom.dclass is not None:
                new_geom.dclass = old_geom.dclass.dclass
            new_geom.group = old_geom.group
            new_geom.solmix = old_geom.solmix
            new_geom.solref = old_geom.solref
            new_geom.solimp = old_geom.solimp
            new_geom.margin = old_geom.margin
            new_geom.gap = old_geom.gap
            new_geom.rgba = old_geom.rgba
            new_geom.density = geom_density
            new_geom.mass = geom_mass
            new_geom.shellinertia = geom_shellinertia

            geom_inertia = bodies_to_inertia_split[body_name][geom_name]

            if "diaginertia" in geom_inertia:
                if should_split:
                    new_body.add(
                        "inertial",
                        mass=geom_inertia["mass"],
                        pos=geom_inertia["ipos"],
                        quat=geom_inertia["iquat"],
                        diaginertia=geom_inertia["diaginertia"],
                    )
                else:
                    if old_body.inertial is not None:
                        old_body.inertial.remove()
                    old_body.add(
                        "inertial",
                        mass=geom_inertia["mass"],
                        pos=geom_inertia["ipos"],
                        quat=geom_inertia["iquat"],
                        diaginertia=geom_inertia["diaginertia"],
                    )
            if should_split:
                old_geom.remove()


def preprocess_mjcf(
    mj_robot: mjcf.RootElement,
    remove_asset: bool = False,
    remove_visual: bool = False,
) -> mjcf.RootElement:
    # WARNING: this function modifies the input mjcf model
    # Because making a deepcopy slows down the function
    # NOTE: we will not handle shellinertia here
    angle = mj_robot.compiler.angle
    mj_robot.compiler.fusestatic = False

    _convert_tracking_sites(mj_robot)
    _normalize_geoms(mj_robot, angle)

    has_collision_class = False
    for dclass in mj_robot.find_all("default"):
        if dclass.dclass == "collision":
            has_collision_class = True

    for feature in ["sensor"]:
        for node in mj_robot.find_all(feature):
            node.remove()

    physics = mjcf.Physics.from_mjcf_model(mj_robot)
    model = physics.model.ptr

    _add_collision_boxes(mj_robot, model, has_collision_class)

    physics = mjcf.Physics.from_mjcf_model(mj_robot)
    model = physics.model.ptr

    _split_multi_collision_geoms(mj_robot, model)

    if remove_visual:
        _remove_visual_geoms(mj_robot, model)

    _cleanup_transform_defaults(mj_robot)

    if remove_asset:
        mj_robot.remove(mj_robot.asset)
    return mj_robot


def _is_empty_xml_node(node: mjcf.Element):
    xml_data = str(node).replace("<", "").replace("/>", "")
    return " " not in xml_data and "..." not in xml_data


def _collect_relevant_ids(
    model: mujoco.MjModel, ignore_ground: bool
) -> tuple[list[int], list[int], dict[int, tuple[int, int]], dict[int, int]]:
    """Collect body, geom and site ids used for tokenization."""
    if ignore_ground:
        included_body_ids = [i for i in range(model.nbody) if not _is_ground(model, i)]
    else:
        included_body_ids = list(range(model.nbody))

    is_contact_geom = np.logical_and(
        model.geom_contype != 0,
        model.geom_conaffinity != 0,
    )

    body_has_contact_geoms = [
        bodyid
        for bodyid in included_body_ids
        if (
            model.body_geomnum[bodyid] > 0
            and is_contact_geom[
                model.body_geomadr[bodyid] : model.body_geomadr[bodyid]
                + model.body_geomnum[bodyid]
            ].any()
        )
    ]

    included_body_ids = list(
        sorted(set(included_body_ids).intersection(set(body_has_contact_geoms)))
    )

    included_geom_ids = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] in included_body_ids
        and model.geom_contype[i] != 0
        and model.geom_conaffinity[i] != 0
    ]

    if any(
        int(model.geom_type[i]) not in ALLOWED_GEOM_PRIMITIVES
        for i in included_geom_ids
    ):
        raise ValueError("Only sphere, capsule, cylinder, and box geoms are supported")

    is_tracking_site = model.site_group[:] == TRACKING_SITE_GROUP

    if len(set(model.site_bodyid[is_tracking_site])) != is_tracking_site.sum():
        raise ValueError("Only one site per body is supported")
    if not (model.site_pos[is_tracking_site] == 0).all():
        raise ValueError("Site should be at their body's origin")
    if not (model.site_quat[is_tracking_site] == np.array([1, 0, 0, 0])).all():
        raise ValueError("Only unit quaternion is supported")

    body_to_site = {
        int(model.site_bodyid[mj_site_id]): (mj_site_id, track_link_idx)
        for track_link_idx, mj_site_id in enumerate(
            np.arange(model.nsite)[is_tracking_site]
        )
    }
    mjsiteid_to_tracklinkid = {
        int(mj_site_id): track_link_idx
        for track_link_idx, mj_site_id in enumerate(
            np.arange(model.nsite)[is_tracking_site]
        )
    }
    return included_body_ids, included_geom_ids, body_to_site, mjsiteid_to_tracklinkid


def _tokenize_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    body_to_site: dict[int, tuple[int, int]],
    links: list[Link],
    dynamic_joints: list[DynamicJoint],
    fixed_joints: list[FixedJoint],
    mjgeomid_to_linkid: dict[int, int],
    mjjntid_to_dyna_jntid: dict[int, int],
) -> None:
    mj_parent_idx = int(model.body_parentid[body_id])
    body_geom_ids = np.arange(
        model.body_geomadr[body_id],
        model.body_geomadr[body_id] + model.body_geomnum[body_id],
    )
    geom_is_contact = np.logical_and(
        model.geom_contype != 0,
        model.geom_conaffinity != 0,
    )
    contact_geom_ids = body_geom_ids[geom_is_contact[body_geom_ids]]
    assert len(contact_geom_ids) < 2, (
        "call `preprocess_mjcf` to split multi-collision geoms"
    )
    tokenized_link_idx = len(links)
    body_mass = model.body_mass[body_id]
    body_diaginertia = model.body_inertia[body_id]
    all_mj_free_joints = list(
        np.arange(model.njnt)[model.jnt_type == mujoco.mjtJoint.mjJNT_FREE]
    )

    free_link_idx = -1
    T_b2b1 = Pose(
        pos=tuple(model.body_pos[body_id]),
        quat_wxyz=tuple(model.body_quat[body_id]),
    )
    body_to_parent_id = [body_id, mj_parent_idx]

    while (
        not geom_is_contact[
            np.arange(
                model.body_geomadr[mj_parent_idx],
                model.body_geomadr[mj_parent_idx] + model.body_geomnum[mj_parent_idx],
            )
        ].any()
        and mj_parent_idx != 0
    ):
        T_b2b1 = Pose.from_matrix(
            Pose(
                pos=tuple(model.body_pos[mj_parent_idx]),
                quat_wxyz=tuple(model.body_quat[mj_parent_idx]),
            ).matrix
            @ T_b2b1.matrix
        )
        mj_parent_idx = model.body_parentid[mj_parent_idx]
        body_to_parent_id.append(mj_parent_idx)

    if model.body_geomnum[mj_parent_idx] == 0:
        T_b2g2 = Pose(pos=(0, 0, 0), quat_wxyz=(1, 0, 0, 0))
        mj_parent_geom_idx = -1
    else:
        parent_geom_indices = np.arange(
            model.body_geomadr[mj_parent_idx],
            model.body_geomadr[mj_parent_idx] + model.body_geomnum[mj_parent_idx],
        )
        parent_contact_geom_indices = parent_geom_indices[
            model.geom_contype[parent_geom_indices] != 0
        ]
        assert len(parent_contact_geom_indices) == 1, (
            "body {} has {} contact geoms".format(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                len(parent_contact_geom_indices),
            )
        )
        mj_parent_geom_idx = parent_contact_geom_indices[0]
        T_b2g2 = Pose(
            pos=tuple(model.geom_pos[mj_parent_geom_idx]),
            quat_wxyz=tuple(model.geom_quat[mj_parent_geom_idx]),
        )

    for parent_body_id in body_to_parent_id[:-1]:
        jnt_indices = np.arange(
            model.body_jntadr[parent_body_id],
            model.body_jntadr[parent_body_id] + model.body_jntnum[parent_body_id],
        )

        body_free_joints = jnt_indices[
            model.jnt_type[jnt_indices] == mujoco.mjtJoint.mjJNT_FREE
        ]
        assert len(body_free_joints) < 2, "body {} has {} free joints".format(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id),
            len(body_free_joints),
        )
        if len(body_free_joints) == 1:
            mj_free_joint_idx = body_free_joints[0]
            free_link_idx = all_mj_free_joints.index(mj_free_joint_idx)
            break

    # count all joints that belong to the body id chain going from
    # the current body to its parent, ignoring only its parent (last index)
    num_joints = model.body_jntnum[body_to_parent_id[:-1]].sum()

    if len(contact_geom_ids) == 1:
        mj_geom_idx = contact_geom_ids[0]
        mjgeomid_to_linkid[int(mj_geom_idx)] = tokenized_link_idx
        T_b1g1 = Pose(
            pos=tuple(model.geom_pos[mj_geom_idx]),
            quat_wxyz=tuple(model.geom_quat[mj_geom_idx]),
        )
        # Compute inertia pose in geom frame using model-space (body-local) transforms
        # This preserves the local frame relationship correctly, unlike world-space transforms
        # which lose the orientation information when geom and inertia have matching rotations
        T_b1i1 = Pose(
            pos=tuple(model.body_ipos[body_id]),
            quat_wxyz=tuple(model.body_iquat[body_id]),
        )
        # T_g1i1 = inertia in geom frame = T_b1g1^-1 @ T_b1i1
        T_g1i1 = Pose.from_matrix(T_b1g1.inverse.matrix @ T_b1i1.matrix)

        track_link_idx = -1
        if body_id in body_to_site:
            mj_site_id, track_link_idx = body_to_site[body_id]

        links.append(
            Link(
                idx=tokenized_link_idx,
                geom_type=GeomType.from_mujoco(model.geom_type[mj_geom_idx]),
                size=tuple(model.geom_size[mj_geom_idx]),
                track_link_idx=track_link_idx,
                free_link_idx=free_link_idx,
                mass=body_mass,
                ipos=T_g1i1.pos,
                iquat=T_g1i1.quat_wxyz,
                diaginertia=body_diaginertia,
                friction=tuple(model.geom_friction[mj_geom_idx]),
                contact_dim=model.geom_condim[mj_geom_idx],
                rgba=tuple(model.geom_rgba[mj_geom_idx]),
            ),
        )
    elif len(contact_geom_ids) == 0:
        raise ValueError(f"no contact geom found for body {body_id}")
    else:
        raise ValueError("more than one collision geometry")

    if num_joints == 0:
        # fixed joint
        fixed_jnt = _tokenize_fixed_joint(
            T_b2g2=T_b2g2,
            T_b1g1=T_b1g1,
            T_b2b1=T_b2b1,
            jnt_pos=np.zeros(3),
            jnt_axis=np.array([0, 0, 1]),
            prev_link_idx=mjgeomid_to_linkid[int(mj_parent_geom_idx)],
            new_link_idx=tokenized_link_idx,
            new_fixed_jnt_idx=len(fixed_joints),
        )
        fixed_joints.append(fixed_jnt)
        return

    jnt_indices = []
    for body_idx in body_to_parent_id[:-1]:
        jnt_indices.extend(
            range(
                model.body_jntadr[body_idx],
                model.body_jntadr[body_idx] + model.body_jntnum[body_idx],
            )
        )
    for j_idx in jnt_indices:
        jnt_type = model.jnt_type[j_idx]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            # free joints are handled by links
            continue
        jnt_pos = model.jnt_pos[j_idx]
        jnt_axis = model.jnt_axis[j_idx]
        jnt_stiffness = model.jnt_stiffness[j_idx]
        joint_range = (
            float(model.jnt_range[j_idx][0]),
            float(model.jnt_range[j_idx][1]),
        )
        dof_idx = model.jnt_dofadr[j_idx]
        armature = model.dof_armature[dof_idx]
        damping = model.dof_damping[dof_idx]
        frictionloss = model.dof_frictionloss[dof_idx]

        qpos_idx = model.jnt_qposadr[j_idx]
        spring_ref = model.qpos_spring[qpos_idx]
        qpos0_ref = model.qpos0[qpos_idx]

        dyna_jnt = _tokenize_joint(
            T_b2g2=T_b2g2,
            T_b1g1=T_b1g1,
            T_b2b1=T_b2b1,
            jnt_pos=jnt_pos,
            jnt_axis=jnt_axis,
            jnt_type=DynamicJointType.from_mujoco(jnt_type),
            prev_link_idx=mjgeomid_to_linkid[int(mj_parent_geom_idx)],
            new_link_idx=tokenized_link_idx,
            new_dyna_jnt_idx=len(dynamic_joints),
            joint_range=joint_range,
            armature=armature,
            damping=damping,
            frictionloss=frictionloss,
            stiffness=jnt_stiffness,
            spring_ref=spring_ref,
            qpos0_ref=qpos0_ref,
        )
        mjjntid_to_dyna_jntid[j_idx] = len(dynamic_joints)
        dynamic_joints.append(dyna_jnt)


def find_contact_geom_in_body(
    model: mujoco.MjModel,
    body_id: int,
) -> int:
    for geom_idx in range(
        model.body_geomadr[body_id],
        model.body_geomadr[body_id] + model.body_geomnum[body_id],
    ):
        contype = model.geom_contype[geom_idx] != 0
        conaffinity = model.geom_conaffinity[geom_idx] != 0
        is_contact = np.logical_and(contype, conaffinity)
        if is_contact:
            return geom_idx
    return -1


def _tokenize_connect_constraints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dynamic_joints: list[DynamicJoint],
    mjgeomid_to_linkid: dict[int, int],
) -> None:
    """Tokenize equality connect and weld constraints into ball joints."""

    for eq_idx in range(model.neq):
        eq_type = model.eq_type[eq_idx]
        body1 = model.eq_obj1id[eq_idx]
        body2 = model.eq_obj2id[eq_idx]
        assert model.eq_objtype[eq_idx] == mujoco.mjtObj.mjOBJ_BODY, (
            "Only body to body constraints are supported"
        )
        geom1 = find_contact_geom_in_body(model, body1)
        geom2 = find_contact_geom_in_body(model, body2)
        eq_data = model.eq_data[eq_idx]
        assert geom1 != -1, "No contact geom found for body 1"
        assert geom2 != -1, "No contact geom found for body 2"
        if eq_type == mujoco.mjtEq.mjEQ_CONNECT:
            T_Wg1 = Pose(
                pos=tuple(data.geom_xpos[geom1]),
                quat_wxyz=tuple(quaternions.mat2quat(data.geom_xmat[geom1])),
            )
            T_Wb1 = Pose(
                pos=tuple(data.xpos[body1]),
                quat_wxyz=tuple(quaternions.mat2quat(data.xmat[body1])),
            )
            T_Wg2 = Pose(
                pos=tuple(data.geom_xpos[geom2]),
                quat_wxyz=tuple(quaternions.mat2quat(data.geom_xmat[geom2])),
            )

            eq_pos = eq_data[:3]
            T_b1J = Pose(pos=tuple(eq_pos), quat_wxyz=(1, 0, 0, 0))
            T_WJ = T_Wb1.matrix @ T_b1J.matrix
            T_g1W = T_Wg1.inverse.matrix
            T_g2W = T_Wg2.inverse.matrix
            T_g1J = Pose.from_matrix(T_g1W @ T_WJ)
            T_g2J = Pose.from_matrix(T_g2W @ T_WJ)
            dyna_jnt = DynamicJoint(
                idx=len(dynamic_joints),
                joint_type=DynamicJointType.BALL,
                connections=frozenset(
                    [
                        JointLinkConnection(
                            link_idx=mjgeomid_to_linkid[int(geom1)],
                            pose=T_g1J.inverse,
                        ),
                        JointLinkConnection(
                            link_idx=mjgeomid_to_linkid[int(geom2)],
                            pose=T_g2J.inverse,
                        ),
                    ]
                ),
                joint_range=(0, 0),
                armature=0.0,
                damping=0.0,
                frictionloss=0.0,
                stiffness=0.0,
                spring_ref=0.0,
                qpos0_ref=0.0,
            )
        elif eq_type == mujoco.mjtEq.mjEQ_WELD:
            assert (
                eq_data
                == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            ).all(), "Weld equality offset is not supported"

            # The weld constraint is between body frames, not geom centers.
            # The convention in JointLinkConnection is that pose is T_Jg (pose of geom
            # in joint frame). For a weld, the "joint" is at the body frame.
            # T_bg is the pose of geom in body frame, so T_Jg = T_bg.
            T_Jg1 = Pose(
                pos=tuple(model.geom_pos[geom1]),
                quat_wxyz=tuple(model.geom_quat[geom1]),
            )
            T_Jg2 = Pose(
                pos=tuple(model.geom_pos[geom2]),
                quat_wxyz=tuple(model.geom_quat[geom2]),
            )

            dyna_jnt = DynamicJoint(
                idx=len(dynamic_joints),
                joint_type=DynamicJointType.BALL,
                connections=frozenset(
                    [
                        JointLinkConnection(
                            link_idx=mjgeomid_to_linkid[int(geom1)],
                            pose=T_Jg1,
                        ),
                        JointLinkConnection(
                            link_idx=mjgeomid_to_linkid[int(geom2)],
                            pose=T_Jg2,
                        ),
                    ]
                ),
                joint_range=(0, 0),
                armature=0.0,
                damping=0.0,
                frictionloss=0.0,
                stiffness=0.0,
                spring_ref=0.0,
                qpos0_ref=0.0,
            )
        else:
            raise ValueError("Only ball and weld connect constraints are supported")

        dynamic_joints.append(dyna_jnt)


def _tokenize_actuators(
    model: mujoco.MjModel, mjjntid_to_dyna_jntid: dict[int, int]
) -> list[Actuator]:
    """Tokenize actuators attached to the model."""

    actuators = []
    assert set(model.actuator_dyntype[:]).issubset(
        {
            mujoco.mjtDyn.mjDYN_NONE,
            mujoco.mjtDyn.mjDYN_FILTEREXACT,
        }
    )
    assert (model.actuator_gaintype[:] == mujoco.mjtGain.mjGAIN_FIXED).all()
    assert (model.actuator_biastype[:] == mujoco.mjtBias.mjBIAS_AFFINE).all()
    assert (model.actuator_gainprm[:, 1:] == 0).all()
    assert (model.actuator_biasprm[:, 0] == 0).all()
    for act_idx in range(model.nu):
        ctrlrange = (
            float(model.actuator_ctrlrange[act_idx][0]),
            float(model.actuator_ctrlrange[act_idx][1]),
        )
        force_range = (
            float(model.actuator_forcerange[act_idx][0]),
            float(model.actuator_forcerange[act_idx][1]),
        )
        is_position_controlled = (
            model.actuator_gainprm[act_idx, 0] == -model.actuator_biasprm[act_idx, 1]
        )
        is_velocity_controlled = (
            model.actuator_gainprm[act_idx, 0] == -model.actuator_biasprm[act_idx, 2]
        )
        assert is_position_controlled or is_velocity_controlled
        actuator_type = (
            ActuatorType.POSITION if is_position_controlled else ActuatorType.VELOCITY
        )
        kp = -model.actuator_biasprm[act_idx, 1]
        kv = -model.actuator_biasprm[act_idx, 2]
        mj_jnt_idx = model.actuator_trnid[act_idx, 0]
        dyna_jnt_idx = mjjntid_to_dyna_jntid[mj_jnt_idx]

        actuators.append(
            Actuator(
                idx=act_idx,
                dyna_joint_idx=dyna_jnt_idx,
                ctrl_range=ctrlrange,
                force_range=force_range,
                kp=kp,
                kv=kv,
                actuator_type=actuator_type,
                gear_ratio=model.actuator_gear[act_idx, 0],
            )
        )
    return actuators


def tokenize(
    mj_robot: mjcf.RootElement,
    ignore_ground: bool = True,
) -> tuple[Robot, dict[int, int], dict[int, int], dict[int, int]]:
    # assert is_empty_xml_node(mj_robot.contact)
    assert _is_empty_xml_node(mj_robot.tendon)
    # assert is_empty_xml_node(mj_robot.sensor)
    # assert is_empty_xml_node(mj_robot.custom)
    assert _is_empty_xml_node(mj_robot.extension)
    physics = mjcf.Physics.from_mjcf_model(mj_robot)
    # this reset is super important.
    # first, it computes forward kinematics, and the resulting poses are used
    # in parsing the kinematic structure of the robot.
    # second, qpos0 is used in calculating the equality constraints.
    model = physics.model.ptr
    data = physics.data.ptr
    physics.reset(0 if model.nkey > 0 else None)
    data.qpos[:] = model.qpos0[:]
    physics.forward()
    links: list[Link] = []
    dynamic_joints: list[DynamicJoint] = []
    fixed_joints: list[FixedJoint] = []

    mjgeomid_to_linkid = {-1: -1}
    mjjntid_to_dyna_jntid = {}
    # MuJoCo doesn't represent fixed joints, so don't need mapping for them

    included_body_ids, _, body_to_site, mjsiteid_to_tracklinkid = _collect_relevant_ids(
        model, ignore_ground
    )
    for body_id in included_body_ids:
        _tokenize_body(
            model=model,
            data=data,
            body_id=body_id,
            body_to_site=body_to_site,
            links=links,
            dynamic_joints=dynamic_joints,
            fixed_joints=fixed_joints,
            mjgeomid_to_linkid=mjgeomid_to_linkid,
            mjjntid_to_dyna_jntid=mjjntid_to_dyna_jntid,
        )

    _tokenize_connect_constraints(model, data, dynamic_joints, mjgeomid_to_linkid)
    actuators = _tokenize_actuators(model, mjjntid_to_dyna_jntid)
    free_link_obs = []
    dyna_jnt_obs = []
    physics.reset(0 if model.nkey > 0 else None)
    free_links = [link for link in links if link.free_link_idx != -1]
    if len(free_links) > 0:
        assert len(free_links) == 1, "at most one free link is supported"
        free_link = free_links[0]
        free_mj_geom_id = next(
            k for k, v in mjgeomid_to_linkid.items() if v == free_link.idx
        )
        # By convention, FreeLinkObs records the world pose of the link's
        # GEOM frame, not its body frame — the detokenized model places the
        # geom at the body origin, so the body frame coincides with the geom
        # frame in the round-trip. See FreeLinkObs.__doc__.
        free_geom_pos = data.geom_xpos[free_mj_geom_id]
        free_geom_quat = quaternions.mat2quat(data.geom_xmat[free_mj_geom_id])
        free_link_obs.append(
            FreeLinkObs(
                free_link_idx=free_link.idx,
                time_idx=0,
                pos=tuple(free_geom_pos),
                quat_wxyz=tuple(free_geom_quat),
            )
        )

    for dyna_jnt in dynamic_joints:
        if dyna_jnt.joint_type == DynamicJointType.BALL:
            continue
        mj_jnt_idx = next(
            k for k, v in mjjntid_to_dyna_jntid.items() if v == dyna_jnt.idx
        )
        mj_jnt = model.joint(mj_jnt_idx)
        dyna_jnt_obs.append(
            DynamicJointState(
                dyna_jnt_idx=dyna_jnt.idx,
                time_idx=0,
                qpos=float(data.qpos[mj_jnt.qposadr][0]),
                qvel=float(data.qvel[mj_jnt.dofadr][0]),
            )
        )

    return (
        Robot(
            links=frozenset(links),
            dynamic_joints=frozenset(dynamic_joints),
            fixed_joints=frozenset(fixed_joints),
            actuators=frozenset(actuators),
            free_link_obs=frozenset(free_link_obs),
            dyna_jnt_obs=frozenset(dyna_jnt_obs),
        ),
        mjjntid_to_dyna_jntid,
        mjgeomid_to_linkid,
        mjsiteid_to_tracklinkid,
    )


def _topological_sort(
    links: list[Link],
    dynamic_joints: list[DynamicJoint],
    fixed_joints: list[FixedJoint],
    root_node: int = -1,
    allow_cycles: bool = True,
) -> list[int]:
    # Create adjacency graph where each edge points away from root
    graph = {root_node: set()}
    active_nodes = {root_node}
    # all free joints are connected to the root node
    for link in links:
        if link.free_link_idx != -1:
            graph[root_node].add(link.idx)
            graph[link.idx] = set()
            active_nodes.add(link.idx)
    processed_dyna_jnts = set()
    processed_fixed_jnts = set()
    while len(active_nodes) > 0:
        node_idx = active_nodes.pop()
        for jnt in [*dynamic_joints, *fixed_joints]:
            if type(jnt) is DynamicJoint and jnt.idx in processed_dyna_jnts:
                continue
            elif type(jnt) is FixedJoint and jnt.idx in processed_fixed_jnts:
                continue
            jnt_conn = {conn.link_idx for conn in jnt.connections}
            if node_idx in jnt_conn:
                other_node_indices = list(jnt_conn - {node_idx})
                if len(other_node_indices) != 1:
                    raise ValueError("Joint has only one connection")
                other_node_idx = other_node_indices[0]
                graph[node_idx].add(other_node_idx)
                if type(jnt) is DynamicJoint:
                    processed_dyna_jnts.add(jnt.idx)
                elif type(jnt) is FixedJoint:
                    processed_fixed_jnts.add(jnt.idx)
                else:
                    raise ValueError(f"Invalid joint type: {type(jnt)}")
                if other_node_idx not in graph:
                    active_nodes.add(other_node_idx)
                    graph[other_node_idx] = set()
    missing_dyna_jnts = set(jnt.idx for jnt in dynamic_joints) - processed_dyna_jnts
    if len(missing_dyna_jnts) > 0:
        raise DetokenizeError(
            f"Missing dynamic joints from topological sort: {missing_dyna_jnts}"
        )
    missing_fixed_jnts = set(jnt.idx for jnt in fixed_joints) - processed_fixed_jnts
    if len(missing_fixed_jnts) > 0:
        raise DetokenizeError(
            f"Missing fixed joints from topological sort: {missing_fixed_jnts}"
        )
    # The BFS above only adds edges in the direction of traversal, so the
    # resulting `graph` is always a DAG. The legacy cycle-handling block here
    # used to assume "both edges are ball joints" without checking, and was
    # in fact unreachable; the actual cycle resolution for parallel kinematic
    # chains lives in `detokenize` (the `num_joints == 2` weld-equality path,
    # which correctly enforces that at least one joint is a ball joint).
    # If a future BFS rewrite produces a real directed cycle, surface it as
    # an explicit error rather than silently dropping an arbitrary edge.
    try:
        ts = TopologicalSorter(graph)
        return list(ts.static_order())
    except CycleError as e:
        raise DetokenizeError(
            f"Topological sort produced a cyclic graph: {e}. "
            "This should not happen with the current BFS construction; "
            "if you see this, the BFS has been changed in a way that makes "
            "cycle handling necessary again — re-introduce a ball-joint "
            "guard before removing edges."
        )


def detokenize(
    tokenized_robot: Robot,
    gravcomp: bool,
) -> tuple[mjcf.RootElement, dict[int, int], dict[int, int], dict[int, int]]:
    root = default_root_element()
    sorted_link_indices = _topological_sort(
        dynamic_joints=list(tokenized_robot.dynamic_joints),
        fixed_joints=list(tokenized_robot.fixed_joints),
        links=list(tokenized_robot.links),
    )

    links = {link.idx: link for link in tokenized_robot.links}
    missing_link_ids = set(links.keys()) - set(sorted_link_indices)
    if len(missing_link_ids) > 0:
        raise DetokenizeError(
            f"Missing link ids from topological sort: {missing_link_ids}"
        )
    dynamic_joints = {jnt.idx: jnt for jnt in tokenized_robot.dynamic_joints}
    fixed_joints = {jnt.idx: jnt for jnt in tokenized_robot.fixed_joints}
    actuators = {act.idx: act for act in tokenized_robot.actuators}
    processed_dyna_jnts = set()

    linkid_to_node = {GROUND_LINK_ID: root.worldbody}

    for link_idx in reversed(sorted_link_indices):
        if link_idx == GROUND_LINK_ID:
            continue
        if link_idx not in links:
            raise DetokenizeError("Link idx not found")
        link = links[link_idx]
        if link.free_link_idx != -1:
            T_g2g1 = Pose(pos=(0, 0, 0), quat_wxyz=(1, 0, 0, 0))
            jnt_pos = np.array([0, 0, 0])
            jnt_axis = np.array([0, 0, 1])
            parent_body = linkid_to_node[GROUND_LINK_ID]
        else:
            dyna_jnts = [
                j
                for j in dynamic_joints.values()
                if any(
                    {link_idx, potential_parent}
                    == {conn.link_idx for conn in j.connections}
                    for potential_parent in linkid_to_node
                )
                and j.idx not in processed_dyna_jnts
            ]
            fixed_jnts = [
                j
                for j in fixed_joints.values()
                if any(
                    {link_idx, potential_parent}
                    == {conn.link_idx for conn in j.connections}
                    for potential_parent in linkid_to_node
                )
            ]

            num_joints = len(dyna_jnts) + len(fixed_jnts)

            if num_joints == 0:
                raise DetokenizeError("No joint found for link")
            if num_joints == 2:
                if DynamicJointType.BALL not in {j.joint_type for j in dyna_jnts}:
                    raise DetokenizeError(
                        "Found cyclic kinematic chain without ball joint"
                    )
                dyna_jnts = list(
                    sorted(
                        dyna_jnts,
                        key=lambda j: int(j.joint_type != DynamicJointType.BALL),
                    )
                )
                # get any ball joint out
                ball_joint = dyna_jnts[0]
                # add this ball joint as a weld equality constraint with relpose
                connections = list(ball_joint.connections)
                c1 = connections[0]
                c2 = connections[1]

                # Get the pose transforms from the joint connections
                T_Jg1 = c1.pose  # pose of geom1 in joint frame
                T_Jg2 = c2.pose  # pose of geom2 in joint frame

                # Compute relative pose from body1 to body2: T_g1g2 = T_g1J @ T_Jg2
                T_g1g2 = Pose.from_matrix(T_Jg1.inverse.matrix @ T_Jg2.matrix)

                # Create weld constraint with relpose
                # relpose format: [x, y, z, qw, qx, qy, qz]
                relpose = list(T_g1g2.pos) + list(T_g1g2.quat_wxyz)

                root.equality.add(
                    "weld",
                    body1=f"link_{c1.link_idx}",
                    body2=f"link_{c2.link_idx}",
                    relpose=relpose,
                    torquescale=0.0,  # torquescale=0 makes weld behave like connect
                )
                processed_dyna_jnts.add(ball_joint.idx)
                dyna_jnts = dyna_jnts[1:]

            jnts = dyna_jnts + fixed_jnts
            if len(jnts) == 0:
                raise DetokenizeError("No joint found for link")

            parent_indices = {
                next(
                    conn.link_idx
                    for conn in jnt.connections
                    if conn.link_idx != link_idx
                )
                for jnt in jnts
            }
            if len(parent_indices) != 1:
                raise DetokenizeError("Multiple parents found for link")
            parent_idx = next(iter(parent_indices))
            parent_body = linkid_to_node[parent_idx]

            base_jnt = dyna_jnts[0] if dyna_jnts else fixed_jnts[0]
            T_g2g1, jnt_pos, jnt_axis = _detokenize_joint(
                jnt=base_jnt,
                child_idx=link_idx,
            )

        child_body = parent_body.add(
            "body",
            name=f"link_{link_idx}",
            pos=T_g2g1.pos,
            quat=T_g2g1.quat_wxyz,
            gravcomp=gravcomp,
        )
        if link.free_link_idx != -1:
            child_body.add(
                "freejoint",
                name=f"free_joint_{link.free_link_idx}",
            )

        linkid_to_node[link_idx] = child_body
        if link.track_link_idx != -1:
            child_body.add(
                "site",
                name=f"site_{link.track_link_idx}",
                group=TRACKING_SITE_GROUP,
            )
            child_body.add(
                "geom",
                name=f"geom_{link_idx}",
                type="sphere",
                size=[0.0001],
                mass=0.0001,
                group=3,  # by mujoco convention, collision geoms are group 3
                contype=1,
                conaffinity=1,
            )
        else:
            child_body.add(
                "geom",
                name=f"geom_{link_idx}",
                type=link.geom_type.value,
                size=(link.size[0], link.size[1], link.size[2]),
                group="3",  # by mujoco convention, collision geoms are group 3
                condim=link.contact_dim,
                friction=link.friction,
                rgba=link.rgba,
            )
            child_body.add(
                "inertial",
                mass=link.mass,
                pos=link.ipos,
                quat=link.iquat,
                diaginertia=link.diaginertia,
            )

        if link.free_link_idx != -1:
            continue

        for jnt in dyna_jnts:
            _, jnt_pos, jnt_axis = _detokenize_joint(
                jnt=jnt,
                child_idx=link_idx,
            )
            child_body.add(
                "joint",
                name=f"dyna_joint_{jnt.idx}",
                axis=jnt_axis,
                pos=jnt_pos,
                type=jnt.joint_type.value,
                range=(jnt.joint_range[0], jnt.joint_range[1]),
                armature=jnt.armature,
                damping=jnt.damping,
                frictionloss=jnt.frictionloss,
                stiffness=jnt.stiffness,
                springref=jnt.spring_ref,
                ref=jnt.qpos0_ref,
                actuatorgravcomp=True,
            )
            processed_dyna_jnts.add(jnt.idx)

    if processed_dyna_jnts != set(dynamic_joints.keys()):
        raise DetokenizeError(
            f"Processed dynamic joints {processed_dyna_jnts} do not match dynamic joints {set(dynamic_joints.keys())}"
        )

    model = mujoco.MjModel.from_xml_string(root.to_xml_string())

    dyna_jntid_to_mjjntid = {}

    linkid_to_mjgeomid = {}
    for mj_geomid in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, mj_geomid)
        if "geom_" in geom_name:
            linkid_to_mjgeomid[int(geom_name.split("geom_")[1])] = mj_geomid

    tracklinkid_to_mjsiteid = {}
    for mj_siteid in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, mj_siteid)
        if site_name is not None and site_name.startswith("site_"):
            track_link_idx = int(site_name.split("site_")[1])
            tracklinkid_to_mjsiteid[track_link_idx] = mj_siteid

    for mj_jntid in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, mj_jntid)
        if "dyna_joint_" in jnt_name:
            dyna_jnt_idx = int(jnt_name.split("dyna_joint_")[1])
            dyna_jntid_to_mjjntid[dyna_jnt_idx] = mj_jntid
        elif "free_joint_" in jnt_name:
            continue
        else:
            raise DetokenizeError(f"Invalid joint type: {jnt_name}")

    for act_idx, act in sorted(actuators.items(), key=lambda x: x[0]):
        if act.dyna_joint_idx not in dyna_jntid_to_mjjntid:
            raise DetokenizeError("Dynamic joint idx not found")
        if (np.abs(act.force_range) < 1e-1).all():  # TODO expose this hardcode
            forcelimited = False
            force_range = None
        else:
            force_range = (act.force_range[0], act.force_range[1])
            forcelimited = True

        if (np.abs(act.ctrl_range) < 1e-1).all():  # TODO expose this hardcode
            ctrllimited = False
            ctrl_range = None
        else:
            ctrl_range = (act.ctrl_range[0], act.ctrl_range[1])
            ctrllimited = True
        if act.actuator_type == ActuatorType.POSITION:
            root.actuator.add(
                "general",
                name=f"actuator_{act_idx}",
                joint=f"dyna_joint_{act.dyna_joint_idx}",
                ctrllimited=ctrllimited,
                ctrlrange=ctrl_range,
                forcelimited=forcelimited,
                forcerange=force_range,
                gaintype="fixed",
                biastype="affine",
                gainprm=(act.kp, 0, 0),
                biasprm=(0, -act.kp, -act.kv),
                gear=[act.gear_ratio] + [0] * 5,
            )
        elif act.actuator_type == ActuatorType.VELOCITY:
            root.actuator.add(
                "general",
                name=f"actuator_{act_idx}",
                joint=f"dyna_joint_{act.dyna_joint_idx}",
                ctrlrange=ctrl_range,
                forcerange=force_range,
                gaintype="fixed",
                biastype="affine",
                gainprm=(act.kv, 0, 0),
                biasprm=(0, 0, -act.kv),
                gear=[act.gear_ratio] + [0] * 5,
            )

    free_link_obs_time_dict = {
        obs.time_idx: obs for obs in tokenized_robot.free_link_obs
    }
    dyna_jnt_obs_time_dict = {}
    for obs in tokenized_robot.dyna_jnt_obs:
        if obs.time_idx not in dyna_jnt_obs_time_dict:
            dyna_jnt_obs_time_dict[obs.time_idx] = []
        dyna_jnt_obs_time_dict[obs.time_idx].append(obs)
    time_indices = sorted(
        set(free_link_obs_time_dict.keys()).union(dyna_jnt_obs_time_dict.keys())
    )

    __physics = mjcf.Physics.from_mjcf_model(root)
    for time_idx in time_indices:
        keyframe = np.zeros(model.nq)
        has_free_link_keyframe = time_idx in free_link_obs_time_dict

        if has_free_link_keyframe:
            free_link_obs = free_link_obs_time_dict[time_idx]
            # FreeLinkObs.pos / .quat_wxyz are the original GEOM world pose
            # (see FreeLinkObs docstring). In the detokenized model the geom
            # sits at the body origin, so writing the geom world pose into
            # the freejoint qpos slots places both the new body and its
            # geom at the original geom's world pose. The original BODY
            # world pose is intentionally not preserved.
            keyframe[:3] = free_link_obs.pos
            keyframe[3:7] = free_link_obs.quat_wxyz
        has_dynamic_joint_keyframe = time_idx in dyna_jnt_obs_time_dict
        if has_dynamic_joint_keyframe:
            dyna_jnt_obs_list = dyna_jnt_obs_time_dict[time_idx]
            for jnt_state in dyna_jnt_obs_list:
                global WARNED_ABOUT_DYNA_JOINT_ERRORS
                try:
                    mj_joint_name = "dyna_joint_" + str(jnt_state.dyna_jnt_idx)
                    mj_jnt = model.joint(mj_joint_name)
                    keyframe[mj_jnt.qposadr] = jnt_state.qpos
                except Exception as e:
                    if not WARNED_ABOUT_DYNA_JOINT_ERRORS:
                        logging.warning(f"detokenize: {e}")
                        WARNED_ABOUT_DYNA_JOINT_ERRORS = True
                    continue

        root.keyframe.add(
            "key",
            name=f"time_{time_idx}",
            qpos=keyframe,
        )

    return (
        root,
        linkid_to_mjgeomid,
        dyna_jntid_to_mjjntid,
        tracklinkid_to_mjsiteid,
    )
