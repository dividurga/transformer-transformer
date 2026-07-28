import copy
import typing
from typing import Optional

import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import MjvOption


def default_root_element(name: str | None = None) -> mjcf.RootElement:
    root = mjcf.RootElement(name)
    root.compiler.angle = "radian"
    root.compiler.autolimits = True

    root.compiler.balanceinertia = True

    # NOTE: setting these didn't help reduce memory usage
    # root.size.nconmax = 100
    # root.size.memory = "10K"
    # getattr(root.visual, "global").offwidth = 64
    # getattr(root.visual, "global").offheight = 64

    collision_class = root.default.add("default", dclass="collision")
    collision_class.geom.contype = 1
    collision_class.geom.conaffinity = 1
    collision_class.geom.group = 3

    return root


def render_opt(opt: Optional[MjvOption] = None):
    if opt is None:
        opt = MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[2] = 1
    opt.sitegroup[:] = 0
    opt.sitegroup[TARGET_SITE_GROUP] = 1
    opt.sitegroup[TRACKING_SITE_GROUP] = 1
    opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    return opt


ALLOWED_GEOM_PRIMITIVES = {
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_BOX,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
}


TRACKING_SITE_GROUP = 1
TARGET_SITE_GROUP = 2


def set_up_default_scene(
    root: mjcf.RootElement,
    add_plane: bool = True,
    add_vis_cam: bool = False,
    plane_z_pos: float = -0.2,
    vis_cam_pos: tuple[float, float, float] = (2.765, 1.761, 0.529),
    vis_cam_xyaxes: tuple[float, float, float, float, float, float] = (
        -0.561,
        0.828,
        0.000,
        0.027,
        0.019,
        0.999,
    ),
):
    root.worldbody.add("light", pos=[0, 0, 3], dir=[0, 0, -1])

    # add assets
    root.asset.add(
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        width="512",
        height="3072",
    )
    root.asset.add(
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        mark="edge",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        markrgb="0.8 0.8 0.8",
        width="300",
        height="300",
    )
    root.asset.add(
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform="true",
        texrepeat="5 5",
        reflectance="0.2",
    )

    if add_plane:
        root.worldbody.add(
            "geom",
            name="floor",
            type="plane",
            size=[0, 0, 0.125],
            material="groundplane",
            pos=[0, 0, plane_z_pos],
            friction="0.8",
            margin="0.001",
            condim="3",
        )

    if add_vis_cam:
        root.worldbody.add(
            "camera",
            name="vis_cam",
            # pos="-0.522 0.892 0.191",
            # xyaxes="-0.796 -0.606 -0.000 0.090 -0.118 0.989",
            pos=vis_cam_pos,
            # pos="1.365 0.861 0.329",
            xyaxes=vis_cam_xyaxes,
        )

    return root


def add_mocap_body_with_site(
    root: mjcf.RootElement,
    name: str,
    site_group: int = 5,
    add_vis_cam: bool = False,
) -> mjcf.RootElement:
    body = root.worldbody.add("body", name=name, mocap=True)
    site = body.add("site", name=f"{name}_site", group=site_group)
    if add_vis_cam:
        body.add(
            "camera",
            name="vis_cam",
            pos="-0.522 0.892 0.191",
            xyaxes="-0.796 -0.606 -0.000 0.090 -0.118 0.989",
        )
    return root


## Inverse Kinematics from dm_control, modified to handle joint limits


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    """Calculates the joint velocities to achieve a specified end effector delta.

    Args:
      jac_joints: The Jacobian of the end effector with respect to the joints. A
        numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
        and `nv` is the number of degrees of freedom.
      delta: The desired end-effector delta. A numpy array of shape `(3,)` or
        `(6,)` containing either position deltas, rotation deltas, or both.
      regularization_strength: (optional) Coefficient of the quadratic penalty
        on joint movements. Default is zero, i.e. no regularization.

    Returns:
      An `(nv,)` numpy array of joint velocities.

    Reference:
      Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
      transpose, pseudoinverse and damped least squares methods.
      https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]


#
def qpos_from_site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    dof_indices: np.ndarray | list[int],
    target_pos: Optional[np.ndarray] = None,
    target_quat: Optional[np.ndarray] = None,
    tol: float = 1e-14,
    rot_weight: float = 1.0,
    regularization_threshold: float = 0.1,
    regularization_strength: float = 3e-2,
    max_update_norm: float = 2.0,
    progress_thresh: float = 20.0,
    cond_num_threshold: float = 1e2,
    max_steps: int = 100,
    inplace: bool = False,
    clip_joint_limits: bool = True,
):
    dtype = data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError()

    update_nv = np.zeros(model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    if not inplace:
        data = copy.deepcopy(data)

    # Ensure that the Cartesian position of the site is up to date.
    mujoco.mj_fwdPosition(model, data)

    # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
    # update them in place, so we can avoid indexing overhead in the main loop.
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    site_xpos = data.site_xpos[site_id]
    site_xmat = data.site_xmat[site_id]

    steps = 0
    success = False

    for steps in range(max_steps):
        err_norm = 0.0

        if target_pos is not None:
            # Translational error.
            err_pos[:] = target_pos - site_xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # Rotational error.
            mujoco.mju_mat2Quat(site_xquat, site_xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            success = True
            break
        else:
            # TODO(b/112141670): Generalize this to other entities besides sites.
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]

            # TODO(b/112141592): This does not take joint limits into consideration.
            reg_strength = (
                regularization_strength if err_norm > regularization_threshold else 0.0
            )
            condition_number = np.linalg.cond(jac_joints)

            if condition_number > cond_num_threshold:
                update_joints = nullspace_method(
                    jac_joints,
                    err,
                    regularization_strength=reg_strength
                    * condition_number
                    / cond_num_threshold,
                )
            else:
                update_joints = nullspace_method(
                    jac_joints,
                    err,
                    regularization_strength=reg_strength,
                )

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            # Write the entries for the specified joints into the full `update_nv`
            # vector.
            update_nv[dof_indices] = update_joints

            # Update `physics.qpos`, taking quaternions into account.
            mujoco.mj_integratePos(model, data.qpos, update_nv, 1)

            if clip_joint_limits:
                np.clip(
                    data.qpos,
                    model.jnt_range[:, 0],
                    model.jnt_range[:, 1],
                    out=data.qpos,
                )

            # Compute the new Cartesian position of the site.
            mujoco.mj_fwdPosition(model, data)

    if not inplace:
        # Our temporary copy ofdata is about to go out of scope, and when
        # it does the underlying mjData pointer will be freed anddata.qpos
        # will be a view onto a block of deallocated memory. We therefore need to
        # make a copy ofdata.qpos whiledata is still alive.
        qpos = data.qpos.copy()
    else:
        # If we're modifyingdata in place then it's fine to return a view.
        qpos = data.qpos
    return qpos, err_norm, steps, success


#
def repeated_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    dof_indices: np.ndarray | list[int],
    target_pos: Optional[np.ndarray] = None,
    target_quat: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_steps: int = 10,
    max_attempts: int = 1,
    inplace: bool = True,
    gamma: float = 1.0,
    rs: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    init_qpos = data.qpos.copy()
    ik_attempt = 0
    if rs is None:
        rs = typing.cast(np.random.RandomState, np.random)
    success = False

    err_to_joint = {}

    while not success and ik_attempt < max_attempts:
        target_q, err_norm, steps, success = qpos_from_site_pose(
            model=model,
            data=data,
            site_name=site_name,
            dof_indices=dof_indices,
            target_quat=target_quat,
            target_pos=target_pos,
            tol=tol,
            max_steps=max_steps,
            inplace=inplace,
        )
        err_to_joint[err_norm] = target_q
        if success:
            break
        elif ik_attempt < max_attempts - 1:
            data.qpos[:] = rs.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        ik_attempt += 1
    if not success:
        target_q = err_to_joint[min(err_to_joint.keys())]
        target_q = init_qpos * (1 - gamma) + target_q * gamma
    return target_q


def find_rigidly_attached_descendant_with_contact_geom(
    mj_model: mujoco.MjModel,
    target_body_id: int,
) -> int:
    body_contact_geomnum = []
    body_contact_geom_ids = []

    for mj_body_id in range(mj_model.nbody):
        geomadr = mj_model.body_geomadr[mj_body_id]
        geomnum = mj_model.body_geomnum[mj_body_id]
        geom_ids = np.arange(geomadr, geomadr + geomnum)
        is_contact_geom = np.logical_and(
            mj_model.geom_contype[geom_ids] != 0,
            mj_model.geom_conaffinity[geom_ids] != 0,
        )
        body_contact_geomnum.append(int(is_contact_geom.sum()))
        body_contact_geom_ids.append(geom_ids[is_contact_geom])
    if body_contact_geomnum[target_body_id] > 0:
        return target_body_id  # self is the descendant with contact geom
    body_nonfree_jntnum = []
    for mj_body_id in range(mj_model.nbody):
        jnt_ids = np.arange(
            mj_model.body_jntadr[mj_body_id],
            mj_model.body_jntadr[mj_body_id] + mj_model.body_jntnum[mj_body_id],
        )
        is_nonfree_jnt = mj_model.jnt_type[jnt_ids] != mujoco.mjtJoint.mjJNT_FREE
        body_nonfree_jntnum.append(int(is_nonfree_jnt.sum()))
    # do DFS to find the descendant with contact geom
    visited = set()
    stack = [target_body_id]
    while stack:
        mj_body_id = stack.pop()
        if mj_body_id in visited:
            continue
        visited.add(mj_body_id)
        if body_contact_geomnum[mj_body_id] > 0:
            return mj_body_id
        for other_body_id in range(mj_model.nbody):
            if (
                mj_model.body_parentid[other_body_id] == mj_body_id
                and body_nonfree_jntnum[other_body_id] == 0
            ):
                stack.append(other_body_id)
    return -1
