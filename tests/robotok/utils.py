import copy
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper import MjvOption
from mujoco import viewer
from transforms3d import affines

from t2.env.mj_utils import TRACKING_SITE_GROUP, set_up_default_scene
from t2.env.transforms import mat_norm
from t2.robotok.io import deserialize, serialize
from t2.robotok.tokenizer import detokenize, preprocess_mjcf, tokenize

"""
TODO also support collision exclusion
  // predefined geom pairs for collision detection; has precedence over exclude
  int*      pair_dim;             // contact dimensionality                   (npair x 1)
  int*      pair_geom1;           // id of geom1                              (npair x 1)
  int*      pair_geom2;           // id of geom2                              (npair x 1)
  int*      pair_signature;       // body1 << 16 + body2                      (npair x 1)
  mjtNum*   pair_solref;          // solver reference: contact normal         (npair x mjNREF)
  mjtNum*   pair_solreffriction;  // solver reference: contact friction       (npair x mjNREF)
  mjtNum*   pair_solimp;          // solver impedance: contact                (npair x mjNIMP)
  mjtNum*   pair_margin;          // detect contact if dist<margin            (npair x 1)
  mjtNum*   pair_gap;             // include in solver if dist<margin-gap     (npair x 1)
  mjtNum*   pair_friction;        // tangent1, 2, spin, roll1, 2              (npair x 5)
TODO also support equality constraints
  // equality constraints
  int*      eq_type;              // constraint type (mjtEq)                  (neq x 1)
  int*      eq_obj1id;            // id of object 1                           (neq x 1)
  int*      eq_obj2id;            // id of object 2                           (neq x 1)
  int*      eq_objtype;           // type of both objects (mjtObj)            (neq x 1)
  mjtByte*  eq_active0;           // initial enable/disable constraint state  (neq x 1)
  mjtNum*   eq_solref;            // constraint solver reference              (neq x mjNREF)
  mjtNum*   eq_solimp;            // constraint solver impedance              (neq x mjNIMP)
  mjtNum*   eq_data;              // numeric data for constraint              (neq x mjNEQDATA)
TODO also compare keyframes
  mjtNum*   key_time;             // key time                                 (nkey x 1)
  mjtNum*   key_qpos;             // key position                             (nkey x nq)
  mjtNum*   key_qvel;             // key velocity                             (nkey x nv)
  mjtNum*   key_act;              // key activation                           (nkey x na)
  mjtNum*   key_mpos;             // key mocap position                       (nkey x nmocap*3)
  mjtNum*   key_mquat;            // key mocap quaternion                     (nkey x nmocap*4)
  mjtNum*   key_ctrl;             // key control                              (nkey x nu)
"""


def render_comparison(
    physics_1: mjcf.Physics, physics_2: mjcf.Physics, camera_id: int = -1
):
    physics_1.reset(0 if physics_1.model.nkey > 0 else None)
    physics_1.forward()
    physics_2.reset(0 if physics_2.model.nkey > 0 else None)
    physics_2.forward()
    fig, axes = plt.subplots(5, 2, figsize=(5, 12))
    scene_option = MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 1
    # render body frame
    scene_option.frame = mujoco.mjtFrame.mjFRAME_BODY
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    img1 = physics_1.render(camera_id=camera_id, scene_option=scene_option)
    img2 = physics_2.render(camera_id=camera_id, scene_option=scene_option)
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("original (body)")
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("detokenized (body)")
    scene_option = MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 1
    # render geom frame
    scene_option.frame = mujoco.mjtFrame.mjFRAME_GEOM
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    img1 = physics_1.render(camera_id=camera_id, scene_option=scene_option)
    img2 = physics_2.render(camera_id=camera_id, scene_option=scene_option)
    axes[1, 0].imshow(img1)
    axes[1, 0].set_title("original (geom)")
    axes[1, 1].imshow(img2)
    axes[1, 1].set_title("detokenized (geom)")
    scene_option = MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 1
    scene_option.sitegroup[:] = 1
    # render actuator frame
    scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    img1 = physics_1.render(camera_id=camera_id, scene_option=scene_option)
    img2 = physics_2.render(camera_id=camera_id, scene_option=scene_option)
    axes[2, 0].imshow(img1)
    axes[2, 0].set_title("original (actuator)")
    axes[2, 1].imshow(img2)
    axes[2, 1].set_title("detokenized (actuator)")
    # render com
    scene_option = MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 1
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    img1 = physics_1.render(camera_id=camera_id, scene_option=scene_option)
    img2 = physics_2.render(camera_id=camera_id, scene_option=scene_option)
    axes[3, 0].imshow(img1)
    axes[3, 0].set_title("original (com)")
    axes[3, 1].imshow(img2)
    axes[3, 1].set_title("detokenized (com)")
    # render inertia
    scene_option = MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 0
    scene_option.geomgroup[2] = 0
    scene_option.geomgroup[3] = 1
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
    img1 = physics_1.render(camera_id=camera_id, scene_option=scene_option)
    img2 = physics_2.render(camera_id=camera_id, scene_option=scene_option)
    axes[4, 0].imshow(img1)
    axes[4, 0].set_title("original (inertia)")
    axes[4, 1].imshow(img2)
    axes[4, 1].set_title("detokenized (inertia)")

    for ax in axes.flatten():
        ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.show()


def check_physics_equal(
    physics_1: mjcf.Physics,
    physics_2: mjcf.Physics,
    num_fk_tests: int = 50,
    num_ctrl_tests: int = 50,
    num_drop_tests: int = 0,
    num_contact_steps_til_success: int = 5,
    # TODO revamp check qpos and qvel after implementing joint correspondence
    check_qpos: bool = True,
    check_qvel: bool = True,
    check_qacc: bool = True,
    disable_contact: bool = False,
    xatol: float = 1e-6,
    xrotatol: float = 5e-3,
    qatol: float = 1e-3,
    qvelatol: float = 1e-3,
    qaccatol: float = 5e-1,
    qaccrtol: float = 1e-3,
    check_free_links: bool = False,
    check_ball_joints: bool = False,
):
    # TODO update all rotation checks to use orientation norm, instead of np.allclose
    model_1 = physics_1.model.ptr
    model_2 = physics_2.model.ptr
    for opt_attr in dir(model_1.opt):
        if not opt_attr.startswith("_"):
            setattr(model_2.opt, opt_attr, getattr(model_1.opt, opt_attr))
    if disable_contact:
        # disable contact
        model_1.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        model_2.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    data_1 = physics_1.data.ptr
    data_2 = physics_2.data.ptr

    rs = np.random.RandomState(0)

    physics_1.reset(0 if physics_1.model.nkey > 0 else None)
    physics_2.reset(0 if physics_2.model.nkey > 0 else None)
    # get all indices that do not correspond to freejoints
    nonfree_qpos_indices_1 = []
    nonfree_qvel_indices_1 = []
    for jntid in range(model_1.njnt):
        if model_1.jnt_type[jntid] != mujoco.mjtJoint.mjJNT_FREE:
            qpos_idx = model_1.jnt_qposadr[jntid]
            nonfree_qpos_indices_1.append(int(qpos_idx))
            qvel_idx = model_1.jnt_dofadr[jntid]
            nonfree_qvel_indices_1.append(int(qvel_idx))
    nonfree_qpos_indices_2 = []
    nonfree_qvel_indices_2 = []
    for jntid in range(model_2.njnt):
        if model_2.jnt_type[jntid] != mujoco.mjtJoint.mjJNT_FREE:
            qpos_idx = model_2.jnt_qposadr[jntid]
            nonfree_qpos_indices_2.append(int(qpos_idx))
            qvel_idx = model_2.jnt_dofadr[jntid]
            nonfree_qvel_indices_2.append(int(qvel_idx))

    # establish geometry correspondence
    geom_xpos_1 = data_1.geom_xpos
    geom_xpos_2 = data_2.geom_xpos
    is_contact_geom_1 = model_1.geom_contype != 0
    is_contact_geom_2 = model_2.geom_contype != 0
    geom_1to2 = []
    geom_2to1_dist = []
    for geom_idx in range(is_contact_geom_1.sum()):
        geom_type = model_1.geom_type[is_contact_geom_1][geom_idx]
        matching_geom_type_2 = model_2.geom_type[is_contact_geom_2] == geom_type
        match_geom_size_2 = np.isclose(
            model_1.geom_size[is_contact_geom_1][geom_idx],
            model_2.geom_size[is_contact_geom_2],
        ).all(axis=-1)

        matching = matching_geom_type_2 & match_geom_size_2
        assert matching.any()
        dist = np.linalg.norm(
            geom_xpos_2[is_contact_geom_2] - geom_xpos_1[is_contact_geom_1][geom_idx],
            axis=1,
        )
        dist[~matching] = np.inf
        matching_idx = int(np.argmin(dist))
        geom_1to2.append(matching_idx)
        geom_2to1_dist.append(dist[matching_idx])
    assert np.all(np.array(geom_2to1_dist) < 1e-8), (
        "some geoms are unmatched: {}".format(geom_2to1_dist)
    )

    assert len(set(geom_1to2)) == len(geom_1to2), "some geoms are indistinguishable"
    geom_2to1_mat = np.array(
        [
            data_1.geom_xmat[is_contact_geom_1][i].reshape(3, 3).T
            @ data_2.geom_xmat[is_contact_geom_2][geom_1to2[i]].reshape(3, 3)
            for i in range(is_contact_geom_1.sum())
        ]
    )
    # establish site correspondence
    is_tracking_site_1 = model_1.site_group[:] == TRACKING_SITE_GROUP
    is_tracking_site_2 = model_2.site_group[:] == TRACKING_SITE_GROUP
    num_tracking_sites_1 = is_tracking_site_1.sum()
    num_tracking_sites_2 = is_tracking_site_2.sum()
    assert num_tracking_sites_1 == num_tracking_sites_2
    if num_tracking_sites_1 > 0:
        if num_tracking_sites_1 == 1:
            site_2to1 = [0]
        else:
            site_xpos_1 = data_1.site_xpos
            site_xpos_2 = data_2.site_xpos
            site_xmat_1 = data_1.site_xmat.reshape(-1, 9)
            site_xmat_2 = data_2.site_xmat.reshape(-1, 9)
            site_2to1 = [
                int(
                    np.argmin(
                        np.linalg.norm(
                            site_xpos_2[is_tracking_site_2]
                            - site_xpos_1[is_tracking_site_1][i],
                            axis=1,
                        )
                        + np.linalg.norm(
                            site_xmat_2[is_tracking_site_2]
                            - site_xmat_1[is_tracking_site_1][i],
                            axis=1,
                        )
                    )
                )
                for i in range(num_tracking_sites_1)
            ]
    else:
        site_2to1 = []

    assert model_1.nu == model_2.nu
    assert model_1.na == model_2.na
    assert np.all(
        model_1.geom_type[is_contact_geom_1]
        == model_2.geom_type[is_contact_geom_2][geom_1to2]
    )
    assert np.all(
        model_1.geom_size[is_contact_geom_1]
        == model_2.geom_size[is_contact_geom_2][geom_1to2]
    )

    # these won't be exactly equal, but should be close
    def get_jnt_pos(m, d, jnt_idx):
        assert jnt_idx != -1
        body_idx = m.jnt_bodyid[jnt_idx]
        body_pose = affines.compose(
            T=d.xpos[body_idx],
            R=d.xmat[body_idx].reshape(3, 3),
            Z=np.ones(3),
        )
        jnt_pos_body = m.jnt_pos[jnt_idx]
        jnt_pos_world = body_pose @ np.concatenate((jnt_pos_body, [1]), axis=0)
        jnt_pos_world = jnt_pos_world[:3]
        return jnt_pos_world

    if model_1.njnt > 0 and (check_qpos or check_qvel or check_qacc):
        jnt_pos_1 = np.array(
            [get_jnt_pos(model_1, data_1, jnt_idx) for jnt_idx in range(model_1.njnt)]
        )
        jnt_pos_2 = np.array(
            [get_jnt_pos(model_2, data_2, jnt_idx) for jnt_idx in range(model_2.njnt)]
        )
        jnt_pos_dist = np.linalg.norm(
            jnt_pos_2[:, None, :] - jnt_pos_1[None, :, :], axis=-1
        )

        is_free_link_1 = model_1.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        is_free_link_2 = model_2.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        if is_free_link_1.any():
            jnt_pos_dist[np.arange(model_1.njnt), is_free_link_2] = np.inf
        assert (is_free_link_1).sum() <= 1, "at most one free joint is allowed"
        jnt_type_1 = model_1.jnt_type
        jnt_type_2 = model_2.jnt_type
        jnt_type_dist = np.ones((model_1.njnt, model_2.njnt)) * np.inf
        jnt_type_dist[jnt_type_1[:, None] == jnt_type_2[None, :]] = 0
        jnt_range_dist = np.ones((model_1.njnt, model_2.njnt)) * np.inf
        for jnt_idx_1 in range(model_1.njnt):
            for jnt_idx_2 in range(model_2.njnt):
                jnt_range_dist[jnt_idx_1, jnt_idx_2] = np.linalg.norm(
                    model_1.jnt_range[jnt_idx_1] - model_2.jnt_range[jnt_idx_2]
                )
        jnt_dynamics_dist = np.ones((model_1.njnt, model_2.njnt)) * np.inf
        for jnt_idx_1 in range(model_1.njnt):
            for jnt_idx_2 in range(model_2.njnt):
                dof_idx_1 = model_1.jnt_dofadr[jnt_idx_1]
                dof_idx_2 = model_2.jnt_dofadr[jnt_idx_2]
                jnt_dynamics_dist[jnt_idx_1, jnt_idx_2] = (
                    int(
                        model_1.dof_damping[dof_idx_1] != model_2.dof_damping[dof_idx_2]
                    )
                    + int(
                        model_1.dof_frictionloss[dof_idx_1]
                        != model_2.dof_frictionloss[dof_idx_2]
                    )
                    + int(
                        model_1.dof_armature[dof_idx_1]
                        != model_2.dof_armature[dof_idx_2]
                    )
                )
        total_dist = jnt_pos_dist + jnt_type_dist + jnt_range_dist
        total_dist = jnt_pos_dist + jnt_type_dist
        assert ((total_dist < 1e-8).sum(axis=1)[~is_free_link_1] == 1).all(), (
            "some joints are indistinguishable"
        )
        jnt_1to2 = np.argmin(total_dist, axis=-1)

        if (is_free_link_1).sum() > 0:
            jnt_1to2[is_free_link_1] = np.where(is_free_link_2)[0][0]
        assert len(set(jnt_1to2)) == len(jnt_1to2), "some joints are indistinguishable"

        qpos_1to2 = np.zeros(model_1.nq, dtype=int)
        qvel_1to2 = np.zeros(model_1.nv, dtype=int)
        qpos_mask_1 = np.ones(model_1.nq, dtype=bool)
        qpos_mask_2 = np.ones(model_2.nq, dtype=bool)
        qvel_mask_1 = np.ones(model_1.nv, dtype=bool)
        qvel_mask_2 = np.ones(model_2.nv, dtype=bool)
        qpos_jnt_type_1 = np.ones(model_1.nq, dtype=int) * -1
        qpos_jnt_type_2 = np.ones(model_2.nq, dtype=int) * -1
        for jnt_idx_1 in range(model_1.njnt):
            # establish qpos correspondence
            start_qpos_idx_1 = model_1.jnt_qposadr[jnt_idx_1]
            num_qpos_1 = 1
            qpos_jnt_type_1[start_qpos_idx_1] = model_1.jnt_type[jnt_idx_1]
            if model_1.jnt_type[jnt_idx_1] == mujoco.mjtJoint.mjJNT_FREE:
                num_qpos_1 = 7  # pos, quat
                qpos_jnt_type_1[start_qpos_idx_1 : start_qpos_idx_1 + num_qpos_1] = (
                    mujoco.mjtJoint.mjJNT_FREE
                )
                if not check_free_links:
                    qpos_mask_1[start_qpos_idx_1 : start_qpos_idx_1 + num_qpos_1] = (
                        False
                    )
            elif model_1.jnt_type[jnt_idx_1] == mujoco.mjtJoint.mjJNT_BALL:
                num_qpos_1 = 4  # quat
                qpos_jnt_type_1[start_qpos_idx_1 : start_qpos_idx_1 + num_qpos_1] = (
                    mujoco.mjtJoint.mjJNT_BALL
                )
                if not check_ball_joints:
                    qpos_mask_1[start_qpos_idx_1 : start_qpos_idx_1 + num_qpos_1] = (
                        False
                    )
            jnt_qpos_addr_1 = np.arange(start_qpos_idx_1, start_qpos_idx_1 + num_qpos_1)

            jnt_idx_2 = np.where(jnt_1to2 == jnt_idx_1)[0][0]

            start_qpos_idx_2 = model_2.jnt_qposadr[jnt_idx_2]
            num_qpos_2 = 1
            qpos_jnt_type_2[start_qpos_idx_2] = model_2.jnt_type[jnt_idx_2]
            if model_2.jnt_type[jnt_idx_2] == mujoco.mjtJoint.mjJNT_FREE:
                num_qpos_2 = 7  # pos, quat
                qpos_jnt_type_2[start_qpos_idx_2 : start_qpos_idx_2 + num_qpos_2] = (
                    mujoco.mjtJoint.mjJNT_FREE
                )
                if not check_free_links:
                    qpos_mask_2[start_qpos_idx_2 : start_qpos_idx_2 + num_qpos_2] = (
                        False
                    )
            elif model_2.jnt_type[jnt_idx_2] == mujoco.mjtJoint.mjJNT_BALL:
                num_qpos_2 = 4  # quat
                qpos_jnt_type_2[start_qpos_idx_2 : start_qpos_idx_2 + num_qpos_2] = (
                    mujoco.mjtJoint.mjJNT_BALL
                )
                if not check_ball_joints:
                    qpos_mask_2[start_qpos_idx_2 : start_qpos_idx_2 + num_qpos_2] = (
                        False
                    )
            jnt_qpos_addr_2 = np.arange(start_qpos_idx_2, start_qpos_idx_2 + num_qpos_2)

            qpos_1to2[jnt_qpos_addr_1] = jnt_qpos_addr_2

            # establish qvel correspondence
            start_qvel_idx_1 = model_1.jnt_dofadr[jnt_idx_1]
            num_qvel_1 = 1
            if model_1.jnt_type[jnt_idx_1] == mujoco.mjtJoint.mjJNT_FREE:
                num_qvel_1 = 6  # lin vel, ang vel
                if not check_free_links:
                    qvel_mask_1[start_qvel_idx_1 : start_qvel_idx_1 + num_qvel_1] = (
                        False
                    )
            elif model_1.jnt_type[jnt_idx_1] == mujoco.mjtJoint.mjJNT_BALL:
                num_qvel_1 = 3  # ang vel
                if not check_ball_joints:
                    qvel_mask_1[start_qvel_idx_1 : start_qvel_idx_1 + num_qvel_1] = (
                        False
                    )
            jnt_qvel_addr_1 = np.arange(start_qvel_idx_1, start_qvel_idx_1 + num_qvel_1)

            start_qvel_idx_2 = model_2.jnt_dofadr[jnt_idx_2]
            num_qvel_2 = 1
            if model_2.jnt_type[jnt_idx_2] == mujoco.mjtJoint.mjJNT_FREE:
                num_qvel_2 = 6  # lin vel, ang vel
                if not check_free_links:
                    qvel_mask_2[start_qvel_idx_2 : start_qvel_idx_2 + num_qvel_2] = (
                        False
                    )
            elif model_2.jnt_type[jnt_idx_2] == mujoco.mjtJoint.mjJNT_BALL:
                num_qvel_2 = 3  # ang vel
                if not check_ball_joints:
                    qvel_mask_2[start_qvel_idx_2 : start_qvel_idx_2 + num_qvel_2] = (
                        False
                    )
            jnt_qvel_addr_2 = np.arange(start_qvel_idx_2, start_qvel_idx_2 + num_qvel_2)

            assert num_qvel_1 == num_qvel_2

            qvel_1to2[jnt_qvel_addr_1] = jnt_qvel_addr_2
        assert len(set(qvel_1to2)) == len(qvel_1to2), "some qvels are indistinguishable"
        assert np.all(
            model_1.dof_frictionloss[:] == model_2.dof_frictionloss[qvel_1to2]
        )
        assert np.all(model_1.dof_damping[:] == model_2.dof_damping[qvel_1to2])
        assert np.all(model_1.dof_armature[:] == model_2.dof_armature[qvel_1to2])
    else:
        check_qpos = False
        check_qvel = False
        check_qacc = False

    act_jnt_pos_1 = []
    act_jnt_pos_2 = []
    actuator_2to1 = []
    if model_1.nu > 0:
        # find actuator based on the pose of the joint it controls
        act_jnt_pos_1 = np.array(
            [
                get_jnt_pos(model_1, data_1, model_1.actuator_trnid[act_idx, 0])
                for act_idx in range(model_1.nu)
            ]
        )
        act_jnt_pos_2 = np.array(
            [
                get_jnt_pos(model_2, data_2, model_2.actuator_trnid[act_idx, 0])
                for act_idx in range(model_2.nu)
            ]
        )
        actuator_pos_dist = np.linalg.norm(
            act_jnt_pos_2[:, None, :] - act_jnt_pos_1[None, :, :], axis=-1
        )

        assert ((actuator_pos_dist < 1e-8).sum(axis=0) == 1).all(), (
            "some actuators are indistinguishable or undetectable"
        )
        actuator_2to1 = [
            int(x)
            for x in np.argmin(
                actuator_pos_dist,
                axis=0,
            )
        ]
        assert actuator_pos_dist[actuator_2to1, np.arange(model_2.nu)].max() < 1e-8, (
            "some actuators aren't matched"
        )
        assert len(set(actuator_2to1)) == len(actuator_2to1), (
            "some actuators are indistinguishable"
        )

    assert np.all(
        model_1.actuator_trntype[:] == model_2.actuator_trntype[actuator_2to1]
    )
    assert np.all(
        model_1.actuator_dyntype[:] == model_2.actuator_dyntype[actuator_2to1]
    )
    assert np.all(
        model_1.actuator_gaintype[:] == model_2.actuator_gaintype[actuator_2to1]
    )
    assert np.all(
        model_1.actuator_biastype[:] == model_2.actuator_biastype[actuator_2to1]
    )
    assert np.all(
        model_1.actuator_ctrlrange[:] == model_2.actuator_ctrlrange[actuator_2to1]
    )
    assert np.all(
        model_1.actuator_forcerange[:] == model_2.actuator_forcerange[actuator_2to1]
    )
    assert np.all(model_1.actuator_dynprm[:] == model_2.actuator_dynprm[actuator_2to1])
    assert np.all(
        model_1.actuator_gainprm[:] == model_2.actuator_gainprm[actuator_2to1]
    )
    assert np.all(
        model_1.actuator_biasprm[:] == model_2.actuator_biasprm[actuator_2to1]
    )

    def check(xatol=xatol, qatol=qatol, check_qpos=check_qpos, check_qvel=check_qvel):
        if model_1.ngeom > 0:
            assert np.allclose(
                data_1.geom_xpos[is_contact_geom_1],
                data_2.geom_xpos[is_contact_geom_2][geom_1to2],
                atol=xatol,
            ), np.abs(
                data_1.geom_xpos[is_contact_geom_1]
                - data_2.geom_xpos[is_contact_geom_2][geom_1to2]
            ).max()
            err_rotmat = np.transpose(
                data_1.geom_xmat[is_contact_geom_1].reshape(-1, 3, 3) @ geom_2to1_mat,
                axes=(0, 2, 1),
            ) @ data_2.geom_xmat[is_contact_geom_2][geom_1to2].reshape(-1, 3, 3)
            err_norms = [
                float(mat_norm(err_rotmat[i])) for i in range(err_rotmat.shape[0])
            ]

            assert (np.array(err_norms) < xrotatol).all(), max(err_norms)
        if num_tracking_sites_1 > 0:
            assert np.allclose(
                data_1.site_xpos[is_tracking_site_1],
                data_2.site_xpos[is_tracking_site_2][site_2to1],
                atol=xatol,
            ), np.abs(
                data_1.site_xpos[is_tracking_site_1]
                - data_2.site_xpos[is_tracking_site_2][site_2to1]
            ).max()
            assert np.allclose(
                data_1.site_xmat[is_tracking_site_1],
                data_2.site_xmat[is_tracking_site_2][site_2to1],
                atol=xrotatol,
            ), np.abs(
                data_1.site_xmat[is_tracking_site_1]
                - data_2.site_xmat[is_tracking_site_2][site_2to1]
            ).max()
        if check_qpos:
            assert np.allclose(
                data_1.qpos[qpos_mask_1],
                data_2.qpos[qpos_1to2][qpos_mask_1],
                atol=qatol,
            ), (
                data_1.qpos[qpos_mask_1] - data_2.qpos[qpos_1to2][qpos_mask_1],
                qpos_1to2[qpos_mask_1],
                np.abs(
                    data_1.qpos[qpos_mask_1] - data_2.qpos[qpos_1to2][qpos_mask_1]
                ).max(),
            )
        if check_qvel:
            assert np.allclose(
                data_1.qvel[qvel_mask_1],
                data_2.qvel[qvel_1to2][qvel_mask_1],
                atol=qvelatol,
            ), (data_1.qvel[qvel_mask_1] - data_2.qvel[qvel_1to2][qvel_mask_1]).max()
        if check_qacc:
            assert np.allclose(
                data_1.qacc[qvel_mask_1],
                data_2.qacc[qvel_1to2][qvel_mask_1],
                atol=qaccatol,
                rtol=qaccrtol,
            ), (data_1.qacc[qvel_mask_1] - data_2.qacc[qvel_1to2][qvel_mask_1]).max()

    # check that the scene is the same at reset
    physics_1.reset(0 if physics_1.model.nkey > 0 else None)
    physics_2.reset(0 if physics_2.model.nkey > 0 else None)
    check()

    perturb_geom_p = np.ones(model_1.ngeom)
    perturb_geom_p[~is_contact_geom_1] = 0
    perturb_geom_p[model_1.geom_type[:] == mujoco.mjtGeom.mjGEOM_PLANE] = 0
    perturb_geom_p = perturb_geom_p / perturb_geom_p.sum()
    for _ in range(num_fk_tests):
        physics_1.reset(0 if physics_1.model.nkey > 0 else None)
        physics_2.reset(0 if physics_2.model.nkey > 0 else None)
        wrench_1 = rs.uniform(-1, 1, 6)  # force is first 3, torque is last 3
        if (perturb_geom_p > 0).any():
            perturb_geom_idx_1 = rs.choice(
                model_1.ngeom, p=perturb_geom_p / perturb_geom_p.sum()
            )
            one_hot = np.zeros(model_1.ngeom)
            one_hot[perturb_geom_idx_1] = 1
            is_contact_idx_1 = np.where(one_hot[is_contact_geom_1])[0][0]
            perturb_geom_idx_2 = geom_1to2[is_contact_idx_1]
            body_idx_1 = model_1.geom_bodyid[perturb_geom_idx_1]
            body_idx_2 = model_2.geom_bodyid[perturb_geom_idx_2]
        else:
            assert model_1.nbody == model_2.nbody
            body_idx_1 = model_1.nbody - 1
            body_idx_2 = model_2.nbody - 1
        point = rs.uniform(-0.1, 0.1, 3) + data_1.xpos[body_idx_1]
        mujoco.mj_applyFT(
            m=model_1,
            d=data_1,
            force=wrench_1[:3],
            torque=wrench_1[3:],
            point=point,
            body=body_idx_1,
            qfrc_target=data_1.qfrc_applied,
        )
        mujoco.mj_applyFT(
            m=model_2,
            d=data_2,
            force=wrench_1[:3],
            torque=wrench_1[3:],
            point=point,
            body=body_idx_2,
            qfrc_target=data_2.qfrc_applied,
        )
        physics_1.step()
        physics_2.step()
        check()
    # return
    scene_option = MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = True
    for ctrl_iter in range(num_ctrl_tests):
        physics_1.reset(0 if physics_1.model.nkey > 0 else None)
        physics_2.reset(0 if physics_2.model.nkey > 0 else None)
        random_ctrl = rs.uniform(
            model_2.actuator_ctrlrange[:, 0],
            model_2.actuator_ctrlrange[:, 1],
        )
        data_1.ctrl[:] = random_ctrl
        data_2.ctrl[actuator_2to1] = random_ctrl
        physics_1.step()
        physics_2.step()
        check()
    if (model_1.jnt_type == mujoco.mjtJoint.mjJNT_FREE).any():
        is_freejoint_1 = model_1.jnt_type[:] == mujoco.mjtJoint.mjJNT_FREE
        is_freejoint_2 = model_2.jnt_type[:] == mujoco.mjtJoint.mjJNT_FREE
        assert is_freejoint_1.sum() == 1
        assert is_freejoint_2.sum() == 1
        freejoint_idx_1 = np.where(is_freejoint_1)[0][0]
        freejoint_idx_2 = np.where(is_freejoint_2)[0][0]
        freejoint_qpos_1 = model_1.jnt_qposadr[freejoint_idx_1]
        freejoint_qpos_2 = model_2.jnt_qposadr[freejoint_idx_2]
        for _ in range(num_drop_tests):
            physics_1.reset(0 if physics_1.model.nkey > 0 else None)
            physics_2.reset(0 if physics_2.model.nkey > 0 else None)
            height = rs.uniform(0.1, 0.2)
            geom_height = data_1.geom_xpos[:, 2]
            geom_idx_1 = np.argmin(geom_height)
            z_displacement = height - geom_height[geom_idx_1]

            data_1.qpos[freejoint_qpos_1 + 2] += z_displacement
            data_2.qpos[freejoint_qpos_2 + 2] += z_displacement
            num_contact_steps = 0

            for _ in range(10000):
                physics_1.step()
                physics_2.step()
                check()
                if len(data_1.contact) > 0:
                    num_contact_steps += 1
                if num_contact_steps >= num_contact_steps_til_success:
                    break


def tokenize_detokenize(
    model: mjcf.RootElement,
    create_viewer: bool = False,
    render: bool = False,
    add_plane: bool = False,
    gravcomp: bool = False,
    dump_xml: bool = False,
    **kwargs,
):
    physics = mjcf.Physics.from_mjcf_model(
        set_up_default_scene(
            copy.deepcopy(model), add_plane=add_plane, add_vis_cam=True
        )
    )
    assert physics is not None
    physics.reset()
    model = preprocess_mjcf(model)
    if dump_xml:
        with open("preprocessed.xml", "w") as f:
            f.write(model.to_xml_string())
    preprocessed_physics = mjcf.Physics.from_mjcf_model(
        set_up_default_scene(
            copy.deepcopy(model), add_plane=add_plane, add_vis_cam=True
        )
    )
    assert preprocessed_physics is not None
    # check_physics_equal(physics, preprocessed_physics, **kwargs)
    tokenized_robot, mjjntid_to_dyna_jntid, mjgeomid_to_linkid, _ = tokenize(model)
    detokenized_model, linkid_to_mjgeomid, dyna_jntid_to_mjjntid, _ = detokenize(
        tokenized_robot, gravcomp=gravcomp
    )
    if dump_xml:
        with open("detokenized.xml", "w") as f:
            f.write(detokenized_model.to_xml_string())
    decoded_physics = mjcf.Physics.from_mjcf_model(
        set_up_default_scene(detokenized_model, add_plane=add_plane, add_vis_cam=True)
    )

    assert decoded_physics is not None
    if create_viewer:
        physics.reset()
        physics.data.ptr.qpos[:] = physics.model.ptr.qpos0[:]
        physics.forward()
        with viewer.launch_passive(
            model=physics.model.ptr,
            data=physics.data.ptr,
        ) as gui:
            gui._opt.geomgroup[3] = True
            while gui.is_running():
                gui.sync()
                # physics.step()
                time.sleep(1 / 60)
        decoded_physics.reset()
        decoded_physics.data.ptr.qpos[:] = decoded_physics.model.ptr.qpos0[:]
        decoded_physics.forward()
        with viewer.launch_passive(
            model=decoded_physics.model.ptr,
            data=decoded_physics.data.ptr,
        ) as gui:
            gui._opt.geomgroup[3] = True
            while gui.is_running():
                gui.sync()
                # decoded_physics.step()
                time.sleep(1 / 60)
    if render:
        render_comparison(physics, decoded_physics)

    check_physics_equal(
        preprocessed_physics,
        decoded_physics,
        **kwargs,
    )

    data_dict = serialize(tokenized_robot, include_states=True)
    serialized_robot = deserialize(data_dict, include_states=True)
    serialized_model = detokenize(serialized_robot, gravcomp=gravcomp)[0]
    serialized_physics = mjcf.Physics.from_mjcf_model(
        set_up_default_scene(
            serialized_model,
            add_plane=add_plane,
            add_vis_cam=True,
        )
    )
    if render:
        render_comparison(physics, serialized_physics)
    assert serialized_physics is not None
    check_physics_equal(
        preprocessed_physics,
        serialized_physics,
        **kwargs,
    )
