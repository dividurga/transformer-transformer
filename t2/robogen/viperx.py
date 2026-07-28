from dm_control import mjcf

from t2.robogen.components import make_robot


VIPERX_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_planar_vary.xml",
    "shoulder": "assets/mjcf/aloha/aloha_shoulder.xml",
    "upper_arm": "assets/mjcf/aloha/aloha_upper_arm.xml",
    "upper_forearm": "assets/mjcf/aloha/aloha_upper_forearm.xml",
    "lower_forearm": "assets/mjcf/aloha/aloha_lower_forearm.xml",
    "wrist_link": "assets/mjcf/aloha/aloha_wrist_link.xml",
    "gripper": "assets/mjcf/aloha/aloha_gripper.xml",
}
viperx, viperx_from_params, _ = make_robot(
    VIPERX_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_WITH_JOINT_DYNAMICS_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_planar_vary.xml",
    "shoulder": "assets/mjcf/aloha/aloha_shoulder_joint_dynamics.xml",
    "upper_arm": "assets/mjcf/aloha/aloha_upper_arm_joint_dynamics.xml",
    "upper_forearm": "assets/mjcf/aloha/aloha_upper_forearm_joint_dynamics.xml",
    "lower_forearm": "assets/mjcf/aloha/aloha_lower_forearm_joint_dynamics.xml",
    "wrist_link": "assets/mjcf/aloha/aloha_wrist_link_joint_dynamics.xml",
    "gripper": "assets/mjcf/aloha/aloha_gripper_joint_dynamics.xml",
}
(
    viperx_with_joint_dynamics,
    viperx_with_joint_dynamics_from_params,
    _,
) = make_robot(
    VIPERX_WITH_JOINT_DYNAMICS_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VARIABLE_DOF_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_full_rotation.xml",
    "base_up": "assets/mjcf/aloha/aloha_base_up.xml",
    "base_down": "assets/mjcf/aloha/aloha_base_down.xml",
    "base_left": "assets/mjcf/aloha/aloha_base_left.xml",
    "base_right": "assets/mjcf/aloha/aloha_base_right.xml",
    "shoulder": "assets/mjcf/aloha/aloha_shoulder.xml",
    "upper_arm": "assets/mjcf/aloha/aloha_upper_arm.xml",
    "upper_forearm": "assets/mjcf/aloha/aloha_upper_forearm_variable.xml",
    "second_upper_forearm": "assets/mjcf/aloha/aloha_second_upper_forearm.xml",
    "lower_forearm": "assets/mjcf/aloha/aloha_lower_forearm_variable_axis.xml",
    "wrist_link": "assets/mjcf/aloha/aloha_wrist_link_variable_axis.xml",
    "gripper": "assets/mjcf/aloha/aloha_gripper_variable_axis.xml",
}


BIMANUAL_ARM_COMPONENTS = {
    "shoulder": "assets/mjcf/aloha/aloha_shoulder.xml",
    "upper_arm": "assets/mjcf/aloha/aloha_upper_arm.xml",
    "upper_forearm": "assets/mjcf/aloha/aloha_upper_forearm.xml",
    "lower_forearm": "assets/mjcf/aloha/aloha_lower_forearm.xml",
    "wrist_link": "assets/mjcf/aloha/aloha_wrist_link.xml",
    "gripper": "assets/mjcf/aloha/aloha_gripper.xml",
}


VIPERX_BIMANUAL_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_bimanual.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
(
    viperx_bimanual_full_variation,
    viperx_bimanual_full_variation_from_params,
    _,
) = make_robot(
    VIPERX_BIMANUAL_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_BIMANUAL_V1_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_bimanual_v1.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""V0 variations + symmetric arm spacing in [0.40m, 0.90m]."""
viperx_bimanual_v1, viperx_bimanual_v1_from_params, _ = make_robot(
    VIPERX_BIMANUAL_V1_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_BIMANUAL_V2_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_bimanual_v2.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""V1 variations + symmetric z-euler yaw of each arm's mount in [-pi/4, pi/4]."""
viperx_bimanual_v2, viperx_bimanual_v2_from_params, _ = make_robot(
    VIPERX_BIMANUAL_V2_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_BIMANUAL_OPPOSING_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_bimanual_opposing.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Bimanual viperx where the two arms face each other across the y-axis."""
(
    viperx_bimanual_opposing,
    viperx_bimanual_opposing_from_params,
    _,
) = make_robot(
    VIPERX_BIMANUAL_OPPOSING_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_BIMANUAL_OPPOSING_V1_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_bimanual_opposing_v1.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Opposing V0 variations + symmetric arm spacing in [0.40m, 0.90m]."""
(
    viperx_bimanual_opposing_v1,
    viperx_bimanual_opposing_v1_from_params,
    _,
) = make_robot(
    VIPERX_BIMANUAL_OPPOSING_V1_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_UPSIDE_DOWN_BIMANUAL_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_upside_down_bimanual.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Upside-down bimanual: arms hang from above pointing forward (still
side-by-side, not opposing each other), with symmetric x and z variation
in the arm mount points."""
(
    viperx_upside_down_bimanual,
    viperx_upside_down_bimanual_from_params,
    _,
) = make_robot(
    VIPERX_UPSIDE_DOWN_BIMANUAL_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_UPSIDE_DOWN_BIMANUAL_V1_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_upside_down_bimanual_v1.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Upside-down V0 variations + symmetric arm spacing in [0.40m, 0.90m]."""
(
    viperx_upside_down_bimanual_v1,
    viperx_upside_down_bimanual_v1_from_params,
    _,
) = make_robot(
    VIPERX_UPSIDE_DOWN_BIMANUAL_V1_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_UPSIDE_DOWN_BIMANUAL_V2_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_upside_down_bimanual_v2.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Upside-down V1 variations + symmetric z-yaw of each arm holder in [-pi/4, pi/4]."""
(
    viperx_upside_down_bimanual_v2,
    viperx_upside_down_bimanual_v2_from_params,
    _,
) = make_robot(
    VIPERX_UPSIDE_DOWN_BIMANUAL_V2_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_VERTICAL_BIMANUAL_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_vertical_bimanual.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Vertical bimanual: arms wall-mounted on opposing left/right walls with the
arm pan axes horizontal and pointing toward each other; symmetric x and z
variation in the arm mount points."""
(
    viperx_vertical_bimanual,
    viperx_vertical_bimanual_from_params,
    _,
) = make_robot(
    VIPERX_VERTICAL_BIMANUAL_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_VERTICAL_BIMANUAL_V1_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_vertical_bimanual_v1.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Vertical V0 variations + symmetric arm spacing in [0.40m, 0.90m]."""
(
    viperx_vertical_bimanual_v1,
    viperx_vertical_bimanual_v1_from_params,
    _,
) = make_robot(
    VIPERX_VERTICAL_BIMANUAL_V1_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_VERTICAL_BIMANUAL_V2_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_vertical_bimanual_v2.xml",
    **BIMANUAL_ARM_COMPONENTS,
}
"""Vertical V1 variations + symmetric z-yaw of each arm holder in [-pi/4, pi/4]."""
(
    viperx_vertical_bimanual_v2,
    viperx_vertical_bimanual_v2_from_params,
    _,
) = make_robot(
    VIPERX_VERTICAL_BIMANUAL_V2_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


(
    variable_dof_viperx_full_variation,
    variable_dof_viperx_full_variation_from_params,
    _,
) = make_robot(
    VARIABLE_DOF_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_BIMANUAL_OPPOSING_70CM_PATH = (
    "assets/mjcf/aloha/aloha_bimanual_opposing_70cm.xml"
)


def viperx_bimanual_opposing_70cm(seed: int = 0, **kwargs) -> mjcf.RootElement:
    """Fixed bimanual viperx: two arms 70cm apart along y, opposing each other.

    No variation of any kind. Each arm's body chain, joint axes, joint ranges,
    geometry and finger placement are a verbatim copy of the partial viperx
    defined in `assets/mjcf/aloha/partial_aloha.xml` (primitive collisions,
    auto-computed inertias, merged gripper body, fingers at +/- 0.028m),
    with `left_`/`right_` name prefixes. `seed` is accepted for API
    compatibility with other robogens but is ignored.
    """
    return mjcf.from_path(VIPERX_BIMANUAL_OPPOSING_70CM_PATH)


VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_PATH = (
    "assets/mjcf/aloha/aloha_upside_down_bimanual_50cm.xml"
)


def viperx_upside_down_bimanual_50cm(
    seed: int = 0, **kwargs
) -> mjcf.RootElement:
    """Fixed upside-down bimanual viperx: two arms 50cm apart along y, hanging
    from above and both facing forward (+x).

    No variation of any kind. Both arms are mounted at z=1.0, rotated pi about
    x so the arm's local +z (the shoulder/upper-arm build axis) points toward
    world -z; local +x is preserved by the x-axis rotation, so both arms share
    the same forward heading. Each arm's body chain, joint axes, joint ranges,
    geometry and finger placement are a verbatim copy of the partial viperx
    defined in `assets/mjcf/aloha/partial_aloha.xml` (primitive collisions,
    auto-computed inertias, merged gripper body, fingers at +/- 0.028m), with
    `left_`/`right_` name prefixes. `seed` is accepted for API compatibility
    with other robogens but is ignored.
    """
    return mjcf.from_path(VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_PATH)




VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_XZ_VARIATION_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_upside_down_bimanual_50cm_xz_variation.xml",
}
"""Upside-down bimanual with fixed 50cm y-spacing and symmetric mount-point
variation: x in [-0.40, 0.00] and z in [0.60, 0.90] (both arms share x and z)."""
(
    viperx_upside_down_bimanual_50cm_xz_variation,
    viperx_upside_down_bimanual_50cm_xz_variation_from_params,
    _,
) = make_robot(
    VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_XZ_VARIATION_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_XZ_LENGTH_VARIATION_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_upside_down_bimanual_50cm_xz_length_variation.xml",
}
"""XZ-variation + symmetric arm-length variation. Both arms share the same
upper-arm extension (total in [10cm, 50cm]) and the same upper-forearm
extension (total in [5cm, 25cm]). Adds two extra uniform parameters."""
(
    viperx_upside_down_bimanual_50cm_xz_length_variation,
    viperx_upside_down_bimanual_50cm_xz_length_variation_from_params,
    _,
) = make_robot(
    VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_XZ_LENGTH_VARIATION_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)


VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_XYZ_LENGTH_VARIATION_COMPONENTS = {
    "base": "assets/mjcf/aloha/aloha_base_upside_down_bimanual_50cm_xyz_length_variation.xml",
}
"""XZ-length variation + symmetric arm-spacing variation along y: arm mount
y in [0.20, 0.30] (right arm mirrored), giving inter-arm spacing in
[40cm, 60cm]. Adds one extra uniform parameter on top of the XZ-length set."""
(
    viperx_upside_down_bimanual_50cm_xyz_length_variation,
    viperx_upside_down_bimanual_50cm_xyz_length_variation_from_params,
    _,
) = make_robot(
    VIPERX_UPSIDE_DOWN_BIMANUAL_50CM_XYZ_LENGTH_VARIATION_COMPONENTS,
    gravity_compensation=True,
    resolve_before_mount=True,
)
