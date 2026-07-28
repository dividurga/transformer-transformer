from t2.robogen.components import make_robot


COMPONENTS = {
    "base": "assets/mjcf/wheeled_bimanual/base.xml",
    "wheeled_box": "assets/mjcf/wheeled_bimanual/wheeled_box.xml",
    "spine": "assets/mjcf/wheeled_bimanual/spine.xml",
    "spine_3dof": "assets/mjcf/wheeled_bimanual/spine_3dof.xml",
    "spine_fixed": "assets/mjcf/wheeled_bimanual/spine_fixed.xml",
    "torso": "assets/mjcf/wheeled_bimanual/torso.xml",
    "torso_lean": "assets/mjcf/wheeled_bimanual/torso_lean.xml",
    "left_arm_7dof": "assets/mjcf/wheeled_bimanual/left_arm_7dof.xml",
    "right_arm_7dof": "assets/mjcf/wheeled_bimanual/right_arm_7dof.xml",
}


wheeled_bimanual, wheeled_bimanual_from_params, wheeled_bimanual_from_choices = (
    make_robot(
        COMPONENTS,
        gravity_compensation=True,
        resolve_before_mount=True,
    )
)
