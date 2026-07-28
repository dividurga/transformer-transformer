from t2.robogen.components import make_robot


COMPONENTS = {
    "base": "assets/mjcf/umi_on_legs_plus_plus/base.xml",
    "quadruped_manipulator": "assets/mjcf/umi_on_legs_plus_plus/quadruped_manipulator.xml",
    "quadruped_manipulator_flipped_legs": "assets/mjcf/umi_on_legs_plus_plus/quadruped_manipulator_flipped_legs.xml",
    "wheeled_quadruped_manipulator": "assets/mjcf/umi_on_legs_plus_plus/wheeled_quadruped_manipulator.xml",
    "arx5": "assets/mjcf/umi_on_legs_plus_plus/arx5.xml",
    "left_leg": "assets/mjcf/umi_on_legs_plus_plus/left_leg.xml",
    "right_leg": "assets/mjcf/umi_on_legs_plus_plus/right_leg.xml",
    "flipped_left_leg": "assets/mjcf/umi_on_legs_plus_plus/flipped_left_leg.xml",
    "flipped_right_leg": "assets/mjcf/umi_on_legs_plus_plus/flipped_right_leg.xml",
    "left_wheeled_leg": "assets/mjcf/umi_on_legs_plus_plus/left_wheeled_leg.xml",
    "right_wheeled_leg": "assets/mjcf/umi_on_legs_plus_plus/right_wheeled_leg.xml",
    "left_linkage_leg": "assets/mjcf/umi_on_legs_plus_plus/left_linkage_leg.xml",
    "right_linkage_leg": "assets/mjcf/umi_on_legs_plus_plus/right_linkage_leg.xml",
    "rail": "assets/mjcf/umi_on_legs_plus_plus/rail.xml",
}


(
    umi_on_legs_plus_plus,
    umi_on_legs_plus_plus_from_params,
    umi_on_legs_plus_plus_from_choices,
) = make_robot(COMPONENTS)
