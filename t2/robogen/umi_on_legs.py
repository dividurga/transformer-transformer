
from t2.robogen.components import grow_components


def umi_on_legs_arm_mount(seed: int):
    components = {"base": "assets/mjcf/umi_on_legs/umi_on_legs_arm_mount.xml"}
    mjcf_model = grow_components(
        seed, components=components, gravity_compensation=False
    )
    return mjcf_model
