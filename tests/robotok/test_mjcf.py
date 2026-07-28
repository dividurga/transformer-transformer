import pytest
from dm_control import mjcf
from utils import tokenize_detokenize


@pytest.mark.parametrize(
    "path,xatol,xrotatol,qatol,qvelatol",
    [
        # successful
        (
            "assets/mjcf/universal_robots_ur5e/ur5e.xml",
            # position/rotation round-trip residuals grew from <5e-4/<1e-3
            # with newer MuJoCo releases, and the rotation residual varies by
            # platform/BLAS (~1.5e-3 on Linux, up to ~2.6e-3 on macOS)
            6e-4,
            3e-3,
            3e-05,
            2e-2,
        ),
        (
            "assets/mjcf/universal_robots_ur10e/ur10e.xml",
            6e-4,
            3e-3,
            4e-5,
            2e-2,
        ),
        (
            "assets/mjcf/anybotics_anymal_c/anymal_c.xml",
            9e-3,
            4e-2,
            1e-2,
            5.0,  # NOTE: very large residual
        ),
        (
            "assets/mjcf/anybotics_anymal_b/anymal_b.xml",
            4e-3,
            2e-2,
            8e-3,
            5.0,  # NOTE: very large residual
        ),
        (
            # NOTE: right now getting non deterministic behavior
            "assets/mjcf/unitree_h1/h1.xml",
            0.0005,
            0.002,
            100,
            100,
        ),
        (
            "assets/mjcf/unitree_go1/go1.xml",
            0.002,
            0.005,
            0.0009,
            0.5,
        ),
        # high residual
        (
            "assets/mjcf/unitree_go2/go2.xml",
            0.0005,
            0.005,
            0.001,
            0.5,
        ),
        (
            "assets/mjcf/unitree_a1/a1.xml",
            0.002,
            0.005,
            0.002,
            1.0,
        ),
        # (
        #     "assets/mjcf/agility_cassie/cassie_collision_enabled.xml",
        #     0.001,
        #     0.02,
        #     -1,
        #     -1,
        #     # 0.002,
        #     # 5.0,
        # ),
        (
            "assets/mjcf/aloha/partial_aloha.xml",
            1e-3,
            1e-2,
            1e-2,
            5e-1,
        ),
        (
            "assets/mjcf/wonik_allegro/left_hand.xml",
            0.001,
            0.05,
            100,
            100,
        ),
    ],
)
def test_tokenize_detokenize_mj_menagerie(
    path: str,
    xatol: float,
    xrotatol: float,
    qatol: float,
    qvelatol: float,
    qaccatol: float = 1.0,
    qaccrtol: float = 1e-1,
    remove_freejoints: bool = False,
):
    mjcf_model = mjcf.from_path(path)

    if remove_freejoints:
        for body_node in mjcf_model.worldbody.all_children():
            if hasattr(body_node, "freejoint") and body_node.freejoint is not None:
                body_node.freejoint.remove()
    if "wonik_allegro" in path:
        # transfer mass from visual geom to collision geom
        # TODO find less hacky way to do this
        geom_dclass = mjcf_model.find("default", "collision")
        geom_dclass.geom.density = 800
        delattr(geom_dclass.geom, "mass")

    # this larger xatol is due to errors accumulated during from to conversion
    # TODO need to parse jnts and qpos/qvel ordering before we can check qpos and qvel
    tokenize_detokenize(
        mjcf_model,
        xatol=xatol,
        xrotatol=xrotatol,
        qatol=qatol,
        qvelatol=qvelatol,
        qaccatol=qaccatol,
        qaccrtol=qaccrtol,
        check_qpos=False,
        check_qvel=False,
        check_qacc=False,
    )


# not supported
models_with_mesh = ["assets/mjcf/hello_robot_stretch_3/stretch.xml"]
feet_only_models = ["assets/mjcf/apptronik_apollo/apptronik_apollo.xml"]
ball_joint_models = ["assets/mjcf/agility_cassie/cassie_collision_enabled.xml"]
contact_exclusion_models = ["assets/mjcf/wonik_allegro/left_hand.xml"]
if __name__ == "__main__":
    test_tokenize_detokenize_mj_menagerie(
        path="assets/mjcf/agility_cassie/cassie_collision_enabled.xml",
        xatol=0.001,
        xrotatol=0.02,
        qatol=-1,
        qvelatol=-1,
    )
