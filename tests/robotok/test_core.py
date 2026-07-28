import numpy as np
import pytest
from transforms3d import euler
from utils import tokenize_detokenize

from t2.env.mj_utils import TRACKING_SITE_GROUP, default_root_element

TEST_LINK_GEOM_ATTRS = {
    "density": 25,
    "type": "capsule",
    "size": [0.04, 0.0],
    "group": 3,
}
TEST_JOINT_ATTRS = {
    "armature": 0.1,
    "damping": 0.01,
    "frictionloss": 0.001,
    # "range": [-6.28319, 6.28319],
    # "range": [-1.57, 1.57],
    "range": [-np.pi, np.pi],
    "type": "hinge",
}


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_random_link(seed: int):
    root = default_root_element()
    rs = np.random.RandomState(seed)
    body = root.worldbody.add(
        "body",
        name="link_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    body.add(
        "geom",
        name="geom_0",
        **{
            **TEST_LINK_GEOM_ATTRS,
            "size": rs.uniform(0.01, 0.1, 3),
        },
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )

    tokenize_detokenize(root)


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_random_link_and_joint(seed: int):
    root = default_root_element()
    rs = np.random.RandomState(seed)
    body = root.worldbody.add(
        "body",
        name="link_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    body.add(
        "geom",
        name="geom_0",
        **{
            **TEST_LINK_GEOM_ATTRS,
            "size": rs.uniform(0.01, 0.1, 3),
        },
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    axis = rs.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    body.add(
        "joint",
        name="joint_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        axis=axis,
        **{
            **TEST_JOINT_ATTRS,
            "type": "hinge",
        },
    )
    tokenize_detokenize(root)


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_multigeom_body(seed: int):
    root = default_root_element()
    rs = np.random.RandomState(seed)
    body = root.worldbody.add(
        "body",
        name="link_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    body.add(
        "geom",
        name="geom_0",
        **{
            **TEST_LINK_GEOM_ATTRS,
            "size": rs.uniform(0.01, 0.1, 3),
            "type": "box",
        },
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    body.add(
        "geom",
        name="geom_1",
        **{
            **TEST_LINK_GEOM_ATTRS,
            "size": rs.uniform(0.01, 0.1, 3),
            "type": "box",
        },
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    axis = rs.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    body.add(
        "joint",
        name="joint_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        axis=axis,
        **{
            **TEST_JOINT_ATTRS,
            "type": "hinge",
        },
    )
    tokenize_detokenize(root)


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_nogeom_body(seed: int):
    root = default_root_element()
    rs = np.random.RandomState(seed)
    body = root.worldbody.add(
        "body",
        name="link_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    body.add(
        "inertial",
        mass=rs.uniform(0.1, 1.0),
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
        diaginertia=[0.01, 0.01, 0.01],
    )
    axis = rs.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    body.add(
        "joint",
        name="joint_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        axis=axis,
        **{
            **TEST_JOINT_ATTRS,
            "type": "hinge",
        },
    )
    tokenize_detokenize(root)


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_multijoint_body(seed: int):
    root = default_root_element()
    rs = np.random.RandomState(seed)
    body = root.worldbody.add(
        "body",
        name="link_0",
    )
    body.add(
        "geom",
        name="geom_0",
        type="capsule",
        size=rs.uniform(0.1, 1.0, 3),
    )
    axis = rs.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    body.add(
        "joint",
        name="joint_0",
        axis=axis,
        type="hinge",
    )

    child_1 = body.add(
        "body",
        name="link_1",
        pos=[0, 0.3, 0],
    )
    child_1.add(
        "geom",
        name="geom_1",
        type="capsule",
        size=rs.uniform(0.1, 1.0, 3),
    )
    axis = rs.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    child_1.add(
        "joint",
        name="joint_1",
        axis=axis,
        type="hinge",
    )

    child_2 = body.add(
        "body",
        name="link_2",
        pos=[0, 0, 0.3],
    )
    child_2.add(
        "geom",
        name="geom_2",
        type="capsule",
        size=rs.uniform(0.1, 1.0, 3),
    )
    axis = rs.uniform(-1, 1, 3)
    axis = axis / np.linalg.norm(axis)
    child_2.add(
        "joint",
        name="joint_2",
        axis=axis,
        type="hinge",
    )

    tokenize_detokenize(
        root,
        add_plane=False,
        check_qacc=False,
        qvelatol=5e-2,
        qaccatol=1e-1,
        qaccrtol=1e-1,
        disable_contact=True,
    )


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_free_body(seed: int):
    root = default_root_element()
    rs = np.random.RandomState(seed)
    body = root.worldbody.add(
        "body",
        name="link_0",
    )
    body.add(
        "geom",
        name="geom_0",
        **{
            **TEST_LINK_GEOM_ATTRS,
            "size": rs.uniform(0.01, 0.1, 3),
        },
    )
    body.add(
        "inertial",
        mass=rs.uniform(0.1, 1.0),
        diaginertia=[0.01, 0.01, 0.01],
        pos=rs.uniform(-0.1, 0.1, 3),
        quat=euler.euler2quat(
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
            rs.uniform(-np.pi, np.pi),
        ),
    )
    body.add("freejoint")
    tokenize_detokenize(root, add_plane=False)


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_multisite_body(seed: int):
    rs = np.random.RandomState(seed)
    root = default_root_element()
    body = root.worldbody.add(
        "body",
        name="link_0",
    )
    body.add("freejoint")
    body.add("geom", name="geom_0", size=rs.uniform(0.1, 1.0, 1), mass=1.0)
    body.add(
        "site",
        name="site_0",
        pos=rs.uniform(-0.1, 0.1, 3),
        group=TRACKING_SITE_GROUP,
    )
    body.add(
        "site",
        name="site_1",
        pos=rs.uniform(-0.1, 0.1, 3),
        group=TRACKING_SITE_GROUP,
    )
    sub_body = body.add("body", name="sub_body", pos=rs.uniform(-0.1, 0.1, 3) + 5)
    sub_body.add(
        "joint",
        name="joint_0",
        type="hinge",
        pos=rs.uniform(-0.1, 0.1, 3) + 5,
        damping=0.1,
    )
    sub_body.add("geom", name="geom_1", size=rs.uniform(0.1, 1.0, 1), mass=1.0)
    sub_body.add(
        "site",
        name="site_2",
        pos=rs.uniform(-0.1, 0.1, 3),
        group=TRACKING_SITE_GROUP,
    )
    # add actuator
    root.actuator.add(
        "position",
        name="actuator_0",
        joint="joint_0",
        ctrlrange=[-1, 1],
        forcerange=[-1, 1],
        kp=10,
        kv=1,
        ctrllimited=True,
    )
    tokenize_detokenize(root, disable_contact=True)


if __name__ == "__main__":
    # test_tokenize_detokenize_multisite_body(24)
    test_tokenize_detokenize_multisite_body(0)
