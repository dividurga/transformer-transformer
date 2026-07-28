import numpy as np
import pytest
from utils import tokenize_detokenize

from t2.env.mj_utils import default_root_element


@pytest.mark.parametrize("seed", range(100))
def test_tokenize_detokenize_ball_joint(seed: int):
    rs = np.random.RandomState(seed)
    root = default_root_element()
    body = root.worldbody.add(
        "body",
        name="link_0",
    )
    body.add(
        "geom",
        name="geom_0",
        type="capsule",
        size=[0.1, 0.4],
    )
    body.add(
        "joint",
        name="joint_0",
        type="ball",
        pos=rs.uniform(-1, 1, 3),
    )

    body2 = body.add(
        "body",
        name="link_1",
        pos=rs.uniform(-1, 1, 3),
    )
    body2.add(
        "geom",
        name="geom_1",
        type="capsule",
        size=[0.1, 0.5],
    )
    tokenize_detokenize(root, xatol=1e-8, xrotatol=1e-4, qatol=1e-8, qvelatol=1e-8)


def test_tokenize_detokenize_balljoint_connect_equality():
    root = default_root_element()
    body1 = root.worldbody.add(
        "body",
        name="link_0",
    )
    body1.add(
        "geom",
        name="geom_0",
        type="capsule",
        size=[0.05, 0.2],
        pos=[0, 0, 0.05],
    )
    body2 = body1.add(
        "body",
        name="link_1",
        pos=[0, 0, 0.4],
        euler=[0, np.pi / 2, 0],
    )
    body2.add(
        "geom",
        name="geom_1",
        type="capsule",
        size=[0.05, 0.3],
        pos=[0, 0, 0.3],
    )
    body2.add("joint", type="ball")

    body3 = body2.add(
        "body",
        name="link_2",
        pos=[0, 0, 0.6],
        euler=[0, np.pi / 2, 0],
    )
    body3.add(
        "geom",
        name="geom_2",
        type="capsule",
        size=[0.05, 0.4],
        pos=[0, 0, 0.4],
    )
    body3.add("joint", type="ball")

    body4 = body3.add(
        "body",
        name="link_3",
        pos=[0, 0, 0.8],
        euler=[0, np.pi / 2, 0],
    )
    body4.add(
        "geom",
        name="geom_3",
        type="capsule",
        size=[0.05, 0.3],
        pos=[0, 0, 0.3],
    )
    body4.add("joint", type="ball")

    root.equality.add("connect", body1="link_3", body2="link_0", anchor=[0, 0, 0.6])
    # disable qpos, qvel, qacc checks because ball joints and connect equalities
    # tokenize to the same object: ball joint tokens. However, during
    # detokenization, if a cycle in the kinematic chain is detected, one joint
    # gets turned into an equality constraint. This means this procedure could
    # lead to "missing joints" in the detokenized model, despite identical dynamics.
    tokenize_detokenize(
        root, xatol=5e-4, check_qpos=False, check_qvel=False, check_qacc=False
    )
