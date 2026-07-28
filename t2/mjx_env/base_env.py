# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the Transformer Transformer authors.

# Modified from mujoco playground
# https://github.com/google-deepmind/mujoco_playground

import abc
from typing import Any, Mapping, Optional, Sequence, Union

import jax
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx


def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    act: Optional[jax.Array] = None,
    mocap_pos: Optional[jax.Array] = None,
    mocap_quat: Optional[jax.Array] = None,
    impl: str | None = None,
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
    device: Optional[jax.Device] = None,
) -> mjx.Data:
    """Initialize MJX Data."""
    contact_kwargs = {} if nconmax is None else {"nconmax": nconmax}
    try:
        data = mjx.make_data(
            model, impl=impl, njmax=njmax, device=device, **contact_kwargs
        )
    except TypeError:
        # mujoco-mjx >= 3.10 renamed nconmax -> naconmax
        if "nconmax" in contact_kwargs:
            contact_kwargs = {"naconmax": contact_kwargs.pop("nconmax")}
        data = mjx.make_data(
            model, impl=impl, njmax=njmax, device=device, **contact_kwargs
        )
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    if act is not None:
        data = data.replace(act=act)
    if mocap_pos is not None:
        data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
    if mocap_quat is not None:
        data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
    return data


def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
    def single_step(data, _):
        data = data.replace(ctrl=action)
        data = mjx.step(model, data)
        return data, None

    return jax.lax.scan(single_step, data, (), n_substeps)[0]


Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[tuple[int, ...], int]]]


@struct.dataclass
class State:
    """Environment state for training and inference."""

    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    info: dict[str, Any]

    def tree_replace(
        self, params: dict[str, Optional[jax.typing.ArrayLike]]
    ) -> "State":
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new


def _tree_replace(
    base: Any,
    attr: Sequence[str],
    val: Optional[jax.typing.ArrayLike],
) -> Any:
    """Sets attributes in a struct.dataclass with values."""
    if not attr:
        return base

    # special case for List attribute
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        raise NotImplementedError("List attributes are not supported.")

    if len(attr) == 1:
        return base.replace(**{attr[0]: val})

    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )


class MjxEnv(abc.ABC):
    action_size: int

    def __init__(
        self,
        ctrl_dt: float,
        sim_dt: float,
        nconmax: Optional[int] = None,
        njmax: Optional[int] = None,
    ):
        self._ctrl_dt = ctrl_dt
        self._sim_dt = sim_dt
        self._nconmax = nconmax
        self._njmax = njmax

    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

    @property
    def dt(self) -> float:
        """Control timestep for the environment."""
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        """Simulation timestep for the environment."""
        return self._sim_dt

    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))

    @property
    def observation_size(self) -> ObservationSize:
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        if isinstance(obs, Mapping):
            return jax.tree_util.tree_map(lambda x: x.shape, obs)
        return obs.shape[-1]

    def render(
        self,
        trajectory: list[mjx.Data],
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
        scene_option: Optional[mujoco.MjvOption] = None,
    ) -> Sequence[np.ndarray]:
        """Renders a trajectory using the MuJoCo renderer."""
        return render_array(
            mujoco_model=self.mj_model,
            trajectory=trajectory,
            height=height,
            width=width,
            camera=camera,
            scene_option=scene_option,
        )

    @property
    def unwrapped(self) -> "MjxEnv":
        return self


def render_array(
    mujoco_model: mujoco.MjModel,
    trajectory: list[mjx.Data],
    height: int = 240,
    width: int = 320,
    camera: str | None = None,
    scene_option: Optional[mujoco.MjvOption] = None,
) -> Sequence[np.ndarray]:
    """Returns a sequence of np.ndarray images using the MuJoCo renderer."""
    renderer = mujoco.Renderer(mujoco_model, height=height, width=width)
    camera = camera or -1
    d = mujoco.MjData(mujoco_model)

    def get_image(data: mjx.Data):
        d.qpos, d.qvel = data.qpos, data.qvel
        d.mocap_pos, d.mocap_quat = data.mocap_pos, data.mocap_quat
        d.xfrc_applied = data.xfrc_applied

        if hasattr(data, "mocap_pos") and hasattr(data, "mocap_quat"):
            d.mocap_pos, d.mocap_quat = data.mocap_pos, data.mocap_quat
        mujoco.mj_forward(mujoco_model, d)
        renderer.update_scene(d, camera=camera, scene_option=scene_option)
        return renderer.render()

    return [get_image(s) for s in trajectory]


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qvel."""
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qpos."""
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def get_qpos_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        qadr = model.jnt_qposadr[jnt]
        qdim = qpos_width(jnt_type)
        index_list.extend(range(qadr, qadr + qdim))
    return np.array(index_list)


def get_qvel_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        vadr = model.jnt_dofadr[jnt]
        vdim = dof_width(jnt_type)
        index_list.extend(range(vadr, vadr + vdim))
    return np.array(index_list)
