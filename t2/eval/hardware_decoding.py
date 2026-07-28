"""
Utilities used for getting discrete and continuous hardware parameters
from a network output.
"""

from pathlib import Path

import numpy as np
import ray
import zarr
from numpy.typing import NDArray


def check_zarr_path(
    masked_hardware_dict: dict[str, NDArray[np.float32 | np.int_]],
    num_links: int,
    num_actuators: int,
    num_dyna_joints: int,
    num_fixed_joints: int,
    zarr_path: Path,
) -> tuple[int, float, dict[str, NDArray[np.float32]]] | None:
    dist_keys = {
        # fixed joints
        "fixed_joint/id": 1.0,
        "fixed_joint/link/id": 1.0,
        "fixed_joint/pos": 1.0,
        "fixed_joint/rotmat": 5.0,
        # dynamic joints
        "dyna_joint/id": 1.0,
        "dyna_joint/link/id": 1.0,
        "dyna_joint/pos": 1.0,
        "dyna_joint/rotmat": 5.0,
        "dyna_joint/type": 1.0,
        "dyna_joint/range": 1.0,
        # "dyna_joint/armature": 1.0,
        # "dyna_joint/damping": 1.0,
        # "dyna_joint/frictionloss": 1.0,
        "dyna_joint/stiffness": 1.0,
        "dyna_joint/spring_ref": 1.0,
        "dyna_joint/qpos0_ref": 1.0,
        # links
        "link/geom_type": 1.0,
        "link/track_link/id": 1.0,
        "link/free_link/id": 1.0,
        "link/geom_size": 1.0,
        "link/mass": 1.0,
        # "link/ipos": 1.0,
        # "link/iquat": 1.0,
        # "link/imat": 1.0,
        # "link/diaginertia": 10.0,
        # "link/friction": 1.0,
        # "link/contact_dim": 1.0,
        "link/rgba": 1.0,
        # "link/id": 1.0,
        # actuators
        "actuator/dyna_joint/id": 1.0,
        # "actuator/ctrl_range": 1.0,
        # "actuator/force_range": 1.0,
        # "actuator/kp": 1.0,
        # "actuator/kv": 1.0,
        # "actuator/gear_ratio": 1.0,
        "actuator/type": 1.0,
        # "actuator/id": 1.0,
    }
    root = zarr.open(zarr_path, mode="r")

    link_ends = root["hardware_meta/link/ends"][:2]
    data_num_links = link_ends[1] - link_ends[0]
    if not data_num_links == num_links:
        return None
    fixed_joint_ends = root["hardware_meta/fixed_joint/ends"][:2]
    data_num_fixed_joints = fixed_joint_ends[1] - fixed_joint_ends[0]
    if not data_num_fixed_joints == num_fixed_joints:
        return None

    dyna_joint_ends = root["hardware_meta/dyna_joint/ends"][:2]
    data_num_dyna_joints = dyna_joint_ends[1] - dyna_joint_ends[0]
    if not data_num_dyna_joints == num_dyna_joints:
        return None

    actuator_ends = root["hardware_meta/actuator/ends"][:2]
    data_num_actuators = actuator_ends[1] - actuator_ends[0]
    if not data_num_actuators == num_actuators:
        return None

    group_seq_lens = {
        "link": num_links,
        "actuator": num_actuators,
        "dyna_joint": num_dyna_joints,
        "fixed_joint": num_fixed_joints,
    }
    num_hardware = root["hardware_meta/seed"].shape[0]
    hardware_dist = np.zeros((num_hardware,), dtype=float)
    dist_per_term = {}
    for group in ["link", "actuator", "dyna_joint", "fixed_joint"]:
        seq_len = group_seq_lens[group]
        data_group = root["hardware/" + group]
        data = data_group[:].reshape(num_hardware, seq_len, -1)
        attr_idx = 0
        data_dict = {}
        for k, attr_dim in sorted(dict(data_group.attrs).items(), key=lambda x: x[0]):
            attr_name = group + "/" + k
            if attr_name not in dist_keys:
                attr_idx += attr_dim
                continue
            data_dict[k] = data[..., attr_idx : attr_idx + attr_dim]
            attr_idx += attr_dim
        for k, data_attr in data_dict.items():
            attr_name = group + "/" + k
            weight = dist_keys[attr_name]
            if group in ["dyna_joint", "fixed_joint"]:
                num_joints = masked_hardware_dict[attr_name].shape[0] // 2
                # in io.py, we do
                # ```py
                # connections = list(sorted(jnt.connections, key=lambda x: x.link_idx))
                # ```
                # so order_1 is [0,1] (smaller link first)
                data_attr = data_attr.reshape(num_hardware, num_joints, 2, -1)
                order_2 = np.argsort(
                    masked_hardware_dict[group + "/link/id"].reshape(num_joints, 2),
                    axis=-1,
                )
                query_data_attr = masked_hardware_dict[attr_name].reshape(
                    num_joints, 2, -1
                )
                query_data_attr = np.array(
                    [
                        _query_data_attr[order]
                        for order, _query_data_attr in zip(order_2, query_data_attr)
                    ]
                )
            else:
                query_data_attr = masked_hardware_dict[attr_name].reshape(
                    1, seq_len, -1
                )
            if attr_name.endswith("/id") or attr_name.endswith("/type"):
                term_dist = np.where(
                    (
                        data_attr.reshape(num_hardware, -1).astype(np.int32)
                        == query_data_attr.reshape(1, -1).astype(np.int32)
                    ).all(axis=-1),
                    0,
                    np.inf,
                )
                hardware_dist += term_dist
            elif attr_name.endswith("/rotmat"):
                rotmat_1 = data_attr.reshape(num_hardware, seq_len, 3, 3)
                rotmat_2 = query_data_attr.reshape(1, seq_len, 3, 3)
                delta_rotmat = (
                    rotmat_2.transpose(0, 1, 3, 2) @ rotmat_1
                )  # (...,...,3,3)
                trace = np.trace(delta_rotmat, axis1=-2, axis2=-1)
                trace = np.clip(trace, min=-1 + 1e-8, max=3 - 1e-8)
                rotation_magnitude = np.arccos((trace - 1) / 2)
                rotation_magnitude = rotation_magnitude % (2 * np.pi)
                rotation_magnitude = np.minimum(
                    rotation_magnitude, 2 * np.pi - rotation_magnitude
                )
                term_dist = rotation_magnitude.mean(axis=1) * weight
                hardware_dist += term_dist
            else:
                term_dist = (
                    np.linalg.norm(
                        data_attr.reshape(num_hardware, -1)
                        - query_data_attr.reshape(1, -1),
                        axis=-1,
                    )
                    * weight
                )
                hardware_dist += term_dist
            dist_per_term[attr_name] = term_dist
    seeds = root["hardware_meta/seed"][:]
    return (
        int(seeds[np.argmin(hardware_dist)]),
        float(np.min(hardware_dist)),
        dist_per_term,
    )


check_zarr_path_ray = ray.remote(check_zarr_path).options(num_cpus=1)


def hardware_to_nearest_neighbor_params(
    masked_hardware_dict: dict[str, NDArray[np.float32 | np.int_]],
    num_links: int,
    num_actuators: int,
    num_dyna_joints: int,
    num_fixed_joints: int,
    robotoken_root_path: str,
    local_only: bool = False,
) -> tuple[list[int], int]:
    zarr_paths = list(Path(robotoken_root_path).glob("*.zarr"))
    zarr_path_to_best_seed = {}
    if local_only:
        results = [
            check_zarr_path(
                masked_hardware_dict=masked_hardware_dict,
                num_links=num_links,
                num_actuators=num_actuators,
                num_dyna_joints=num_dyna_joints,
                num_fixed_joints=num_fixed_joints,
                zarr_path=str(zarr_path),
            )
            for zarr_path in zarr_paths
        ]
    else:
        tasks = []
        for zarr_path in zarr_paths:
            tasks.append(
                check_zarr_path_ray.remote(
                    masked_hardware_dict=masked_hardware_dict,
                    num_links=num_links,
                    num_actuators=num_actuators,
                    num_dyna_joints=num_dyna_joints,
                    num_fixed_joints=num_fixed_joints,
                    zarr_path=str(zarr_path),
                )
            )

        results = ray.get(tasks)

    for zarr_path, result in zip(zarr_paths, results):
        if result is None:
            continue
        zarr_path_to_best_seed[str(zarr_path)] = result

    if len(zarr_path_to_best_seed) == 0:
        raise ValueError("No matching archetype found")
    min_dist = min(zarr_path_to_best_seed.values(), key=lambda x: x[1])[1]

    for k, (seed, dist, _) in zarr_path_to_best_seed.items():
        if dist > min_dist:
            continue
        choices = list(map(int, k.split("/")[-1].split(".")[0]))
        return choices, seed
    raise ValueError("No best seed found")
