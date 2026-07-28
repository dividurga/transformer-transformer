import dataclasses
import datetime
import logging
import os
import shutil
import socket
import tempfile
import warnings
from typing import Any

# Suppress Zarr v3 warnings before importing zarr
# (we're aware these features aren't in the official spec yet)
warnings.filterwarnings(
    "ignore",
    message=r".*does not have a Zarr V3 specification.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*is currently not part in the Zarr format 3 specification.*",
)
# Also suppress by warning class name for more complete coverage
try:
    from zarr.core.dtype.npy.string import UnstableSpecificationWarning
    warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
except ImportError:
    pass

import git
import numpy as np
import ray
import tqdm
import zarr
import zarr.storage
from filelock import FileLock
from numpy.typing import NDArray
from ray.experimental import tqdm_ray

from t2.utils.misc import sorted_dict

ZarrDiskStore = zarr.storage.LocalStore | zarr.storage.ZipStore


def init_root(
    path: str,
    store_cls: type[ZarrDiskStore] = zarr.storage.LocalStore,
    root_metadata: dict[str, float | str | int | bool] | None = None,
    overwrite: bool = False,
) -> tuple[zarr.Group, ZarrDiskStore]:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {path}")
        else:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    attrs = {}
    attrs["metadata"] = {
        "git": git.Repo(search_parent_directories=True).head.object.hexsha,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
    }
    if root_metadata is not None:
        attrs.update(root_metadata)
    store_kwargs = {}
    if store_cls is zarr.storage.ZipStore:
        store_kwargs["mode"] = "w"
    store = store_cls(path, **store_kwargs)
    root = zarr.group(store, attributes=attrs)
    return root, store


COMPRESSOR = zarr.codecs.BloscCodec(
    cname="lz4hc",
    clevel=7,
    shuffle=zarr.codecs.BloscShuffle.noshuffle,
)

data_type_conversion = {
    np.dtype(np.float32): np.float32,
    np.dtype(np.float64): np.float32,
    np.dtype(np.int32): np.int32,
    np.dtype(np.int64): np.int32,
    np.dtype(np.bool_): bool,
    np.dtype(bool): bool,
}


@dataclasses.dataclass
class ZarrConfig:
    rollout_chunk: int = 100
    hardware_chunk: int = 32
    rollout_shard_multiplier: int = 100
    hardware_shard_multiplier: int = 100
    compressor: Any = COMPRESSOR


def append_hardware(
    hardware: dict[str, np.ndarray],
    data_path: str,
    zarr_config: ZarrConfig | None = None,
) -> int:
    """
    Append hardware data to a zarr store.

    Args:
        hardware: Dictionary containing hardware data
        data_path: Path to the zarr store
        zarr_config: Zarr config
    """
    if zarr_config is None:
        zarr_config = ZarrConfig()
    lock = FileLock(data_path + ".lock")
    with lock:
        if not os.path.exists(data_path):
            root, store = init_root(
                path=data_path,
                store_cls=zarr.storage.LocalStore,
            )
        else:
            store = zarr.storage.LocalStore(data_path)
            root = zarr.group(store)

        hardware_metadata = {
            k.split("metadata/")[1]: v
            for k, v in hardware.items()
            if k.startswith("metadata/")
        }
        assert "seed" in hardware_metadata

        # dump hardware data
        hardware_group_keys = [
            "link",
            "dyna_joint",
            "fixed_joint",
            "actuator",
        ]
        hardware_groups = {
            group_key: {
                "data": [],
                "attr_dims": {},
            }
            for group_key in hardware_group_keys
        }
        hardware_seq_len = {group_key: None for group_key in hardware_group_keys}
        for k, v in sorted_dict(hardware).items():
            prefix = k.split("/")[0]
            if prefix == "metadata":
                continue
            subgroup_key = k[len(prefix) + 1 :]
            if prefix not in hardware_groups:
                logging.warning(f"Unknown hardware group: {prefix}")
                continue
            group = hardware_groups[prefix]
            group["data"].append(v)
            group["attr_dims"][subgroup_key] = v.shape[-1]
            if hardware_seq_len[prefix] is None:
                hardware_seq_len[prefix] = v.shape[0]
            else:
                assert hardware_seq_len[prefix] == v.shape[0]

        hardware_metadata = {
            "dyna_joint/ends": -1,
            "fixed_joint/ends": -1,
            "link/ends": -1,
            "actuator/ends": -1,
            **hardware_metadata,
        }
        for group_key, group in hardware_groups.items():
            full_key = f"hardware/{group_key}"
            subgroup_key = group_key.split("/")[0]
            data = np.concatenate(group["data"], axis=-1)
            if full_key not in root:
                subroot = root.create_array(
                    full_key,
                    shape=data.shape,
                    dtype=np.float32,
                    chunks=(zarr_config.hardware_chunk, *list(data.shape[1:])),
                    shards=(
                        zarr_config.hardware_chunk
                        * zarr_config.hardware_shard_multiplier,
                        *list(data.shape[1:]),
                    ),
                    compressors=zarr_config.compressor,
                )
                for k, v in sorted_dict(group["attr_dims"]).items():
                    subroot.attrs[k] = v
                subroot[:] = data
                if hardware_metadata[subgroup_key + "/ends"] == -1:
                    hardware_metadata[subgroup_key + "/ends"] = data.shape[0]
                else:
                    assert hardware_metadata[subgroup_key + "/ends"] == data.shape[0]
            else:
                # append to existing array
                root[full_key].append(data)
                # check that attributes match
                for k, v in group["attr_dims"].items():
                    assert root[full_key].attrs[k] == v
                if hardware_metadata[subgroup_key + "/ends"] == -1:
                    hardware_metadata[subgroup_key + "/ends"] = root[full_key].shape[0]
                else:
                    assert (
                        hardware_metadata[subgroup_key + "/ends"]
                        == root[full_key].shape[0]
                    )

        hardware_idx = None

        for meta_key, meta_value in hardware_metadata.items():
            is_string = isinstance(meta_value, str)
            meta_value = np.array([meta_value])
            if f"hardware_meta/{meta_key}" not in root:
                if is_string:
                    dtype = meta_value.dtype  # Use numpy's inferred string dtype
                elif meta_key.endswith("/ends") or meta_key == "seed":
                    dtype = np.uint64
                elif (
                    meta_key == "predicted_value"
                    or meta_key.startswith("actual_value")
                    or meta_key == "optimize_time"
                ):
                    dtype = np.float32
                else:
                    raise ValueError(f"Unknown meta key: {meta_key}")
                subroot = root.create_array(
                    f"hardware_meta/{meta_key}",
                    shape=(1,),
                    dtype=dtype,
                    chunks=(zarr_config.hardware_chunk,),
                    shards=(
                        zarr_config.hardware_chunk
                        * zarr_config.hardware_shard_multiplier,
                    ),
                    compressors=None,
                )
                subroot[:] = meta_value
            else:
                root[f"hardware_meta/{meta_key}"].append(meta_value)
            if hardware_idx is None:
                hardware_idx = int(root[f"hardware_meta/{meta_key}"].shape[0] - 1)
            else:
                assert hardware_idx == root[f"hardware_meta/{meta_key}"].shape[0] - 1
    lock.release()
    assert hardware_idx is not None
    return hardware_idx


ALLOWED_ROLLOUT_GROUPS = [
    "track_link_obs",
    "free_link_obs",
    "dyna_joint_obs",
    "actuator_obs",
    "ctrl",
    "metric",
    "target_pose",
    "done",
]


def append_episode(
    hardware_idx: int,
    episode_seed: int,
    episode_data: dict[str, NDArray[np.float32 | np.int_ | np.bool_]],
    data_path: str,
    zarr_config: ZarrConfig | None = None,
):
    """
    Append episode data to a zarr store.

    Args:
        hardware_idx: Index of the hardware data
        episode_seed: Seed used for the episode
        episode_data: Dictionary containing episode data
        data_path: Path to the zarr store
        zarr_config: Zarr config
    """
    if zarr_config is None:
        zarr_config = ZarrConfig()
    lock = FileLock(data_path + ".lock")
    with lock:
        if not os.path.exists(data_path):
            root, store = init_root(
                path=data_path,
                store_cls=zarr.storage.LocalStore,
            )
        else:
            store = zarr.storage.LocalStore(data_path)
            root = zarr.group(store)

        episode_meta = {
            "ends": -1,
            "hardware_id": hardware_idx,
            "seed": episode_seed,
        }

        # dump rollout data
        rollout_groups = {}
        for k, v in sorted_dict(episode_data).items():
            prefix = k.split("/")[0]
            subgroup_key = k[len(prefix) + 1 :]
            assert prefix in ALLOWED_ROLLOUT_GROUPS
            if prefix not in rollout_groups:
                rollout_groups[prefix] = {
                    "data": [],
                    "attr_dims": {},
                }
            group = rollout_groups[prefix]
            group["data"].append(v)
            group["attr_dims"][subgroup_key] = v.shape[-1]

        for group_key, group in rollout_groups.items():
            full_key = f"rollout/{group_key}"
            data = np.concatenate(group["data"], axis=-1)
            empty_data = np.prod(data.shape) == 0
            if full_key not in root:
                save_dtype = data_type_conversion.get(np.dtype(data.dtype), data.dtype)
                if empty_data:
                    subroot = root.create_array(
                        full_key,
                        shape=data.shape,
                        dtype=save_dtype,
                        compressors=zarr_config.compressor,
                    )
                else:
                    subroot = root.create_array(
                        full_key,
                        shape=data.shape,
                        dtype=save_dtype,
                        chunks=(
                            zarr_config.rollout_chunk,
                            *list(data.shape[1:]),
                        ),
                        shards=(
                            zarr_config.rollout_chunk
                            * zarr_config.rollout_shard_multiplier,
                            *list(data.shape[1:]),
                        ),
                        compressors=zarr_config.compressor,
                    )
                for k, v in sorted_dict(group["attr_dims"]).items():
                    subroot.attrs[k] = v
                subroot[:] = data
            else:
                # append to existing array
                root[full_key].append(data)
                # check that attributes match
                for k, v in group["attr_dims"].items():
                    assert root[full_key].attrs[k] == v

            rollout_size = root[full_key].shape[0]
            if episode_meta["ends"] == -1:
                episode_meta["ends"] = rollout_size
            else:
                assert episode_meta["ends"] == rollout_size

        for meta_key, meta_value in episode_meta.items():
            meta_value = np.array([meta_value])
            if f"rollout_meta/{meta_key}" not in root:
                subroot = root.create_array(
                    f"rollout_meta/{meta_key}",
                    shape=(1,),
                    dtype=np.uint64,
                    chunks=(zarr_config.rollout_chunk,),
                    shards=(
                        zarr_config.rollout_chunk
                        * zarr_config.rollout_shard_multiplier,
                    ),
                    compressors=None,
                )
                subroot[:] = meta_value
            else:
                root[f"rollout_meta/{meta_key}"].append(meta_value)
        store.close()
    lock.release()


def get_flattened_leaf_keys(root: zarr.Group, prefix: str = "") -> list[str]:
    flattened_keys = []
    for key in root.keys():
        if isinstance(root[key], zarr.Group):
            flattened_keys.extend(
                get_flattened_leaf_keys(root[key], os.path.join(prefix, key))
            )
        elif isinstance(root[key], zarr.Array):
            flattened_keys.append(os.path.join(prefix, key))
    return flattened_keys


def concat_zarr_stores(
    from_paths: list[str],
    to_path: str,
    root_metadata: dict[str, float | str | int | bool] | None = None,
    zarr_config: ZarrConfig | None = None,
    use_pbar: bool = True,
    overwrite: bool = False,
    flattened_keys: list[str] | None = None,
) -> tuple[zarr.Group, ZarrDiskStore]:
    if zarr_config is None:
        zarr_config = ZarrConfig()
    lock = FileLock(to_path + ".lock")
    with lock:
        if not os.path.exists(to_path):
            target_root, target_store = init_root(
                path=to_path,
                store_cls=zarr.storage.LocalStore,
                root_metadata=root_metadata,
            )
        else:
            if not overwrite:
                raise FileExistsError(f"Dataset already exists: {to_path}")
            target_store = zarr.storage.LocalStore(to_path)
            target_root = zarr.group(target_store)

    from_stores = {}
    from_roots = {}
    leaf_shapes = {}
    data_attrs = {}
    for from_path in from_paths:
        from_stores[from_path] = zarr.storage.LocalStore(from_path)
        from_roots[from_path] = zarr.group(from_stores[from_path])
        if flattened_keys is None:
            flattened_keys = get_flattened_leaf_keys(from_roots[from_path])
        for key in flattened_keys:
            if key not in leaf_shapes:
                leaf_shapes[key] = []
            leaf_shapes[key].append(from_roots[from_path][key].shape)
            if key not in data_attrs:
                data_attrs[key] = dict(from_roots[from_path][key].attrs)
            else:
                assert data_attrs[key] == dict(from_roots[from_path][key].attrs)
    int_dtype_range = {
        np.int32: (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        np.int64: (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        np.uint32: (np.iinfo(np.uint32).min, np.iinfo(np.uint32).max),
        np.uint64: (np.iinfo(np.uint64).min, np.iinfo(np.uint64).max),
    }
    idx_dtype = np.uint64
    with tqdm.tqdm(
        sorted_dict(leaf_shapes).items(),
        desc="Concatenating zarr stores",
        disable=not use_pbar,
    ) as pbar:
        for k, shapes in pbar:
            ref_shape = shapes[0]
            key_requires_mask = False  # only needs masks if not all keys match shapes
            if len(ref_shape) > 1:
                for shape in shapes[1:]:
                    assert ref_shape[-1] == shape[-1], (
                        f"Last dimension mismatch: {ref_shape[-1]} != {shape[-1]}, {k}"
                    )
                    if ref_shape[1:] != shape[1:]:
                        ref_shape = tuple([max(x, y) for x, y in zip(ref_shape, shape)])
                        key_requires_mask = True
            concat_shape = (sum(shape[0] for shape in shapes), *ref_shape[1:])
            mask_shape = concat_shape[
                :-1
            ]  # last dimension must match, so we can ignore it for masks

            if k.startswith("rollout"):
                chunk_size = zarr_config.rollout_chunk
                shard_size = chunk_size * zarr_config.rollout_shard_multiplier
            elif k.startswith("hardware"):
                chunk_size = zarr_config.hardware_chunk
                shard_size = chunk_size * zarr_config.hardware_shard_multiplier
            else:
                raise ValueError(f"Unknown group: {k}")

            # Detect dtype from first source array
            first_from_path = from_paths[0]
            first_dtype = from_roots[first_from_path][k].dtype
            is_string_array = np.issubdtype(first_dtype, np.str_) or (
                hasattr(first_dtype, "name") and "string" in first_dtype.name.lower()
            )

            dtype = np.float32
            # TODO parse dtype from previous datasets
            compressor = zarr_config.compressor
            if is_string_array:
                dtype = str  # Use string dtype for string arrays
                compressor = None
            elif (
                k.startswith("hardware_meta/") or k.startswith("rollout_meta/")
            ) and not ("value" in k or "optimize_time" in k):
                dtype = idx_dtype
                compressor = None

            created_target_root = False

            ends_offset = 0
            hardware_offset = 0
            idx = 0
            for from_path in tqdm.tqdm(
                from_paths,
                desc=f"Copying {k}",
                dynamic_ncols=True,
                leave=False,
                disable=not use_pbar,
            ):
                try:
                    if is_string_array:
                        # Don't cast string arrays
                        data = np.array(from_roots[from_path][k][:])
                    else:
                        data = np.array(from_roots[from_path][k][:]).astype(dtype)
                except Exception as e:
                    logging.error(f"Error copying {k} from {from_path}")
                    raise e
                if not created_target_root:
                    if is_string_array:
                        # String arrays need special handling - use the data's dtype directly
                        array_dtype = data.dtype
                    else:
                        array_dtype = data_type_conversion.get(
                            np.dtype(data.dtype), dtype
                        )
                    target_root.create_array(
                        k,
                        shape=concat_shape,
                        dtype=array_dtype,
                        chunks=(chunk_size, *list(concat_shape[1:])),
                        shards=(shard_size, *list(concat_shape[1:])),
                        compressors=compressor,
                        attributes=data_attrs[k],
                    )
                    if key_requires_mask:
                        # Store mask as sibling array with _mask suffix (not /mask child)
                        # to avoid Zarr v3 restriction on child nodes under arrays
                        target_root.create_array(
                            f"{k}_mask",
                            shape=mask_shape,
                            dtype=bool,
                            chunks=(chunk_size, *list(mask_shape[1:])),
                            shards=(shard_size, *list(mask_shape[1:])),
                            compressors=compressor,
                        )
                    created_target_root = True
                if k.endswith("/ends"):
                    data = data + ends_offset
                    ends_offset = data.max()
                    data_max = data.max()
                    data_min = data.min()
                    assert data_min >= int_dtype_range[idx_dtype][0]
                    assert data_max <= int_dtype_range[idx_dtype][1]
                elif k == "rollout_meta/hardware_id":
                    data = data + hardware_offset
                    hardware_offset = data.max() + 1
                if data.shape[1:] != ref_shape[1:]:
                    assert key_requires_mask
                    assert len(ref_shape) == 3, (
                        "special case, only support 3d for now (T, L, D)"
                        + "where T is time, L is length, and D is dimension"
                    )
                    mask = np.zeros((data.shape[0], ref_shape[1]), dtype=bool)
                    is_padding_indices = ref_shape[1] - data.shape[1]
                    assert is_padding_indices >= 0
                    mask[:, -is_padding_indices:] = True
                    target_root[f"{k}_mask"][idx : idx + data.shape[0]] = mask
                    target_root[k][idx : idx + data.shape[0], : data.shape[1]] = data
                else:
                    target_root[k][idx : idx + data.shape[0]] = data
                idx += data.shape[0]

    for store in from_stores.values():
        store.close()

    return target_root, target_store


@ray.remote
def recursive_concat_zarr_stores(
    from_paths: list[str],
    to_path: str,
    pbar: tqdm_ray.tqdm | None = None,
    max_from_paths: int = 10,
):
    if not from_paths:
        raise ValueError("from_paths cannot be empty")

    if len(from_paths) <= max_from_paths:
        concat_zarr_stores(
            from_paths=from_paths,
            to_path=to_path,
            use_pbar=False,
        )
        if pbar is not None:
            pbar.update.remote(len(from_paths))
        return to_path
    else:
        midpoint = len(from_paths) // 2
        path_groups = [
            from_paths[midpoint:],
            from_paths[:midpoint],
        ]
        to_path_groups = []

        try:
            # Create temp directories
            to_path_groups = [
                tempfile.mkdtemp(suffix=".zarr") for i in range(len(path_groups))
            ]

            # Process in parallel
            ray.get(
                [
                    recursive_concat_zarr_stores.remote(
                        path_group,
                        to_path_group,
                        pbar=pbar,
                    )
                    for path_group, to_path_group in zip(path_groups, to_path_groups)
                ]
            )

            # Final concatenation
            concat_zarr_stores(
                from_paths=to_path_groups,
                to_path=to_path,
                use_pbar=False,
            )
        finally:
            # Always cleanup temp directories
            for to_path_group in to_path_groups:
                if os.path.exists(to_path_group):
                    try:
                        shutil.rmtree(to_path_group)
                    except Exception as e:
                        logging.warning(f"Failed to cleanup {to_path_group}: {e}")

        return to_path


if __name__ == "__main__":
    concat_zarr_stores(
        from_paths=["/tmp/test1.zarr", "/tmp/test2.zarr"],
        to_path="/tmp/test_concat.zarr",
    )
