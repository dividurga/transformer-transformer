from typing import Set

import numpy as np
import torch
import torch.nn as nn
import tqdm
import zarr

from t2.data.normalization import Normalization
from t2.model.modality import Modality, TensorFn
from t2.utils.misc import sorted_dict

QUANTILES = [0.0, 100]


def compute_dataset_normalization(
    path: str,
    hardware_groups: Set[str],
    rollout_groups: Set[str],
    modalities: dict[str, tuple[Modality, dict[str, TensorFn]]],
    skip_ids: bool = True,
    max_seq_len: int = 1_000_000,  # beyond this length, just subsample to estimate normalization
) -> nn.ModuleDict:
    store = zarr.storage.LocalStore(path, read_only=True)
    root = zarr.open(store, mode="r")
    # cache group roots
    group_roots = {}
    for group in hardware_groups:
        group_roots[group] = root[f"hardware/{group}"]
    for group in rollout_groups:
        group_roots[group] = root[f"rollout/{group}"]

    normalization_dict = nn.ModuleDict()

    group_pbar = tqdm.tqdm(
        rollout_groups,
        dynamic_ncols=True,
    )

    for rollout_group in group_pbar:
        group_root = group_roots[rollout_group]
        group_pbar.set_description(f"Computing {rollout_group} normalization")
        attrs = group_root.attrs
        dim_idx = 0
        if rollout_group in modalities:
            modality, attr_encoders = modalities[rollout_group]
        else:
            modality = None
            attr_encoders = {}
        if group_root.shape[0] > max_seq_len:
            group_indices = np.random.choice(
                group_root.shape[0], max_seq_len, replace=False
            )
            group_root = group_root[group_indices]
        for attr_name, attr_dim in sorted_dict(attrs).items():
            assert type(attr_dim) == int
            data_dict_key = f"{rollout_group}/{attr_name}"
            if skip_ids and data_dict_key.endswith("/id"):
                dim_idx += attr_dim
                continue
            assert data_dict_key not in normalization_dict
            data = np.array(group_root[:, ..., dim_idx : dim_idx + attr_dim])
            if attr_name in attr_encoders:
                data = attr_encoders[attr_name](torch.from_numpy(data)).numpy()
            if modality is not None and attr_name in modality.attrs:
                dim = modality.attrs[attr_name]
                assert data.shape[-1] == dim
                data = data.reshape(-1, dim)
            else:
                assert data.shape[-1] == attr_dim
                data = data.reshape(-1, attr_dim)
            if data_dict_key.endswith("/rotmat"):
                # Rotation matrices should always have range [-1, 1]
                normalization_dict[data_dict_key] = Normalization(
                    quantiles={
                        "q0/0": torch.full((data.shape[-1],), -1.0),
                        "q100/0": torch.full((data.shape[-1],), 1.0),
                    },
                )
            else:
                normalization_dict[data_dict_key] = Normalization(
                    quantiles={
                        f"q{q:.1f}".replace(".", "/"): torch.from_numpy(
                            np.percentile(data, q, axis=0)
                        )
                        for q in QUANTILES
                    },
                )
            dim_idx += attr_dim
        assert dim_idx == group_root.shape[-1]

    group_pbar.close()

    group_pbar = tqdm.tqdm(
        hardware_groups,
        dynamic_ncols=True,
    )
    for hardware_group in group_pbar:
        group_root = group_roots[hardware_group]
        group_pbar.set_description(f"Computing {hardware_group} normalization")
        attrs = group_root.attrs
        if group_root.shape[0] > max_seq_len:
            group_indices = np.random.choice(
                group_root.shape[0], max_seq_len, replace=False
            )
            group_root = group_root[group_indices]
        dim_idx = 0
        if hardware_group in modalities:
            modality, attr_encoders = modalities[hardware_group]
        else:
            modality = None
            attr_encoders = {}
        for attr_name, attr_dim in sorted_dict(attrs).items():
            assert type(attr_dim) == int
            data_dict_key = f"{hardware_group}/{attr_name}"
            if skip_ids and data_dict_key.endswith("/id"):
                dim_idx += attr_dim
                continue
            assert data_dict_key not in normalization_dict
            data = np.array(group_root[..., dim_idx : dim_idx + attr_dim])
            if attr_name in attr_encoders and attr_name in modality.attrs:
                # has encoder and should encode
                data = attr_encoders[attr_name](torch.from_numpy(data)).numpy()
            if modality is not None and attr_name in modality.attrs:
                dim = modality.attrs[attr_name]
                assert data.shape[-1] == dim
                data = data.reshape(-1, dim)
            else:
                assert data.shape[-1] == attr_dim
                data = data.reshape(-1, attr_dim)
            if data_dict_key.endswith("/rotmat"):
                # Rotation matrices should always have range [-1, 1]
                normalization_dict[data_dict_key] = Normalization(
                    quantiles={
                        "q0/0": torch.full((data.shape[-1],), -1.0),
                        "q100/0": torch.full((data.shape[-1],), 1.0),
                    },
                )
            else:
                normalization_dict[data_dict_key] = Normalization(
                    quantiles={
                        f"q{q:.1f}".replace(".", "/"): torch.from_numpy(
                            np.percentile(data, q, axis=0)
                        )
                        for q in QUANTILES
                    },
                )
            dim_idx += attr_dim
        assert dim_idx == group_root.shape[-1]

    group_pbar.close()

    return normalization_dict


def save_normalization_dict(normalization: nn.ModuleDict, path: str) -> None:
    """Save normalization ModuleDict to ``path`` using ``torch.save``."""
    torch.save(normalization.state_dict(), path)


def load_normalization_dict(path: str) -> nn.ModuleDict:
    """Load a normalization ModuleDict saved by :func:`save_normalization_dict`."""
    state_dict = torch.load(path, map_location="cpu")
    return normalization_dict_from_state_dict(state_dict)


def normalization_dict_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> nn.ModuleDict:
    """Convert a state dict produced from a normalization ModuleDict back into the ModuleDict."""
    names = {k.split(".")[0] for k in state_dict.keys()}
    modules = {}
    for name in sorted(names):
        quantiles = {
            k.split(name + ".quantiles.")[1]: state_dict[k]
            for k in state_dict.keys()
            if k.startswith(name + ".quantiles.")
        }
        modules[name] = Normalization(
            quantiles=quantiles,
        )
    return nn.ModuleDict(modules)


if __name__ == "__main__":
    import hydra

    with hydra.initialize(config_path="../../config", version_base="1.3"):
        cfg = hydra.compose(config_name="train")
    model = hydra.utils.instantiate(cfg.model)
    modalities = {}
    for task in model.tasks.values():
        for adapter in task.adapters.values():
            modalities[adapter.modality.name] = (
                adapter.modality,
                adapter.attr_encoders,
            )

    normalization_dict = compute_dataset_normalization(
        path="/tmp/test.zarr",
        hardware_groups={"actuator", "joint", "link"},
        rollout_groups={
            "ctrl",
            "link_obs",
            "joint_obs",
            "actuator_obs",
            "metric",
            "target_pose",
        },
        modalities=modalities,
    )
    for k, v in sorted(normalization_dict.items(), key=lambda x: x[0]):
        print(k)
        print(v.vmin)
        print(v.vmax)
