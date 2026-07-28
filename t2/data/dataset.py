import gc
import logging
import os
import shutil
from typing import Optional, Set

import numpy as np
import torch
import tqdm
import zarr
from zarr.core.group import Group

from t2.data.pad import pad
from t2.utils.misc import sorted_dict


class TimestepSampler:
    # samples forward in time
    def __init__(self, num_rollout_steps: int, step_size: int = 1):
        self.num_rollout_steps = num_rollout_steps
        self.step_size = step_size

    def __call__(
        self, rollout_idx: int, episode_starts: int, episode_ends: int
    ) -> tuple[int, np.ndarray]:
        indices = np.arange(
            rollout_idx,
            rollout_idx + self.num_rollout_steps * self.step_size,
            self.step_size,
        )
        return rollout_idx, indices


class CenteredTimestepSampler(TimestepSampler):
    def __call__(
        self, rollout_idx: int, episode_starts: int, episode_ends: int
    ) -> tuple[int, np.ndarray]:
        num_before = (self.num_rollout_steps - 1) // 2
        num_after = self.num_rollout_steps - num_before - 1
        indices = np.arange(
            rollout_idx - num_before * self.step_size,
            rollout_idx + (num_after + 1) * self.step_size,
            self.step_size,
        )
        return rollout_idx, indices


class RandomTimestepSampler(TimestepSampler):
    def __call__(
        self, rollout_idx: int, episode_starts: int, episode_ends: int
    ) -> tuple[int, np.ndarray]:
        episode_len = episode_ends - episode_starts
        indices = np.random.choice(
            episode_len, size=self.num_rollout_steps, replace=False
        )  # this will error out of episode length is too short
        indices = indices + episode_starts
        return indices.min(), indices


class EvenTimestepSampler(TimestepSampler):
    def __call__(
        self, rollout_idx: int, episode_starts: int, episode_ends: int
    ) -> tuple[int, np.ndarray]:
        # Sample indices evenly spaced across the episode range [starts, ends)
        episode_len = episode_ends - episode_starts
        num_samples = min(self.num_rollout_steps, max(episode_len, 0))

        if num_samples > 0:
            # Use linspace to distribute indices evenly, then cast to int
            indices = np.linspace(
                episode_starts, episode_ends - 1, num=num_samples, dtype=int
            )
        else:
            indices = np.zeros(0)

        # Pad to fixed length with masked-out zeros (consistent with RandomTimestepSampler)
        if len(indices) < self.num_rollout_steps:
            pad_len = self.num_rollout_steps - len(indices)
            indices = np.concatenate((indices, np.zeros(pad_len)))

        return episode_starts, indices


class T2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        group_seq_lens: dict[str, int],
        hardware_groups: Set[str],
        rollout_groups: Set[str],
        timestep_sampler: TimestepSampler,
        group_time_offsets: dict[str, int],
        load_into_memory: bool = False,
        use_pbar: bool = True,
        min_episode_length: int = 64,
        fast_launch: bool = False,
        use_cached_indices: bool = False,
        filter_episode_seeds: Optional[list[int]] = None,
        filter_hardware_seeds: Optional[list[int]] = None,
    ):
        self.path = path
        self.store = zarr.storage.LocalStore(path, read_only=True)
        self.root = zarr.open(self.store, mode="r")

        if load_into_memory:
            mem_store = zarr.storage.MemoryStore()
            self._copy_group(self.root, zarr.group(store=mem_store))
            self.store = mem_store
            self.root = zarr.open(self.store, mode="r")
        self.group_seq_lens = dict(group_seq_lens)
        # group_time_offsets used to support features like
        # - observe one action in the past
        # - observe one goal in the future
        self.group_time_offsets = group_time_offsets
        self.hardware_groups = hardware_groups
        self.rollout_groups = rollout_groups
        self.timestep_sampler = timestep_sampler
        # cache group roots
        self.group_roots = {}
        self.group_mask_roots = {}
        for group in self.hardware_groups:
            self.group_roots[group] = self.root[f"hardware/{group}"]
            # Check for legacy /mask path (Zarr v2) and new _mask path (Zarr v3)
            legacy_mask_key = f"hardware/{group}/mask"
            mask_key = f"hardware/{group}_mask"
            legacy_mask_path = path + "/" + legacy_mask_key
            mask_path = path + "/" + mask_key
            assert not os.path.exists(mask_path) and not os.path.exists(legacy_mask_path), (
                "don't expect mask for hardware groups"
            )
        for group in self.rollout_groups:
            self.group_roots[group] = self.root[f"rollout/{group}"]
            # Check for legacy /mask path (Zarr v2) first, then new _mask path (Zarr v3)
            legacy_mask_key = f"rollout/{group}/mask"
            mask_key = f"rollout/{group}_mask"
            legacy_mask_path = path + "/" + legacy_mask_key
            mask_path = path + "/" + mask_key

            # Try new _mask location first (Zarr v3), fall back to legacy /mask (Zarr v2)
            if os.path.exists(mask_path):
                logging.info(f"Loading mask for {group} from {mask_path}")
                self.group_mask_roots[group] = zarr.open(
                    zarr.storage.LocalStore(mask_path, read_only=True),
                    mode="r",
                )
            elif os.path.exists(legacy_mask_path):
                logging.info(f"Loading legacy mask for {group} from {legacy_mask_path}")
                self.group_mask_roots[group] = zarr.open(
                    zarr.storage.LocalStore(legacy_mask_path, read_only=True),
                    mode="r",
                )
        indices_path = path.replace(".zarr", ".zarr.idx")
        reindex_dataset = not os.path.exists(indices_path) or not use_cached_indices

        if reindex_dataset:
            fast_launch = False

        if not fast_launch:
            rollout_ends = self.root["rollout_meta/ends"][:].astype(np.uint64)
            rollout_seeds = self.root["rollout_meta/seed"][:].astype(np.uint64)
            rollout_hardware_ids = self.root["rollout_meta/hardware_id"][:].astype(
                np.uint64
            )
            hardware_seeds = self.root["hardware_meta/seed"][:].astype(np.uint64)
            hardware_actuator_ends = self.root["hardware_meta/actuator/ends"][:].astype(
                np.uint64
            )
            hardware_dyna_joint_ends = self.root["hardware_meta/dyna_joint/ends"][
                :
            ].astype(np.uint64)
            hardware_fixed_joint_ends = self.root["hardware_meta/fixed_joint/ends"][
                :
            ].astype(np.uint64)
            hardware_link_ends = self.root["hardware_meta/link/ends"][:].astype(
                np.uint64
            )

            seq_len_errors = []

            for meta_key, meta_data in [
                ("rollout_steps", rollout_ends),
                ("actuator", hardware_actuator_ends),
                ("dyna_joint", hardware_dyna_joint_ends),
                ("fixed_joint", hardware_fixed_joint_ends),
                ("link", hardware_link_ends),
            ]:
                if len(meta_data) == 1:
                    continue
                diffs = np.diff(meta_data, prepend=0).astype(int)
                if meta_key == "rollout_steps":
                    diffs = diffs[diffs >= min_episode_length]
                logging.info(
                    f"len({meta_key}): {diffs.min()}, {diffs.max()}, {int(diffs.mean())}"
                )
                if meta_key == "rollout_steps":
                    continue
                if group_seq_lens[meta_key] < diffs.max():
                    seq_len_errors.append(
                        (meta_key, int(diffs.max()), group_seq_lens[meta_key])
                    )
            if len(seq_len_errors) > 0:
                raise ValueError(f"Group sequence length errors: {seq_len_errors}")
        else:
            logging.info("Fast launch, skipping sequence length checks")

        if reindex_dataset:
            if os.path.exists(indices_path):
                # remove indices file
                shutil.rmtree(indices_path)
            # Precompute starts for all episodes and hardware
            episode_starts_arr = np.concatenate(([0], rollout_ends[:-1]))
            episode_lens = rollout_ends - episode_starts_arr
            valid_mask = episode_lens >= min_episode_length

            # Filter valid episodes
            valid_episode_idxs = np.where(valid_mask)[0]
            n_indices = int(episode_lens[valid_episode_idxs].sum())
            indices = np.empty((n_indices, 13), dtype=np.uint64)

            # For fast lookup
            actuator_starts_arr = np.concatenate(([0], hardware_actuator_ends[:-1]))
            dyna_joint_starts_arr = np.concatenate(([0], hardware_dyna_joint_ends[:-1]))
            fixed_joint_starts_arr = np.concatenate(
                ([0], hardware_fixed_joint_ends[:-1])
            )
            link_starts_arr = np.concatenate(([0], hardware_link_ends[:-1]))

            row_idx = 0
            for episode_idx in tqdm.tqdm(
                valid_episode_idxs,
                dynamic_ncols=True,
                desc="Indexing dataset",
                disable=not use_pbar,
            ):
                episode_starts = episode_starts_arr[episode_idx]
                episode_ends = rollout_ends[episode_idx]
                episode_seed = rollout_seeds[episode_idx]
                if (
                    filter_episode_seeds is not None
                    and episode_seed not in filter_episode_seeds
                ):
                    continue
                hardware_id = rollout_hardware_ids[episode_idx]
                hardware_seed = hardware_seeds[hardware_id]
                if (
                    filter_hardware_seeds is not None
                    and hardware_seed not in filter_hardware_seeds
                ):
                    continue
                actuator_starts = actuator_starts_arr[hardware_id]
                actuator_ends = hardware_actuator_ends[hardware_id]
                dyna_joint_starts = dyna_joint_starts_arr[hardware_id]
                dyna_joint_ends = hardware_dyna_joint_ends[hardware_id]
                fixed_joint_starts = fixed_joint_starts_arr[hardware_id]
                fixed_joint_ends = hardware_fixed_joint_ends[hardware_id]
                link_starts = link_starts_arr[hardware_id]
                link_ends = hardware_link_ends[hardware_id]
                rollout_range = np.arange(episode_starts, episode_ends, dtype=np.uint64)

                n_steps = int(episode_ends - episode_starts)
                block = np.empty((n_steps, 13), dtype=np.uint64)
                block[:, 0] = hardware_seed
                block[:, 1] = actuator_starts
                block[:, 2] = actuator_ends
                block[:, 3] = dyna_joint_starts
                block[:, 4] = dyna_joint_ends
                block[:, 5] = fixed_joint_starts
                block[:, 6] = fixed_joint_ends
                block[:, 7] = link_starts
                block[:, 8] = link_ends
                block[:, 9] = episode_seed
                block[:, 10] = rollout_range
                block[:, 11] = episode_starts
                block[:, 12] = episode_ends

                indices[row_idx : row_idx + n_steps] = block
                row_idx += n_steps

            self.indices = indices[:row_idx]
            zarr.save(indices_path, self.indices)
        else:
            self.indices = zarr.open(indices_path, mode="r")

    def __len__(self):
        return self.indices.shape[0]

    def get_rollout_data(
        self,
        curr_rollout_idx: int,
        episode_starts: int,
        episode_ends: int,
        rollout_idxs: np.ndarray,
    ):
        data_dict = {}
        for rollout_group in self.rollout_groups:
            group_rollout_idxs = rollout_idxs.copy() + self.group_time_offsets.get(
                rollout_group, 0
            )
            clipped_rollout_idxs = group_rollout_idxs.clip(
                episode_starts, episode_ends - 1
            )
            # allow masks before the episode, but not after
            group_root = self.group_roots[rollout_group]
            group_data = group_root[clipped_rollout_idxs]
            attrs = dict(group_root.attrs)
            dim_idx = 0
            for attr_name, attr_dim in sorted_dict(attrs).items():
                assert type(attr_dim) is int
                data_dict_key = f"{rollout_group}/{attr_name}"
                data_dict[data_dict_key] = np.array(
                    group_data[..., dim_idx : dim_idx + attr_dim]
                )
                data_dict[data_dict_key][
                    np.logical_or(
                        group_rollout_idxs < episode_starts,
                        group_rollout_idxs >= episode_ends,
                    )
                ] = 0.0  # pad with 0.0
                dim_idx += attr_dim

                data_seq_shape = data_dict[data_dict_key].shape[:-1]

                parent_group = "/".join(data_dict_key.split("/")[:-1])
                time_key = f"{parent_group}/time/id"
                if time_key not in data_dict:
                    relative_time_id = group_rollout_idxs - curr_rollout_idx
                    data_dict[time_key] = np.zeros(data_seq_shape + (1,))
                    data_dict[time_key][:] = relative_time_id[:, None, None]

                mask_key = f"{parent_group}/mask"
                if mask_key in data_dict:
                    continue
                # mask not present, so add mask

                # first check if mask is stored in data
                # otherwise, create it from data
                if parent_group in self.group_mask_roots:
                    mask_root = self.group_mask_roots[parent_group]
                    mask_data = mask_root[clipped_rollout_idxs].reshape(
                        data_seq_shape + (1,)
                    )
                    data_dict[mask_key] = mask_data
                else:
                    data_dict[mask_key] = np.zeros(
                        data_seq_shape + (1,),
                        dtype=bool,
                    )
        return data_dict

    def get_hardware_data(
        self,
        act_slice: slice,
        dyna_joint_slice: slice,
        fixed_joint_slice: slice,
        link_slice: slice,
    ):
        data_dict = {}
        for hardware_group in self.hardware_groups:
            if hardware_group == "actuator":
                data_slice = act_slice
            elif hardware_group == "dyna_joint":
                data_slice = dyna_joint_slice
            elif hardware_group == "fixed_joint":
                data_slice = fixed_joint_slice
            elif hardware_group == "link":
                data_slice = link_slice
            else:
                raise ValueError(f"Unknown hardware group: {hardware_group}")

            group_root = self.group_roots[hardware_group]
            group_data = group_root[data_slice, ...]
            attrs = dict(group_root.attrs)
            dim_idx = 0
            for attr_name, attr_dim in sorted_dict(attrs).items():
                assert type(attr_dim) is int
                data_dict_key = f"{hardware_group}/{attr_name}"
                assert data_dict_key not in data_dict
                data_dict[data_dict_key] = np.array(
                    group_data[..., dim_idx : dim_idx + attr_dim]
                )
                dim_idx += attr_dim
            assert dim_idx == group_root.shape[-1]
        return data_dict

    def __getitem__(self, idx: int):
        if idx % 10000 == 0:
            gc.collect()
        (
            hardware_seed,
            actuator_starts,
            actuator_ends,
            dyna_joint_starts,
            dyna_joint_ends,
            fixed_joint_starts,
            fixed_joint_ends,
            link_starts,
            link_ends,
            episode_seed,
            rollout_idx,
            episode_starts,
            episode_ends,
        ) = self.indices[idx]
        act_slice = slice(actuator_starts, actuator_ends)
        dyna_joint_slice = slice(dyna_joint_starts, dyna_joint_ends)
        fixed_joint_slice = slice(fixed_joint_starts, fixed_joint_ends)
        link_slice = slice(link_starts, link_ends)

        curr_rollout_idx, rollout_idxs = self.timestep_sampler(
            int(rollout_idx), int(episode_starts), int(episode_ends)
        )
        data_dict = {}
        data_dict.update(
            self.get_rollout_data(
                curr_rollout_idx=curr_rollout_idx,
                episode_starts=int(episode_starts),
                episode_ends=int(episode_ends),
                rollout_idxs=rollout_idxs,
            )
        )
        data_dict.update(
            self.get_hardware_data(
                act_slice,
                dyna_joint_slice,
                fixed_joint_slice,
                link_slice,
            )
        )

        data_dict = pad(
            data_dict,
            self.group_seq_lens,
        )
        data_dict["hardware_seed"] = np.array([hardware_seed])
        data_dict["episode_seed"] = np.array([episode_seed])
        data_dict["rollout_idx"] = np.array([rollout_idx])
        data_dict["episode_starts"] = np.array([episode_starts])
        data_dict["episode_ends"] = np.array([episode_ends])
        return data_dict

    @staticmethod
    def _copy_group(src: Group, dest: Group, compressors=None) -> None:
        """Recursively copy ``src`` into ``dest``."""
        for name in src.array_keys():
            arr = src[name]
            dest_arr = dest.create_array(
                name,
                shape=arr.shape,
                dtype=arr.dtype,
                chunks=arr.chunks,
                fill_value=getattr(arr, "fill_value", None),
                compressors=compressors,
            )
            dest_arr[...] = arr[...]
            dest_arr.attrs.update(dict(arr.attrs))
        for name in src.group_keys():
            sub_src = src[name]
            sub_dest = dest.create_group(name)
            sub_dest.attrs.update(dict(sub_src.attrs))
            T2Dataset._copy_group(sub_src, sub_dest)
