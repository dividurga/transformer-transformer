import numpy as np

TWO_DIMENSIONAL_GROUPS = {
    "free_link_obs",
    "dyna_joint_obs",
    "actuator_obs",
    "track_link_obs",
    "ctrl",
    "target_pose",
    "metric",
}


def pad(
    data_dict: dict[str, np.ndarray],
    group_seq_lens: dict[str, int],
    pad_repeat_along_time: bool = False,
):
    padded_data_dict = {}
    group_seq_len = {}
    # assume data_dict is unbatched
    rollout_steps = int(group_seq_lens["rollout_steps"])
    for k, v in data_dict.items():
        if k.endswith("/id"):
            assert len(k.split("/")) >= 3, (
                "Should only load secondary ids (used for references) from the dataset, such as "
                "`dyna_joint/link/id` or `link/track_link/id`, but not primary ids. This is because "
                "primary ids are used as positional embeddings at inference time, and so we must know"
                "them apriori. During training, these primary ids are added using `AddPositionId` augmentation."
                f"Got {k}."
            )
        group = k.split("/")[0]
        group_mask_key = group + "/mask"
        seq_len = v.shape[0]
        target_seq_len = group_seq_lens[group]
        if group in TWO_DIMENSIONAL_GROUPS:
            assert v.ndim == 3
            target_seq_len *= rollout_steps

        # if we don't have data for the mask to fill the tensor, then the remaining values
        # of the padded mask should be True (meaning `is_padding` for torch self attention)
        fill_tensor_fn = np.zeros if not k.endswith("/mask") else np.ones
        mask_not_yet_present = group_mask_key not in padded_data_dict
        mask_not_in_data = group_mask_key not in data_dict
        should_add_mask = mask_not_yet_present and mask_not_in_data

        if group in TWO_DIMENSIONAL_GROUPS:
            target_element_len = -1
            if group.endswith("_obs"):
                target_element_len = group_seq_lens[group.replace("_obs", "")]
            elif group in {"ctrl", "target_pose", "metric"}:
                target_element_len = group_seq_lens[group]
            else:
                raise ValueError(f"Unknown group: {group}")
            assert target_element_len != -1, f"target_element_len is -1 for {k}"
            shape = (
                rollout_steps,
                group_seq_lens[group],
                *v.shape[2:],
            )
            time_dim, seq_dim = v.shape[:2]
            padded_data_dict[k] = fill_tensor_fn(shape, dtype=v.dtype)
            padded_data_dict[k][:time_dim, :seq_dim] = v
            if pad_repeat_along_time and time_dim > 0 and time_dim < rollout_steps:
                padded_data_dict[k][time_dim:, :seq_dim] = padded_data_dict[k][
                    [time_dim - 1], :seq_dim
                ]
            if group not in group_seq_len:
                group_seq_len[group] = (time_dim, seq_dim)
                if should_add_mask:
                    padded_data_dict[group_mask_key] = np.ones(
                        (rollout_steps, group_seq_lens[group], 1), dtype=bool
                    )
                    padded_data_dict[group_mask_key][:time_dim, :seq_dim] = False
            else:
                assert group_seq_len[group] == (time_dim, seq_dim)
            padded_data_dict[k] = padded_data_dict[k].reshape(
                target_seq_len, *v.shape[2:]
            )
        else:
            shape = (target_seq_len, *v.shape[1:])
            padded_data_dict[k] = fill_tensor_fn(shape, dtype=v.dtype)
            # zero pad
            padded_data_dict[k][:seq_len] = v
            if group not in group_seq_len:
                group_seq_len[group] = seq_len
                if should_add_mask:
                    padded_data_dict[group_mask_key] = np.zeros(
                        (target_seq_len, 1), dtype=bool
                    )
                    padded_data_dict[group_mask_key][seq_len:] = True
            else:
                assert group_seq_len[group] == seq_len

    padded_data_dict = {
        k: (v.reshape(-1, v.shape[1]) if not k.endswith("/mask") else v.reshape(-1, 1))
        for k, v in padded_data_dict.items()
    }
    return padded_data_dict
