import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from dm_control import mjcf
from omegaconf import OmegaConf
from visualize_robotoken_diffusion import (
    build_anchor_positions,
    parse_physics_state,
)

from t2.env.mj_utils import set_up_default_scene
from t2.robotok.io import deserialize
from t2.robotok.token import GROUND_LINK_ID
from t2.robotok.tokenizer import detokenize

# Discrete/categorical keys copied from the final denoising step into every
# intermediate step, so geometry topology stays fixed while values denoise.
OVERRIDE_KEYS: list[str] = list(
    OmegaConf.load(Path(__file__).parent.parent / "config" / "hardware_gen_vis.yaml")[
        "override_keys"
    ]
)


def post_process_robot_dict(
    robot_dict: dict[str, torch.Tensor],
    last_robot_dict: dict[str, torch.Tensor],
    override_keys: list[str],
) -> dict[str, np.ndarray]:
    # Tensors are already unbatched by the optimizer (see
    # visualize_robotoken_diffusion.py, which selects the best seed before
    # caching), so there is no batch dimension to strip here.
    robot_dict_cpu = {k: v.cpu().numpy() for k, v in robot_dict.items()}
    for key in override_keys:
        if key not in last_robot_dict:
            continue
        robot_dict_cpu[key] = last_robot_dict[key].cpu().numpy()
    masked_robot_dict = {}
    # preprocess robot dict to remove masked predictions
    for k, v in robot_dict_cpu.items():
        if k.endswith("/mask"):
            continue
        group = k.split("/")[0]
        mask_key = group + "/mask"
        mask = last_robot_dict[mask_key].reshape(-1).cpu().numpy().astype(bool)
        if mask_key == "link/mask":
            if len(v.shape) == 2:
                mask = (
                    torch.from_numpy(mask[:, None]).expand_as(torch.from_numpy(v)).cpu()
                )
            # keep all vectors
            masked_robot_dict[k] = v
            if k == "link/geom_type":
                masked_robot_dict[k] = np.where(
                    mask, np.ones_like(v), masked_robot_dict[k]
                )
            elif k == "link/track_link/id":
                masked_robot_dict[k] = np.where(
                    mask, -np.ones_like(v), masked_robot_dict[k]
                )
            elif k == "link/free_link/id":
                target = -np.ones_like(v)
                masked_robot_dict[k] = np.where(mask, target, masked_robot_dict[k])
            elif k == "link/geom_size":
                masked_robot_dict[k] = np.where(
                    mask, np.ones_like(v) * 0.0001, masked_robot_dict[k]
                )
            elif k == "link/mass":
                masked_robot_dict[k] = np.where(
                    mask, np.ones_like(v) * 0.1, masked_robot_dict[k]
                )
            elif k == "link/ipos":
                masked_robot_dict[k] = np.where(
                    mask, np.zeros_like(v), masked_robot_dict[k]
                )
            elif k == "link/iquat":
                target = np.zeros_like(v)
                target[..., 0] = 1
                masked_robot_dict[k] = np.where(mask, target, masked_robot_dict[k])
            elif k == "link/imat":
                target = np.zeros_like(v)
                target[..., :] = np.identity(3).reshape(-1)
                masked_robot_dict[k] = np.where(mask, target, masked_robot_dict[k])
            elif k == "link/inertia":
                raise Exception()
            elif k == "link/diaginertia":
                target = np.zeros_like(v)
                target[..., 0] = 0.01
                target[..., 1] = 0.02
                target[..., 2] = 0.03
                masked_robot_dict[k] = np.where(mask, target, masked_robot_dict[k])
            elif k == "link/friction":
                masked_robot_dict[k] = np.where(
                    mask, np.zeros_like(v), masked_robot_dict[k]
                )
            elif k == "link/contact_dim":
                masked_robot_dict[k] = np.where(
                    mask, np.ones_like(v) * 3, masked_robot_dict[k]
                )
            elif k == "link/rgba":
                target = np.random.uniform(0, 1, size=v.shape)
                target[..., 3] = 1.0
                masked_robot_dict[k] = np.where(
                    mask,
                    target,
                    masked_robot_dict[k],
                )
        else:
            masked_robot_dict[k] = v[~mask]
    # need to add this many fixed joints
    num_new_free_link_obs = int(last_robot_dict["link/mask"].sum().item())
    new_fixed_joints = {}
    new_fixed_joints["fixed_joint/id"] = np.repeat(
        np.arange(num_new_free_link_obs) + max(masked_robot_dict["fixed_joint/id"]) + 1,
        2,
        0,
    )[:, None]
    new_fixed_joints["fixed_joint/pos"] = np.repeat(
        np.zeros((num_new_free_link_obs, 3)), 2, 0
    )
    new_fixed_joints["fixed_joint/rotmat"] = np.repeat(
        np.repeat(np.identity(3).reshape(-1)[None, :], num_new_free_link_obs, 0), 2, 0
    )
    new_fixed_joints["fixed_joint/link/id"] = np.stack(
        (
            np.ones_like(masked_robot_dict["link/id"][-num_new_free_link_obs:])
            * GROUND_LINK_ID,
            masked_robot_dict["link/id"][-num_new_free_link_obs:],
        ),
        axis=1,
    ).reshape(-1)
    for k in new_fixed_joints.keys():
        masked_robot_dict[k] = np.concatenate(
            (masked_robot_dict[k], new_fixed_joints[k]), axis=0
        )

    masked_robot_dict["link/geom_type"] = (
        masked_robot_dict["link/geom_type"].astype(int) % 4
    )
    masked_robot_dict["link/geom_size"] = np.maximum(
        masked_robot_dict["link/geom_size"], 0.0001
    )

    return masked_robot_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain several robots' diffusion denoising processes into one "
        "looping animation. Feed each robot's *_robot_dicts.pkl (produced by "
        "scripts/visualize_robotoken_diffusion.py) in the order they should appear; "
        "the output pickles plug into scripts/render_pickle.py via "
        "--state-times for Blender rendering."
    )
    parser.add_argument(
        "--robot_dicts",
        type=str,
        nargs="+",
        required=True,
        help="Ordered *_robot_dicts.pkl paths, one per robot in the loop.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="looping_diffusion",
        help="Prefix for the output <prefix>.pkl and <prefix>_state_times.pkl.",
    )
    args = parser.parse_args()

    GRAVCOMP = False
    NUM_INTERPOLATION_STEPS = 5
    DIFFUSION_DURATION = 2.0
    HOLD_TIME = 0.5
    paths = args.robot_dicts
    path_robot_dicts = {}
    for path in paths:
        with open(path, "rb") as f:
            robot_dicts = pickle.load(f)
        path_robot_dicts[str(path)] = robot_dicts

    states = []
    state_times = []
    # chain diffusion processes together
    for path1, path2 in zip(paths + [paths[0]], paths[1:] + paths[:2]):
        robot_dicts_1 = path_robot_dicts[str(path1)]
        last_robot_dict_1 = robot_dicts_1[-1]
        robot_dicts_2 = path_robot_dicts[str(path2)]
        first_robot_dict_2 = robot_dicts_2[0]
        processed_robot_dicts = [
            post_process_robot_dict(robot_dict, last_robot_dict_1, OVERRIDE_KEYS)
            for robot_dict in robot_dicts_1
        ]
        anchor_positions = build_anchor_positions(
            processed_robot_dicts[-1],
            gravcomp=GRAVCOMP,
        )
        curr_time = max(state_times) if len(state_times) > 0 else 0
        state_times.extend(
            (np.linspace(0, 1, len(robot_dicts_1)) ** 4) * DIFFUSION_DURATION
            + curr_time
        )
        processed_robot_dicts = processed_robot_dicts + [processed_robot_dicts[-1]] * 2
        state_times.append(max(state_times) + HOLD_TIME / 2)
        state_times.append(max(state_times) + HOLD_TIME / 2)
        unit_time = np.diff(
            (DIFFUSION_DURATION * np.linspace(0, 1, len(robot_dicts_1)) ** 4)
        )[0]
        for step in range(NUM_INTERPOLATION_STEPS + 1):
            noisy_interpolation_dict = {}
            for k in last_robot_dict_1.keys():
                v_1 = last_robot_dict_1[k].cpu()
                v_2 = first_robot_dict_2[k].cpu()
                alpha = step / NUM_INTERPOLATION_STEPS
                if (
                    k.endswith("type")
                    or k.endswith("/contact_dim")
                    or k.endswith("/mass")
                    or k.endswith("range")
                    or k.endswith("inertia")
                    or k.endswith("/id")
                    or k.endswith("/mask")
                ):
                    v = v_1
                else:
                    v = v_1 * (1 - alpha) + v_2 * alpha
                    if k.endswith("/pos") or k.endswith("/geom_size"):
                        norm = torch.linalg.norm(v_1 - v_2)
                        std = np.sin(alpha * torch.pi) * norm * 0.01
                        v = v + torch.randn(size=v.shape) * std

                noisy_interpolation_dict[k] = v
            processed_robot_dicts.append(
                post_process_robot_dict(
                    noisy_interpolation_dict, last_robot_dict_1, OVERRIDE_KEYS
                )
            )
            state_times.append(max(state_times) + unit_time)

        for robot_dict in processed_robot_dicts:
            # create simulation
            tokenized_robot = deserialize(robot_dict, include_states=True)
            (
                mjcf_model,
                _,
                _,
                _,
            ) = detokenize(
                tokenized_robot,
                gravcomp=GRAVCOMP,
            )
            mjcf_model = set_up_default_scene(
                mjcf_model, add_plane=True, add_vis_cam=True, plane_z_pos=0.0
            )
            getattr(mjcf_model.visual, "global").offheight = 1080
            getattr(mjcf_model.visual, "global").offwidth = 1920
            physics = mjcf.Physics.from_mjcf_model(mjcf_model)
            physics.reset(0)
            assert physics is not None
            states.append(
                parse_physics_state(physics, anchor_positions=anchor_positions)
            )

    with open(f"{args.output_prefix}.pkl", "wb") as f:
        pickle.dump(states, f)
    with open(f"{args.output_prefix}_state_times.pkl", "wb") as f:
        pickle.dump(state_times, f)
