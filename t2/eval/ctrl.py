import itertools
import os
import pickle
import tempfile
import time
import typing
from typing import Any, Callable

import numpy as np
import ray
import torch
from ray.experimental import tqdm_ray
from ray.util.queue import Queue

from t2.data.pad import pad
from t2.env.base_env import ObsDict
from t2.env.remote_classes import RemoteTrackEnvRunner
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv
from t2.eval.utils import summarize_rollout
from t2.io.schema import concat_zarr_stores

remote_tqdm = ray.remote(tqdm_ray.tqdm)


def env_obs_to_batch(
    obs_list: list[ObsDict],
    prev_action_list: list[np.ndarray],
    seq_len_cfg: dict[str, int],
    device: torch.device,
    batch_process_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    group_time_offsets: dict[str, int],
) -> dict[str, torch.Tensor]:
    data_dict = {}
    rollout_steps = seq_len_cfg["rollout_steps"]
    for obs in obs_list:
        # pad all observation terms to rollout_steps
        repeated_obs = {}
        for k, v in obs.items():
            if k.startswith("metadata"):
                continue
            group = k.split("/")[0]
            if (
                group.endswith("_obs") or group in {"ctrl", "target_pose", "metric"}
            ) and v.shape[0] == 1:
                repeated_obs[k] = v.repeat(rollout_steps, axis=0)
            else:
                repeated_obs[k] = v

        for k, v in pad(repeated_obs, seq_len_cfg).items():
            if k not in data_dict:
                data_dict[k] = []
            data_dict[k].append(v)
    batch = {
        k: (np.stack(v, axis=0) if "time/id" not in k else np.stack(v, axis=0))
        for k, v in data_dict.items()
    }

    batch = {
        k: torch.tensor(
            v,
            dtype=(torch.bool if np.issubdtype(v.dtype, np.bool_) else torch.float32),
            device=device,
        )
        for k, v in batch.items()
    }
    batch = batch_process_fn(batch)
    # NOTE: this assumes the ctrl task uses the `forward` timestep sampler
    rollout_steps_indices = torch.arange(rollout_steps, device=device)
    batch["dyna_joint_obs/time/id"] = (
        rollout_steps_indices[None, :, None]
        .repeat(len(obs_list), 1, seq_len_cfg["dyna_joint_obs"])
        .reshape(
            len(obs_list),
            seq_len_cfg["dyna_joint_obs"] * rollout_steps,
            1,
        )
    ) + group_time_offsets.get("dyna_joint_obs", 0)
    batch["free_link_obs/time/id"] = (
        rollout_steps_indices[None, :, None]
        .repeat(len(obs_list), 1, seq_len_cfg["free_link_obs"])
        .reshape(
            len(obs_list),
            seq_len_cfg["free_link_obs"] * rollout_steps,
            1,
        )
    ) + group_time_offsets.get("free_link_obs", 0)
    batch["track_link_obs/time/id"] = (
        rollout_steps_indices[None, :, None]
        .repeat(len(obs_list), 1, seq_len_cfg["track_link_obs"])
        .reshape(
            len(obs_list),
            seq_len_cfg["track_link_obs"] * rollout_steps,
            1,
        )
    ) + group_time_offsets.get("track_link_obs", 0)
    batch["actuator_obs/time/id"] = (
        rollout_steps_indices[None, :, None]
        .repeat(len(obs_list), 1, seq_len_cfg["actuator_obs"])
        .reshape(
            len(obs_list),
            seq_len_cfg["actuator_obs"] * rollout_steps,
            1,
        )
    ) + group_time_offsets.get("actuator_obs", 0)
    batch["ctrl/time/id"] = (
        rollout_steps_indices[None, :, None]
        .repeat(len(obs_list), 1, seq_len_cfg["actuator"])
        .reshape(
            len(obs_list),
            seq_len_cfg["actuator"] * rollout_steps,
            1,
        )
    ) + group_time_offsets.get("ctrl", 0)

    batch["ctrl/actuator/id"] = torch.arange(seq_len_cfg["actuator"], device=device)[
        None, :, None
    ].repeat(len(obs_list), rollout_steps, 1)
    batch["ctrl/target_qpos"] = torch.zeros(
        len(obs_list),
        rollout_steps,
        seq_len_cfg["ctrl"],
        1,
        device=device,
    )
    batch["ctrl/mask"] = torch.ones(
        len(obs_list),
        rollout_steps,
        seq_len_cfg["ctrl"],
        1,
        device=device,
        dtype=torch.bool,
    )

    # this won't be replaced by noise when dont_noise_timesteps==[-1]
    # for the control task
    for batch_idx, prev_action in enumerate(prev_action_list):
        batch["ctrl/target_qpos"][batch_idx, :, : len(prev_action), :] = (
            torch.from_numpy(prev_action.copy()).to(device)[None, :, None]
        )
        batch["ctrl/mask"][batch_idx, :, : len(prev_action), :] = False
    batch["ctrl/target_qpos"] = batch["ctrl/target_qpos"].reshape(
        len(obs_list), rollout_steps * seq_len_cfg["ctrl"], 1
    )
    batch["ctrl/mask"] = batch["ctrl/mask"].reshape(
        len(obs_list), rollout_steps * seq_len_cfg["ctrl"], 1
    )

    return batch


def policy_server(
    decoder: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    device: torch.device,
    seq_len_cfg: dict[str, int],
    obs_queue: Queue,
    runner_pool: ray.util.ActorPool,
    batch_process_fn: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    group_time_offsets: dict[str, int],
    timeout: float = 0.5,  # one env step shouldn't take longer than this
    max_empty_queue_times: int = 20,
    batch_size: int = 16,
    pbar: tqdm_ray.tqdm | None = None,
    use_inference_mode: bool = True,
):
    empty_queue_times = 0
    num_consecutive_queue_empties = 0
    while runner_pool.has_next():
        try:
            queued_obs = obs_queue.get_nowait_batch(
                num_items=min(batch_size, obs_queue.size())
            )
            if len(queued_obs) == 0:
                empty_queue_times += 1
                if empty_queue_times < max_empty_queue_times:
                    time.sleep(timeout)
                    pbar.update.remote(0)
                    continue
                else:
                    # queue empty twice in a row, done running
                    break
            empty_queue_times = 0
            num_consecutive_queue_empties = 0

            obs_list = [q[0] for q in queued_obs]
            prev_action_list = [q[1] for q in queued_obs]
            hardware_dicts = [q[2] for q in queued_obs]
            action_queues = [q[3] for q in queued_obs]

            batch = env_obs_to_batch(
                obs_list=[
                    {**obs, **hardware_dict}
                    for obs, hardware_dict in zip(obs_list, hardware_dicts)
                ],
                prev_action_list=prev_action_list,
                seq_len_cfg=seq_len_cfg,
                device=device,
                group_time_offsets=group_time_offsets,
                batch_process_fn=batch_process_fn,
            )

            with torch.inference_mode(use_inference_mode):
                decode_dict = decoder(batch)
            actions = (
                decode_dict["ctrl/target_qpos"]
                .cpu()
                .numpy()
                .reshape(
                    len(obs_list),
                    seq_len_cfg["rollout_steps"],
                    seq_len_cfg["actuator"],
                )
            )
            action_masks = (
                batch["ctrl/mask"]
                .cpu()
                .numpy()
                .reshape(
                    len(obs_list),
                    seq_len_cfg["rollout_steps"],
                    seq_len_cfg["actuator"],
                )
            )

            action_time_offset = group_time_offsets.get("ctrl", 0)
            assert action_time_offset <= 0, "action time offset should be non-positive"
            # for instance, if action_time_offset == -1, then the agent observes one
            # action in the past to make its current decision
            num_past_actions = -action_time_offset
            # thus, we will only return actions starting from the current time step
            # onwards, filtering the first num_past_actions

            for action, action_queue, action_mask in zip(
                actions[:, num_past_actions:],
                action_queues,
                action_masks[:, num_past_actions:],
            ):
                n_steps = action.shape[0]
                masked_action = action[~action_mask].reshape(n_steps, -1)
                action_queue.put(
                    masked_action,
                    block=False,
                )
            if pbar is not None:
                pbar.update.remote(len(obs_list))
        except ray.util.queue.Empty:
            # done running
            num_consecutive_queue_empties += 1
            time.sleep(timeout)
            if num_consecutive_queue_empties > max_empty_queue_times:
                break
        except Exception as e:
            print("[run_policy] error", e)
            import traceback

            print(traceback.format_exc())
            raise e
            break


def evaluate_policy(
    decoder: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    device: torch.device,
    num_processes: int,
    runner: EnvRunner,
    seq_len_cfg: dict[str, int],
    hardware_traj_pairs_generator: Callable[[], list[tuple[int, int]]],
    log_dir: str | None = None,
    episode_len: int | None = None,
    policy_server_kwargs: dict[str, Any] = {},
    num_policies_per_gpu: int = 1,
    use_pbar: bool = True,
    action_horizon: int = 1,
) -> dict[str, float]:
    original_episode_len = runner.env.episode_len
    if episode_len is not None:
        runner.env.episode_len = episode_len
    assert isinstance(runner.env, TrackEnv)
    track_env = typing.cast(TrackEnv, runner.env)
    track_env.obs_time_indices = np.arange(
        seq_len_cfg["rollout_steps"]
    ) + policy_server_kwargs["group_time_offsets"].get("target_pose", 0)
    track_env.include_obs_time_indices = "relative"
    runner.log_dir = log_dir
    runner_pool = ray.util.ActorPool(
        [
            RemoteTrackEnvRunner.remote(runner=runner, action_horizon=action_horizon)
            for _ in range(num_processes)
        ]
    )
    hardware_traj_pairs = hardware_traj_pairs_generator()
    obs_queue = Queue()
    result_iter = runner_pool.map_unordered(
        lambda runner_actor, hardware_traj_pair: runner_actor.run_episode.remote(
            hardware_seed=hardware_traj_pair[0],
            traj_idx=hardware_traj_pair[1],
            obs_queue=obs_queue,
        ),
        hardware_traj_pairs,
    )
    assert isinstance(runner.env, TrackEnv)
    env = typing.cast(TrackEnv, runner.env)
    trajs = pickle.load(open(env.pickle_path, "rb"))
    traj_lens = [len(traj) for traj in trajs]
    total_steps = 0
    for _, traj_idx in hardware_traj_pairs:
        # add 1 because we end one state past the end of the trajectory
        # so that our dataset support the typical
        # "state" "action" "reward" "done" "info" "next state" structure
        total_steps += min(traj_lens[traj_idx], env.episode_len) + 1
    pbar = (
        remote_tqdm.remote(
            total=int(total_steps),
            desc="running policy",
        )
        if use_pbar
        else None
    )
    # get all gpu ids
    num_gpus = ray.cluster_resources().get("GPU", 0)
    if num_gpus < 1:
        raise RuntimeError(
            "Control evaluation requires at least one GPU visible to Ray: "
            "policy servers are scheduled with a fractional GPU each."
        )
    num_policy_servers = int(num_gpus * num_policies_per_gpu)
    policy_server_fn = ray.remote(policy_server).options(
        num_gpus=1 / num_policies_per_gpu - 0.01
    )

    policy_servers = [
        policy_server_fn.remote(
            decoder=decoder,
            device=device,
            seq_len_cfg=seq_len_cfg,
            obs_queue=obs_queue,
            runner_pool=runner_pool,
            pbar=pbar,
            **policy_server_kwargs,
        )
        for _ in range(num_policy_servers)
    ]
    ray.wait(policy_servers, timeout=0.1)
    data_paths = list(result_iter)
    ray.wait(policy_servers)
    if pbar is not None:
        ray.get(pbar.close.remote())
    if log_dir is not None:
        data_path = os.path.join(log_dir, "ctrl_eval_summary.zarr")
        concat_zarr_stores(
            from_paths=data_paths,
            to_path=data_path,
            use_pbar=False,
        )
        summary_stats = summarize_rollout(data_path)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = os.path.join(tmp_dir, "data.zarr")
            concat_zarr_stores(
                from_paths=data_paths,
                to_path=data_path,
                use_pbar=False,
            )
            summary_stats = summarize_rollout(data_path)

    runner.env.episode_len = original_episode_len

    return {k: torch.tensor(v, device=device) for k, v in summary_stats.items()}


def cross_product(
    list1: list[int],
    list2: list[int],
):
    return list(itertools.product(list1, list2))


def identity(*args):
    return args
