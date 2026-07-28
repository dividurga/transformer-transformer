import functools
import hashlib
import json
import logging
import os
import pickle
import shutil
from typing import Any, Callable

import hydra
import numpy as np
import ray
import torch
from dm_control import mjcf
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

import wandb
from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.rl_runner import RLEnvRunner
from t2.env.track_env import TrackEnv
from t2.eval.hardware_decoding import hardware_to_nearest_neighbor_params
from t2.eval.hardware_optimization import label_episode_with_actual_value
from t2.eval.utils import (
    extract_masked_hardware,
    log_summary_stats,
    run_model_upfront_phase,
    summarize_rollout,
)
from t2.io.schema import concat_zarr_stores
from t2.model.t2 import setup_decoder
from t2.robotok.io import deserialize
from t2.robotok.tokenizer import DetokenizeError, preprocess_mjcf, tokenize
from t2.train.augment import AddPositionId
from t2.train.utils import set_torch_configs
from t2.utils.misc import (
    ensure_wandb_run_metadata,
    flatten_dict,
    seed_everything,
)


@ray.remote(num_cpus=1)
def evaluate_hardware(
    runner_cfg: DictConfig,
    robogen_choices: list[int],
    robogen_seed: int,
    rl_ckpt_path: str,
    log_dir: str,
    data_path: str,
    traj_idx: int,
    predicted_value: float,
    optimize_time: float,
    post_process_episode_data_fn: Callable[
        [BaseEnv, dict[str, Any], HardwareDict], tuple[dict[str, Any], HardwareDict]
    ],
    reset_qpos: NDArray[np.float32] | None = None,
):
    runner_cfg.env.robot_generator.choices = robogen_choices
    runner_cfg.env.robot_generator.num_uniforms = (
        13  # HARDCODED, but should look up from data
    )
    runner_cfg.ckpt_path = rl_ckpt_path

    def reset_fn(
        _env: BaseEnv,
        _seed: int,
        _traj_idx: int,
        _reset_qpos: NDArray[np.float32] | None = reset_qpos,
    ) -> tuple[ObsDict, InfoDict, HardwareDict]:
        # this function is identical to `TrackEnv.reset_sim_and_target_traj`
        # but optionally sets the init qpos to the predicted qpos, if provided
        assert isinstance(_env, TrackEnv)
        _env.traj_idx = _traj_idx
        _env.reset_no_obs(seed=_seed)
        if _reset_qpos is not None:
            _env.d.qpos[:] = _reset_qpos

        _env.p.forward()  # might be redundant here
        obs = _env.get_obs()
        info = _env.get_info()
        return obs, info, _env.hardware_dict

    def hardware_reset_fn(
        _env: BaseEnv,
        _seed: int,
    ) -> None:
        _env.reset_hardware(hardware_seed=_seed)
        _env.hardware_dict["metadata/predicted_value"] = np.array([predicted_value])
        _env.hardware_dict["metadata/optimize_time"] = np.array([optimize_time])

    reset_to_traj_fn = functools.partial(reset_fn, _traj_idx=traj_idx)
    runner: RLEnvRunner = hydra.utils.instantiate(runner_cfg, log_dir=log_dir)
    runner.run_episodes(
        hardware_seed=robogen_seed,
        episode_seeds=[traj_idx],
        policy_fn=None,
        reset_fn=reset_to_traj_fn,
        post_process_episode_data_fn=post_process_episode_data_fn,
        hardware_reset_fn=hardware_reset_fn,
        data_path=data_path,
    )
    del runner


@hydra.main(
    config_path="../config",
    config_name="evaluate_hardware_opt_rl",
    version_base="1.3",
)
def main(cfg):
    set_torch_configs()
    seed = 0
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_cfg_path = os.path.dirname(cfg.ckpt_path) + "/cfg.pkl"
    policy_cfg = pickle.load(open(policy_cfg_path, "rb"))
    OmegaConf.resolve(policy_cfg.seq_len)
    cfg.timestep_sampler.num_rollout_steps = int(policy_cfg.seq_len["rollout_steps"])
    cfg.eval_fn.num_seeds_per_datapoint = cfg.num_seeds_per_datapoint
    flattened_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True), sep="/")  # type: ignore
    wandb.init(
        project="transformer-transformer",
        config=flattened_cfg,
        tags=cfg.tags,
    )
    assert wandb.run is not None
    ensure_wandb_run_metadata(wandb.run)

    OmegaConf.resolve(policy_cfg)

    # Parse composition config for multi-trajectory optimization
    composition_enabled = getattr(cfg, "composition", {}).get("enabled", False)
    compose_sample_names = None
    num_composed = 1
    compose_method = "avg"

    if composition_enabled:
        compose_sample_names = list(cfg.composition.sample_names)
        compose_method = cfg.composition.method
        # Determine num_composed from the first trajectory group
        # Convert to container to handle OmegaConf ListConfig
        traj_indices = OmegaConf.to_container(cfg.eval_fn.traj_indices, resolve=True)
        assert isinstance(traj_indices, list)
        if len(traj_indices) > 0 and isinstance(traj_indices[0], (list, tuple)):
            num_composed = len(traj_indices[0])
        logging.info(
            f"Composition enabled: sample_names={compose_sample_names}, "
            f"num_composed={num_composed}, method={compose_method}"
        )

    hardware_generator = setup_decoder(
        policy_cfg,
        ckpt_path=cfg.ckpt_path,
        device=device,
        task_name=cfg.task_name,
        num_inference_steps=cfg.num_inference_steps,
        num_repeats_per_step=cfg.num_repeats_per_step,
        clip_samples_in_guidance=cfg.clip_samples_in_guidance,
        use_flash_attn=cfg.use_flash_attn,
        eta=cfg.eta,
        use_mixed_precision=cfg.use_mixed_precision,
        use_torch_compile=cfg.use_torch_compile,
        # Composition parameters
        compose_sample_names=compose_sample_names,
        num_composed=num_composed,
        compose_method=compose_method,
    )

    batch_process_fn = hydra.utils.instantiate(
        policy_cfg.datasets.clean.batch_process_fn,
    )
    add_pos_id = next(
        aug for aug in batch_process_fn.augmentations if type(aug) is AddPositionId
    )

    choice_to_ckpt: dict[str, str] = json.load(
        open(cfg.get("choice_to_ckpt_path", "choice_to_ckpt_path.json"), "r")
    )

    trajs = pickle.load(open(cfg.pickle_path, "rb"))
    timestep_sampler = hydra.utils.instantiate(cfg.timestep_sampler)
    seq_len_cfg = dict(policy_cfg.seq_len)

    # Default output_cache_dir to wandb run dir if not set
    output_cache_dir = getattr(
        cfg.eval_fn.hardware_optimizer_fn, "output_cache_dir", None
    )
    if output_cache_dir is None:
        output_cache_dir = os.path.join(wandb.run.dir, "hardware_cache")

    # Get run_model_upfront from config (defaults to False if not set)
    run_model_upfront = getattr(cfg, "run_model_upfront", False)
    if run_model_upfront:
        logging.info(
            "run_model_upfront=True: Will pre-compute all model outputs, "
            "then unload model to free GPU memory."
        )

    max_traj_len = policy_cfg.model.pos_embs.time.embedding.num_embeddings - 1

    optimizer = hydra.utils.instantiate(cfg.eval_fn.hardware_optimizer_fn)(
        hardware_generator=hardware_generator,
        seq_len_cfg=seq_len_cfg,
        timestep_sampler=timestep_sampler,
        center_traj=cfg.eval_fn.runner.env.center_traj,
        add_pos_id=add_pos_id,
        device=device,
        output_cache_dir=output_cache_dir,
        max_traj_len=max_traj_len,
    )

    post_process_episode_data_fn = functools.partial(
        label_episode_with_actual_value,
        reward_fns=optimizer.reward_fns,
    )

    # Phase 1: Pre-compute all model outputs if run_model_upfront is enabled
    if run_model_upfront:
        run_model_upfront_phase(
            optimizer=optimizer,
            trajs=trajs,
            traj_indices=list(cfg.eval_fn.traj_indices),
            num_seeds_per_datapoint=cfg.num_seeds_per_datapoint,
            optimize_hardware=cfg.eval_fn.optimize_hardware,
        )

    def hardware_reset_fn(_seed: int, _traj_idx: int):
        # Warmup is handled internally by the optimizer
        robot_dict, predicted_value, optimize_time = optimizer.optimize(
            trajs=[trajs[_traj_idx]], seed=_seed
        )
        logging.info(
            f"value: {predicted_value:.02f} for traj {_traj_idx}, took {optimize_time:.1f}s"
        )
        robot_dict = {k: v.cpu().numpy() for k, v in robot_dict.items()}

        (
            masked_hardware_dict,
            num_links,
            num_actuators,
            num_dyna_joints,
            num_fixed_joints,
        ) = extract_masked_hardware(robot_dict)
        choices, seed = hardware_to_nearest_neighbor_params(
            masked_hardware_dict,
            num_links,
            num_actuators,
            num_dyna_joints,
            num_fixed_joints,
            cfg.robotoken_root_path,
        )
        ckpt_path = choice_to_ckpt["".join(map(str, choices))]
        tokenized_robot = deserialize(masked_hardware_dict, include_states=True)
        reset_qpos = None
        if cfg.use_predicted_reset_qpos:
            mjcf_model = hydra.utils.instantiate(
                cfg.eval_fn.runner.env.robot_generator, choices=choices, num_uniforms=13
            )(seed=seed)
            mjcf_model = preprocess_mjcf(mjcf_model)
            _, mjjntid_to_dyna_jntid, _, _ = tokenize(mjcf_model)
            physics = mjcf.Physics.from_mjcf_model(mjcf_model)
            reset_qpos = np.zeros(physics.model.nq)
            for free_link_obs in filter(
                lambda x: x.time_idx == 0, tokenized_robot.free_link_obs
            ):
                # assume there is only one free link and it is the first few qpos
                reset_qpos[:3] = free_link_obs.pos
                reset_qpos[3:7] = free_link_obs.quat_wxyz
            for dyna_jnt_obs in filter(
                lambda x: x.time_idx == 0, tokenized_robot.dyna_jnt_obs
            ):
                for mj_jnt_idx, dyna_jnt_idx in mjjntid_to_dyna_jntid.items():
                    if dyna_jnt_idx == dyna_jnt_obs.dyna_jnt_idx:
                        mj_jnt = physics.model.joint(mj_jnt_idx)
                        reset_qpos[mj_jnt.qposadr] = dyna_jnt_obs.qpos
        return choices, seed, ckpt_path, reset_qpos, optimize_time, predicted_value

    def pick_random_hardware(
        _seed: int,
    ) -> tuple[list[int], int, str, NDArray[np.float32] | None, float, float]:
        rs = np.random.RandomState(_seed)
        choices = rs.choice(list(choice_to_ckpt.keys()))
        seed = rs.randint(0, np.iinfo(np.int32).max)
        ckpt_path = choice_to_ckpt[choices]
        choices = list(map(int, choices))
        return choices, seed, ckpt_path, None, 0.0, 0.0

    data_paths = []

    ray_tasks = []

    validity = []

    for traj_idx in cfg.eval_fn.traj_indices:
        hardware_reset_for_traj_fn = functools.partial(
            hardware_reset_fn, _traj_idx=traj_idx
        )
        for hardware_iter in range(cfg.num_seeds_per_datapoint):
            hashobj = hashlib.sha256(f"{hardware_iter}{traj_idx}".encode())
            rs = np.random.RandomState(
                int(hashobj.hexdigest(), 16) % (np.iinfo(np.int32).max)
            )
            hardware_seed = rs.randint(0, np.iinfo(np.int32).max)
            if cfg.eval_fn.optimize_hardware:
                try:
                    (
                        robogen_choices,
                        robogen_seed,
                        rl_ckpt_path,
                        reset_qpos,
                        optimize_time,
                        predicted_value,
                    ) = hardware_reset_for_traj_fn(_seed=hardware_seed)
                except DetokenizeError:
                    logging.warning(
                        f"attempt: {hardware_iter}, traj: {traj_idx}, DetokenizeError"
                    )
                    validity.append(False)
                    continue
                except np.linalg.LinAlgError:
                    # IK didn't converge, most likely because joint transforms was bad
                    logging.warning(
                        f"attempt: {hardware_iter}, traj: {traj_idx}, LinAlgError"
                    )
                    validity.append(False)
                    continue
                except Exception as e:
                    logging.error(
                        f"attempt: {hardware_iter}, traj: {traj_idx}, error: {e}"
                    )
                    validity.append(False)
                    continue
            else:
                (
                    robogen_choices,
                    robogen_seed,
                    rl_ckpt_path,
                    reset_qpos,
                    optimize_time,
                    predicted_value,
                ) = pick_random_hardware(_seed=hardware_seed)
            log_dir = f"{wandb.run.dir}/traj{traj_idx:02d}_seed{hardware_seed:06d}"
            data_path = f"{log_dir}/traj{traj_idx:02d}_seed{hardware_seed:06d}.zarr"
            data_paths.append(data_path)
            ray_tasks.append(
                evaluate_hardware.remote(
                    runner_cfg=cfg.eval_fn.runner,
                    robogen_choices=robogen_choices,
                    robogen_seed=robogen_seed,
                    rl_ckpt_path=rl_ckpt_path,
                    log_dir=log_dir,
                    data_path=data_path,
                    traj_idx=traj_idx,
                    reset_qpos=reset_qpos,
                    predicted_value=predicted_value,
                    optimize_time=optimize_time,
                    post_process_episode_data_fn=post_process_episode_data_fn,
                )
            )
            validity.append(True)
            ray.wait(ray_tasks, timeout=0.0)  # push tasks to start running
            if len(ray_tasks) > cfg.max_concurrent_tasks:
                import gc

                logging.info("getting tasks")
                ray.get(ray_tasks)
                ray_tasks = []
                ray.shutdown()
                gc.collect()
    ray.get(ray_tasks)

    avg_validity = float(np.mean(validity))
    if avg_validity == 0:
        wandb.log(data={"validity": avg_validity})
        logging.error("No valid hardware found")
        return

    concat_zarr_stores(
        from_paths=data_paths,
        to_path=f"{wandb.run.dir}/hardware_opt.zarr",
        root_metadata=OmegaConf.to_container(cfg),
    )
    for data_path in data_paths:
        shutil.rmtree(data_path)
        lock_path = data_path + ".lock"
        if os.path.exists(lock_path):
            os.remove(lock_path)

    summary_stats = summarize_rollout(f"{wandb.run.dir}/hardware_opt.zarr")
    summary_stats["validity"] = float(np.mean(validity))
    wandb.log(data=summary_stats)

    log_summary_stats(summary_stats)


if __name__ == "__main__":
    main()
