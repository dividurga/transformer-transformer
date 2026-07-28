import logging
import os
import pickle

import hydra
import torch
from omegaconf import OmegaConf

import wandb
from t2.model.t2 import setup_decoder
from t2.train.augment import (
    AddPositionId,
)
from t2.train.utils import set_torch_configs
from t2.utils.misc import (
    ensure_wandb_run_metadata,
    flatten_dict,
    seed_everything,
)


@hydra.main(
    config_path="../config",
    config_name="evaluate_hardware_opt",
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
        # Determine num_composed from trajectory groups
        # Convert to container to handle OmegaConf ListConfig
        traj_indices_raw = OmegaConf.to_container(
            cfg.eval_fn.traj_indices, resolve=True
        )
        assert isinstance(traj_indices_raw, list), "traj_indices must be a list"
        if len(traj_indices_raw) > 0 and isinstance(traj_indices_raw[0], (list, tuple)):
            # Validate that all trajectory groups have the same size
            # This is required because ComposedMultiModalDiffusionTask uses a fixed
            # num_composed for tensor reshaping in _compose_pred_noise
            group_sizes = [len(group) for group in traj_indices_raw]
            if len(set(group_sizes)) > 1:
                raise ValueError(
                    f"All trajectory groups must have the same size when composition "
                    f"is enabled. Found groups with sizes: {group_sizes}. "
                    f"The ComposedMultiModalDiffusionTask requires consistent group "
                    f"sizes for tensor reshaping operations."
                )
            num_composed = group_sizes[0]
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
    timestep_sampler = hydra.utils.instantiate(cfg.timestep_sampler)
    seq_len_cfg = dict(policy_cfg.seq_len)
    center_traj = cfg.eval_fn.runner.env.center_traj

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

    eval_fn = hydra.utils.call(cfg.eval_fn)
    summary_stats = eval_fn(
        decoder=hardware_generator,
        hardware_optimizer_fn=lambda decoder: hydra.utils.instantiate(
            cfg.eval_fn.hardware_optimizer_fn
        )(
            hardware_generator=decoder,
            seq_len_cfg=seq_len_cfg,
            timestep_sampler=timestep_sampler,
            center_traj=center_traj,
            add_pos_id=add_pos_id,
            device=device,
            output_cache_dir=output_cache_dir,
            max_traj_len=max_traj_len,
        ),
        log_dir=wandb.run.dir,
        run_model_upfront=run_model_upfront,
    )

    wandb.log(data=summary_stats)

    for k, v in summary_stats.items():
        if k.startswith("metric/actuator"):
            continue
        if (
            any(k.endswith(suffix) for suffix in ["/q95", "/q50", "/mean"])
            or k
            in {
                "metric/reward/sum",
                "predicted_value",
                "optimize_time",
            }
            or k.startswith("actual_value")
        ):
            logging.info(f"{k}: {v:.2f}")
        elif any(k.endswith(suffix) for suffix in ["/any"]):
            logging.info(f"{k}: {v * 100:.1f}%")


if __name__ == "__main__":
    main()
