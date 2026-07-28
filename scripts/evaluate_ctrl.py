import logging
import os
import pickle
from typing import Callable

import hydra
import ray
import torch
from omegaconf import OmegaConf

import wandb
from t2.data.dataset import TimestepSampler
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId
from t2.utils.misc import flatten_dict, seed_everything


@hydra.main(
    config_path="../config",
    config_name="evaluate_ctrl",
    version_base="1.3",
)
def main(cfg):
    seed = 0
    seed_everything(seed)
    policy_cfg_path = os.path.dirname(cfg.ckpt_path) + "/cfg.pkl"
    policy_cfg = pickle.load(open(policy_cfg_path, "rb"))
    OmegaConf.resolve(policy_cfg.ctrl_seq_len)
    cfg.eval_fn.seq_len_cfg = policy_cfg.ctrl_seq_len
    flattened_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True), sep="/")  # type: ignore
    wandb.init(
        project="transformer-transformer",
        config=flattened_cfg,
        tags=["ctrl-eval"],
    )
    assert wandb.run is not None
    ray.init(num_cpus=cfg.eval_fn.num_processes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: env_obs_to_batch currently assumes the ctrl task uses the
    # `forward` timestep sampler
    assert (
        type(
            hydra.utils.instantiate(
                policy_cfg.datasets.ctrl_rand.dataset.timestep_sampler
            )
        )
        is TimestepSampler
    )

    cfg.eval_fn.policy_server_kwargs.group_time_offsets = (
        policy_cfg.datasets.ctrl_rand.dataset.group_time_offsets
    )
    logging.info(
        "time_offset: " + str(cfg.eval_fn.policy_server_kwargs.group_time_offsets)
    )

    batch_process_fn = hydra.utils.instantiate(
        policy_cfg.datasets.ctrl_rand.batch_process_fn
    )
    batch_process_fn.augmentations = [
        aug for aug in batch_process_fn.augmentations if type(aug) is AddPositionId
    ]

    policy = setup_decoder(
        policy_cfg,
        ckpt_path=cfg.ckpt_path,
        device=device,
        task_name=cfg.task_name,
        num_inference_steps=cfg.num_inference_steps,
        num_repeats_per_step=cfg.num_repeats_per_step,
        clip_samples_in_guidance=cfg.clip_samples_in_guidance,
        deterministic=cfg.deterministic,
        use_flash_attn=cfg.use_flash_attn,
        use_torch_compile=cfg.use_torch_compile,
        use_mixed_precision=cfg.use_mixed_precision,
    )
    guidance_fn: Callable | None = None
    if "diffusion_guidance" in cfg and cfg.diffusion_guidance is not None:
        guidance_cfg = cfg.diffusion_guidance
        guidance_fn = hydra.utils.instantiate(guidance_cfg)
        logging.info(
            "Using diffusion guidance: %s",
            guidance_cfg.get("_target_", "custom guidance"),
        )

    def policy_with_optional_guidance(
        batch: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        kwargs["guidance_fn"] = guidance_fn
        return policy(
            batch,
            **kwargs,
        )

    eval_fn = hydra.utils.call(cfg.eval_fn)
    stats = eval_fn(
        decoder=policy_with_optional_guidance,
        device=device,
        log_dir=wandb.run.dir,
        policy_server_kwargs={
            **cfg.eval_fn.policy_server_kwargs,
            "batch_process_fn": batch_process_fn,
        },
    )
    wandb.log({k: v.item() for k, v in stats.items()})
    for k, v in stats.items():
        if k.startswith("metric/actuator"):
            continue
        if (
            any(k.endswith(suffix) for suffix in ["/q95", "/q50", "/mean"])
            or k == "metric/reward/sum"
        ):
            print(f"{k}: {v:.3f}")
        elif any(k.endswith(suffix) for suffix in ["/any"]):
            print(f"{k}: {v * 100:.1f}%")


if __name__ == "__main__":
    main()
