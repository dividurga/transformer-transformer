import gc
import logging
import os
import pickle
from datetime import timedelta
from typing import Callable, Optional
import ray
import hydra
import torch
import torch.multiprocessing
import torch.nn as nn
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin
from diffusers.training_utils import EMAModel
from omegaconf import OmegaConf

import wandb
from t2.data.normalization_utils import (
    compute_dataset_normalization,
    load_normalization_dict,
    save_normalization_dict,
)
from t2.model.t2 import T2
from t2.train.augment import ComposeAugmentation, TransformAugmentation
from t2.train.utils import set_torch_configs
from t2.utils.misc import flatten_dict, seed_everything

torch.multiprocessing.set_sharing_strategy("file_system")


def infinite_iterator(dataloader: torch.utils.data.DataLoader):
    """Yield batches from the dataloader indefinitely without caching."""
    while True:
        for batch in dataloader:
            yield batch


def train_epoch(
    accelerator: Accelerator,
    model: nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    dataset_task_names: dict[str, list[str]],
    dataset_batch_process_fn: dict[
        str,
        Optional[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]],
    ],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    max_grad_norm: float,
    curr_train_step: int,
    num_batches: int = 1,
    ema: Optional[EMAModel] = None,
    log_stats_every_n_steps: int = 1,
    is_eval: bool = False,
    gc_collect_every_n_steps: int = 1000,
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, float] = {}
    total_num_steps = (
        int(num_batches / accelerator.gradient_accumulation_steps)
        if not is_eval
        else num_batches
    )
    pbar = tqdm.tqdm(
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
        total=total_num_steps,
    )
    for original_batches in zip(*dataloaders.values()):
        # Load original batches once per iteration
        original_batches = [
            (k, dataset_task_names[k], batch)
            for k, batch in zip(dataloaders.keys(), original_batches)
        ]

        if curr_train_step % gc_collect_every_n_steps == 0:
            gc.collect()

        with accelerator.accumulate(model):
            # Run the same batch with different augmentations
            batches = []
            for k, task_names, batch in original_batches:
                # Apply augmentation (different each time due to randomness)
                batch_process_fn = dataset_batch_process_fn[k]
                if batch_process_fn is not None:
                    batch = batch_process_fn(batch)
                batches.append((task_names, batch))

            outputs = model(batches=batches, is_eval=is_eval)

            # Accumulate stats from all augmentation repeats
            with torch.no_grad():
                for key, value in outputs.items():
                    if key not in sums:
                        sums[key] = torch.zeros_like(value)
                        counts[key] = 0
                    sums[key] += value
                    counts[key] += 1

            if not is_eval:
                # Backward pass for each augmentation repeat
                accelerator.backward(outputs["loss"])

                # Single optimizer step after all augmentation repeats
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if ema is not None and accelerator.sync_gradients:
                    ema.step(model.parameters())
                if (
                    curr_train_step % log_stats_every_n_steps == 0
                    and accelerator.is_main_process
                ):
                    # NOTE not gathering logs from other processes
                    # Log stats from the last augmentation repeat
                    accelerator.log(
                        {
                            "lr": scheduler.get_last_lr()[0],
                            **{k: v.item() for k, v in outputs.items()},
                        },
                        step=curr_train_step,
                    )
                if accelerator.sync_gradients:
                    curr_train_step += 1
                    pbar.update(1)
            else:
                pbar.update(1)
                curr_train_step += 1
    with torch.no_grad():
        return {k: sums[k] / counts[k] for k in sums}


EVAL_FN_USES_RAY = {"ctrl"}


@hydra.main(
    config_path="../config",
    config_name=os.path.splitext(os.path.basename(__file__))[0],
    version_base="1.3",
)
def main(cfg):
    # Set random seed
    seed_everything(cfg.seed, workers=True)
    set_torch_configs()
    flattened_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True), sep="/")  # type: ignore

    # Initialize W&B
    plugin = GradientAccumulationPlugin(
        sync_with_dataloader=False,
        num_steps=cfg.accelerator.gradient_accumulation_steps,
        sync_each_batch=False,
    )
    accelerator = Accelerator(
        mixed_precision=cfg.accelerator.mixed_precision,
        gradient_accumulation_plugin=plugin,
        log_with=cfg.accelerator.log_with,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=1000000000))],
    )
    flattened_cfg["effective_batch_size"] = int(
        accelerator.num_processes
        * cfg.train.batch_size
        * accelerator.gradient_accumulation_steps
    )
    logger = get_logger(__name__)
    logger.info("effective batch size: %d", flattened_cfg["effective_batch_size"])
    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    wandb_cfg["config"] = flattened_cfg
    accelerator.init_trackers(
        project_name="transformer-transformer",
        config=flattened_cfg,
        init_kwargs={"wandb": wandb_cfg},
    )
    if accelerator.is_main_process:
        assert wandb.run is not None
        pickle.dump(cfg, open(f"{wandb.run.dir}/cfg.pkl", "wb"))

    device = accelerator.device
    with accelerator.main_process_first():
        datasets = hydra.utils.instantiate(cfg.datasets)
        dataset_task_names = {k: v["tasks"] for k, v in datasets.items()}
        train_dataloaders = {
            k: hydra.utils.instantiate(
                cfg.dataloader,
                batch_size=v.get("batch_size", cfg.train.batch_size),
                dataset=v["dataset"],
                sampler=torch.utils.data.RandomSampler(
                    data_source=v["dataset"],
                    replacement=True,
                    num_samples=v.get("batch_size", cfg.train.batch_size)
                    * cfg.train.num_batches_per_epoch
                    * accelerator.num_processes,
                ),
            )
            for k, v in datasets.items()
        }
        val_dataloaders = {
            k: hydra.utils.instantiate(
                cfg.dataloader,
                batch_size=v.get("batch_size", cfg.train.val_batch_size),
                dataset=v["dataset"],
                sampler=torch.utils.data.RandomSampler(
                    data_source=v["dataset"],
                    replacement=False,
                    num_samples=cfg.train.num_eval_batches_per_epoch
                    * v.get("batch_size", cfg.train.val_batch_size)
                    * accelerator.num_processes,
                ),
            )
            for k, v in datasets.items()
        }

    # Initialize model
    model: T2 = hydra.utils.instantiate(
        cfg.model,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("number of parameters: %.2fM" % (n_params / 1e6))

    with accelerator.main_process_first():
        # compute normalization
        modalities = {}
        for task in model.tasks.values():
            for adapter in task.adapters.values():
                modalities[adapter.modality.name] = (
                    adapter.modality,
                    adapter.attr_encoders,
                )
        normalization_dicts = {}
        for k, v in datasets.items():
            dataset = v["dataset"]
            cache_norm_path = dataset.path.replace(".zarr", ".norm.pt")
            if not os.path.exists(cache_norm_path):
                normalization_dict = compute_dataset_normalization(
                    path=dataset.path,
                    hardware_groups=dataset.hardware_groups,
                    rollout_groups=dataset.rollout_groups,
                    modalities=modalities,
                )
                save_normalization_dict(normalization_dict, cache_norm_path)
                normalization_dicts[k] = normalization_dict

            assert os.path.exists(cache_norm_path)
            normalization_dicts[k] = load_normalization_dict(cache_norm_path)

        normalization_dict = next(iter(normalization_dicts.values()))
        # combine normalization dicts
        for norm_term in normalization_dict.keys():
            norm = normalization_dict[norm_term]
            for other_norms in normalization_dicts.values():
                norm.vmin.data[:] = torch.min(
                    norm.vmin.data,
                    other_norms[norm_term].vmin.data,
                )
                norm.vmax.data[:] = torch.max(
                    norm.vmax.data,
                    other_norms[norm_term].vmax.data,
                )
                if cfg.train.normalization_clip is not None:
                    norm.clip = cfg.train.normalization_clip
                else:
                    assert norm.clip == other_norms[norm_term].clip

        # assign normalization to model
        for task in model.tasks.values():
            for adapter in task.adapters.values():
                for norm_name, norm in normalization_dict.items():
                    norm_group = norm_name.split("/")[0]
                    if norm_group == adapter.modality.name:
                        attr_name = "/" + "/".join(norm_name.split("/")[1:])
                        adapter.modality.attr_norms[attr_name] = norm.to(device)

    # Initialize optimizer, scheduler, and loss function
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    num_training_steps = cfg.train.num_batches_per_epoch * cfg.train.num_epochs
    scheduler = hydra.utils.instantiate(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )

    epoch = 0
    curr_train_step = 0

    for k, v in train_dataloaders.items():
        train_dataloaders[k] = accelerator.prepare(v)
    for k, v in val_dataloaders.items():
        val_dataloaders[k] = accelerator.prepare(v)

    model, optimizer = accelerator.prepare(model, optimizer)
    ema = hydra.utils.instantiate(cfg.ema, parameters=model.parameters())

    if cfg.train.load_ckpt_path is not None:
        logger.info(f"Loading checkpoint from {cfg.train.load_ckpt_path}")
        ckpt = torch.load(cfg.train.load_ckpt_path, map_location=device)
        model.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        curr_train_step = ckpt["curr_train_step"]
        epoch = ckpt["epoch"] + 1
        ema.load_state_dict(ckpt["ema"])

    eval_fns = {
        eval_name: hydra.utils.call(
            eval_cfg,
        )
        for eval_name, eval_cfg in cfg.evals.items()
    }
    if accelerator.is_main_process:
        # Do a test run with the ctrl eval fn, which is typical to fail
        with torch.inference_mode():
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            unwrapped_model = accelerator.unwrap_model(
                model, keep_fp32_wrapper=False, keep_torch_compile=False
            )
            if "ctrl" in unwrapped_model.tasks and "ctrl" in eval_fns:
                stats = eval_fns["ctrl"](
                    decoder=lambda batch,
                    seed=None,
                    _model=unwrapped_model: _model.tasks["ctrl"].decode(
                        batch=batch,
                        backbone=_model.backbone,
                        pos_embs=_model.pos_embs,
                        seed=seed,
                    ),
                    device=device,
                    episode_len=1,
                )
            ray.shutdown()
            gc.collect()
            ema.restore(model.parameters())
            model.train()

    dataset_batch_process_fn = {k: v["batch_process_fn"] for k, v in datasets.items()}
    transform_augmentations = []
    for k, v in dataset_batch_process_fn.items():
        if isinstance(v, TransformAugmentation):
            transform_augmentations.append(v)
        elif isinstance(v, ComposeAugmentation):
            for aug in v.augmentations:
                if isinstance(aug, TransformAugmentation):
                    transform_augmentations.append(aug)

    if len(transform_augmentations) > 0:
        max_displacement = max(
            [
                transform_aug.pos_aug_magnitude.abs().max().item()
                for transform_aug in transform_augmentations
            ]
        )
        logging.info(
            "Augmenting normalization ranges for position fields by"
            + f" {max_displacement:.02f}m",
        )
        affected_pose_fields = transform_augmentations[0].affected_pose_fields
        for pose_field in affected_pose_fields:
            pos_field = pose_field + "/pos"
            assert pos_field in normalization_dict
            norm = normalization_dict[pos_field]
            # Use the maximum absolute range across all three axes
            abs_vmin = norm.vmin.abs()
            abs_vmax = norm.vmax.abs()
            max_range = torch.max(torch.max(abs_vmin, abs_vmax))
            # Add augmentation magnitude to the range
            max_range += max_displacement
            # Set symmetric range for all axes
            # NOTE: this may be an overly conservative range
            norm.vmin.data[:] = -max_range
            norm.vmax.data[:] = max_range

    # Training loop
    with accelerator.autocast():
        for epoch in range(epoch, cfg.train.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.train.num_epochs}")

            model.train()
            train_stats = train_epoch(
                accelerator=accelerator,
                model=model,
                dataloaders=train_dataloaders,
                dataset_task_names=dataset_task_names,
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=cfg.train.max_grad_norm,
                curr_train_step=curr_train_step,
                log_stats_every_n_steps=cfg.train.log_stats_every_n_steps,
                is_eval=False,
                dataset_batch_process_fn=dataset_batch_process_fn,
                ema=ema,
                num_batches=cfg.train.num_batches_per_epoch,
            )
            curr_train_step += int(
                cfg.train.num_batches_per_epoch
                / accelerator.gradient_accumulation_steps
            )
            stats = {
                "train/" + k: v.mean().item()
                for k, v in accelerator.gather(train_stats).items()
            }

            if epoch % cfg.train.eval_every_n_epochs == 0:
                model.eval()
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                with torch.inference_mode():
                    if cfg.train.run_val:
                        val_stats = train_epoch(
                            accelerator=accelerator,
                            model=model,
                            dataloaders=val_dataloaders,
                            dataset_task_names=dataset_task_names,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            max_grad_norm=cfg.train.max_grad_norm,
                            curr_train_step=curr_train_step,
                            log_stats_every_n_steps=cfg.train.log_stats_every_n_steps,
                            is_eval=True,
                            dataset_batch_process_fn=dataset_batch_process_fn,
                            ema=ema,
                            num_batches=cfg.train.num_eval_batches_per_epoch,
                        )
                        stats.update(
                            {
                                f"val/{k}": v.mean().item()
                                for k, v in accelerator.gather(val_stats).items()
                            }
                        )
                    torch.cuda.empty_cache()
                    # run evals
                    if accelerator.is_main_process:
                        for eval_name, eval_fn in eval_fns.items():
                            if eval_name not in unwrapped_model.tasks:
                                continue
                            unwrapped_model = accelerator.unwrap_model(
                                model,
                                # some evaluation code uses ray, which doesn't support fp32 wrapper
                                # or torch compile
                                keep_fp32_wrapper=eval_name not in EVAL_FN_USES_RAY,
                                keep_torch_compile=eval_name not in EVAL_FN_USES_RAY,
                            )
                            task_model = unwrapped_model.tasks[eval_name]
                            use_flash_attn = task_model.use_flash_attn
                            task_model.use_flash_attn = (
                                eval_name not in EVAL_FN_USES_RAY and use_flash_attn
                            )

                            def decoder(
                                batch: dict[str, torch.Tensor],
                                seed=None,
                                guidance_fn=None,
                                _task_model=task_model,
                            ):
                                return _task_model.decode(
                                    batch=batch,
                                    backbone=unwrapped_model.backbone,
                                    pos_embs=unwrapped_model.pos_embs,
                                    seed=seed,
                                    guidance_fn=guidance_fn,
                                )

                            try:
                                eval_log_dir = (
                                    wandb.run.dir + f"/{eval_name}/{epoch:03d}/"
                                )
                                os.makedirs(eval_log_dir, exist_ok=True)
                                eval_stats = eval_fn(
                                    decoder=decoder,
                                    device=device,
                                    log_dir=eval_log_dir,
                                )
                                stats.update(
                                    {
                                        f"eval/{eval_name}/{k}": v.mean().item()
                                        for k, v in eval_stats.items()
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error in eval {eval_name}: {e}")
                                task_model.use_flash_attn = use_flash_attn
                                continue
                            finally:
                                task_model.use_flash_attn = use_flash_attn

                        for k, v in stats.items():
                            logger.info(f"{k}: {v:.03f}")

                        accelerator.log(stats, step=curr_train_step)
                ray.shutdown()
                gc.collect()
                ema.restore(model.parameters())

            if epoch % cfg.train.ckpt_every_n_epochs == 0:
                if accelerator.is_main_process:
                    accelerator.save(
                        {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "normalization": normalization_dict.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "epoch": epoch,
                            "curr_train_step": curr_train_step,
                        },
                        f"{wandb.run.dir}/{epoch:03d}.pt",
                    )

        # Save the final model
        if accelerator.is_main_process:
            accelerator.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "normalization": normalization_dict.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "curr_train_step": curr_train_step,
                },
                f"{wandb.run.dir}/final.pt",
            )
        accelerator.end_training()


if __name__ == "__main__":
    main()
