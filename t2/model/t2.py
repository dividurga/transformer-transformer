import gc
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, overload

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from t2.data.normalization_utils import normalization_dict_from_state_dict
from t2.model.core import DiT
from t2.model.task.task import (
    ComposedMultiModalDiffusionTask,
    MultiModalDiffusionTask,
    MultiModalTask,
)
from t2.model.utils import initialize_weights


class T2(nn.Module):
    def __init__(
        self,
        backbone: DiT,
        pos_embs: dict[str, nn.Module],
        tasks: dict[str, MultiModalTask],
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embs = nn.ModuleDict(pos_embs)
        self.tasks = nn.ModuleDict(tasks)
        initialize_weights(self)

    def forward(
        self,
        batches: list[tuple[list[str], dict[str, torch.Tensor]]],
        is_eval: bool,
    ) -> dict[str, torch.Tensor]:
        losses = {"loss": 0.0}
        for task_names, batch in batches:
            for name, task in self.tasks.items():
                if name not in task_names:
                    continue
                task_losses = task(
                    batch=batch,
                    backbone=self.backbone,
                    pos_embs=self.pos_embs,
                    is_eval=is_eval,
                )
                losses["loss"] = task_losses.pop("loss") + losses["loss"]
                losses.update({f"{name}/{k}": v for k, v in task_losses.items()})
        return losses  # type: ignore


T2Decoder = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
T2DecoderWithIntermediates = Callable[
    [dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]
]


@dataclass
class DecoderBundle:
    """Bundle containing a decoder callable and references to models for unloading.

    This allows callers to unload the model from GPU after pre-computation to free memory.

    When return_intermediates=True, the decoder returns a list of decoded samples
    at each denoising step (useful for visualization).
    """

    decoder: T2Decoder | T2DecoderWithIntermediates
    model: "T2"
    task_model: MultiModalDiffusionTask
    device: torch.device
    return_intermediates: bool = field(default=False)

    @overload
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        seed: Optional[int] = None,
        guidance_fn: Optional[
            Callable[
                [dict[str, torch.Tensor], dict[str, torch.Tensor]],
                dict[str, torch.Tensor],
            ]
        ] = None,
        *,
        return_intermediates: Literal[False] = ...,
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        seed: Optional[int] = None,
        guidance_fn: Optional[
            Callable[
                [dict[str, torch.Tensor], dict[str, torch.Tensor]],
                dict[str, torch.Tensor],
            ]
        ] = None,
        *,
        return_intermediates: Literal[True] = ...,
    ) -> list[dict[str, torch.Tensor]]: ...

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        seed: Optional[int] = None,
        guidance_fn: Optional[
            Callable[
                [dict[str, torch.Tensor], dict[str, torch.Tensor]],
                dict[str, torch.Tensor],
            ]
        ] = None,
        *,
        return_intermediates: Optional[bool] = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Call the decoder.

        Args:
            batch: Input batch with conditioning information.
            seed: Random seed for reproducibility.
            guidance_fn: Optional guidance function for classifier-free guidance.
            return_intermediates: If provided, override the bundle's setting.
                If None, use self.return_intermediates.

        Returns:
            If return_intermediates is False: Final decoded sample dict.
            If return_intermediates is True: List of decoded sample dicts.
        """
        # Use provided value or fall back to bundle's setting
        use_intermediates = (
            return_intermediates
            if return_intermediates is not None
            else self.return_intermediates
        )
        if use_intermediates != self.return_intermediates:
            raise ValueError(
                f"Cannot override return_intermediates at call time. "
                f"Bundle was created with return_intermediates={self.return_intermediates}, "
                f"but called with return_intermediates={return_intermediates}. "
                f"Create a new DecoderBundle with the desired setting."
            )
        # Note: decoder uses closure variable for return_intermediates, not a parameter
        return self.decoder(batch, seed=seed, guidance_fn=guidance_fn)  # type: ignore

    def unload_model(self) -> None:
        """Unload the model from GPU to free memory.

        Moves the model to CPU and clears CUDA cache.
        """
        self.model.to("cpu")
        self.task_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def setup_decoder(
    cfg: DictConfig,
    ckpt_path: str,
    device: torch.device,
    task_name: str,
    num_inference_steps: Optional[int] = None,
    num_repeats_per_step: int = 1,
    use_ema: bool = True,
    disable_grad: bool = True,
    clip_samples_in_guidance: bool = True,
    deterministic: bool = False,
    use_flash_attn: bool = True,
    eta: float = 0.0,
    use_mixed_precision: bool = True,
    use_torch_compile: bool = True,
    # Composition parameters for multi-trajectory optimization
    compose_sample_names: Optional[list[str]] = None,
    num_composed: int = 1,
    compose_method: str = "avg",
    # Intermediate sample collection for visualization
    return_intermediates: bool = False,
) -> DecoderBundle:
    """Set up a decoder for hardware generation.

    Args:
        cfg: Model configuration (from policy checkpoint).
        ckpt_path: Path to model checkpoint.
        device: Torch device for inference.
        task_name: Name of the task to use for decoding.
        num_inference_steps: Number of denoising steps. If None, uses default.
        num_repeats_per_step: Number of times to repeat each timestep.
        use_ema: Whether to use EMA weights from checkpoint.
        disable_grad: Whether to disable gradient computation.
        clip_samples_in_guidance: Whether to clip samples during guidance.
        deterministic: Whether to use deterministic sampling.
        use_flash_attn: Whether to use flash attention.
        eta: DDIM eta parameter.
        use_mixed_precision: Whether to use bfloat16 mixed precision.
        use_torch_compile: Whether to compile the decoder with torch.compile.
        compose_sample_names: Sample names for composition (multi-traj optimization).
        num_composed: Number of trajectories to compose.
        compose_method: Composition method ("avg" or "sum").
        return_intermediates: If True, decoder returns list of decoded samples
            at each denoising step (useful for visualization). If False, returns
            only the final decoded sample.

    Returns:
        DecoderBundle containing the decoder callable and model references.
    """
    t2 = hydra.utils.instantiate(cfg.model)
    t2 = t2.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalization_dict = normalization_dict_from_state_dict(ckpt["normalization"])

    # assign normalization to model
    for task in t2.tasks.values():
        for adapter in task.adapters.values():
            for norm_name, norm in normalization_dict.items():
                norm_group = norm_name.split("/")[0]
                if norm_group == adapter.modality.name:
                    attr_name = "/" + "/".join(norm_name.split("/")[1:])
                    adapter.modality.attr_norms[attr_name] = norm.to(device)

    if use_ema:
        for s_param, param in zip(ckpt["ema"]["shadow_params"], t2.parameters()):
            param.data.copy_(s_param.data)
    else:
        # training may have wrapped the model (torch.compile / accelerate),
        # prefixing state dict keys with _orig_mod. and/or module.
        state_dict = {
            k.removeprefix("_orig_mod.").removeprefix("module."): v
            for k, v in ckpt["model"].items()
        }
        t2.load_state_dict(state_dict)

    t2.eval()
    if task_name not in t2.tasks:
        raise ValueError(
            f"Task {task_name} not found in model: {list(t2.tasks.keys())}"
        )

    if disable_grad:
        for p in t2.parameters():
            p.requires_grad = False

    task_model: MultiModalDiffusionTask = t2.tasks[task_name]
    task_model.use_flash_attn = use_flash_attn

    # If composition is enabled, wrap the task with ComposedMultiModalDiffusionTask
    if compose_sample_names is not None and num_composed > 1:
        original_task = task_model
        # Get hidden_dim from timestep embedding (output dim of linear layers)
        hidden_dim = original_task.timestep_embedding.linear2.out_features
        task_model = ComposedMultiModalDiffusionTask(
            adapters=dict(original_task.adapters),
            use_flash_attn=original_task.use_flash_attn,
            hidden_dim=hidden_dim,
            num_inference_steps=original_task.num_inference_steps,
            noise_scheduler=original_task.noise_scheduler,
            timestep_mode=original_task.timestep_mode,
            num_eval_seeds=original_task.num_eval_seeds,
            compose_sample_names=compose_sample_names,
            num_composed=num_composed,
            compose_method=compose_method,
        )
        # Copy over the timestep embedding weights
        task_model.timestep_embedding.load_state_dict(
            original_task.timestep_embedding.state_dict()
        )
        task_model.to(device)
        task_model.eval()

    @torch.compile(disable=not use_torch_compile)
    def decoder(
        batch: dict[str, torch.Tensor],
        seed: Optional[int] = None,
        guidance_fn: Optional[
            Callable[
                [dict[str, torch.Tensor], dict[str, torch.Tensor]],
                dict[str, torch.Tensor],
            ]
        ] = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        with torch.autocast(
            device_type=str(device), dtype=torch.bfloat16, enabled=use_mixed_precision
        ):
            return task_model.decode(
                batch=batch,
                backbone=t2.backbone,
                pos_embs=t2.pos_embs,
                num_inference_steps=num_inference_steps,
                num_repeats_per_step=num_repeats_per_step,
                seed=seed,
                guidance_fn=guidance_fn,
                clip_samples_in_guidance=clip_samples_in_guidance,
                deterministic=deterministic,
                eta=eta,
                return_intermediates=return_intermediates,
            )

    return DecoderBundle(
        decoder=decoder,
        model=t2,
        task_model=task_model,
        device=device,
        return_intermediates=return_intermediates,
    )


def create_random_batch(
    cfg: DictConfig,
    batch_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Create a random batch based entirely on the config definition."""

    seq_lens = OmegaConf.to_container(cfg.seq_len, resolve=True)
    tasks_cfg = OmegaConf.to_container(cfg.model.tasks, resolve=True)

    modalities = {}
    for task in tasks_cfg.values():  # pyright: ignore
        for adapter in task.get("adapters", {}).values():
            mod_cfg = adapter.get("modality")
            if mod_cfg is None:
                continue
            name = mod_cfg.get("name", None)
            if name is None:
                continue
            modal = modalities.setdefault(
                name,
                {
                    "attrs": {},
                    "id_attrs": set(),
                    "attr_encoders": {},
                    "max_seq_len": mod_cfg["max_seq_len"],
                },
            )
            modal["attrs"].update(mod_cfg.get("attrs", {}))
            modal["id_attrs"].update(mod_cfg.get("id_attrs", []))
            modal["attr_encoders"].update(adapter.get("attr_encoders", {}))

    batch = {}
    for name, modal in modalities.items():
        seq_len = modal["max_seq_len"]
        batch[f"{name}/mask"] = torch.zeros(
            batch_size, seq_len, 1, dtype=torch.bool, device=device
        )
        for attr, dim in modal["attrs"].items():
            encoder_cfg = modal["attr_encoders"].get(attr)
            if encoder_cfg is not None and "num_bits" in encoder_cfg:
                bits = encoder_cfg["num_bits"]
                if attr.endswith("/id"):
                    group = attr.split("/")[-2]
                    max_val = seq_lens.get(group, 2)  # pyright: ignore
                else:
                    max_val = 2**bits
                tensor = torch.randint(
                    0, max_val, (batch_size, seq_len, 1), device=device
                )
                batch[f"{name}/{attr}"] = tensor.to(dtype)
            elif encoder_cfg is not None and not attr.endswith("/id"):
                # continuous-valued encoder (e.g. log encoding): sample values
                # in its valid (positive) input range, at the pre-encoding width
                # (SignedLogEncoding doubles width with a sign channel)
                raw_dim = dim
                if str(encoder_cfg.get("_target_", "")).endswith("SignedLogEncoding"):
                    raw_dim = dim // 2
                shape = (
                    (batch_size, seq_len, raw_dim)
                    if raw_dim > 1
                    else (batch_size, seq_len)
                )
                batch[f"{name}/{attr}"] = torch.rand(*shape, device=device, dtype=dtype)
            elif attr.endswith("/id") or attr == "id":
                group = attr.split("/")[-2] if "/" in attr else name
                max_val = seq_lens.get(group, 2)  # pyright: ignore
                batch[f"{name}/{attr}"] = torch.randint(
                    0, max_val, (batch_size, seq_len), device=device
                ).to(dtype)
            else:
                shape = (batch_size, seq_len, dim) if dim > 1 else (batch_size, seq_len)
                batch[f"{name}/{attr}"] = torch.randn(
                    *shape, device=device, dtype=dtype
                )

        for id_attr in modal["id_attrs"]:
            if id_attr in modal["attrs"]:
                continue
            group = id_attr.split("/")[-2] if "/" in id_attr else name
            max_val = seq_lens.get(group, 2)  # pyright: ignore
            batch[f"{name}/{id_attr}"] = torch.randint(
                0, max_val, (batch_size, seq_len), device=device
            ).to(dtype)

    return batch
