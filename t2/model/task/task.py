import logging
import typing
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, OrderedDict, cast, overload

import torch
import torch.nn as nn
import tqdm
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel

from t2.model.core import DiT, TimestepEmbedder
from t2.model.modality import (
    DiffusionModalityAdapter,
    ModalityAdapter,
)
from t2.utils.misc import get_batch_device, get_batch_size, sorted_dict


class MultiModalTask(ABC, nn.Module):
    """
    Each modality the task requires will provide an
    adapter than handles input preparation to and loss
    computation from the backbone.
    """

    def __init__(
        self, adapters: dict[str, ModalityAdapter], use_flash_attn: bool = False
    ):
        super().__init__()
        # this is a promise that no masks will be used
        self.use_flash_attn = use_flash_attn
        self.adapters = cast(
            OrderedDict,
            nn.ModuleDict(
                OrderedDict(
                    [(name, adapter) for name, adapter in sorted_dict(adapters).items()]
                )
            ),
        )

    @abstractmethod
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        is_eval: bool,
    ) -> dict[str, torch.Tensor]:
        pass


class MultiModalDiffusionTask(MultiModalTask):
    """
    Multi-modal joint diffusion task, where all modalities
    are modelled with a single diffusion process. It does not
    support different diffusion schedules for different modalities.

    Args:
        adapters: List of modality adapters.
        hidden_dim: Hidden dimension of the backbone.
    """

    TIMESTEP_MODES = ["condition", "context", "add"]

    def __init__(
        self,
        adapters: dict[str, ModalityAdapter],
        use_flash_attn: bool,
        hidden_dim: int,
        # diffusion support
        num_inference_steps: int,
        noise_scheduler: DDIMScheduler,
        timestep_mode: str,
        num_eval_seeds: int,
    ):
        super().__init__(adapters=adapters, use_flash_attn=use_flash_attn)
        self.timestep_embedding = TimestepEmbedder(hidden_dim)
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = noise_scheduler
        self.timestep_mode = timestep_mode
        self.num_eval_seeds = num_eval_seeds
        assert timestep_mode in self.TIMESTEP_MODES, (
            f"Invalid timestep mode: {timestep_mode}"
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        is_eval: bool,
    ) -> dict[str, torch.Tensor]:
        device = get_batch_device(batch)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,  # type: ignore
            (get_batch_size(batch),),
            device=device,
        ).long()

        t_emb = self.timestep_embedding(timesteps)
        backbone_ins = {}
        backbone_masks = OrderedDict()
        backbone_conds: dict[str, torch.Tensor] = {}

        adapter_targets = {}
        adapter_batch = {
            "timesteps": timesteps,
            # Hack to use shared noise scheduler
            "noise_scheduler": self.noise_scheduler,  # type: ignore
            **batch,
        }
        for name, adapter in sorted_dict(self.adapters).items():
            data = adapter.prepare_tokens(
                adapter_batch,
                pos_embs=pos_embs,
            )
            if "input_seq" in data:
                backbone_ins[name] = data["input_seq"]
            if "seq_mask" in data:
                assert "input_seq" in data, "masks received but no input sequence"
                assert data["input_seq"].shape[:2] == data["seq_mask"].shape[:2], (
                    "mask and input sequence must have same batch and sequence dimensions"
                )
                backbone_masks[name] = data["seq_mask"]
            if "target_seq" in data:
                adapter_targets[name] = data["target_seq"]
            if "cond" in data:
                backbone_conds[name] = data["cond"]

        # prepare backbone inputs
        if self.timestep_mode == "condition":
            backbone_conds["t"] = t_emb
        elif self.timestep_mode == "context":
            backbone_ins["t"] = t_emb[:, None, :]
            backbone_masks["t"] = torch.zeros_like(timesteps[:, None], dtype=torch.bool)
        elif self.timestep_mode == "add":
            backbone_ins = {k: v + t_emb[:, None, :] for k, v in backbone_ins.items()}
        if len(backbone_conds) > 0:
            cond = torch.zeros_like(t_emb)
            assert all(v.shape == cond.shape for v in backbone_conds.values()), (
                "all cond terms must have same shape, but got"
                + ", ".join([f"{k}: {v.shape}" for k, v in backbone_conds.items()])
            )
            for cond_term in backbone_conds.values():
                cond += cond_term
        else:
            cond = None

        sorted_backbone_ins = sorted_dict(backbone_ins)
        sorted_input_seq = torch.cat(list(sorted_backbone_ins.values()), dim=1)
        if len(backbone_masks) > 0:
            sorted_masks = sorted_dict(backbone_masks)
            sorted_mask_seq = torch.cat(list(sorted_masks.values()), dim=1)
        else:
            sorted_mask_seq = None
        if self.use_flash_attn:
            if sorted_mask_seq is not None and sorted_mask_seq.any():
                logging.error(
                    f"nothing should be masked but found {sorted_mask_seq.sum()} items"
                )
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                sorted_feat_seq = backbone(
                    x=sorted_input_seq,
                    cond=cond,
                )
        else:
            sorted_feat_seq = backbone(
                x=sorted_input_seq,
                key_padding_mask=sorted_mask_seq,
                cond=cond,
            )

        losses = {
            "loss": torch.tensor(0.0, device=device),
        }
        idx = 0
        for name, input_seq in sorted_backbone_ins.items():
            # parse out feature sequence in same order as input_seq
            seq_len = input_seq.shape[1]
            if name in self.adapters:
                adapter = self.adapters[name]
                target_seq = adapter_targets.get(name, None)
                seq_mask = backbone_masks.get(name, None)
                if adapter.loss_weight != 0.0:
                    losses[name] = adapter.get_loss(
                        batch={
                            "feat_seq": sorted_feat_seq[:, idx : idx + seq_len],
                            "target_seq": target_seq,
                            "cond": cond,
                            "seq_mask": seq_mask,
                        }
                    )
                    losses["loss"] += losses[name] * adapter.loss_weight
            idx += seq_len
        if is_eval:
            decoded = {}
            for _ in range(self.num_eval_seeds):
                decoded_seed = self.decode(
                    batch=batch,
                    backbone=backbone,
                    pos_embs=pos_embs,
                    use_pbar=False,
                )
                for k, v in decoded_seed.items():
                    if k not in decoded:
                        decoded[k] = []
                    decoded[k].append(v)
            decoded = {k: torch.stack(v, dim=1) for k, v in decoded.items()}
            for k, pred in decoded.items():
                adapter_name = k.split("/")[0]
                adapter = self.adapters[adapter_name]
                diffusion_adapter = typing.cast(DiffusionModalityAdapter, adapter)

                attr_key = k.split(adapter_name + "/")[-1]
                override_noise_prob = diffusion_adapter.attrs_noise_probs.get(
                    attr_key, 0.0
                )
                no_override = attr_key not in diffusion_adapter.attrs_noise_probs
                key_is_an_output = (
                    diffusion_adapter.default_diffuse_prob == 1.0 and no_override
                ) or (override_noise_prob > 0.0)
                if not key_is_an_output:
                    continue
                target = batch[k].to(pred.dtype)  # (batch, seq_len, attr_dim)
                target = target.unsqueeze(1).repeat(
                    1, self.num_eval_seeds, *([1] * (target.ndim - 1))
                )
                pred = pred.view(
                    target.shape
                )  # (batch, num_eval_seeds, seq_len, attr_dim)
                mask = batch[adapter_name + "/mask"].squeeze(dim=-1)

                if k.endswith("/mask"):
                    adapter = self.adapters[k.split("/")[0]]
                    if not adapter.only_use_mask_for_padding:
                        continue
                    metric = (pred == target).float()
                    metric_key = "accuracy"
                elif pred.dtype in [torch.long, torch.bool, torch.short]:
                    metric = (
                        (pred == target)
                        .all(dim=-1)[~mask[:, None, :].expand(pred.shape[:3])]
                        .float()
                    )
                    metric_key = "accuracy"
                else:
                    metric = (
                        torch.nn.functional.mse_loss(pred, target, reduction="none")[
                            ~mask[:, None, :].expand(pred.shape[:3])
                        ]
                        .mean(dim=-1)
                        .float()
                    )
                    metric_key = "mse"

                losses[f"{k}/{metric_key}"] = metric.mean()
        return losses  # type: ignore

    def _init_denoising_state(
        self,
        batch: dict[str, torch.Tensor],
        pos_embs: nn.ModuleDict,
        batch_size: int,
        device: torch.device,
        seed: Optional[int],
        deterministic: bool,
    ) -> tuple[
        OrderedDict,
        OrderedDict,
        OrderedDict,
        dict[str, ModalityAdapter],
        dict[str, nn.Parameter | torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, slice],
        int,
        Optional[torch.Generator],
    ]:
        """Prepare state dictionaries for the decoding loop."""

        backbone_conds: OrderedDict[str, torch.Tensor] = OrderedDict()
        backbone_ins: OrderedDict[str, torch.Tensor] = OrderedDict()
        backbone_masks: OrderedDict[str, torch.Tensor] = OrderedDict()
        denoising_adapters: dict[str, ModalityAdapter] = {}
        samples: dict[str, nn.Parameter | torch.Tensor] = {}
        # `dont_update_masks`` is True if samples should be updated during
        # denoising process and is per attribute dimension
        dont_update_masks: dict[str, torch.Tensor] = {}
        seq_slices: dict[str, slice] = {}
        seq_idx = 0

        generator = None

        if seed is not None and not deterministic:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        keys = list(self.adapters.keys())
        if self.timestep_mode == "context":
            keys = ["t"] + keys

        for name in sorted(keys):
            if name == "t":
                seq_slices[name] = slice(seq_idx, seq_idx + 1)
                seq_idx += 1
                continue

            adapter = self.adapters[name]

            if batch.get(name + "/mask", None) is not None:
                assert batch[name + "/mask"].shape == (
                    batch_size,
                    adapter.modality.max_seq_len,
                ) or batch[name + "/mask"].shape == (
                    batch_size,
                    adapter.modality.max_seq_len,
                    1,
                )
                backbone_masks[name] = batch[name + "/mask"].reshape(
                    batch_size, adapter.modality.max_seq_len
                )
            else:
                backbone_masks[name] = torch.zeros(
                    (
                        batch_size,
                        adapter.modality.max_seq_len,
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            if isinstance(adapter, DiffusionModalityAdapter):
                diffusion_adapter = typing.cast(DiffusionModalityAdapter, adapter)
                dont_noise = diffusion_adapter.get_dont_noise_mask(batch)
                dont_update_masks[name] = dont_noise
                if not diffusion_adapter.only_use_mask_for_padding:
                    # this means for this modality, the mask is actually used
                    # for masking too (and thus, whether or not samples get updated)
                    # during denoising process
                    dont_update_masks[name] = torch.logical_or(
                        backbone_masks[name][..., None].expand_as(dont_noise),
                        dont_noise,
                    )
                else:
                    # None are masked, so the diffusion process should diffuse
                    # everything, including the padding tokens
                    backbone_masks[name][:] = False
                denoising_adapters[name] = adapter
                if dont_noise.any():
                    target_seq = adapter.modality.to_tensor(
                        batch, encoders=adapter.attr_encoders, normalize=True
                    )
                    samples[name] = torch.where(
                        dont_noise,
                        target_seq,
                        (
                            torch.randn(
                                (
                                    batch_size,
                                    adapter.modality.max_seq_len,
                                    adapter.modality.dim,
                                ),
                                device=device,
                                generator=generator,
                            )
                            if not deterministic
                            else torch.zeros(
                                (
                                    batch_size,
                                    adapter.modality.max_seq_len,
                                    adapter.modality.dim,
                                ),
                                device=device,
                            )
                        ),
                    )
                else:
                    samples[name] = (
                        torch.randn(
                            (
                                batch_size,
                                adapter.modality.max_seq_len,
                                adapter.modality.dim,
                            ),
                            device=device,
                            generator=generator,
                        )
                        if not deterministic
                        else torch.zeros(
                            (
                                batch_size,
                                adapter.modality.max_seq_len,
                                adapter.modality.dim,
                            ),
                            device=device,
                        )
                    )

            else:
                data = adapter.prepare_tokens(batch, pos_embs)
                if "input_seq" in data:
                    backbone_ins[name] = data["input_seq"]
                if "seq_mask" in data:
                    assert "input_seq" in data, "masks received but no input sequence"
                    assert data["input_seq"].shape[:2] == data["seq_mask"].shape[:2], (
                        "mask and input sequence must have same batch and sequence dimensions"
                    )
                    backbone_masks[name] = data["seq_mask"].reshape(
                        batch_size, adapter.modality.max_seq_len
                    )
                if "cond" in data:
                    backbone_conds[name] = data["cond"]

            if name in backbone_ins or name in samples:
                seq_slices[name] = slice(
                    seq_idx, seq_idx + adapter.modality.max_seq_len
                )
                seq_idx += adapter.modality.max_seq_len

        return (
            backbone_conds,
            backbone_ins,
            backbone_masks,
            denoising_adapters,
            samples,
            dont_update_masks,
            seq_slices,
            seq_idx,
            generator,
        )

    def _prepare_backbone_inputs(
        self,
        batch: dict[str, torch.Tensor],
        pos_embs: nn.ModuleDict,
        samples: dict[str, nn.Parameter],
        backbone_ins: OrderedDict,
    ) -> None:
        """Update ``backbone_ins`` with projected samples and pos embeddings."""
        for name, sample in samples.items():
            backbone_ins[name] = self.adapters[name].in_proj(sample)
            for id_attr in sorted(self.adapters[name].modality.id_attrs):
                full_attr = self.adapters[name].modality.name + "/" + id_attr
                pos_ids = batch[full_attr]
                pos_emb_group = full_attr.split("/")[-2]
                backbone_ins[name] += pos_embs[pos_emb_group](pos_ids.long()).view(
                    backbone_ins[name].shape
                )

    def _apply_timestep(
        self,
        t_emb: torch.Tensor,
        batch_size: int,
        backbone_ins: OrderedDict,
        backbone_masks: OrderedDict,
        backbone_conds: OrderedDict,
        device: torch.device,
    ) -> None:
        if self.timestep_mode == "condition":
            backbone_conds["t"] = t_emb
        elif self.timestep_mode == "context":
            backbone_ins["t"] = t_emb[:, None, :]
            backbone_masks["t"] = torch.zeros(
                (batch_size, 1), device=device, dtype=torch.bool
            )
        elif self.timestep_mode == "add":
            for k in backbone_ins.keys():
                backbone_ins[k] = backbone_ins[k] + t_emb[:, None, :]

    def _build_cond(
        self, t_emb: torch.Tensor, backbone_conds: OrderedDict
    ) -> Optional[torch.Tensor]:
        if len(backbone_conds) == 0:
            return None
        cond = torch.zeros_like(t_emb)
        assert all(v.shape == cond.shape for v in backbone_conds.values()), (
            "all cond terms must have same shape, but got"
            + ", ".join([f"{k}: {v.shape}" for k, v in backbone_conds.items()])
        )
        for cond_term in backbone_conds.values():
            cond += cond_term
        return cond

    def _run_backbone(
        self,
        backbone: DiT,
        backbone_ins: OrderedDict,
        backbone_masks: OrderedDict,
        cond: Optional[torch.Tensor],
        seq_idx: int,
    ) -> torch.Tensor:
        x = torch.cat(list(sorted_dict(backbone_ins).values()), dim=1)
        key_padding_mask = None
        if len(backbone_masks) > 0:
            key_padding_mask = torch.cat(
                list(sorted_dict(backbone_masks).values()), dim=1
            )
        assert x.shape[1] == seq_idx, f"{x.shape[1]} != {seq_idx}"
        if self.use_flash_attn:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return backbone(x=x, cond=cond)
        else:
            return backbone(x=x, key_padding_mask=key_padding_mask, cond=cond)

    def _denoise_samples(
        self,
        t: torch.Tensor,
        batch: dict[str, torch.Tensor],
        backbone_outputs: torch.Tensor,
        samples: dict[str, nn.Parameter | torch.Tensor],
        dont_update_masks: dict[str, torch.Tensor],
        seq_slices: dict[str, slice],
        generator: Optional[torch.Generator],
        cond: Optional[torch.Tensor],
        denoising_adapters: dict[str, ModalityAdapter],
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ],
        clip_samples_in_guidance: bool,
        eta: float = 0.0,
    ) -> None:
        # just tensors, not parameters
        latest_samples: dict[str, torch.Tensor] = {}

        for name in samples.keys():
            out_features = backbone_outputs[:, seq_slices[name]]
            if self.adapters[name].loss_weight == 0.0:
                continue
            pred_noise = self.adapters[name].out_proj(x=out_features, cond=cond)

            dont_update = dont_update_masks[name]
            latest_samples[name] = torch.where(
                dont_update,
                samples[name],
                self.noise_scheduler.step(
                    eta=eta,
                    model_output=pred_noise,
                    timestep=t,
                    sample=samples[name],
                    generator=generator,
                ).prev_sample,  # type: ignore
            )

        if guidance_fn is None:
            for name in latest_samples.keys():
                samples[name] = latest_samples[name]
        else:
            guidances = guidance_fn(
                {
                    **batch,
                    **self._decode_samples(
                        denoising_adapters,
                        latest_samples,
                        clip_samples=clip_samples_in_guidance,
                    ),
                },
                {**batch, **samples},
            )

            for name in samples.keys():
                if self.adapters[name].loss_weight == 0.0:
                    continue
                if name not in guidances:
                    samples[name] = latest_samples[name]
                    continue
                out_features = backbone_outputs[:, seq_slices[name]]
                pred_noise = self.adapters[name].out_proj(x=out_features, cond=cond)
                guidance = guidances[name]
                # NOTE we're adding, not subtracting like the original classifier
                # guidance paper because these are gradients w.r.t to cost (minimize)
                # not logprob (maximize)
                pred_noise = (
                    pred_noise
                    + (1 - self.noise_scheduler.alphas_cumprod[t]).sqrt() * guidance
                )

                dont_update = dont_update_masks[name]
                samples[name] = torch.where(
                    dont_update,
                    samples[name],
                    self.noise_scheduler.step(
                        eta=eta,
                        model_output=pred_noise,
                        timestep=t,
                        sample=samples[name],
                        generator=generator,
                    ).prev_sample,  # type: ignore
                )

    def _decode_samples(
        self,
        denoising_adapters: dict[str, ModalityAdapter],
        samples: dict[str, nn.Parameter | torch.Tensor],
        clip_samples: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        return {
            k: v
            for name, adapter in denoising_adapters.items()
            for k, v in adapter.modality.from_tensor(
                samples[name],
                decoders=adapter.attr_decoders,
                unnormalize=True,
                clip=clip_samples,
            ).items()
        }

    @overload
    def decode(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        num_inference_steps: Optional[int] = None,
        num_repeats_per_step: int = 1,
        use_pbar: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ] = None,
        clip_samples_in_guidance: bool = True,
        eta: float = 0.0,
        return_intermediates: Literal[False] = False,
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def decode(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        num_inference_steps: Optional[int] = None,
        num_repeats_per_step: int = 1,
        use_pbar: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ] = None,
        clip_samples_in_guidance: bool = True,
        eta: float = 0.0,
        return_intermediates: Literal[True] = ...,
    ) -> list[dict[str, torch.Tensor]]: ...

    def decode(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        num_inference_steps: Optional[int] = None,
        num_repeats_per_step: int = 1,
        use_pbar: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ] = None,
        clip_samples_in_guidance: bool = True,
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Decode samples from diffusion model.

        Args:
            batch: Input batch with conditioning information.
            backbone: DiT backbone network.
            pos_embs: Position embeddings module dict.
            num_inference_steps: Number of denoising steps. If None, uses task default.
            num_repeats_per_step: Number of times to repeat each timestep.
            use_pbar: Whether to show a progress bar.
            seed: Random seed for reproducibility.
            deterministic: If True, use deterministic sampling.
            guidance_fn: Optional guidance function for classifier-free guidance.
            clip_samples_in_guidance: Whether to clip samples during guidance.
            eta: DDIM eta parameter (0=deterministic, 1=DDPM).
            return_intermediates: If True, return list of decoded samples at each
                denoising step (plus initial noise). Useful for visualization.

        Returns:
            If return_intermediates=False: Final decoded sample dict.
            If return_intermediates=True: List of decoded sample dicts, one per
                denoising step plus initial noise. Intermediate samples use
                clip_samples=False; final sample uses clip_samples=True.
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        device = get_batch_device(batch)
        batch_size = get_batch_size(batch)

        (
            backbone_conds,
            backbone_ins,
            backbone_masks,
            denoising_adapters,
            samples,
            dont_update_masks,
            seq_slices,
            seq_idx,
            generator,
        ) = self._init_denoising_state(
            batch=batch,
            pos_embs=pos_embs,
            batch_size=batch_size,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )
        if len(backbone_masks) > 0:
            # either all modalities should have masks or none should have masks
            no_mask_keys = set(backbone_ins.keys()) - set(backbone_masks.keys())
            if len(no_mask_keys) > 0:
                raise ValueError(
                    f"No mask keys: {no_mask_keys}. All modalities must have a mask."
                )

        # Collect intermediate samples if requested
        intermediate_samples: list[dict[str, torch.Tensor]] = []
        if return_intermediates:
            # Capture initial (noisy) samples
            intermediate_samples.append(
                {
                    k: v.detach()
                    for k, v in self._decode_samples(
                        denoising_adapters, samples, clip_samples=False
                    ).items()
                }
            )

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = (
            self.noise_scheduler.timesteps[:, None]
            .repeat(1, num_repeats_per_step)
            .reshape(-1)
        )
        if use_pbar:
            pbar = tqdm.tqdm(
                timesteps,
                desc="Denoising",
                dynamic_ncols=True,
            )
        else:
            pbar = timesteps

        # start denoising process
        for t in pbar:
            # Clear gradients for all parameters
            samples = {k: nn.Parameter(v.detach()) for k, v in samples.items()}

            self._prepare_backbone_inputs(
                batch=batch,
                pos_embs=pos_embs,
                samples=samples,
                backbone_ins=backbone_ins,
            )
            t_emb = self.timestep_embedding(t.expand(batch_size))
            self._apply_timestep(
                t_emb,
                batch_size,
                backbone_ins,
                backbone_masks,
                backbone_conds,
                device,
            )

            cond = self._build_cond(t_emb, backbone_conds)
            backbone_outputs = self._run_backbone(
                backbone,
                backbone_ins,
                backbone_masks,
                cond,
                seq_idx,
            )

            self._denoise_samples(
                t=t,
                batch=batch,
                backbone_outputs=backbone_outputs,
                samples=samples,
                dont_update_masks=dont_update_masks,
                seq_slices=seq_slices,
                generator=generator,
                cond=cond,
                denoising_adapters=denoising_adapters,
                guidance_fn=guidance_fn,
                clip_samples_in_guidance=clip_samples_in_guidance,
                eta=eta,
            )

            # Capture intermediate samples after each denoising step
            if return_intermediates:
                intermediate_samples.append(
                    {
                        k: v.detach()
                        for k, v in self._decode_samples(
                            denoising_adapters, samples, clip_samples=False
                        ).items()
                    }
                )

        # Decode diffusion samples (final, with clipping)
        result = {
            k: v.detach()
            for k, v in self._decode_samples(
                denoising_adapters, samples, clip_samples=True
            ).items()
        }

        if return_intermediates:
            # Replace the last intermediate sample with the properly clipped final result
            # (the last intermediate was decoded with clip_samples=False)
            intermediate_samples[-1] = result
            return intermediate_samples

        return result

class ComposedMultiModalDiffusionTask(MultiModalDiffusionTask):
    """
    Multi-modal diffusion task with composed noise prediction for multi-trajectory optimization.

    This class extends MultiModalDiffusionTask to support composing predicted noise outputs
    from multiple parallel diffusion processes (e.g., one per trajectory) by averaging or
    summing them before the scheduler step. This allows optimizing for designs that work
    well across multiple target trajectories simultaneously.

    The batch dimension is organized as: [traj0_sample0, traj1_sample0, ..., trajN_sample0,
    traj0_sample1, traj1_sample1, ..., trajN_sample1, ...] where N = num_composed - 1.

    Args:
        compose_sample_names: List of sample names to apply composition to (e.g., ["link", "dyna_joint"]).
        num_composed: Number of parallel diffusion processes to compose (typically one per trajectory).
        compose_method: Method for combining noise predictions, either "avg" or "sum".
        **kwargs: Arguments passed to MultiModalDiffusionTask.
    """

    def __init__(
        self,
        compose_sample_names: list[str],
        num_composed: int,
        compose_method: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compose_sample_names = compose_sample_names
        self.num_composed = num_composed
        self.compose_method = compose_method
        assert compose_method in [
            "avg",
            "sum",
        ], f"Invalid compose_method: {compose_method}. Must be 'avg' or 'sum'."

    def _compose_pred_noise(
        self, pred_noise: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Combine predicted noise across composed subbatch.

        Args:
            pred_noise: Predicted noise tensor of shape (batch_size, seq_len, dim).
            batch_size: Total batch size (num_composed * num_samples_per_traj).

        Returns:
            Composed noise with same shape, where each subbatch has identical values.
        """
        # Reshape to (num_samples_per_traj, num_composed, seq_len, dim)
        reshaped = pred_noise.reshape(
            batch_size // self.num_composed,
            self.num_composed,
            *pred_noise.shape[1:],
        )
        if self.compose_method == "avg":
            composed = reshaped.mean(dim=1, keepdim=True)
        else:  # sum
            composed = reshaped.sum(dim=1, keepdim=True)
        # Repeat to all positions in the subbatch and reshape back
        repeated = composed.repeat(1, self.num_composed, *([1] * (pred_noise.ndim - 1)))
        return repeated.reshape_as(pred_noise)

    def _ensure_composed_consistency(
        self,
        samples: dict[str, nn.Parameter | torch.Tensor],
        batch_size: int,
    ) -> dict[str, nn.Parameter | torch.Tensor]:
        """
        Ensure all samples in each composed subbatch have identical values.

        This is called after _init_denoising_state to ensure identical initialization,
        and after each denoising step when eta > 0 (to handle stochastic noise injection).

        Args:
            samples: Dictionary of sample tensors.
            batch_size: Total batch size.

        Returns:
            Updated samples dict with consistency enforced for composed terms.
        """
        for name in self.compose_sample_names:
            if name not in samples:
                continue
            sample = samples[name]
            # Reshape to (num_samples_per_traj, num_composed, seq_len, dim)
            reshaped = sample.reshape(
                batch_size // self.num_composed,
                self.num_composed,
                *sample.shape[1:],
            )
            # Take first sample in each subbatch and repeat to all positions
            first = reshaped[:, 0:1].repeat(
                1, self.num_composed, *([1] * (sample.ndim - 1))
            )
            # Preserve the type (Parameter or Tensor)
            if isinstance(sample, nn.Parameter):
                samples[name] = nn.Parameter(first.reshape_as(sample))
            else:
                samples[name] = first.reshape_as(sample)
        return samples

    def _denoise_samples(
        self,
        t: torch.Tensor,
        batch: dict[str, torch.Tensor],
        backbone_outputs: torch.Tensor,
        samples: dict[str, nn.Parameter | torch.Tensor],
        dont_update_masks: dict[str, torch.Tensor],
        seq_slices: dict[str, slice],
        generator: Optional[torch.Generator],
        cond: Optional[torch.Tensor],
        denoising_adapters: dict[str, ModalityAdapter],
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ],
        clip_samples_in_guidance: bool,
        eta: float = 0.0,
    ) -> None:
        """
        Denoise samples with composed noise prediction for specified modalities.

        For modalities in compose_sample_names, the predicted noise is averaged/summed
        across the composed subbatch before being passed to the scheduler step.
        """
        batch_size = backbone_outputs.shape[0]
        latest_samples: dict[str, torch.Tensor] = {}

        for name in samples.keys():
            out_features = backbone_outputs[:, seq_slices[name]]
            if self.adapters[name].loss_weight == 0.0:
                continue
            pred_noise = self.adapters[name].out_proj(x=out_features, cond=cond)

            # Apply composition for specified sample names
            if name in self.compose_sample_names and self.num_composed > 1:
                pred_noise = self._compose_pred_noise(pred_noise, batch_size)

            dont_update = dont_update_masks[name]
            latest_samples[name] = torch.where(
                dont_update,
                samples[name],
                self.noise_scheduler.step(
                    eta=eta,
                    model_output=pred_noise,
                    timestep=t,
                    sample=samples[name],
                    generator=generator,
                ).prev_sample,  # type: ignore
            )

        if guidance_fn is None:
            for name in latest_samples.keys():
                samples[name] = latest_samples[name]
        else:
            guidances = guidance_fn(
                {
                    **batch,
                    **self._decode_samples(
                        denoising_adapters,
                        latest_samples,
                        clip_samples=clip_samples_in_guidance,
                    ),
                },
                {**batch, **samples},
            )

            for name in samples.keys():
                if self.adapters[name].loss_weight == 0.0:
                    continue
                if name not in guidances:
                    samples[name] = latest_samples[name]
                    continue
                out_features = backbone_outputs[:, seq_slices[name]]
                pred_noise = self.adapters[name].out_proj(x=out_features, cond=cond)

                # Apply composition for specified sample names
                if name in self.compose_sample_names and self.num_composed > 1:
                    pred_noise = self._compose_pred_noise(pred_noise, batch_size)

                guidance = guidances[name]
                # NOTE we're adding, not subtracting like the original classifier
                # guidance paper because these are gradients w.r.t to cost (minimize)
                # not logprob (maximize)
                pred_noise = (
                    pred_noise
                    + (1 - self.noise_scheduler.alphas_cumprod[t]).sqrt() * guidance
                )

                dont_update = dont_update_masks[name]
                samples[name] = torch.where(
                    dont_update,
                    samples[name],
                    self.noise_scheduler.step(
                        eta=eta,
                        model_output=pred_noise,
                        timestep=t,
                        sample=samples[name],
                        generator=generator,
                    ).prev_sample,  # type: ignore
                )

    @overload
    def decode(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        num_inference_steps: Optional[int] = None,
        num_repeats_per_step: int = 1,
        use_pbar: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ] = None,
        clip_samples_in_guidance: bool = True,
        eta: float = 0.0,
        return_intermediates: Literal[False] = False,
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def decode(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        num_inference_steps: Optional[int] = None,
        num_repeats_per_step: int = 1,
        use_pbar: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ] = None,
        clip_samples_in_guidance: bool = True,
        eta: float = 0.0,
        return_intermediates: Literal[True] = ...,
    ) -> list[dict[str, torch.Tensor]]: ...

    def decode(
        self,
        batch: dict[str, torch.Tensor],
        backbone: DiT,
        pos_embs: nn.ModuleDict,
        num_inference_steps: Optional[int] = None,
        num_repeats_per_step: int = 1,
        use_pbar: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        guidance_fn: Optional[
            Callable[
                [
                    dict[str, torch.Tensor],
                    dict[str, nn.Parameter | torch.Tensor],
                ],
                dict[str, torch.Tensor],
            ]
        ] = None,
        clip_samples_in_guidance: bool = True,
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """
        Decode with composed diffusion for multi-trajectory optimization.

        This extends the base decode method to:
        1. Ensure composed samples have identical initialization after _init_denoising_state
        2. Enforce consistency after each denoising step when eta > 0 (stochastic sampling)

        Args:
            batch: Input batch with conditioning information.
            backbone: DiT backbone network.
            pos_embs: Position embeddings module dict.
            num_inference_steps: Number of denoising steps. If None, uses task default.
            num_repeats_per_step: Number of times to repeat each timestep.
            use_pbar: Whether to show a progress bar.
            seed: Random seed for reproducibility.
            deterministic: If True, use deterministic sampling.
            guidance_fn: Optional guidance function for classifier-free guidance.
            clip_samples_in_guidance: Whether to clip samples during guidance.
            eta: DDIM eta parameter (0=deterministic, 1=DDPM).
            return_intermediates: If True, return list of decoded samples at each
                denoising step (plus initial noise). Useful for visualization.

        Returns:
            If return_intermediates=False: Final decoded sample dict.
            If return_intermediates=True: List of decoded sample dicts, one per
                denoising step plus initial noise. Intermediate samples use
                clip_samples=False; final sample uses clip_samples=True.
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        device = get_batch_device(batch)
        batch_size = get_batch_size(batch)

        (
            backbone_conds,
            backbone_ins,
            backbone_masks,
            denoising_adapters,
            samples,
            dont_update_masks,
            seq_slices,
            seq_idx,
            generator,
        ) = self._init_denoising_state(
            batch=batch,
            pos_embs=pos_embs,
            batch_size=batch_size,
            device=device,
            seed=seed,
            deterministic=deterministic,
        )

        # Ensure identical initialization for composed samples
        if self.num_composed > 1 and len(self.compose_sample_names) > 0:
            samples = self._ensure_composed_consistency(samples, batch_size)

        if len(backbone_masks) > 0:
            # either all modalities should have masks or none should have masks
            no_mask_keys = set(backbone_ins.keys()) - set(backbone_masks.keys())
            if len(no_mask_keys) > 0:
                raise ValueError(
                    f"No mask keys: {no_mask_keys}. All modalities must have a mask."
                )

        # Collect intermediate samples if requested
        intermediate_samples: list[dict[str, torch.Tensor]] = []
        if return_intermediates:
            # Capture initial (noisy) samples
            intermediate_samples.append(
                {
                    k: v.detach()
                    for k, v in self._decode_samples(
                        denoising_adapters, samples, clip_samples=False
                    ).items()
                }
            )

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = (
            self.noise_scheduler.timesteps[:, None]
            .repeat(1, num_repeats_per_step)
            .reshape(-1)
        )
        if use_pbar:
            pbar = tqdm.tqdm(
                timesteps,
                desc="Denoising",
                dynamic_ncols=True,
            )
        else:
            pbar = timesteps

        # start denoising process
        for t in pbar:
            # Clear gradients for all parameters
            samples = {k: nn.Parameter(v.detach()) for k, v in samples.items()}

            self._prepare_backbone_inputs(
                batch=batch,
                pos_embs=pos_embs,
                samples=samples,
                backbone_ins=backbone_ins,
            )
            t_emb = self.timestep_embedding(t.expand(batch_size))
            self._apply_timestep(
                t_emb,
                batch_size,
                backbone_ins,
                backbone_masks,
                backbone_conds,
                device,
            )

            cond = self._build_cond(t_emb, backbone_conds)
            backbone_outputs = self._run_backbone(
                backbone,
                backbone_ins,
                backbone_masks,
                cond,
                seq_idx,
            )

            self._denoise_samples(
                t=t,
                batch=batch,
                backbone_outputs=backbone_outputs,
                samples=samples,
                dont_update_masks=dont_update_masks,
                seq_slices=seq_slices,
                generator=generator,
                cond=cond,
                denoising_adapters=denoising_adapters,
                guidance_fn=guidance_fn,
                clip_samples_in_guidance=clip_samples_in_guidance,
                eta=eta,
            )

            # When eta > 0, the scheduler injects noise, breaking consistency.
            # Enforce consistency after each step by picking the first sample.
            if (
                eta > 0.0
                and self.num_composed > 1
                and len(self.compose_sample_names) > 0
            ):
                samples = self._ensure_composed_consistency(samples, batch_size)

            # Capture intermediate samples after each denoising step
            if return_intermediates:
                intermediate_samples.append(
                    {
                        k: v.detach()
                        for k, v in self._decode_samples(
                            denoising_adapters, samples, clip_samples=False
                        ).items()
                    }
                )

        # Decode diffusion samples (final, with clipping)
        result = {
            k: v.detach()
            for k, v in self._decode_samples(
                denoising_adapters, samples, clip_samples=True
            ).items()
        }

        if return_intermediates:
            # Replace the last intermediate sample with the properly clipped final result
            # (the last intermediate was decoded with clip_samples=False)
            intermediate_samples[-1] = result
            return intermediate_samples

        return result


