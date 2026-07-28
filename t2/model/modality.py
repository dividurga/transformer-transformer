from typing import Callable, Optional, Union, cast

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from t2.data.normalization import Normalization
from t2.model.core import AdaLnLinear
from t2.utils.misc import get_batch_device, get_batch_size, sorted_dict

TensorFn = Union[Callable[[torch.Tensor], torch.Tensor], nn.Module]


class Modality(nn.Module):
    def __init__(
        self,
        name: str,
        attrs: dict[str, int],
        id_attrs: list[str],
        attr_norms: nn.ModuleDict,
        pad_value: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.name = name
        self.attrs = attrs
        self.id_attrs = id_attrs
        self.attr_norms = attr_norms
        self.pad_value = pad_value
        self.max_seq_len = max_seq_len

    @property
    def dim(self) -> int:
        return sum(self.attrs.values())

    def to_tensor(
        self,
        batch: dict[str, torch.Tensor],
        encoders: Optional[dict[str, TensorFn]] = None,
        normalize: bool = True,
        clip: Optional[bool] = None,
    ) -> torch.Tensor:
        # encode before normalize (opposite order from from_tensor)
        outs = []
        for field_name, dim in sorted_dict(self.attrs).items():
            tensor = batch[self.name + "/" + field_name]
            if encoders is not None and field_name in encoders:
                tensor = encoders[field_name](tensor)
            tensor = tensor.reshape(len(tensor), self.max_seq_len, -1)
            assert tensor.shape[-1] == dim, (
                f"{self.name}/{field_name} shape {tensor.shape} does not match dim {dim}"
            )
            attr_norm_name = f"/{field_name}"

            if normalize and attr_norm_name in self.attr_norms:
                normalizer = self.attr_norms[attr_norm_name]
                assert isinstance(normalizer, Normalization)
                tensor = normalizer.normalize(tensor, clip=clip)
            outs.append(tensor)

        return torch.cat(outs, dim=-1).float()

    def from_tensor(
        self,
        x: torch.Tensor,
        decoders: Optional[dict[str, TensorFn]] = None,
        unnormalize: bool = True,
        clip: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        # unnormalize before decode (opposite order from to_tensor)
        outs = {
            f"{self.name}/mask": torch.isclose(
                x,
                torch.tensor([self.pad_value], device=x.device, dtype=x.dtype),
                atol=1e-1,
                rtol=1e-1,
            ).all(dim=-1),
        }
        idx = 0
        for field_name, dim in sorted_dict(self.attrs).items():
            out_name = f"{self.name}/{field_name}"
            outs[out_name] = x[..., idx : idx + dim]
            idx += dim
            attr_norm_name = f"/{field_name}"
            if unnormalize and attr_norm_name in self.attr_norms:
                normalizer = self.attr_norms[attr_norm_name]
                assert isinstance(normalizer, Normalization)
                outs[out_name] = normalizer.unnormalize(outs[out_name], clip=clip)
            if decoders is not None and field_name in decoders:
                outs[out_name] = decoders[field_name](outs[out_name])
        return outs


class ModalityAdapter(nn.Module):
    """
    Handles preparing input and target for a modality
    to be used in the backbone
    """

    def __init__(
        self,
        modality: Modality,
        loss_weight: float,
        hidden_dim: int,
        attr_decoders: dict[str, nn.Module],
        attr_encoders: dict[str, nn.Module],
        only_use_mask_for_padding: bool,
        # loss fn shouldn't reduce
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        # if weight_loss_evenly_by_batch=True,loss from each batch should be normalized by seq len
        weight_loss_evenly_by_batch: bool = False,
        input_noise_std: float = 0.0,
    ):
        super().__init__()
        assert loss_fn(torch.zeros(42), torch.zeros(42)).shape == torch.Size([42]), (
            "loss function should not reduce"
        )
        self.modality = modality
        self.loss_weight = loss_weight
        self.attr_encoders = attr_encoders
        self.attr_decoders = attr_decoders
        self.in_proj = nn.Linear(modality.dim, hidden_dim)
        self.out_proj = (
            nn.Linear(hidden_dim, modality.dim)
            if self.loss_weight > 0.0
            else nn.Identity()
        )
        self.only_use_mask_for_padding = only_use_mask_for_padding
        self.loss_fn = loss_fn
        self.weight_loss_evenly_by_batch = weight_loss_evenly_by_batch
        self.input_noise_std = input_noise_std

    def prepare_tokens(
        self,
        batch: dict[str, torch.Tensor],
        pos_embs: dict[str, nn.Module] | nn.ModuleDict,
    ) -> dict[str, torch.Tensor]:
        return_dict = {}
        target_seq = self.modality.to_tensor(
            batch, encoders=self.attr_encoders, normalize=True
        )

        # Handle sequence masks
        mask = batch.get(self.modality.name + "/mask", None)
        if mask is not None:
            target_seq = torch.where(
                mask.expand_as(target_seq), self.modality.pad_value, target_seq
            )
            if self.only_use_mask_for_padding:
                return_dict["seq_mask"] = torch.zeros_like(mask)
            else:
                return_dict["seq_mask"] = mask

        # Project target and apply post-encoder position embeddings
        return_dict["target_seq"] = target_seq

        # Process input sequence
        input_seq = target_seq.clone()
        input_seq = input_seq + torch.randn_like(input_seq) * self.input_noise_std
        input_seq = self.in_proj(input_seq)
        for id_attr in sorted(self.modality.id_attrs):
            full_attr = self.modality.name + "/" + id_attr
            pos_ids = batch[full_attr]
            path = full_attr.split("/")
            assert path[-1] == "id"
            pos_emb_group = path[-2]
            input_seq = input_seq + pos_embs[pos_emb_group](pos_ids.long()).view(
                input_seq.shape
            )
        return_dict["input_seq"] = input_seq

        return return_dict

    def get_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["feat_seq"]
        target = batch["target_seq"]
        err = self.loss_fn(self.out_proj(features), target)
        mask = batch["seq_mask"]
        if mask is not None:
            if self.weight_loss_evenly_by_batch:
                err_zeroed = torch.where(
                    mask.expand_as(err), torch.zeros_like(err), err
                ).reshape(err.shape[0], -1)
                err_zeroed_sum = err_zeroed.sum(dim=-1)
                nonmasked_sum = (
                    (~mask).expand_as(err).reshape(err.shape[0], -1).sum(dim=-1)
                )
                err = err_zeroed_sum / nonmasked_sum
            else:
                err = err[~mask.expand_as(err)]
        return err.mean()


class CLSPooling(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str,
        batch_first: bool,
        in_dim: int,
        out_dim: int,
        cls_dropout: float,
    ):
        super().__init__()
        self.transformer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    batch_first=batch_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.cls = nn.Parameter(torch.randn(d_model))
        self.out_proj = nn.Linear(d_model, out_dim)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.drop = nn.Parameter(torch.randn(out_dim))
        self.drop_prob = cls_dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(x.shape) == 3
        x = self.in_proj(x)
        x = torch.cat([self.cls.unsqueeze(0).repeat(len(x), 1, 1), x], dim=1)
        if mask is not None:
            mask = torch.cat(
                [
                    torch.zeros(len(mask), 1, device=mask.device, dtype=torch.bool),
                    mask,
                ],
                dim=1,
            )
        for layer in self.transformer:
            x = layer(x, src_key_padding_mask=mask)
        x = self.out_proj(x[:, 0, :])
        # only drop out during training
        if self.training:
            should_drop = (
                torch.rand(x.size(0), device=x.device) < self.drop_prob
            ).unsqueeze(-1)

            drop_vec = self.drop.unsqueeze(0).expand_as(x)
            x = torch.where(should_drop, drop_vec, x)
        return x


class ConditioningModalityAdapter(ModalityAdapter):
    def __init__(self, pooling_encoder: nn.Module, **kwargs):
        kwargs["loss_weight"] = 0.0
        super().__init__(**kwargs)
        self.pooling_encoder = pooling_encoder

    def prepare_tokens(
        self,
        batch: dict[str, torch.Tensor],
        pos_embs: dict[str, nn.Module] | nn.ModuleDict,
    ) -> dict[str, torch.Tensor]:
        return_dict = super().prepare_tokens(batch, pos_embs)
        input_seq = return_dict.pop("input_seq")
        cond = self.pooling_encoder(input_seq, return_dict.get("seq_mask", None))
        return_dict["cond"] = cond
        if "seq_mask" in return_dict:
            return_dict.pop("seq_mask")
        return return_dict


class DiffusionModalityAdapter(ModalityAdapter):
    def __init__(
        self,
        attrs_noise_probs: dict[str, float],
        default_diffuse_prob: float,
        dont_noise_timesteps: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attrs_noise_probs = attrs_noise_probs
        self.default_diffuse_prob = default_diffuse_prob
        assert self.default_diffuse_prob in {0.0, 1.0}
        self.out_proj = (
            AdaLnLinear(in_dim=kwargs["hidden_dim"], out_dim=self.modality.dim)
            if self.loss_weight > 0.0
            else nn.Identity()
        )
        self.dont_noise_timesteps = dont_noise_timesteps

    def get_dont_noise_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = get_batch_size(batch)
        device = get_batch_device(batch)
        diffuse_probs = self.attrs_noise_probs

        if len(diffuse_probs) == 0:
            dont_noise_mask = (
                torch.rand(
                    (b, self.modality.max_seq_len, self.modality.dim),
                    device=device,
                )
                > self.default_diffuse_prob
            )
        else:
            dont_noise_mask = self.modality.to_tensor(
                batch,
                {
                    attr_name: lambda x, attr_name=attr_name: torch.rand(
                        (
                            self.attr_encoders[attr_name](x).shape
                            if attr_name in self.attr_encoders
                            else x.shape
                        ),
                        device=device,
                    )
                    > (
                        diffuse_probs.get(
                            attr_name,
                            self.default_diffuse_prob,
                        )
                    )
                    for attr_name in sorted(self.modality.attrs)
                },
                normalize=False,
            ).bool()  # to_tensor casts to float; the mask must stay boolean

        time_id_key = self.modality.name + "/time/id"
        if self.dont_noise_timesteps is not None and time_id_key in batch:
            time_data = batch[time_id_key]
            for timestep in self.dont_noise_timesteps:
                is_timestep = time_data == timestep
                is_timestep = is_timestep.expand_as(dont_noise_mask)
                dont_noise_mask[is_timestep] = True
        return dont_noise_mask

    def prepare_tokens(
        self,
        batch: dict[str, torch.Tensor],
        pos_embs: dict[str, nn.Module] | nn.ModuleDict,
    ) -> dict[str, torch.Tensor]:
        return_dict = {}
        target_seq = self.modality.to_tensor(
            batch, encoders=self.attr_encoders, normalize=True
        )

        mask = batch.get(self.modality.name + "/mask", None)
        if mask is not None:
            target_seq = torch.where(
                mask.expand_as(target_seq), self.modality.pad_value, target_seq
            )
            if self.only_use_mask_for_padding:
                return_dict["seq_mask"] = torch.zeros_like(mask)
            else:
                return_dict["seq_mask"] = mask

        timesteps = batch["timesteps"]
        noise_scheduler = cast(DDIMScheduler, batch["noise_scheduler"])
        # randomly drop noise for each attribute
        dont_noise = self.get_dont_noise_mask(batch)

        noise_seq = torch.randn_like(target_seq)
        input_seq = noise_scheduler.add_noise(
            original_samples=target_seq,
            noise=noise_seq,
            timesteps=timesteps,  # type: ignore
        )
        input_seq = torch.where(dont_noise, target_seq, input_seq)
        noise_seq = torch.where(dont_noise, torch.zeros_like(noise_seq), noise_seq)

        # Apply pre-encoder position embeddings to target
        input_seq = self.in_proj(input_seq)

        for id_attr in sorted(self.modality.id_attrs):
            full_attr = self.modality.name + "/" + id_attr
            pos_ids = batch[full_attr]
            path = full_attr.split("/")
            assert path[-1] == "id"
            pos_emb_group = path[-2]
            input_seq = input_seq + pos_embs[pos_emb_group](pos_ids.long()).view(
                input_seq.shape
            )

        return_dict["input_seq"] = input_seq
        return_dict["target_seq"] = noise_seq
        return return_dict

    def get_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat_seq = batch["feat_seq"]
        target_seq = batch["target_seq"]
        cond = batch["cond"]
        err = self.loss_fn(self.out_proj(x=feat_seq, cond=cond), target_seq)
        mask = batch["seq_mask"]
        if mask is not None:
            if self.weight_loss_evenly_by_batch:
                err_zeroed = torch.where(
                    mask.expand_as(err), torch.zeros_like(err), err
                ).reshape(err.shape[0], -1)
                err_zeroed_sum = err_zeroed.sum(dim=-1)
                nonmasked_sum = (
                    (~mask).expand_as(err).reshape(err.shape[0], -1).sum(dim=-1)
                )
                err = err_zeroed_sum / nonmasked_sum
            else:
                err = err[~mask.expand_as(err)]
        return err.mean()
