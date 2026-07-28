import logging
from typing import Optional

import torch

RANGE_RATIO_THRESHOLD = 50  # if the ratio of the full range to the reduced range is greater than this threshold, use the reduced range


class Normalization(torch.nn.Module):
    EPS = 1e-4

    def __init__(
        self,
        quantiles: dict[
            str, torch.Tensor
        ],  # e.g. `q0/0`, `q05/0`, ..., `q95/0`, `q100/0`
        min_key: str = "q0/0",
        max_key: str = "q100/0",
        clip: bool = False,
    ):
        super().__init__()
        quantiles = {
            k: torch.nn.Parameter(v, requires_grad=False) for k, v in quantiles.items()
        }
        self.quantiles = torch.nn.ParameterDict(quantiles)

        # the following logic is a heuristic to determine whether
        # the normalization range likely contain outlier values
        # if so, then we should use the reduced range to avoid
        # normalizing with respect to outlier values
        if set(
            {
                "q0/0",
                "q1/0",
                "q99/0",
                "q100/0",
            }
        ).issubset(set(quantiles.keys())):
            full_range = torch.abs(quantiles["q100/0"] - quantiles["q0/0"]).max()

            reduced_range = torch.maximum(
                torch.abs(quantiles["q99/0"] - quantiles["q1/0"]).max(),
                torch.tensor(self.EPS),
            )
            ratio = full_range / reduced_range
            if ratio > RANGE_RATIO_THRESHOLD:
                unique_values = torch.unique(torch.cat(list(quantiles.values())))
                # sometimes, one attribute is binary, either 0 or 1000. In these cases,
                # we're fine using the full range
                if len(unique_values) != 2:
                    logging.warning(
                        f"Normalization range is too large: {ratio}, using reduced range."
                    )
                    min_key = "q1/0"
                    max_key = "q99/0"

        self.vmin = self.quantiles[min_key]
        self.vmax = self.quantiles[max_key]
        self.clip = clip

    def normalize(self, x: torch.Tensor, clip: Optional[bool] = None) -> torch.Tensor:
        vrange = self.vmax - self.vmin
        vrange = torch.max(vrange, torch.tensor(self.EPS))
        x = (x - self.vmin) / vrange
        if clip is None:
            clip = self.clip
        if clip:
            x = torch.clamp(x, 0, 1)
        return (x - 0.5) * 2

    def unnormalize(self, x: torch.Tensor, clip: Optional[bool] = None) -> torch.Tensor:
        vrange = self.vmax - self.vmin
        vrange = torch.max(vrange, torch.tensor(self.EPS))
        if clip is None:
            clip = self.clip
        if clip:
            x = torch.clamp(x, -1, 1)
        x = (x + 1) / 2
        return x * vrange + self.vmin
