import os
from typing import Any, List, MutableMapping, OrderedDict

import accelerate
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))


def limit_threads(n: int = 1):
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["BLOSC_NUM_THREADS"] = str(n)


def seed_everything(seed: int, workers: bool = True):
    accelerate.utils.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if workers:
        torch.cuda.manual_seed_all(seed)


def ensure_wandb_run_metadata(run) -> None:
    """Write files/wandb-metadata.json for offline wandb runs.

    scripts/plot_hardware_opt.py discovers runs by globbing for
    wandb-metadata.json, but WANDB_MODE=offline never writes that file.
    Recreate it with the fields the plotter reads (program, args, host, git
    commit) so offline evaluation runs stay plottable. No-op when the file
    already exists (i.e. online mode).
    """
    import json
    import socket
    import subprocess
    import sys

    metadata_path = os.path.join(run.dir, "wandb-metadata.json")
    if os.path.exists(metadata_path):
        return
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(sys.argv[0])) or None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        commit = ""
    os.makedirs(run.dir, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "program": sys.argv[0],
                "args": sys.argv[1:],
                "host": socket.gethostname(),
                "git": {"commit": commit, "remote": ""},
            },
            f,
            indent=2,
        )


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, List):
            if len(v) == 0:
                items.append((new_key, []))
            elif isinstance(v[0], MutableMapping):
                for idx in range(len(v)):
                    items.extend(
                        flatten_dict(v[idx], f"{new_key}/{idx}", sep=sep).items()
                    )
            else:
                for idx in range(len(v)):
                    items.append((f"{new_key}/{idx}", v[idx]))
        else:
            items.append((new_key, v))
    return dict(items)


def add_text_to_image(
    image: np.ndarray,
    texts: list[str],
    positions: list[tuple[int, int]],
    color="rgb(0, 0, 0)",
    fontsize=18,
):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default(size=fontsize)
    for text, pos in zip(texts, positions):
        draw.text(pos, text, fill=color, font=font, spacing=0.5)
    return np.array(pil_image)


def sorted_dict(d: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(sorted(d.items(), key=lambda x: x[0]))


def get_batch_device(batch: dict[str, torch.Tensor]) -> torch.device:
    """Return the device of the first tensor in ``batch``."""
    return next(iter(batch.values())).device


def get_batch_size(batch: dict[str, torch.Tensor]) -> int:
    """Return the batch size of the first tensor in ``batch``."""
    return next(iter(batch.values())).shape[0]
