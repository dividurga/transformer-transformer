import os

import torch


def set_torch_configs(disable_compilation_logs: bool = True):
    # the default is 8, which is too small for our modalities
    torch._dynamo.config.recompile_limit = 128
    torch.set_float32_matmul_precision("high")
    if disable_compilation_logs:
        os.environ["TORCH_LOGS"] = "-inductor"
