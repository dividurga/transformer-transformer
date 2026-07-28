import copy
from typing import Callable

import hydra
import numpy as np
import ray
import tqdm
from dm_control import mjcf
from omegaconf import OmegaConf

from t2.io.schema import append_hardware, init_root
from t2.robotok.io import serialize
from t2.robotok.tokenizer import preprocess_mjcf, tokenize
from t2.utils.ray import wait_with_pbar


def generate_robogen_token(
    choices: list[int],
    seed: int,
    num_uniforms: int,
    robogen_from_params: Callable[[list[int], list[float]], mjcf.RootElement],
):
    rs = np.random.RandomState(seed)
    uniforms = rs.uniform(0, 1, num_uniforms).tolist()
    mjcf_model = robogen_from_params(choices=copy.copy(choices), uniforms=uniforms)
    mjcf_model = preprocess_mjcf(
        mjcf_model,
    )
    tokenized_robot, _, _, _ = tokenize(
        mj_robot=mjcf_model,
    )  # type: ignore
    hardware_dict = serialize(tokenized_robot)
    hardware_dict["metadata/seed"] = np.array(seed)
    return hardware_dict


@hydra.main(
    config_path="../config",
    config_name="generate_robogen_tokens",
    version_base="1.3",
)
def main(cfg):
    hardware_seeds = list(range(cfg.num_hardware))
    ray.init(num_cpus=cfg.num_processes)

    @ray.remote(num_cpus=1)
    def generate_robogen_token_remote(
        choices: list[int],
        seeds: list[int],
        num_uniforms: int,
        robogen_from_params: Callable[[list[int], list[float]], mjcf.RootElement],
    ):
        data = []
        for seed in seeds:
            data.append(
                (
                    seed,
                    generate_robogen_token(
                        choices, seed, num_uniforms, robogen_from_params
                    ),
                )
            )
        return data

    robogen_fn = hydra.utils.instantiate(cfg.robogen_from_params)
    desc = f"robogen : {cfg.choices}"

    hardware_seed_groups = np.array_split(
        hardware_seeds, min(cfg.num_processes, len(hardware_seeds))
    )
    results = wait_with_pbar(
        {
            desc: [
                generate_robogen_token_remote.remote(
                    choices=cfg.choices,
                    seeds=seeds,
                    num_uniforms=cfg.num_uniforms,
                    robogen_from_params=robogen_fn,
                )
                for seeds in hardware_seed_groups
            ]
        }
    )[desc]

    root, store = init_root(
        cfg.output_path,
        root_metadata={
            "choices": OmegaConf.to_container(cfg.choices, resolve=True),
            "robogen_fn": OmegaConf.to_container(cfg.robogen_from_params),
            "num_uniforms": cfg.num_uniforms,
        },
    )
    flattened_results = [result for group in results for result in group]
    for hardware_seed, hardware_dict in tqdm.tqdm(
        flattened_results,
        desc="Appending hardware",
        dynamic_ncols=True,
    ):
        hardware_idx = append_hardware(
            hardware=hardware_dict,
            data_path=cfg.output_path,
        )
    store.close()


if __name__ == "__main__":
    main()
