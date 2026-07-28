import copy
from typing import Callable

import hydra
import numpy as np
import ray
import zarr
from dm_control import mjcf
from omegaconf import OmegaConf

from t2.env.mj_utils import add_mocap_body_with_site, set_up_default_scene
from t2.io.schema import init_root
from t2.utils.ray import wait_with_pbar


def generate_robogen_mjcf(
    choices: list[int],
    seed: int,
    num_uniforms: int,
    robogen_from_params: Callable[[list[int], list[float]], mjcf.RootElement],
    data_attrs: list[str],
):
    rs = np.random.RandomState(seed)
    uniforms = rs.uniform(0, 1, num_uniforms).tolist()
    mjcf_model = robogen_from_params(choices=copy.copy(choices), uniforms=uniforms)
    mjcf_model = add_mocap_body_with_site(
        set_up_default_scene(mjcf_model, add_plane=True, plane_z_pos=0.0),
        name="target",
    )
    mj_model = mjcf.Physics.from_mjcf_model(mjcf_model).model.ptr
    data = {}
    for attr in data_attrs:
        data[attr] = getattr(mj_model, attr)
    return data


@hydra.main(
    config_path="../config",
    config_name="generate_robogen_tokens",
    version_base="1.3",
)
def main(cfg):
    hardware_seeds = list(range(cfg.num_hardware))
    ray.init(num_cpus=cfg.num_processes)

    @ray.remote(num_cpus=1)
    def generate_robogen_mjcf_remote(
        choices: list[int],
        seeds: list[int],
        num_uniforms: int,
        robogen_from_params: Callable[[list[int], list[float]], mjcf.RootElement],
        data_attrs: list[str],
    ):
        data = []
        for seed in seeds:
            data.append(
                generate_robogen_mjcf(
                    choices, seed, num_uniforms, robogen_from_params, data_attrs
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
                generate_robogen_mjcf_remote.remote(
                    choices=cfg.choices,
                    seeds=seeds,
                    num_uniforms=cfg.num_uniforms,
                    robogen_from_params=robogen_fn,
                    data_attrs=cfg.data_attrs,
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
    for attr in cfg.data_attrs:
        data = np.stack([result[attr] for result in flattened_results], axis=0)
        root.create_array(
            attr,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(100, *list(data.shape[1:])),
            shards=(
                500,
                *list(data.shape[1:]),
            ),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=9),
        )
        root[attr][:] = data
    store.close()


if __name__ == "__main__":
    main()
