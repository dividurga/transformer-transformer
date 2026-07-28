import os

import jax
from brax.training.agents.ppo import networks as ppo_networks
from dm_control import mjcf

from t2.env.mj_utils import (
    TARGET_SITE_GROUP,
    add_mocap_body_with_site,
    set_up_default_scene,
)
from t2.mjx_env.utils import render_policy
from t2.robogen.umi_on_legs_plus_plus import umi_on_legs_plus_plus

#  Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import hydra
import jax
from absl import logging

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)


@hydra.main(
    config_path="../config", config_name="run_rl_procedural", version_base="1.2"
)
def main(cfg):
    def mj_model_fn():
        mjcf_model = umi_on_legs_plus_plus(seed=2)
        physics = mjcf.Physics.from_mjcf_model(
            add_mocap_body_with_site(
                set_up_default_scene(mjcf_model, add_plane=True, plane_z_pos=0.0),
                name="target",
                site_group=TARGET_SITE_GROUP,
            )
        )
        return physics.model.ptr

    eval_env = hydra.utils.instantiate(cfg.env, mj_model_fn=mj_model_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(0)

    def random_gaussian_policy(x, rng=rng):
        policy_rng, rng = jax.random.split(rng)
        return jax.random.normal(policy_rng, (eval_env.action_size)) * 0.5, {}

    ppo_network = hydra.utils.instantiate(cfg.train.network_factory)(
        eval_env.observation_size,
        eval_env.action_size,
        lambda x, y: x,
    )

    make_policy = ppo_networks.make_inference_fn(ppo_network)

    policy_params = ppo_network.policy_network.init(rng)
    randomly_init_policy = make_policy((None, policy_params))

    render_policy(
        reset_fn=jit_reset,
        step_fn=jit_step,
        policy=random_gaussian_policy,
        env=eval_env,
        video_path="video.mp4",
        camera="body_cam",
    )


if __name__ == "__main__":
    main()
