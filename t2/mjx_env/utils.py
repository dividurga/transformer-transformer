import textwrap
import time
from typing import Callable

import imageio
import jax
import jax.numpy as jnp
import mujoco
from tqdm import tqdm

from t2.env.mj_utils import TARGET_SITE_GROUP, TRACKING_SITE_GROUP
from t2.mjx_env.base_env import MjxEnv, State
from t2.utils.misc import add_text_to_image


def render(
    rollout: list[State],
    env: MjxEnv,
    camera: str | None = None,
    height: int = 224,
    width: int = 224,
    fontsize: int = 10,
):
    scene_option = mujoco.MjvOption()
    # render site frames
    scene_option.geomgroup[:4] = 1
    scene_option.sitegroup[:] = 0

    scene_option.sitegroup[TARGET_SITE_GROUP] = 1
    scene_option.sitegroup[TRACKING_SITE_GROUP] = 1
    scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE

    frames = []
    for state, frame in zip(
        tqdm(rollout, desc="Rendering", dynamic_ncols=True),
        env.render(
            [state.data for state in rollout],
            height=height,
            width=width,
            camera=camera,
            scene_option=scene_option,
        ),
    ):
        text = ""
        for k, v in state.metrics.items():
            if (
                k.startswith("done/")
                or (k.startswith("pos_err") and not k.endswith("pos_err"))
                or (k.startswith("orn_err") and not k.endswith("orn_err"))
                or k == {"termination", "reward"}
            ):
                continue
            text += f"{k}: {v:.2f} | "

        frame = add_text_to_image(
            frame,
            texts=[textwrap.fill(text, width=50, break_long_words=False)],
            positions=[(2, 2)],
            color="rgb(255, 255, 255)",
            fontsize=fontsize,
        )
        frames.append(frame)
    return frames


def render_policy(
    policy: Callable[[State], tuple[jnp.ndarray, jnp.ndarray]],
    step_fn: Callable[[State, jnp.ndarray], State],
    reset_fn: Callable[[jax.random.PRNGKey], State],
    env: MjxEnv,
    video_path: str,
    seed: int = 0,
    camera: str | None = None,
):
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)
    rollout = [state]
    pbar = tqdm(
        total=env.max_episode_length,
        desc="Rolling out",
        dynamic_ncols=True,
    )

    # TODO refactor into jax scan loop
    while not state.done:
        act_rng, rng = jax.random.split(rng)
        start_time = time.time()
        ctrl, _ = policy(state.obs, act_rng)
        end_time = time.time()
        if len(rollout) == 1:
            print(f"Policy Jit time: {float(end_time - start_time):.1f} seconds")
        start_time = time.time()
        state = step_fn(state, ctrl)
        end_time = time.time()
        if len(rollout) == 1:
            print(f"Step Jit time: {float(end_time - start_time):.1f} seconds\n")
        rollout.append(state)
        if state.done:
            break
        pbar.update(1)
    pbar.close()

    # Render and save the rollout
    with imageio.get_writer(video_path, mode="I", fps=1 / env.dt) as writer:
        frames = render(rollout, env, camera=camera)
        for frame in tqdm(frames, desc="Saving", dynamic_ncols=True):
            writer.append_data(frame)
    return video_path
