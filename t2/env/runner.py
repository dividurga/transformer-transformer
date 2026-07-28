import os
import time
from typing import Any, Callable, Optional

import imageio
import mujoco
import numpy as np
import tqdm
from mujoco import viewer

from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.mj_utils import TARGET_SITE_GROUP, TRACKING_SITE_GROUP, render_opt
from t2.io.schema import append_episode, append_hardware


class EnvRunner:
    """
    Environment runner that handles running a single environment.

    This class is responsible for:
    1. Executing episodes in the environment
    2. Collecting and storing data from environment runs
    3. Handling rendering and logging
    """

    def __init__(
        self,
        env: BaseEnv,
        log_dir: str | None = None,
        render: bool = False,
        render_fps: int = 20,
        use_gui: bool = False,
        default_policy: Optional[Callable[[ObsDict], dict[str, np.ndarray]]] = None,
        default_reset_fn: Optional[
            Callable[[BaseEnv, int], tuple[ObsDict, InfoDict, HardwareDict]]
        ] = None,
        dump_hardware_every_episode: bool = False,
    ):
        """
        Initialize the environment runner.

        Args:
            env: The environment to run
            log_dir: Directory to save logs and videos
            render: Whether to render and save videos
            render_fps: FPS for rendered videos
        """
        self.env = env
        self.log_dir = log_dir
        self.render = render
        self.render_fps = render_fps
        self.render_every_n_steps = int(np.round((1 / env.ctrl_dt) / self.render_fps))
        self.video_writer = None
        self.use_gui = use_gui
        self.__gui = None
        self.default_policy = default_policy
        self.default_reset_fn = default_reset_fn
        if self.default_reset_fn is None:
            self.default_reset_fn = lambda _env, _seed: _env.reset(seed=_seed)
        self.dump_hardware_every_episode = dump_hardware_every_episode
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        self.total_steps = 0
        self.step_pbar = None

    def post_hardware_reset(
        self,
        hardware_seed: int,
    ):
        if self.use_gui:
            if self.__gui is not None:
                if not self.__gui.is_running():
                    self.close()
                    exit()
                self.__gui.close()
            self.__gui = viewer.launch_passive(data=self.env.d, model=self.env.m)
            render_opt(self.__gui._opt)
            self.__gui._opt.geomgroup[3] = 1
            self.__gui._opt.sitegroup[:] = 0
            self.__gui._opt.sitegroup[TARGET_SITE_GROUP] = 1
            self.__gui._opt.sitegroup[TRACKING_SITE_GROUP] = 1
            self.__gui._opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self.env.p.forward()
            self.__gui.sync()

    def run_episode(
        self,
        seed: int,
        policy_fn: Optional[Callable[[ObsDict], dict[str, np.ndarray]]] = None,
        reset_fn: Optional[
            Callable[[BaseEnv, int], tuple[ObsDict, InfoDict, HardwareDict]]
        ] = None,
        max_steps: Optional[int] = None,
        render_path: str | None = None,
    ) -> tuple[dict[str, Any], HardwareDict]:
        """
        Run a single episode using the provided policy function.

        Args:
            policy_fn: Function that takes an observation and returns an action.
                If None, the runner's default policy is used if available; otherwise
                actions will be sampled randomly from the action space
            max_steps: Maximum number of steps to run for. If None, run until done
            render_path: Path to save the video if rendering

        Returns:
            Dictionary containing episode statistics
        """
        if policy_fn is None:
            policy_fn = self.default_policy

        if reset_fn is None:
            reset_fn = self.default_reset_fn

        assert reset_fn is not None

        # Reset environment
        obs, info, hardware = reset_fn(self.env, seed)
        if self.__gui is not None:
            if not self.__gui.is_running():
                self.close()
                exit()
            self.__gui.sync()

        # Setup rendering if needed
        if self.render:
            if render_path is None and self.log_dir is not None:
                render_path = os.path.join(
                    self.log_dir,
                    f"episode_h{self.env.hardware_seed:05d}_" + f"s{seed:05d}.mp4",
                )
            else:
                raise ValueError("`render_path` is required to render.")

            # Initialize video writer if we're rendering
            self.video_writer = imageio.get_writer(
                render_path, mode="I", fps=self.render_fps
            )

        done = False
        step_count = 0

        episode_data = {
            "ctrl/target_qpos": [],
            "ctrl/observed_qpos": [],
            "metric/reward": [],
        }

        # Run episode
        while True:
            # Render frame if needed
            if (
                self.video_writer is not None
                and step_count % self.render_every_n_steps == 0
            ):
                rgb = self.env.render()
                if rgb is not None:
                    self.video_writer.append_data(rgb)

            # Get action from policy or sample random action
            if policy_fn is not None:
                ctrls = policy_fn(obs)
                # this functionality support adding noise to the control
                target_qpos = ctrls["ctrl/target_qpos"]
                actual_qpos = (
                    target_qpos
                    if "ctrl/observed_qpos" not in ctrls
                    else ctrls["ctrl/observed_qpos"]
                )
            else:
                target_qpos = self.env.action_space.sample()
                actual_qpos = target_qpos

            # Store data
            for k, v in {**obs, **info}.items():
                if k not in episode_data:
                    episode_data[k] = []
                if not hasattr(v, "__len__"):
                    v = [[v]]
                episode_data[k].append(v)

            episode_data["ctrl/target_qpos"].append(target_qpos[..., None])
            episode_data["ctrl/observed_qpos"].append(actual_qpos[..., None])

            if max_steps is not None and step_count >= max_steps or done:
                episode_data["metric/reward"].append([[0.0]])
                break

            actual_qpos = actual_qpos[: len(self.env.d.ctrl)]

            # Execute action in environment
            next_obs, reward, done, info = self.env.step(actual_qpos)

            if self.__gui is not None:
                if not self.__gui.is_running():
                    self.close()
                    exit()
                self.__gui.sync()
                time.sleep(self.env.sim_dt * self.env.num_repeat_actions)

            episode_data["metric/reward"].append([[reward]])

            # Update for next iteration
            obs = next_obs
            step_count += 1
            self.total_steps += 1
            if self.step_pbar is not None:
                self.step_pbar.update(1)

        # Render final frame if needed
        if self.video_writer is not None:
            rgb = self.env.render()
            if rgb is not None:
                self.video_writer.append_data(rgb)
            self.video_writer.close()
            self.video_writer = None

        return (
            {k: np.array(v) for k, v in episode_data.items()},
            hardware,
        )

    def run_episodes(
        self,
        hardware_seed: int,
        episode_seeds: list[int],
        policy_fn: Callable[[ObsDict], dict[str, np.ndarray]] | None = None,
        hardware_reset_fn: Callable[[BaseEnv, int], None] | None = None,
        reset_fn: (
            Callable[[BaseEnv, int], tuple[ObsDict, InfoDict, HardwareDict]] | None
        ) = None,
        post_process_episode_data_fn: (
            Callable[
                [BaseEnv, dict[str, Any], HardwareDict],
                tuple[dict[str, Any], HardwareDict],
            ]
            | None
        ) = None,
        data_path: str | None = None,
        max_steps: int | None = None,
        use_pbar: bool = False,
    ):
        """
        Run multiple episodes.

        Args:
            policy_fn: Function that takes an observation and returns an action
            reset_fn: Function that resets the environment
            post_process_episode_data_fn: Function that post-processes the episode data
            max_steps: Maximum steps per episode
            hardware_seed: Seed for hardware randomization
            episode_seeds: List of seeds for episodes

        Returns:
            List of episode results

        Note:
            The same episode seed can be given multiple times
        """

        if hardware_reset_fn is None:

            def __hardware_reset_fn(_env, _hardware_seed):
                _env.reset_hardware(hardware_seed=_hardware_seed)

            hardware_reset_fn = __hardware_reset_fn

        hardware_reset_fn(self.env, hardware_seed)
        self.post_hardware_reset(hardware_seed=hardware_seed)

        hardware_idx = None

        if use_pbar:
            pbar = tqdm.tqdm(
                episode_seeds,
                desc="Running episodes",
                dynamic_ncols=True,
            )
            self.step_pbar = tqdm.tqdm(
                desc="Running steps",
                dynamic_ncols=True,
                total=(max_steps if max_steps is not None else self.env.episode_len),
            )
        else:
            pbar = episode_seeds

        for seed in pbar:
            # Run episode
            episode_data, hardware = self.run_episode(
                seed=seed,
                policy_fn=policy_fn,
                reset_fn=reset_fn,
                max_steps=max_steps,
            )
            if post_process_episode_data_fn is not None:
                episode_data, hardware = post_process_episode_data_fn(
                    self.env, episode_data, hardware
                )
            if data_path is not None:
                if hardware_idx is None:
                    hardware_idx = append_hardware(
                        hardware=hardware,
                        data_path=data_path,
                    )

                append_episode(
                    hardware_idx=hardware_idx,
                    episode_seed=seed,
                    episode_data=episode_data,
                    data_path=data_path,
                )
                if self.dump_hardware_every_episode:
                    hardware_idx = None

            if self.step_pbar is not None:
                self.step_pbar.reset()

    def close(self):
        """Clean up resources."""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if self.__gui is not None:
            self.__gui.close()
            self.__gui = None
        self.env.close()
