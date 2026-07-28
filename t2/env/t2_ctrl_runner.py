import os
import pickle
import typing
from typing import Any, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from t2.env.base_env import BaseEnv, HardwareDict, InfoDict, ObsDict
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv
from t2.eval.ctrl import env_obs_to_batch
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId


def _to_resolved_dict(obj: Any) -> dict[str, Any]:
    if OmegaConf.is_config(obj):
        out = OmegaConf.to_container(obj, resolve=True)  # type: ignore[arg-type]
        assert isinstance(out, dict)
        return out
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected dict or OmegaConf config, got {type(obj)}")


class T2CtrlEnvRunner(EnvRunner):
    """
    EnvRunner that runs a trained T2 control policy (task: ctrl) online.

    This mirrors the preprocessing/postprocessing used by `t2.eval.ctrl.policy_server`
    but for a single environment instance (no Ray policy server).
    """

    def __init__(
        self,
        ckpt_path: str,
        task_name: str = "ctrl",
        device: str = "cuda",
        qpos_noise: float = 0.0,
        num_inference_steps: Optional[int] = None,
        num_repeats_per_step: int = 1,
        clip_samples_in_guidance: bool = True,
        deterministic: bool = False,
        use_flash_attn: bool = True,
        use_torch_compile: bool = True,
        use_mixed_precision: bool = True,
        eta: float = 0.0,
        group_time_offsets: Optional[dict[str, int]] = None,
        **kwargs,
    ):
        self.ckpt_path = os.path.abspath(ckpt_path)
        self.task_name = task_name
        self.device = torch.device(device)
        self.qpos_noise = float(qpos_noise)

        policy_cfg_path = os.path.join(os.path.dirname(self.ckpt_path), "cfg.pkl")
        policy_cfg = pickle.load(open(policy_cfg_path, "rb"))

        # ctrl_seq_len is often expressed via resolvers; resolve before use.
        try:
            OmegaConf.resolve(policy_cfg.ctrl_seq_len)
        except Exception:
            # Some older configs may not need explicit resolution.
            pass

        # Used by `env_obs_to_batch` and by TrackEnv for choosing target pose horizon.
        self.seq_len_cfg = typing.cast(
            dict[str, int], _to_resolved_dict(policy_cfg.ctrl_seq_len)
        )

        default_group_time_offsets = _to_resolved_dict(
            policy_cfg.datasets.ctrl_rand.dataset.group_time_offsets
        )
        self.group_time_offsets: dict[str, int] = {
            **default_group_time_offsets,
            **({} if group_time_offsets is None else group_time_offsets),
        }

        batch_process_fn = hydra.utils.instantiate(
            policy_cfg.datasets.ctrl_rand.batch_process_fn
        )
        # At inference time we only want position ids
        add_pos_id = next(
            aug for aug in batch_process_fn.augmentations if type(aug) is AddPositionId
        )
        self.batch_process_fn = add_pos_id

        self.decoder = setup_decoder(
            policy_cfg,
            ckpt_path=self.ckpt_path,
            device=self.device,
            task_name=self.task_name,
            num_inference_steps=num_inference_steps,
            num_repeats_per_step=num_repeats_per_step,
            clip_samples_in_guidance=clip_samples_in_guidance,
            deterministic=deterministic,
            use_flash_attn=use_flash_attn,
            use_torch_compile=use_torch_compile,
            use_mixed_precision=use_mixed_precision,
            eta=eta,
        )

        # Updated on every control step.
        self._prev_action: np.ndarray | None = None

        reset_fn = (
            kwargs["default_reset_fn"]
            if kwargs.get("default_reset_fn", None) is not None
            else (lambda _env, _seed: _env.reset(seed=_seed))
        )
        assert reset_fn is not None

        def augmented_reset_fn(
            _env: BaseEnv, _seed: int
        ) -> tuple[ObsDict, InfoDict, HardwareDict]:
            obs, info, hardware = reset_fn(_env, _seed)
            # Initialize prev_action for this hardware.
            self._prev_action = np.zeros((_env.d.ctrl.shape[0],), dtype=np.float32)
            return obs, info, hardware

        super().__init__(
            default_policy=self.run_t2_ctrl_policy,
            **{**kwargs, "default_reset_fn": augmented_reset_fn},
        )

        rollout_steps = int(self.seq_len_cfg["rollout_steps"])
        target_pose_offset = int(self.group_time_offsets.get("target_pose", 0))
        self.obs_time_indices = np.arange(rollout_steps) + target_pose_offset

    @property
    def tracking_env(self) -> TrackEnv:
        assert isinstance(self.env, TrackEnv)
        return self.env

    def post_hardware_reset(self, hardware_seed: int):
        super().post_hardware_reset(hardware_seed=hardware_seed)

        # Initialize prev_action for this hardware.
        self._prev_action = np.zeros((self.env.d.ctrl.shape[0],), dtype=np.float32)

    def _obs_to_model_input(self, obs: ObsDict) -> ObsDict:
        track_env = typing.cast(TrackEnv, self.env)
        # Match the shape conventions used in `RemoteTrackEnvRunner.policy_fn`.
        reshaped_obs: ObsDict = {}
        for k, v in obs.items():
            group = k.split("/")[0]
            if v.ndim == 2:
                if group == "target_pose":
                    # target pose observations are sampled
                    # as trajectories of end effectors
                    # all other observation terms only include
                    # the current timestep for now
                    v = v.reshape(-1, track_env.n_end_effectors, v.shape[-1])
                else:
                    v = v[None, :, :]
            reshaped_obs[k] = v
        return reshaped_obs

    def run_t2_ctrl_policy(self, obs: ObsDict) -> dict[str, np.ndarray]:
        assert self._prev_action is not None, "prev_action not initialized (reset?)"
        assert isinstance(self.env, TrackEnv), "T2CtrlEnvRunner requires TrackEnv"
        task_obs = self.tracking_env.sample_target_poses(
            self.obs_time_indices + self.env.episode_step,
            include_obs_time_indices="relative",
        )

        batch = env_obs_to_batch(
            obs_list=[
                {
                    **self._obs_to_model_input({**obs, **task_obs}),
                    **self.env.hardware_dict,
                }
            ],
            prev_action_list=[self._prev_action.copy()],
            seq_len_cfg=self.seq_len_cfg,
            device=self.device,
            batch_process_fn=self.batch_process_fn,
            group_time_offsets=self.group_time_offsets,
        )

        with torch.inference_mode():
            decode_dict = self.decoder(batch)

        rollout_steps = int(self.seq_len_cfg["rollout_steps"])
        max_actuators = int(self.seq_len_cfg["actuator"])

        actions = (
            decode_dict["ctrl/target_qpos"]
            .detach()
            .cpu()
            .numpy()
            .reshape(rollout_steps, max_actuators)
        )
        action_masks = (
            batch["ctrl/mask"]
            .detach()
            .cpu()
            .numpy()
            .reshape(rollout_steps, max_actuators)
        )

        action_time_offset = int(self.group_time_offsets.get("ctrl", 0))
        assert action_time_offset <= 0, "action time offset should be non-positive"
        num_past_actions = -action_time_offset
        actions = actions[num_past_actions:]
        action_masks = action_masks[num_past_actions:]

        # Use current-step action and drop padded actuator slots.
        curr_action = actions[0]
        curr_mask = action_masks[0]
        ctrl = curr_action[~curr_mask].astype(np.float32)

        noisy_ctrl = ctrl.copy()
        if self.qpos_noise > 0:
            noisy_ctrl += (
                self.env.rs.randn(*noisy_ctrl.shape).astype(np.float32)
                * self.qpos_noise
            )

        # Update prev_action for next step conditioning.
        self._prev_action = ctrl.copy()

        return {
            "ctrl/target_qpos": ctrl,
            "ctrl/observed_qpos": noisy_ctrl,
        }
