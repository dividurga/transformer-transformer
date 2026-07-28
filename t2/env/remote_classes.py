import tempfile
import typing
from typing import Optional

import numpy as np
import ray
from ray.util.queue import Queue

from t2.env.base_env import BaseEnv, ObsDict
from t2.env.mink_runner import MinkEnvRunner
from t2.env.bimanual_mink_runner import BimanualMinkEnvRunner
from t2.env.wheeled_bimanual_mink_runner import WheeledBimanualMinkEnvRunner
from t2.env.runner import EnvRunner
from t2.env.track_env import TrackEnv


@ray.remote
class RemoteTrackEnvRunner:
    def __init__(self, runner: EnvRunner, action_horizon: int):
        self.__runner = runner
        self.action_horizon = action_horizon
        assert action_horizon == 1
        assert isinstance(runner.env, TrackEnv)

    def run_episode(
        self,
        hardware_seed: int,
        traj_idx: int,
        obs_queue: Queue,
        max_steps: Optional[int] = None,
    ):
        prev_action = []

        def reset_to_traj_and_seed(_env: BaseEnv, seed: int, _traj_idx: int = traj_idx):
            track_env = typing.cast(TrackEnv, _env)

            prev_action.clear()
            if type(self.__runner) is MinkEnvRunner:
                raise NotImplementedError("Handle qpos to ctrl mapping for mink runner")
                track_env.traj_idx = _traj_idx
                track_env.reset_no_obs(seed=seed)
                track_env.p.forward()
                mink_env_runner = typing.cast(MinkEnvRunner, self.__runner)
                if mink_env_runner.randomize_default_posture:
                    target_q = _env.rs.uniform(
                        _env.m.jnt_range[:, 0], _env.m.jnt_range[:, 1]
                    )
                else:
                    target_q = _env.d.qpos[:]
                mink_env_runner.posture_task.set_target(target_q)
                mink_result = mink_env_runner.mink_policy(
                    obs={}, mink_max_iters=mink_env_runner.mink_reset_max_iters
                )
                track_env.d.qpos[:] = mink_result["ctrl/target_qpos"]
                track_env.d.ctrl[:] = mink_result["ctrl/target_qpos"]
                track_env.p.forward()
                retval = (
                    track_env.get_obs(),
                    track_env.get_info(),
                    track_env.hardware_dict,
                )
                prev_action.extend(mink_result["ctrl/target_qpos"])
            elif type(self.__runner) is WheeledBimanualMinkEnvRunner:
                runner = typing.cast(WheeledBimanualMinkEnvRunner, self.__runner)
                track_env.traj_idx = _traj_idx
                track_env.reset_no_obs(seed=seed)
                if runner.randomize_default_posture:
                    target_qpos = track_env.rs.uniform(
                        track_env.m.jnt_range[:, 0], track_env.m.jnt_range[:, 1]
                    )
                else:
                    target_qpos = track_env.d.qpos[:]
                runner.mink_tasks["posture"].set_target(target_qpos)
                original_posture_cost = runner.mink_tasks["posture"].cost.copy()
                reset_posture_cost = original_posture_cost.copy()
                reset_posture_cost[:3] *= runner.reset_base_posture_cost_multiplier
                reset_posture_cost[3:] *= runner.reset_body_posture_cost_multiplier
                runner.mink_tasks["posture"].set_cost(reset_posture_cost)
                mink_result = runner.mink_policy(
                    obs={}, mink_max_iters=runner.mink_reset_max_iters
                )
                runner.mink_tasks["posture"].set_cost(original_posture_cost)
                track_env.d.qpos[list(runner.qpos_to_ctrl_map.keys())] = mink_result[
                    "ctrl/target_qpos"
                ][list(runner.qpos_to_ctrl_map.values())]
                track_env.d.ctrl[:] = mink_result["ctrl/target_qpos"]
                track_env.p.forward()
                obs = track_env.get_obs()
                info = track_env.get_info()
                hardware = track_env.hardware_dict
                retval = obs, info, hardware
                prev_action.extend(mink_result["ctrl/target_qpos"])
            elif type(self.__runner) is BimanualMinkEnvRunner:
                runner = typing.cast(BimanualMinkEnvRunner, self.__runner)
                track_env.traj_idx = _traj_idx
                track_env.reset_no_obs(seed=seed)
                if runner.randomize_default_posture:
                    target_qpos = track_env.rs.uniform(
                        track_env.m.jnt_range[:, 0], track_env.m.jnt_range[:, 1]
                    )
                else:
                    target_qpos = track_env.d.qpos[:]
                runner.posture_task.set_target(target_qpos)
                mink_result = runner.mink_policy(
                    obs={}, mink_max_iters=runner.mink_reset_max_iters
                )
                track_env.d.qpos[:] = mink_result["ctrl/target_qpos"]
                track_env.d.ctrl[:] = mink_result["ctrl/target_qpos"]
                track_env.p.forward()
                retval = (
                    track_env.get_obs(),
                    track_env.get_info(),
                    track_env.hardware_dict,
                )
                prev_action.extend(mink_result["ctrl/target_qpos"])
            else:
                retval = track_env.reset_sim_and_target_traj(
                    seed=seed, target_traj_idx=_traj_idx
                )
                prev_action.extend([0.0] * self.__runner.env.d.ctrl.shape[0])
            return retval

        action_queue = Queue()

        def policy_fn(obs: ObsDict):
            track_env = typing.cast(TrackEnv, self.__runner.env)
            # send off obs with a return queue
            reshaped_obs = {}
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
            obs_queue.put(
                (
                    reshaped_obs,
                    np.array(prev_action),
                    self.__runner.env.hardware_dict,
                    action_queue,
                )
            )
            action = action_queue.get()
            # TODO: support action_horizon > 1
            assert self.action_horizon == 1
            action = action[0]
            prev_action.clear()
            prev_action.extend(action.tolist())
            return {"ctrl/target_qpos": action}

        data_path = tempfile.mkdtemp()
        self.__runner.run_episodes(
            hardware_seed=hardware_seed,
            episode_seeds=[traj_idx],
            policy_fn=policy_fn,
            reset_fn=reset_to_traj_and_seed,  # type: ignore
            max_steps=max_steps,
            data_path=data_path,
        )
        return data_path
