# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the Transformer Transformer authors.

import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training import acting
from brax.training.types import (
    Metrics,
    Policy,
    PolicyParams,
    PRNGKey,
    Transition,
)

State = envs.State
Env = envs.Env


class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[PolicyParams], Policy],
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        """Init.

        Args:
          eval_env: Batched environment to run evals on.
          eval_policy_fn: Function returning the policy from the policy parameters.
          num_eval_envs: Each env will run 1 episode in parallel for each eval.
          episode_length: Maximum length of an episode.
          action_repeat: Number of physics steps per env step.
          key: RNG key.
        """
        self._key = key
        self._eval_walltime = 0.0

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(
            policy_params: PolicyParams, key: PRNGKey
        ) -> tuple[State, Transition]:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return acting.generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                key,
                unroll_length=episode_length // action_repeat,
                extra_fields=["pos_err", "orn_err"],
            )

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state, transition = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()

        epoch_eval_time = time.time() - t

        not_done = transition.discount == 1
        orn_err = transition.extras["state_extras"]["orn_err"][not_done]
        pos_err = transition.extras["state_extras"]["pos_err"][not_done]
        metrics = {}

        for suffix, fn in [
            ("/mean", jnp.mean),
            ("/q100", lambda x: jnp.quantile(x, 1.0)),
            ("/q95", lambda x: jnp.quantile(x, 0.95)),
            ("/q50", lambda x: jnp.quantile(x, 0.75)),
            ("/q50", lambda x: jnp.quantile(x, 0.5)),
            ("/q25", lambda x: jnp.quantile(x, 0.25)),
            ("/q05", lambda x: jnp.quantile(x, 0.05)),
            ("/q00", lambda x: jnp.quantile(x, 0.00)),
        ]:
            metrics.update(
                {
                    f"eval/summary/pos_err{suffix}": fn(pos_err),
                    f"eval/summary/orn_err{suffix}": fn(orn_err),
                }
            )
        for suffix, fn in [
            ("/mean", np.mean),
            ("/std", np.std),
        ]:
            metrics.update(
                {
                    f"eval/episode/{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/std_episode_length"] = np.std(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            "eval/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray
