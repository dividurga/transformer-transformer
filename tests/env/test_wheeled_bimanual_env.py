"""
Test script to verify that TrackEnv works with the wheeled bimanual robot
which has 2 end effectors.
"""

import os

import hydra
import pytest

from t2.eval.utils import summarize_rollout


@pytest.mark.skipif(
    not os.path.exists("data/bimanual_dish_washing.pkl"),
    reason="requires the released UMI dishwashing trajectories (see docs/starter.md)",
)
def test_wheeled_bimanual_env():
    """Test that the wheeled bimanual environment can be instantiated and run."""
    with hydra.initialize(config_path="../../config", version_base=None):
        cfg = hydra.compose(config_name="datagen_wheeled_bimanual")

    cfg.runner.env.pickle_path = "data/bimanual_dish_washing.pkl"
    runner = hydra.utils.instantiate(
        cfg.runner, use_gui=False, render=True, log_dir="renders"
    )
    # with tempfile.TemporaryDirectory(suffix=".zarr") as temp_dir:
    temp_dir = "test_bimanual_env.zarr"
    runner.run_episodes(hardware_seed=0, episode_seeds=[0], data_path=temp_dir)
    runner.close()
    summary_stats = summarize_rollout(temp_dir)
    print("rewards", summary_stats["metric/reward/sum"])


if __name__ == "__main__":
    test_wheeled_bimanual_env()
