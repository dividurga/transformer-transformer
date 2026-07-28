"""Visualize hardware diffusion denoising process.

This script renders the intermediate states of the diffusion model as it denoises
from random noise to a coherent robot structure. It captures the progressive
refinement of robot morphology through the diffusion process.

Outputs:
    - Video (.mp4) showing the denoising progression
    - Pickled robot dicts at each diffusion timestep
    - Pickled physics states for each frame
    - Combined state visualization (optional, for multi-frame renders)

Usage:
    python scripts/visualize_robotoken_diffusion.py \\
        ckpt_path=/path/to/checkpoint.pt \\
        pickle_path=/path/to/trajectories.pkl \\
        render=true

Configuration:
    See config/hardware_gen_vis.yaml for all available options.
"""

# =============================================================================
# SECTION: Imports
# =============================================================================

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import hydra
import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import tqdm
from dm_control import mjcf
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from transforms3d import quaternions

from t2.data.dataset import TimestepSampler
from t2.env.mj_utils import render_opt, set_up_default_scene
from t2.eval.diffusion_guidance import HardwareOptimizer
from t2.model.t2 import setup_decoder
from t2.robotok.io import deserialize
from t2.robotok.tokenizer import detokenize
from t2.train.augment import AddPositionId, ComposeAugmentation
from t2.utils.misc import add_text_to_image, seed_everything

# =============================================================================
# SECTION: Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION: Visualization Hardware Optimizer
# =============================================================================


class VisualizationHardwareOptimizer(HardwareOptimizer):
    """HardwareOptimizer subclass that returns intermediate diffusion samples.

    This is used for visualization to capture the denoising progression.
    Warmup is always skipped since timing accuracy isn't needed for visualization.

    The key differences from the base class:
    - `run_hardware_generator` calls decoder with `return_intermediates=True`
    - `_run_optimization_core` handles list of dicts and returns all intermediates
    - Cache stores list[dict] instead of single dict
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Always skip warmup for visualization - timing accuracy doesn't matter
        self._ran_warmup = True

    def run_hardware_generator(
        self, batch: dict[str, torch.Tensor], seed: int
    ) -> list[dict[str, torch.Tensor]]:
        """Run hardware generator and return all intermediate samples.

        Args:
            batch: Input batch with conditioning information.
            seed: Random seed for reproducibility.

        Returns:
            List of decoded dicts, one per denoising step, each merged with batch.
        """
        with torch.inference_mode():
            # Call decoder - return_intermediates=True was set in setup_decoder()
            # so the decoder already returns intermediates by default
            decoded_list = self.hardware_generator(batch, seed=seed)
        # Merge batch into each intermediate
        return [{**batch, **d} for d in decoded_list]

    def _run_optimization_core(
        self, trajs: list[NDArray[np.float32]], seed: int
    ) -> tuple[list[dict[str, torch.Tensor]], float]:
        """Run optimization returning list of intermediate samples.

        Args:
            trajs: List of trajectory arrays to optimize for.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (list of intermediate hardware dicts, highest reward value).
        """
        assert self.num_seeds % self.batch_size == 0
        num_iters = int(self.num_seeds // self.batch_size)
        highest_value = -float("inf")
        best_intermediates: list[dict[str, torch.Tensor]] | None = None

        # Prepare batch from trajectories
        batch = self._prepare_batch(trajs)
        num_trajs = len(trajs)

        # Repeat batch for batch_size samples per trajectory
        batch = {
            k: v.repeat(self.batch_size, *([1] * (v.ndim - 1)))
            for k, v in batch.items()
        }

        rs = np.random.RandomState(seed)
        for _ in range(num_iters):
            intermediates = self.run_hardware_generator(
                batch, seed=rs.randint(0, np.iinfo(np.int32).max)
            )
            # Use final sample for reward calculation
            decoded = intermediates[-1]

            # Compute reward (may be empty for visualization-only use)
            if self.reward_fns:
                value = sum(
                    reward_fn(decoded) for reward_fn in self.reward_fns.values()
                )
                assert value.ndim == 1 and value.shape[0] == self.batch_size * num_trajs
                value = value.reshape(self.batch_size, num_trajs)
                # Set the average value over all trajectories to each seed
                value[:] = value.mean(dim=1, keepdim=True)
                value = value.reshape(self.batch_size * num_trajs)
                best_seed_idx = value.argmax(dim=0)
                curr_highest_value = value[best_seed_idx].item()
            else:
                # No reward functions - just use first seed (index 0)
                best_seed_idx = 0
                curr_highest_value = 0.0

            if curr_highest_value > highest_value or best_intermediates is None:
                highest_value = curr_highest_value
                # Extract best seed from all intermediates
                best_intermediates = [
                    {k: v[best_seed_idx] for k, v in d.items()} for d in intermediates
                ]

        assert best_intermediates is not None
        return best_intermediates, highest_value

    def _load_from_cache(
        self, cache_key: str
    ) -> tuple[list[dict[str, torch.Tensor]], float, float] | None:
        """Try to load results from cache.

        Args:
            cache_key: The cache key to look up.

        Returns:
            Tuple of (list of decoded dicts, predicted_value, optimize_time) if found,
            None otherwise.
        """
        if self.output_cache_dir is None:
            return None
        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None
        try:
            cached = torch.load(
                cache_path, map_location=self.device, weights_only=False
            )
            logging.info(f"Cache hit: {cache_key[:16]}...")
            return (
                cached["decoded"],  # list[dict] for visualization
                cached["predicted_value"],
                cached["optimize_time"],
            )
        except Exception as e:
            logging.warning(f"Failed to load cache {cache_path}: {e}")
            return None

    def _save_to_cache(
        self,
        cache_key: str,
        decoded: list[dict[str, torch.Tensor]],
        predicted_value: float,
        optimize_time: float,
    ) -> None:
        """Save results to cache.

        Args:
            cache_key: The cache key.
            decoded: List of decoded hardware dicts (intermediates).
            predicted_value: The predicted reward value.
            optimize_time: Time taken for optimization.
        """
        if self.output_cache_dir is None:
            return
        cache_path = self._get_cache_path(cache_key)
        try:
            torch.save(
                {
                    "decoded": decoded,
                    "predicted_value": predicted_value,
                    "optimize_time": optimize_time,
                },
                cache_path,
            )
            logging.debug(f"Saved to cache: {cache_key[:16]}...")
        except Exception as e:
            logging.warning(f"Failed to save cache {cache_path}: {e}")

    def optimize(
        self, trajs: list[NDArray[np.float32]], seed: int = 0
    ) -> tuple[list[dict[str, torch.Tensor]], float, float]:
        """Optimize hardware for the given trajectories, returning intermediates.

        Args:
            trajs: List of trajectory arrays to optimize for.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (list of intermediate hardware dicts, highest reward value,
            optimization time in seconds).

        Raises:
            RuntimeError: If in cache-only mode and the result is not cached.
        """
        import time

        # Check cache first
        cache_key = self._compute_cache_key(trajs, seed)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # If in cache-only mode and we get here, it's a cache miss - raise error
        if self._cache_only_mode:
            raise RuntimeError(
                f"Cache miss in cache-only mode for cache_key={cache_key[:16]}... "
                f"(seed={seed}). The model has been unloaded and cannot run inference. "
                "Ensure all inputs were pre-computed before enabling cache-only mode."
            )

        # Note: warmup is skipped for visualization (self._ran_warmup = True in __init__)

        # Run the actual optimization with timing
        start_time = time.time()
        best_intermediates, highest_value = self._run_optimization_core(trajs, seed)
        optimize_time = float(time.time() - start_time)

        # Save to cache
        self._save_to_cache(cache_key, best_intermediates, highest_value, optimize_time)

        return best_intermediates, highest_value, optimize_time


# =============================================================================
# SECTION: Robot Processing
# =============================================================================


def post_process_robot_dict(
    robot_dict: dict[str, torch.Tensor],
    last_robot_dict: dict[str, torch.Tensor],
    override_keys: list[str],
) -> dict[str, np.ndarray]:
    """Post-process a robot dict by applying overrides and removing masked entries.

    For intermediate diffusion steps, certain keys (like discrete IDs and
    categorical values) should be copied from the final robot dict to ensure
    consistency. This function also removes masked (invalid) entries.

    Args:
        robot_dict: Robot dictionary from a diffusion timestep (already unbatched).
        last_robot_dict: Robot dictionary from the final diffusion timestep (already unbatched).
        override_keys: List of keys to copy from last_robot_dict.

    Returns:
        Processed robot dict with numpy arrays, masked entries removed.
    """
    # Convert to numpy - tensors are already unbatched by the optimizer
    robot_dict_cpu = {k: v.cpu().numpy() for k, v in robot_dict.items()}

    # Override specified keys with values from final timestep
    for key in override_keys:
        if key not in last_robot_dict:
            continue
        robot_dict_cpu[key] = last_robot_dict[key].cpu().numpy()

    # Remove masked predictions
    masked_robot_dict = {}
    for k, v in robot_dict_cpu.items():
        if k.endswith("/mask"):
            continue
        group = k.split("/")[0]
        mask_key = group + "/mask"
        mask = last_robot_dict[mask_key].reshape(-1).cpu().numpy().astype(bool)
        masked_robot_dict[k] = v[~mask]

    # Ensure valid geom types and sizes
    masked_robot_dict["link/geom_type"] = masked_robot_dict["link/geom_type"] % 4
    masked_robot_dict["link/geom_size"] = np.maximum(
        masked_robot_dict["link/geom_size"], 0.0001
    )

    return masked_robot_dict


def build_anchor_positions(
    robot_dict: dict[str, np.ndarray],
    gravcomp: bool,
) -> np.ndarray:
    """Build anchor positions from the final robot state.

    Creates a physics simulation from the robot dict and extracts geom
    positions to use as anchors for distance-based colorization.

    Args:
        robot_dict: Processed robot dictionary.
        gravcomp: Whether to enable gravity compensation.

    Returns:
        Array of shape (num_geoms, 3) with world-space geom positions.

    Raises:
        ValueError: If physics simulation fails to build.
    """
    tokenized_robot = deserialize(robot_dict, include_states=True)
    mjcf_model, _, _, _ = detokenize(tokenized_robot, gravcomp=gravcomp)
    mjcf_model = set_up_default_scene(
        mjcf_model, add_plane=True, add_vis_cam=True, plane_z_pos=0.0
    )
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)
    physics.reset(0)

    if physics is None:
        raise ValueError("Failed to build physics for anchor positions.")

    return np.array(
        [
            np.array(physics.data.geom_xpos[geom_id])
            for geom_id in range(physics.model.ngeom)
        ]
    )


# =============================================================================
# SECTION: Physics State Parsing
# =============================================================================


def parse_physics_state(
    physics,
    anchor_positions: Optional[np.ndarray] = None,
    max_distance: float = 2.0,
    colormap_name: str = "jet",
    prev_data_dict: Optional[dict[str, list]] = None,
    use_original_rgba: bool = True,
) -> dict[str, list]:
    """Parse physics state into a dictionary of geom properties.

    Extracts geom types, sizes, colors, positions, and orientations from
    a MuJoCo physics state. Optionally colors geoms based on distance from
    anchor positions.

    Args:
        physics: dm_control Physics object.
        anchor_positions: Reference positions for distance-based coloring.
            If None, uses current positions.
        max_distance: Maximum distance for colormap normalization.
        colormap_name: Matplotlib colormap name for distance coloring.
        prev_data_dict: Previous frame's data for quaternion sign consistency.
        use_original_rgba: If True, use model's original colors. If False,
            use distance-based colormap.

    Returns:
        Dictionary with keys:
        - link/geom_type: List of geom type integers
        - link/geom_size: List of (3,) size tuples
        - link/rgba: List of (4,) RGBA tuples
        - link/pos: List of (3,) position tuples
        - link/quat_wxyz: List of (4,) quaternion tuples (wxyz order)
    """
    data_dict: dict[str, list] = {
        "link/geom_type": [],
        "link/geom_size": [],
        "link/rgba": [],
        "link/pos": [],
        "link/quat_wxyz": [],
    }

    # Get current geom positions
    positions = np.array(
        [
            np.array(physics.data.geom_xpos[geom_id])
            for geom_id in range(physics.model.ngeom)
        ]
    )

    # Handle anchor positions for distance coloring
    if anchor_positions is None:
        anchor_positions = positions.copy()
    else:
        anchor_positions = np.asarray(anchor_positions)
        if anchor_positions.shape[0] != positions.shape[0]:
            # Align shapes by copying available anchors
            aligned = positions.copy()
            overlap = min(anchor_positions.shape[0], positions.shape[0])
            aligned[:overlap] = anchor_positions[:overlap]
            anchor_positions = aligned

    # Compute distance-based colors
    distances = np.linalg.norm(positions - anchor_positions, axis=1)
    normalized = np.clip(distances / max_distance, 0.0, 1.0) ** 0.5
    colormap = plt.get_cmap(colormap_name)
    colors = colormap(normalized)
    colors[:, 3] = 1.0  # Full opacity

    # Extract per-geom data
    for geom_id in range(physics.model.ngeom):
        data_dict["link/geom_type"].append(int(physics.model.geom_type[geom_id]))
        data_dict["link/geom_size"].append(
            [float(x) for x in physics.model.geom_size[geom_id]]
        )

        if use_original_rgba:
            data_dict["link/rgba"].append(
                tuple(float(x) for x in physics.model.geom_rgba[geom_id])
            )
        else:
            data_dict["link/rgba"].append(tuple(float(x) for x in colors[geom_id]))

        data_dict["link/pos"].append([float(x) for x in positions[geom_id]])

        # Convert rotation matrix to quaternion
        quat = quaternions.mat2quat(physics.data.geom_xmat[geom_id])

        # Maintain quaternion sign consistency with previous frame
        if prev_data_dict is not None:
            prev_quat = prev_data_dict["link/quat_wxyz"][geom_id]
            if np.linalg.norm(quat - prev_quat, ord=1) > np.linalg.norm(
                -quat - prev_quat, ord=1
            ):
                quat = -quat

        data_dict["link/quat_wxyz"].append(tuple(float(x) for x in quat))

    return data_dict


# =============================================================================
# SECTION: Rendering Utilities
# =============================================================================


def render_robot_frame(
    physics,
    diffusion_timestep: int,
    episode_timestep: int,
    cfg: DictConfig,
) -> np.ndarray:
    """Render a single frame of the robot visualization.

    Args:
        physics: dm_control Physics object.
        diffusion_timestep: Current diffusion denoising step index.
        episode_timestep: Current episode/trajectory timestep.
        cfg: Configuration with render settings.

    Returns:
        RGB image array with text overlay.
    """
    scene_opt = render_opt()
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    scene_opt.geomgroup[:] = 1

    img = physics.render(
        camera_id=0,
        height=cfg.render_height,
        width=cfg.render_width,
        scene_option=scene_opt,
    )

    # Add text overlay
    text_positions = [tuple(pos) for pos in cfg.text_positions]
    img = add_text_to_image(
        img,
        [
            f"diffusion timestep: {diffusion_timestep}",
            f"episode timestep: {episode_timestep}",
        ],
        text_positions,
        fontsize=cfg.text_fontsize,
        color="rgb(255, 255, 255)",
    )

    return img


# =============================================================================
# SECTION: Main Pipeline
# =============================================================================


def generate_hardware_visualization(
    cfg: DictConfig,
    robot_dicts: list[dict[str, torch.Tensor]],
    seed: int,
    output_dir: Path,
    override_keys: list[str],
) -> None:
    """Generate visualization from pre-computed robot dicts.

    This is the main visualization pipeline that:
    1. Post-processes the robot dicts
    2. Renders each diffusion timestep
    3. Saves videos and state files

    Args:
        cfg: Visualization configuration.
        robot_dicts: List of robot dictionaries (one per diffusion timestep),
            pre-computed by the VisualizationHardwareOptimizer.
        seed: Random seed (for output filename).
        output_dir: Directory for saving outputs (videos, states).
        override_keys: Keys to override from final robot dict.
    """
    last_robot_dict = robot_dicts[-1]

    # Get episode timesteps from observation data
    episode_timesteps = torch.unique(
        robot_dicts[-1]["dyna_joint_obs/time/id"][
            ~last_robot_dict["dyna_joint_obs/mask"]
        ]
    )

    # Post-process all robot dicts
    processed_robot_dicts = [
        post_process_robot_dict(robot_dict, last_robot_dict, override_keys)
        for robot_dict in robot_dicts
    ]

    # Get robot statistics for filename
    last_deserialized_robot = deserialize(
        processed_robot_dicts[-1], include_states=True
    )
    num_dyna_jnts = len(last_deserialized_robot.dynamic_joints)
    num_fixed_jnts = len(last_deserialized_robot.fixed_joints)
    num_actuators = len(last_deserialized_robot.actuators)

    output_prefix = (
        f"hardware_diffusion_a{num_actuators}_d{num_dyna_jnts}"
        f"_f{num_fixed_jnts}_s{seed:02d}"
    )

    # Build anchor positions from final robot
    anchor_positions = build_anchor_positions(
        processed_robot_dicts[-1],
        gravcomp=cfg.gravcomp,
    )

    images: list[np.ndarray] = []
    states: list[dict[str, list]] = []
    logging.disable(logging.CRITICAL)

    # Render each diffusion timestep
    for diffusion_timestep, robot_dict in enumerate(
        tqdm.tqdm(
            processed_robot_dicts,
            desc="Rendering",
            dynamic_ncols=True,
            disable=not cfg.use_pbar,
        )
    ):
        # Create physics simulation
        tokenized_robot = deserialize(robot_dict, include_states=True)
        mjcf_model, _, _, _ = detokenize(tokenized_robot, gravcomp=cfg.gravcomp)
        mjcf_model = set_up_default_scene(
            mjcf_model, add_plane=True, add_vis_cam=True, plane_z_pos=0.0
        )

        # Set render resolution
        getattr(mjcf_model.visual, "global").offheight = cfg.render_height
        getattr(mjcf_model.visual, "global").offwidth = cfg.render_width

        physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        assert physics is not None

        # Determine which episode timesteps to render
        is_last_diffusion_step = diffusion_timestep == len(processed_robot_dicts) - 1
        vis_timesteps = (
            episode_timesteps if is_last_diffusion_step else episode_timesteps[:1]
        )

        if not cfg.render:
            vis_timesteps = []

        for key_frame_idx, vis_timestep in enumerate(vis_timesteps):
            vis_timestep_int = int(vis_timestep.item())
            physics.reset(key_frame_idx)

            img = render_robot_frame(physics, diffusion_timestep, vis_timestep_int, cfg)

            # Repeat frames for last diffusion step
            if is_last_diffusion_step:
                images.extend([img] * cfg.last_diffusion_step_frame_repeat)
            else:
                images.append(img)

            # Capture physics state
            states.append(
                parse_physics_state(
                    physics,
                    anchor_positions=anchor_positions,
                    max_distance=cfg.max_distance,
                    colormap_name=cfg.colormap_name,
                    prev_data_dict=states[-1] if len(states) > 0 else None,
                    use_original_rgba=cfg.use_original_rgba,
                )
            )
    logging.disable(logging.NOTSET)

    # Early return if no images rendered
    if len(images) == 0:
        logger.warning("No images rendered, skipping output files.")
        return

    # Save physics states
    states_path = output_dir / f"{output_prefix}_states.pkl"
    with open(states_path, "wb") as f:
        pickle.dump(states, f)
    logger.info(f"Saved physics states to {states_path}")

    # Save video
    video_path = output_dir / f"{output_prefix}.mp4"
    with imageio.get_writer(str(video_path), mode="I", fps=cfg.video_fps) as writer:
        # Add final frame hold
        all_frames = images + [images[-1]] * cfg.final_frame_hold_count
        for img in all_frames:
            writer.append_data(img)  # type: ignore[attr-defined]
    logger.info(f"Saved video to {video_path}")


def get_output_dir(cfg: DictConfig) -> Path:
    """Get or create the output directory.

    Args:
        cfg: Hydra configuration with optional output_cache_dir field.

    Returns:
        Path to the output directory. If output_cache_dir is None or not set,
        creates a temporary directory.
    """
    output_dir = getattr(cfg, "output_cache_dir", None)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hardware_vis_output_")
        logger.info(f"Using temporary output directory: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    return Path(output_dir)


@hydra.main(
    config_path="../config",
    config_name="hardware_gen_vis",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for hardware diffusion visualization.

    Loads model and trajectories, then generates visualizations for
    each seed in the configured range.
    """
    # Verify correct optimizer class is configured
    assert "VisualizationHardwareOptimizer" in cfg.hardware_optimizer._target_, (
        f"Must use VisualizationHardwareOptimizer for this script, "
        f"got {cfg.hardware_optimizer._target_}"
    )

    # Initialize
    seed_everything(cfg.seed_start)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy config from checkpoint directory
    policy_cfg_path = os.path.dirname(cfg.ckpt_path) + "/cfg.pkl"
    with open(policy_cfg_path, "rb") as f:
        policy_cfg = pickle.load(f)

    cfg.timestep_sampler.num_rollout_steps = int(policy_cfg.seq_len["rollout_steps"])
    OmegaConf.resolve(policy_cfg)

    # Get output directory
    output_dir = get_output_dir(cfg)

    # Set up decoder with intermediate sample collection enabled
    # return_intermediates=True is required so the decoder accepts that argument at call time
    hardware_generator_bundle = setup_decoder(
        policy_cfg,
        ckpt_path=cfg.ckpt_path,
        device=device,
        task_name=cfg.task_name,
        num_inference_steps=cfg.num_inference_steps,
        num_repeats_per_step=cfg.num_repeats_per_step,
        clip_samples_in_guidance=cfg.clip_samples_in_guidance,
        use_flash_attn=cfg.use_flash_attn,
        use_mixed_precision=cfg.use_mixed_precision,
        use_torch_compile=cfg.use_torch_compile,
        eta=cfg.eta,
        return_intermediates=True,
    )

    # Set up augmentation - extract AddPositionId for the optimizer
    augment = hydra.utils.instantiate(policy_cfg.datasets.clean.batch_process_fn)
    add_pos_id: AddPositionId
    if isinstance(augment, ComposeAugmentation):
        add_pos_id_list = [
            aug for aug in augment.augmentations if isinstance(aug, AddPositionId)
        ]
        assert len(add_pos_id_list) == 1, "Expected exactly one AddPositionId"
        add_pos_id = add_pos_id_list[0]
    elif isinstance(augment, AddPositionId):
        add_pos_id = augment
    else:
        raise ValueError(f"Unexpected augmentation type: {type(augment)}")

    # Set up timestep sampler
    sampler: TimestepSampler = hydra.utils.instantiate(cfg.timestep_sampler)

    # Load trajectories
    with open(cfg.pickle_path, "rb") as f:
        trajs = pickle.load(f)

    # Create the VisualizationHardwareOptimizer directly
    # Note: We don't use hydra.utils.instantiate() here because it can't handle
    # the DecoderBundle object which has complex Union type annotations
    optimizer = VisualizationHardwareOptimizer(
        hardware_generator=hardware_generator_bundle,
        reward_fns={},  # No reward functions needed for visualization
        num_seeds=cfg.hardware_optimizer.num_seeds,
        batch_size=cfg.hardware_optimizer.batch_size,
        seq_len_cfg=OmegaConf.to_container(policy_cfg.seq_len, resolve=True),
        timestep_sampler=sampler,
        center_traj=cfg.center_traj,
        add_pos_id=add_pos_id,
        device=device,
        max_traj_len=cfg.max_traj_len,
        output_cache_dir=cfg.hardware_optimizer.output_cache_dir,
    )

    # Get override keys from config
    override_keys = list(cfg.override_keys)

    # Generate visualizations for each seed
    for seed in range(cfg.seed_start, cfg.seed_end):
        try:
            logger.info(f"Generating hardware for seed={seed}, traj_idx={cfg.traj_idx}")

            # Get trajectory
            traj = trajs[cfg.traj_idx]

            # Use the optimizer to get robot_dicts (handles caching internally)
            robot_dicts, predicted_value, optimize_time = optimizer.optimize(
                trajs=[traj], seed=seed
            )
            logger.info(
                f"Optimization complete: value={predicted_value:.4f}, "
                f"time={optimize_time:.2f}s, num_intermediates={len(robot_dicts)}"
            )

            # Generate visualization from the robot dicts
            generate_hardware_visualization(
                cfg=cfg,
                robot_dicts=robot_dicts,
                seed=seed,
                output_dir=output_dir,
                override_keys=override_keys,
            )
        except Exception as e:
            logger.error(f"Error for seed {seed}: {e}")
            raise


if __name__ == "__main__":
    main()
