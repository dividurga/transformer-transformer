import argparse
import pickle
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from transforms3d import affines, euler


def axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: Array of shape (..., 3) containing axis-angle vectors

    Returns:
        Array of shape (..., 3, 3) containing rotation matrices
    """
    original_shape = axis_angle.shape[:-1]
    axis_angle_flat = axis_angle.reshape(-1, 3)
    rotmats = R.from_rotvec(axis_angle_flat).as_matrix()
    return rotmats.reshape(*original_shape, 3, 3)


def poses_to_transform_matrices(
    positions: np.ndarray, rotmats: np.ndarray
) -> np.ndarray:
    """Convert positions and rotation matrices to 4x4 transformation matrices.

    Args:
        positions: Array of shape (T, 3)
        rotmats: Array of shape (T, 3, 3)

    Returns:
        Array of shape (T, 4, 4) containing transformation matrices
    """
    T = len(positions)
    transforms = np.zeros((T, 4, 4))
    for i in range(T):
        transforms[i] = affines.compose(
            T=positions[i],
            R=rotmats[i],
            Z=np.ones(3),
        )
    return transforms


def resample_trajectory(
    positions: np.ndarray,
    rotmats: np.ndarray,
    input_dt: float,
    output_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample trajectory from input_dt to output_dt.

    Args:
        positions: Array of shape (T_in, 3)
        rotmats: Array of shape (T_in, 3, 3)
        input_dt: Input time step in seconds
        output_dt: Output time step in seconds

    Returns:
        Tuple of (positions, rotmats) resampled to output_dt
    """
    T_in = len(positions)
    input_times = np.arange(T_in) * input_dt
    max_time = input_times[-1]
    output_times = np.arange(0, max_time, output_dt)

    # Interpolate positions
    interp_positions = np.stack(
        [np.interp(output_times, input_times, positions[:, i]) for i in range(3)],
        axis=1,
    )

    # Interpolate rotations using SLERP
    rotations = R.from_matrix(rotmats)
    slerp = Slerp(times=input_times, rotations=rotations)
    interp_rotmats = slerp(output_times).as_matrix()

    return interp_positions, interp_rotmats


def process_episode(
    robot0_pos: np.ndarray,
    robot0_rot_axis_angle: np.ndarray,
    robot1_pos: np.ndarray,
    robot1_rot_axis_angle: np.ndarray,
    input_dt: float,
    output_dt: float,
    global_transform: np.ndarray,
    local_transform: np.ndarray,
    center_init_xy_pos: bool,
) -> np.ndarray:
    """Process a single episode and return bimanual trajectory.

    Args:
        robot0_pos: Left arm positions (T, 3)
        robot0_rot_axis_angle: Left arm rotations in axis-angle (T, 3)
        robot1_pos: Right arm positions (T, 3)
        robot1_rot_axis_angle: Right arm rotations in axis-angle (T, 3)
        input_dt: Input time step in seconds
        output_dt: Output time step in seconds

    Returns:
        Array of shape (T_out, 2, 4, 4) containing bimanual trajectory
    """
    # Convert axis-angle to rotation matrices
    robot0_rotmats = axis_angle_to_rotmat(robot0_rot_axis_angle)
    robot1_rotmats = axis_angle_to_rotmat(robot1_rot_axis_angle)

    # Resample trajectories
    robot0_pos_resampled, robot0_rotmats_resampled = resample_trajectory(
        robot0_pos, robot0_rotmats, input_dt, output_dt
    )
    robot1_pos_resampled, robot1_rotmats_resampled = resample_trajectory(
        robot1_pos, robot1_rotmats, input_dt, output_dt
    )

    # Convert to 4x4 transformation matrices
    robot0_transforms = poses_to_transform_matrices(
        robot0_pos_resampled, robot0_rotmats_resampled
    )
    robot1_transforms = poses_to_transform_matrices(
        robot1_pos_resampled, robot1_rotmats_resampled
    )
    robot0_transforms = global_transform @ robot0_transforms @ local_transform
    robot1_transforms = global_transform @ robot1_transforms @ local_transform

    if center_init_xy_pos:
        init_xy_pos = np.mean(
            [robot0_transforms[0, :2, 3], robot1_transforms[0, :2, 3]], axis=0
        )
        robot0_transforms[:, :2, 3] = robot0_transforms[:, :2, 3] - init_xy_pos
        robot1_transforms[:, :2, 3] = robot1_transforms[:, :2, 3] - init_xy_pos

    # Stack to create bimanual trajectory: (T, 2, 4, 4)
    # robot1 (right) is end effector 0, robot0 (left) is end effector 1
    bimanual_transforms = np.stack([robot1_transforms, robot0_transforms], axis=1)

    return bimanual_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process UMI bimanual zarr dataset to pickle trajectories"
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        help="Path to the zarr dataset",
        default="data/bimanual_dish_washing.zarr",
        choices=["data/bimanual_dish_washing.zarr", "data/bimanual_cloth_folding.zarr"],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output pickle file",
    )
    parser.add_argument(
        "--output_dt",
        type=float,
        default=0.02,
        help="Output time step in seconds (default: 0.02 = 50Hz)",
    )
    parser.add_argument(
        "--input_hz",
        type=float,
        default=59.94,
        help="Input recording rate in Hz (default: 59.94)",
    )
    parser.add_argument(
        "--center_init_xy_pos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center initial xy position (default: True)",
    )
    parser.add_argument(
        "--global_z_rotation",
        type=float,
        default=-1.5708,
        help="Global z-axis rotation in radians (default: -1.5708 = -90 degrees)",
    )
    parser.add_argument(
        "--global_z_pos_offset",
        type=float,
        default=0.4,
        help="Global z-axis position offset in meters (default: 0.4)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of episodes for training set (default: 0.9, i.e., 90%% train, 10%% test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    args = parser.parse_args()

    # Import zarr here to handle different API versions
    import zarr

    zarr_path = pathlib.Path(args.zarr_path)
    input_dt = 1.0 / args.input_hz

    print(f"Loading zarr dataset from {zarr_path}")
    print(f"Input rate: {args.input_hz} Hz (dt={input_dt:.6f}s)")
    print(f"Output rate: {1.0 / args.output_dt:.2f} Hz (dt={args.output_dt}s)")

    # Open zarr dataset - handle different zarr versions
    try:
        # zarr v3
        z = zarr.open_group(str(zarr_path), mode="r")
    except TypeError:
        # zarr v2
        z = zarr.open(str(zarr_path), mode="r")

    # Load arrays
    robot0_eef_pos = np.array(z["data/robot0_eef_pos"])
    robot0_eef_rot = np.array(z["data/robot0_eef_rot_axis_angle"])
    robot1_eef_pos = np.array(z["data/robot1_eef_pos"])
    robot1_eef_rot = np.array(z["data/robot1_eef_rot_axis_angle"])
    episode_ends = np.array(z["meta/episode_ends"])

    print(f"Loaded {len(episode_ends)} episodes")
    print(f"Total timesteps: {len(robot0_eef_pos)}")

    global_transform = affines.compose(
        T=np.array([0, 0, args.global_z_pos_offset]),
        R=euler.euler2mat(0, 0, args.global_z_rotation),
        Z=np.ones(3),
    )
    local_transform = affines.compose(
        T=np.zeros(3),
        R=euler.euler2mat(0, 0, 0),
        Z=np.ones(3),
    )

    # Process each episode
    trajs = []
    episode_timesteps = []
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    for ep_idx, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        # Extract episode data
        robot0_pos_ep = robot0_eef_pos[start:end]
        robot0_rot_ep = robot0_eef_rot[start:end]
        robot1_pos_ep = robot1_eef_pos[start:end]
        robot1_rot_ep = robot1_eef_rot[start:end]

        # Process episode
        bimanual_traj = process_episode(
            robot0_pos_ep,
            robot0_rot_ep,
            robot1_pos_ep,
            robot1_rot_ep,
            input_dt,
            args.output_dt,
            global_transform,
            local_transform,
            args.center_init_xy_pos,
        )

        trajs.append(bimanual_traj)
        episode_timesteps.append(len(bimanual_traj))

        if (ep_idx + 1) % 50 == 0:
            print(f"Processed {ep_idx + 1}/{len(episode_ends)} episodes")

    print(f"\nProcessed {len(trajs)} episodes")
    print(
        f"Episode lengths - min: {min(episode_timesteps)}, "
        f"max: {max(episode_timesteps)}, mean: {np.mean(episode_timesteps):.1f}"
    )

    # Split into train/test sets
    np.random.seed(args.seed)
    n_episodes = len(trajs)
    n_train = int(n_episodes * args.train_split)
    indices = np.random.permutation(n_episodes)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_trajs = [trajs[i] for i in train_indices]
    test_trajs = [trajs[i] for i in test_indices]

    if "dish_washing" in args.zarr_path:
        assert args.seed == 0, (
            "when seed is 0, trajectory 21 of the test set is flipped"
            + " between left and right, we need to flip it back"
        )
        # for index 21, left and right is flipped
        test_trajs[21][:, [0, 1], :, :] = test_trajs[21][:, [1, 0], :, :]

    print(f"\nTrain/test split: {len(train_trajs)} train, {len(test_trajs)} test")

    # Save to pickle files
    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create train and test output paths by appending _train and _test before extension
    stem = output_path.stem
    suffix = output_path.suffix
    train_output_path = output_path.parent / f"{stem}_train{suffix}"
    test_output_path = output_path.parent / f"{stem}_test{suffix}"

    with open(train_output_path, "wb") as f:
        pickle.dump(train_trajs, f)
    print(f"Saved {len(train_trajs)} train trajectories to {train_output_path}")

    with open(test_output_path, "wb") as f:
        pickle.dump(test_trajs, f)
    print(f"Saved {len(test_trajs)} test trajectories to {test_output_path}")
