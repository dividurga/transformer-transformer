import argparse
import json
import pathlib
import pickle
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from transforms3d import affines, euler


def parse_iphumi_json(json_path: pathlib.Path):
    """Parse an iPhumi JSON. Returns (utc_times_sec, poses, side) or None if invalid.

    `side` is the contents of the `side` field in the JSON ("left" or "right"),
    or None if absent.
    """
    data = json.load(open(json_path))
    if (
        type(data) is not dict
        or "poseTransforms" not in data
        or len(data["poseTransforms"]) == 0
    ):
        return None
    times_sec = np.array(
        [
            datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            for t in data["poseTimes"]
        ]
    )
    poses = [np.array(p) for p in data["poseTransforms"]]
    side = data.get("side", None)
    return times_sec, poses, side


def resample_arm_traj(
    times_abs: np.ndarray,
    poses: list,
    sampling_times: np.ndarray,
    global_transform: np.ndarray,
    local_transform: np.ndarray,
) -> np.ndarray:
    """Apply transforms and slerp/interp the trajectory at `sampling_times`.

    `times_abs` and `sampling_times` are both absolute UTC seconds.
    Returns array of shape (len(sampling_times), 4, 4).
    """
    positions = []
    rotations = []
    for pose in poses:
        pose = global_transform @ pose @ local_transform
        pos, rotmat = affines.decompose(pose)[:2]
        positions.append(pos)
        rotations.append(R.from_matrix(rotmat))
    positions = np.array(positions)
    rotations = R.concatenate(rotations)

    slerp = Slerp(times=times_abs, rotations=rotations)
    interp_rotmats = slerp(sampling_times).as_matrix()
    interp_positions = np.stack(
        [
            np.interp(sampling_times, times_abs, positions[:, i])
            for i in range(3)
        ],
        axis=1,
    )

    out = np.zeros((len(sampling_times), 4, 4))
    for i in range(len(sampling_times)):
        out[i] = affines.compose(
            T=interp_positions[i], R=interp_rotmats[i], Z=np.ones(3)
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_dt", type=float, default=0.02)
    parser.add_argument(
        "--zero_min_z",
        action="store_true",
        help="Shift each trajectory so its lowest end-effector z is 0.",
    )
    args = parser.parse_args()

    dir_path = pathlib.Path(args.dir_path)
    json_paths = list(dir_path.rglob("*.json"))

    # Group iPhumi pose JSONs by their parent directory. Each demonstration
    # directory contains a `left.json` and a `right.json`; together they form
    # one bimanual episode.
    by_parent: dict = defaultdict(dict)
    for json_path in json_paths:
        parsed = parse_iphumi_json(json_path)
        if parsed is None:
            continue
        times_sec, poses, side = parsed
        if side not in ("left", "right"):
            continue
        by_parent[json_path.parent][side] = (times_sec, poses)

    global_transform = affines.compose(
        T=np.zeros(3),
        R=euler.euler2mat(0, 0, -np.pi / 2)
        @ euler.euler2mat(0, 0, 0)
        @ euler.euler2mat(np.pi / 2, 0, 0),
        Z=np.ones(3),
    )
    local_transform = affines.compose(
        T=np.zeros(3),
        R=euler.euler2mat(0, np.pi, 0)
        @ euler.euler2mat(0, 0, np.pi)
        @ euler.euler2mat(0, 0, 0),
        Z=np.ones(3),
    )

    trajs = []
    episode_timesteps = []
    for parent in sorted(by_parent.keys()):
        sides = by_parent[parent]
        if "left" not in sides or "right" not in sides:
            print(f"Skipping {parent}: only has sides {sorted(sides.keys())}")
            continue
        left_times, left_poses = sides["left"]
        right_times, right_poses = sides["right"]

        # Sample within the absolute-time overlap of the two recordings, so
        # both arms have valid data at every interpolation point.
        t_start = max(left_times.min(), right_times.min())
        t_end = min(left_times.max(), right_times.max())
        if t_end <= t_start:
            print(f"Skipping {parent}: no time overlap")
            continue
        sampling_times = np.arange(t_start, t_end, args.output_dt)

        left_traj = resample_arm_traj(
            left_times,
            left_poses,
            sampling_times,
            global_transform,
            local_transform,
        )
        right_traj = resample_arm_traj(
            right_times,
            right_poses,
            sampling_times,
            global_transform,
            local_transform,
        )

        # axis=1 index 0 = left, index 1 = right. Matches bimanual ViperX
        # (`track_link_idx` 0 is the +Y `left:shoulder_mount` site, 1 is the
        # -Y `right:shoulder_mount` site) and the convention produced by
        # scripts/process_umi_data.py.
        bimanual_traj = np.stack([left_traj, right_traj], axis=1)
        if args.zero_min_z:
            bimanual_traj[:, :, 2, 3] -= bimanual_traj[:, :, 2, 3].min()
        trajs.append(bimanual_traj)
        episode_timesteps.append(len(sampling_times))

    if len(episode_timesteps) > 0:
        print(
            f"Episode lengths - min: {min(episode_timesteps)}, "
            f"max: {max(episode_timesteps)}, "
            f"mean: {np.mean(episode_timesteps):.1f}"
        )
    print(f"Total bimanual episodes: {len(trajs)}")

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(trajs, open(output_path, "wb"))
    print(f"Saved to {output_path}")
