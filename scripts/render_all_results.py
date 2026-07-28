"""Render evaluation rollouts to supplementary videos via Blender.

Usage:
    python scripts/render_all_results.py --spec scripts/render_spec_example.yaml \
        --zarr_root /path/to/eval_results/ --render_root supp_renders/

The spec YAML describes which robots/tasks/methods to render, the eval zarr
path and trajectory indices for each, and the Blender template per robot.
See scripts/render_spec_example.yaml for the format.
"""

import argparse
import glob
import os
import subprocess
import sys

import yaml

RENDER_SCRIPT = "scripts/render_pickle.py"


def process_render(
    robot_name: str,
    task_name: str,
    method_name: str,
    zarr_path: str,
    traj_indices: list[int] | None,
    render_root: str,
    blender_template: str,
):
    dir_path = os.path.join(render_root, robot_name, task_name, method_name)
    os.makedirs(dir_path, exist_ok=True)
    cmd = [
        "python",
        "scripts/visualize_dataset.py",
        f"dataset.path={zarr_path}",
        "visualization.dt=0.0",
        (
            "dataset.rollout_groups=[ctrl,track_link_obs,dyna_joint_obs,actuator_obs,target_pose]"
            if robot_name != "quadruped"
            else "dataset.rollout_groups=[ctrl,track_link_obs,dyna_joint_obs,actuator_obs,target_pose,free_link_obs]"
        ),
        "dataset.use_cached_indices=false",
        f"pickle_output_root={dir_path}",
        "visualization.disable_gui=true",
    ]
    if traj_indices is not None:
        cmd.extend(
            [f"dataset.filter_episode_seeds=[{','.join(map(str, traj_indices))}]"]
        )
    subprocess.run(cmd, check=True)
    # this generates pickle files in the pickle_output_root
    # now we need to render the pickles using Blender
    pickle_files = glob.glob(os.path.join(dir_path, "*.pkl"))
    for pickle_file in pickle_files:
        print(f"Rendering {pickle_file}")
        render_dir = pickle_file.replace(".pkl", "_renders/")
        base_name = os.path.basename(pickle_file).replace(".pkl", "")

        # Render pickle using Blender via render_pickle.py
        render_cmd = [
            sys.executable,
            RENDER_SCRIPT,
            "--pickle",
            os.path.abspath(pickle_file),
            "--output",
            os.path.abspath(render_dir),
            "--template",
            blender_template,
            "--fps",
            "50",
        ]
        print(f"Running: {' '.join(render_cmd)}")
        subprocess.run(render_cmd, check=True)

        # fill alpha in pngs with white background, then convert to mp4
        output_video = os.path.join(dir_path, f"{base_name}.mp4")
        frame_pattern = os.path.join(
            os.path.abspath(render_dir), f"{base_name}_frame_%04d.png"
        )

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # overwrite output
            "-framerate",
            "50",
            "-start_number",
            "1",  # frames start at 0001
            "-i",
            frame_pattern,
            "-vf",
            # Split input, fill one copy with white, overlay original on top
            "split[s0][s1];[s0]drawbox=c=white:replace=1:t=fill[bg];[bg][s1]overlay=format=auto",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "50",  # ensure output framerate
            output_video,
        ]
        print(f"Running: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Created video: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render evaluation rollouts to videos via Blender."
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to a YAML spec describing what to render "
        "(see scripts/render_spec_example.yaml)",
    )
    parser.add_argument(
        "--zarr_root",
        default="",
        help="Root directory prepended to the zarr paths in the spec",
    )
    parser.add_argument(
        "--render_root",
        default="supp_renders/",
        help="Output directory for pickles, frames, and videos",
    )
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f)

    for robot_name, robot_data in spec["robots"].items():
        blender_template = robot_data["template"]
        for task_name, task_data in robot_data["tasks"].items():
            for method_name, method_data in task_data.items():
                print(f"Rendering {robot_name} {task_name} {method_name}")
                process_render(
                    robot_name=robot_name,
                    task_name=task_name,
                    method_name=method_name,
                    zarr_path=os.path.join(args.zarr_root, method_data["path"]),
                    traj_indices=method_data.get("traj_indices", None),
                    render_root=args.render_root,
                    blender_template=blender_template,
                )
