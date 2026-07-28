#!/usr/bin/env python3
"""Launch Blender to render a physics state pickle file as PNGs.

Runs Blender in background mode with a template .blend scene, imports the
pickle animation data via the MuJoCo Diffusion Importer, and renders all
frames as PNG images using Cycles on GPU.

Usage:
    python scripts/render_pickle.py \
        --pickle /path/to/states.pkl \
        --output /path/to/output_dir/ \
        --template blender_templates/vis_template_quadruped.blend \
        [options]

Examples:
    # Basic render
    python scripts/render_pickle.py -p states.pkl -o renders/ -t scene.blend

    # Custom resolution and samples
    python scripts/render_pickle.py -p states.pkl -o renders/ -t scene.blend \
        --resolution 1920x1080 --samples 256

    # Render a subset of frames
    python scripts/render_pickle.py -p states.pkl -o renders/ -t scene.blend \
        --start-frame 10 --end-frame 50

    # Also encode PNGs to mp4 (fills alpha with white background)
    python scripts/render_pickle.py -p states.pkl -o renders/ -t scene.blend --video

    # Use explicit per-state timestamps
    python scripts/render_pickle.py -p states.pkl -o renders/ -t scene.blend \
        --state-times times.pkl --state-time-scale 2.0
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Render a physics state pickle file using Blender (Cycles, GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--pickle",
        "-p",
        type=str,
        required=True,
        help="Path to the pickle file containing animation data",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for rendered PNG images",
    )
    parser.add_argument(
        "--template",
        "-t",
        type=str,
        required=True,
        help="Blender template .blend file (e.g. blender_templates/vis_template_quadruped.blend)",
    )

    # Animation options
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Animation frames per second (default: 50)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override number of animation timesteps (default: data length)",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="linear",
        choices=[
            "linear",
            "ease_in",
            "ease_in_quintic",
            "ease_out",
            "ease_out_quintic",
            "ease_in_out",
            "step",
        ],
        help="Time interpolation mode (default: linear)",
    )

    # Import options
    parser.add_argument(
        "--state-times",
        type=str,
        default=None,
        help="Pickle file with explicit timestamps for each state",
    )
    parser.add_argument(
        "--state-time-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to state times (default: 1.0)",
    )
    parser.add_argument(
        "--position-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="3D offset applied to all geometry positions",
    )

    # Rendering options
    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        default=None,
        help="Render resolution as WIDTHxHEIGHT (e.g. 1920x1080)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=None,
        help="Number of Cycles render samples",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First frame to render (default: 1)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to render (default: last animation frame)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame (default: 1)",
    )

    # Blender options
    parser.add_argument(
        "--blender",
        type=str,
        default="blender",
        help="Path to the Blender executable (default: blender)",
    )
    parser.add_argument(
        "--engine",
        "-E",
        type=str,
        default="CYCLES",
        help="Blender render engine (default: CYCLES)",
    )

    # Video output
    parser.add_argument(
        "--video",
        action="store_true",
        help="Also create an mp4 video from the rendered PNGs",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="Video FPS (default: same as --fps)",
    )

    args = parser.parse_args()

    # Resolve and validate paths
    pickle_path = os.path.abspath(args.pickle)
    template_path = os.path.abspath(args.template)
    output_dir = os.path.abspath(args.output)

    if not os.path.isfile(pickle_path):
        print(f"Error: pickle file not found: {pickle_path}")
        sys.exit(1)
    if not os.path.isfile(template_path):
        print(f"Error: template file not found: {template_path}")
        sys.exit(1)
    if args.state_times is not None and not os.path.isfile(args.state_times):
        print(f"Error: state-times file not found: {args.state_times}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Locate the Blender-side script next to this file
    blender_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "blender_import_and_render.py"
    )
    if not os.path.isfile(blender_script):
        print(f"Error: blender script not found: {blender_script}")
        sys.exit(1)

    # Build the Blender command
    cmd = [
        args.blender,
        "--background",
        template_path,
        # Without this, Blender exits 0 even when the python script raises,
        # and failures masquerade as successful no-op renders.
        "--python-exit-code",
        "1",
        "--python",
        blender_script,
        "-E",
        args.engine,
        "--",
        "--pickle",
        pickle_path,
        "--output",
        output_dir,
        "--fps",
        str(args.fps),
        "--interpolation",
        args.interpolation,
        "--frame-step",
        str(args.frame_step),
    ]

    if args.resolution is not None:
        cmd.extend(["--resolution", args.resolution])
    if args.samples is not None:
        cmd.extend(["--samples", str(args.samples)])
    if args.start_frame is not None:
        cmd.extend(["--start-frame", str(args.start_frame)])
    if args.end_frame is not None:
        cmd.extend(["--end-frame", str(args.end_frame)])
    if args.timesteps is not None:
        cmd.extend(["--timesteps", str(args.timesteps)])
    if args.state_times is not None:
        cmd.extend(["--state-times", os.path.abspath(args.state_times)])
    if args.state_time_scale != 1.0:
        cmd.extend(["--state-time-scale", str(args.state_time_scale)])
    if args.position_offset is not None:
        cmd.extend(["--position-offset"] + [str(x) for x in args.position_offset])

    print("=" * 60)
    print("Render Pickle")
    print("=" * 60)
    print(f"Pickle:        {pickle_path}")
    print(f"Output:        {output_dir}")
    print(f"Template:      {template_path}")
    print(f"Engine:        {args.engine}")
    print(f"FPS:           {args.fps}")
    print(f"Interpolation: {args.interpolation}")
    if args.resolution:
        print(f"Resolution:    {args.resolution}")
    if args.samples:
        print(f"Samples:       {args.samples}")
    if args.timesteps:
        print(f"Timesteps:     {args.timesteps}")
    if args.state_times:
        print(f"State times:   {os.path.abspath(args.state_times)}")
        print(f"Time scale:    {args.state_time_scale}")
    if args.position_offset:
        print(f"Pos offset:    {args.position_offset}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Blender exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Optionally create mp4 from rendered PNGs
    if args.video:
        video_fps = args.video_fps if args.video_fps is not None else args.fps
        base_name = os.path.splitext(os.path.basename(args.pickle))[0]
        frame_pattern = os.path.join(output_dir, f"{base_name}_frame_%04d.png")
        output_video = os.path.join(output_dir, f"{base_name}.mp4")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(video_fps),
            "-start_number",
            "1",
            "-i",
            frame_pattern,
            "-vf",
            "split[s0][s1];[s0]drawbox=c=white:replace=1:t=fill[bg];[bg][s1]overlay=format=auto",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(video_fps),
            output_video,
        ]
        print(f"\nCreating video: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Created video: {output_video}")

    print("\nDone.")


if __name__ == "__main__":
    main()
