"""
Blender script to import a MuJoCo diffusion pickle and render all frames as PNGs.

Usage:
    blender --background template.blend --python scripts/blender_import_and_render.py -- \
        --pickle /path/to/animation.pkl \
        --output /path/to/output_dir \
        [--fps 50] \
        [--resolution 1920x1080] \
        [--samples 128]

This script requires the MuJoCo Diffusion Importer addon to be installed and enabled
in the template.blend file.

Arguments:
    --pickle: Path to the pickle file containing animation data
    --output: Output directory for rendered PNG images
    --fps: Frames per second for the animation (default: 50)
    --resolution: Resolution as WIDTHxHEIGHT (default: scene settings)
    --samples: Number of render samples for Cycles (default: scene settings)
    --start-frame: Start frame to render (default: 1)
    --end-frame: End frame to render (default: last frame of animation)
    --frame-step: Render every Nth frame (default: 1, render all frames)
"""

import argparse
import os
import sys

import bpy


def parse_args():
    """Parse command line arguments after the -- separator."""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Import MuJoCo diffusion pickle and render all frames"
    )
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
        "--fps",
        type=int,
        default=50,
        help="Frames per second for the animation (default: 50)",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        default=None,
        help="Resolution as WIDTHxHEIGHT (e.g., 1920x1080)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=None,
        help="Number of render samples (Cycles only)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Start frame to render (default: 1)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End frame to render (default: last frame)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame (default: 1)",
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
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Number of animation timesteps (default: data length)",
    )
    parser.add_argument(
        "--state-times",
        type=str,
        default=None,
        help="Path to pickle file with explicit timestamps for each state",
    )
    parser.add_argument(
        "--state-time-scale",
        type=float,
        default=1.0,
        help="Scale factor for state times (default: 1.0)",
    )
    parser.add_argument(
        "--position-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="3D offset applied to all geometry positions",
    )

    return parser.parse_args(argv)


def import_pickle_animation(
    pickle_path: str,
    fps: int,
    interpolation_mode: str,
    timesteps: int | None = None,
    state_times_filepath: str | None = None,
    state_time_scale: float = 1.0,
    position_offset: tuple[float, float, float] | None = None,
):
    """
    Import animation from pickle file using the MuJoCo Diffusion Importer addon.

    Returns the name of the created collection.
    """
    # Import the addon module (it's registered in Blender)
    # The create_animation function is available from the addon
    try:
        # Try to import from the addon module if available
        from import_diffusion_blender import create_animation
    except ImportError:
        # If not available as module, define inline (for standalone use)
        # This requires the addon code to be accessible
        addon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "import_diffusion_blender.py",
        )
        if os.path.exists(addon_path):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "import_diffusion_blender", addon_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            create_animation = module.create_animation
        else:
            raise ImportError(
                f"Could not find import_diffusion_blender.py at {addon_path}"
            )

    # Call create_animation with the pickle file
    create_animation(
        pickle_filepath=pickle_path,
        fps=fps,
        timesteps=timesteps,
        interpolation_mode=interpolation_mode,
        state_times_filepath=state_times_filepath,
        state_time_scale=state_time_scale,
        position_offset=position_offset,
    )

    # Return the collection name (derived from pickle filename)
    collection_name = os.path.splitext(os.path.basename(pickle_path))[0]
    return collection_name


def set_collection_visibility(active_collection_name: str) -> bool:
    """
    Set the specified collection as visible and hide all others.
    """
    if active_collection_name not in bpy.data.collections:
        print(f"Warning: Collection '{active_collection_name}' not found.")
        print("Available collections:")
        for col in bpy.data.collections:
            print(f"  - {col.name}")
        return False

    view_layer = bpy.context.view_layer
    layer_collection = view_layer.layer_collection

    def set_visibility_recursive(layer_col, target_name):
        is_target = layer_col.collection.name == target_name

        if layer_col.collection.name != "Scene Collection":
            layer_col.exclude = not is_target
            layer_col.collection.hide_viewport = not is_target
            layer_col.collection.hide_render = not is_target

        for child in layer_col.children:
            if is_target:
                child.exclude = False
                child.collection.hide_viewport = False
                child.collection.hide_render = False
                for grandchild in child.children:
                    set_visibility_recursive(grandchild, grandchild.collection.name)
            else:
                set_visibility_recursive(child, target_name)

    for child in layer_collection.children:
        set_visibility_recursive(child, active_collection_name)

    print(f"Set collection '{active_collection_name}' as visible.")
    return True


def render_animation(
    output_dir: str,
    collection_name: str,
    resolution: tuple[int, int] | None = None,
    samples: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_step: int = 1,
):
    """
    Render all frames of the animation to the output directory.
    """
    bpy.data.scenes[0].render.engine = "CYCLES"

    # Pick the best compute backend that actually has a device attached.
    # Assignability is not enough: the enum lists whatever was compiled into
    # this Blender build (every official Linux/Windows build offers OPTIX,
    # CUDA, HIP and ONEAPI regardless of the hardware present), while macOS
    # builds offer only NONE/METAL and raise TypeError for the rest. So try
    # each backend and keep the first one that enumerates a non-CPU device.
    prefs = bpy.context.preferences.addons["cycles"].preferences
    has_gpu = False
    for device_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = device_type
        except TypeError:
            continue  # not compiled into this build
        prefs.get_devices()
        if any(d.type != "CPU" for d in prefs.devices):
            has_gpu = True
            break
    if not has_gpu:
        prefs.compute_device_type = "NONE"
        prefs.get_devices()
    print(f"Cycles compute device type: {prefs.compute_device_type}")

    # Enable every GPU device the chosen backend found; render on CPU if none
    for d in prefs.devices:
        d.use = d.type != "CPU"
        print(f"Device {d.name} ({d.type}) use: {d.use}")

    bpy.context.scene.cycles.device = "GPU" if has_gpu else "CPU"
    bpy.context.scene.cycles.denoising_use_gpu = has_gpu

    bpy.context.scene.render.use_overwrite = False

    os.makedirs(output_dir, exist_ok=True)

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"

    if resolution is not None:
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        print(f"Set resolution to {resolution[0]}x{resolution[1]}")

    if samples is not None and scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
        print(f"Set render samples to {samples}")

    # Determine frame range
    if start_frame is None:
        start_frame = scene.frame_start
    if end_frame is None:
        end_frame = scene.frame_end

    print(f"Rendering frames {start_frame} to {end_frame} (step: {frame_step})")

    # Set output path pattern
    # Blender will substitute frame number
    output_pattern = os.path.join(output_dir, f"{collection_name}_frame_")
    scene.render.filepath = output_pattern

    # Render animation
    total_frames = (end_frame - start_frame) // frame_step + 1
    rendered = 0

    for frame in range(start_frame, end_frame + 1, frame_step):
        scene.frame_set(frame)
        output_path = f"{output_pattern}{frame:04d}.png"
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        rendered += 1
        print(f"Rendered frame {frame} ({rendered}/{total_frames}): {output_path}")

    print(f"\nCompleted rendering {rendered} frames to {output_dir}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Blender Import and Render")
    print("=" * 60)
    print(f"Pickle file: {args.pickle}")
    print(f"Output directory: {args.output}")
    print(f"FPS: {args.fps}")
    print(f"Interpolation: {args.interpolation}")
    if args.timesteps:
        print(f"Timesteps: {args.timesteps}")
    if args.resolution:
        print(f"Resolution: {args.resolution}")
    if args.samples:
        print(f"Samples: {args.samples}")
    if args.state_times:
        print(f"State times: {args.state_times}")
        print(f"Time scale: {args.state_time_scale}")
    if args.position_offset:
        print(f"Position offset: {args.position_offset}")
    print("=" * 60)

    # Validate pickle file exists
    if not os.path.exists(args.pickle):
        print(f"Error: Pickle file not found: {args.pickle}")
        sys.exit(1)

    # Parse resolution
    resolution = None
    if args.resolution:
        try:
            width, height = args.resolution.lower().split("x")
            resolution = (int(width), int(height))
        except ValueError:
            print(f"Error: Invalid resolution format '{args.resolution}'")
            sys.exit(1)

    # Import the animation
    print("\n[1/3] Importing animation from pickle...")
    collection_name = import_pickle_animation(
        args.pickle,
        args.fps,
        args.interpolation,
        timesteps=args.timesteps,
        state_times_filepath=args.state_times,
        state_time_scale=args.state_time_scale,
        position_offset=tuple(args.position_offset) if args.position_offset else None,
    )
    print(f"Created collection: {collection_name}")

    # Set collection visibility
    print("\n[2/3] Setting collection visibility...")
    set_collection_visibility(collection_name)

    # Render all frames
    print("\n[3/3] Rendering animation...")
    render_animation(
        output_dir=args.output,
        collection_name=collection_name,
        resolution=resolution,
        samples=args.samples,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
