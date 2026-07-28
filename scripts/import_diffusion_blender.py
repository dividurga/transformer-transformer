bl_info = {
    "name": "MuJoCo Diffusion Importer",
    "author": "T2 Project",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > MuJoCo",
    "description": "Import MuJoCo diffusion animation data from pickle files",
    "category": "Import-Export",
}

import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper
import pickle
import os
from mathutils import Quaternion, Vector

# MuJoCo geometry type constants
MJGEOM_PLANE = 0
MJGEOM_HFIELD = 1
MJGEOM_SPHERE = 2
MJGEOM_CAPSULE = 3
MJGEOM_ELLIPSOID = 4
MJGEOM_CYLINDER = 5
MJGEOM_BOX = 6
MJGEOM_MESH = 7
MJGEOM_SDF = 8

# Supported geometry types
SUPPORTED_GEOM_TYPES = {MJGEOM_SPHERE, MJGEOM_CAPSULE, MJGEOM_CYLINDER, MJGEOM_BOX}


def load_pickle_data(filepath):
    """Load the pickle file containing the animation data."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def interpolate_frame_index(
    data_index, num_data_frames, target_timesteps, interpolation_mode="linear"
):
    """
    Map data frame index to target frame index based on interpolation mode.

    Args:
        data_index: Index in the original data (0 to num_data_frames-1)
        num_data_frames: Number of frames in the original data
        target_timesteps: Desired number of timesteps in the animation
        interpolation_mode: Type of interpolation ('linear', 'ease_in', 'ease_out', 'ease_in_out', 'step')

    Returns:
        Frame number in the target animation (1-based)
    """
    # Normalize to 0-1 range
    t = data_index / (num_data_frames - 1) if num_data_frames > 1 else 0

    # Apply interpolation function
    if interpolation_mode == "linear":
        mapped_t = t
    elif interpolation_mode == "ease_in":
        # Quadratic ease in
        mapped_t = t * t
    elif interpolation_mode == "ease_in_quintic":
        mapped_t = t**4
    elif interpolation_mode == "ease_out":
        # Quadratic ease out
        mapped_t = 1 - (1 - t) * (1 - t)
    elif interpolation_mode == "ease_out_quintic":
        # Quadratic ease out
        mapped_t = 1 - (1 - t) ** 4
    elif interpolation_mode == "ease_in_out":
        # Smooth step (cubic hermite)
        mapped_t = t * t * (3 - 2 * t)
    elif interpolation_mode == "step":
        # Step function (no interpolation)
        mapped_t = t
    else:
        raise ValueError(f"Unknown interpolation mode: {interpolation_mode}")

    # Map to target frame range (1-based)
    target_frame = 1 + int(mapped_t * (target_timesteps - 1))
    return target_frame


def create_sphere(name):
    """Create a unit sphere mesh."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    # Shade smooth
    bpy.ops.object.shade_auto_smooth()
    return obj


def create_box(name):
    """Create a unit box mesh."""
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    # Shade smooth
    bpy.ops.object.shade_auto_smooth()
    return obj


def create_cylinder(name):
    """Create a unit cylinder mesh."""
    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    # Shade smooth
    bpy.ops.object.shade_auto_smooth()
    return obj


def create_capsule_with_components(name):
    """
    Create a capsule using separate objects for cylinder and caps parented to an empty.
    This allows independent scaling of radius vs length while appearing as a single unit.
    The empty is hidden in viewport and render.
    Boolean modifiers merge the spheres into the cylinder for seamless rendering.
    """
    # Create empty parent
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    parent = bpy.context.active_object
    parent.name = name
    parent.empty_display_size = 0.0  # Hide the empty completely
    parent.hide_viewport = False  # Parent visibility controls children
    parent.hide_render = False

    # Create cylinder body
    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))
    cylinder = bpy.context.active_object
    cylinder.name = f"{name}_cylinder"
    cylinder.parent = parent
    bpy.ops.object.shade_auto_smooth()

    # Create top sphere cap
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 1.0))
    top_cap = bpy.context.active_object
    top_cap.name = f"{name}_top_cap"
    top_cap.parent = parent
    bpy.ops.object.shade_auto_smooth()

    # Create bottom sphere cap
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, -1.0))
    bottom_cap = bpy.context.active_object
    bottom_cap.name = f"{name}_bottom_cap"
    bottom_cap.parent = parent
    bpy.ops.object.shade_auto_smooth()

    # Add boolean modifiers to cylinder to merge with sphere caps
    # This creates a seamless capsule shape when rendered
    bool_top = cylinder.modifiers.new(name="Boolean_Top", type="BOOLEAN")
    bool_top.operation = "UNION"
    bool_top.object = top_cap
    bool_top.solver = "EXACT"

    bool_bottom = cylinder.modifiers.new(name="Boolean_Bottom", type="BOOLEAN")
    bool_bottom.operation = "UNION"
    bool_bottom.object = bottom_cap
    bool_bottom.solver = "EXACT"

    # Hide the sphere caps from viewport and render
    # They must remain in the scene for boolean modifiers to work
    top_cap.hide_viewport = True
    top_cap.hide_render = True
    bottom_cap.hide_viewport = True
    bottom_cap.hide_render = True

    # Store component references
    parent["capsule_cylinder"] = cylinder
    parent["capsule_top_cap"] = top_cap
    parent["capsule_bottom_cap"] = bottom_cap
    parent["is_capsule"] = True

    return parent


def create_geometry(geom_type, name):
    """Create a unit geometry based on type."""
    if geom_type == MJGEOM_SPHERE:
        return create_sphere(name)
    elif geom_type == MJGEOM_BOX:
        return create_box(name)
    elif geom_type == MJGEOM_CYLINDER:
        return create_cylinder(name)
    elif geom_type == MJGEOM_CAPSULE:
        return create_capsule_with_components(name)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


def apply_capsule_scale(parent_obj, radius, half_length):
    """
    Apply proper scaling to a capsule's components.
    Radius scales all components uniformly in XY.
    Half_length only scales the cylinder's Z and repositions caps.
    """
    if not parent_obj.get("is_capsule"):
        return

    cylinder = parent_obj.get("capsule_cylinder")
    top_cap = parent_obj.get("capsule_top_cap")
    bottom_cap = parent_obj.get("capsule_bottom_cap")

    if cylinder and top_cap and bottom_cap:
        # Scale cylinder: radius in XY, half_length in Z
        cylinder.scale = Vector((radius, radius, half_length))

        # Scale caps: only by radius (uniform sphere scaling)
        top_cap.scale = Vector((radius, radius, radius))
        bottom_cap.scale = Vector((radius, radius, radius))

        # Position caps at the ends of the cylinder
        top_cap.location = Vector((0, 0, half_length))
        bottom_cap.location = Vector((0, 0, -half_length))


def get_or_create_base_material():
    """
    Get or create the shared 'Base Material' that reads color from the Object Color property.
    All geometries share this single material, differing only in their object color.

    The material uses an Object Info node to read the per-object color, which can be
    animated via keyframes on each object's `color` property.
    """
    mat_name = "Base Material"

    # Return existing material if already created
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    # Create new material
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get the Principled BSDF node
    bsdf = nodes.get("Principled BSDF")

    # Create Object Info node to read per-object color
    obj_info = nodes.new(type="ShaderNodeObjectInfo")
    obj_info.location = (-300, 300)
    obj_info.label = "Per-Object Color"

    # Create Separate Color node to extract alpha from object color
    # Object color in Blender is RGBA, but Object Info only outputs RGB
    # We'll use a custom property for alpha instead

    # Connect Color output to Base Color input
    links.new(obj_info.outputs["Color"], bsdf.inputs["Base Color"])

    # For alpha, we use the alpha channel of the object color
    # Blender's Object Info node outputs the object's viewport display color
    # which includes alpha in the 4th component
    # However, the "Color" output is RGB only, so we need another approach

    # Create an Attribute node to read a custom "alpha" attribute
    attr_node = nodes.new(type="ShaderNodeAttribute")
    attr_node.attribute_name = "alpha"
    attr_node.attribute_type = "OBJECT"
    attr_node.location = (-300, 100)

    # Connect alpha attribute to BSDF alpha
    links.new(attr_node.outputs["Fac"], bsdf.inputs["Alpha"])

    # Set other BSDF properties
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Subsurface Weight"].default_value = 1.0
        bsdf.inputs["Subsurface Scale"].default_value = 0.5

    # Enable transparency blending
    mat.blend_method = "BLEND"

    return mat


def apply_material_to_object(obj, mat):
    """Apply material to object, handling both mesh objects and capsule parents."""
    if obj.get("is_capsule"):
        # Apply to all capsule components
        cylinder = obj.get("capsule_cylinder")
        top_cap = obj.get("capsule_top_cap")
        bottom_cap = obj.get("capsule_bottom_cap")

        for component in [cylinder, top_cap, bottom_cap]:
            if component and component.data.materials:
                component.data.materials[0] = mat
            elif component:
                component.data.materials.append(mat)
    else:
        # Apply to regular mesh object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def set_object_color(obj, rgba):
    """
    Set the object color property for an object, handling both regular meshes and capsules.
    The object color is read by the shared Base Material via the Object Info node.

    Args:
        obj: Blender object (mesh or capsule parent)
        rgba: Tuple of (r, g, b, a) color values
    """
    if obj.get("is_capsule"):
        # Set color on all capsule components
        cylinder = obj.get("capsule_cylinder")
        top_cap = obj.get("capsule_top_cap")
        bottom_cap = obj.get("capsule_bottom_cap")

        for component in [cylinder, top_cap, bottom_cap]:
            if component:
                # Set viewport display color (RGBA)
                component.color = (rgba[0], rgba[1], rgba[2], rgba[3])
                # Set custom alpha property for shader
                component["alpha"] = rgba[3]
    else:
        # Set color on regular mesh object
        obj.color = (rgba[0], rgba[1], rgba[2], rgba[3])
        # Set custom alpha property for shader
        obj["alpha"] = rgba[3]


def keyframe_object_color(obj, rgba, frame_num):
    """
    Set and keyframe the object color for animation.

    Args:
        obj: Blender object (mesh or capsule parent)
        rgba: Tuple of (r, g, b, a) color values
        frame_num: Frame number to insert keyframe
    """
    if obj.get("is_capsule"):
        # Keyframe color on all capsule components
        cylinder = obj.get("capsule_cylinder")
        top_cap = obj.get("capsule_top_cap")
        bottom_cap = obj.get("capsule_bottom_cap")

        for component in [cylinder, top_cap, bottom_cap]:
            if component:
                component.color = (rgba[0], rgba[1], rgba[2], rgba[3])
                component["alpha"] = rgba[3]
                component.keyframe_insert(data_path="color", frame=frame_num)
                component.keyframe_insert(data_path='["alpha"]', frame=frame_num)
    else:
        obj.color = (rgba[0], rgba[1], rgba[2], rgba[3])
        obj["alpha"] = rgba[3]
        obj.keyframe_insert(data_path="color", frame=frame_num)
        obj.keyframe_insert(data_path='["alpha"]', frame=frame_num)


def mujoco_quat_to_blender(quat_wxyz):
    """Convert MuJoCo quaternion (w,x,y,z) to Blender quaternion (w,x,y,z)."""
    return Quaternion((quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))


def duplicate_hidden_axes(name, collection):
    """
    Duplicate the 'hidden_axes' object, move it to the specified collection,
    and make it visible.

    Args:
        name: Name for the duplicated object
        collection: Blender collection to move the duplicate into

    Returns:
        The duplicated object, or None if 'hidden_axes' doesn't exist
    """
    # Find the hidden_axes object
    hidden_axes = bpy.data.objects.get("hidden_axes")
    if hidden_axes is None:
        print("Warning: 'hidden_axes' object not found in scene")
        return None

    # Duplicate the object (and all its children/hierarchy)
    # First, select only the hidden_axes and its children
    bpy.ops.object.select_all(action="DESELECT")

    # Select the hidden_axes and all its descendants
    def select_hierarchy(obj):
        obj.select_set(True)
        for child in obj.children:
            select_hierarchy(child)

    select_hierarchy(hidden_axes)
    bpy.context.view_layer.objects.active = hidden_axes

    # Duplicate
    bpy.ops.object.duplicate()

    # Get the duplicated parent (the active object after duplication)
    duplicated = bpy.context.active_object
    duplicated.name = name

    # Collect all duplicated objects (parent and children)
    duplicated_objects = [duplicated]

    def collect_children(obj):
        for child in obj.children:
            duplicated_objects.append(child)
            collect_children(child)

    collect_children(duplicated)

    # Move all duplicated objects to the target collection
    for obj in duplicated_objects:
        # Unlink from all current collections
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        # Link to target collection
        collection.objects.link(obj)

        # Make visible
        obj.hide_viewport = False
        obj.hide_render = False

    return duplicated


def create_target_pose_axes(animation_data, collection, num_targets):
    """
    Create axes objects for target poses by duplicating 'hidden_axes'.

    Args:
        animation_data: List of animation data dicts
        collection: Blender collection to add axes to
        num_targets: Number of target poses to create axes for

    Returns:
        List of created axes objects (one per target), or empty list if failed
    """
    axes_objects = []

    for target_idx in range(num_targets):
        name = f"target_pose_{target_idx}"
        axes = duplicate_hidden_axes(name, collection)
        if axes is not None:
            axes_objects.append(axes)
            print(f"Created target pose axes: {name}")
        else:
            print(f"Failed to create target pose axes: {name}")

    return axes_objects


def detect_geometry_changes(animation_data, geom_id):
    """
    Detect frames where geometry type changes for a given geom_id.
    Returns a list of (data_index, new_geom_type) tuples.
    """
    changes = []
    prev_type = None

    for data_idx, data_dict in enumerate(animation_data):
        current_type = data_dict["link/geom_type"][geom_id]

        if prev_type is not None and current_type != prev_type:
            changes.append((data_idx, current_type))

        prev_type = current_type

    return changes


def create_animation(
    pickle_filepath,
    fps=30,
    timesteps=None,
    interpolation_mode="linear",
    state_times_filepath=None,
    state_time_scale: float = 1.0,  # scale the state times by this factor
    position_offset: (
        tuple[float, float, float] | None
    ) = None,  # 3D offset applied to all geometry positions
):
    """
    Main function to create animation from pickle file with shape morphing support.

    Args:
        pickle_filepath: Path to the pickle file containing animation data
        fps: Frames per second for the animation
        timesteps: Number of timesteps in the final animation (default: length of data list)
        interpolation_mode: Type of time interpolation ('linear', 'ease_in', 'ease_out',
                           'ease_in_out', 'step'). Ignored if state_times_filepath is provided.
        state_times_filepath: Optional path to pickle file containing timestamps for each state.
                              When provided, uses these timestamps for keyframing instead of
                              the default interpolation function.
        state_time_scale: Scale the state times by this factor
        position_offset: Optional 3D offset (x, y, z) to apply to all geometry positions
    """
    # Load data
    print(f"Loading pickle file: {pickle_filepath}")
    animation_data = load_pickle_data(pickle_filepath)

    # Load state times if provided
    state_times = None
    if state_times_filepath is not None:
        print(f"Loading state times from: {state_times_filepath}")
        state_times = load_pickle_data(state_times_filepath)
        if not isinstance(state_times, (list, tuple)) and not hasattr(
            state_times, "__len__"
        ):
            raise ValueError("state_times_filepath must contain a list/array of times")
        print(f"Loaded {len(state_times)} state timestamps")
        state_times = [time * state_time_scale for time in state_times]
    if not isinstance(animation_data, list):
        raise ValueError("Expected pickle file to contain a list of data_dicts")

    num_data_frames = len(animation_data)
    print(f"Loaded {num_data_frames} data frames")

    if num_data_frames == 0:
        print("No frames to animate")
        return

    # Set timesteps
    if state_times is not None:
        # When state_times provided, compute total duration and derive timesteps
        min_time = min(state_times)
        max_time = max(state_times)
        total_duration = max_time - min_time
        if timesteps is None:
            # Default to duration * fps frames
            timesteps = max(1, int(total_duration * fps))
        print(
            f"State times range: {min_time:.3f}s to {max_time:.3f}s (duration: {total_duration:.3f}s)"
        )
        print("Using explicit state timestamps for keyframing")
    else:
        if timesteps is None:
            timesteps = num_data_frames
        print(f"Interpolation mode: {interpolation_mode}")
    print(f"Target timesteps: {timesteps}")

    # Get base name from pickle file for collection
    base_name = os.path.splitext(os.path.basename(pickle_filepath))[0]

    # Create or get collection
    if base_name in bpy.data.collections:
        collection = bpy.data.collections[base_name]
        # Clear existing objects in the collection
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        collection = bpy.data.collections.new(base_name)
        bpy.context.scene.collection.children.link(collection)

    print(f"Using collection: {base_name}")

    # Get number of geometries from first frame
    first_frame = animation_data[0]
    num_geoms = len(first_frame["link/geom_type"])
    print(f"Number of geometries: {num_geoms}")

    # Create or get the shared base material
    base_material = get_or_create_base_material()
    print(f"Using shared material: {base_material.name}")

    # Track geometry objects for each entity across type changes
    geom_objects_timeline = {}

    for geom_id in range(num_geoms):
        initial_type = first_frame["link/geom_type"][geom_id]

        if initial_type not in SUPPORTED_GEOM_TYPES:
            print(
                f"Skipping unsupported geometry type {initial_type} for geom {geom_id}"
            )
            geom_objects_timeline[geom_id] = []
            continue

        # Detect all geometry type changes
        type_changes = detect_geometry_changes(animation_data, geom_id)

        # Create objects for each geometry type segment
        segments = []
        current_data_idx = 0
        current_type = initial_type

        # Create initial geometry
        name = f"geom_{geom_id}_type{current_type}_d{current_data_idx}"
        obj = create_geometry(current_type, name)
        collection.objects.link(obj)
        if current_type == MJGEOM_CAPSULE:
            collection.objects.link(obj["capsule_cylinder"])
            collection.objects.link(obj["capsule_top_cap"])
            collection.objects.link(obj["capsule_bottom_cap"])
            bpy.context.scene.collection.objects.unlink(obj["capsule_cylinder"])
            bpy.context.scene.collection.objects.unlink(obj["capsule_top_cap"])
            bpy.context.scene.collection.objects.unlink(obj["capsule_bottom_cap"])

        # Unlink from scene collection to avoid duplicates
        if obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(obj)

        # Apply shared material and set initial vertex color
        rgba = first_frame["link/rgba"][geom_id]
        apply_material_to_object(obj, base_material)
        set_object_color(obj, rgba)

        for change_data_idx, new_type in type_changes:
            # End current segment
            segments.append((current_data_idx, change_data_idx - 1, obj, current_type))

            # Create new geometry for new type
            current_data_idx = change_data_idx
            current_type = new_type
            name = f"geom_{geom_id}_type{new_type}_d{current_data_idx}"
            obj = create_geometry(new_type, name)
            collection.objects.link(obj)
            if current_type == MJGEOM_CAPSULE:
                collection.objects.link(obj["capsule_cylinder"])
                collection.objects.link(obj["capsule_top_cap"])
                collection.objects.link(obj["capsule_bottom_cap"])
                bpy.context.scene.collection.objects.unlink(obj["capsule_cylinder"])
                bpy.context.scene.collection.objects.unlink(obj["capsule_top_cap"])
                bpy.context.scene.collection.objects.unlink(obj["capsule_bottom_cap"])

            # Unlink from scene collection
            if obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(obj)

            # Apply shared material and set initial vertex color for new object
            rgba = animation_data[change_data_idx]["link/rgba"][geom_id]
            apply_material_to_object(obj, base_material)
            set_object_color(obj, rgba)

        # Add final segment
        segments.append((current_data_idx, num_data_frames - 1, obj, current_type))

        geom_objects_timeline[geom_id] = segments
        print(f"Created geom {geom_id} with {len(segments)} segment(s)")

        # Initialize all objects as hidden
        for start_idx, end_idx, obj, geom_type in segments:
            obj.hide_viewport = True
            obj.hide_render = True
            # Also hide capsule components if applicable
            if obj.get("is_capsule"):
                for comp in [
                    obj.get("capsule_cylinder"),
                    obj.get("capsule_top_cap"),
                    obj.get("capsule_bottom_cap"),
                ]:
                    if comp:
                        comp.hide_viewport = True
                        comp.hide_render = True

    # Set up animation
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = timesteps

    # Animate each geometry
    for data_idx, data_dict in enumerate(animation_data):
        # Map data index to target frame
        if state_times is not None:
            # Use explicit timestamps for keyframing
            state_time = state_times[data_idx]
            # Map time to frame: frame = (time - min_time) / total_duration * (timesteps - 1) + 1
            if total_duration > 0:
                normalized_time = (state_time - min_time) / total_duration
            else:
                normalized_time = 0
            frame_num = int(normalized_time * (timesteps - 1)) + 1
        else:
            frame_num = interpolate_frame_index(
                data_idx, num_data_frames, timesteps, interpolation_mode
            )
        bpy.context.scene.frame_set(frame_num)

        for geom_id in range(num_geoms):
            segments = geom_objects_timeline.get(geom_id, [])
            if not segments:
                continue

            # Find the active segment for this data index
            active_obj = None
            active_type = None
            for start_idx, end_idx, obj, geom_type in segments:
                if start_idx <= data_idx <= end_idx:
                    active_obj = obj
                    active_type = geom_type
                    break

            if active_obj is None:
                continue

            # Get frame data
            geom_type = data_dict["link/geom_type"][geom_id]
            geom_size = data_dict["link/geom_size"][geom_id]
            pos = data_dict["link/pos"][geom_id]
            quat_wxyz = data_dict["link/quat_wxyz"][geom_id]
            rgba = data_dict["link/rgba"][geom_id]

            # Set location (apply offset if provided)
            location = Vector(pos)
            if position_offset is not None:
                location += Vector(position_offset)
            active_obj.location = location

            # Set rotation
            active_obj.rotation_mode = "QUATERNION"
            active_obj.rotation_quaternion = mujoco_quat_to_blender(quat_wxyz)

            # Set scale based on geometry type
            if geom_type == MJGEOM_SPHERE:
                radius = geom_size[0]
                active_obj.scale = Vector((radius, radius, radius))
            elif geom_type == MJGEOM_BOX:
                active_obj.scale = Vector(geom_size[:3])
            elif geom_type == MJGEOM_CYLINDER:
                radius = geom_size[0]
                half_length = geom_size[1] if len(geom_size) > 1 else 1.0
                active_obj.scale = Vector((radius, radius, half_length))
            elif geom_type == MJGEOM_CAPSULE:
                radius = geom_size[0]
                half_length = geom_size[1] if len(geom_size) > 1 else 1.0
                apply_capsule_scale(active_obj, radius, half_length)

            # Update and keyframe object color (read by shared Base Material)
            keyframe_object_color(active_obj, rgba, frame_num)

            # Insert keyframes for transform
            active_obj.keyframe_insert(data_path="location", frame=frame_num)
            active_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)

            # Keyframe scale or capsule components
            if geom_type == MJGEOM_CAPSULE:
                cylinder = active_obj.get("capsule_cylinder")
                top_cap = active_obj.get("capsule_top_cap")
                bottom_cap = active_obj.get("capsule_bottom_cap")

                if cylinder:
                    cylinder.keyframe_insert(data_path="scale", frame=frame_num)
                if top_cap:
                    top_cap.keyframe_insert(data_path="scale", frame=frame_num)
                    top_cap.keyframe_insert(data_path="location", frame=frame_num)
                if bottom_cap:
                    bottom_cap.keyframe_insert(data_path="scale", frame=frame_num)
                    bottom_cap.keyframe_insert(data_path="location", frame=frame_num)
            else:
                active_obj.keyframe_insert(data_path="scale", frame=frame_num)

            # Hide other segments at this frame
            for start_idx, end_idx, obj, _ in segments:
                is_active = obj == active_obj

                # Set visibility
                obj.hide_viewport = not is_active
                obj.hide_render = not is_active
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
                obj.keyframe_insert(data_path="hide_render", frame=frame_num)

                # Handle capsule component visibility
                if obj.get("is_capsule"):
                    cylinder = obj.get("capsule_cylinder")
                    top_cap = obj.get("capsule_top_cap")
                    bottom_cap = obj.get("capsule_bottom_cap")
                    if cylinder:
                        cylinder.hide_viewport = not is_active
                        cylinder.hide_render = not is_active
                        cylinder.keyframe_insert(
                            data_path="hide_viewport", frame=frame_num
                        )
                        cylinder.keyframe_insert(
                            data_path="hide_render", frame=frame_num
                        )

                    for comp in [top_cap, bottom_cap]:
                        if comp:
                            comp.hide_viewport = True
                            comp.hide_render = True
                            comp.keyframe_insert(
                                data_path="hide_viewport", frame=frame_num
                            )

    # ==========================================================================
    # Target Pose Animation
    # ==========================================================================

    # Check if target pose data exists in the animation data
    has_target_pose = (
        "target_pose/pos" in first_frame
        and "target_pose/quat" in first_frame
        and len(first_frame["target_pose/pos"]) > 0
    )

    if has_target_pose:
        num_targets = len(first_frame["target_pose/pos"])
        print(f"Found {num_targets} target pose(s), creating axes...")

        # Create axes objects for each target
        target_axes = create_target_pose_axes(animation_data, collection, num_targets)

        if target_axes:
            print(f"Animating {len(target_axes)} target pose axes...")

            # Animate target poses
            for data_idx, data_dict in enumerate(animation_data):
                # Skip if this frame doesn't have target pose data
                if (
                    "target_pose/pos" not in data_dict
                    or "target_pose/quat" not in data_dict
                ):
                    continue

                target_positions = data_dict["target_pose/pos"]
                target_quats = data_dict["target_pose/quat"]

                # Map data index to target frame (same logic as geometry animation)
                if state_times is not None:
                    state_time = state_times[data_idx]
                    if total_duration > 0:
                        normalized_time = (state_time - min_time) / total_duration
                    else:
                        normalized_time = 0
                    frame_num = int(normalized_time * (timesteps - 1)) + 1
                else:
                    frame_num = interpolate_frame_index(
                        data_idx, num_data_frames, timesteps, interpolation_mode
                    )

                # Animate each target
                for target_idx, axes in enumerate(target_axes):
                    if target_idx >= len(target_positions) or target_idx >= len(
                        target_quats
                    ):
                        continue

                    pos = target_positions[target_idx]
                    quat = target_quats[target_idx]

                    # Set location (apply offset if provided)
                    location = Vector(pos)
                    if position_offset is not None:
                        location += Vector(position_offset)
                    axes.location = location

                    # Set rotation
                    axes.rotation_mode = "QUATERNION"
                    axes.rotation_quaternion = mujoco_quat_to_blender(quat)

                    # Insert keyframes
                    axes.keyframe_insert(data_path="location", frame=frame_num)
                    axes.keyframe_insert(
                        data_path="rotation_quaternion", frame=frame_num
                    )

            print(f"Target pose animation complete for {len(target_axes)} target(s)")
    else:
        print("No target pose data found in animation data")

    # ==========================================================================
    # Tracked Link Animation
    # ==========================================================================

    # Check if tracked link observation data exists in the animation data
    has_track_link_obs = (
        "track_link_obs/pos" in first_frame
        and "track_link_obs/quat" in first_frame
        and len(first_frame["track_link_obs/pos"]) > 0
    )

    if has_track_link_obs:
        num_tracked_links = len(first_frame["track_link_obs/pos"])
        print(f"Found {num_tracked_links} tracked link(s), creating axes...")

        # Create axes objects for each tracked link
        tracked_link_axes = []
        for link_idx in range(num_tracked_links):
            name = f"track_link_{link_idx}"
            axes = duplicate_hidden_axes(name, collection)
            if axes is not None:
                tracked_link_axes.append(axes)
                print(f"Created tracked link axes: {name}")
            else:
                print(f"Failed to create tracked link axes: {name}")

        if tracked_link_axes:
            print(f"Animating {len(tracked_link_axes)} tracked link axes...")

            # Animate tracked links
            for data_idx, data_dict in enumerate(animation_data):
                # Skip if this frame doesn't have tracked link data
                if (
                    "track_link_obs/pos" not in data_dict
                    or "track_link_obs/quat" not in data_dict
                ):
                    continue

                track_positions = data_dict["track_link_obs/pos"]
                track_quats = data_dict["track_link_obs/quat"]

                # Map data index to target frame (same logic as geometry animation)
                if state_times is not None:
                    state_time = state_times[data_idx]
                    if total_duration > 0:
                        normalized_time = (state_time - min_time) / total_duration
                    else:
                        normalized_time = 0
                    frame_num = int(normalized_time * (timesteps - 1)) + 1
                else:
                    frame_num = interpolate_frame_index(
                        data_idx, num_data_frames, timesteps, interpolation_mode
                    )

                # Animate each tracked link
                for link_idx, axes in enumerate(tracked_link_axes):
                    if link_idx >= len(track_positions) or link_idx >= len(track_quats):
                        continue

                    pos = track_positions[link_idx]
                    quat = track_quats[link_idx]

                    # Set location (apply offset if provided)
                    location = Vector(pos)
                    if position_offset is not None:
                        location += Vector(position_offset)
                    axes.location = location

                    # Set rotation
                    axes.rotation_mode = "QUATERNION"
                    axes.rotation_quaternion = mujoco_quat_to_blender(quat)

                    # Insert keyframes
                    axes.keyframe_insert(data_path="location", frame=frame_num)
                    axes.keyframe_insert(
                        data_path="rotation_quaternion", frame=frame_num
                    )

            print(
                f"Tracked link animation complete for {len(tracked_link_axes)} link(s)"
            )
    else:
        print("No tracked link observation data found in animation data")

    print(
        f"Animation created with {timesteps} frames from {num_data_frames} data frames"
    )


# =============================================================================
# Blender Add-on Classes
# =============================================================================


class MUJOCO_PG_ImportSettings(PropertyGroup):
    """Property group for MuJoCo diffusion import settings."""

    pickle_filepath: StringProperty(
        name="Animation Data",
        description="Path to pickle file containing animation data",
        default="",
        subtype="FILE_PATH",
    )

    fps: IntProperty(
        name="FPS",
        description="Frames per second for the animation",
        default=50,
        min=1,
        max=240,
    )

    use_custom_timesteps: BoolProperty(
        name="Custom Timesteps",
        description="Use custom number of timesteps instead of data length",
        default=False,
    )

    timesteps: IntProperty(
        name="Timesteps",
        description="Number of timesteps in the final animation (leave unchecked to use data length)",
        default=100,
        min=1,
    )

    interpolation_mode: EnumProperty(
        name="Interpolation",
        description="Type of time interpolation",
        items=[
            ("linear", "Linear", "Linear interpolation"),
            ("ease_in", "Ease In", "Quadratic ease in"),
            ("ease_in_quintic", "Ease In (Quintic)", "Quintic ease in"),
            ("ease_out", "Ease Out", "Quadratic ease out"),
            ("ease_out_quintic", "Ease Out (Quintic)", "Quintic ease out"),
            ("ease_in_out", "Ease In/Out", "Smooth step cubic hermite"),
            ("step", "Step", "Step function (no interpolation)"),
        ],
        default="linear",
    )

    use_state_times: BoolProperty(
        name="Use State Times",
        description="Use explicit timestamps from a separate file",
        default=False,
    )

    state_times_filepath: StringProperty(
        name="State Times",
        description="Path to pickle file containing timestamps for each state",
        default="",
        subtype="FILE_PATH",
    )

    state_time_scale: FloatProperty(
        name="Time Scale",
        description="Scale factor for state times",
        default=1.0,
        min=0.001,
        soft_max=10.0,
    )

    use_position_offset: BoolProperty(
        name="Use Position Offset",
        description="Apply a 3D offset to all geometry positions",
        default=False,
    )

    position_offset: FloatVectorProperty(
        name="Position Offset",
        description="3D offset applied to all geometry positions",
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
    )


class MUJOCO_OT_ImportAnimation(Operator, ImportHelper):
    """Import MuJoCo diffusion animation from pickle file(s)."""

    bl_idname = "mujoco.import_animation"
    bl_label = "Import Animation"
    bl_description = "Import MuJoCo diffusion animation from pickle file(s). Supports multiple selection"
    bl_options = {"REGISTER", "UNDO"}

    # File browser filter
    filter_glob: StringProperty(
        default="*.pkl;*.pickle",
        options={"HIDDEN"},
    )

    # Directory for multi-file selection
    directory: StringProperty(
        subtype="DIR_PATH",
    )

    # Collection of selected files (for multi-select)
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
    )

    def execute(self, context):
        settings = context.scene.mujoco_import_settings

        # Collect all selected filepaths from the files collection
        filepaths = []
        if self.files:
            for f in self.files:
                if f.name:
                    filepath = os.path.join(self.directory, f.name)
                    if os.path.exists(filepath):
                        filepaths.append(filepath)
                        print(f"[MuJoCo Import] Found file: {filepath}")

        # Fallback to single filepath if files collection is empty or didn't yield valid paths
        if not filepaths and self.filepath:
            if os.path.exists(self.filepath):
                filepaths = [self.filepath]
                print(f"[MuJoCo Import] Using single filepath: {self.filepath}")

        if not filepaths:
            self.report({"WARNING"}, "No valid files selected")
            return {"CANCELLED"}

        print(f"[MuJoCo Import] Importing {len(filepaths)} file(s)")

        # Update settings with first filepath (for UI display)
        settings.pickle_filepath = filepaths[0]

        # Import all selected files
        success_count = 0
        fail_count = 0
        for filepath in filepaths:
            print(f"[MuJoCo Import] Importing: {filepath}")
            result = import_animation_from_filepath(context, filepath)
            if result == {"FINISHED"}:
                success_count += 1
                print(f"[MuJoCo Import] Success: {filepath}")
            else:
                fail_count += 1
                print(f"[MuJoCo Import] Failed: {filepath}")

        if success_count == 0:
            self.report({"ERROR"}, "Failed to import any animations")
            return {"CANCELLED"}

        self.report(
            {"INFO"}, f"Imported {success_count} of {len(filepaths)} animation(s)"
        )
        return {"FINISHED"}

    def invoke(self, context, event):
        settings = context.scene.mujoco_import_settings
        if settings.pickle_filepath:
            self.filepath = settings.pickle_filepath
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class MUJOCO_OT_ImportFromPath(Operator):
    """Import animation using the path specified in settings."""

    bl_idname = "mujoco.import_from_path"
    bl_label = "Import from Path"
    bl_description = "Import animation using the filepath in settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        return import_animation_from_settings(context)


class MUJOCO_OT_BrowseStateTimes(Operator, ImportHelper):
    """Browse for state times pickle file."""

    bl_idname = "mujoco.browse_state_times"
    bl_label = "Browse State Times"
    bl_description = "Browse for state times pickle file"

    filter_glob: StringProperty(
        default="*.pkl;*.pickle",
        options={"HIDDEN"},
    )

    def execute(self, context):
        settings = context.scene.mujoco_import_settings
        settings.state_times_filepath = self.filepath
        return {"FINISHED"}


def import_animation_from_filepath(context, filepath):
    """Execute import for a single filepath using current settings."""
    settings = context.scene.mujoco_import_settings

    # Validate filepath
    filepath = bpy.path.abspath(filepath)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {"CANCELLED"}

    # Prepare arguments
    timesteps = settings.timesteps if settings.use_custom_timesteps else None

    state_times_filepath = None
    if settings.use_state_times and settings.state_times_filepath:
        state_times_filepath = bpy.path.abspath(settings.state_times_filepath)

    position_offset = None
    if settings.use_position_offset:
        position_offset = tuple(settings.position_offset)

    # Call the main function
    try:
        create_animation(
            pickle_filepath=filepath,
            fps=settings.fps,
            timesteps=timesteps,
            interpolation_mode=settings.interpolation_mode,
            state_times_filepath=state_times_filepath,
            state_time_scale=settings.state_time_scale,
            position_offset=position_offset,
        )
    except Exception as e:
        print(f"Error importing animation from {filepath}: {e}")
        return {"CANCELLED"}

    return {"FINISHED"}


def import_animation_from_settings(context):
    """Execute import using current settings (single file from UI path)."""
    settings = context.scene.mujoco_import_settings

    if not settings.pickle_filepath:
        return {"CANCELLED"}

    return import_animation_from_filepath(context, settings.pickle_filepath)


class MUJOCO_PT_ImportPanel(Panel):
    """Panel for MuJoCo diffusion import in the 3D View sidebar."""

    bl_label = "MuJoCo Diffusion Import"
    bl_idname = "MUJOCO_PT_import_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MuJoCo"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mujoco_import_settings

        # File selection (shows last imported file, use Import button for multi-select)
        box = layout.box()
        box.label(text="Animation Data", icon="FILE")
        box.prop(settings, "pickle_filepath", text="")

        # Animation settings
        box = layout.box()
        box.label(text="Animation Settings", icon="ANIM")
        box.prop(settings, "fps")

        # Timesteps with toggle
        row = box.row()
        row.prop(settings, "use_custom_timesteps", text="")
        sub = row.row()
        sub.enabled = settings.use_custom_timesteps
        sub.prop(settings, "timesteps")

        # Interpolation mode
        box.prop(settings, "interpolation_mode")

        # State times (optional)
        box = layout.box()
        box.label(text="State Times (Optional)", icon="TIME")
        box.prop(settings, "use_state_times")
        if settings.use_state_times:
            row = box.row(align=True)
            row.prop(settings, "state_times_filepath", text="")
            row.operator("mujoco.browse_state_times", text="", icon="FILEBROWSER")
            box.prop(settings, "state_time_scale")

        # Position offset (optional)
        box = layout.box()
        box.label(text="Position Offset (Optional)", icon="ORIENTATION_GLOBAL")
        box.prop(settings, "use_position_offset")
        if settings.use_position_offset:
            box.prop(settings, "position_offset", text="")

        # Import button
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("mujoco.import_from_path", text="Import Animation", icon="IMPORT")


# =============================================================================
# Registration
# =============================================================================

classes = (
    MUJOCO_PG_ImportSettings,
    MUJOCO_OT_ImportAnimation,
    MUJOCO_OT_ImportFromPath,
    MUJOCO_OT_BrowseStateTimes,
    MUJOCO_PT_ImportPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mujoco_import_settings = bpy.props.PointerProperty(
        type=MUJOCO_PG_ImportSettings
    )


def unregister():
    del bpy.types.Scene.mujoco_import_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
