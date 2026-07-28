import numpy as np
from transforms3d import axangles, quaternions


def quat_norm(q: np.ndarray) -> float:
    mat = quaternions.quat2mat(q)
    return mat_norm(mat)


def mat_norm(mat: np.ndarray) -> float:
    trace = np.trace(mat)
    trace = np.clip(trace, min=-1 + 1e-8, max=3 - 1e-8)
    rotation_magnitude = np.arccos((trace - 1) / 2)
    rotation_magnitude = rotation_magnitude % (2 * np.pi)
    rotation_magnitude = min(rotation_magnitude, 2 * np.pi - rotation_magnitude)
    return rotation_magnitude


def axangle2quat(axangle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axangle)
    axis = axangle / angle
    return quaternions.mat2quat(axangles.axangle2mat(axis=axis, angle=angle))


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Calculate the rotation matrix that rotates normalized vector vec1 to vec2.

    Parameters:
        vec1: The starting normalized vector (numpy array)
        vec2: The ending normalized vector (numpy array)

    Returns:
        rotation_matrix: 3x3 rotation matrix (numpy array)
    """
    # Ensure input vectors are numpy arrays and normalized
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    # If vectors are parallel, return identity matrix
    if np.allclose(vec1, vec2):
        return np.eye(3)
    # If vectors are antiparallel, rotate 180° around any perpendicular axis
    elif np.allclose(vec1, -vec2):
        # Find a perpendicular vector to rotate around
        perpendicular = np.array([1, 0, 0])
        if np.allclose(perpendicular, vec1) or np.allclose(perpendicular, -vec1):
            perpendicular = np.array([0, 1, 0])
        perpendicular = perpendicular - np.dot(perpendicular, vec1) * vec1
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        # Create rotation matrix for 180° rotation around perpendicular axis
        return axangles.axangle2mat(
            axis=perpendicular,
            angle=np.pi,
            is_normalized=True,
        )

    # Calculate rotation axis (cross product)
    axis = np.cross(vec1, vec2)
    axis = axis / np.linalg.norm(axis)

    # Calculate rotation angle (dot product)
    angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    # Use axis-angle rotation to create rotation matrix
    return axangles.axangle2mat(
        axis=axis,
        angle=angle,
        is_normalized=True,
    )


def mjuu_z2quat(vec):
    """
    NOTE: this should be equivalent to rotation_matrix_from_vectors
    but this is a mujoco function reimplemented in python to reduce
    floating point residual accumulation compared to mujoco
    """
    z = np.array([0, 0, 1])
    quat = np.zeros(4)

    # Compute cross product
    quat[1:] = np.cross(z, vec)

    # Compute norm
    s = np.linalg.norm(quat[1:])

    if s < 1e-10:
        quat[0] = 0  # This value is implicit in C++
        quat[1] = 1
        quat[2] = 0
        quat[3] = 0
    else:
        ang = np.arctan2(s, vec[2])
        quat[0] = np.cos(ang / 2)
        quat[1:] *= np.sin(ang / 2) / s  # Normalize and scale

    return quat
