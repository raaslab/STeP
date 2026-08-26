from __future__ import annotations

import math
import numpy as np


def _hat(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out.astype(np.float32)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    w = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float32)
    k = w / theta
    K = _hat(k)
    R = np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return orthonormalize(R)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = orthonormalize(R).astype(np.float64)
    trace = float(np.trace(R))
    cos_theta = max(min((trace - 1.0) / 2.0, 1.0), -1.0)
    theta = math.acos(cos_theta)

    if theta < 1e-7:
        return np.zeros(3, dtype=np.float32)

    if abs(theta - math.pi) < 1e-4:
        diag = np.diag(R)
        axis = np.sqrt(np.maximum((diag + 1.0) / 2.0, 0.0))
        axis = axis.astype(np.float64)
        if axis[0] > 1e-5:
            axis[1] = (R[0, 1] + R[1, 0]) / (4.0 * axis[0])
            axis[2] = (R[0, 2] + R[2, 0]) / (4.0 * axis[0])
        elif axis[1] > 1e-5:
            axis[0] = (R[0, 1] + R[1, 0]) / (4.0 * axis[1])
            axis[2] = (R[1, 2] + R[2, 1]) / (4.0 * axis[1])
        else:
            axis[0] = (R[0, 2] + R[2, 0]) / (4.0 * axis[2])
            axis[1] = (R[1, 2] + R[2, 1]) / (4.0 * axis[2])
        n = float(np.linalg.norm(axis)) + 1e-8
        return (theta * axis / n).astype(np.float32)

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )
    axis = axis / (2.0 * math.sin(theta))
    return (theta * axis).astype(np.float32)


def orientation_error_rotvec(current_R: np.ndarray, target_R: np.ndarray) -> np.ndarray:
    current_R = orthonormalize(current_R)
    target_R = orthonormalize(target_R)
    R_err = target_R @ current_R.T
    return matrix_to_rotvec(R_err)
