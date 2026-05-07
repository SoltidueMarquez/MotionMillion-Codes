"""
Encode world-space SMPL-22 joints + global joint rotations into MotionMillion vector_272 (T, 272).

Layout matches 动作序列损坏检测/generate_corrupt_data.py and decode in utils/motion_process.py:
  0:2   root xz velocities (heading-frame convention, see recover_from_local_position)
  2:8   per-frame heading rotation in 6D (Zhou), R_diff[0]=R_cum[0], R_diff[t]=R_cum[t]@R_cum[t-1].T
  8:74  positions without heading (22*3)
  74:140 joint velocities in heading frame (22*3)
  140:272 joint rotations 6D (22*6); stored root = R_cum @ R_global_root, other joints = R_global
"""

from __future__ import annotations

import numpy as np
import torch

from utils.motion_process import accumulate_rotations, recover_from_local_position, recover_from_local_rotation
from utils.face_z_align_util import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

# SMPL kinematic parents for body joints 0..21 (pelvis .. right_wrist)
SMPL22_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)
NJOINT = 22


def _rotation_y(angle: np.ndarray | float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _yaw_rotation_from_root(R_root: np.ndarray) -> np.ndarray:
    """World Y-axis rotation that captures horizontal facing from root orientation (+Z forward in body)."""
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    f = R_root @ ez
    theta = np.arctan2(f[0], f[2])
    return _rotation_y(theta)


def smpl22_fk_global_rotations(poses_66: np.ndarray) -> np.ndarray:
    """
    poses_66: (T, 66) = global orient (3) + body_pose (21*3) in axis-angle, SMPL convention.
    Returns global rotation matrices (T, 22, 3, 3) float64.
    """
    t = torch.from_numpy(poses_66.astype(np.float64))
    t = t.view(-1, 22, 3)
    R_local = axis_angle_to_matrix(t).numpy()
    tlen = R_local.shape[0]
    R_global = np.zeros((tlen, NJOINT, 3, 3), dtype=np.float64)
    for j in range(NJOINT):
        p = SMPL22_PARENTS[j]
        if p < 0:
            R_global[:, j] = R_local[:, j]
        else:
            R_global[:, j] = np.matmul(R_global[:, p], R_local[:, j])
    return R_global


def encode_world_joints_and_rotations_to_vector272(
    joints_world: np.ndarray,
    global_rot_mats: np.ndarray,
) -> np.ndarray:
    """
    joints_world: (T, 22, 3) world positions (e.g. SMPL joints 0..21 from SMPL-H).
    global_rot_mats: (T, 22, 3, 3) global rotation per joint (FK from poses_66).
    """
    joints_world = np.asarray(joints_world, dtype=np.float64)
    global_rot_mats = np.asarray(global_rot_mats, dtype=np.float64)
    t = joints_world.shape[0]
    if joints_world.shape != (t, NJOINT, 3):
        raise ValueError(f"joints_world must be (T,22,3), got {joints_world.shape}")
    if global_rot_mats.shape != (t, NJOINT, 3, 3):
        raise ValueError(f"global_rot_mats must be (T,22,3,3), got {global_rot_mats.shape}")

    R_yaw = np.stack([_yaw_rotation_from_root(global_rot_mats[i, 0]) for i in range(t)], axis=0)
    R_cum = np.transpose(R_yaw, (0, 2, 1))

    R_diff = np.zeros((t, 3, 3), dtype=np.float64)
    R_diff[0] = R_cum[0]
    for i in range(1, t):
        R_diff[i] = R_cum[i] @ R_cum[i - 1].T

    b = np.zeros((t, 3), dtype=np.float64)
    b[:, 0] = joints_world[:, 0, 0]
    b[:, 2] = joints_world[:, 0, 2]

    vel_xyz = np.zeros((t, 3), dtype=np.float64)
    vel_xyz[0, 0] = b[0, 0]
    vel_xyz[0, 2] = b[0, 2]
    for i in range(1, t):
        vel_xyz[i, 0] = b[i, 0] - b[i - 1, 0]
        vel_xyz[i, 2] = b[i, 2] - b[i - 1, 2]

    feat_vx = np.zeros(t, dtype=np.float64)
    feat_vz = np.zeros(t, dtype=np.float64)
    feat_vx[0] = vel_xyz[0, 0]
    feat_vz[0] = vel_xyz[0, 2]
    for i in range(1, t):
        col = R_cum[i - 1] @ vel_xyz[i]
        feat_vx[i] = col[0]
        feat_vz[i] = col[2]

    U = joints_world.copy()
    U[:, :, 0] -= b[:, None, 0]
    U[:, :, 2] -= b[:, None, 2]

    pos_no = np.einsum("tnj,tij->tni", U, R_cum)

    local_vel = np.zeros((t, NJOINT, 3), dtype=np.float64)
    for i in range(t - 1):
        d = joints_world[i + 1] - joints_world[i]
        local_vel[i] = np.einsum("nj,ij->ni", d, R_cum[i])
    if t > 1:
        local_vel[t - 1] = local_vel[t - 2]

    # 将全局旋转转换为局部旋转 (Local Rotations)
    R_local = np.zeros_like(global_rot_mats)
    for j in range(NJOINT):
        p = SMPL22_PARENTS[j]
        if p < 0:
            R_local[:, j] = global_rot_mats[:, j]
        else:
            # R_local = R_parent^T @ R_global
            R_local[:, j] = np.matmul(np.transpose(global_rot_mats[:, p], (0, 2, 1)), global_rot_mats[:, j])
            
    R_stored = R_local.copy()
    # 根节点需要额外减去 yaw 旋转
    R_stored[:, 0] = np.matmul(R_cum, R_local[:, 0])

    out = np.zeros((t, 272), dtype=np.float32)
    out[:, 0] = feat_vx.astype(np.float32)
    out[:, 1] = feat_vz.astype(np.float32)

    R_diff_t = torch.from_numpy(R_diff.reshape(-1, 3, 3).astype(np.float32))
    out[:, 2:8] = matrix_to_rotation_6d(R_diff_t).numpy()
    out[:, 8:74] = pos_no.reshape(t, -1).astype(np.float32)
    out[:, 74:140] = local_vel.reshape(t, -1).astype(np.float32)

    R6 = torch.from_numpy(R_stored.reshape(-1, 3, 3).astype(np.float32))
    out[:, 140:272] = matrix_to_rotation_6d(R6).numpy().reshape(t, -1)

    return out


def global_rot_mats_from_vector272(x: np.ndarray, njoint: int = NJOINT) -> np.ndarray:
    """Invert the rotation packing in recover_from_local_rotation (no SMPL FK)."""
    x = np.asarray(x, dtype=np.float32)
    nfrm = x.shape[0]
    R_stored = (
        rotation_6d_to_matrix(torch.from_numpy(x[:, 140:].reshape(-1, 6)))
        .numpy()
        .reshape(nfrm, njoint, 3, 3)
    )
    R_diff = rotation_6d_to_matrix(torch.from_numpy(x[:, 2:8])).numpy()
    R_cum = accumulate_rotations(R_diff)
    inv_heading = np.transpose(R_cum, (0, 2, 1))
    R_global = R_stored.copy()
    R_global[:, 0] = np.matmul(inv_heading, R_stored[:, 0])
    return R_global.astype(np.float64)


def roundtrip_self_test_from_vector272(x: np.ndarray, njoint: int = NJOINT) -> dict:
    """
    Decode positions/rotations from x, re-encode, compare to x.
    Works for any valid vector_272 sample (e.g. from encode or dataset).
    """
    x = np.asarray(x, dtype=np.float32)
    joints = recover_from_local_position(x, njoint)
    R_glob = global_rot_mats_from_vector272(x, njoint)
    x2 = encode_world_joints_and_rotations_to_vector272(joints.astype(np.float64), R_glob)
    smpl85 = recover_from_local_rotation(x, njoint)
    smpl85_2 = recover_from_local_rotation(x2, njoint)
    joints2 = recover_from_local_position(x2, njoint)
    return {
        "max_abs_smpl85": float(np.abs(smpl85 - smpl85_2).max()),
        "max_abs_joints": float(np.abs(joints - joints2).max()),
        "max_abs_x": float(np.abs(x - x2).max()),
    }


def roundtrip_self_test_synthetic(t: int = 48, seed: int = 0) -> dict:
    """Build motion from random SMPL-like poses + FK positions (SMPL not needed)."""
    rng = np.random.default_rng(seed)
    poses = rng.normal(0, 0.15, (t, 66)).astype(np.float64)
    R_glob = smpl22_fk_global_rotations(poses)
    ez = np.array([0.0, 0.0, 0.05])
    joints = np.zeros((t, NJOINT, 3), dtype=np.float64)
    for i in range(t):
        for j in range(NJOINT):
            joints[i, j] = R_glob[i, j] @ ez * (j + 1) * 0.02
        joints[i, :, 1] += 1.0
        joints[i, :, 0] += np.linspace(0, 0.3, t)[i]
    x = encode_world_joints_and_rotations_to_vector272(joints, R_glob)
    return roundtrip_self_test_from_vector272(x)


if __name__ == "__main__":
    m = roundtrip_self_test_synthetic()
    print("roundtrip_self_test_synthetic:", m)
