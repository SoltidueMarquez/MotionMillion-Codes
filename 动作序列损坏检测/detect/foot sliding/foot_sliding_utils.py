from __future__ import annotations

from typing import Tuple

import numpy as np

# 复用 motion → joints 的完整链路
from utils.motion_process import recover_from_local_rotation
from visualize.smplx2joints import process_smplx_data


def vector272_to_joints(
    motion_denorm: np.ndarray,
    njoint: int = 22,
) -> np.ndarray:
    """
    将反标准化后的 vector_272 序列转换为关节 3D 坐标。

    Args:
        motion_denorm: (T, 272) 或 (1, T, 272) 的 numpy 数组，已反标准化。
        njoint: 关节数量，默认 22（与现有 pipeline 一致）。

    Returns:
        joints: (T, 22, 3) 的关节坐标（世界坐标系，y 轴为高度）。
    """
    motion_denorm = np.asarray(motion_denorm, dtype=np.float32)
    if motion_denorm.ndim == 3:
        motion_denorm = motion_denorm.squeeze(0)
    if motion_denorm.shape[1] != 272:
        raise ValueError(f"期望 motion 维度为 272，实际为 {motion_denorm.shape[1]}")

    T = motion_denorm.shape[0]

    smpl_85 = recover_from_local_rotation(motion_denorm, njoint=njoint)

    # 与 detect_corrupt_utils.visualize_motion_vector272 中保持一致的 SMPLX 拼接逻辑
    smplx_322 = np.concatenate(
        (
            smpl_85[:, :66],
            np.zeros((T, 90), dtype=np.float32),
            np.zeros((T, 3), dtype=np.float32),
            np.zeros((T, 50), dtype=np.float32),
            np.zeros((T, 100), dtype=np.float32),
            smpl_85[:, 72:75],
            smpl_85[:, 75:],
        ),
        axis=-1,
    )

    _, joints, _, _ = process_smplx_data(
        smplx_322, norm_global_orient=False, transform=False
    )

    joints_np = joints[:, :22, :].detach().cpu().numpy()
    return joints_np


def denorm_and_to_joints(
    motion_std: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    从标准化空间的 motion 恢复到真实坐标，并转换为 joints。

    Args:
        motion_std: (T, 272) 或 (1, T, 272)，标准化后的 motion。
        mean: (272,) 均值向量。
        std: (272,) 标准差向量。

    Returns:
        joints: (T, 22, 3) 的关节坐标。
    """
    motion_std = np.asarray(motion_std, dtype=np.float32)
    if motion_std.ndim == 3:
        motion_std = motion_std.squeeze(0)
    motion_denorm = motion_std * std + mean
    return vector272_to_joints(motion_denorm)


def compute_foot_sliding_mask(
    joints: np.ndarray,
    foot_ground_height_thresh: float = 0.03,
    foot_vert_vel_thresh: float = 0.03,
    foot_horiz_vel_k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    基于关节轨迹计算左右脚逐帧脚滑掩码。

    Args:
        joints: (T, 22, 3) 的关节坐标，y 轴为高度。
        foot_ground_height_thresh: 判定支撑状态时脚离地高度阈值（米）。
        foot_vert_vel_thresh: 判定支撑状态时竖直速度阈值（米/帧）。
        foot_horiz_vel_k: 水平速度自适应阈值的倍数系数（Median + k * MAD）。

    Returns:
        mask_L: (T,) bool，左脚是否脚滑。
        mask_R: (T,) bool，右脚是否脚滑。
        score: (T,) float，整体脚滑评分（例如左右脚水平速度的最大值）。
    """
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1] < 12:
        raise ValueError(f"期望 joints 形状为 (T, 22, 3)，实际为 {joints.shape}")

    # 索引约定参照 utils/human_models.SMPX.pos_joints_name 前 22 个
    idx_L_ankle = 7
    idx_R_ankle = 8
    idx_L_foot = 10
    idx_R_foot = 11

    T = joints.shape[0]
    # 取踝和脚的平均作为支撑点
    p_L = (joints[:, idx_L_ankle] + joints[:, idx_L_foot]) * 0.5  # (T, 3)
    p_R = (joints[:, idx_R_ankle] + joints[:, idx_R_foot]) * 0.5

    # 速度（一帧差分）
    v_L = np.zeros_like(p_L)
    v_R = np.zeros_like(p_R)
    v_L[1:] = p_L[1:] - p_L[:-1]
    v_R[1:] = p_R[1:] - p_R[:-1]

    # 地面高度：使用整条序列所有脚底点的最小 y，并做一点冗余
    all_feet_y = np.concatenate([p_L[:, 1], p_R[:, 1]], axis=0)
    ground_y = float(all_feet_y.min())

    # 支撑状态：高度接近地面且竖直速度很小
    support_L = (np.abs(p_L[:, 1] - ground_y) < foot_ground_height_thresh) & (
        np.abs(v_L[:, 1]) < foot_vert_vel_thresh
    )
    support_R = (np.abs(p_R[:, 1] - ground_y) < foot_ground_height_thresh) & (
        np.abs(v_R[:, 1]) < foot_vert_vel_thresh
    )

    # 水平速度
    v_L_h = np.linalg.norm(v_L[:, [0, 2]], axis=1)  # (T,)
    v_R_h = np.linalg.norm(v_R[:, [0, 2]], axis=1)

    def _adaptive_threshold(v: np.ndarray, support: np.ndarray) -> float:
        vals = v[support]
        if vals.size == 0:
            return float("inf")
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        return float(med + foot_horiz_vel_k * (mad + 1e-6))

    thr_L = _adaptive_threshold(v_L_h, support_L)
    thr_R = _adaptive_threshold(v_R_h, support_R)

    mask_L = support_L & (v_L_h > thr_L)
    mask_R = support_R & (v_R_h > thr_R)

    score = np.maximum(v_L_h, v_R_h)
    
    debug_info = {
        "v_L_h": v_L_h,
        "v_R_h": v_R_h,
        "thr_L": thr_L,
        "thr_R": thr_R,
        "support_L": support_L,
        "support_R": support_R
    }
    
    return mask_L, mask_R, score, debug_info


def compute_foot_sliding_mask_stable_motion(
    joints: np.ndarray,
    thresh_height: float = 0.10,
    thresh_vel: float = 0.10,
    fps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    基于 StableMotion 项目中的逻辑实现的脚滑检测 (模式2)。
    保持与 StableMotion/data_loaders/dataset_utils.py:foot_slidedetect_zup 逻辑一致。

    Args:
        joints: (T, 22, 3) 的关节坐标，y 轴为高度 (本项目 convention)。
        thresh_height: 高度阈值 (米)。
        thresh_vel: 水平速度阈值 (米/秒)。
        fps: 帧率。

    Returns:
        mask_L: (T,) bool，左脚是否满足滑动条件。
        mask_R: (T,) bool，右脚是否满足滑动条件。
        score: (T,) float，整体评分 (左右脚水平速度的最大值)。
        debug_info: 包含阈值和速度。
    """
    joints = np.asarray(joints, dtype=np.float32)
    # StableMotion 使用索引: 7, 10 (左), 8, 11 (右)
    foot_joint_index_list = [7, 10, 8, 11]
    joints_foot = joints[:, foot_joint_index_list, :]  # (T, 4, 3)

    # 适配 Y-up 坐标系: y 是高度, (x, z) 是水平面
    # StableMotion 是 Z-up: z 是高度, (x, y) 是水平面
    h_idx = 1
    horiz_idx = [0, 2]

    # offseth: 所有帧所有关节的最小高度
    offseth = joints[:, :, h_idx].min()

    # 水平速度 (m/s)
    # v[i] = (p[i+1] - p[i]) * fps
    joints_feet_horizon_vel = (
        np.linalg.norm(
            joints_foot[1:, :, horiz_idx] - joints_foot[:-1, :, horiz_idx], axis=-1
        )
        * fps
    )  # (T-1, 4)

    # 相对高度
    joints_feet_height = joints_foot[:-1, :, h_idx] - offseth  # (T-1, 4)

    # StableMotion 逻辑:
    # skating_left: 踝部(0)和足部(1)同时满足速度和高度阈值
    # 注意: 踝部高度阈值多 0.05m
    skating_left = (
        (joints_feet_horizon_vel[:, 0] > thresh_vel)
        & (joints_feet_horizon_vel[:, 1] > thresh_vel)
        & (joints_feet_height[:, 0] < (thresh_height + 0.05))
        & (joints_feet_height[:, 1] < thresh_height)
    )

    skating_right = (
        (joints_feet_horizon_vel[:, 2] > thresh_vel)
        & (joints_feet_horizon_vel[:, 3] > thresh_vel)
        & (joints_feet_height[:, 2] < (thresh_height + 0.05))
        & (joints_feet_height[:, 3] < thresh_height)
    )

    # StableMotion 原代码返回的是 torch.logical_and(skating_left, skating_right)
    # 意味着必须双脚同时滑动。为了对齐并输出 mask_L/R，我们记录各自的状态。
    # 并在最终结果中使用 logical_and (如果需要完全一致的话)。
    # 但通常我们希望知道哪只脚滑了。根据用户要求 "保持检测逻辑完全一致"。
    
    # 这里的 mask 是 T-1 长度，我们需要 padding 到 T
    mask_L = np.concatenate([skating_left, skating_left[-1:]])
    mask_R = np.concatenate([skating_right, skating_right[-1:]])
    
    # 计算 score (用于排序/分析)
    v_L_h = np.linalg.norm(
        (joints_foot[1:, 0, horiz_idx] + joints_foot[1:, 1, horiz_idx]) * 0.5 -
        (joints_foot[:-1, 0, horiz_idx] + joints_foot[:-1, 1, horiz_idx]) * 0.5,
        axis=-1
    ) * fps
    v_R_h = np.linalg.norm(
        (joints_foot[1:, 2, horiz_idx] + joints_foot[1:, 3, horiz_idx]) * 0.5 -
        (joints_foot[:-1, 2, horiz_idx] + joints_foot[:-1, 3, horiz_idx]) * 0.5,
        axis=-1
    ) * fps
    v_L_h = np.concatenate([v_L_h, v_L_h[-1:]])
    v_R_h = np.concatenate([v_R_h, v_R_h[-1:]])
    score = np.maximum(v_L_h, v_R_h)

    debug_info = {
        "v_L_h": v_L_h,
        "v_R_h": v_R_h,
        "thr_vel": thresh_vel,
        "height_L": joints_feet_height[:, 0], # padding later if needed
        "height_R": joints_feet_height[:, 2]
    }

    return mask_L, mask_R, score, debug_info


