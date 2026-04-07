"""
动作序列损坏检测 - 生成损坏数据脚本

对 vector_272 格式的 .npy 动作序列施加随机帧区间、随机类型的损坏，
输出损坏副本及可供 run_detect.py、calibrate_threshold.py 使用的 txt 文件列表。

参考 StableMotion 的 motion_artifacts_smpl 语义，在 vector_272 上实现等价损坏逻辑。
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# 确保既能导入仓库顶层模块，也能导入 detect 子目录下的辅助脚本。
_SCRIPT_DIR = Path(__file__).resolve().parent
_FEATURE_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _FEATURE_ROOT.parent
_DETECT_DIR = _FEATURE_ROOT / "detect"

for _p in (_REPO_ROOT, _FEATURE_ROOT, _DETECT_DIR, _SCRIPT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# vector_272 维度定义（与 utils/motion_process.py 一致）
NJOINT = 22
DIM_ROOT_VEL = 2       # 0:2   velocities_root_xy_no_heading
DIM_HEADING = 6        # 2:8   global_heading_diff_rot
DIM_POSITIONS = 66     # 8:74  positions_no_heading (22*3)
DIM_VELOCITIES = 66    # 74:140 local velocities (22*3)
DIM_ROTATIONS = 132    # 140:272 joint 6D rotations (22*6)

IDX_ROOT_VEL = slice(0, 2)
IDX_HEADING = slice(2, 8)
IDX_POSITIONS = slice(8, 8 + 3 * NJOINT)
IDX_VELOCITIES = slice(8 + 3 * NJOINT, 8 + 6 * NJOINT)
IDX_ROTATIONS = slice(8 + 6 * NJOINT, 8 + 12 * NJOINT)

# 关节索引（与 HumanML 22 关节一致）
HML_LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]  # pelvis, hips, knees, ankles, feet
HML_LEFT_LEG_JOINTS = [1, 4, 7, 10]   # left_hip, left_knee, left_ankle, left_foot
HML_RIGHT_LEG_JOINTS = [2, 5, 8, 11]  # right_hip, right_knee, right_ankle, right_foot

# CORRUPT_TYPES = ["jittering", "foot sliding", "over smooth", "drifting"]
CORRUPT_TYPES = ["drifting"]

def _get_joint_indices_for_jittering() -> List[int]:
    """随机选择要施加 jittering 的关节子集 (对齐 StableMotion 逻辑)"""
    choice = np.random.choice(4, p=[0.4, 0.3, 0.15, 0.15])
    if choice == 0:
        n = random.randint(1, NJOINT)
        return random.sample(list(range(NJOINT)), n)
    elif choice == 1:
        # 参考项目: HML_LOWER_BODY_JOINTS[random.choice((0, 1, 3, 5, 7)):]
        start = random.choice([0, 1, 3, 5, 7])
        return HML_LOWER_BODY_JOINTS[start:]
    elif choice == 2:
        start = random.choice(range(len(HML_LEFT_LEG_JOINTS)))
        return HML_LEFT_LEG_JOINTS[start:]
    else:
        start = random.choice(range(len(HML_RIGHT_LEG_JOINTS)))
        return HML_RIGHT_LEG_JOINTS[start:]


def _apply_jittering(
    motion: np.ndarray,
    aug_interval: int,
    aug_length: int,
) -> None:
    """对选定区间的关节位置和/或旋转加高斯噪声，可选二次高斯平滑"""
    base_scale = 0.5
    gaussian_noise_std = 0.3  # 提升到 0.3 以应对归一化数据 (原参考值为 0.1)
    rd = np.random.random() * 0.5 + 0.5
    s, e = aug_interval, aug_interval + aug_length

    joints_selected = _get_joint_indices_for_jittering()

    # 生成与参考项目完全一致的噪声项
    noise_term = np.clip(
        np.random.randn(len(motion), NJOINT, 3).astype(np.float32) * gaussian_noise_std * rd,
        -base_scale,
        base_scale,
    )
    noise_term_rot = np.clip(
        np.random.randn(len(motion), NJOINT, 6).astype(np.float32) * gaussian_noise_std * rd,
        -base_scale,
        base_scale,
    )

    # 对 positions (8:74) 施加噪声
    for j in joints_selected:
        j_start = 8 + j * 3
        j_end = j_start + 3
        motion[s:e, j_start:j_end] += noise_term[s:e, j]

    # 对 rotations (140:272) 施加噪声
    for j in joints_selected:
        j_start = 140 + j * 6
        j_end = j_start + 6
        motion[s:e, j_start:j_end] += noise_term_rot[s:e, j]

    # 25% 概率额外做高斯平滑 (使用 truncate 替代 radius 以兼容旧版本 scipy)
    if np.random.random() < 0.25:
        radius = float(round(6 * (np.random.random() * 2 + 2)))
        truncate = radius / 4.0
        for j in joints_selected:
            j_start = 8 + j * 3
            j_end = j_start + 3
            motion[s:e, j_start:j_end] = gaussian_filter1d(
                motion[:, j_start:j_end],
                sigma=4,
                axis=0,
                truncate=truncate,
                mode="nearest",
            )[s:e]
        for j in joints_selected:
            j_start = 140 + j * 6
            j_end = j_start + 6
            motion[s:e, j_start:j_end] = gaussian_filter1d(
                motion[:, j_start:j_end],
                sigma=4,
                axis=0,
                truncate=truncate,
                mode="nearest",
            )[s:e]


def _apply_over_smooth(
    motion: np.ndarray,
    aug_interval: int,
    aug_length: int,
) -> None:
    """对选定区间的 positions 和 rotations 做高斯平滑 (使用 truncate 替代 radius 以兼容旧版本 scipy)"""
    radius = float(round(6 * (np.random.random() * 2 + 2)))
    truncate = radius / 4.0
    s, e = aug_interval, aug_interval + aug_length

    # 参考项目 StableMotion 对整个 poses (相当于这里的 rotations) 做平滑
    # 为了保证 vector_272 内部的一致性，我们对位置和旋转都做平滑
    motion[s:e, IDX_POSITIONS] = gaussian_filter1d(
        motion[:, IDX_POSITIONS],
        sigma=4,
        axis=0,
        truncate=truncate,
        mode="nearest",
    )[s:e]
    motion[s:e, IDX_ROTATIONS] = gaussian_filter1d(
        motion[:, IDX_ROTATIONS],
        sigma=4,
        axis=0,
        truncate=truncate,
        mode="nearest",
    )[s:e]


def _apply_foot_sliding(
    motion: np.ndarray,
    aug_interval: int,
    aug_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对根速度 (0:2) 在区间内施加缩放，并叠加与当前速度同方向的固定最小偏移，
    避免原速度接近 0 时仅缩放仍几乎为 0 导致脚滑检测漏检。
    """
    scale = 0.1
    # 与 drifting 中 root_drift_vel 量级一致的上界参考，使用固定最小偏移
    min_offset = 0.025
    vel_eps = 1e-6
    s, e = aug_interval, aug_interval + aug_length
    mlen = len(motion)

    # 缩放系数（逐帧随机 1.0~1.1）
    diag_vec = np.ones((mlen,), dtype=np.float32)
    diag_vec[s:e] += scale * np.random.random((aug_length,)).astype(np.float32)

    # 损坏前根速度（用于统计与方向）
    old_vel = motion[s:e, IDX_ROOT_VEL].astype(np.float32, copy=True)
    old_vel_norms = np.linalg.norm(old_vel, axis=1)

    # 每帧单位方向：与当前速度同向；近零则随机水平单位向量
    norms_col = old_vel_norms.astype(np.float32)[:, np.newaxis]
    norms_safe = np.maximum(norms_col, np.float32(vel_eps))
    rnd = np.random.randn(aug_length, 2).astype(np.float32)
    rnd_norm = np.linalg.norm(rnd, axis=1, keepdims=True)
    rnd_norm = np.maximum(rnd_norm, np.float32(vel_eps))
    rnd_unit = rnd / rnd_norm
    mask_ok = (old_vel_norms >= vel_eps)[:, np.newaxis]
    vel_dir = np.where(mask_ok, old_vel / norms_safe, rnd_unit)

    # 先缩放整段序列的根速度
    motion[:, IDX_ROOT_VEL] *= diag_vec[:, None]
    # 再在损坏区间叠加同向固定最小偏移
    motion[s:e, IDX_ROOT_VEL] += vel_dir * np.float32(min_offset)

    new_vel = motion[s:e, IDX_ROOT_VEL]
    new_vel_norms = np.linalg.norm(new_vel, axis=1)

    return old_vel_norms, new_vel_norms


def _apply_drifting(
    motion: np.ndarray,
    aug_interval: int,
    aug_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """在区间内对根速度施加漂移速度 (对齐 StableMotion)，返回损坏前后根速度模长。"""
    s, e = aug_interval, aug_interval + aug_length
    old_vel = motion[s:e, IDX_ROOT_VEL].astype(np.float32, copy=True)
    old_vel_norms = np.linalg.norm(old_vel, axis=1)

    # 生成漂移速度方向和量 (参考 StableMotion)
    root_drift_dir = np.random.randn(1, 2).astype(np.float32) + np.random.randn(aug_length, 2).astype(np.float32) * 0.1
    root_drift_vel = np.random.random((aug_length, 1)).astype(np.float32) * 0.025
    root_drift_dir /= np.linalg.norm(root_drift_dir, keepdims=True)
    root_drift_vel = root_drift_vel * root_drift_dir

    # 修正：在速度空间直接累加漂移速度 root_drift_vel
    # 在 StableMotion 中是 trans += cumsum(drift_vel)，等价于 vel += drift_vel
    motion[s:e, IDX_ROOT_VEL] += root_drift_vel

    new_vel = motion[s:e, IDX_ROOT_VEL]
    new_vel_norms = np.linalg.norm(new_vel, axis=1)
    return old_vel_norms, new_vel_norms


def pin_motion_to_origin(motion: np.ndarray) -> np.ndarray:
    """
    将动作钉在原地：去除根节点位移和朝向变化。
    
    vector_272 布局:
      0:2   根节点水平速度 (velocities_root_xy)
      2:8   每帧朝向旋转增量 (global_heading_diff_rot, 6D)
      8:74  去除了朝向的关节位置 (positions_no_heading)
    """
    # 1. 清除根节点水平速度 (0:2)
    motion[:, 0:2] = 0.0
    
    # 2. 清除朝向变化 (2:8) -> 设为 6D 单位旋转 [1, 0, 0, 0, 1, 0]
    # vector_272 存储的是增量旋转 R_diff
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 2:8] = identity_6d
    
    # 3. 确保关节位置 (8:74) 中的 Pelvis XZ 为 0
    # Pelvis 是关节 0, 坐标在 8, 9, 10 (X, Y, Z)
    # motion[:, 8] = 0.0  # Pelvis X
    # motion[:, 10] = 0.0 # Pelvis Z
    # 通常 encode 时已为 0，此处保持现状
    
    return motion


def corrupt_motion_vector272(
    motion: np.ndarray,
    aug_interval: int,
    aug_length: int,
    aug_type: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """对 motion 的指定区间施加指定类型损坏，原地修改。foot sliding / drifting 返回损坏前后根速度模长。"""
    if aug_type == "jittering":
        _apply_jittering(motion, aug_interval, aug_length)
    elif aug_type == "over smooth":
        _apply_over_smooth(motion, aug_interval, aug_length)
    elif aug_type == "foot sliding":
        return _apply_foot_sliding(motion, aug_interval, aug_length)
    elif aug_type == "drifting":
        return _apply_drifting(motion, aug_interval, aug_length)
    else:
        raise NotImplementedError(f"Unknown aug_type: {aug_type}")
    return None


def _intervals_with_types_to_str(
    intervals_0based: List[Tuple[int, int, str]],
) -> str:
    """
    将带类型的 0-based 区间列表转为 1-based 区间字符串，格式为 [s,e,type]。
    不进行合并，以便保留原始损坏类型信息。
    """
    if not intervals_0based:
        return "[]"
    res = []
    for s, e, t in intervals_0based:
        res.append(f"[{s+1},{e+1},{t}]")
    return ",".join(res)


def _intervals_to_mask_then_str(
    intervals_0based: List[Tuple[int, int]],
    seq_len: int,
) -> str:
    """
    将 0-based 区间列表合并为 bool mask，再转为 1-based 区间字符串。
    与 detect_corrupt_utils.corrupt_frames_to_intervals 输出格式一致。
    支持处理重叠区间。
    """
    if not intervals_0based:
        return "[]"
    mask = np.zeros(seq_len, dtype=bool)
    for s, e in intervals_0based:
        # e 是 inclusive 的 0-based index
        s_idx = max(0, min(s, seq_len - 1))
        e_idx = max(s_idx, min(e, seq_len - 1))
        mask[s_idx : e_idx + 1] = True
    
    if not mask.any():
        return "[]"
        
    intervals = []
    in_run = False
    start = 0
    for i in range(len(mask)):
        if mask[i] and not in_run:
            in_run = True
            start = i
        elif not mask[i] and in_run:
            in_run = False
            intervals.append((start + 1, i))  # 1-based inclusive
    if in_run:
        intervals.append((start + 1, len(mask)))
    return ",".join(f"[{s},{e}]" for s, e in intervals)


def collect_npy_files(
    input_dir: Path,
    file_list_path: Optional[str],
    recursive: bool,
) -> List[Path]:
    """收集要处理的 .npy 文件列表"""
    if file_list_path and os.path.exists(file_list_path):
        with open(file_list_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        paths = []
        for name in names:
            base = name[:-4] if name.endswith(".npy") else name
            p = input_dir / f"{base}.npy"
            if p.exists():
                paths.append(p)
            else:
                print(f"警告: 文件不存在，跳过: {p}")
        return paths
    if recursive:
        return sorted(input_dir.rglob("*.npy"))
    return sorted(input_dir.glob("*.npy"))


def run_generate(
    input_dir: str,
    output_dir: str,
    file_list_path: Optional[str] = None,
    min_length: int = 64,
    seed: Optional[int] = None,
    recursive: bool = True,
    suffix: str = "_corrupt",
    visualize: bool = False,
    vis_fps: int = 30,
    pin_to_origin: bool = False,
) -> None:
    """
    主流程：遍历文件，施加损坏，保存并生成 corrupt_list.txt。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 开启原地化时过滤掉根节点位移损坏类型
    current_corrupt_types = CORRUPT_TYPES
    if pin_to_origin:
        current_corrupt_types = [t for t in CORRUPT_TYPES if t not in ["foot sliding", "drifting"]]
        print(f"提示: 原地化已启用。已过滤根节点位移损坏类型。当前可用损坏类型: {current_corrupt_types}")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    output_path.mkdir(parents=True, exist_ok=True)

    files = collect_npy_files(input_path, file_list_path, recursive)
    if not files:
        print("未找到任何 .npy 文件")
        return

    corrupt_entries: List[str] = []
    good_entries: List[str] = []
    gt_entries: List[Tuple[str, str, int]] = []  # (name, gt_intervals, seq_len)
    footsliding_vel_records: List[Tuple[str, int, float, float]] = []  # (name, frame_idx, old_v, new_v)
    drifting_vel_records: List[Tuple[str, int, float, float]] = []

    for fp in tqdm(files, desc="生成损坏数据"):
        try:
            motion = np.load(fp).astype(np.float32)
        except Exception as e:
            print(f"跳过 {fp}: 加载失败 {e}")
            continue

        if motion.ndim != 2 or motion.shape[1] != 272:
            print(f"跳过 {fp}: 形状 {motion.shape} 非 (T, 272)")
            continue

        mlen = len(motion)
        if mlen < min_length:
            continue

        # 如果开启了原地化，先处理 motion
        if pin_to_origin:
            motion = pin_motion_to_origin(motion)

        # 1. 先随机选择本次序列要施加的损坏类型列表 (对齐 StableMotion)
        aug_types_selected = random.sample(current_corrupt_types, random.randint(1, len(current_corrupt_types)))
        
        motion_corrupt = motion.copy()
        gt_intervals_with_types: List[Tuple[int, int, str]] = []

        # 2. 对每种类型独立分配区间
        for aug_type in aug_types_selected:
            # 对齐参考项目的 aug_length 生成逻辑 (* 5)
            # 参考项目: aug_length = min(mlen - 2, int(random.randint(5, min(50, mlen - 2)) * 5))
            aug_length = min(mlen - 2, int(random.randint(5, min(50, mlen - 2)) * 5))
            aug_interval = random.randint(1, mlen - aug_length) # 1-based start in StableMotion logic
            
            # 施加损坏并收集速度统计 (foot sliding / drifting)
            res = corrupt_motion_vector272(motion_corrupt, aug_interval, aug_length, aug_type)
            if res is not None:
                old_v_norms, new_v_norms = res
                stem = fp.stem
                for i_offset, (ov, nv) in enumerate(zip(old_v_norms, new_v_norms)):
                    rec = (stem, aug_interval + i_offset, float(ov), float(nv))
                    if aug_type == "foot sliding":
                        footsliding_vel_records.append(rec)
                    elif aug_type == "drifting":
                        drifting_vel_records.append(rec)
            
            # 记录 GT 区间 (对齐 StableMotion det_mask 逻辑)
            # StableMotion: det_mask[aug_interval - 1: aug_interval + aug_length + 1] = 1
            # 注意: 这里统一为 0-based 区间
            gt_s = max(0, aug_interval - 1)
            gt_e = min(mlen - 1, aug_interval + aug_length)
            gt_intervals_with_types.append((gt_s, gt_e, aug_type))

        # 计算相对路径
        try:
            rel = fp.relative_to(input_path)
        except ValueError:
            rel = fp.name

        # 复制或保存 good 副本到 output_dir
        out_good_full = output_path / rel
        out_good_full.parent.mkdir(parents=True, exist_ok=True)
        if pin_to_origin:
            # 开启原地化时，保存原地化后的 motion
            np.save(out_good_full, motion)
        else:
            shutil.copy2(fp, out_good_full)

        # 输出损坏文件，加 _corrupt 后缀
        stem = rel.stem
        out_rel = rel.parent / f"{stem}{suffix}.npy"
        out_full = output_path / out_rel
        out_full.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_full, motion_corrupt)

        # 记录到 txt：相对 output_dir 的路径，无 .npy
        good_entry = str(rel.with_suffix("")).replace("\\", "/")
        corrupt_entry = str(out_rel.with_suffix("")).replace("\\", "/")
        good_entries.append(good_entry)
        corrupt_entries.append(corrupt_entry)

        # 记录 GT 损坏区间，供 evaluate_detect 对比
        gt_intervals_str = _intervals_with_types_to_str(gt_intervals_with_types)
        gt_entries.append((corrupt_entry, gt_intervals_str, mlen))

        # 可视化：保存原始与损坏动作视频到 output_dir/vis/{rel_stem}/
        if visualize:
            try:
                from detect.detect_corrupt_utils import visualize_motion_vector272
                vis_folder = output_path / "vis" / rel.with_suffix("")
                vis_folder.mkdir(parents=True, exist_ok=True)
                visualize_motion_vector272(motion, str(vis_folder / "input.mp4"), fps=vis_fps)
                visualize_motion_vector272(motion_corrupt, str(vis_folder / "corrupt.mp4"), fps=vis_fps)
            except Exception as e:
                import warnings
                warnings.warn(f"可视化失败 {stem}: {e}")

    # 写入 good_list.txt 与 corrupt_list.txt
    good_list_path = output_path / "good_list.txt"
    corrupt_list_path = output_path / "corrupt_list.txt"
    with open(good_list_path, "w", encoding="utf-8") as f:
        for e in good_entries:
            f.write(e + "\n")
    with open(corrupt_list_path, "w", encoding="utf-8") as f:
        for e in corrupt_entries:
            f.write(e + "\n")

    # 写入 ground_truth_intervals.csv，供 evaluate_detect 评估检测准确性
    gt_csv_path = output_path / "ground_truth_intervals.csv"
    with open(gt_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "gt_intervals", "seq_len"])
        for name, gt_intervals, seq_len in gt_entries:
            writer.writerow([name, gt_intervals, seq_len])

    # 写入 footsliding_vel_stats.csv，用于验证召回率猜想
    if footsliding_vel_records:
        stats_csv_path = output_path / "footsliding_vel_stats.csv"
        with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "frame_idx", "old_v", "new_v"])
            for rec in footsliding_vel_records:
                writer.writerow(rec)
        print(f"  footsliding_vel_stats.csv -> {stats_csv_path}")

    # 写入 drifting_vel_stats.csv，用于分析 drifting 根速度变化是否过小
    if drifting_vel_records:
        drift_stats_path = output_path / "drifting_vel_stats.csv"
        with open(drift_stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "frame_idx", "old_v", "new_v"])
            for rec in drifting_vel_records:
                writer.writerow(rec)
        print(f"  drifting_vel_stats.csv -> {drift_stats_path}")

    print(f"完成: 生成 {len(good_entries)} 个完好副本、{len(corrupt_entries)} 个损坏文件")
    print(f"  good_list.txt -> {good_list_path}")
    print(f"  corrupt_list.txt -> {corrupt_list_path}")
    print(f"  ground_truth_intervals.csv -> {gt_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成损坏动作数据，用于动作序列损坏检测测试"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入目录（含原始 .npy 文件）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录（存放损坏副本及 corrupt_list.txt）",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default=None,
        help="文件列表 txt，每行一个相对 input-dir 的路径。不提供则扫描 input-dir",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=64,
        help="最小序列长度，过短则跳过",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="目录扫描时不递归子目录",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_corrupt",
        help="损坏文件后缀，默认 _corrupt",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="保存原始与损坏动作视频到 output-dir/vis/{name}/input.mp4 与 corrupt.mp4",
    )
    parser.add_argument(
        "--vis-fps",
        type=int,
        default=30,
        help="可视化视频帧率",
    )
    parser.add_argument(
        "--pin-to-origin",
        action="store_true",
        help="开启原地化：生成数据时清除根节点位移和朝向变化，使动作停留在原点",
    )
    args = parser.parse_args()

    run_generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_list_path=args.file_list,
        min_length=args.min_length,
        seed=args.seed,
        recursive=not args.no_recursive,
        suffix=args.suffix,
        visualize=args.visualize,
        vis_fps=args.vis_fps,
        pin_to_origin=args.pin_to_origin,
    )


if __name__ == "__main__":
    main()
