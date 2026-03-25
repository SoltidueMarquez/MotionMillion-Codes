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

# 确保项目根目录在 path 中，以便导入 detect_corrupt_utils
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

CORRUPT_TYPES = ["jittering", "foot sliding", "over smooth", "drifting"]
# CORRUPT_TYPES = ["jittering", "over smooth"]

def _get_joint_indices_for_jittering() -> List[int]:
    """随机选择要施加 jittering 的关节子集"""
    choice = np.random.choice(4, p=[0.4, 0.3, 0.15, 0.15])
    if choice == 0:
        n = random.randint(1, NJOINT)
        return random.sample(list(range(NJOINT)), n)
    elif choice == 1:
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
    gaussian_noise_std = 0.1
    rd = np.random.random() * 0.5 + 0.5
    s, e = aug_interval, aug_interval + aug_length

    joints_selected = _get_joint_indices_for_jittering()

    # 对 positions (8:74) 施加噪声
    for j in joints_selected:
        j_start = 8 + j * 3
        j_end = j_start + 3
        noise = np.clip(
            np.random.randn(aug_length, 3).astype(np.float32) * gaussian_noise_std * rd,
            -base_scale,
            base_scale,
        )
        motion[s:e, j_start:j_end] += noise

    # 对 rotations (140:272) 施加噪声
    for j in joints_selected:
        j_start = 140 + j * 6
        j_end = j_start + 6
        noise = np.clip(
            np.random.randn(aug_length, 6).astype(np.float32) * gaussian_noise_std * rd,
            -base_scale,
            base_scale,
        )
        motion[s:e, j_start:j_end] += noise

    # 25% 概率额外做高斯平滑（与 StableMotion 一致，只写回区间 [s,e]）
    if np.random.random() < 0.25:
        truncate = 6 * (np.random.random() * 2 + 2) / 4  # 等价于 radius/sigma，sigma=4
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
    """对选定区间的 positions 或 rotations 做高斯平滑"""
    truncate = 6 * (np.random.random() * 2 + 2) / 4  # 等价于 radius/sigma，sigma=4
    s, e = aug_interval, aug_interval + aug_length

    # 随机选 positions 或 rotations 或两者
    if np.random.random() < 0.5:
        motion[s:e, IDX_POSITIONS] = gaussian_filter1d(
            motion[:, IDX_POSITIONS],
            sigma=4,
            axis=0,
            truncate=truncate,
            mode="nearest",
        )[s:e]
    else:
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
) -> None:
    """对根速度 (0:2) 在区间内施加缩放，模拟脚滑（与 StableMotion disp_matrix 等价）"""
    scale = 0.1
    s, e = aug_interval, aug_interval + aug_length
    mlen = len(motion)

    root_vel = motion[:, IDX_ROOT_VEL].copy()
    diag_vec = np.ones((mlen,), dtype=np.float32)
    diag_vec[s:e] += scale * np.random.random((aug_length,)).astype(np.float32)
    # 等价于 cumsum(root_vel * diag_vec)：修改后的根位移由缩放后的速度累积得到
    motion[:, IDX_ROOT_VEL] = root_vel * diag_vec[:, None]


def _apply_drifting(
    motion: np.ndarray,
    aug_interval: int,
    aug_length: int,
) -> None:
    """在区间内对根速度累加漂移，区间后帧保持末帧漂移"""
    s, e = aug_interval, aug_interval + aug_length
    root_drift_dir = np.random.randn(1, 2).astype(np.float32) + np.random.randn(aug_length, 2).astype(np.float32) * 0.1
    root_drift_vel = np.random.random((aug_length, 1)).astype(np.float32) * 0.025
    root_drift_dir /= np.linalg.norm(root_drift_dir, keepdims=True)
    root_drift_vel = root_drift_vel * root_drift_dir
    root_drift_dist = np.cumsum(root_drift_vel, axis=0)
    motion[s:e, IDX_ROOT_VEL] += root_drift_dist
    if e < len(motion):
        motion[e:, IDX_ROOT_VEL] += root_drift_dist[-1:]


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
) -> None:
    """对 motion 的指定区间施加指定类型损坏，原地修改"""
    if aug_type == "jittering":
        _apply_jittering(motion, aug_interval, aug_length)
    elif aug_type == "over smooth":
        _apply_over_smooth(motion, aug_interval, aug_length)
    elif aug_type == "foot sliding":
        _apply_foot_sliding(motion, aug_interval, aug_length)
    elif aug_type == "drifting":
        _apply_drifting(motion, aug_interval, aug_length)
    else:
        raise NotImplementedError(f"Unknown aug_type: {aug_type}")


def _intervals_to_mask_then_str(
    intervals_0based: List[Tuple[int, int]],
    seq_len: int,
) -> str:
    """
    将 0-based 区间列表合并为 bool mask，再转为 1-based 区间字符串。
    与 detect_corrupt_utils.corrupt_frames_to_intervals 输出格式一致。
    """
    if not intervals_0based:
        return "[]"
    mask = np.zeros(seq_len, dtype=bool)
    for s, e in intervals_0based:
        s = max(0, min(s, seq_len - 1))
        e = max(s, min(e, seq_len - 1))
        mask[s : e + 1] = True
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
    num_intervals: Tuple[int, int] = (1, 3),
    min_interval: int = 5,
    max_interval: int = 50,
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

        # 随机选择区间数量
        n_intervals = random.randint(num_intervals[0], min(num_intervals[1], mlen // min_interval))
        n_intervals = max(1, n_intervals)

        motion_corrupt = motion.copy()
        gt_intervals_0based: List[Tuple[int, int]] = []

        for _ in range(n_intervals):
            aug_length = random.randint(
                min_interval,
                min(max_interval, mlen - 2),
            )
            aug_interval = random.randint(0, mlen - aug_length)
            aug_types = random.sample(current_corrupt_types, random.randint(1, len(current_corrupt_types)))
            for aug_type in aug_types:
                corrupt_motion_vector272(motion_corrupt, aug_interval, aug_length, aug_type)
            # drifting 与 foot sliding 都会使根位移传播到序列末尾
            propagates_to_end = "drifting" in aug_types or "foot sliding" in aug_types
            gt_end = mlen - 1 if propagates_to_end else aug_interval + aug_length - 1
            gt_intervals_0based.append((aug_interval, gt_end))

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
        gt_intervals_str = _intervals_to_mask_then_str(gt_intervals_0based, mlen)
        gt_entries.append((corrupt_entry, gt_intervals_str, mlen))

        # 可视化：保存原始与损坏动作视频到 output_dir/vis/{rel_stem}/
        if visualize:
            try:
                from detect_corrupt_utils import visualize_motion_vector272
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
        "--num-intervals",
        type=str,
        default="1,3",
        help="每个文件随机损坏区间数量范围，如 '1,3' 表示 1~3 个区间",
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=5,
        help="最小区间帧数",
    )
    parser.add_argument(
        "--max-interval",
        type=int,
        default=50,
        help="最大区间帧数",
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

    num_lo, num_hi = map(int, args.num_intervals.split(","))
    num_intervals = (num_lo, num_hi)

    run_generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_list_path=args.file_list,
        num_intervals=num_intervals,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
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
