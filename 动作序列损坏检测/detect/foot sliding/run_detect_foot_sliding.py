"""
动作序列脚滑检测 - 批量检测脚本

接口风格对齐 run_detect.py，但不依赖 FSQ 重建误差，
而是基于关节空间的脚部速度/高度启发式检测脚滑。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# region 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
_DETECT_DIR = _SCRIPT_DIR.parent
_FEATURE_ROOT = _DETECT_DIR.parent
_REPO_ROOT = _FEATURE_ROOT.parent
_ANALYZE_DIR = _FEATURE_ROOT / "analyze"

for _p in (_REPO_ROOT, _FEATURE_ROOT, _DETECT_DIR, _ANALYZE_DIR, _SCRIPT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
# endregion

from dataset_corrupt_detection import create_dataloader
from detect_corrupt_utils import corrupt_frames_to_intervals
from evaluate_detect import load_gt_csv, parse_intervals_to_mask
from foot_sliding_utils import (
    denorm_and_to_joints,
    compute_foot_sliding_mask,
    compute_foot_sliding_mask_stable_motion,
)
from detect_corrupt_utils import visualize_motion_vector272


def run_batch_detect_foot_sliding(
    motion_dir: str,
    output_csv: str,
    file_list_path: Optional[str] = None,
    batch_size: int = 1,
    motion_type: str = "vector_272",
    unit_length: int = 2,
    min_length: int = 64,
    recursive: bool = True,
    output_dir: Optional[str] = None,
    visualize_num: int = 0,
    vis_fps: int = 30,
    gt_csv_path: Optional[str] = None,
    foot_ground_height_thresh: float = 0.03,
    foot_vert_vel_thresh: float = 0.03,
    foot_horiz_vel_k: float = 3.0,
    mode: str = "original",
    fps: int = 30,
) -> None:
    """
    批量脚滑检测：逐条读取 motion，转换到关节空间，生成脚滑帧区间并写入 CSV。

    Args:
        mode: "original" 或 "stable_motion"。
        fps: 帧率，用于速度计算。
    """
    loader, dataset = create_dataloader(
        motion_dir=motion_dir,
        file_list_path=file_list_path,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        motion_type=motion_type,
        unit_length=unit_length,
        min_length=min_length,
        recursive=recursive,
    )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # 可视化输出目录：若未指定，则使用 output_csv 所在目录
    vis_output_dir = output_dir if output_dir else os.path.dirname(output_csv) or "."
    if visualize_num > 0:
        os.makedirs(vis_output_dir, exist_ok=True)

    # 加载 GT 标注（用于可视化叠加）
    gt_dict = {}
    if gt_csv_path and os.path.exists(gt_csv_path):
        gt_dict = load_gt_csv(gt_csv_path)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "corrupt_intervals", "mean_sliding_score"])

        sample_idx = 0
        for batch in tqdm(loader, desc="脚滑检测中"):
            motion, names = batch  # motion: (B, T, 272)
            if isinstance(names, (list, tuple)):
                names = list(names)
            else:
                names = [names]

            for i in range(motion.shape[0]):
                m = motion[i : i + 1]  # (1, T, 272) tensor
                name = names[i] if i < len(names) else str(i)
                name_stem = name[:-4] if name.endswith(".npy") else name

                m_np = m.numpy()
                joints = denorm_and_to_joints(
                    m_np,
                    mean=dataset.mean,
                    std=dataset.std,
                )

                if mode == "stable_motion":
                    mask_L, mask_R, score = compute_foot_sliding_mask_stable_motion(
                        joints,
                        thresh_height=foot_ground_height_thresh,
                        thresh_vel=foot_vert_vel_thresh,
                        fps=fps,
                    )
                    # StableMotion 的核心逻辑是必须两只脚同时滑动
                    corrupt_mask = mask_L & mask_R
                else:
                    mask_L, mask_R, score = compute_foot_sliding_mask(
                        joints,
                        foot_ground_height_thresh=foot_ground_height_thresh,
                        foot_vert_vel_thresh=foot_vert_vel_thresh,
                        foot_horiz_vel_k=foot_horiz_vel_k,
                    )
                    corrupt_mask = mask_L | mask_R

                intervals_str = corrupt_frames_to_intervals(corrupt_mask)
                mean_score = float(score.mean()) if score.size > 0 else 0.0

                writer.writerow(
                    [name_stem, intervals_str, f"{mean_score:.6f}"]
                )

                # 可视化：仅对前 visualize_num 个样本生成视频
                if visualize_num > 0 and sample_idx < visualize_num:
                    motion_denorm = (
                        m_np.squeeze(0) * dataset.std + dataset.mean
                    )  # (T,272)
                    T = motion_denorm.shape[0]

                    # GT mask（若存在）
                    gt_mask = None
                    name_norm = name_stem.replace("\\", "/")
                    if name_norm in gt_dict:
                        gt_intervals, seq_len = gt_dict[name_norm]
                        # 以检测序列实际长度为准构造 mask
                        gt_mask = parse_intervals_to_mask(gt_intervals, T)

                    detected_mask = corrupt_mask.astype(bool)

                    folder = os.path.join(
                        vis_output_dir, name_stem.replace("\\", "/")
                    )
                    os.makedirs(folder, exist_ok=True)
                    out_path = os.path.join(folder, "footsliding.mp4")

                    # 复用现有 vector_272 可视化工具，并叠加 GT / 脚滑检测结果
                    visualize_motion_vector272(
                        motion_denorm,
                        out_path,
                        fps=vis_fps,
                        gt_corrupt_mask=gt_mask,
                        detected_corrupt_mask=detected_mask,
                    )

                sample_idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量脚滑检测（基于关节空间启发式规则）"
    )
    parser.add_argument(
        "--motion-dir",
        type=str,
        required=True,
        help="运动文件根目录，如 dataset/MotionMillion/motion_data/vector_272",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="detect_results_foot_sliding.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        default=None,
        help="文件列表 txt 路径，每行一个文件名（相对 motion-dir）。不提供则扫描目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批大小，建议 1（序列长度可变）",
    )
    parser.add_argument(
        "--motion-type",
        type=str,
        default="vector_272",
        help="运动类型，用于 mean/std 路径",
    )
    parser.add_argument(
        "--unit-length",
        type=int,
        default=2,
        help="长度对齐单位（2^down_t）",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=64,
        help="最小序列长度，过短则跳过",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="目录扫描时不递归子目录",
    )

    # 脚滑检测特有参数
    parser.add_argument(
        "--mode",
        type=str,
        choices=["original", "stable_motion"],
        default="original",
        help="检测模式：'original' 为项目原始逻辑, 'stable_motion' 为 StableMotion 逻辑",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="运动帧率 (影响速度计算, 仅对 stable_motion 模式生效)。StableMotion 默认为 20",
    )
    parser.add_argument(
        "--foot-ground-height-thresh",
        type=float,
        default=0.03,
        help="判定支撑状态时脚与地面的高度阈值（米）",
    )
    parser.add_argument(
        "--foot-vert-vel-thresh",
        type=float,
        default=0.03,
        help="判定支撑状态时脚的竖直速度阈值（米/帧）",
    )
    parser.add_argument(
        "--foot-horiz-vel-k",
        type=float,
        default=3.0,
        help="水平速度自适应阈值的倍数系数（Median + k * MAD）",
    )
    # 可视化相关参数
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可视化输出根目录，未指定则使用 output-csv 所在目录",
    )
    parser.add_argument(
        "--visualize-num",
        type=int,
        default=0,
        help="仅对前 N 个样本输出可视化视频，0 表示不输出可视化",
    )
    parser.add_argument(
        "--vis-fps",
        type=int,
        default=30,
        help="可视化视频帧率",
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        default=None,
        help="ground_truth_intervals.csv 路径，提供则在视频中叠加 GT 标注",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.motion_dir):
        print(f"错误：运动目录不存在 {args.motion_dir}")
        raise SystemExit(1)

    run_batch_detect_foot_sliding(
        motion_dir=args.motion_dir,
        output_csv=args.output_csv,
        file_list_path=args.file_list,
        batch_size=args.batch_size,
        motion_type=args.motion_type,
        unit_length=args.unit_length,
        min_length=args.min_length,
        recursive=not args.no_recursive,
        mode=args.mode,
        fps=args.fps,
        foot_ground_height_thresh=args.foot_ground_height_thresh,
        foot_vert_vel_thresh=args.foot_vert_vel_thresh,
        foot_horiz_vel_k=args.foot_horiz_vel_k,
        output_dir=args.output,
        visualize_num=args.visualize_num,
        vis_fps=args.vis_fps,
        gt_csv_path=args.gt_csv,
    )
    print(f"脚滑检测完成，结果已保存到 {args.output_csv}")


if __name__ == "__main__":
    main()

