"""
动作序列损坏检测 - 批量检测脚本

命令行入口，遍历 motion 文件，计算误差，写入 CSV。
支持目录扫描或文件列表两种输入方式。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

# region 路径配置
# 将项目根目录和本目录加入 path，便于导入
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# endregion

from tqdm import tqdm

from dataset_corrupt_detection import create_dataloader
from detect_corrupt_utils import (
    compute_quantization_error,
    compute_reconstruction_error,
    detect_corrupt_per_frame,
    load_detector,
)


# region 批量检测逻辑
def run_batch_detect(
    motion_dir: str,
    output_csv: str,
    ckpt_path: str,
    threshold: float = 0.1,
    metric: str = "recon",
    file_list_path: Optional[str] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    frame_level: bool = False,
    **dataset_kwargs,
) -> None:
    """
    批量检测：遍历 motion 文件，计算误差，写入 CSV。

    流程：加载模型 -> 创建 DataLoader -> 逐 batch 计算误差 -> 写入 CSV。
    frame_level=False：每行记录 文件名、误差值、是否损坏（True/False）。
    frame_level=True：每行记录 文件名、损坏帧区间（如 [1,17],[22,78]）、整段 mean_error（可选）。

    Args:
        motion_dir: 运动文件根目录
        output_csv: 输出 CSV 路径
        ckpt_path: 模型检查点路径
        threshold: 判定阈值
        metric: 'recon' 或 'quant'
        file_list_path: 文件列表 txt，不提供则扫描 motion_dir
        batch_size: 批大小，建议 1（序列长度可变）
        device: 计算设备
        frame_level: 是否启用帧级检测，输出损坏帧区间
        **dataset_kwargs: 传给 create_dataloader 的 dataset 参数
    """
    net, _ = load_detector(ckpt_path, device=device)
    loader, dataset = create_dataloader(
        motion_dir=motion_dir,
        file_list_path=file_list_path,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        **dataset_kwargs,
    )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if frame_level:
            writer.writerow(["name", "corrupt_intervals", "mean_error"])
        else:
            if metric == "recon":
                compute_fn = compute_reconstruction_error
            else:
                compute_fn = compute_quantization_error
            writer.writerow(["name", "error", "corrupt"])

        for batch in tqdm(loader, desc="检测中"):
            motion, names = batch
            if isinstance(names, (list, tuple)):
                names = list(names)
            else:
                names = [names]
            # batch_size=1 时 motion 为 (1, T, 272)
            for i in range(motion.shape[0]):
                m = motion[i : i + 1]
                name = names[i] if i < len(names) else str(i)
                if frame_level:
                    _, _, intervals_str = detect_corrupt_per_frame(
                        net, m, threshold, metric=metric, device=None
                    )
                    mean_err = compute_reconstruction_error(
                        net, m, reduction="mean"
                    ) if metric == "recon" else compute_quantization_error(
                        net, m, reduction="mean"
                    )
                    mean_val = mean_err.item() if hasattr(mean_err, "item") else float(mean_err)
                    writer.writerow([name, intervals_str, f"{mean_val:.6f}"])
                else:
                    err = compute_fn(net, m, reduction="mean")
                    err_val = err.item() if hasattr(err, "item") else float(err)
                    corrupt = err_val > threshold
                    writer.writerow([name, f"{err_val:.6f}", corrupt])
# endregion


# region 命令行入口
def main() -> None:
    parser = argparse.ArgumentParser(description="批量动作序列损坏检测")
    parser.add_argument(
        "--motion-dir",
        type=str,
        required=True,
        help="运动文件根目录，如 dataset/MotionMillion/motion_data/vector_272",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="detect_results.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/pretrained_models/fsq_net_6000000.pth",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="判定阈值，误差超过则判为损坏",
    )
    parser.add_argument(
        "--metric",
        choices=["recon", "quant"],
        default="recon",
        help="检测指标：recon=重构误差，quant=量化误差",
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
        "--device",
        type=str,
        default=None,
        help="计算设备，默认自动选择 cuda/cpu",
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
    parser.add_argument(
        "--frame-level",
        action="store_true",
        help="启用帧级检测，输出损坏帧区间（如 [1,17],[22,78]）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"错误：检查点不存在 {args.ckpt}")
        print("请先下载或训练得到 fsq_net_6000000.pth，参见 ENV_CONFIG.md")
        sys.exit(1)

    if not os.path.isdir(args.motion_dir):
        print(f"错误：运动目录不存在 {args.motion_dir}")
        sys.exit(1)

    run_batch_detect(
        motion_dir=args.motion_dir,
        output_csv=args.output_csv,
        ckpt_path=args.ckpt,
        threshold=args.threshold,
        metric=args.metric,
        file_list_path=args.file_list,
        batch_size=args.batch_size,
        device=args.device,
        frame_level=args.frame_level,
        motion_type=args.motion_type,
        unit_length=args.unit_length,
        min_length=args.min_length,
        recursive=not args.no_recursive,
    )
    print(f"检测完成，结果已保存到 {args.output_csv}")
# endregion


if __name__ == "__main__":
    main()
