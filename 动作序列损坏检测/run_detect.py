"""
动作序列损坏检测 - 批量检测脚本

命令行入口，遍历 motion 文件，计算误差，写入 CSV。
支持目录扫描或文件列表两种输入方式。
"""

# 使类型注解支持前向引用（如 Optional[str]），兼容 Python 3.7+
from __future__ import annotations

import argparse   # 解析命令行参数
import csv        # 读写 CSV 文件
import os         # 路径、目录操作
import sys        # 修改 sys.path 以导入项目模块
from pathlib import Path
from typing import Optional  # 可选类型，如 Optional[str] 表示 str 或 None

# region 路径配置
# 获取本脚本所在目录的绝对路径（即 动作序列损坏检测/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 = 动作序列损坏检测 的上一级（即 MotionMillion-Codes/）
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
# 将项目根目录和本目录加入 Python 搜索路径，这样 import models、dataset_corrupt_detection 等才能找到
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# endregion

from tqdm import tqdm  # 进度条显示

# 从同目录下的模块导入：数据加载器、检测工具函数
from dataset_corrupt_detection import create_dataloader
from detect_corrupt_utils import (
    compute_quantization_error,   # 计算量化误差
    compute_reconstruction_error, # 计算重构误差
    detect_corrupt_per_frame,     # 帧级检测，返回损坏区间
    load_detector,                # 加载预训练 FSQ-VQ-VAE 模型
)


# region 批量检测逻辑
def run_batch_detect(
    motion_dir: str,           # 运动文件根目录，如 dataset/MotionMillion/motion_data/vector_272
    output_csv: str,           # 输出 CSV 文件路径
    ckpt_path: str,            # 预训练模型权重路径（fsq_net_6000000.pth）
    threshold: float = 0.1,    # 判定阈值：误差超过此值则判为损坏
    metric: str = "recon",     # 检测指标：'recon'=重构误差，'quant'=量化误差
    file_list_path: Optional[str] = None,  # 可选：指定 txt 文件，每行一个相对路径，不提供则扫描 motion_dir
    batch_size: int = 1,       # 批大小，建议 1（因为不同序列长度可能不同，pad 后 batch>1 也可）
    device: Optional[str] = None,  # 计算设备，None 表示自动选 cuda/cpu
    frame_level: bool = False,  # 是否启用帧级检测：True 时输出损坏帧区间，False 时只输出整段是否损坏
    smooth_sigma: float = 0,  # 时序平滑 sigma，>0 启用，推荐 2~3
    merge_gap: int = 0,       # 区间合并：间隔<=此值的相邻区间合并，推荐 2~5
    min_interval_len: int = 0,  # 短区间过滤：长度<此值的区间丢弃，推荐 2~4
    **dataset_kwargs,          # 其他参数会传给 create_dataloader（如 motion_type, unit_length, min_length, recursive）
) -> None:
    """
    批量检测：遍历 motion 文件，计算误差，写入 CSV。

    流程：加载模型 -> 创建 DataLoader -> 逐 batch 计算误差 -> 写入 CSV。
    frame_level=False：每行记录 文件名、误差值、是否损坏（True/False）。
    frame_level=True：每行记录 文件名、损坏帧区间（如 [1,17],[22,78]）、整段 mean_error（可选）。
    """
    # 加载预训练 FSQ-VQ-VAE 模型，返回 (net, args)，这里只用 net
    net, _ = load_detector(ckpt_path, device=device)
    # 创建数据加载器：从 motion_dir 扫描或按 file_list_path 读取，返回 (loader, dataset)
    loader, dataset = create_dataloader(
        motion_dir=motion_dir,
        file_list_path=file_list_path,
        batch_size=batch_size,
        num_workers=0,      # 0 表示主进程加载，避免多进程与 CUDA 冲突
        shuffle=False,      # 不打乱顺序，便于结果与文件名一一对应
        **dataset_kwargs,
    )

    # 确保输出 CSV 所在目录存在，os.path.dirname(output_csv) 可能为空则用 "."
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    # 以写模式打开 CSV，newline="" 避免 Windows 下多空行，encoding="utf-8" 支持中文
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if frame_level:
            # 帧级模式：表头为 文件名、损坏帧区间、整段平均误差
            writer.writerow(["name", "corrupt_intervals", "mean_error"])
        else:
            # 整段模式：根据 metric 选择误差计算函数
            if metric == "recon":
                compute_fn = compute_reconstruction_error
            else:
                compute_fn = compute_quantization_error
            # 表头：文件名、误差值、是否损坏
            writer.writerow(["name", "error", "corrupt"])

        # 遍历每个 batch，tqdm 显示进度条
        for batch in tqdm(loader, desc="检测中"):
            # batch 为 (motion, names)：motion 形状 (B, T, 272)，names 为文件名列表或单个字符串
            motion, names = batch
            # 统一 names 为列表，便于按索引取
            if isinstance(names, (list, tuple)):
                names = list(names)
            else:
                names = [names]
            # 遍历 batch 内每条样本（batch_size=1 时只循环一次）
            for i in range(motion.shape[0]):
                # 取出第 i 条动作，保持 (1, T, 272) 形状以符合模型输入
                m = motion[i : i + 1]
                # 对应文件名，若 names 长度不足则用索引代替
                name = names[i] if i < len(names) else str(i)
                if frame_level:
                    # 帧级检测：返回 (每帧误差, 损坏掩码, 区间字符串)
                    # 这里只用 intervals_str，如 "[1,17],[22,78]" 或 "[]"
                    _, _, intervals_str = detect_corrupt_per_frame(
                        net, m, threshold, metric=metric, device=None,
                        smooth_sigma=smooth_sigma,
                        merge_gap=merge_gap,
                        min_interval_len=min_interval_len,
                    )
                    # 计算整段平均误差（用于 CSV 中展示）
                    mean_err = compute_reconstruction_error(
                        net, m, reduction="mean"
                    ) if metric == "recon" else compute_quantization_error(
                        net, m, reduction="mean"
                    )
                    # 转为 Python float（兼容 Tensor）
                    mean_val = mean_err.item() if hasattr(mean_err, "item") else float(mean_err)
                    writer.writerow([name, intervals_str, f"{mean_val:.6f}"])
                else:
                    # 整段检测：计算整段平均误差
                    err = compute_fn(net, m, reduction="mean")
                    err_val = err.item() if hasattr(err, "item") else float(err)
                    # 误差超过阈值则判为损坏
                    corrupt = err_val > threshold
                    writer.writerow([name, f"{err_val:.6f}", corrupt])
# endregion


# region 命令行入口
def main() -> None:
    """命令行入口：解析参数，校验路径，调用 run_batch_detect 执行批量检测。"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="批量动作序列损坏检测")
    # 必选：运动文件根目录
    parser.add_argument(
        "--motion-dir",
        type=str,
        required=True,
        help="运动文件根目录，如 dataset/MotionMillion/motion_data/vector_272",
    )
    # 输出 CSV 路径，默认当前目录下的 detect_results.csv
    parser.add_argument(
        "--output-csv",
        type=str,
        default="detect_results.csv",
        help="输出 CSV 路径",
    )
    # 预训练模型权重路径
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/pretrained_models/fsq_net_6000000.pth",
        help="模型检查点路径",
    )
    # 判定阈值：误差超过此值则判为损坏
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="判定阈值，误差超过则判为损坏",
    )
    # 检测指标：recon=重构误差（motion vs rec），quant=量化误差（z vs codes）
    parser.add_argument(
        "--metric",
        choices=["recon", "quant"],
        default="recon",
        help="检测指标：recon=重构误差，quant=量化误差",
    )
    # 可选：指定 txt 文件，每行一个相对 motion-dir 的文件名；不提供则扫描 motion_dir 下所有 .npy
    parser.add_argument(
        "--file-list",
        type=str,
        default=None,
        help="文件列表 txt 路径，每行一个文件名（相对 motion-dir）。不提供则扫描目录",
    )
    # 批大小，建议 1（序列长度不一，pad 后 batch>1 也可工作）
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批大小，建议 1（序列长度可变）",
    )
    # 计算设备，None 时自动选 cuda（有 GPU）或 cpu
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备，默认自动选择 cuda/cpu",
    )
    # 运动类型，用于确定 mean/std 路径（如 dataset/MotionMillion/mean_std/vector_272/）
    parser.add_argument(
        "--motion-type",
        type=str,
        default="vector_272",
        help="运动类型，用于 mean/std 路径",
    )
    # 长度对齐单位：序列长度会对齐到 unit_length 的整数倍（与 Encoder 的 stride_t 相关）
    parser.add_argument(
        "--unit-length",
        type=int,
        default=2,
        help="长度对齐单位（2^down_t）",
    )
    # 最小序列长度，低于此长度的文件会被跳过
    parser.add_argument(
        "--min-length",
        type=int,
        default=64,
        help="最小序列长度，过短则跳过",
    )
    # store_true：不加此参数时为 False，加了 --no-recursive 则为 True
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="目录扫描时不递归子目录",
    )
    # 启用帧级检测时，输出损坏帧区间而非仅整段是否损坏
    parser.add_argument(
        "--frame-level",
        action="store_true",
        help="启用帧级检测，输出损坏帧区间（如 [1,17],[22,78]）",
    )
    # 精度优化参数（帧级检测时生效）
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0,
        help="时序平滑 sigma，>0 时对 err_per_frame 做高斯平滑，推荐 2~3，0=关闭",
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=0,
        help="区间合并：间隔<=此值的相邻区间合并，推荐 2~5，0=关闭",
    )
    parser.add_argument(
        "--min-interval-len",
        type=int,
        default=0,
        help="短区间过滤：长度<此值的区间丢弃，推荐 2~4，0=关闭",
    )
    # 解析命令行，得到 args 对象
    args = parser.parse_args()

    # 校验：模型文件必须存在
    if not os.path.exists(args.ckpt):
        print(f"错误：检查点不存在 {args.ckpt}")
        print("请先下载或训练得到 fsq_net_6000000.pth，参见 ENV_CONFIG.md")
        sys.exit(1)

    # 校验：运动目录必须存在
    if not os.path.isdir(args.motion_dir):
        print(f"错误：运动目录不存在 {args.motion_dir}")
        sys.exit(1)

    # 调用批量检测函数，将 args 中的参数传入
    # recursive=not args.no_recursive：默认递归扫描子目录，加 --no-recursive 则不递归
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
        smooth_sigma=args.smooth_sigma,
        merge_gap=args.merge_gap,
        min_interval_len=args.min_interval_len,
        motion_type=args.motion_type,
        unit_length=args.unit_length,
        min_length=args.min_length,
        recursive=not args.no_recursive,
    )
    print(f"检测完成，结果已保存到 {args.output_csv}")
# endregion


# 当直接运行此脚本时（python run_detect.py ...）执行 main
if __name__ == "__main__":
    main()
