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
from typing import Dict, Literal, Optional, Tuple  # 可选类型，如 Optional[str] 表示 str 或 None

# region 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
_FEATURE_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _FEATURE_ROOT.parent
_DETECT_DIR = _FEATURE_ROOT / "detect"
_ANALYZE_DIR = _FEATURE_ROOT / "analyze"

# 既要能导入仓库顶层模块，也要兼容 detect/analyze 跨目录平铺导入。
for _p in (_REPO_ROOT, _FEATURE_ROOT, _DETECT_DIR, _ANALYZE_DIR, _SCRIPT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
# endregion

import torch  # type: ignore[import-untyped]
from tqdm import tqdm  # 进度条显示

# 从同目录下的模块导入：数据加载器、检测工具函数
from dataset_corrupt_detection import create_dataloader
from detect_corrupt_utils import (
    compute_quantization_error,   # 计算量化误差
    compute_reconstruction_error, # 计算重构误差
    compute_reconstruction_error_per_part, # 各部位重构误差分析
    detect_corrupt_per_frame,     # 帧级检测，返回损坏区间
    load_detector,                # 加载预训练 FSQ-VQ-VAE 模型
    save_detection_visualizations,  # 保存输入与重建动作视频
)
from analyze.evaluate_detect import load_gt_csv, parse_intervals_to_mask


# region 批量检测逻辑
def run_batch_detect(
    motion_dir: str,           # 运动文件根目录，如 dataset/MotionMillion/motion_data/vector_272
    output_csv: str,           # 输出 CSV 文件路径
    ckpt_path: str,            # 预训练模型权重路径（fsq_net_6000000.pth）
    threshold: float = 0.1,    # 判定阈值：误差超过此值则判为损坏
    metric: Literal["recon", "quant"] = "recon",     # 检测指标：'recon'=重构误差，'quant'=量化误差
    file_list_path: Optional[str] = None,  # 可选：指定 txt 文件，每行一个相对路径，不提供则扫描 motion_dir
    batch_size: int = 1,       # 批大小，建议 1（因为不同序列长度可能不同，pad 后 batch>1 也可）
    device: Optional[str] = None,  # 计算设备，None 表示自动选 cuda/cpu
    frame_level: bool = False,  # 是否启用帧级检测：True 时输出损坏帧区间，False 时只输出整段是否损坏
    output_dir: Optional[str] = None,  # 可视化输出根目录，--visualize-num 时使用
    visualize_num: int = 0,    # 仅对前 N 个样本做可视化，0 表示不可视化
    vis_fps: int = 30,         # 可视化视频帧率
    gt_csv_path: Optional[str] = None,  # GT 标注文件路径，未提供则不做 GT 叠加
    overlay: bool = False,     # 是否采用重叠可视化模式（输入与重建叠加，显示部位误差）
    pin_to_origin: bool = False, # 是否开启原地化检测
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

    # 可视化输出目录：visualize_num > 0 时若未指定 output_dir 则用 output_csv 所在目录
    vis_output_dir = output_dir if output_dir else os.path.dirname(output_csv) or "."
    if visualize_num > 0:
        os.makedirs(vis_output_dir, exist_ok=True)

    # 加载 GT 标注（用于可视化叠加）
    gt_dict: Dict[str, Tuple[str, int]] = {}
    if gt_csv_path and os.path.exists(gt_csv_path):
        gt_dict = load_gt_csv(gt_csv_path)

    # 以写模式打开 CSV，newline="" 避免 Windows 下多空行，encoding="utf-8" 支持中文
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        compute_fn = None
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
        sample_idx = 0
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
                # 去掉 .npy 后缀作为文件夹名
                name_stem = name[:-4] if name.endswith(".npy") else name

                # 可视化：仅对前 visualize_num 个样本保存视频
                if visualize_num > 0 and sample_idx < visualize_num:
                    dev = next(net.parameters()).device
                    with torch.no_grad():
                        rec, _, _, _, _ = net(m.to(dev))
                    
                    # 损坏标记
                    _, corrupt_mask, _ = detect_corrupt_per_frame(
                        net, m, threshold, metric=metric, device=None,
                        pin_to_origin=pin_to_origin, mean=dataset.mean, std=dataset.std
                    )
                    detected_mask = corrupt_mask.cpu().numpy()
                    gt_mask = None
                    gt_intervals_str = None
                    name_norm = name_stem.replace("\\", "/")
                    if name_norm in gt_dict:
                        gt_intervals_str, T = gt_dict[name_norm]
                        gt_mask = parse_intervals_to_mask(gt_intervals_str, T)
                    
                    # 各部位误差（用于叠加模式）
                    err_parts_np = None
                    if overlay:
                        err_parts = compute_reconstruction_error_per_part(
                            net, m, pin_to_origin=pin_to_origin, mean=dataset.mean, std=dataset.std
                        )
                        err_parts_np = err_parts.cpu().numpy()

                    save_detection_visualizations(
                        m, rec, name_stem, vis_output_dir,
                        dataset.mean, dataset.std, fps=vis_fps,
                        gt_corrupt_mask=gt_mask,
                        detected_corrupt_mask=detected_mask,
                        overlay=overlay,
                        per_part_errors=err_parts_np,
                        pin_to_origin=pin_to_origin,
                        gt_intervals_str=gt_intervals_str,
                    )
                sample_idx += 1

                if frame_level:
                    # 帧级检测：返回 (每帧误差, 损坏掩码, 区间字符串)
                    # 这里只用 intervals_str，如 "[1,17],[22,78]" 或 "[]"
                    _, _, intervals_str = detect_corrupt_per_frame(
                        net, m, threshold, metric=metric, device=None,
                        pin_to_origin=pin_to_origin, mean=dataset.mean, std=dataset.std
                    )
                    # 计算整段平均误差（用于 CSV 中展示）
                    mean_err = compute_reconstruction_error(
                        net, m, reduction="mean", pin_to_origin=pin_to_origin, mean=dataset.mean, std=dataset.std
                    ) if metric == "recon" else compute_quantization_error(
                        net, m, reduction="mean"
                    )
                    # 转为 Python float（兼容 Tensor）
                    mean_val = mean_err.item() if hasattr(mean_err, "item") else float(mean_err)
                    writer.writerow([name, intervals_str, f"{mean_val:.6f}"])
                else:
                    # 整段检测：计算整段平均误差
                    err = compute_reconstruction_error(
                        net, m, reduction="mean", pin_to_origin=pin_to_origin, mean=dataset.mean, std=dataset.std
                    ) if metric == "recon" else compute_quantization_error(
                        net, m, reduction="mean"
                    )
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
    # 可视化：保存输入与重建动作视频
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可视化输出根目录，--visualize-num 时使用；未指定则使用 output-csv 所在目录",
    )
    parser.add_argument(
        "--visualize-num",
        type=int,
        default=0,
        help="仅对前 N 个样本做可视化，0 表示不可视化；大于样本总数则全部可视化",
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
        help="ground_truth_intervals.csv 路径（generate_corrupt_data 输出），提供则在视频中叠加 GT 标注",
    )
    # 叠加可视化：输入与重建重叠，并标注各部位误差
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="是否采用重叠可视化模式（输入与重建叠加，显示部位误差）",
    )
    # 原地化检测：去除根节点位移带来的误差
    parser.add_argument(
        "--pin-to-origin",
        action="store_true",
        help="开启原地化检测：去除输入与重建动作序列的根节点位移与朝向变化进行误差计算",
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
        output_dir=args.output,
        visualize_num=args.visualize_num,
        vis_fps=args.vis_fps,
        gt_csv_path=args.gt_csv,
        overlay=args.overlay,
        pin_to_origin=args.pin_to_origin,
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
