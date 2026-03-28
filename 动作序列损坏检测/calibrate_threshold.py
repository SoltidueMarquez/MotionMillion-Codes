"""
动作序列损坏检测 - 阈值标定与评估

在已知完好与已知损坏的样本上运行检测，统计误差分布，
绘制 ROC/Precision-Recall 曲线，输出推荐阈值及 AUC、F1 等指标。

对接 detect_corrupt_utils 与 dataset_corrupt_detection，复用检测逻辑。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

# region 路径配置
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
    compute_quantization_error_per_frame,
    compute_reconstruction_error,
    compute_reconstruction_error_per_frame,
    compute_reconstruction_error_per_part,
    load_detector,
    visualize_motion_overlay_vector272,
)


# region 误差收集
def collect_part_errors_analysis(
    motion_dir: str,
    good_list_path: str,
    ckpt_path: str,
    device: Optional[str] = None,
    visualize_num: int = 0,
    vis_output_dir: Optional[str] = None,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    pin_to_origin: bool = False,
    **dataset_kwargs,
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    量化分析：计算每个部位的重构误差。
    返回: (total_frame_errors, per_sequence_part_errors, filenames)
    """
    net, _ = load_detector(ckpt_path, device=device)

    loader, dataset = create_dataloader(
        motion_dir=motion_dir,
        file_list_path=good_list_path,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        **dataset_kwargs,
    )

    per_sequence_part_errors = []
    filenames = []

    if mean is None or std is None:
        mean, std = dataset.mean, dataset.std

    sample_idx = 0
    for batch in tqdm(loader, desc="正在分析各部位 FSQ 重建损坏 (仅完好样本)"):
        motion, name_list = batch
        # motion: (1, T, 272)
        name = name_list[0]
        
        # 计算每帧每个部位的误差
        # (T, 7)
        err_parts = compute_reconstruction_error_per_part(
            net, motion, pin_to_origin=pin_to_origin, mean=mean, std=std
        )
        err_np = err_parts.cpu().numpy()
        per_sequence_part_errors.append(err_np)
        filenames.append(name)

        # 可视化重叠动作视频
        if visualize_num > 0 and sample_idx < visualize_num:
            dev = next(net.parameters()).device
            with torch.no_grad():
                rec, _, _, _, _ = net(motion.to(dev))
            
            # 如果开启原地化，反标准化前进行处理
            if pin_to_origin:
                from detect_corrupt_utils import pin_motion_to_origin
                motion_pinned = pin_motion_to_origin(motion.clone(), mean, std)
                rec_pinned = pin_motion_to_origin(rec.clone(), mean, std)
            else:
                motion_pinned = motion
                rec_pinned = rec

            # 反标准化
            motion_np = motion_pinned.squeeze(0).cpu().numpy()
            rec_np = rec_pinned.squeeze(0).cpu().numpy()
            motion_denorm = motion_np * std + mean
            rec_denorm = rec_np * std + mean
            
            out_name = f"{name}_overlay.mp4"
            out_path = os.path.join(vis_output_dir or ".", "overlay_viz", out_name)
            
            visualize_motion_overlay_vector272(
                motion_denorm, rec_denorm, out_path,
                per_frame_errors=err_np, fps=30
            )

        sample_idx += 1

    total_frame_errors = np.concatenate(per_sequence_part_errors, axis=0)
    return total_frame_errors, per_sequence_part_errors, filenames


def collect_errors(
    motion_dir: str,
    good_list_path: str,
    corrupt_list_path: str,
    ckpt_path: str,
    metric: str = "recon",
    device: Optional[str] = None,
    pin_to_origin: bool = False,
    **dataset_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在完好与损坏样本上分别计算误差，返回 (errors, labels)。

    labels: 0=完好, 1=损坏
    """
    net, _ = load_detector(ckpt_path, device=device)
    if metric == "recon":
        compute_fn = compute_reconstruction_error
    else:
        compute_fn = compute_quantization_error

    def _load_file_list(path: str) -> Optional[List[str]]:
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    good_list = _load_file_list(good_list_path)
    corrupt_list = _load_file_list(corrupt_list_path)
    if not good_list or not corrupt_list:
        raise ValueError("good-list 与 corrupt-list 均需提供且文件存在")

    errors_list: List[float] = []
    labels_list: List[int] = []

    # 获取 dataset 实例以取得 mean/std (对于 pin_to_origin 必选)
    temp_loader, temp_dataset = create_dataloader(
        motion_dir=motion_dir,
        file_list_path=good_list_path,
        batch_size=1,
        num_workers=0,
        **dataset_kwargs,
    )
    mean, std = temp_dataset.mean, temp_dataset.std

    for file_list_path, label in [(good_list_path, 0), (corrupt_list_path, 1)]:
        loader, _ = create_dataloader(
            motion_dir=motion_dir,
            file_list_path=file_list_path,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            **dataset_kwargs,
        )
        for batch in tqdm(loader, desc=f"采集 {'完好' if label == 0 else '损坏'} 样本误差"):
            motion, _ = batch
            if metric == "recon":
                err = compute_fn(net, motion, reduction="mean", pin_to_origin=pin_to_origin, mean=mean, std=std)
            else:
                err = compute_fn(net, motion, reduction="mean")
            err_val = err.item() if hasattr(err, "item") else float(err)
            errors_list.append(err_val)
            labels_list.append(label)

    return np.array(errors_list), np.array(labels_list)


def collect_good_only_errors(
    motion_dir: str,
    good_list_path: str,
    ckpt_path: str,
    metric: str = "recon",
    frame_level: bool = False,
    device: Optional[str] = None,
    pin_to_origin: bool = False,
    **dataset_kwargs,
) -> Tuple[np.ndarray, float]:
    """
    仅在完好样本上计算误差，返回 (per_sample_errors, max_error)。

    用于「完好视频零误报」标定：阈值取 max_error 可确保完好样本不会被判为损坏。
    frame_level=True 时采集每帧误差的 max，与 run_detect --frame-level 一致。
    """
    net, _ = load_detector(ckpt_path, device=device)

    def _load_file_list(path: str) -> Optional[List[str]]:
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    good_list = _load_file_list(good_list_path)
    if not good_list:
        raise ValueError("good-list 必须提供且文件存在")

    loader, dataset = create_dataloader(
        motion_dir=motion_dir,
        file_list_path=good_list_path,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        **dataset_kwargs,
    )
    mean, std = dataset.mean, dataset.std

    all_errors: List[float] = []
    per_sample_max: List[float] = []

    for batch in tqdm(loader, desc="采集完好样本误差（good-only 标定）"):
        motion, _ = batch
        if frame_level:
            if metric == "recon":
                err_tensor = compute_reconstruction_error_per_frame(
                    net, motion, pin_to_origin=pin_to_origin, mean=mean, std=std
                )
            else:
                err_tensor = compute_quantization_error_per_frame(net, motion)
            err_np = err_tensor.cpu().numpy().ravel()
            all_errors.extend(err_np.tolist())
            per_sample_max.append(float(np.max(err_np)))
        else:
            if metric == "recon":
                err = compute_reconstruction_error(
                    net, motion, reduction="mean", pin_to_origin=pin_to_origin, mean=mean, std=std
                )
            else:
                err = compute_quantization_error(net, motion, reduction="mean")
            err_val = err.item() if hasattr(err, "item") else float(err)
            all_errors.append(err_val)
            per_sample_max.append(err_val)

    max_error = float(np.max(all_errors))
    return np.array(per_sample_max), max_error
# endregion


# region 指标计算
def compute_roc_metrics(
    errors: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    计算 ROC 曲线所需指标。

    Returns:
        thresholds: 阈值序列
        tpr: True Positive Rate
        fpr: False Positive Rate
        auc: AUC 值
    """
    # 按误差升序排列，阈值取相邻误差中点
    order = np.argsort(errors)
    sorted_errors = errors[order]
    sorted_labels = labels[order]
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([]), np.array([]), 0.0

    # 阈值：在排序后误差之间插入（含端点）
    thresholds = np.unique(sorted_errors)
    thresholds = np.concatenate([[thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]])

    tpr_list = []
    fpr_list = []
    for th in thresholds:
        pred = (errors > th).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    # AUC: 梯形积分。需按 fpr 升序排列，否则 np.trapezoid 会因 x 递减而得负值
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    # np.trapz 在 NumPy 2.0 中被弃用并由 np.trapezoid 取代。
    # 这里我们优先使用 np.trapezoid 以适应新版 NumPy。
    if hasattr(np, "trapezoid"):
        auc_val = np.trapezoid(tpr, fpr)
    elif hasattr(np, "trapz"):
        auc_val = getattr(np, "trapz")(tpr, fpr)
    else:
        # 手动计算梯形积分作为回退
        auc_val = np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2.0)
    
    auc = float(auc_val)

    return thresholds, tpr, fpr, auc


def find_best_f1_threshold(
    errors: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """找到使 F1 最大的阈值及对应 F1"""
    thresholds = np.unique(errors)
    best_f1 = 0.0
    best_th = thresholds[0] if len(thresholds) > 0 else 0.0

    for th in thresholds:
        pred = (errors > th).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1
# endregion


# region 绘图与输出
def save_roc_plot(
    errors: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """绘制 ROC 曲线并保存"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 避免中文乱码：优先使用支持 CJK 的字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("警告：未安装 matplotlib，跳过绘图")
        return

    thresholds, tpr, fpr, auc = compute_roc_metrics(errors, labels)
    if len(thresholds) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC 曲线
    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 误差分布直方图
    good_errors = errors[labels == 0]
    corrupt_errors = errors[labels == 1]
    axes[1].hist(good_errors, bins=30, alpha=0.6, label="完好", color="green", density=True)
    axes[1].hist(corrupt_errors, bins=30, alpha=0.6, label="损坏", color="red", density=True)
    axes[1].set_xlabel("Error")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC 曲线已保存到 {output_path}")
# endregion


# region 主流程
def main() -> None:
    parser = argparse.ArgumentParser(
        description="阈值标定：仅 good-list 时取完好样本最大误差（零误报）；提供 corrupt-list 时用 ROC/F1 评估"
    )
    parser.add_argument("--motion-dir", type=str, required=True, help="运动文件根目录")
    parser.add_argument("--good-list", type=str, required=True, help="完好样本文件列表 txt")
    parser.add_argument(
        "--corrupt-list",
        type=str,
        default=None,
        help="损坏样本文件列表 txt；不提供则走 good-only 标定（完好零误报）",
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/pretrained_models/fsq_net_6000000.pth")
    parser.add_argument("--metric", choices=["recon", "quant"], default="recon")
    parser.add_argument(
        "--frame-level",
        action="store_true",
        help="帧级标定，与 run_detect --frame-level 一致；不指定则为序列级",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="阈值安全余量，threshold = max_error + margin，默认 0",
    )
    parser.add_argument(
        "--analyze-parts",
        action="store_true",
        help="量化各部位重建损坏（仅 good-list 模式），记录每帧各部位误差",
    )
    parser.add_argument(
        "--visualize-num",
        type=int,
        default=0,
        help="仅对前 N 个样本做可视化，0 表示不可视化",
    )
    parser.add_argument(
        "--pin-to-origin",
        action="store_true",
        help="开启原地化分析：在计算重构误差和可视化时清除根节点位移和朝向变化",
    )
    parser.add_argument("--output-plot", type=str, default="动作序列损坏检测/calibrate_roc.png")
    parser.add_argument("--output-csv", type=str, default=None, help="保存每样本误差与标签供调试")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--motion-type", type=str, default="vector_272")
    parser.add_argument("--unit-length", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"错误：检查点不存在 {args.ckpt}")
        sys.exit(1)

    dataset_kwargs = {
        "motion_type": args.motion_type,
        "unit_length": args.unit_length,
        "min_length": args.min_length,
    }

    if not args.corrupt_list:
        if args.analyze_parts:
            # 量化分析：计算每个部位的重建损坏，记录详细数据
            print("正在进行各部位重建损坏量化分析（仅完好样本）...")
            vis_dir = os.path.dirname(args.output_csv) if args.output_csv else None
            total_frame_errors, per_seq_errors, filenames = collect_part_errors_analysis(
                motion_dir=args.motion_dir,
                good_list_path=args.good_list,
                ckpt_path=args.ckpt,
                device=args.device,
                visualize_num=args.visualize_num,
                vis_output_dir=vis_dir,
                pin_to_origin=args.pin_to_origin,
                **dataset_kwargs,
            )

            part_names = ["h1", "h2L", "h2R", "h3L", "h3R", "h4", "h"]
            avg_per_part = total_frame_errors.mean(axis=0)

            print("\n" + "=" * 50)
            print("各部位 FSQ 重建损坏量化分析结果")
            print("=" * 50)
            for i, name in enumerate(part_names):
                print(f"{name}: mean={avg_per_part[i]:.6f}, std={total_frame_errors[:, i].std():.6f}")
            print("=" * 50)

            if args.output_csv:
                os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
                with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    # 表头: [file_name, frame_idx, h1, h2L, h2R, h3L, h3R, h4, h]
                    w.writerow(["file_name", "frame_idx"] + part_names)
                    for seq_err, fname in zip(per_seq_errors, filenames):
                        for t in range(len(seq_err)):
                            row = [fname, t] + [f"{e:.6f}" for e in seq_err[t]]
                            w.writerow(row)
                print(f"详细部位误差数据已保存到 {args.output_csv}")
            return

        # good-only 标定：取完好样本最大误差，确保完好视频零误报
        print("正在采集完好样本误差（good-only 标定，目标：完好视频零误报）...")
        per_sample_errors, max_error = collect_good_only_errors(
            motion_dir=args.motion_dir,
            good_list_path=args.good_list,
            ckpt_path=args.ckpt,
            metric=args.metric,
            frame_level=args.frame_level,
            device=args.device,
            pin_to_origin=args.pin_to_origin,
            **dataset_kwargs,
        )
        n_good = len(per_sample_errors)
        recommended_th = max_error + args.margin

        print(f"完好样本数: {n_good}")

        print("\n" + "=" * 50)
        print("标定结果（good-only）")
        print("=" * 50)
        print(f"metric: {args.metric}")
        print(f"误差粒度: {'帧级' if args.frame_level else '序列级'}")
        print(f"完好样本误差: mean={per_sample_errors.mean():.6f}, std={per_sample_errors.std():.6f}")
        print(f"最大误差: {max_error:.6f}")
        print(f"推荐阈值: {recommended_th:.6f}" + (f" (max + margin={args.margin})" if args.margin else ""))
        print(f"建议 run_detect 使用: --threshold {recommended_th:.6f}")
        if args.frame_level:
            print("（请同时使用 run_detect --frame-level）")
        print("=" * 50)

        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
            with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["error", "label"])
                for e in per_sample_errors:
                    w.writerow([f"{e:.6f}", 0])
            print(f"每样本误差已保存到 {args.output_csv}")
    else:
        # 双列表 ROC/F1 模式（兼容旧用法）
        print("正在采集完好与损坏样本的误差...")
        errors, labels = collect_errors(
            motion_dir=args.motion_dir,
            good_list_path=args.good_list,
            corrupt_list_path=args.corrupt_list,
            ckpt_path=args.ckpt,
            metric=args.metric,
            device=args.device,
            pin_to_origin=args.pin_to_origin,
            **dataset_kwargs,
        )

        n_good = (labels == 0).sum()
        n_corrupt = (labels == 1).sum()
        print(f"完好样本数: {n_good}, 损坏样本数: {n_corrupt}")

        if n_good == 0 or n_corrupt == 0:
            print("错误：完好与损坏样本均需至少 1 个")
            sys.exit(1)

        thresholds, tpr, fpr, auc = compute_roc_metrics(errors, labels)
        best_th, best_f1 = find_best_f1_threshold(errors, labels)

        print("\n" + "=" * 50)
        print("标定结果（ROC/F1）")
        print("=" * 50)
        print(f"metric: {args.metric}")
        print(f"完好样本误差: mean={errors[labels==0].mean():.6f}, std={errors[labels==0].std():.6f}")
        print(f"损坏样本误差: mean={errors[labels==1].mean():.6f}, std={errors[labels==1].std():.6f}")
        print(f"AUC: {auc:.4f}")
        print(f"推荐阈值（F1 最大）: {best_th:.6f}, F1={best_f1:.4f}")
        print(f"建议 run_detect 使用: --threshold {best_th:.6f}")
        print("=" * 50)

        if args.output_plot:
            save_roc_plot(errors, labels, args.output_plot)

        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
            with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["error", "label"])
                for e, l in zip(errors, labels):
                    w.writerow([f"{e:.6f}", l])
            print(f"每样本误差已保存到 {args.output_csv}")
# endregion


if __name__ == "__main__":
    main()
