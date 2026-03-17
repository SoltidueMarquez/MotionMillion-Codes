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
    compute_reconstruction_error,
    load_detector,
)


# region 误差收集
def collect_errors(
    motion_dir: str,
    good_list_path: str,
    corrupt_list_path: str,
    ckpt_path: str,
    metric: str = "recon",
    device: Optional[str] = None,
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
            err = compute_fn(net, motion, reduction="mean")
            err_val = err.item() if hasattr(err, "item") else float(err)
            errors_list.append(err_val)
            labels_list.append(label)

    return np.array(errors_list), np.array(labels_list)
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

    # AUC: 梯形积分。需按 fpr 升序排列，否则 np.trapz 会因 x 递减而得负值
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    auc = np.trapz(tpr, fpr)

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
    parser = argparse.ArgumentParser(description="阈值标定：在已知完好/损坏样本上评估，输出推荐阈值")
    parser.add_argument("--motion-dir", type=str, required=True, help="运动文件根目录")
    parser.add_argument("--good-list", type=str, required=True, help="完好样本文件列表 txt")

    parser.add_argument("--corrupt-list", type=str, required=True, help="损坏样本文件列表 txt")
    parser.add_argument("--ckpt", type=str, default="checkpoints/pretrained_models/fsq_net_6000000.pth")
    parser.add_argument("--metric", choices=["recon", "quant"], default="recon")
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

    print("正在采集完好与损坏样本的误差...")
    errors, labels = collect_errors(
        motion_dir=args.motion_dir,
        good_list_path=args.good_list,
        corrupt_list_path=args.corrupt_list,
        ckpt_path=args.ckpt,
        metric=args.metric,
        device=args.device,
        motion_type=args.motion_type,
        unit_length=args.unit_length,
        min_length=args.min_length,
    )

    n_good = (labels == 0).sum()
    n_corrupt = (labels == 1).sum()
    print(f"完好样本数: {n_good}, 损坏样本数: {n_corrupt}")

    if n_good == 0 or n_corrupt == 0:
        print("错误：完好与损坏样本均需至少 1 个")
        sys.exit(1)

    # 统计
    thresholds, tpr, fpr, auc = compute_roc_metrics(errors, labels)
    best_th, best_f1 = find_best_f1_threshold(errors, labels)

    print("\n" + "=" * 50)
    print("标定结果")
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
