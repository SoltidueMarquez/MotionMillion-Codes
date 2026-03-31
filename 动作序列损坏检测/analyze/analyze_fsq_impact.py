"""
动作序列损坏检测 - FSQ 动作重建误差对损坏检测影响的量化分析

对比 corrupt_list 与 good_list 的检测结果，结合 GT，量化分析 FSQ 重建误差
对损坏检测精度的影响程度。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# region 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
_FEATURE_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _FEATURE_ROOT.parent

for _p in (_REPO_ROOT, _FEATURE_ROOT, _SCRIPT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
# endregion

import numpy as np

from evaluate_detect import load_gt_csv, load_detect_csv, parse_intervals_to_mask


def _good_name_from_corrupt(corrupt_name: str) -> str:
    """从 corrupt 名称推导对应的 good 名称。如 000000_corrupt -> 000000"""
    if corrupt_name.endswith("_corrupt"):
        return corrupt_name[:-8]  # 移除 "_corrupt"
    return corrupt_name


def analyze_fsq_impact(
    gt_csv_path: str,
    good_detect_csv_path: str,
    corrupt_detect_csv_path: str,
    output_path: Optional[str] = None,
    output_detail_csv: Optional[str] = None,
) -> Dict:
    """
    量化分析 FSQ 动作重建误差对损坏检测的影响。

    Returns:
        包含各指标的字典
    """
    gt_data = load_gt_csv(gt_csv_path)
    good_detect = load_detect_csv(good_detect_csv_path)
    corrupt_detect = load_detect_csv(corrupt_detect_csv_path)

    # 建立对齐：GT/corrupt 使用 corrupt_name，good 使用 good_name
    # 仅分析 GT 中有的 corrupt 样本，且 good/corrupt 检测结果均存在的
    corrupt_names = set(gt_data.keys()) & set(corrupt_detect.keys())
    if not corrupt_names:
        raise ValueError(
            "GT 与 corrupt 检测 CSV 无共同文件。"
            f"GT 样本数: {len(gt_data)}, corrupt 检测样本数: {len(corrupt_detect)}"
        )

    good_names = {_good_name_from_corrupt(c) for c in corrupt_names}
    missing_good = good_names - set(good_detect.keys())
    if missing_good:
        raise ValueError(
            f"good 检测 CSV 中缺少 {len(missing_good)} 个对应样本，例如: {list(missing_good)[:3]}"
        )

    # 1. good_list：FSQ 诱导误报
    fp_good_total = 0
    total_good_frames = 0
    per_sample_fp_good: List[int] = []
    per_sample_frames: List[int] = []

    # 2. corrupt_list：TP/FP/FN/TN
    tp_total = fp_corrupt_total = fn_total = tn_total = 0

    # 3. 序列级对齐（用于相关系数）
    fp_good_per_seq: List[int] = []
    fp_corrupt_per_seq: List[int] = []

    eps = 1e-8

    for corrupt_name in sorted(corrupt_names):
        good_name = _good_name_from_corrupt(corrupt_name)
        gt_intervals, seq_len = gt_data[corrupt_name]
        pred_good = good_detect[good_name]
        pred_corrupt = corrupt_detect[corrupt_name]

        gt_mask = parse_intervals_to_mask(gt_intervals, seq_len)
        pred_good_mask = parse_intervals_to_mask(pred_good, seq_len)
        pred_corrupt_mask = parse_intervals_to_mask(pred_corrupt, seq_len)

        # good：完好数据上任何检测均为误报（GT 视为全 0）
        fp_good = int(pred_good_mask.sum())
        fp_good_total += fp_good
        total_good_frames += seq_len
        per_sample_fp_good.append(fp_good)
        per_sample_frames.append(seq_len)

        # corrupt：与 GT 对比
        tp_total += int((gt_mask & pred_corrupt_mask).sum())
        fp_corrupt = int((~gt_mask & pred_corrupt_mask).sum())
        fp_corrupt_total += fp_corrupt
        fn_total += int((gt_mask & ~pred_corrupt_mask).sum())
        tn_total += int((~gt_mask & ~pred_corrupt_mask).sum())

        fp_good_per_seq.append(fp_good)
        fp_corrupt_per_seq.append(fp_corrupt)

    n_samples = len(corrupt_names)
    fsq_fpr = fp_good_total / (total_good_frames + eps)

    precision_corrupt = tp_total / (tp_total + fp_corrupt_total + eps)
    recall_corrupt = tp_total / (tp_total + fn_total + eps)
    iou_corrupt = tp_total / (tp_total + fp_corrupt_total + fn_total + eps)

    # FSQ attribution：FP_good 占 corrupt FP 的比例（下界）
    fsq_attribution = fp_good_total / (fp_corrupt_total + eps)

    # 若剔除 FSQ 基线误报后的 Precision 潜力
    fp_corrupt_remaining = max(0, fp_corrupt_total - fp_good_total)
    precision_potential = tp_total / (tp_total + fp_corrupt_remaining + eps)
    precision_improvement = (precision_potential - precision_corrupt) / (precision_corrupt + eps)

    # Spearman 相关系数（good FP vs corrupt FP，按序列）
    fp_good_arr = np.array(fp_good_per_seq)
    fp_corrupt_arr = np.array(fp_corrupt_per_seq)
    if len(fp_good_arr) > 1 and np.std(fp_good_arr) > 0 and np.std(fp_corrupt_arr) > 0:
        from scipy import stats

        spearman_r, _ = stats.spearmanr(fp_good_arr, fp_corrupt_arr)
        spearman_r = float(spearman_r) if not np.isnan(spearman_r) else 0.0
    else:
        spearman_r = 0.0

    results = {
        "n_samples": n_samples,
        "fp_good_total": fp_good_total,
        "total_good_frames": total_good_frames,
        "fsq_fpr": fsq_fpr,
        "tp": tp_total,
        "fp_corrupt": fp_corrupt_total,
        "fn": fn_total,
        "tn": tn_total,
        "precision": precision_corrupt,
        "recall": recall_corrupt,
        "iou": iou_corrupt,
        "fsq_attribution": fsq_attribution,
        "precision_potential": precision_potential,
        "precision_improvement": precision_improvement,
        "spearman_r": spearman_r,
        "per_sample_fp_good": per_sample_fp_good,
        "per_sample_frames": per_sample_frames,
        "corrupt_names": sorted(corrupt_names),
    }

    # 输出报告
    lines = [
        "======== FSQ 动作重建误差对损坏检测的影响分析 ========",
        "",
        "1. good_list（完好数据）— FSQ 诱导误报",
        f"   - 样本数: {n_samples}",
        f"   - 总帧数: {total_good_frames}",
        f"   - 误报帧数: {fp_good_total}",
        f"   - FSQ 帧级误报率: {fsq_fpr * 100:.2f}%",
        "",
        "2. corrupt_list（损坏数据）— 现有评估",
        f"   - TP={tp_total} FP={fp_corrupt_total} FN={fn_total} TN={tn_total}",
        f"   - Precision={precision_corrupt:.4f} Recall={recall_corrupt:.4f} IoU={iou_corrupt:.4f}",
        "",
        "3. FSQ 影响估计",
        f"   - FP 中至少可归因 FSQ 的比例（下界）: {fsq_attribution * 100:.1f}%",
        f"   - 若剔除 FSQ 基线误报，Precision 潜力: {precision_potential:.4f}",
        f"   - Precision 相对提升潜力: {precision_improvement * 100:.1f}%",
        "",
        "4. 序列级关联",
        f"   - good FP 与 corrupt FP 的 Spearman 相关系数: {spearman_r:.3f}",
        "==================================================",
    ]
    text = "\n".join(lines)
    print(text)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n结果已保存到 {output_path}")

    if output_detail_csv:
        os.makedirs(os.path.dirname(output_detail_csv) or ".", exist_ok=True)
        import csv

        with open(output_detail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["corrupt_name", "good_name", "seq_len", "fp_good", "fp_corrupt", "fp_good_rate"]
            )
            for i, corrupt_name in enumerate(sorted(corrupt_names)):
                good_name = _good_name_from_corrupt(corrupt_name)
                sl = per_sample_frames[i]
                fpg = per_sample_fp_good[i]
                fpc = fp_corrupt_per_seq[i]
                rate = fpg / (sl + eps) * 100
                writer.writerow([corrupt_name, good_name, sl, fpg, fpc, f"{rate:.2f}%"])
        print(f"序列级明细已保存到 {output_detail_csv}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="量化分析 FSQ 动作重建误差对损坏检测的影响"
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        required=True,
        help="ground_truth_intervals.csv 路径",
    )
    parser.add_argument(
        "--good-detect-csv",
        type=str,
        required=True,
        help="good_list 检测结果 CSV 路径（threshold-good/detect_results_corrupt.csv）",
    )
    parser.add_argument(
        "--corrupt-detect-csv",
        type=str,
        required=True,
        help="corrupt_list 检测结果 CSV 路径（threshold/detect_results_corrupt.csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选，汇总报告输出文件路径",
    )
    parser.add_argument(
        "--output-detail-csv",
        type=str,
        default=None,
        help="可选，序列级明细 CSV 输出路径",
    )
    args = parser.parse_args()

    if not os.path.exists(args.gt_csv):
        print(f"错误：GT CSV 不存在 {args.gt_csv}")
        sys.exit(1)
    if not os.path.exists(args.good_detect_csv):
        print(f"错误：good 检测 CSV 不存在 {args.good_detect_csv}")
        sys.exit(1)
    if not os.path.exists(args.corrupt_detect_csv):
        print(f"错误：corrupt 检测 CSV 不存在 {args.corrupt_detect_csv}")
        sys.exit(1)

    analyze_fsq_impact(
        gt_csv_path=args.gt_csv,
        good_detect_csv_path=args.good_detect_csv,
        corrupt_detect_csv_path=args.corrupt_detect_csv,
        output_path=args.output,
        output_detail_csv=args.output_detail_csv,
    )


if __name__ == "__main__":
    main()
