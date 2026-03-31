"""
动作序列损坏检测 - 准确性评估脚本

将 run_detect.py --frame-level 输出的检测 CSV 与 generate_corrupt_data.py 输出的
ground_truth_intervals.csv 对比，计算 Precision、Recall、IoU 等指标。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_intervals_to_mask(intervals_str: str, seq_len: int) -> np.ndarray:
    """
    将 1-based 区间字符串解析为帧级 bool mask。

    输入：如 "[1,17],[22,78]" 或 "[1,17,jittering],[22,78,drifting]"
    输出：(seq_len,) bool 数组，1-based 帧号对应 0-based 下标
    """
    mask = np.zeros(seq_len, dtype=bool)
    if not intervals_str or intervals_str.strip() == "[]":
        return mask
    
    # 兼容 [s,e] 和 [s,e,type] 格式
    # (\d+) 是数字，([^\]]+) 是直到 ] 之前的字符（可能是 type）
    pattern = r"\[(\d+),(\d+)(?:,([^\]]+))?\]"
    for m in re.finditer(pattern, intervals_str):
        s, e = int(m.group(1)), int(m.group(2))
        # 1-based 转 0-based
        idx_start = max(0, s - 1)
        idx_end = min(seq_len - 1, e - 1)
        if idx_start <= idx_end:
            mask[idx_start : idx_end + 1] = True
    return mask


def parse_intervals_with_types(intervals_str: str) -> List[Tuple[int, int, str]]:
    """
    从字符串中提取带类型的区间。
    返回: [(s, e, type), ...]，其中 s, e 是 1-based
    """
    if not intervals_str or intervals_str.strip() == "[]":
        return []
    
    results = []
    pattern = r"\[(\d+),(\d+)(?:,([^\]]+))?\]"
    for m in re.finditer(pattern, intervals_str):
        s, e = int(m.group(1)), int(m.group(2))
        t = m.group(3) if m.group(3) else "unknown"
        results.append((s, e, t))
    return results


def _normalize_name(name: str) -> str:
    """统一路径分隔符为 /，便于 GT 与检测 CSV 按 name 对齐"""
    return name.replace("\\", "/")


def load_gt_csv(path: str) -> Dict[str, Tuple[str, int]]:
    """加载 ground_truth_intervals.csv，返回 {name: (gt_intervals, seq_len)}"""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _normalize_name(row.get("name", "").strip())
            gt_intervals = row.get("gt_intervals", "").strip()
            seq_len = int(row.get("seq_len", 0))
            if name:
                result[name] = (gt_intervals, seq_len)
    return result


def load_detect_csv(path: str) -> Dict[str, str]:
    """加载检测 CSV（run_detect --frame-level 输出），返回 {name: corrupt_intervals}"""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _normalize_name(row.get("name", "").strip())
            corrupt_intervals = row.get("corrupt_intervals", "").strip()
            if name:
                result[name] = corrupt_intervals
    return result


def evaluate(
    gt_csv_path: str,
    detect_csv_path: str,
    output_path: Optional[str] = None,
) -> Tuple[int, int, int, int, float, float, float]:
    """
    对比 GT 与检测结果，计算帧级 TP/FP/FN/TN 及 Precision、Recall、IoU。
    支持按损坏类型进行细分统计。
    """
    gt_data = load_gt_csv(gt_csv_path)
    detect_data = load_detect_csv(detect_csv_path)

    common_names = set(gt_data.keys()) & set(detect_data.keys())
    if not common_names:
        raise ValueError(
            f"GT 与检测 CSV 无共同文件。GT 样本数: {len(gt_data)}, 检测样本数: {len(detect_data)}"
        )

    # 总体统计
    tp = fp = fn = tn = 0
    eps = 1e-8

    # 按类型统计: {type: {"tp": 0, "fn": 0, "total_gt_frames": 0}}
    # 注意: FP 和 TN 难以直接按类型归类（因为背景是通用的），
    # 所以类型统计主要关注该类型的召回率 (Recall)。
    type_stats: Dict[str, Dict[str, int]] = {}

    for name in sorted(common_names):
        gt_intervals_str, seq_len = gt_data[name]
        pred_intervals_str = detect_data[name]

        gt_mask = parse_intervals_to_mask(gt_intervals_str, seq_len)
        pred_mask = parse_intervals_to_mask(pred_intervals_str, seq_len)

        tp += int((gt_mask & pred_mask).sum())
        fp += int((~gt_mask & pred_mask).sum())
        fn += int((gt_mask & ~pred_mask).sum())
        tn += int((~gt_mask & ~pred_mask).sum())

        # 类型细分统计
        gt_parts = parse_intervals_with_types(gt_intervals_str)
        for s, e, t in gt_parts:
            if t not in type_stats:
                type_stats[t] = {"tp": 0, "fn": 0}
            
            # 该特定损坏区间的 mask
            part_mask = np.zeros(seq_len, dtype=bool)
            idx_s, idx_e = max(0, s-1), min(seq_len-1, e-1)
            part_mask[idx_s:idx_e+1] = True
            
            # 在这个特定区间内，检测对了多少帧，漏了多少帧
            type_stats[t]["tp"] += int((part_mask & pred_mask).sum())
            type_stats[t]["fn"] += int((part_mask & ~pred_mask).sum())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    lines = [
        "================ 损坏检测准确性评估 ================",
        f"文件数: {len(common_names)}",
        f"总体指标: TP={tp} FP={fp} FN={fn} TN={tn}",
        f"总体结果: Precision={precision:.4f} Recall={recall:.4f} IoU={iou:.4f}",
        "",
        "---------------- 按损坏类型分析 (Recall) ----------------"
    ]
    
    for t in sorted(type_stats.keys()):
        t_tp = type_stats[t]["tp"]
        t_fn = type_stats[t]["fn"]
        t_recall = t_tp / (t_tp + t_fn + eps)
        lines.append(f"类型 [{t:15s}]: Recall={t_recall:.4f} (TP={t_tp}, FN={t_fn})")
    
    lines.append("==================================================")
    text = "\n".join(lines)
    print(text)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"结果已保存到 {output_path}")

    return tp, fp, fn, tn, precision, recall, iou


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估损坏检测准确性：对比 GT 与检测结果，计算 Precision/Recall/IoU"
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        required=True,
        help="ground_truth_intervals.csv 路径（由 generate_corrupt_data.py 生成）",
    )
    parser.add_argument(
        "--detect-csv",
        type=str,
        required=True,
        help="检测结果 CSV 路径（run_detect --frame-level 输出）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选，汇总指标输出文件路径",
    )
    args = parser.parse_args()

    if not os.path.exists(args.gt_csv):
        print(f"错误：GT CSV 不存在 {args.gt_csv}")
        sys.exit(1)
    if not os.path.exists(args.detect_csv):
        print(f"错误：检测 CSV 不存在 {args.detect_csv}")
        sys.exit(1)

    evaluate(
        gt_csv_path=args.gt_csv,
        detect_csv_path=args.detect_csv,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
