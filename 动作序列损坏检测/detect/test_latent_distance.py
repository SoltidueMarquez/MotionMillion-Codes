import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve

# ==========================================
# 模块导入与环境配置
# ==========================================
# 确保能正确导入项目内的模块，通过将项目的根目录及相关子目录加入 sys.path
_DETECT_DIR = Path(__file__).resolve().parent
_FEATURE_ROOT = _DETECT_DIR.parent
_REPO_ROOT = _FEATURE_ROOT.parent

for _p in (_REPO_ROOT, _FEATURE_ROOT, _DETECT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# 导入现有的模型加载器和量化误差计算核心函数
# compute_quantization_error_per_frame 的核心思想：
# 它只运行 Encoder 得到隐向量 z，然后运行 FSQ 量化器寻找最近的码字（code/level）
# 计算 z 和 码字 之间的距离（残差 diff），最后将其映射回帧级别
from detect_corrupt_utils import load_detector, compute_quantization_error_per_frame


# ==========================================
# 辅助函数定义
# ==========================================
def load_file_list(list_path: str) -> list:
    """
    读取提供的数据列表文件（如 good_list.txt 或 corrupt_list.txt）
    返回一个包含相对路径字符串的列表，自动去除空白符
    """
    with open(list_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def parse_gt_csv(csv_path: str) -> dict:
    """
    解析 Ground Truth (GT) 真实损坏区间的 CSV 文件。
    读取每一行记录，提取文件名以及对应的真实损坏区间 intervals。
    
    返回字典结构：{ 'filename': {'intervals': [(start1, end1), ...], 'seq_len': len} }
    注意：这里的 interval 采用 1-based inclusive 格式，即从 1 开始且包含头尾。
    """
    gt_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name']
            intervals_str = row['gt_intervals']
            seq_len = int(row['seq_len'])
            
            intervals = []
            if intervals_str and intervals_str != "[]":
                import re
                # 使用正则表达式匹配形如 [start,end] 或 [start,end,type] 的格式
                pattern = r"\[(\d+),(\d+)(?:,([^\]]+))?\]"
                for m in re.finditer(pattern, intervals_str):
                    s, e = int(m.group(1)), int(m.group(2))
                    intervals.append((s, e)) # 1-based inclusive
            gt_dict[name] = {
                'intervals': intervals,
                'seq_len': seq_len
            }
    return gt_dict


# ==========================================
# 主流程：隐空间量化误差（距离）测试实验
# ==========================================
def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="隐空间量化误差可行性实验测试脚本 (Proof of Concept)")
    parser.add_argument("--motion-dir", type=str, required=True, help="动作数据的根目录")
    parser.add_argument("--good-list", type=str, required=True, help="正常动作序列的文件列表")
    parser.add_argument("--corrupt-list", type=str, required=True, help="损坏动作序列的文件列表")
    parser.add_argument("--ckpt", type=str, required=True, help="预训练的全身体 FSQ 模型权重路径")
    parser.add_argument("--gt-csv", type=str, required=True, help="包含真实损坏区间 Ground Truth 的 CSV 文件")
    parser.add_argument("--output-csv", type=str, required=True, help="输出最终统计分析结果和最佳阈值的 CSV 文件")
    parser.add_argument("--visualize-num", type=int, default=5, help="要生成并保存的曲线对比图的样本数量")
    parser.add_argument("--frame-level", action="store_true", help="表明本次实验是在帧级别进行检测评估")
    args = parser.parse_args()

    # 2. 模型加载
    # 核心优势：我们不需要重新训练任何模型，完全白嫖预训练 VQ-VAE 编码器的表征能力
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.ckpt}...")
    net, _ = load_detector(args.ckpt, device=device)
    net.eval() # 必须设置为 evaluation 模式

    # 3. 准备数据列表与 Ground Truth (GT)
    good_files = load_file_list(args.good_list)
    corrupt_files = load_file_list(args.corrupt_list)
    
    if len(good_files) != len(corrupt_files):
        print(f"警告：正常文件列表数量({len(good_files)})与损坏文件列表数量({len(corrupt_files)})不一致！将按最小数量进行截断匹配。")
    
    gt_info = parse_gt_csv(args.gt_csv)
    
    num_samples = min(len(good_files), len(corrupt_files))
    motion_dir = Path(args.motion_dir)
    
    # 准备可视化曲线保存的输出目录
    vis_dir = Path(args.output_csv).parent / "vis_latent_distance"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 用于收集所有帧的误差值和真实标签，以便最后计算全局最佳阈值 (Optimal Threshold) 和 AUC
    all_errors = []
    all_labels = []
    
    # 收集每条序列单独的统计信息
    sequence_stats = []
    
    print(f"Starting to process {num_samples} pairs of sequences...")
    for i in tqdm(range(num_samples)):
        good_rel_path = good_files[i]
        corrupt_rel_path = corrupt_files[i]
        
        good_path = motion_dir / f"{good_rel_path}.npy"
        corrupt_path = motion_dir / f"{corrupt_rel_path}.npy"
        
        if not good_path.exists() or not corrupt_path.exists():
            continue
            
        good_motion = np.load(good_path)
        corrupt_motion = np.load(corrupt_path)
        
        T = good_motion.shape[0]
        
        # =========================================================
        # 核心环节：绕过 Decoder，仅在隐空间计算量化误差！
        # =========================================================
        # 导师理论依据：
        # - 正常动作在隐空间中能够找到匹配的离散码字 (Codebook)，z 与 码字 的距离非常小
        # - 损坏动作（异常模式）在预训练的字典里不存在，强行寻找最近的码字会导致巨大的量化残差距离
        # 这样做极大地减少了推理时间，因为无需 Decoder 重建庞大的动作序列。
        err_good = compute_quantization_error_per_frame(net, good_motion, device).cpu().numpy()
        err_corrupt = compute_quantization_error_per_frame(net, corrupt_motion, device).cpu().numpy()
        
        # 构建当前损坏序列 (corrupt motion) 对应的 GT 布尔遮罩 (Mask)
        gt_mask = np.zeros(T, dtype=bool)
        intervals = []
        if corrupt_rel_path in gt_info:
            intervals = gt_info[corrupt_rel_path]['intervals']
            for s, e in intervals:
                # GT 的 s 和 e 是 1-based inclusive 的，转为 0-based index
                gt_mask[s-1:e] = True
        else:
            # Fallback 策略：如果路径名对不上，尝试使用文件名 (stem) 进行匹配
            stem = Path(corrupt_rel_path).stem
            matched_key = None
            for k in gt_info.keys():
                if Path(k).stem == stem:
                    matched_key = k
                    break
            if matched_key:
                intervals = gt_info[matched_key]['intervals']
                for s, e in intervals:
                    gt_mask[s-1:e] = True
        
        # 收集全局评估数据：
        # 对正常序列而言，所有帧都是完好的（标签为 False）
        all_errors.extend(err_good.tolist())
        all_labels.extend([False] * T)
        
        # 对损坏序列而言，根据人为设定的 GT 区间标记真实状态（标签来自 gt_mask）
        all_errors.extend(err_corrupt.tolist())
        all_labels.extend(gt_mask.tolist())
        
        # 记录单条序列的统计详情（用于观察 GT 内和 GT 外残差分布的显著差异）
        max_err_in_gt = err_corrupt[gt_mask].max() if gt_mask.any() else 0.0
        mean_err_in_gt = err_corrupt[gt_mask].mean() if gt_mask.any() else 0.0
        max_err_out_gt = err_corrupt[~gt_mask].max() if (~gt_mask).any() else 0.0
        mean_err_out_gt = err_corrupt[~gt_mask].mean() if (~gt_mask).any() else 0.0
        
        sequence_stats.append({
            'name': corrupt_rel_path,
            'max_err_in_gt': max_err_in_gt,
            'mean_err_in_gt': mean_err_in_gt,
            'max_err_out_gt': max_err_out_gt,
            'mean_err_out_gt': mean_err_out_gt,
            'gt_intervals': str(intervals)
        })
        
        # =========================================================
        # 可视化部分：绘制量化误差曲线图
        # =========================================================
        # 为了直观给导师展示方案的可行性，绘制前 N 个样本的距离曲线
        if i < args.visualize_num:
            plt.figure(figsize=(12, 6))
            
            # 蓝线：正常动作在隐空间的残差（应保持低且平稳）
            plt.plot(err_good, label='Normal Motion (err_good)', color='blue', alpha=0.7)
            # 红线：带有损坏的动作在隐空间的残差
            plt.plot(err_corrupt, label='Corrupted Motion (err_corrupt)', color='red', alpha=0.8)
            
            # 使用浅黄色高亮标出 GT 的真实损坏区间
            # 预期现象：在黄色的高亮区域内，红线（err_corrupt）应该会产生非常明显的陡增（Spike/Peak）
            for s, e in intervals:
                plt.axvspan(s-1, e-1, color='yellow', alpha=0.3, label='GT Corrupt Interval' if s == intervals[0][0] else "")
                
            plt.title(f"Latent Quantization Error (Distance) - {Path(corrupt_rel_path).stem}")
            plt.xlabel("Frame")
            plt.ylabel("Quantization Error (Squared L2 Diff)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            save_path = vis_dir / f"latent_dist_{Path(corrupt_rel_path).stem}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()

    # =========================================================
    # 全局数据分析与最佳阈值 (Optimal Threshold) 搜索
    # =========================================================
    # 因为不能确定一个固定的阈值，所以我们通过汇总所有的测试数据，
    # 动态搜索能够得到最佳 F1 Score 的划分点，并计算 ROC AUC 评价指标
    print("Calculating optimal threshold and evaluation metrics...")
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    if all_labels.any():
        # 基于 Precision-Recall 曲线搜索能使 F1 得分最大化的最佳阈值
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_errors)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
        
        # 计算 ROC AUC（Area Under Curve）：衡量利用量化误差区分损坏帧能力的“黄金指标”
        # AUC 越接近 1，证明这种“仅在隐空间检查量化残差”的方法越有效、越行得通！
        fpr, tpr, _ = roc_curve(all_labels, all_errors)
        roc_auc = auc(fpr, tpr)
        
        print(f"Optimal Threshold: {best_threshold:.6f}")
        print(f"Best F1 Score: {best_f1:.4f} (Precision: {best_precision:.4f}, Recall: {best_recall:.4f})")
        print(f"ROC AUC: {roc_auc:.4f}")
    else:
        best_threshold = 0.0
        roc_auc = 0.0
        print("未找到任何含有 GT 损坏的帧，无法计算评价指标。")

    # =========================================================
    # 输出详细统计报告到 CSV
    # =========================================================
    # 将寻找出的全局最佳阈值和各项统计指标写入 CSV，方便后续整理汇报
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Overall Analysis"])
        writer.writerow(["Optimal Threshold", f"{best_threshold:.6f}"])
        if all_labels.any():
            writer.writerow(["Best F1 Score", f"{best_f1:.4f}"])
            writer.writerow(["Best Precision", f"{best_precision:.4f}"])
            writer.writerow(["Best Recall", f"{best_recall:.4f}"])
            writer.writerow(["ROC AUC", f"{roc_auc:.4f}"])
        else:
            writer.writerow(["Note", "No GT anomalies found."])
        
        writer.writerow([])
        writer.writerow(["Sequence Details"])
        writer.writerow([
            "name", "max_err_in_gt", "mean_err_in_gt", 
            "max_err_out_gt", "mean_err_out_gt", "gt_intervals"
        ])
        # 写入每个单独序列的详细信息，方便导师审查具体用例中“损坏内”和“损坏外”的误差对比
        for stat in sequence_stats:
            writer.writerow([
                stat['name'], 
                f"{stat['max_err_in_gt']:.6f}", 
                f"{stat['mean_err_in_gt']:.6f}", 
                f"{stat['max_err_out_gt']:.6f}", 
                f"{stat['mean_err_out_gt']:.6f}", 
                stat['gt_intervals']
            ])

    print(f"Results saved to {out_csv}")
    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()
