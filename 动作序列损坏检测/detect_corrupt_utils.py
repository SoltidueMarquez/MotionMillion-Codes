"""
动作序列损坏检测 - 核心工具模块

基于 MotionMillion 预训练 FSQ-VQ-VAE，提供两种检测指标：
1. 重构误差：L2(rec_motion, motion)
2. 量化误差：||z_projected - codes||（有效码本空间内）

不依赖文本/分词器，仅使用 encoder + quantizer + decoder。
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

# region 路径与导入配置
# 本脚本位于 动作序列损坏检测/ 子目录，需将项目根目录加入 sys.path，
# 才能正确导入 models.vqvae、options 等上层模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from einops import rearrange
from einops import pack as einops_pack
from einops import unpack as einops_unpack
# endregion


# region 辅助函数：einops 包装
# FSQ 内部使用 pack_one/unpack_one 进行张量形状的打包与解包，
# 此处复现相同接口，避免修改 models/FSQ.py
def _pack_one(t, pattern):
    """将张量按 pattern 打包，用于统一 (b, t, d) -> (b, n, d) 等变换"""
    return einops_pack([t], pattern)


def _unpack_one(t, ps, pattern):
    """将打包后的张量按 pattern 解包回原始形状"""
    return einops_unpack(t, ps, pattern)[0]
# endregion


# region 模型参数构建
def _get_detection_args(
    ckpt_path: Optional[str] = None,
    dataname: str = "motionmillion",
    motion_type: str = "vector_272",
) -> Namespace:
    """
    构建与训练一致的模型参数（不解析命令行）。

    为什么需要单独构建：加载 checkpoint 时，模型结构必须与训练时完全一致，
    否则 load_state_dict 会失败。train_tokenizer.sh 使用的参数（如 nb_code=65536、
    down_t=1、use_patcher 等）需在此处复现。
    """
    args = Namespace(
        dataname=dataname,
        quantizer="FSQ",
        nb_code=65536,
        code_dim=512,
        output_emb_width=512,
        down_t=1,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm="LN",
        kernel_size=3,
        use_patcher=True,
        patch_size=1,
        patch_method="haar",
        use_attn=False,
        causal=False,
        motion_type=motion_type,
        text_type="texts",
        version="version1",
    )
    if ckpt_path is not None:
        args.resume_pth = ckpt_path
    return args
# endregion


# region 模型加载
def load_detector(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    dataname: str = "motionmillion",
    motion_type: str = "vector_272",
):
    """
    加载预训练 FSQ-VQ-VAE 模型用于损坏检测。

    流程：构建参数 -> 创建 HumanVQVAE -> 加载权重 -> 设为 eval 模式。
    不依赖文本编码器或分词器，仅使用动作编码/量化/解码部分。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    args = _get_detection_args(ckpt_path, dataname, motion_type)

    import models.vqvae as vqvae

    # 创建模型：结构与 train_tokenizer 训练时一致
    net = vqvae.HumanVQVAE(
        args,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.activation,
        norm=args.norm,
        kernel_size=args.kernel_size,
        use_patcher=args.use_patcher,
        patch_size=args.patch_size,
        patch_method=args.patch_method,
        use_attn=args.use_attn,
    )

    # 加载权重：checkpoint 可能为 {'net': state_dict} 或直接为 state_dict
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("net", ckpt)
    # 分布式训练保存的 key 带 'module.' 前缀，需去掉以匹配单卡结构
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    net.to(device)
    return net, args
# endregion


# region 重构误差计算
def compute_reconstruction_error(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    计算重构误差 L2(rec_motion, motion)。

    原理：正常动作经 encode -> quantize -> decode 后，重建应接近原输入；
    损坏动作偏离训练分布，重建误差会更大。使用 MSE 作为 L2 的平方形式。
    """
    if device is None:
        device = next(net.parameters()).device
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    motion = motion.to(device)
    # 单条输入 (T, 272) 需加 batch 维 -> (1, T, 272)
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)

    with torch.no_grad():
        rec, _, _, _, _ = net(motion)
        err = torch.nn.functional.mse_loss(rec, motion, reduction=reduction)
    return err
# endregion


# region 量化误差计算
def compute_quantization_error(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    计算量化误差 ||z_bounded - codes_level||（有效码本空间内）。

    原理：FSQ 将连续隐向量 z 量化到离散格点。正常动作的 z 应靠近格点，
    量化残差小；损坏动作的 z 偏离格点，残差大。此处复现 FSQ 的
    project_in -> bound -> quantize 流程，不修改 models/FSQ.py。
    """
    if device is None:
        device = next(net.parameters()).device
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    motion = motion.to(device)
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)

    vqvae_inner = net.vqvae
    quantizer = vqvae_inner.quantizer
    if quantizer.__class__.__name__ != "FSQ":
        raise ValueError("量化误差仅支持 FSQ 量化器")

    with torch.no_grad():
        # 1. 预处理：(b, T, 272) -> (b, 272, T)，与 encoder 输入格式一致
        x_in = vqvae_inner.preprocess(motion)
        # 2. 编码：得到连续隐向量 z，形状 (b, d, t)
        z = vqvae_inner.encoder(x_in)

        # 3. 与 FSQ.forward 一致的维度变换，便于调用 project_in 和 quantize
        z = rearrange(z, "b d ... -> b ... d")  # (b, t, d)
        z, ps = _pack_one(z, "b * d")           # (b, n, d)，n = t
        z = quantizer.project_in(z)             # 投影到有效码本维度（如 6D）
        z = rearrange(z, "b n (c d) -> b n c d", c=quantizer.num_codebooks)  # (b, n, 1, 6)

        # 4. 量化上下文：FSQ 在 float32 下量化，避免半精度舍入误差
        from torch.amp import autocast
        from contextlib import nullcontext
        from functools import partial
        force_f32 = getattr(quantizer, "force_quantization_f32", True)
        ctx = partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext

        with ctx():
            orig_dtype = z.dtype
            if force_f32 and orig_dtype not in (torch.float32, torch.float64):
                z = z.float()
            # 5. 有界化：将 z 限制到量化等级有效范围
            z_bounded = quantizer.bound(z)
            # 6. 量化：round_ste(bound(z)) 再归一化到 [-1,1]
            codes = quantizer.quantize(z)
            # 7. 将 codes 还原到 level 空间，与 z_bounded 同尺度，才能计算残差
            half_width = quantizer._levels // 2  # 每维半宽，如 [4,4,4,2,2,2]
            codes_level = codes * half_width
            diff = z_bounded - codes_level  # 量化残差
            err = (diff * diff).sum(dim=-1)  # 每 token 的 L2^2
            if reduction == "mean":
                err = err.mean()
            elif reduction == "sum":
                err = err.sum()
    return err
# endregion


# region 帧级误差计算
def compute_reconstruction_error_per_frame(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    计算每帧重构误差，返回形状 (T,) 或 (b, T)。

    rec 与 motion 均为 (b, T, 272)，MSE reduction="none" 得 (b,T,272)，
    对 272 维 mean 得每帧误差。
    """
    if device is None:
        device = next(net.parameters()).device
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    motion = motion.to(device)
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)

    with torch.no_grad():
        rec, _, _, _, _ = net(motion)
        err = torch.nn.functional.mse_loss(rec, motion, reduction="none")  # (b, T, 272)
        err = err.mean(dim=2)  # (b, T)
    if err.shape[0] == 1:
        err = err.squeeze(0)  # (T,)
    return err


def compute_quantization_error_per_frame(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    计算每帧量化误差。latent 时间维度为 T/2，每个 token 对应约 2 帧，
    通过 repeat_interleave 映射回帧级。
    """
    if device is None:
        device = next(net.parameters()).device
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    motion = motion.to(device)
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)

    T = motion.shape[1]
    vqvae_inner = net.vqvae
    quantizer = vqvae_inner.quantizer
    if quantizer.__class__.__name__ != "FSQ":
        raise ValueError("量化误差仅支持 FSQ 量化器")

    with torch.no_grad():
        x_in = vqvae_inner.preprocess(motion)
        z = vqvae_inner.encoder(x_in)
        z = rearrange(z, "b d ... -> b ... d")
        z, ps = _pack_one(z, "b * d")
        z = quantizer.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=quantizer.num_codebooks)

        from torch.amp import autocast
        from contextlib import nullcontext
        from functools import partial
        force_f32 = getattr(quantizer, "force_quantization_f32", True)
        ctx = partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext

        with ctx():
            if force_f32 and z.dtype not in (torch.float32, torch.float64):
                z = z.float()
            z_bounded = quantizer.bound(z)
            codes = quantizer.quantize(z)
            half_width = quantizer._levels // 2
            codes_level = codes * half_width
            diff = z_bounded - codes_level
            err_token = (diff * diff).sum(dim=-1)  # (b, n), n = T/2

        # 映射到帧级：每个 token 对应 2 帧
        err = err_token.repeat_interleave(2, dim=1)  # (b, 2*n)
        if err.shape[1] < T:
            # T 为奇数时，最后一帧复制最后一个 token 的误差
            pad = err[:, -1:].expand(-1, T - err.shape[1])
            err = torch.cat([err, pad], dim=1)
        elif err.shape[1] > T:
            err = err[:, :T]

    if err.shape[0] == 1:
        err = err.squeeze(0)  # (T,)
    return err


def corrupt_frames_to_intervals(
    corrupt_mask: Union[torch.Tensor, np.ndarray],
) -> str:
    """
    将每帧损坏标签转为区间字符串，1-based inclusive。

    输入：corrupt_mask (T,) bool
    输出：如 "[1,17],[22,78]" 或 "[]"
    """
    if isinstance(corrupt_mask, torch.Tensor):
        corrupt_mask = corrupt_mask.cpu().numpy()
    corrupt_mask = np.asarray(corrupt_mask, dtype=bool).ravel()
    if corrupt_mask.size == 0 or not corrupt_mask.any():
        return "[]"

    # 找到连续 True 的 run
    intervals = []
    in_run = False
    start = 0
    for i in range(len(corrupt_mask)):
        if corrupt_mask[i] and not in_run:
            in_run = True
            start = i
        elif not corrupt_mask[i] and in_run:
            in_run = False
            intervals.append((start + 1, i))  # 1-based inclusive
    if in_run:
        intervals.append((start + 1, len(corrupt_mask)))  # 1-based inclusive

    return ",".join(f"[{s},{e}]" for s, e in intervals)


def detect_corrupt_per_frame(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    threshold: float,
    metric: Literal["recon", "quant"] = "recon",
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    帧级检测：返回每帧误差、损坏掩码、损坏区间字符串。

    Returns:
        per_frame_err: (T,) 每帧误差
        corrupt_mask: (T,) bool 每帧是否损坏
        intervals_str: 如 "[1,17],[22,78]" 或 "[]"
    """
    if metric == "recon":
        per_frame_err = compute_reconstruction_error_per_frame(net, motion, device)
    else:
        per_frame_err = compute_quantization_error_per_frame(net, motion, device)

    corrupt_mask = per_frame_err > threshold
    intervals_str = corrupt_frames_to_intervals(corrupt_mask)
    return per_frame_err, corrupt_mask, intervals_str
# endregion


# region 损坏判定
def detect_corrupt(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    threshold: float,
    metric: Literal["recon", "quant"] = "recon",
    device: Optional[torch.device] = None,
) -> Tuple[float, bool]:
    """
    单次检测：计算误差并判定是否损坏。

    误差超过 threshold 则判为损坏。阈值需在标定集上根据业务需求确定。
    """
    if metric == "recon":
        err = compute_reconstruction_error(net, motion, device, reduction="mean")
    else:
        err = compute_quantization_error(net, motion, device, reduction="mean")
    err_val = err.item() if isinstance(err, torch.Tensor) else float(err)
    return err_val, err_val > threshold
# endregion


# region 命令行测试入口
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/pretrained_models/fsq_net_6000000.pth")
    ap.add_argument("--motion", type=str, default=None, help="单条 .npy 路径，不提供则用随机数据测试")
    ap.add_argument("--metric", choices=["recon", "quant"], default="recon")
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"检查点不存在: {args.ckpt}")
        print("请先下载或训练得到 fsq_net_6000000.pth，参见 ENV_CONFIG.md")
        sys.exit(1)

    net, _ = load_detector(args.ckpt)
    if args.motion and os.path.exists(args.motion):
        motion = np.load(args.motion)
        if motion.shape[0] < 64:
            print("警告：序列过短，建议至少 64 帧")
    else:
        motion = np.random.randn(96, 272).astype(np.float32) * 0.1  # 随机测试

    err, corrupt = detect_corrupt(net, motion, args.threshold, args.metric)
    print(f"metric={args.metric}, error={err:.6f}, corrupt={corrupt} (threshold={args.threshold})")
# endregion
