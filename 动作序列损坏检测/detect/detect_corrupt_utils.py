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
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch

# region 路径与导入配置
_DETECT_DIR = Path(__file__).resolve().parent
_FEATURE_ROOT = _DETECT_DIR.parent
_REPO_ROOT = _FEATURE_ROOT.parent

# 仓库根用于导入 models/utils/visualize，detect 目录用于兼容平铺模块引用。
for _p in (_REPO_ROOT, _FEATURE_ROOT, _DETECT_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

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
    pin_to_origin: bool = False,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
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
        
        # 如果开启原地化，清除根节点位移带来的误差
        if pin_to_origin:
            motion = pin_motion_to_origin(motion.clone(), mean, std)
            rec = pin_motion_to_origin(rec, mean, std)

        # 计算重建前后的 MSE 误差
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
    pin_to_origin: bool = False,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
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

        # 如果开启原地化，清除根节点位移带来的误差
        if pin_to_origin:
            motion = pin_motion_to_origin(motion.clone(), mean, std)
            rec = pin_motion_to_origin(rec, mean, std)

        err = torch.nn.functional.mse_loss(rec, motion, reduction="none")  # (b, T, 272)
        err = err.mean(dim=2)  # (b, T)
    if err.shape[0] == 1:
        err = err.squeeze(0)  # (T,)
    return err


def pin_motion_to_origin(
    motion: Union[torch.Tensor, np.ndarray],
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    将动作钉在原地：去除根节点位移和朝向变化。
    支持 torch.Tensor (b, T, 272) 或 np.ndarray (T, 272)。
    如果提供了 mean 和 std，则在标准化空间进行 pinning。
    """
    is_torch = isinstance(motion, torch.Tensor)
    identity_6d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    if is_torch:
        device = motion.device
        dtype = motion.dtype
        # 如果是 float16/bfloat16，可能需要转为 float32 处理
        id_6d = torch.tensor(identity_6d, device=device, dtype=dtype)
        if mean is not None:
            # 在标准化空间进行 pinning
            m = torch.as_tensor(mean, device=device, dtype=dtype)
            s = torch.as_tensor(std, device=device, dtype=dtype)
            v_root = (torch.zeros(2, device=device, dtype=dtype) - m[0:2]) / s[0:2]
            v_head = (id_6d - m[2:8]) / s[2:8]
            motion[..., 0:2] = v_root
            motion[..., 2:8] = v_head
        else:
            # 在原始空间或假设 mean=0/std=1 空间
            motion[..., 0:2] = 0.0
            motion[..., 2:8] = id_6d
    else:
        if mean is not None:
            v_root = (np.zeros(2) - mean[0:2]) / std[0:2]
            v_head = (np.array(identity_6d) - mean[2:8]) / std[2:8]
            motion[..., 0:2] = v_root
            motion[..., 2:8] = v_head
        else:
            motion[..., 0:2] = 0.0
            motion[..., 2:8] = np.array(identity_6d, dtype=np.float32)

    return motion


def get_vector272_part_indices() -> List[List[int]]:
    """
    获取 vector_272 各个部位的索引列表。
    返回列表的顺序对应: h1, h2L, h2R, h3L, h3R, h4, h
    与 StableMotion 保持逻辑一致。
    """
    # 0..22 joints, 272 dims total
    def _get_joint_indices(joint_ids):
        indices = []
        for jid in joint_ids:
            # Pos: 8:74 (3*22)
            indices.extend(range(8 + jid * 3, 8 + (jid + 1) * 3))
            # Vel: 74:140 (3*22)
            indices.extend(range(74 + jid * 3, 74 + (jid + 1) * 3))
            # Rot: 140:272 (6*22)
            indices.extend(range(140 + jid * 6, 140 + (jid + 1) * 6))
        return indices

    # h1: Root (trajectory, heading, pelvis pos/vel)
    h1_ids = list(range(0, 8))  # traj + heading
    h1_ids.extend(range(8 + 0 * 3, 8 + 1 * 3))  # pelvis pos
    h1_ids.extend(range(74 + 0 * 3, 74 + 1 * 3))  # pelvis vel

    # h2L: L-Leg (1, 4, 7, 10)
    h2L_ids = _get_joint_indices([1, 4, 7, 10])

    # h2R: R-Leg (2, 5, 8, 11)
    h2R_ids = _get_joint_indices([2, 5, 8, 11])

    # h3L: L-Arm (13, 16, 18, 20)
    h3L_ids = _get_joint_indices([13, 16, 18, 20])

    # h3R: R-Arm (14, 17, 19, 21)
    h3R_ids = _get_joint_indices([14, 17, 19, 21])

    # h4: Trunk/Head (3, 6, 9, 12, 15) + Pelvis Rot (140:146)
    h4_ids = _get_joint_indices([3, 6, 9, 12, 15])
    h4_ids.extend(range(140 + 0 * 6, 140 + 1 * 6))  # pelvis rot

    # h: Overall
    h_ids = list(range(272))

    return [h1_ids, h2L_ids, h2R_ids, h3L_ids, h3R_ids, h4_ids, h_ids]


def compute_reconstruction_error_per_part(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
    pin_to_origin: bool = False,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    """
    计算每帧各个部位的重构误差。
    返回形状 (T, 7)，最后一维对应 h1..h4 (含 L/R) 和 h。
    """
    if device is None:
        device = next(net.parameters()).device
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    motion = motion.to(device)
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)

    part_indices = get_vector272_part_indices()

    with torch.no_grad():
        rec, _, _, _, _ = net(motion)
        
        # 如果开启原地化，清除根节点位移带来的误差
        if pin_to_origin:
            motion = pin_motion_to_origin(motion.clone(), mean, std)
            rec = pin_motion_to_origin(rec, mean, std)

        # (b, T, 272)
        err_full = torch.nn.functional.mse_loss(rec, motion, reduction="none")

        per_part_errors = []
        for indices in part_indices:
            # 在指定维度上求均值
            part_err = err_full[:, :, indices].mean(dim=2)  # (b, T)
            per_part_errors.append(part_err)

        # (7, b, T) -> (b, T, 7)
        res = torch.stack(per_part_errors, dim=-1)

    if res.shape[0] == 1:
        res = res.squeeze(0)  # (T, 7)
    return res


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
    pin_to_origin: bool = False,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    帧级检测：返回每帧误差、损坏掩码、损坏区间字符串。

    Returns:
        per_frame_err: (T,) 每帧误差
        corrupt_mask: (T,) bool 每帧是否损坏
        intervals_str: 如 "[1,17],[22,78]" 或 "[]"
    """
    if metric == "recon":
        per_frame_err = compute_reconstruction_error_per_frame(net, motion, device, pin_to_origin, mean, std)
    else:
        # 量化误差不涉及反标准化后的位移清除
        per_frame_err = compute_quantization_error_per_frame(net, motion, device)

    corrupt_mask = per_frame_err > threshold
    intervals_str = corrupt_frames_to_intervals(corrupt_mask)
    return per_frame_err, corrupt_mask, intervals_str
# endregion


# region 动作可视化
def _draw_part_errors_overlay(
    frame: np.ndarray,
    errors: np.ndarray,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    part_names: List[str] = ["h1", "h2L", "h2R", "h3L", "h3R", "h4", "h"],
) -> np.ndarray:
    """
    在帧的右上角显示各部位的重构误差。
    frame: (H, W, 3) or (H, W, 4)
    errors: (7,) 每个部位的误差值
    colors: (7,) 每个部位对应的颜色 (R, G, B)，若为 None 则默认白色
    """
    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    h, w = frame.shape[:2]
    # 在右上角绘制一个半透明背景
    margin = 10
    line_h = 25
    rect_w = 200 # 稍微加宽以适应中文
    rect_h = line_h * (len(part_names) + 1) + margin
    
    top_left = (w - rect_w - margin, margin)
    bottom_right = (w - margin, margin + rect_h)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (40, 40, 40), -1)
    # 混合原图与矩形，实现半透明 (alpha=0.6)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 使用 PIL 绘制文本以支持中文
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体，Windows 常见路径
    font_paths = [
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "D:\\Windows\\Fonts\\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 16)
                break
            except:
                continue
    
    if font is None:
        font = ImageFont.load_default()

    # 绘制标题
    draw.text((top_left[0] + 5, top_left[1] + 5), "各部位重构误差 (MSE)", font=font, fill=(220, 220, 220))
    
    # 中文名称映射
    CHINESE_NAMES = {
        "h1": "根节点",
        "h2L": "左腿",
        "h2R": "右腿",
        "h3L": "左臂",
        "h3R": "右臂",
        "h4": "躯干/头",
        "h": "总体"
    }
    
    default_color = (255, 255, 255) # 改为白色
    for i, (name, err) in enumerate(zip(part_names, errors)):
        disp_name = CHINESE_NAMES.get(name, name)
        text = f"{disp_name}: {err:.6f}"
        pos = (top_left[0] + 10, top_left[1] + 25 + (i + 1) * (line_h - 2))
        text_color = colors[i] if colors is not None else default_color
        # PIL 使用 RGB
        draw.text(pos, text, font=font, fill=text_color)

    return np.array(img_pil)


def visualize_motion_overlay_vector272(
    motion_orig: np.ndarray,
    motion_rec: np.ndarray,
    output_path: str,
    per_frame_errors: Optional[np.ndarray] = None,
    fps: int = 30,
    title: str = "Original (Blue) vs Reconstructed (Red)",
    gt_corrupt_mask: Optional[np.ndarray] = None,
    detected_corrupt_mask: Optional[np.ndarray] = None,
    gt_intervals_str: Optional[str] = None,
) -> bool:
    """
    将原始动作与重建动作重叠可视化。
    motion_orig, motion_rec: (T, 272) 已反标准化的动作
    per_frame_errors: (T, 7) 每帧每个部位的误差值
    """
    try:
        from utils.motion_process import recover_from_local_rotation
        from visualize.smplx2joints import process_smplx_data
        import imageio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import io
    except ImportError as e:
        import warnings
        warnings.warn(f"可视化依赖未安装，跳过: {e}")
        return False

    try:
        T = motion_orig.shape[0]
        
        # 计算序列统计信息用于自适应异常检测
        part_medians = None
        part_mads = None
        diff_medians = None
        diff_mads = None
        if per_frame_errors is not None:
            # 使用稳健统计量 (Median/MAD) 替代 Mean/Std，防止离群值掩盖真实异常
            # 每个部位的统计信息
            part_medians = np.median(per_frame_errors, axis=0)
            part_mads = np.median(np.abs(per_frame_errors - part_medians), axis=0)
            
            # 帧间跳变的统计信息 (mutation detection)
            diffs = np.abs(np.diff(per_frame_errors, axis=0))
            diff_medians = np.median(diffs, axis=0)
            diff_mads = np.median(np.abs(diffs - diff_medians), axis=0)

        def _get_joints(m):
            smpl_85 = recover_from_local_rotation(m, njoint=22)
            smplx_322 = np.concatenate((
                smpl_85[:, :66], np.zeros((T, 90)), np.zeros((T, 3)),
                np.zeros((T, 50)), np.zeros((T, 100)),
                smpl_85[:, 72:75], smpl_85[:, 75:]
            ), axis=-1)
            _, joints, _, _ = process_smplx_data(smplx_322, norm_global_orient=False, transform=False)
            return joints[:, :22, :].detach().cpu().numpy()

        joints_orig = _get_joints(motion_orig)
        joints_rec = _get_joints(motion_rec)

        # 动力学链 (SMPL-22)
        kinetic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        
        # 渲染循环
        frames = []
        # 提高 DPI 使骨骼更清晰
        fig = plt.figure(figsize=(10, 10), dpi=120)
        
        for t in range(T):
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            
            # 让视角跟随原始动作的根节点，实现自动缩放和中心化
            root = joints_orig[t, 0]
            # view_radius 决定了缩放程度。2.0 米半径能看清全身动作
            view_radius = 2.0
            
            ax.set_xlim3d([root[0] - view_radius, root[0] + view_radius])
            ax.set_ylim3d([root[1] - view_radius, root[1] + view_radius])
            # 假设 Z 是高度方向
            ax.set_zlim3d([0, view_radius * 2])
            
            ax.view_init(elev=20, azim=-90)
            ax.set_title(title)
            ax.axis('off')

            # 绘制跟随角色的局部地面网格
            grid_res = 11
            grid_x, grid_y = np.meshgrid(
                np.linspace(root[0] - view_radius, root[0] + view_radius, grid_res),
                np.linspace(root[1] - view_radius, root[1] + view_radius, grid_res)
            )
            ax.plot_surface(grid_x, grid_y, np.zeros_like(grid_x), alpha=0.1, color='gray')

            # 绘制两个骨架，增加线宽
            for chain in kinetic_chain:
                # 原始: 蓝色
                ax.plot(joints_orig[t, chain, 0], joints_orig[t, chain, 1], joints_orig[t, chain, 2], 
                        linewidth=4.5, color='blue', alpha=0.8, label='Original' if chain == kinetic_chain[0] and t == 0 else "")
                # 重建: 红色
                ax.plot(joints_rec[t, chain, 0], joints_rec[t, chain, 1], joints_rec[t, chain, 2], 
                        linewidth=4.5, color='red', alpha=0.8, label='Reconstructed' if chain == kinetic_chain[0] and t == 0 else "")
            
            # 转为图像
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120)
            buf.seek(0)
            frame = imageio.imread(buf)
            buf.close()

            # 叠加误差信息
            if per_frame_errors is not None:
                # 颜色定义 (R, G, B) - 这里的帧是 RGB 格式
                WHITE = (255, 255, 255) # 默认白色
                YELLOW = (255, 255, 0)
                RED = (255, 0, 0)
                
                curr_errors = per_frame_errors[t]
                colors = []
                for i in range(len(curr_errors)):
                    err = curr_errors[i]
                    # 自适应突变检测 (与上一帧对比)
                    mutation = False
                    if t > 0:
                        prev_err = per_frame_errors[t-1, i]
                        diff = abs(err - prev_err)
                        # 突变判定：超过跳变中位数的 4.0 倍 MAD (使用更严格的阈值捕捉极端跳变)
                        if diff > diff_medians[i] + 4.0 * (diff_mads[i] + 1e-6):
                            mutation = True
                    
                    # 自适应大误差检测
                    large = False
                    # 大误差判定：超过该序列中位数的 3.0 倍 MAD
                    if err > part_medians[i] + 3.0 * (part_mads[i] + 1e-6):
                        large = True
                    
                    if mutation:
                        colors.append(RED)
                    elif large:
                        colors.append(YELLOW)
                    else:
                        colors.append(WHITE)
                
                frame = _draw_part_errors_overlay(frame, curr_errors, colors=colors)
            
            frames.append(frame)

        plt.close(fig)
        
        # 叠加 GT/Det 标注 (frames 是列表，转成 numpy 数组处理)
        frames_np = np.array(frames, dtype=np.uint8)
        frames_np = _draw_annotation_overlay(frames_np, gt_corrupt_mask, detected_corrupt_mask, gt_intervals_str=gt_intervals_str)
        
        out_dir = os.path.dirname(output_path)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        imageio.mimsave(output_path, frames_np, fps=fps)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        import warnings
        warnings.warn(f"重叠可视化失败 {output_path}: {e}")
        return False


def _draw_annotation_overlay(
    frames: np.ndarray,
    gt_corrupt_mask: Optional[np.ndarray],
    detected_corrupt_mask: Optional[np.ndarray],
    gt_intervals_str: Optional[str] = None,
) -> np.ndarray:
    """
    在每帧左上角叠加 GT/Det 标注文本。
    frames: (T, H, W, C), C 为 3 或 4
    gt_intervals_str: 带类型的 GT 区间字符串，用于显示具体的损坏类型
    """
    if gt_corrupt_mask is None and detected_corrupt_mask is None:
        return frames
    try:
        import cv2
    except ImportError:
        return frames

    T = frames.shape[0]
    out = np.array(frames, dtype=np.uint8, copy=True)
    has_alpha = out.shape[-1] == 4

    # 预解析 GT 区间以获得每帧的类型信息
    frame_types: List[set] = [set() for _ in range(T)]
    if gt_intervals_str and gt_intervals_str != "[]":
        import re
        pattern = r"\[(\d+),(\d+)(?:,([^\]]+))?\]"
        for m in re.finditer(pattern, gt_intervals_str):
            s, e = int(m.group(1)), int(m.group(2))
            t = m.group(3) if m.group(3) else "corrupt"
            for i in range(max(0, s-1), min(T, e)):
                frame_types[i].add(t)

    for i in range(T):
        frame = out[i]
        rgb = frame[..., :3] if has_alpha else frame
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        parts = [f"Frame {i}"]
        if gt_corrupt_mask is not None and i < len(gt_corrupt_mask):
            if gt_corrupt_mask[i]:
                types = frame_types[i]
                type_str = ",".join(sorted(list(types))) if types else "corrupt"
                parts.append(f"GT: {type_str}")
            else:
                parts.append("GT: OK")
        
        if detected_corrupt_mask is not None and i < len(detected_corrupt_mask):
            parts.append(f"Det: {'corrupt' if detected_corrupt_mask[i] else 'OK'}")
        
        text = " | ".join(parts)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame_bgr, (5, 5), (15 + tw, 15 + th), (40, 40, 40), -1)
        cv2.putText(frame_bgr, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        rgb_out = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if has_alpha:
            out[i] = np.dstack([rgb_out, frame[..., 3]])
        else:
            out[i] = rgb_out
    return out


def visualize_motion_vector272(
    motion_denorm: np.ndarray,
    output_path: str,
    fps: int = 30,
    gt_corrupt_mask: Optional[np.ndarray] = None,
    detected_corrupt_mask: Optional[np.ndarray] = None,
    gt_intervals_str: Optional[str] = None,
) -> bool:
    """
    将已反标准化的 vector_272 动作序列可视化为视频并保存。

    流程：recover_from_local_rotation -> smpl_85 -> process_smplx_data -> plot_3d_motion -> imageio.mimsave。
    依赖 visualize.smplx2joints（SMPL-X、CUDA）、imageio、matplotlib。

    Args:
        motion_denorm: 已反标准化的 motion (T, 272) 或 (1, T, 272)
        output_path: 输出视频路径，建议 .mp4
        fps: 视频帧率
        gt_corrupt_mask: 可选，(T,) bool，GT 标注的损坏帧
        detected_corrupt_mask: 可选，(T,) bool，检测判定的损坏帧
        gt_intervals_str: 带类型的 GT 区间字符串，用于显示具体的损坏类型

    Returns:
        True 若成功，False 若失败（不抛出异常）
    """
    try:
        from utils.motion_process import recover_from_local_rotation
        from visualize.plot_3d_global import plot_3d_motion
        from visualize.smplx2joints import process_smplx_data
        import imageio
    except ImportError as e:
        import warnings
        warnings.warn(f"可视化依赖未安装，跳过: {e}")
        return False

    try:
        motion_denorm = np.asarray(motion_denorm, dtype=np.float32)
        if motion_denorm.ndim == 3:
            motion_denorm = motion_denorm.squeeze(0)
        if motion_denorm.shape[1] != 272:
            return False

        smpl_85 = recover_from_local_rotation(motion_denorm, njoint=22)

        # smpl_85 -> smplx_322（与 inference_single.visualize_smplx_85 一致）
        smplx_322 = np.concatenate(
            (
                smpl_85[:, :66],
                np.zeros((smpl_85.shape[0], 90)),
                np.zeros((smpl_85.shape[0], 3)),
                np.zeros((smpl_85.shape[0], 50)),
                np.zeros((smpl_85.shape[0], 100)),
                smpl_85[:, 72:75],
                smpl_85[:, 75:],
            ),
            axis=-1,
        )

        vert, joints, motion, faces = process_smplx_data(
            smplx_322, norm_global_orient=False, transform=False
        )
        xyz = joints[:, :22, :].reshape(-1, 22, 3).detach().cpu().numpy()

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        img = plot_3d_motion([xyz, None, None])
        frames = np.array(img, dtype=np.uint8)
        frames = _draw_annotation_overlay(frames, gt_corrupt_mask, detected_corrupt_mask, gt_intervals_str=gt_intervals_str)
        imageio.mimsave(output_path, frames, fps=fps)
        return True
    except Exception as e:
        import warnings
        warnings.warn(f"可视化失败 {output_path}: {e}")
        return False


def save_detection_visualizations(
    motion: Union[torch.Tensor, np.ndarray],
    rec: Union[torch.Tensor, np.ndarray],
    name: str,
    output_dir: str,
    mean: np.ndarray,
    std: np.ndarray,
    fps: int = 30,
    gt_corrupt_mask: Optional[np.ndarray] = None,
    detected_corrupt_mask: Optional[np.ndarray] = None,
    overlay: bool = False,
    per_part_errors: Optional[np.ndarray] = None,
    pin_to_origin: bool = False,
    gt_intervals_str: Optional[str] = None,
) -> bool:
    """
    保存检测时的输入与重建动作视频。
    
    默认模式：保存到 output_dir/{name}/input.mp4 与 reconstructed.mp4。
    叠加模式（overlay=True）：保存到 output_dir/{name}/overlay.mp4，原始与重建重叠，并标注部位误差。

    motion、rec 为标准化空间，会先 inv_transform 再可视化。
    失败时打印警告并返回 False，不中断流程。

    Args:
        motion: 输入动作 (T, 272) 或 (1, T, 272)，已标准化
        rec: 重建动作，形状同 motion
        name: 相对路径名，不含 .npy（如 subdir/motion_corrupt）
        output_dir: 可视化输出根目录
        mean: 272 维均值
        std: 272 维标准差
        fps: 视频帧率
        gt_corrupt_mask: 可选，GT 标注的损坏帧 (T,) bool
        detected_corrupt_mask: 可选，检测判定的损坏帧 (T,) bool
        overlay: 是否采用重叠可视化模式
        per_part_errors: (T, 7) 各部位每帧误差，overlay=True 时使用
        pin_to_origin: 是否在可视化前将动作钉在原地
        gt_intervals_str: 可选，带类型的 GT 区间字符串
    """
    try:
        # 在反标准化前（或反标准化后均可，pin_motion_to_origin 支持两种）执行 pinning
        if pin_to_origin:
            motion = pin_motion_to_origin(motion.clone() if hasattr(motion, "clone") else motion.copy(), mean, std)
            rec = pin_motion_to_origin(rec.clone() if hasattr(rec, "clone") else rec.copy(), mean, std)

        motion_np = motion.cpu().numpy() if isinstance(motion, torch.Tensor) else np.asarray(motion)
        rec_np = rec.cpu().numpy() if isinstance(rec, torch.Tensor) else np.asarray(rec)
        if motion_np.ndim == 3:
            motion_np = motion_np.squeeze(0)
        if rec_np.ndim == 3:
            rec_np = rec_np.squeeze(0)

        motion_denorm = motion_np * std + mean
        rec_denorm = rec_np * std + mean

        folder = os.path.join(output_dir, name.replace("\\", "/"))
        os.makedirs(folder, exist_ok=True)

        if overlay:
            # 重叠可视化模式：仅输出一个视频
            out_path = os.path.join(folder, "overlay.mp4")
            return visualize_motion_overlay_vector272(
                motion_denorm, rec_denorm, out_path,
                per_frame_errors=per_part_errors, fps=fps,
                gt_corrupt_mask=gt_corrupt_mask,
                detected_corrupt_mask=detected_corrupt_mask,
                gt_intervals_str=gt_intervals_str,
            )
        else:
            # 默认模式：输出两个独立视频
            ok1 = visualize_motion_vector272(
                motion_denorm, os.path.join(folder, "input.mp4"), fps=fps,
                gt_corrupt_mask=gt_corrupt_mask, detected_corrupt_mask=detected_corrupt_mask,
                gt_intervals_str=gt_intervals_str,
            )
            ok2 = visualize_motion_vector272(
                rec_denorm, os.path.join(folder, "reconstructed.mp4"), fps=fps,
                gt_corrupt_mask=gt_corrupt_mask, detected_corrupt_mask=detected_corrupt_mask,
                gt_intervals_str=gt_intervals_str,
            )
            return ok1 or ok2
    except Exception as e:
        import warnings
        warnings.warn(f"保存检测可视化失败 {name}: {e}")
        return False
# endregion


# region 损坏判定
def detect_corrupt(
    net: torch.nn.Module,
    motion: Union[torch.Tensor, np.ndarray],
    threshold: float,
    metric: Literal["recon", "quant"] = "recon",
    device: Optional[torch.device] = None,
    pin_to_origin: bool = False,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Tuple[float, bool]:
    """
    单次检测：计算误差并判定是否损坏。

    误差超过 threshold 则判为损坏。阈值需在标定集上根据业务需求确定。
    """
    if metric == "recon":
        err = compute_reconstruction_error(net, motion, device, reduction="mean", pin_to_origin=pin_to_origin, mean=mean, std=std)
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
