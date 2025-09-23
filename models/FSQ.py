"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1

本文件实现了 FSQ（Finite Scalar Quantization，有限标量量化）模块，用于离散化连续表示，常用于 VQ-VAE 等模型。
核心思想是将每一维特征独立量化到若干离散等级上，并通过整数组合得到整体码本索引。
"""

from __future__ import annotations  # 未来注解语法支持
from functools import wraps, partial  # 装饰器工具与偏函数
from contextlib import nullcontext  # 空上下文管理器
from typing import List, Tuple  # 类型注解

import torch  # 张量与自动求导库
import torch.nn as nn  # 神经网络模块
from torch.nn import Module  # 基类 Module
from torch import Tensor, int32  # 类型别名
from torch.amp import autocast  # 自动混合精度上下文

from einops import rearrange, pack, unpack  # 张量维度变换工具

# 辅助函数（helper functions）

"""函数说明
输入参数：
- v：任意类型对象
功能：
- 判断对象是否为非 None
执行流程：
- 返回布尔值 v is not None
"""
def exists(v):  # 判断变量是否存在（非 None）
    return v is not None  # True 表示存在

"""函数说明
输入参数：
- *args：可变参数列表（任意类型）
功能：
- 从左到右返回第一个非 None 的参数；若都为 None，返回 None
执行流程：
- 遍历 args，使用 exists 判断，返回第一个为真的项
"""
def default(*args):  # 级联默认值选择器
    for arg in args:  # 依次检查参数
        if exists(arg):  # 若该参数存在
            return arg  # 返回该参数
    return None  # 若均不存在，返回 None
    
"""函数说明
输入参数：
- fn：可调用函数
功能：
- 返回一个包装器：当传入的第一个参数 x 为 None 时，直接返回 None；否则调用原函数 fn
执行流程：
- 定义 inner，检查 x 是否存在，存在则调用 fn，否则返回 x（即 None）
"""
def maybe(fn):  # 安全调用装饰器工厂
    @wraps(fn)  # 保留原函数签名与元信息
    def inner(x, *args, **kwargs):  # x 为被检查对象
        if not exists(x):  # 若 x 为 None
            return x  # 直接返回 None
        return fn(x, *args, **kwargs)  # 否则调用 fn
    return inner  # 返回包装函数

"""函数说明
输入参数：
- t：Tensor，任意形状
- pattern：einops 的 pack 模式字符串
功能：
- 使用 einops.pack 包装单个张量，以统一接口
执行流程：
- 调用 pack([t], pattern)
"""
def pack_one(t, pattern):  # 单张量 pack 包装
    return pack([t], pattern)  # 返回 (张量, 辅助信息)

"""函数说明
输入参数：
- t：Tensor，经 pack 处理后的张量
- ps：pack 返回的辅助信息（元组/列表）
- pattern：einops 的 unpack 模式字符串
功能：
- 对单个张量进行 unpack 并取第一个结果
执行流程：
- 调用 unpack(t, ps, pattern) 并取第 0 个元素
"""
def unpack_one(t, ps, pattern):  # 单张量 unpack 解包
    return unpack(t, ps, pattern)[0]  # 取第一个（对应 pack 时的单元素）

# 张量相关辅助

"""函数说明
输入参数：
- z：Tensor，任意形状与 dtype 的张量
功能：
- 采用 STE（Straight-Through Estimator）实现的四舍五入：前向使用 round，反向梯度穿透
执行流程：
- 先 z.round() 得到 zhat；返回 z + (zhat - z).detach()，从而反向梯度等同于恒等映射
输出：
- Tensor，与 z 同形状
"""
def round_ste(z: Tensor) -> Tensor:  # 带直通估计的取整
    zhat = z.round()  # 前向量化取整
    return z + (zhat - z).detach()  # 反向对 z 传递梯度

# 主体类实现（FSQ）

class FSQ(Module):  # 有限标量量化器
    """类说明
    主要参数：
    - levels：List[int]，每个维度的量化等级数，例如 [8, 8, 8] 表示 3 维、每维 8 个等级
    - dim：输入特征维度（若为 None，默认使用 len(levels) * num_codebooks）
    - num_codebooks：并行的独立码本数（factorized codebooks）
    - keep_num_codebooks_dim：是否保留码本维度（True 时输出形状分离 c 维）
    - scale：可选缩放（未在此实现中使用）
    - allowed_dtypes：强制量化步骤允许的 dtype（默认为 (float32, float64)）
    - channel_first：输入/输出是否为通道优先格式（如 (b, d, ...)）
    - projection_has_bias：线性投影是否带 bias
    - return_indices：是否返回离散码索引
    - force_quantization_f32：是否强制量化在 float32 中执行以稳定数值

    主要功能：
    - 将输入特征映射到若干并行码本的低维空间，逐维量化后再映射回原维度
    - 提供编码（codes_to_indices）与解码（indices_to_codes/dequantize）能力
    - 计算码本使用的困惑度与激活比例
    """
    def __init__(
        self,
        levels: List[int],  # 每一维的量化等级数量
        dim: int | None = None,  # 输入特征维度
        num_codebooks = 1,  # 码本数量
        keep_num_codebooks_dim: bool | None = None,  # 是否保留码本维度
        scale: float | None = None,  # 可选缩放系数
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),  # 量化允许的 dtype
        channel_first: bool = False,  # 是否通道在前
        projection_has_bias: bool = True,  # 线性层是否带偏置
        return_indices = True,  # 是否返回离散索引
        force_quantization_f32 = True  # 是否在量化中强制 float32
    ):
        super().__init__()  # 初始化 Module
        _levels = torch.tensor(levels, dtype=int32)  # 将等级列表保存为 Tensor[int32]
        self.register_buffer("_levels", _levels, persistent = False)  # 注册为缓冲区（不随模型保存持久化）

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)  # 进制基数，用于多维索引到一维编码
        self.register_buffer("_basis", _basis, persistent = False)  # 注册缓冲区

        self.scale = scale  # 预留缩放参数

        codebook_dim = len(levels)  # 每个码本的向量维度（等于 levels 的长度）
        self.codebook_dim = codebook_dim  # 保存码本维度

        effective_codebook_dim = codebook_dim * num_codebooks  # 总的（合并后）码本维度
        self.num_codebooks = num_codebooks  # 并行码本数
        self.effective_codebook_dim = effective_codebook_dim  # 保存总维度

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)  # 多码本时默认保留 c 维
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)  # 多码本必须保留 c 维
        self.keep_num_codebooks_dim = keep_num_codebooks_dim  # 标记是否保留 c 维

        self.dim = default(dim, len(_levels) * num_codebooks)  # 输入维度，默认每码本维度相加

        self.channel_first = channel_first  # 记录通道顺序

        has_projections = self.dim != effective_codebook_dim  # 判断是否需要投影层
        # 将输入映射到码本维度（若维度不一致）
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        # 将码本维度映射回输出维度
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections  # 记录是否存在投影

        self.return_indices = return_indices  # 是否返回索引
        if return_indices:
            self.codebook_size = self._levels.prod().item()  # 码本大小 = 各维等级数乘积
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))  # 隐式码本（所有组合对应的代码）
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)  # 注册缓冲
     
        self.allowed_dtypes = allowed_dtypes  # 允许的 dtype
        self.force_quantization_f32 = force_quantization_f32  # 是否强制使用 float32 量化

    """函数说明
    输入参数：
    - z：Tensor，形状 (..., d)，d = codebook_dim（或 broadcast 到该维度）
    - eps：float，微小余量，防止数值边界问题，默认 1e-3
    功能：
    - 将输入 z 限制到量化等级范围（中心化后为 [-half_l, half_l]），考虑偶数等级时的 0.5 偏移
    执行流程：
    - 计算每维的半宽 half_l 与 offset，根据偶/奇数等级数处理中心偏移，再用 tanh 限制范围
    输出：
    - Tensor，与 z 同形状，值域被限制
    """
    def bound(self, z, eps: float = 1e-3):  # 对输入进行有界化
        half_l = (self._levels - 1) * (1 + eps) / 2  # 每维半宽
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)  # 偶数等级则中心偏移 0.5
        shift = (offset / half_l).atanh()  # 计算用于平移的 atanh 值
        return (z + shift).tanh() * half_l - offset  # tanh 限幅，再缩放回目标区间

    """函数说明
    输入参数：
    - z：Tensor，形状 (..., c, d) 或 (..., d)，d = codebook_dim
    功能：
    - 对每维执行有界化 + 取整（STE），并归一化到 [-1, 1]
    执行流程：
    - 调用 bound 有界化，再用 round_ste 取整，然后除以 half_width 完成归一化
    输出：
    - Tensor，形状与 z 相同，值域约为 [-1, 1]
    """
    def quantize(self, z):  # 量化并归一化
        quantized = round_ste(self.bound(z))  # 有界化后取整（STE）
        half_width = self._levels // 2  # 半宽用于归一化到 [-1, 1]
        return quantized / half_width  # 归一化
    
    """函数说明
    输入参数：
    - zhat_normalized：Tensor，已归一化代码（约在 [-1, 1]）
    功能：
    - 将归一化代码缩放回非中心化等级索引区间 [0, levels-1]
    执行流程：
    - 乘以 half_width 再加上 half_width
    """
    def _scale_and_shift(self, zhat_normalized):  # 归一化代码 -> 非中心化等级
        half_width = self._levels // 2  # 半宽
        return (zhat_normalized * half_width) + half_width  # 缩放平移
    
    """函数说明
    输入参数：
    - zhat：Tensor，非中心化等级（[0, levels-1]）
    功能：
    - 将等级值还原为中心化并归一化的代码（约在 [-1, 1]）
    执行流程：
    - 减去 half_width，再除以 half_width
    """
    def _scale_and_shift_inverse(self, zhat):  # 非中心化等级 -> 归一化代码
        half_width = self._levels // 2  # 半宽
        return (zhat - half_width) / half_width  # 逆向缩放平移

    """函数说明
    输入参数：
    - indices：Tensor，形状 (...,)，码本的一维整型索引
    功能：
    - 将一维索引还原为每维等级索引，再映射为中心化归一化代码
    执行流程：
    - 调用 indices_to_level_indices，再调用 _scale_and_shift_inverse
    输出：
    - Tensor，形状 (..., d)，d = codebook_dim
    """
    def _indices_to_codes(self, indices):  # 一维索引 -> 归一化代码
        level_indices = self.indices_to_level_indices(indices)  # 还原为逐维索引
        codes = self._scale_and_shift_inverse(level_indices)  # 转为归一化代码
        return codes  # 返回代码

    """函数说明
    输入参数：
    - zhat：Tensor，形状 (..., d)，d = codebook_dim，中心化归一化代码（[-1,1]）
    功能：
    - 将每维代码映射到非中心化等级，并按进制组合为一维索引
    执行流程：
    - _scale_and_shift 得到 [0, levels-1]，与 _basis 相乘后在最后一维求和
    输出：
    - Tensor[int32]，形状 (...,)
    """
    def codes_to_indices(self, zhat):  # 代码 -> 一维索引
        assert zhat.shape[-1] == self.codebook_dim  # 维度检查
        zhat = self._scale_and_shift(zhat)  # 归一化代码转非中心化等级
        return (zhat * self._basis).sum(dim=-1).to(int32)  # 进制展开为一维索引

    """函数说明
    输入参数：
    - indices：Tensor[int]，形状 (...,)
    功能：
    - 将一维索引展开为逐维的非中心化等级索引（每维范围 [0, levels_i-1]）
    执行流程：
    - 扩展末维，使用整除与取模对每维求值
    输出：
    - Tensor，形状 (..., d)
    """
    def indices_to_level_indices(self, indices):  # 一维索引 -> 逐维等级索引
        indices = rearrange(indices, '... -> ... 1')  # 扩展维度以便广播
        codes_non_centered = (indices // self._basis) % self._levels  # 进制展开 + 逐维取模
        return codes_non_centered  # 返回逐维索引

    """函数说明
    输入参数：
    - indices：Tensor[int]，形状可为 (b, n, c) 或 (b, n) 或更低维；c 为码本数（当 keep_num_codebooks_dim=True 时）
    功能：
    - 将索引还原为连续代码，并映射回输出维度与布局
    执行流程：
    - 调用 _indices_to_codes 得到逐维代码；若保留 c 维则合并 (c d)->d；随后 project_out；
      若输入是图像/视频或 channel_first=True，则将通道移动到前面
    输出：
    - Tensor，形状与 forward 的输出对齐（通常为 (b, d, ...)）
    """
    def indices_to_codes(self, indices):  # 索引 -> 连续代码（解码）
        assert exists(indices)  # 索引必须存在

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))  # 根据维度猜测是否为图像/视频

        codes = self._indices_to_codes(indices)  # 先得到逐维归一化代码

        if self.keep_num_codebooks_dim:  # 若保留 c 维，合并 (c d)
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)  # 投影回输出维度

        if is_img_or_video or self.channel_first:  # 恢复通道优先布局
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes  # 返回连续代码
    
    """函数说明
    输入参数：
    - code_idx：Tensor[int]，形状 (N,) 或 (N,)，为已展平的码本索引序列
    功能：
    - 统计码本使用分布，计算困惑度（perplexity）与激活比例（使用过的码数量占比）
    执行流程：
    - 根据索引构建 one-hot 计数，得到每个码的出现次数；
      归一化为概率分布后计算信息熵的指数即困惑度；
      激活比例为出现次数>0 的码数除以码本大小
    输出：
    - (perplexity: Tensor[float], activate: Tensor[float])
    注意：
    - 使用 no_grad，统计过程不参与梯度
    """
    @torch.no_grad()
    def compute_perplexity(self, code_idx) :  # 计算困惑度与激活率
        # Calculate new centres / 统计 one-hot
        code_onehot = torch.zeros(self.codebook_size, code_idx.shape[0], device=code_idx.device)  # 形状 [codebook_size, N]
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)  # 根据索引填充 1

        code_count = code_onehot.sum(dim=-1)  # 每个码的计数
        prob = code_count / torch.sum(code_count)  # 概率分布
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))  # 困惑度（避免 log(0)）
        activate = torch.sum(code_count > 0).float() / self.codebook_size  # 激活比例
        return perplexity, activate  # 返回统计值

    """函数说明
    输入参数：
    - indices：Tensor[int]，一维索引（形状可广播）
    功能：
    - 解量化：将离散索引映射为连续表示，并投影回输出维度
    执行流程：
    - _indices_to_codes -> project_out
    输出：
    - Tensor，连续表示
    """
    def dequantize(self, indices):  # 解量化
        codes = self._indices_to_codes(indices)  # 索引 -> 归一化代码
        out = self.project_out(codes)  # 投影回输出维度
        return out  # 返回解码结果
    
    """函数说明
    输入参数：
    - z：Tensor
      常见形状：
      - (b, d, ...) 当 channel_first=True 时或图像/视频布局
      - (b, ..., d) 当通道在最后
      其中 d = self.dim（若不同会自动通过 project_in 调整）
    功能：
    - 将输入映射到码本空间，进行逐维量化，返回量化后的连续表示、索引及统计信息
    执行流程：
    1) 统一到 (b, n, d) 形式；
    2) project_in 到有效码本维度，并 reshape 为 (b, n, c, d_codebook)；
    3) 在指定的精度上下文中调用 quantize 得到 codes；可选计算 indices；
    4) 合并 (c d) 并还原 dtype；
    5) project_out 回到输出维度；
    6) 如需，恢复通道与空间布局；
    7) 计算困惑度与激活率；
    8) 返回 (out, indices, dummy_loss, perplexity, activate, indices)
    输出：
    - out：Tensor，量化后的连续表示，形状与输入布局一致（通道维为 d）
    - indices：Tensor[int]，离散索引（可能保留 c 维）
    - dummy_loss：Tensor[float]，占位损失（0）
    - perplexity：Tensor[float]，码本困惑度
    - activate：Tensor[float]，码本激活比例
    - indices：重复返回，兼容外部期望的接口
    """
    def forward(self, z):  # 前向量化与反量化主流程
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        # is_img_or_video = z.ndim >= 4
        # need_move_channel_last = is_img_or_video or self.channel_first

        need_move_channel_last = True  # 统一将通道移到最后以便 pack

        # 将输入标准化为 (batch, seq, dimension)
        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')  # 通道后移
            z, ps = pack_one(z, 'b * d')  # 展平成 (b, n, d)，并获取还原信息 ps

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'  # 维度检查

        z = self.project_in(z)  # 若需要，映射到有效码本维度

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)  # 拆分出码本维 c 与每码本维 d

        # 决定是否强制量化步骤使用 float32 精度
        force_f32 = self.force_quantization_f32  # 标志位
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext  # 上下文选择

        with quantization_context():  # 进入量化精度上下文
            orig_dtype = z.dtype  # 记录原始 dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:  # 若强制 f32 且原 dtype 不在允许集合
                z = z.float()  # 转为 float32

            codes = self.quantize(z)  # 执行量化，得到归一化代码
            # returning indices could be optional

            indices = None  # 默认不返回索引

            if self.return_indices:  # 如需索引
                indices = self.codes_to_indices(codes)  # 代码 -> 一维索引

            codes = rearrange(codes, 'b n c d -> b n (c d)')  # 合并回 (b, n, d)

            codes = codes.type(orig_dtype)  # 恢复原 dtype

        # 投影回输出维度
        out = self.project_out(codes)

        # 还原通道与空间维度
        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')  # 还原到 (b, ..., d)
            out = rearrange(out, 'b ... d -> b d ...')  # 通道前移

            indices = maybe(unpack_one)(indices, ps, 'b * c')  # 若存在索引，同样还原 seq 维

        if not self.keep_num_codebooks_dim and self.return_indices:  # 若不保留 c 维
            indices = maybe(rearrange)(indices, '... 1 -> ...')  # 去除多余维度

        # 返回量化结果与统计
        perplexity, activate = self.compute_perplexity(indices.reshape(-1).to(torch.int64))  # 统计指标

        dummy_loss = torch.tensor(0.0, device=indices.device)  # 占位损失

        return out, indices, dummy_loss, perplexity, activate, indices  # 返回多个项