![image-20250922202842453](FSQ%E7%9B%B8%E5%85%B3%E6%A2%B3%E7%90%86.assets/image-20250922202842453.png)

#### 1.主流程梳理（forward）

```python
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
```

##### 主调用流程（forward 主路径）

- 入口：FSQ.forward(z)
- 1）标准化形状
  - 将输入从 (b, d, ...) 统一变换为 (b, ..., d)，再用 pack_one 展平为 (b, n, d)，保留还原信息 ps。

- 2）线性映射到码本维度
  - project_in：若 self.dim != codebook_dim * num_codebooks，将 (b, n, d) 映射为 (b, n, c*d_codebook)，否则恒等。

- 3）拆分码本维度
  - rearrange 到 (b, n, c, d_codebook)，其中 c=self.num_codebooks，d_codebook=self.codebook_dim。

- 4）量化（核心）

  - 在特定精度上下文中执行：

    - quantize(z) → bound(z) → round_ste(...) → 返回归一化量化代码，形状仍为 (b, n, c, d_codebook)。

    - 若 return_indices=True，则 codes_to_indices(codes) 得到离散索引 (b, n, c)（或后续视配置去除 c 维）。

- 5）合并维度与还原 dtype
  - rearrange(codes, 'b n c d -> b n (c d)')，再转回原始 dtype。

- 6）线性映射回输出维度
  - project_out：将 (b, n, c*d_codebook) 映射回 (b, n, d)（或恒等）。

- 7）还原输入布局

  - 用 unpack_one(out, ps, 'b * d') 还原到 (b, ..., d)，再 rearrange 为 (b, d, ...)。

  - 若存在 indices，同样 unpack_one(indices, ps, 'b * c')，必要时按配置移除 c 维。

- 8）统计指标
  - compute_perplexity(indices.reshape(-1)) 计算困惑度与激活率。

- 9）返回
  - (out, indices, dummy_loss(=0), perplexity, activate, indices)。

##### 量化子流程

- quantize(z) 调用链

  - bound(z)：按 levels 将每维限制到合适区间（考虑偶数等级的 0.5 偏移），输出与 z 同形状。

  - round_ste(...)：前向四舍五入，反向用直通估计传梯度。

  - 将取整结果除以 half_width = levels // 2，归一化到约 [-1, 1]。

- 用途：供 forward 的主量化，以及编码/解码函数中的中间形态参考。

##### 索引与代码相互转换

- 代码→索引：codes_to_indices(zhat)

  - 断言最后一维等于 codebook_dim。

  - _scale_and_shift(zhat)：把归一化代码（约 [-1,1]）映射到非中心化等级 [0, levels-1]。

  - 与 _basis 按最后一维相乘求和，得到一维索引（进制展开），输出形状为 (...,) 的 int32。

- 索引→代码（两条路径）

  - 低层：_indices_to_codes(indices)

    - indices_to_level_indices(indices)：把一维索引展开为每维等级索引，形状 (..., d)。

    - _scale_and_shift_inverse(level_indices)：把 [0, levels-1] 映射回中心化归一化代码（约 [-1,1]），输出 (..., d)。

  - 高层：indices_to_codes(indices)

    - 先走 _indices_to_codes。

    - 若 keep_num_codebooks_dim=True，将 (..., c, d) 合并为 (..., c*d)。

    - project_out 投影回输出维。

    - 若输入为图像/视频或 channel_first=True，将通道移到前，最终形状与 (b, d, ...) 对齐。

##### 解量化（索引到连续表示）

- dequantize(indices)

- _indices_to_codes(indices) 得到归一化代码 (..., d)。

- project_out 投影回输出维度，得到连续表示。

- 这是不涉及空间/通道还原的简洁版本；如需与 forward 对齐布局，可参考 indices_to_codes。

##### 指标统计

- compute_perplexity(code_idx)

  - 输入为展平的一维索引 (N,)。

  - 构造 one-hot 统计每个码的使用次数 → 概率分布 → 计算困惑度 exp(-sum(p log p))。

  - 激活率为使用过的码比例。

##### 辅助与工具函数

- exists(v)：判断非 None。

- default(*args)：返回第一个非 None。

- maybe(fn)：包装函数，若首参为 None 则直接返回 None，否则调用 fn。

- pack_one(t, pattern) / unpack_one(t, ps, pattern)：einops 的单张量打包/解包，供形状标准化与还原。

- round_ste(z)：取整的直通估计。

##### 模块级依赖关系（概览）

- forward

  - pack_one / rearrange

  - project_in

  - rearrange 到 (b, n, c, d_codebook)

  - quantize

    - bound

    - round_ste

  - 可选 codes_to_indices
    - _scale_and_shift

  - rearrange 合并 (c d)

  - project_out

  - unpack_one / rearrange 还原布局

  - compute_perplexity
- indices_to_codes

  - _indices_to_codes

    - indices_to_level_indices

    - _scale_and_shift_inverse

  - 可选合并 c 维

  - project_out

  - 可选 rearrange 调整通道
- dequantize

  - _indices_to_codes
- project_out





#### 2.为什么要用FSQ

https://spaces.ac.cn/archives/9826/comment-page-2?replyTo=25381

 **FSQ（Finite Scalar Quantization）** 是用来替代 **VQ-VAE** 中的 **VQ（Vector Quantization，向量量化）** 部分的一种更简单、更高效的方法。

##### 1. **VQ-VAE 是做什么的？**

VQ-VAE 是一种自编码器（AutoEncoder），它的作用是将连续的图像数据转换成离散的整数序列（类似于文本中的单词编号），这样就可以用类似处理文本的方式（比如用GPT）来生成图像。

##### 2. **VQ 的复杂性**

传统的 VQ 方法需要维护一个“编码表”（Codebook），通过计算最接近的编码向量来量化输入，训练过程中还需要设计**复杂的梯度回传机制**（如 Straight-Through Estimator, STE）和**额外的损失函数**来稳定训练。

##### 3. **FSQ 的简单之处**

FSQ 的做法非常直接：**对编码向量的每一维进行“四舍五入”**。具体来说，它将每一维的值通过一个sigmoid函数压缩到0~1之间，再乘以一个整数（比如L-1），最后四舍五入取整。这样就实现了离散化，不需要编码表，也不需要额外的损失函数。

##### 4. **FSQ 的优势**

- **更简单**：没有编码表，不需要额外的损失项。
- **更稳定**：避免了VQ中常见的“编码表坍缩”问题（即很多编码向量没被用到）。
- **效果更好**：在编码表较大时，FSQ在图像重建和生成任务上表现优于VQ。

##### 5. **FSQ 的局限性**

当编码表很小的时候（<1000），FSQ的效果可能不如VQ，因为它对编码向量的维度有限制（通常只有几维），而VQ可以支持更高维的编码。

##### 使用FSQ的主要原因有以下几点：

1. **简化模型结构**：FSQ不需要维护复杂的编码表，也不需要设计多个损失函数来稳定训练，大大降低了实现难度。
2. **训练更稳定**：VQ方法容易出现“编码表坍缩”问题（即很多编码向量没有被充分利用），而FSQ通过简单的四舍五入机制避免了这一问题。
3. **效果更好**：当编码表较大时（比如编码数量超过1000），FSQ在图像重建和生成任务上的表现优于VQ。
4. **收敛更快**：由于FSQ的结构更简单，梯度回传更直接，模型训练速度更快。





#### 3.基于llama的结构：







#### 4.函数dequantize()函数的意义：

- 简单说：dequantize 用来把“离散索引”直接还原成“连续向量表示”，方便在推理/解码阶段用离散码重建连续特征。

- 典型用途

  - 从已保存的码本索引重建连续特征，喂给后续网络（解码器/渲染器等）。

  - 采样后（先采样索引）快速转换为连续代码向量，不必走完整的 forward 流程。

  - 离线/后处理场景：你只有 indices，没有原始输入 z，也不需要形状还原或通道搬运。

- 它做了哪些事

  - indices → 每维等级索引（indices_to_level_indices）→ 归一化代码（*scale_and_shift_inverse）。*

  - 再通过 project_out 映射回输出维度，得到连续表示。

- 和 indices_to_codes 的区别

  - 相同点：两者都会把索引还原为连续表示，并用 project_out 投影回输出维。

  - 不同点：

    - indices_to_codes 会根据 keep_num_codebooks_dim、channel_first、是否为图像/视频等，做维度合并/通道位置调整；更贴近 forward 的完整形状语义。

    - dequantize 不处理这些布局细节，只做“索引→代码→投影”的最短路径，接口更轻量，适合你明确只要连续向量、不需要重排的情况。

- 何时用哪个

  - 需要与前向输出布局、通道位置完全对齐时：用 indices_to_codes。

  - 只想把索引变成连续特征向量，后续自己处理形状/不需要还原布局时：用 dequantize。