![](FSQ%E7%9B%B8%E5%85%B3%E6%A2%B3%E7%90%86.assets/image-20250922202842453.png)

#### 1.主流程梳理（forward）

**输入预处理** → 2. **投影变换** → 3. **维度重组** → **4. 量化处理（本代码块）** → 5. **输出投影** → 6. **维度还原**

- 功能：
    - 将输入映射到码本空间，进行逐维量化，返回量化后的连续表示、索引及统计信息
- 执行流程：
    1) 统一到 (b, n, d) 形式；
    2) project_in 到有效码本维度，并 reshape 为 (b, n, c, d_codebook)；
    3) 在指定的精度上下文中调用 quantize 得到 codes；可选计算 indices；
    4) 合并 (c d) 并还原 dtype；
    5) project_out 回到输出维度；
    6) 如需，恢复通道与空间布局；
    7) 计算困惑度与激活率；
    8) 返回 (out, indices, dummy_loss, perplexity, activate, indices)
- 输出：
    - out：Tensor，量化后的连续表示，形状与输入布局一致（通道维为 d）
    - indices：Tensor[int]，离散索引（可能保留 c 维）
    - dummy_loss：Tensor[float]，占位损失（0）
    - perplexity：Tensor[float]，码本困惑度
    - activate：Tensor[float]，码本激活比例
    - indices：重复返回，兼容外部期望的接口

```python
	def forward(self, z):

    # 判断是否需要将通道维度移到最后（默认True）
    # 如果是图像/视频数据(ndim>=4)或指定channel_first，通常需要此操作
    need_move_channel_last = True  # 统一将通道移到最后以便 pack

    # 标准化输入格式：将输入张量统一转换为(batch, seq, dimension)形式
    if need_move_channel_last:
        # 将通道维度从第1位移动到末尾： (b, d, ...) -> (b, ..., d)
        z = rearrange(z, 'b d ... -> b ... d')
        # 打包张量：将空间维度展平为序列维度
        # 输入: (b, s1, s2, ..., d) 输出: (b, n, d) 其中 n = s1*s2*...
        z, ps = pack_one(z, 'b * d')  # ps保存了原始形状信息用于后续解包
```
##### 1.确保输入张量的特征维度与FSQ模型期望的维度完全一致
防止因维度不匹配导致的运行时错误或计算异常。

```python
    assert z.shape[-1] == self.dim, \
        f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
```


##### 2.将输入投影到有效码本维度（如果设置的话）

```python
    # 输入: (b, n, dim) 输出: (b, n, effective_codebook_dim)
    z = self.project_in(z)
```
相关的是在init函数的166行
```python
# 将输入映射到码本维度（若维度不一致）
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()

```


##### 3.将原来的c*d维度拆分为两个独立维度

```python
    # 重新排列维度，拆分出码本数量维度
    # 输入: (b, n, c*d) 输出: (b, n, c, d) 其中c=num_codebooks, d=codebook_dim
    z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)
```


##### 4.创建量化上下文

```python
    # 决定是否强制量化步骤使用float32精度
    force_f32 = self.force_quantization_f32
    # 创建量化上下文：强制f32时使用autocast禁用半精度，否则使用空上下文
    quantization_context = partial(autocast, 'cuda', enabled=False) if force_f32 else nullcontext
```


##### 5.量化处理的核⼼操作部分

1. **精度管理**：控制量化计算的数值精度
2. **核心量化**：将连续特征转换为离散代码
3. **索引生成**：可选地生成离散码本索引
4. **维度重组**：调整输出格式以适应后续处理
5. **类型恢复**：保持输出数据类型的一致性

###### 数值变化示例

假设一个简单情况：

- 量化等级：`levels = [4, 4]`（每维4个离散值）
- 输入值：`z = [[[0.2, -0.3]]]`（shape: (1, 1, 2)）

处理过程：

1. **有界化**：限制值在有效范围内
2. **取整**：`round(0.2 * 1.5 + 1.5) = 2`, `round(-0.3 * 1.5 + 1.5) = 1`
3. **归一化**：`2/1.5 ≈ 1.33`, `1/1.5 ≈ 0.67`（然后裁剪到[-1,1]）
4. **索引计算**：如`[2, 1]`可能对应一维索引`2 * 4 + 1 = 9`

```python
    with quantization_context():  # 进入量化精度上下文,创建受控的计算环境，确保量化操作在指定精度下进行,避免低精度(float16/bfloat16)下的舍入误差
        
        orig_dtype = z.dtype  # 保存输入张量的原始数据类型(torch.float16, torch.float32等)
        # 如果需要强制float32且当前dtype不在允许列表中，转换为float32
        if force_f32 and orig_dtype not in self.allowed_dtypes:
            z = z.float()
```
执行量化操作，得到归一化代码
```python
        # 输入: (b, n, c, d) 输出: (b, n, c, d) （取值范围[-1, 1]）
        codes = self.quantize(z)
```
相关函数如下：
```python
def quantize(self, z):  # 量化并归一化
        quantized = round_ste(self.bound(z))  # 有界化后取整（STE）
        half_width = self._levels // 2  # 半宽用于归一化到 [-1, 1]
        return quantized / half_width  # 归一化
```
​	1)**有界化处理**：首先调用`self.bound(z)`将输入值限制在特定范围内

```python
# 计算每维的半宽和偏移量 
half_l = (self._levels - 1) * (1 + eps) / 2 offset = torch.where(self._levels % 2 == 0, 0.5, 0.0) shift = (offset / half_l).atanh() return (z + shift).tanh() * half_l - offset
```
​	2)**取整操作**：使用STE(Straight-Through Estimator)技巧进行四舍五入

```python
zhat = z.round()  # 前向传播时取整 
return z + (zhat - z).detach()  # 反向传播时梯度穿透
```
​	3)**归一化**：将整数值缩放到[-1, 1]范围

```python
half_width = self._levels // 2 return quantized / half_width  # 归一化到[-1, 1]
```
索引生成（可选），将多维离散代码转换为一维索引
```python
        indices = None  # 初始化索引为None
        
        # 如果设置返回索引，计算量化后的一维索引
        if self.return_indices:
            # 输入: (b, n, c, d) 输出: (b, n, c) 或 (b, n) 取决于keep_num_codebooks_dim
            indices = self.codes_to_indices(codes)
```
重新组合维度，将码本维度合并回特征维度，为后续的投影操作做准备
```python
        # 输入: (b, n, c, d) 输出: (b, n, c*d)
        codes = rearrange(codes, 'b n c d -> b n (c d)')
        
        # 恢复原始数据类型，确保与后续网络层的类型兼容性
        codes = codes.type(orig_dtype)
```


##### 6.投影回输出维度

```python
    # 输入: (b, n, effective_codebook_dim) 输出: (b, n, dim)
    out = self.project_out(codes)
```


##### 7.还原通道与空间维度

```python
    # 还原通道与空间维度（如果需要）
    if need_move_channel_last:
        # 解包张量：将序列维度还原为空间维度
        # 输入: (b, n, dim) 输出: (b, s1, s2, ..., dim)
        out = unpack_one(out, ps, 'b * d')
        # 将通道维度移回原来的位置： (b, ..., d) -> (b, d, ...)
        out = rearrange(out, 'b ... d -> b d ...')
        
        # 如果存在索引，同样进行解包操作
        indices = maybe(unpack_one)(indices, ps, 'b * c')

    # 如果不保留码本数量维度且返回索引，去除多余的维度
    if not self.keep_num_codebooks_dim and self.return_indices:
        indices = maybe(rearrange)(indices, '... 1 -> ...')

    # 计算困惑度和激活率（如果返回索引）
    perplexity, activate = self.compute_perplexity(
        indices.reshape(-1).to(torch.int64))  # 将索引展平为一维

    # 创建占位损失（为了兼容某些接口）
    dummy_loss = torch.tensor(0.0, device=indices.device)

    # 返回多个输出：量化后的连续表示、索引、占位损失、统计指标
    return out, indices, dummy_loss, perplexity, activate, indices
```

### 输入输出维度变化总结

| 步骤 | 操作     | 输入维度                                   | 输出维度                       | 说明             |
| :--- | :------- | :----------------------------------------- | :----------------------------- | :--------------- |
| 1    | 输入     | (b, d, s1, s2, ...) 或 (b, s1, s2, ..., d) | -                              | 多种可能输入格式 |
| 2    | 通道重排 | (b, d, ...)                                | (b, ..., d)                    | 统一通道在最后   |
| 3    | 打包     | (b, s1, s2, ..., d)                        | (b, n, d)                      | 空间维度展平     |
| 4    | 投影输入 | (b, n, dim)                                | (b, n, effective_codebook_dim) | 线性变换         |
| 5    | 重排     | (b, n, c*d)                                | (b, n, c, d)                   | 拆分码本维度     |
| 6    | 量化     | (b, n, c, d)                               | (b, n, c, d)                   | 取值范围变化     |
| 7    | 索引计算 | (b, n, c, d)                               | (b, n, c) 或 (b, n)            | 可选步骤         |
| 8    | 重排     | (b, n, c, d)                               | (b, n, c*d)                    | 合并码本维度     |
| 9    | 投影输出 | (b, n, effective_codebook_dim)             | (b, n, dim)                    | 线性变换         |
| 10   | 解包     | (b, n, dim)                                | (b, s1, s2, ..., dim)          | 恢复空间维度     |
| 11   | 通道重排 | (b, ..., d)                                | (b, d, ...)                    | 恢复通道位置     |

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

##### 1. **VQ-VAE 用途**

VQ-VAE 是一种自编码器（AutoEncoder），它的作用是将连续的图像数据转换成离散的整数序列（类似于文本中的单词编号），这样就可以用类似处理文本的方式（比如用GPT）来生成图像。

##### 2. **VQ 原理**

传统的 VQ 方法需要维护一个“编码表”（Codebook），通过计算最接近的编码向量来量化输入，训练过程中还需要设计**复杂的梯度回传机制**（如 Straight-Through Estimator, STE）和**额外的损失函数**来稳定训练。

##### 3. **FSQ 原理**

FSQ 的做法非常直接：**对编码向量的每一维进行“四舍五入”**。具体来说，它将每一维的值通过一个sigmoid函数压缩到0~1之间，再乘以一个整数（比如L-1），最后四舍五入取整。这样就实现了离散化，不需要编码表，也不需要额外的损失函数。

##### 4. **FSQ 优势**

- **更简单**：没有编码表，不需要额外的损失项。
- **更稳定**：避免了VQ中常见的“编码表坍缩”问题（即很多编码向量没被用到）。
- **效果更好**：在编码表较大时，FSQ在图像重建和生成任务上表现优于VQ。

##### 5. **FSQ 局限性**

当编码表很小的时候（<1000），FSQ的效果可能不如VQ，因为它对编码向量的维度有限制（通常只有几维），而VQ可以支持更高维的编码。

##### 使用FSQ的主要原因：

1. **简化模型结构**：FSQ不需要维护复杂的编码表，也不需要设计多个损失函数来稳定训练，大大降低了实现难度。
2. **训练更稳定**：VQ方法容易出现“编码表坍缩”问题（即很多编码向量没有被充分利用），而FSQ通过简单的四舍五入机制避免了这一问题。
3. **效果更好**：当编码表较大时（比如编码数量超过1000），FSQ在图像重建和生成任务上的表现优于VQ。
4. **收敛更快**：由于FSQ的结构更简单，梯度回传更直接，模型训练速度更快。





#### 3.基于llama的结构：

##### 完整处理流水线

```python
def visualize_smplx_85(data, title=None, output_path='./recon_272/0_14_rot_new3.mp4', fps=60):    
"""    
可视化SMPLX数据，生成动作视频   
输入:        
	data: numpy数组, 形状通常为 (nframes, 85) 或 (1, nframes, 85)，SMPLX格式的动作数据。            
	85维数据包含：前66维是身体姿态参数（轴角表示），随后是6维零填充，3维根节点位移，10维零填充。        
	title: 字符串, 可选的标题，当前函数未直接使用，可能供其他可视化函数使用。        output_path: 字符串, 输出视频文件路径。        
	fps: 整数, 生成视频的帧率。
输出:        
	无直接返回值。函数会生成一个GIF动画和一个MP4视频文件，保存在output_path指定的路径。    
"""    
	# 将输入数据赋值给局部变量    
	smplx_85_data = data    
	# 检查输入数据的维度，如果为3维（例如批量处理时是[1, nframes, 85]），则压缩掉第一个批次维度    
	if len(smplx_85_data.shape) == 3:       
		smplx_85_data = np.squeeze(smplx_85_data, axis=0)    # 调用函数将85维的SMPLX数据转换为322维的完整SMPLX数据格式（通过零填充其他参数）    # 需要查看 smplx85_2_smplx322 函数的实现以确认输出维度，推测为 (nframes, 322)    	
		smplx_85_data = smplx85_2_smplx322(smplx_85_data)    # 处理SMPLX数据，获取网格顶点、关节位置、动作数据和面信息    
		# 输入: smplx_85_data (nframes, 322), norm_global_orient=False, transform=False    
		# 输出:     
			#   vert: 顶点坐标，具体维度需查看 process_smplx_data 函数，推测为 (nframes, 10475, 3) 
			#   joints: 关节坐标，具体维度需查看 process_smplx_data 函数，推测为 (nframes, n_joints, 3)。此项目通常使用22个关节。   
			#   motion: 可能包含其他运动信息，具体需查看函数实现。    
			#   faces: 网格的面信息，用于渲染。    
		vert, joints, motion, faces = process_smplx_data(smplx_85_data, norm_global_orient=False, transform=False)    
		# 从所有关节中提取前22个主要关节的3D坐标，并确保数据在CPU上且转换为numpy数组    
		# 操作: joints[:, :22, :] 选取前22个关节 -> reshape 确保形状为 (nframes, 22, 3)    
		# 输出 xyz: numpy数组, 形状为 (nframes, 22, 3)    
		xyz = joints[:, :22, :].reshape(-1, 22, 3).detach().cpu().numpy()    # 创建输出文件所在的目录，如果目录不存在则创建    
		os.makedirs(os.path.dirname(output_path), exist_ok=True)    
		# 调用 plot_3d_motion 函数生成3D动作序列的图像帧列表    
		# 输入: 一个列表，包含关节坐标数据 [xyz, None, None]。后两个None可能是为其他数据预留的位置。    
		# 输出 img: 一个列表，包含一系列图像帧（例如PIL图像对象或numpy数组），用于生成动画。    
		img = plot_3d_motion([xyz, None, None])    
		# 使用 imageio 将图像帧列表保存为GIF动画文件    
		imageio.mimsave(output_path, np.array(img), fps=fps)    
		# 使用 moviepy 读取刚才生成的GIF文件    
		out_video = mp.VideoFileClip(output_path)    # 将视频文件转换为MP4格式并保存，文件名通过替换扩展名得到    
		out_video.write_videofile(output_path.replace('.gif', '.mp4'))
```
2. **运动量化**：运动数据通过FSQ被离散化为token序列
3. **特征融合**：文本特征与运动token在嵌入空间对齐
4. **自回归生成**：LLaMA基于文本条件生成运动token序列
5. **运动重建**：生成的离散token通过FSQ解码器恢复为连续运动

inferance_batch里首先将输入文本转换为高维特征表示，支持CLIP和T5两种编码方式，为后续的多模态融合提供统一的特征空间。
```python
if args.text_encode == 'clip':
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)
    feat_clip_text = clip_model.encode_text(text).float()
elif args.text_encode == 'flan-t5-xl':
    tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xl')
    text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xl')
    feat_clip_text = text_encoder(input_ids=cap_inputs.input_ids, 
                                attention_mask=cap_inputs.attention_mask).last_hidden_state
```

FSQ的核心量化操作 将连续运动序列离散化为有限的标量等级，生成类似文本token的离散表示，为LLaMA处理做准备。
```python
def quantize(self, z):
    quantized = round_ste(self.bound(z))  # 有界化后取整（STE）
    half_width = self._levels // 2  # 半宽用于归一化
    return quantized / half_width  # 归一化到[-1, 1]
```

llama作为核心的生成引擎，处理文本和运动token的序列建模任务，通过自回归方式生成动作序列。
```python
class LLaMAHF(nn.Module):
    def __init__(self, config: LLaMAHFConfig) -> None:
        super().__init__()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size-1, bias=False)
        self.transformer = nn.ModuleDict(
            dict(wte=nn.Embedding(config.vocab_size, config.n_embd),
                 h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                 ln_f=RMSNorm(config.n_embd))
        )
        self.llama_proj = nn.Linear(config.clip_dim, config.n_embd)
```






#### 4.函数dequantize()函数的意义：

​	在FSQ（有限标量量化）架构中，`dequantize()`函数扮演着**从离散表示重建连续特征**的关键角色，它是量化过程的逆操作，实现了从离散索引空间回到连续特征空间的映射。

​	`dequantize()`是FSQ模型中的**解码器功能函数**，主要作用是将离散的码本索引转换回连续的特征表示，完成"索引 → 代码 → 连续特征"的反向转换过程。

![image-20250923171559554](FSQ%E7%9B%B8%E5%85%B3%E6%A2%B3%E7%90%86.assets/image-20250923171559554.png)

`dequantize()`函数在整体架构中处于**解码阶段**：



1. **编码路径**：连续特征 → 投影 → 量化 → 离散索引
2. **解码路径**：离散索引 → 解量化 → 投影 → 连续特征（本函数）

##### 主要应用场景

- 在文本到动作生成等任务中，模型首先生成离散的索引序列，然后通过 `dequantize()`将这些索引转换回连续的动作表示。

- 研究人员可以使用此函数将学习到的离散表示转换回可解释的连续空间，进行可视化和分析。

- 在评估重建质量时，通过比较原始输入与解量化后的输出，计算重构误差等指标。