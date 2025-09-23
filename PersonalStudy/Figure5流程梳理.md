![image-20250922132055947](Figure5%E6%B5%81%E7%A8%8B%E6%A2%B3%E7%90%86.assets/image-20250922132055947.png)

### Transformer部分

![image-20250921152359617](Figure5%E6%B5%81%E7%A8%8B%E6%A2%B3%E7%90%86.assets/image-20250921152359617.png)

#### 1. **Wavelet Transform预处理（Motion Encoder前端）**

- **对应文档**：`models/vqvae.py`
- **关键代码段**：9-72行初始化vqvae

```python
class VQVAE_251(nn.Module):
    def __init__(self,...use_patcher=False,patch_size=1,patch_method="haar",...):
...
        if args.causal:
            print('use causal conv !!!')
            self.encoder = CausalEncoder(..., use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method, use_attn=use_attn)
            self.decoder = CausalDecoder(..., use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method, use_attn=use_attn)
        else:
            self.encoder = Encoder(..., use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method)
            self.decoder = Decoder(..., use_patcher=use_patcher, patch_size=patch_size, patch_method=patch_method)
...
```

- **说明**：通过`use_patcher`和`patch_method="haar"`参数启用小波变换（Haar wavelet），作为Motion Encoder的前置处理。该部分在论文中描述为**wavelet transformation预处理运动数据**，旨在减少FSQ离散化带来的高频信息损失。



#### 2. **Motion Encoder（特征提取）**

- **对应文档**：`models/vqvae.py`
- **关键代码段**：86-103行：

```python
def encode(self, x):
        
        if self.quant in ["LFQ", "BSQ", "FSQ"]:
            N, T, _ = x.shape
            x_in = self.preprocess(x)
            x_encoder = self.encoder(x_in)
            _, code_idx, _, _, _, _ = self.quantizer(x_encoder)
            code_idx = code_idx.view(N, -1)
            return code_idx
        else:
            N, T, _ = x.shape
            x_in = self.preprocess(x)
            x_encoder = self.encoder(x_in)
            x_encoder = self.postprocess(x_encoder)
            x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
            code_idx = self.quantizer.quantize(x_encoder)
            code_idx = code_idx.view(N, -1)
            return code_idx
```

- **说明**：编码器将预处理后的运动数据转换为潜在特征，输出维度为`output_emb_width`（默认为512维）。该模块是FSQ量化的输入前端。



#### 3. **Autoregressive Transformer（运动生成）**

- **对应文档**：`models/lit_llama/model_hf.py`、inference_batch.py

- **关键代码段**：sample与forward_sample部分、inference_batch.py的__main__部分

<details>
    <summary>
        1. sample函数
    </summary>
    <ul>
        <li>
            <strong>函数功能：</strong>使用自回归方式逐步生成文本序列
        </li>
        <li>
            <strong>输入参数：</strong>
            <ul>
                <li>clip_feature: CLIP提取的视觉特征，作为生成的条件</li>
                <li>y_mask: 掩码张量，用于确定CLIP特征的有效长度</li>
                <li>if_categorial: 是否使用随机采样（否则使用贪心采样）</li>
            </ul>
        </li>
        <li>
            <strong>函数过程：</strong>
            <ul>
                <li>禁用梯度计算，因为这是推理阶段。</li>
                <li>循环最多51次（即最多生成50个token，因为第一次循环时k=0，然后最多再生成50个，所以最多50个token）。</li>
                <li>在第一次循环时，序列x为空，然后调用forward_sample得到logits（此时只使用CLIP特征，没有文本token）。</li>
                <li>取最后一个时间步的logits，然后通过softmax得到概率分布。</li>
                <li>根据if_categorial选择采样方式：
                    <ul>
                        <li>如果为True，则使用分类分布进行随机采样，并检查采样到的token是否为结束token（vocab_size-2），如果是则停止生成。</li>
                        <li>如果为False，则使用贪心采样（取概率最大的token），同样检查是否为结束token。</li>
                    </ul>
                </li>
                <li>将新生成的token添加到序列中。</li>
                <li>如果达到最大长度（50个token），则返回序列（去掉最后一个token，因为最后一个可能是结束token或者第51个token，但循环最多51次，实际上我们只想要50个，所以这里返回时去掉最后一个）。</li>
                <li>如果生成过程中遇到结束token，则提前终止并返回序列。</li>
                <li>注意：结束token的索引是vocab_size-2。</li>
            </ul>
        </li>
    </ul>
</details>

```python
    @torch.no_grad() # 禁用梯度计算，用于推理阶段
    def sample(self, clip_feature, y_mask, if_categorial=False):  # 自回归采样生成序列
        for k in range(51):  # 最多生成50个token（k从0到50，共51次迭代）
            if k == 0:  # 第一次迭代
                x = []  # 序列为空
            else:  # 后续迭代
                x = xs  # 使用之前生成的所有token
            logits = self.forward_sample(x, clip_feature, y_mask)  # 前向传播获取模型输出
            logits = logits[:, -1, :]  # 只取最后一个时间步的logits
            probs = F.softmax(logits, dim=-1)  # 转换为概率分布
            if if_categorial:  # 分类采样（随机采样）
                dist = Categorical(probs)  # 创建分类分布
                idx = dist.sample()  # 从分布中采样一个token
                if idx == self.config.vocab_size -2:  # 检查是否遇到结束token
                    break  # 遇到结束token则停止生成
                idx = idx.unsqueeze(-1)  # 增加维度以匹配序列格式
            else:  # 贪心采样
                _, idx = torch.topk(probs, k=1, dim=-1)  # 选择概率最高的token

                if idx[0] == self.config.vocab_size - 2:  # 检查是否结束token
                    break  # 遇到结束token则停止生成

            # 将新生成的token添加到序列中
            if k == 0:  # 第一次迭代
                xs = idx  # 直接赋值
            else:  # 后续迭代
                xs = torch.cat((xs, idx), dim=1)  # 拼接新token到序列末尾
            
            if k == 50:  # 如果达到最大长度限制
                return xs[:, :-1]  # 返回除最后一个token外的所有token
        
        if k == 0:  # 如果第一次迭代就遇到结束token
            return torch.ones(1,1).to(clip_feature.device).long()  # 返回默认token
        else:  # 正常情况
            return xs  # 返回生成的完整序列
```

<details>
    <summary>
        2. forward_sample函数
    </summary>
    <ul>
        <li>
            <strong>函数功能：</strong>基于CLIP视觉特征和已生成的文本token，预测下一个token的概率分布
        </li>
        <li>
            <strong>输入参数：</strong>
            <ul>
                <li>idx: 已生成的token序列，形状为(batch_size, sequence_length)</li>
                <li>clip_feature: CLIP模型提取的视觉特征，形状为(batch_size, feature_length, feature_dim)</li>
                <li>y_mask: 掩码张量，用于确定CLIP特征的有效长度</li>
            </ul>
        </li>
        <li>
            <strong>具体步骤：</strong>
            <ul>
                <li>首先获取文本特征的长度（即CLIP特征的长度）</li>
                <li>判断当前是否第一次生成（即idx是否为空）：
                    <ul>
                        <li>如果是，则只使用投影后的CLIP特征，并且通过y_mask[0]的求和（即有效长度）来截取CLIP特征</li>
                        <li>如果不是，则检查当前序列长度是否超过模型的最大块大小（block_size），然后将idx通过词嵌入层转换为嵌入向量，并与投影后的CLIP特征（同样截取有效长度）进行拼接</li>
                    </ul>
                </li>
                <li>将得到的特征输入到transformer的各个块中，每一块都会使用y_mask（可能是用于调整注意力掩码）进行处理</li>
                <li>通过最后的层归一化</li>
                <li>通过语言模型头（lm_head）得到logits</li>
            </ul>
        </li>
    </ul>
</details>

```python
    def forward_sample(self, idx: torch.Tensor, clip_feature: torch.Tensor, y_mask) -> torch.Tensor:  # 前向传播采样函数
        text_length = clip_feature.shape[1]  # 获取文本(Clip)特征的长度
        if len(idx) == 0:  # 如果输入序列为空（第一次生成）
            x = self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :]  # 只使用投影后的clip特征
        else:  # 如果已有生成的token序列
            _, t = idx.size()  # 获取序列长度
            assert (  # 检查序列长度是否超过模型限制
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            # forward the LLaMA model itself  # 前向传播LLaMA模型
            x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)  # 将token转换为嵌入向量
            x = torch.cat((self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :],x), dim=1)  # 将clip特征和token嵌入拼接

        for block in self.transformer.h:  # 遍历所有transformer块
            x = block(x, y_mask)  # 通过transformer块处理
        x = self.transformer.ln_f(x)  # 应用最终的层归一化

        logits = self.lm_head(x)  # (b, t, vocab_size)  # 通过语言模型头获取logits

        return logits  # 返回logits用于后续采样
```



#### 4. **Motion Decoder（运动重建）**

- **对应文档**：文档5 (`models/vqvae.py`)
- **关键代码段**：

```python
def forward(self, x):
        
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in) # 256, 512, 16
         
        ## quantization
        if self.quant in ["LFQ", "BSQ"]:
            x_quantized, _, loss, perplexity, activate, indices = self.quantizer(x_encoder)
        elif self.quant == "FSQ":
            x_quantized, _, loss, perplexity, activate, indices = self.quantizer(x_encoder)
        else:
            x_quantized, loss, perplexity, activate, indices  = self.quantizer(x_encoder) # (256, 512, 16)
        
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity, activate, indices


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
```

- **说明**：解码器将量化后的特征（`x_quantized`）重建为运动序列。该步骤在论文中强调通过**逆小波变换**（inverse wavelet transform）恢复平滑运动，具体通过`patch_method="haar"`参数实现。



#### 5. **Inverse Wavelet Transform（后处理）**

- **对应文档**：`models/vqvae.py`
- **关键代码段**：9-72行初始化vqvae
- **说明**：逆小波变换作为解码器的后处理步骤，用于将频域特征转换回时域运动数据，减少抖动现象（如论文第4.2节所述）。



### 调用梳理：

inference_batch.py直接调用的**LLMA模型model_hf.py**加载transformer的sample，

inference_batch.py 两处使用 vqvae.py：初始化与解码

- 初始化/加载 VQ-VAE（获取码本规模、加载权重）

![image-20250922204539656](Figure5%E6%B5%81%E7%A8%8B%E6%A2%B3%E7%90%86.assets/image-20250922204539656.png)

- 将 model_hf.sample 生成的索引序列“反量化+解码”为连续动作

![image-20250922204631915](Figure5%E6%B5%81%E7%A8%8B%E6%A2%B3%E7%90%86.assets/image-20250922204631915.png)

这里调用的就是vqvae的forward_decoder函数：

![image-20250922204702178](Figure5%E6%B5%81%E7%A8%8B%E6%A2%B3%E7%90%86.assets/image-20250922204702178.png)

而model_hf.py 从不调用 vqvae.py，它只负责根据文本特征自回归生成“离散动作索引序列”（token ids），即 trans_encoder.sample(...) 的返回值。