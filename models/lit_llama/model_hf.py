"""Full definition of a LLaMA Language Model, all of it in this single file.

基于nanoGPT实现的完整LLaMA语言模型定义: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors  # 忽略mypy类型检查错误
import math  # 数学函数库
from dataclasses import dataclass  # 数据类装饰器

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
from torch.nn import functional as F  # 神经网络函数
from typing_extensions import Self  # 类型注解扩展
from typing import Optional  # 可选类型注解
from transformers.modeling_utils import PreTrainedModel  # HuggingFace预训练模型基类
from torch.distributions import Categorical  # 分类分布
import torch.nn.functional as F  # 神经网络函数（重复导入）
import numpy as np  # 数值计算库

@dataclass  # 数据类装饰器，自动生成__init__等方法
class LLaMAHFConfig:  # LLaMA模型配置类
    block_size: int = 4096  # 最大序列长度（上下文窗口大小）
    vocab_size: int = 32000  # 词汇表大小
    n_layer: int = 32  # Transformer层数
    n_head: int = 32  # 注意力头数
    n_embd: int = 4096  # 嵌入维度
    @classmethod  # 类方法装饰器
    def from_name(cls, name: str) -> Self:  # 根据名称创建配置实例
        return cls(**llama_configs[name])  # 使用预定义配置创建实例


llama_configs = {  # 预定义的LLaMA模型配置字典
    "44M": dict(n_layer=8, n_head=8, n_embd=512),  # 44M参数模型配置
    "111M": dict(n_layer=12, n_head=12, n_embd=768),  # 111M参数模型配置
    "343M": dict(n_layer=24, n_head=16, n_embd=1024),  # 343M参数模型配置
    "775M": dict(n_layer=36, n_head=20, n_embd=1280),  # 775M参数模型配置
    "1B": dict(n_layer=48, n_head=24, n_embd=1536),  # 1B参数模型配置
    "3B": dict(n_layer=24, n_head=32, n_embd=3200),  # 3B参数模型配置
    "5B": dict(n_layer=24, n_head=32, n_embd=4096),  # 5B参数模型配置
    "6B": dict(n_layer=28, n_head=32, n_embd=4096),  # 6B参数模型配置
    "7B": dict(n_layer=36, n_head=32, n_embd=4096),  # 7B参数模型配置
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),  # 13B参数模型配置
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),  # 30B参数模型配置
    "65B": dict(n_layer=80, n_head=64, n_embd=8192)  # 65B参数模型配置
}


class LLaMAHF(nn.Module):  # LLaMA语言模型主类，继承自nn.Module
    """
    功能：基于LLaMA架构的条件语言模型，用于文本到动作的生成
    输入参数：
        config: LLaMAHFConfig - 模型配置参数
    主要组件：
        - lm_head: 语言模型头，输出词汇表概率分布
        - transformer: 包含词嵌入、多层Block和层归一化
        - llama_proj: CLIP特征投影层，将视觉特征映射到模型维度
    """
    def __init__(self, config: LLaMAHFConfig) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        assert config.vocab_size is not None  # 确保词汇表大小已定义
        assert config.block_size is not None  # 确保块大小已定义
        self.config = config  # 保存配置参数

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size-1, bias=False)  # 语言模型头，输出维度为vocab_size-1
        self.transformer = nn.ModuleDict(  # Transformer模块字典
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入层，将token ID映射为向量
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 多层Transformer Block
                ln_f=RMSNorm(config.n_embd),  # 最终的层归一化
            )
        )

        self.llama_proj = nn.Linear(config.clip_dim, config.n_embd)  # CLIP特征投影层
        if config.tie_weights:  # 如果启用权重绑定
            self._tie_or_clone_weights(self.lm_head, self.transformer.wte)  # 绑定语言模型头和词嵌入权重

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):  # 权重绑定或克隆函数
        """
        功能：根据是否使用TorchScript来绑定或克隆模块权重
        输入参数：
            output_embeddings: 输出嵌入层（如lm_head）
            input_embeddings: 输入嵌入层（如wte）
        实现过程：将输入嵌入的权重复制给输出嵌入，并处理偏置项
        """
        output_embeddings.weight = input_embeddings.weight  # 直接复制权重

        if getattr(output_embeddings, "bias", None) is not None:  # 如果输出嵌入有偏置项
            output_embeddings.bias.data = nn.functional.pad(  # 对偏置项进行填充
                output_embeddings.bias.data,  # 原始偏置数据
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],  # 计算需要填充的长度
                ),
                "constant",  # 使用常数填充
                0,  # 填充值为0
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):  # 如果两个模块都有特征维度属性
            output_embeddings.out_features = input_embeddings.num_embeddings  # 同步输出特征数

    def get_input_embeddings(self):  # 获取输入嵌入层
        return self.transformer.wte  # 返回词嵌入层
    
    def set_input_embeddings(self, value):  # 设置输入嵌入层
        self.transformer.wte = value  # 更新词嵌入层

    def get_output_embeddings(self):  # 获取输出嵌入层
        return self.lm_head  # 返回语言模型头
    
    def set_output_embeddings(self, new_embeddings):  # 设置输出嵌入层
        self.lm_head = new_embeddings  # 更新语言模型头

    def _init_weights(self, module: nn.Module) -> None:  # 权重初始化函数
        """
        功能：对模型参数进行初始化
        输入参数：
            module: 需要初始化的模块
        实现过程：使用正态分布初始化线性层和嵌入层的权重
        """
        if isinstance(module, nn.Linear):  # 如果是线性层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))  # 使用缩放的正态分布初始化
        elif isinstance(module, nn.Embedding):  # 如果是嵌入层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))  # 使用缩放的正态分布初始化

    """
    函数功能：使用自回归方式逐步生成文本序列
    输入参数：
        clip_feature: CLIP提取的视觉特征，作为生成的条件
        y_mask: 掩码张量，用于确定CLIP特征的有效长度
        if_categorial: 是否使用随机采样（否则使用贪心采样）
    函数过程：
        禁用梯度计算，因为这是推理阶段。
        循环最多51次（即最多生成50个token，因为第一次循环时k=0，然后最多再生成50个，所以最多50个token）。
        在第一次循环时，序列x为空，然后调用forward_sample得到logits（此时只使用CLIP特征，没有文本token）。
        取最后一个时间步的logits，然后通过softmax得到概率分布。
        根据if_categorial选择采样方式：
            如果为True，则使用分类分布进行随机采样，并检查采样到的token是否为结束token（vocab_size-2），如果是则停止生成。
            如果为False，则使用贪心采样（取概率最大的token），同样检查是否为结束token。
        将新生成的token添加到序列中。
        如果达到最大长度（50个token），则返回序列（去掉最后一个token，因为最后一个可能是结束token或者第51个token，但循环最多51次，实际上我们只想要50个，所以这里返回时去掉最后一个）。
        如果生成过程中遇到结束token，则提前终止并返回序列。
        注意：结束token的索引是vocab_size-2。
    """        
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

                if idx[0] == self.config.vocab_size - 2:  # 检查是否遇到结束token
                    break  # 遇到结束token则停止生成

            # append to the sequence and continue  # 将新生成的token添加到序列中
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
    
    
    """
    函数功能:基于CLIP视觉特征和已生成的文本token,预测下一个token的概率分布
    输入参数：
        idx: 已生成的token序列,形状为(batch_size, sequence_length)
        clip_feature: CLIP模型提取的视觉特征,形状为(batch_size, feature_length, feature_dim)
        y_mask: 掩码张量,用于确定CLIP特征的有效长度
    具体步骤：
        首先获取文本特征的长度（即CLIP特征的长度）
        判断当前是否第一次生成（即idx是否为空）：
            如果是，则只使用投影后的CLIP特征，并且通过y_mask[0]的求和（即有效长度）来截取CLIP特征。
            如果不是，则检查当前序列长度是否超过模型的最大块大小（block_size），然后将idx通过词嵌入层转换为嵌入向量，并与投影后的CLIP特征（同样截取有效长度）进行拼接。
        将得到的特征输入到transformer的各个块中，每一块都会使用y_mask（可能是用于调整注意力掩码）进行处理。
        通过最后的层归一化。
        通过语言模型头（lm_head）得到logits。
    """
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

    def forward(self, idx: torch.Tensor, clip_feature: torch.Tensor, y_mask) -> torch.Tensor:  # 前向传播函数（训练时使用）
        """
        功能：执行模型的前向传播，用于训练阶段
        输入参数：
            idx: torch.Tensor - 输入token序列，形状为(batch_size, seq_len)
            clip_feature: torch.Tensor - CLIP特征，形状为(batch_size, text_length, clip_dim)
            y_mask: torch.Tensor - 掩码张量，指示文本部分位置
        输出：
            torch.Tensor - 输出logits，形状为(batch_size, seq_len, vocab_size)
        实现过程：处理输入序列 -> 替换文本部分为CLIP特征 -> Transformer处理 -> 输出投影
        """
        text_length = clip_feature.shape[1]  # 获取文本特征长度
        if len(idx) == 0:  # 如果输入序列为空
            x = self.llama_proj(clip_feature)[:, :int(y_mask[0].sum()), :]  # 只使用投影后的CLIP特征
        else:  # 如果有输入序列
            _, t = idx.size()  # 获取序列长度
            assert (  # 检查序列长度是否超过模型限制
                t <= self.config.block_size
            ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            # forward the LLaMA model itself  # 前向传播LLaMA模型
            x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)  # 将token转换为嵌入向量

            # replace text_length tokens with clip_feature  # 用CLIP特征替换文本长度的token
            expanded_mask = y_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # 扩展掩码到嵌入维度
            result = torch.where(expanded_mask == 1, self.llama_proj(clip_feature), x[:, :text_length, :])  # 根据掩码替换文本部分
            result = torch.cat((result, x[:, text_length:, :]), dim=1)  # 拼接替换后的文本部分和剩余序列
            x = result  # 更新输入

        for block in self.transformer.h:  # 遍历所有Transformer块
            x = block(x, y_mask)  # 通过Transformer块处理
        x = self.transformer.ln_f(x)  # 应用最终的层归一化

        logits = self.lm_head(x)  # (b, t, vocab_size)  # 通过语言模型头获取logits

        return logits  # 返回logits
    
    def resize_token_embeddings(  # 调整token嵌入大小函数
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, using_old_initilization: bool = False
    ) -> nn.Embedding:
        """
        功能：调整模型输入token嵌入矩阵的大小
        输入参数：
            new_num_tokens: Optional[int] - 新的token数量，增加会在末尾添加新初始化的向量，减少会移除末尾的向量
            pad_to_multiple_of: Optional[int] - 将嵌入矩阵填充到指定值的倍数，有利于Tensor Cores性能
            using_old_initilization: bool - 是否使用旧初始化方法
        输出：
            nn.Embedding - 调整后的输入token嵌入模块
        实现过程：调用内部调整函数 -> 更新配置 -> 重新绑定权重
        
        如果new_num_tokens != config.vocab_size，则调整输入token嵌入矩阵的大小。
        如果模型类有tie_weights()方法，会在之后处理权重绑定。
        
        参数说明：
            new_num_tokens (`int`, *optional*):
                嵌入矩阵中的新token数量。增加大小会在末尾添加新初始化的向量。
                减少大小会移除末尾的向量。如果未提供或为None，则只返回输入token模块的指针而不做任何操作。
            pad_to_multiple_of (`int`, *optional*):
                如果设置，会将嵌入矩阵填充到提供值的倍数。如果new_num_tokens设置为None，
                则只将嵌入填充到pad_to_multiple_of的倍数。
                
                这对于在计算能力>=7.5(Volta)的NVIDIA硬件上启用Tensor Cores特别有用，
                或者对于TPU，序列长度为128的倍数会有好处。更多详情请参考：
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        返回：
            `torch.nn.Embedding`: 指向模型输入token嵌入模块的指针。
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)  # 调用内部调整函数
        if new_num_tokens is None and pad_to_multiple_of is None:  # 如果两个参数都为None
            return model_embeds  # 直接返回嵌入模块

        # Update base model and current model config  # 更新基础模型和当前模型配置
        self.config.vocab_size = model_embeds.weight.shape[0]  # 更新配置中的词汇表大小
        self.vocab_size = model_embeds.weight.shape[0]  # 更新实例变量中的词汇表大小

        # Tie weights again if needed  # 如果需要，重新绑定权重
        # self.tie_weights()  # 权重绑定（已注释）

        return model_embeds  # 返回调整后的嵌入模块
    
    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):  # 内部调整token嵌入函数
        """
        功能：内部调整token嵌入矩阵的大小
        输入参数：
            new_num_tokens: 新的token数量
            pad_to_multiple_of: 填充到指定倍数
        输出：
            nn.Embedding - 调整后的输入嵌入模块
        实现过程：获取旧嵌入 -> 创建新嵌入 -> 设置梯度 -> 更新模型 -> 调整语言模型头
        """
        old_embeddings = self.get_input_embeddings()  # 获取当前输入嵌入
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)  # 创建调整后的嵌入
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad  # 获取旧嵌入的梯度状态
        new_embeddings.requires_grad_(old_embeddings_requires_grad)  # 设置新嵌入的梯度状态
        self.set_input_embeddings(new_embeddings)  # 更新模型的输入嵌入

        # Update new_num_tokens with the actual size of new_embeddings  # 用新嵌入的实际大小更新new_num_tokens
        if pad_to_multiple_of is not None:  # 如果指定了填充倍数
            # if is_deepspeed_zero3_enabled():  # 如果启用了DeepSpeed ZeRO-3（已注释）
            #     import deepspeed  # 导入DeepSpeed（已注释）

            #     with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):  # 收集参数（已注释）
            #         new_num_tokens = new_embeddings.weight.shape[0]  # 更新token数量（已注释）
            # else:  # 否则（已注释）
            new_num_tokens = new_embeddings.weight.shape[0]  # 更新token数量

        # if word embeddings are not tied, make sure that lm head is resized as well  # 如果词嵌入没有绑定，确保语言模型头也被调整
        # if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:  # 原始条件（已注释）
        if self.get_output_embeddings() is not None and not False:  # 如果输出嵌入存在且权重未绑定
            old_lm_head = self.get_output_embeddings()  # 获取当前语言模型头
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)  # 创建调整后的语言模型头
            # if hasattr(old_lm_head, "_hf_hook"):  # 如果有HuggingFace钩子（已注释）
            #     hook = old_lm_head._hf_hook  # 获取钩子（已注释）
            #     add_hook_to_module(new_lm_head, hook)  # 添加钩子到新模块（已注释）
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad  # 获取旧语言模型头的梯度状态
            new_lm_head.requires_grad_(old_lm_head_requires_grad)  # 设置新语言模型头的梯度状态
            self.set_output_embeddings(new_lm_head)  # 更新模型的输出嵌入

        return self.get_input_embeddings()  # 返回调整后的输入嵌入
    
    def _get_resized_embeddings(  # 获取调整大小的嵌入函数
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        功能：从提供的token嵌入模块构建调整大小的嵌入模块
        输入参数：
            old_embeddings: nn.Embedding - 要调整大小的旧嵌入
            new_num_tokens: Optional[int] - 嵌入矩阵中的新token数量
            pad_to_multiple_of: Optional[int] - 填充到指定倍数
        输出：
            nn.Embedding - 调整后的嵌入模块
        实现过程：验证参数 -> 计算新大小 -> 创建新嵌入 -> 复制权重 -> 初始化新权重
        
        增加大小会在末尾添加新初始化的向量，减少大小会移除末尾的向量
        
        参数说明：
            old_embeddings (`torch.nn.Embedding`):
                要调整大小的旧嵌入。
            new_num_tokens (`int`, *optional*):
                嵌入矩阵中的新token数量。
                
                增加大小会在末尾添加新初始化的向量。减少大小会移除末尾的向量。
                如果未提供或为None，则只返回输入token模块的指针而不做任何操作。
            pad_to_multiple_of (`int`, *optional*):
                如果设置，会将嵌入矩阵填充到提供值的倍数。如果new_num_tokens设置为None，
                则只将嵌入填充到pad_to_multiple_of的倍数。
                
                这对于在计算能力>=7.5(Volta)的NVIDIA硬件上启用Tensor Cores特别有用，
                或者对于TPU，序列长度为128的倍数会有好处。更多详情请参考：
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        返回：
            `torch.nn.Embedding`: 指向调整后的嵌入模块的指针，如果new_num_tokens为None则返回旧嵌入模块
        """

        if pad_to_multiple_of is not None:  # 如果指定了填充倍数
            if not isinstance(pad_to_multiple_of, int):  # 检查是否为整数
                raise ValueError(  # 抛出值错误
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"  # 错误信息
                )
            if new_num_tokens is None:  # 如果新token数量为None
                new_num_tokens = old_embeddings.weight.shape[0]  # 使用旧嵌入的大小
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of  # 向上对齐到倍数
        else:  # 如果没有指定填充倍数
            print(  # 打印警告信息
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"  # 警告信息
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."  # 性能警告
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"  # 参考指南
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"  # 链接
            )

        if new_num_tokens is None:  # 如果新token数量为None
            return old_embeddings  # 直接返回旧嵌入

        # if is_deepspeed_zero3_enabled():  # 如果启用了DeepSpeed ZeRO-3（已注释）
        if False:  # 禁用DeepSpeed ZeRO-3
            import deepspeed  # 导入DeepSpeed（已注释）

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):  # 收集参数（已注释）
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()  # 获取旧嵌入大小（已注释）
        else:  # 否则
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()  # 获取旧嵌入大小

        # if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():  # 如果大小相同且未启用DeepSpeed（已注释）
        if old_num_tokens == new_num_tokens and not False:  # 如果大小相同
            return old_embeddings  # 直接返回旧嵌入

        if not isinstance(old_embeddings, nn.Embedding):  # 检查是否为嵌入类型
            raise TypeError(  # 抛出类型错误
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"  # 错误信息
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"  # 建议
                f" {nn.Embedding}."  # 类型要求
            )

        # Build new embeddings  # 构建新嵌入

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init  # 使用DeepSpeed ZeRO-3时不应使用DeepSpeed初始化创建新嵌入
        # because the shape of the new embedding layer is used across various modeling files  # 因为新嵌入层的形状在多个建模文件中使用
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading  # 以及更新配置词汇表大小。使用DeepSpeed初始化时形状为0会导致
        # to errors when training.  # 训练时出错
        new_embeddings = nn.Embedding(  # 创建新嵌入层
            new_num_tokens,  # 新token数量
            old_embedding_dim,  # 旧嵌入维度
            device=old_embeddings.weight.device,  # 设备
            dtype=old_embeddings.weight.dtype,  # 数据类型
        )

        # initialize all new embeddings (in particular added tokens)  # 初始化所有新嵌入（特别是添加的token）
        self._init_weights(new_embeddings)  # 初始化权重

        # Copy token embeddings from the previous weights  # 从之前的权重复制token嵌入

        # numbers of tokens to copy  # 要复制的token数量
        n = min(old_num_tokens, new_num_tokens)  # 取较小值

        # if is_deepspeed_zero3_enabled():  # 如果启用了DeepSpeed ZeRO-3（已注释）
        if False:  # 禁用DeepSpeed ZeRO-3
            import deepspeed  # 导入DeepSpeed（已注释）

            params = [old_embeddings.weight, new_embeddings.weight]  # 参数列表（已注释）
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):  # 收集参数（已注释）
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]  # 复制权重（已注释）
        else:  # 否则
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]  # 复制权重

        return new_embeddings  # 返回新嵌入


    def _get_resized_lm_head(  # 获取调整大小的语言模型头函数
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Linear:
        """
        功能：从提供的旧线性模块构建调整大小的线性模块
        输入参数：
            old_lm_head: nn.Linear - 要调整大小的旧语言模型头线性层
            new_num_tokens: Optional[int] - 线性矩阵中的新token数量
            transposed: Optional[bool] - 是否转置，默认为False
        输出：
            nn.Linear - 调整后的线性模块
        实现过程：检查参数 -> 获取旧模块大小 -> 创建新模块 -> 复制权重 -> 初始化新权重
        
        增加大小会在末尾添加新初始化的向量，减少大小会移除末尾的向量
        
        参数说明：
            old_lm_head (`torch.nn.Linear`):
                要调整大小的旧语言模型头线性层。
            new_num_tokens (`int`, *optional*):
                线性矩阵中的新token数量。
                
                增加大小会在末尾添加新初始化的向量。减少大小会移除末尾的向量。
                如果未提供或为None，则只返回输入token模块的指针而不做任何操作。
            transposed (`bool`, *optional*, 默认为`False`):
                是否转置old_lm_head。如果为True，old_lm_head.size()为`lm_head_dim, vocab_size`，
                否则为`vocab_size, lm_head_dim`。

        返回：
            `torch.nn.Linear`: 指向调整后的线性模块的指针，如果new_num_tokens为None则返回旧模块
        """
        if new_num_tokens is None:  # 如果新token数量为None
            return old_lm_head  # 直接返回旧语言模型头

        # if is_deepspeed_zero3_enabled():  # 如果启用了DeepSpeed ZeRO-3（已注释）
        if False:  # 禁用DeepSpeed ZeRO-3
            import deepspeed  # 导入DeepSpeed（已注释）

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):  # 收集参数（已注释）
                old_num_tokens, old_lm_head_dim = (  # 获取旧模块大小（已注释）
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()  # 根据是否转置获取大小（已注释）
                )
        else:  # 否则
            old_num_tokens, old_lm_head_dim = (  # 获取旧模块大小
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()  # 根据是否转置获取大小
            )

        # if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():  # 如果大小相同且未启用DeepSpeed（已注释）
        if old_num_tokens == new_num_tokens and not False:  # 如果大小相同
            return old_lm_head  # 直接返回旧语言模型头

        if not isinstance(old_lm_head, nn.Linear):  # 检查是否为线性层类型
            raise TypeError(  # 抛出类型错误
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"  # 错误信息
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"  # 建议
                f" {nn.Linear}."  # 类型要求
            )

        # Build new lm head  # 构建新语言模型头
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)  # 根据是否转置确定形状
        has_new_lm_head_bias = old_lm_head.bias is not None  # 检查是否有偏置项

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init  # 使用DeepSpeed ZeRO-3时不应使用DeepSpeed初始化创建新嵌入
        # because the shape of the new embedding layer is used across various modeling files  # 因为新嵌入层的形状在多个建模文件中使用
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading  # 以及更新配置词汇表大小。使用DeepSpeed初始化时形状为0会导致
        # to errors when training.  # 训练时出错
        new_lm_head = nn.Linear(  # 创建新语言模型头
            *new_lm_head_shape,  # 展开形状参数
            bias=has_new_lm_head_bias,  # 是否使用偏置
            device=old_lm_head.weight.device,  # 设备
            dtype=old_lm_head.weight.dtype,  # 数据类型
        )

        # initialize new lm head (in particular added tokens)  # 初始化新语言模型头（特别是添加的token）
        self._init_weights(new_lm_head)  # 初始化权重

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)  # 要复制的token数量

        # if is_deepspeed_zero3_enabled():  # 如果启用了DeepSpeed ZeRO-3（已注释）
        if False:  # 禁用DeepSpeed ZeRO-3
            import deepspeed  # 导入DeepSpeed（已注释）

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]  # 参数列表（已注释）
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):  # 收集参数（已注释）
                self._copy_lm_head_original_to_resized(  # 复制语言模型头（已注释）
                    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias  # 参数（已注释）
                )
        else:  # 否则
            self._copy_lm_head_original_to_resized(  # 复制语言模型头
                new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias  # 参数
            )

        return new_lm_head  # 返回新语言模型头

    def _copy_lm_head_original_to_resized(  # 复制语言模型头从原始到调整大小
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        """
        功能：将旧语言模型头的权重复制到新语言模型头
        输入参数：
            new_lm_head: 新语言模型头
            old_lm_head: 旧语言模型头
            num_tokens_to_copy: 要复制的token数量
            transposed: 是否转置
            has_new_lm_head_bias: 是否有偏置项
        实现过程：根据是否转置复制权重 -> 如果有偏置则复制偏置
        """
        # Copy old lm head weights to new lm head  # 将旧语言模型头权重复制到新语言模型头
        if not transposed:  # 如果没有转置
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]  # 直接复制权重
        else:  # 如果转置了
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]  # 转置复制权重

        # Copy bias weights to new lm head  # 将偏置权重复制到新语言模型头
        if has_new_lm_head_bias:  # 如果有偏置项
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]  # 复制偏置

    @classmethod  # 类方法装饰器
    def from_name(cls, name: str) -> Self:  # 根据名称创建实例
        return cls(LLaMAHFConfig.from_name(name))  # 使用配置名称创建实例


class Block(nn.Module):  # Transformer Block类，继承自nn.Module
    """
    功能：单个Transformer Block，包含自注意力机制和前馈网络
    输入参数：
        config: LLaMAHFConfig - 模型配置参数
    主要组件：
        - rms_1: 第一个RMS归一化层（注意力前）
        - attn: 长度感知的因果自注意力机制
        - rms_2: 第二个RMS归一化层（MLP前）
        - mlp: 多层感知机前馈网络
    实现过程：使用Pre-LN架构，先归一化再计算注意力/MLP，最后残差连接
    """
    def __init__(self, config: LLaMAHFConfig) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.rms_1 = RMSNorm(config.n_embd)  # 第一个RMS归一化层
        # self.attn = CausalSelfAttention(config)  # 标准因果自注意力（已注释）
        self.attn = LengthCausalSelfAttention(config)  # 长度感知的因果自注意力
        self.rms_2 = RMSNorm(config.n_embd)  # 第二个RMS归一化层
        
        # self.use_moe = use_moe  # 是否使用混合专家模型（已注释）
        # if use_moe:  # 如果使用MoE
        #     # n_embed, num_experts, top_k, dropout  # MoE参数注释
        #     # self.smoe = SparseMoE(config.n_embd, config.n_experts, config.top_k, 0.1)  # 稀疏MoE（已注释）
        #     self.smoe = Qwen2MoeSparseMoeBlock(config)  # Qwen2 MoE块（已注释）
        # else:  # 如果不使用MoE
        #     self.mlp = MLP(config)  # 标准MLP（已注释）
        self.mlp = MLP(config)  # 多层感知机前馈网络

    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        """
        功能：执行Transformer Block的前向传播
        输入参数：
            x: torch.Tensor - 输入张量，形状为(batch_size, seq_len, n_embd)
            y_mask: torch.Tensor - 掩码张量，用于注意力机制
        输出：
            torch.Tensor - 处理后的张量，形状与输入相同
        实现过程：Pre-LN架构，先归一化再计算，最后残差连接
        """
        x = x + self.attn(self.rms_1(x), y_mask)  # 自注意力：先归一化，再注意力，最后残差连接
        # if self.use_moe:  # 如果使用MoE（已注释）
        #     x = x + self.smoe(self.rms_2(x))  # MoE前馈网络（已注释）
        # else:  # 否则使用标准MLP（已注释）
        x = x + self.mlp(self.rms_2(x))  # MLP前馈网络：先归一化，再MLP，最后残差连接
        return x  # 返回处理后的张量


class CausalSelfAttention(nn.Module):  # 因果自注意力机制类
    """
    功能：实现标准的因果自注意力机制，支持RoPE位置编码
    输入参数：
        config: LLaMAHFConfig - 模型配置参数
    主要组件：
        - c_attn: QKV投影层，将输入映射为查询、键、值
        - c_proj: 输出投影层
        - rope_cache: RoPE位置编码缓存
    实现过程：多头注意力 + RoPE位置编码 + Flash Attention
    """
    def __init__(self, config: LLaMAHFConfig) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        assert config.n_embd % config.n_head == 0  # 确保嵌入维度能被头数整除

        # key, query, value projections for all heads, but in a batch  # 为所有头批量计算QKV投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)  # QKV投影层，输出3倍嵌入维度
        # output projection  # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)  # 输出投影层

        self.n_head = config.n_head  # 注意力头数
        self.n_embd = config.n_embd  # 嵌入维度
        self.block_size = config.block_size  # 最大序列长度
        self.rope_cache = None  # RoPE缓存，延迟初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        """
        功能：执行因果自注意力计算
        输入参数：
            x: torch.Tensor - 输入张量，形状为(batch_size, seq_len, n_embd)
        输出：
            torch.Tensor - 注意力输出，形状与输入相同
        实现过程：QKV投影 -> 多头重塑 -> RoPE编码 -> Flash Attention -> 输出投影
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)  # 获取批次大小、序列长度、嵌入维度

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim  # 为所有头批量计算QKV并调整维度
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # 将投影结果分割为Q、K、V

        head_size = C // self.n_head  # 计算每个头的维度
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)  # 重塑K为多头格式
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)  # 重塑Q为多头格式
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)  # 重塑V为多头格式

        if self.rope_cache is None:  # 如果RoPE缓存未初始化
            # cache for future forward calls  # 为后续前向调用创建缓存
            self.rope_cache = build_rope_cache(  # 构建RoPE缓存
                seq_len=self.block_size,  # 序列长度
                n_elem=self.n_embd // self.n_head,  # 每个头的元素数
                dtype=x.dtype,  # 数据类型
                device=x.device,  # 设备
            )

        q = apply_rope(q, self.rope_cache)  # 对Q应用RoPE位置编码
        k = apply_rope(k, self.rope_cache)  # 对K应用RoPE位置编码

        # efficient attention using Flash Attention CUDA kernels  # 使用Flash Attention CUDA内核的高效注意力
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)  # 执行缩放点积注意力

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side  # 重新组装所有头的输出

        # output projection  # 输出投影
        y = self.c_proj(y)  # 通过输出投影层

        return y  # 返回注意力输出

class LengthCausalSelfAttention(nn.Module):  # 长度感知的因果自注意力机制类
    """
    功能：实现支持文本-动作混合注意力的长度感知因果自注意力机制
    输入参数：
        config: LLaMAHFConfig - 模型配置参数
    主要组件：
        - c_attn: QKV投影层
        - c_proj: 输出投影层
        - rope_cache: RoPE位置编码缓存
    实现过程：多头注意力 + RoPE位置编码 + 混合注意力掩码 + Flash Attention
    特点：支持文本和动作序列的混合注意力，文本部分可以相互关注
    """
    def __init__(self, config: LLaMAHFConfig) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        assert config.n_embd % config.n_head == 0  # 确保嵌入维度能被头数整除

        # key, query, value projections for all heads, but in a batch  # 为所有头批量计算QKV投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)  # QKV投影层
        # output projection  # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)  # 输出投影层

        self.n_head = config.n_head  # 注意力头数
        self.n_embd = config.n_embd  # 嵌入维度
        self.block_size = config.block_size  # 最大序列长度
        self.rope_cache = None  # RoPE缓存，延迟初始化
        
    def forward(self, x: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        """
        功能：执行长度感知的因果自注意力计算
        输入参数：
            x: torch.Tensor - 输入张量，形状为(batch_size, seq_len, n_embd)
            y_mask: torch.Tensor - 掩码张量，指示文本部分的位置
        输出：
            torch.Tensor - 注意力输出，形状与输入相同
        实现过程：QKV投影 -> 多头重塑 -> RoPE编码 -> 混合注意力掩码 -> Flash Attention -> 输出投影
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)  # 获取批次大小、序列长度、嵌入维度

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim  # 为所有头批量计算QKV并调整维度
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # 将投影结果分割为Q、K、V

        head_size = C // self.n_head  # 计算每个头的维度
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)  # 重塑K为多头格式
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)  # 重塑Q为多头格式
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)  # 重塑V为多头格式
        
        if self.rope_cache is None:  # 如果RoPE缓存未初始化
            # cache for future forward calls  # 为后续前向调用创建缓存
            self.rope_cache = build_rope_cache(  # 构建RoPE缓存
                seq_len=self.block_size,  # 序列长度
                n_elem=self.n_embd // self.n_head,  # 每个头的元素数
                dtype=x.dtype,  # 数据类型
                device=x.device,  # 设备
            )
            
        q = apply_rope(q, self.rope_cache)  # 对Q应用RoPE位置编码
        k = apply_rope(k, self.rope_cache)  # 对K应用RoPE位置编码

        # 构建注意力掩码，支持文本-动作混合注意力  # 构建混合注意力掩码
        attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device)  # 创建全1的注意力掩码
        attn_mask = torch.tril(attn_mask)  # 转换为下三角矩阵（因果掩码）
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)  # 扩展到批次维度

        text_mask = y_mask.unsqueeze(2)*y_mask.unsqueeze(1)  # 创建文本-文本注意力掩码
        text_mask = F.pad(text_mask, (0, T-y_mask.shape[1], 0, T-y_mask.shape[1]), mode='constant', value=0)  # 填充到完整序列长度
        attn_mask = torch.logical_or(attn_mask, text_mask)  # 合并因果掩码和文本掩码
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(1), dropout_p=0.0, is_causal=False)  # 执行缩放点积注意力

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 重新组装所有头的输出

        y = self.c_proj(y)  # 通过输出投影层

        return y  # 返回注意力输出


class MLP(nn.Module):  # 多层感知机类
    """
    功能：实现SwiGLU激活函数的前馈网络
    输入参数：
        config: LLaMAHFConfig - 模型配置参数
    主要组件：
        - c_fc1: 第一个线性层，用于SwiGLU的gate分支
        - c_fc2: 第二个线性层，用于SwiGLU的value分支
        - c_proj: 输出投影层
    实现过程：SwiGLU激活函数 = SiLU(gate) * value，然后通过输出投影
    特点：使用SwiGLU激活函数，比标准ReLU更高效
    """
    def __init__(self, config: LLaMAHFConfig) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        hidden_dim = 4 * config.n_embd  # 隐藏层维度为嵌入维度的4倍
        n_hidden = int(2 * hidden_dim / 3)  # 计算隐藏层大小，使用2/3比例
        N = 256  # 对齐参数
        # ensure n_hidden is multiple of N  # 确保隐藏层大小是N的倍数
        n_hidden = ((n_hidden - 1) // N) * N + N  # 向上对齐到N的倍数

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)  # 第一个线性层（gate分支）
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)  # 第二个线性层（value分支）
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)  # 输出投影层

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        """
        功能：执行SwiGLU前馈网络计算
        输入参数：
            x: torch.Tensor - 输入张量，形状为(batch_size, seq_len, n_embd)
        输出：
            torch.Tensor - 前馈网络输出，形状与输入相同
        实现过程：SwiGLU(gate, value) = SiLU(gate) * value，然后输出投影
        """
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)  # SwiGLU激活：SiLU(gate) * value
        x = self.c_proj(x)  # 输出投影
        return x  # 返回前馈网络输出


class RMSNorm(nn.Module):  # RMS归一化类
    """
    功能：实现根均方层归一化（Root Mean Square Layer Normalization）
    
    基于论文：https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    BSD 3-Clause License: https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE
    
    输入参数：
        size: int - 归一化的特征维度
        dim: int - 归一化的维度，默认为-1（最后一维）
        eps: float - 数值稳定性参数，默认为1e-5
    主要组件：
        - scale: 可学习的缩放参数
        - eps: 数值稳定性参数
    实现过程：计算RMS，然后进行归一化和缩放
    特点：比LayerNorm更简单，不需要计算均值
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.scale = nn.Parameter(torch.ones(size))  # 可学习的缩放参数
        self.eps = eps  # 数值稳定性参数
        self.dim = dim  # 归一化维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播函数
        """
        功能：执行RMS归一化
        输入参数：
            x: torch.Tensor - 输入张量
        输出：
            torch.Tensor - 归一化后的张量，形状与输入相同
        实现过程：计算RMS -> 归一化 -> 缩放
        """
        # NOTE: the original RMSNorm paper implementation is not equivalent  # 注意：原始RMSNorm论文实现不相等
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)  # 原始实现（已注释）
        # rms_x = norm_x * d_x ** (-1. / 2)  # 原始实现（已注释）
        # x_normed = x / (rms_x + self.eps)  # 原始实现（已注释）
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)  # 计算均方根
        x_normed = x * torch.rsqrt(norm_x + self.eps)  # 归一化：x / sqrt(mean(x^2) + eps)
        return self.scale * x_normed  # 应用可学习缩放参数


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:  # 构建RoPE缓存函数
    """
    功能：构建旋转位置编码（RoPE）的缓存
    输入参数：
        seq_len: int - 序列长度
        n_elem: int - 每个头的元素数量
        dtype: torch.dtype - 数据类型
        device: torch.device - 设备
        base: int - 频率基数，默认为10000
    输出：
        torch.Tensor - RoPE缓存，形状为(seq_len, n_elem//2)
    实现过程：计算频率 -> 位置索引 -> 外积 -> 极坐标转换
    
    基于论文：Enhanced Transformer with Rotary Position Embedding
    来源：https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$  # 计算频率参数
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))  # 计算频率倒数

    # Create position indexes `[0, 1, ..., seq_len - 1]`  # 创建位置索引
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)  # 生成位置索引张量

    # Calculate the product of position index and $\theta_i$  # 计算位置索引和频率的外积
    idx_theta = torch.outer(seq_idx, theta)  # 外积：位置 × 频率

    # Compute cache. Because polar only takes float32 or float64, we need to cast  # 计算缓存，因为极坐标只支持float32或float64
    # when working with 16 bit floats (float16 or bfloat16)  # 当使用16位浮点数时需要类型转换
    dtypes_requiring_casting = [torch.float16, torch.bfloat16, torch.int8]  # 需要类型转换的数据类型
    working_dtype = (  # 工作数据类型
        torch.float32 if dtype in dtypes_requiring_casting else dtype  # 如果是16位类型则使用float32
    )
    complex_dtype = (  # 复数数据类型
        torch.complex32 if dtype in dtypes_requiring_casting else torch.complex64  # 如果是16位类型则使用complex32
    )
    cache = torch.polar(  # 极坐标转换
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)  # 幅度为1，角度为idx_theta
    ).to(complex_dtype)  # 转换为复数类型
    return cache  # 返回RoPE缓存


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:  # 应用RoPE函数
    """
    功能：将旋转位置编码应用到输入张量
    输入参数：
        x: torch.Tensor - 输入张量，形状为(batch_size, n_head, seq_len, head_dim)
        rope_cache: torch.Tensor - RoPE缓存
    输出：
        torch.Tensor - 应用RoPE后的张量，形状与输入相同
    实现过程：维度调整 -> 复数转换 -> 旋转乘法 -> 实数转换 -> 维度恢复
    """
    x = x.transpose(1, 2)  # 调整维度：(batch_size, seq_len, n_head, head_dim)

    # truncate to support variable sizes  # 截断以支持可变长度
    T = x.size(1)  # 获取序列长度
    rope_cache = rope_cache[:T]  # 截断缓存到实际序列长度
    
    # cast because `view_as_complex` does not support 16 bit tensors  # 类型转换，因为view_as_complex不支持16位张量
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # 转换为复数：将最后两维合并为复数
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))  # 调整缓存维度以匹配输入
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)  # 旋转乘法：xc * rope_cache，然后转换回实数
    return x_out.transpose(1, 2).type_as(x)  # 恢复原始维度和数据类型
