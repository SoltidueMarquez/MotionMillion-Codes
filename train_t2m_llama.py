import os 
# import multiprocessing
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库

# 训练监控和工具
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录
from torch.distributions import Categorical  # 用于采样token预测（teacher forcing变体）
import json  # 用于保存配置信息
import clip  # CLIP文本编码器，用于文本特征提取

# 项目内部模块
import options.option_transformer as option_trans  # Transformer训练的参数配置
import models.vqvae as vqvae  # VQ-VAE模型，用于将动作编码为离散token
import utils.utils_model as utils_model  # 模型相关的工具函数
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig  # LLaMA Transformer模型及其配置
from transformers import T5EncoderModel, T5Tokenizer  # T5文本编码器替代CLIP

# 分布式训练
from accelerate import Accelerator  # Hugging Face的分布式训练库，支持多GPU、混合精度等

# 学习率调度器
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR  # 学习率调度器
# 数据集加载器
from dataset import dataset_TM_train, dataset_TM_train_motionmillion  # 文本到动作的训练数据集

# ==================== 自定义学习率调度器 ====================
# Warmup + Cosine Decay调度器：训练初期线性增加学习率，然后按余弦函数衰减
# 为什么这样做：
# - Warmup可以避免训练初期的大梯度导致模型不稳定
# - Cosine decay可以让模型在训练后期更精细地调整参数
class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0, resume_trans=None):
        self.optimizer = optimizer
        self.warmup_iters = 12000  # warmup迭代次数，前12000步使用warmup
        self.total_iters = total_iters  # 总迭代次数
        self.min_lr = min_lr  # 学习率的最小值
        self.resume_trans = resume_trans  # 用于恢复训练时的检查点路径
        
        # 使用LambdaLR实现线性warmup：学习率从0线性增加到目标学习率
        # 这样可以避免训练初期的大梯度导致的不稳定
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        
        # 使用CosineAnnealingLR实现余弦衰减：学习率按余弦函数从最大值衰减到最小值
        # 余弦衰减可以让训练后期更平滑地收敛
        self.cosine_scheduler = CosineAnnealingLR(optimizer, 
                                                  T_max=total_iters - warmup_iters,  # 衰减阶段的总步数
                                                  eta_min=min_lr)  # 最小学习率
            
    def warmup_lambda(self, current_iter):
        """计算当前迭代的学习率倍数（lambda函数）"""
        # 在warmup期间，学习率线性增加：从0增加到1
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        # warmup结束后，lambda=1，不再由LambdaLR修改学习率
        return 1.0

    def step(self, current_iter):
        """根据当前迭代步数更新学习率"""
        # 如果还在warmup阶段，使用warmup调度器
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            # warmup结束，使用余弦衰减调度器
            self.cosine_scheduler.step()
        
        # 添加内存清理：每步后清理CUDA缓存，避免内存泄漏
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def state_dict(self):
        """保存调度器状态（用于检查点保存）"""
        return {
            'warmup_scheduler' : self.warmup_scheduler.state_dict(),
            'cosine_scheduler' : self.cosine_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态（用于检查点恢复）"""
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])


# Warmup + Constant调度器：训练初期线性增加学习率，然后保持恒定
# 为什么这样做：
# - Warmup避免训练初期的梯度爆炸
# - 恒定学习率更简单，适合某些场景
class WarmupConstantScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0, resume_trans=None):
        self.optimizer = optimizer
        self.warmup_iters = 12000  # warmup迭代次数
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.resume_trans = resume_trans
        
        # 使用LambdaLR实现线性warmup
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
            
    def warmup_lambda(self, current_iter):
        """计算当前迭代的学习率倍数"""
        # 在warmup期间，学习率线性增加：从0增加到1
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        # warmup结束后，lambda=1，保持学习率不变
        return 1.0

    def step(self, current_iter):
        """根据当前迭代步数更新学习率"""
        # 如果还在warmup阶段，使用warmup调度器
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            # warmup结束，不更新学习率（保持恒定）
            pass
        
        # 添加内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def state_dict(self):
        """保存调度器状态"""
        return {
            'warmup_scheduler' : self.warmup_scheduler.state_dict()
        }
        
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
 

# ==================== 辅助函数 ====================
def cycle(iterable):
    """创建一个无限循环的迭代器"""
    # 用于数据加载器，保证训练时可以无限循环获取数据
    while True:
        for x in iterable:
            yield x

def train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device):
    """
    执行一次训练迭代的前向传播
    参数：
        feat_clip_text: 文本特征的CLIP编码 [batch_size, seq_len, clip_dim]
        m_tokens: 动作的离散token编码 [batch_size, seq_len]
        m_tokens_len: 每个序列的实际长度
        y_mask: 文本mask
        trans_encoder: Transformer编码器
        args: 训练参数
        comp_device: 计算设备（CPU/GPU）
    
    返回：
        cls_pred: 模型对每个token位置的预测 [batch_size, seq_len, nb_code]
        target: 真实的token标签 [batch_size, seq_len]
    """
    
    # 将tokens移动到计算设备（GPU/CPU）
    m_tokens, m_tokens_len = m_tokens.to(comp_device), m_tokens_len.to(comp_device)
    bs = m_tokens.shape[0]  # batch_size

    target = m_tokens  # 目标token序列（用于计算损失）
    target = target.to(comp_device)
    
    input_index = target  # 输入就是目标序列

    # ==================== 实施"Masking"策略（类似BERT的Masked LM） ====================
    # 这是一种正则化技术，用于提升模型的鲁棒性
    # 为什么要这样做：
    # 1. 随机mask掉一些token，让模型学习从部分信息预测完整序列
    # 2. 这类似于BERT的训练方式，可以让模型更强大
    # 3. 通过在部分token被替换的序列上训练，模型学会更好地理解上下文
    
    if args.pkeep == -1:
        # 如果pkeep=-1，随机选择mask的概率
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                        device=input_index.device))
    else:
        # 使用固定的pkeep作为mask概率（例如0.9表示保留90%的token）
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                        device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    
    # 生成随机的token来替换被mask的位置
    r_indices = torch.randint_like(input_index, args.nb_code)
    
    # 应用mask：mask=1的位置保留原token，mask=0的位置替换为随机token
    a_indices = mask*input_index+(1-mask)*r_indices

    # ==================== 前向传播 ====================
    # trans_encoder接收：
    # - a_indices: 部分被mask的动作token序列
    # - feat_clip_text: 文本特征
    # - y_mask: 文本mask
    # 输出：对每个token位置的预测（所有可能token的概率分布）
    cls_pred = trans_encoder(a_indices, feat_clip_text, y_mask)
    
    # 确保tensor在内存中是连续的（某些PyTorch操作需要连续内存）
    cls_pred = cls_pred.contiguous()
    

    return cls_pred, target


# ==================== 主训练函数 ====================
def main():

    # region ==================== 第一步：初始化实验目录 ====================
    # 解析命令行参数（批次大小、学习率、数据路径等）
    args = option_trans.get_args_parser()
    
    # 设置随机种子，确保结果可复现
    torch.manual_seed(args.seed)
    
    # 调试模式：将实验名称设为'debug'
    if args.debug:
        args.exp_name = 'debug'
    
    # 创建输出目录：用于保存模型检查点、日志等
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

    os.makedirs(args.out_dir, exist_ok = True)
    # endregion
    
    # region ==================== 第二步：GPU内存优化设置 ====================
    # 为什么要这样做：
    # - 大模型训练容易显存不足，需要优化内存管理
    # - expandable_segments允许PyTorch更灵活地管理CUDA内存
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' 
       
    if torch.cuda.is_available():
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 获取GPU信息（用于监控）
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**3
        print(f"GPU总内存: {total_memory:.2f} GB")
        
        # 强制垃圾回收，释放未使用的内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    # endregion

    # 调试模式：每1次迭代打印一次
    if args.debug:
        args.print_iter = 1

    # region ==================== 第三步：初始化Accelerator ====================
    # 为什么要使用Accelerator：
    # 1. 自动处理多GPU训练
    # 2. 支持混合精度训练（fp16/bf16），节省内存并加速
    # 3. 支持梯度累积，即使批次很小也能模拟大批次训练
    # 4. 简化分布式训练代码
    accelerator = Accelerator(mixed_precision=args.mixed_precision,  # 混合精度类型
                              gradient_accumulation_steps=args.gradient_accumulation_steps)  # 梯度累积步数
    
    comp_device = accelerator.device  # 获取实际的计算设备（主GPU）
    # endregion

    # region ==================== 第四步：初始化日志系统 ====================
    # logger: 用于保存训练日志到文件
    logger = utils_model.get_logger(args.out_dir)
    # writer: 用于TensorBoard可视化（损失、准确率等）
    writer = SummaryWriter(args.out_dir)
    # 将训练参数保存到日志文件中
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    # endregion    

    # 打印调试信息
    print("=" * 50)
    print("开始初始化LLaMA Transformer训练")
    print(f"实验名称: {args.exp_name}")
    print(f"输出目录: {args.out_dir}")
    print(f"设备: {comp_device}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps}")
    if torch.cuda.is_available():
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 50)

    # ==================== 第五步：加载WordVectorizer ====================
    # 用于文本的词向量编码（如果使用词汇表编码方式）
    from utils.word_vectorizer import WordVectorizer

    # region ==================== 第六步：初始化文本编码器 ====================
    # 为什么要用文本编码器：
    # - 将文本描述转换为连续的向量表示
    # - 这些向量作为条件输入，指导模型生成对应的动作序列
    # - 可以理解为"告诉模型想生成什么样的动作"
    
    if args.text_encode == 'clip':
        # 使用CLIP作为文本编码器（视觉-语言模型）
        # jit=False: 关闭JIT编译，因为训练时需要动态图
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)
        clip.model.convert_weights(clip_model)
        
        # 冻结文本编码器的参数（不训练，只用于特征提取）
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    elif args.text_encode == 'flan-t5-xl':
        # 使用Flan-T5-XL作为文本编码器（更大的T5模型，2048维）
        tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xl', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xl', local_files_only=True).to(device=comp_device)

        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 2048  # 设置特征维度为2048
        logger.info(f'Flan-t5-xl loaded')
    elif args.text_encode == 'flan-t5-xxl':
        # 使用Flan-T5-XXL作为文本编码器（更大的模型，4096维）
        tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True).to(device=comp_device)
        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 4096  # 设置特征维度为4096
        logger.info(f'Flan-t5-xxl loaded')
    else:
        raise ValueError(f'Unknown text encoder: {args.text_encode}')
    # endregion

    # region ==================== 第七步：初始化VQ-VAE模型 ====================
    # 为什么要用VQ-VAE：
    # - 将连续的动作序列编码为离散的token序列
    # - 这是一个预训练的模型，将动作压缩为codebook中的code
    # - Transformer只学习在token级别生成，而不是在原始动作空间
    # - 这类似于GPT处理语言：先在单词级tokenize，再在token级训练
    
    net = vqvae.HumanVQVAE(args,  # 使用args定义不同的量化器参数
                        args.nb_code,  # codebook大小
                        args.code_dim,  # code维度
                        args.output_emb_width,  # 输出embedding宽度
                        args.down_t,  # 时序下采样倍数
                        args.stride_t,  # 卷积步长
                        args.width,  # 网络宽度
                        args.depth,  # 网络深度
                        args.dilation_growth_rate,  # 膨胀率
                        args.vq_act,  # 激活函数类型
                        args.vq_norm,  # 归一化方式
                        args.kernel_size,  # 卷积核大小
                        args.use_patcher,  # 是否使用patcher
                        args.patch_size,  # patch大小
                        args.patch_method,  # patch方法
                        args.use_attn)  # 是否使用注意力机制
    
    # 从VQ-VAE中获取实际的codebook大小
    args.nb_code = net.vqvae.quantizer.codebook_size
    # endregion
    
    # region ==================== 第八步：初始化LLaMA Transformer模型 ====================
    # 为什么要用Transformer：
    # - Transformer可以学习文本条件到动作token的映射
    # - 使用自回归生成：每次预测下一个token
    # - LLaMA架构提供强大的表示学习能力
    
    config = LLaMAHFConfig.from_name(args.pretrained_llama)
    config.block_size = args.block_size  # 最大序列长度
    config.vocab_size = args.nb_code + 2  # 词汇表大小 = codebook大小 + 2（特殊token：起始和结束）
    config.clip_dim = args.clip_dim  # 文本特征维度
    
    # 权重绑定（可选）：减少参数量
    config.tie_weights = args.tie_weights
    
    print(config)
    trans_encoder = LLaMAHF(config)

    # 打印模型参数量（仅在主进程中）
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info(f'Trans encoder total parameters: {total_params:,}')
    # endregion
    
    # region ==================== 第九步：加载预训练的VQ-VAE ====================
    # 为什么只加载VQ-VAE，不训练它：
    # - VQ-VAE已经预训练好，可以稳定地将动作编码为token
    # - Transformer负责学习token级的生成，所以只需要训练Transformer
    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')['net']
    # 处理分布式训练后的key名（去掉'module.'前缀）
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
    net.eval()  # 设为评估模式（不训练）
    net.to(comp_device)
    
    # 打印VQ-VAE模型信息
    print("=" * 50)
    print("VQ-VAE模型加载完毕")
    print(f"Codebook大小: {args.nb_code}")
    print(f"量化器类型: {args.quantizer}")
    print("=" * 50)
    # endregion

    # ==================== 第十步：初始化训练状态变量 ====================
    nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.  # nb_iter: 当前迭代次数，avg_loss_cls: 平均损失，avg_acc: 平均准确率
     
    # region ==================== 第十一步：恢复训练（如果提供了检查点） ====================
    # 用于从之前的检查点继续训练
    if args.resume_trans is not None:
        print ('loading transformer checkpoint from {}'.format(args.resume_trans))
        state_dict = torch.load(args.resume_trans, map_location='cpu')
        ckpt = state_dict['trans']
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        trans_encoder.load_state_dict(ckpt, strict=True)
        nb_iter = state_dict['nb_iter']
        print(f'loading transformer checkpoint from {args.resume_trans}, nb_iter: {nb_iter}')
        nb_iter = nb_iter + 1
    else:
        nb_iter = 0  # 从0开始训练
        
    # 设置Transformer为训练模式
    trans_encoder.train()
    trans_encoder.to(comp_device)
    # endregion
    
    # region ==================== 第十二步：设置混合精度（可选） ====================
    # 混合精度的好处：
    # - 节省内存：fp16占一半内存，bf16同样节省内存
    # - 加速训练：某些硬件对半精度运算更快
    # - 需要注意：可能影响数值稳定性
    if args.mixed_precision == 'fp16':
        trans_encoder = trans_encoder.half()  # 使用fp16
    elif args.mixed_precision == 'bf16':
        trans_encoder = trans_encoder.bfloat16()  # 使用bf16
    # endregion
    
    # region ==================== 第十三步：初始化优化器和学习率调度器 ====================
    # 优化器：负责更新模型参数（梯度下降）
    # 学习率调度器：动态调整学习率
    
    if args.mixed_precision == 'bf16':
        eps = 1e-06  # bf16的数值稳定性更好，可以用更大的epsilon
    else:
        eps = 1e-08  # fp16和fp32用更小的epsilon

    # 初始化优化器（Adam等）
    optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer, eps)
    
    # 选择学习率调度策略
    if args.lr_scheduler_type == 'MultiStepLR':
        # 多步长衰减：在指定里程碑处降低学习率
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    elif args.lr_scheduler_type == 'CosineDecayScheduler':
        # Warmup + Cosine衰减（推荐）：训练更稳定
        scheduler = WarmupCosineDecayScheduler(optimizer, 
                                              args.total_iter//10//args.gradient_accumulation_steps,  # warmup步数
                                              args.total_iter//args.gradient_accumulation_steps,  # 总步数
                                              resume_trans=args.resume_trans)
    elif args.lr_scheduler_type == 'ConstantScheduler':
        # Warmup + 恒定学习率：更简单的方式
        scheduler = WarmupConstantScheduler(optimizer, 
                                           args.total_iter//10//args.gradient_accumulation_steps, 
                                           args.total_iter//args.gradient_accumulation_steps, 
                                           resume_trans=args.resume_trans)
    else:
        raise ValueError(f'Unknown learning rate scheduler: {args.lr_scheduler}')
    
    # 用于计算准确率的变量
    right_num = 0  # 预测正确的token数量
    nb_sample_train = 0  # 训练样本总数
    # endregion
    
    # region ==================== 第十四步：设置数据路径 ====================
    # 根据数据集名称设置VQ码本的路径
    if args.dataname == 'motionmillion':
        args.vq_dir = os.path.join("./dataset/MotionMillion", f'{args.vq_name}')
        args.prob_dir = os.path.join("./dataset/MotionMillion", f'{args.vq_name}' + '_prob.npy')
    elif args.dataname == 'motionx_0717':
        args.vq_dir = os.path.join("./dataset/Motion-X_++_v.1.0.717", f'{args.vq_name}')
        args.prob_dir = os.path.join("./dataset/Motion-X_++_v.1.0.717", f'{args.vq_name}' + '_prob.npy')
    elif args.dataname == 'kit':
        args.vq_dir = os.path.join("./dataset/KIT-ML", f'{args.vq_name}')
        args.prob_dir = os.path.join("./dataset/KIT-ML", f'{args.vq_name}' + '_prob.npy')
    elif args.dataname == 't2m':
        args.vq_dir = os.path.join("./dataset/HumanML3D", f'{args.vq_name}')
        args.prob_dir = os.path.join("./dataset/HumanML3D", f'{args.vq_name}' + '_prob.npy')

    print("=" * 50)
    print("开始LLaMA Transformer训练!")
    print(f"总迭代次数: {args.total_iter}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"LLaMA模型大小: {args.pretrained_llama}")
    print("=" * 50)
    # endregion
    
    # region ==================== 第十五步：定义损失函数 ====================
    # 交叉熵损失：用于预测下一个token
    # ignore_index: 忽略填充token（索引为nb_code+1的token）
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=args.nb_code+1)
    # endregion

    # ==================== 第十六步：加载训练数据集 ====================
    # 数据集会提供：(文本描述, 动作token序列)
    if args.dataname == 'motionmillion':
        train_loader = dataset_TM_train_motionmillion.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, args.train_split, clip_model, args.text_encode, args.text_sum_way, comp_device, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug, num_workers=args.num_workers)
    else:
        train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, 'train', clip_model, args.text_encode, args.text_sum_way, comp_device, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug)

    # 加载词汇向量器（用于某些数据集）
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    
    # ==================== 第十七步：使用Accelerator准备训练 ====================
    # accelerate.prepare()会：
    # 1. 将模型/优化器/数据分配到正确的GPU
    # 2. 包装模型以支持分布式训练
    # 3. 设置混合精度训练
    if args.dataname == 'motionmillion':
        # MotionMillion数据集不需要VQ-VAE
        clip_model, trans_encoder, optimizer, train_loader = accelerator.prepare(clip_model, trans_encoder, optimizer, train_loader)
    else:
        # 其他数据集需要VQ-VAE
        clip_model, trans_encoder, net, optimizer, train_loader= accelerator.prepare(clip_model, trans_encoder, net, optimizer, train_loader)

    # 如果是从检查点恢复，需要加载优化器和调度器状态
    if args.resume_trans is not None:
        unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        unwrapped_scheduler = accelerator.unwrap_model(scheduler)
        unwrapped_optimizer.load_state_dict(state_dict['optimizer'])
        unwrapped_scheduler.load_state_dict(state_dict['scheduler'])
    
    # 创建无限循环的数据迭代器
    train_loader_iter = cycle(train_loader)


    # ==================== 第十八步：开始训练循环 ====================
    # 这是训练的核心循环：重复执行前向传播、计算损失、反向传播、更新参数
    
    while nb_iter <= args.total_iter:
        # 从数据加载器中获取一个批次的数据
        batch = next(train_loader_iter)

        # accelerator.accumulate()用于梯度累积
        # 梯度累积的作用：即使GPU内存有限，也能模拟大批次的训练效果
        with accelerator.accumulate(trans_encoder):
            # ==================== 数据解包和预处理 ====================
            clip_text, m_tokens, m_tokens_len, feat_clip_text, y_mask, text_tokens_len = batch
            
            # 确保mask和文本特征的维度正确
            if len(y_mask.shape) == 1:
                y_mask = y_mask.unsqueeze(1)
                feat_clip_text = feat_clip_text.unsqueeze(1)
            
            # region ==================== 内存监控（我加的） ====================
            # 用于监控GPU内存使用情况，帮助调试内存问题
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"迭代 {nb_iter}: GPU内存使用 {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            # endregion
            
            # 检查序列长度（用于调试）
            if nb_iter <= 10:
                print(f"迭代 {nb_iter}: m_tokens shape: {m_tokens.shape}, feat_clip_text shape: {feat_clip_text.shape}")
                print(f"模型block_size: {trans_encoder.config.block_size}")
                
            # ==================== 前向传播 ====================
            # 调用train_one_iter执行：
            # 1. 对token序列进行mask（随机替换一些token）
            # 2. 输入Transformer得到预测
            cls_pred, target = train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device)
            bs = target.shape[0]  # batch size

            loss_cls = 0.0
            
            # ==================== 准备损失计算 ====================
            # 为什么是这样的shift：
            # - cls_pred[..., :-1, :]: 预测序列（去掉最后一个位置）
            # - target[..., 1:]: 真实序列（去掉第一个位置）
            # - 这是自回归模型的标准训练方式：
            #   输入: [start, token1, token2, ..., token_n-1]
            #   预测: [token1, token2, ..., token_n]
            #   即：用前n-1个token预测后n-1个token
            
            cls_pred = cls_pred[..., :-1, :].contiguous()  # [batch_size, seq_len-1, nb_code]
            target = target[..., 1:].contiguous().to(torch.int64)  # [batch_size, seq_len-1]

            # ==================== 计算损失 ====================
            # 交叉熵损失：衡量预测的token概率分布与真实token的距离
            loss_cls = loss_ce(cls_pred.view(-1, cls_pred.shape[-1]), target.view(-1))

            # ==================== 计算准确率（用于监控训练进度） ====================
            probs = torch.softmax(cls_pred.float(), dim=-1)  # 将logits转换为概率
            if args.if_maxtest:
                # 使用最大概率的token（贪婪解码）
                _, cls_pred_index = torch.max(probs, dim=-1)
            else:
                # 从概率分布中采样（更接近实际生成时的过程）
                dist = Categorical(probs)
                cls_pred_index = dist.sample()
            
            # 统计准确率：排除填充token
            token_mask = (target != args.nb_code+1)
            right_num += ((cls_pred_index == target) & token_mask).sum().item()
            nb_sample_train += token_mask.sum().item()

            # ==================== 反向传播和优化 ====================
            # 清空上一步的梯度
            optimizer.zero_grad()
            # 反向传播：计算梯度
            accelerator.backward(loss_cls)

            # ==================== 梯度累积逻辑 ====================
            # 只有当累积了足够步数后，才真正更新参数
            if accelerator.sync_gradients:
                # 更新模型参数（真正的优化器step）
                optimizer.step()

                # 立即清理优化器相关的临时内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ==================== 更新学习率 ====================
                # 根据训练进度调整学习率
                if args.lr_scheduler_type == 'CosineDecayScheduler' or args.lr_scheduler_type == 'ConstantScheduler':
                    # Warmup调度器需要传入当前迭代次数
                    scheduler.step(nb_iter//args.gradient_accumulation_steps)
                else:
                    # 标准调度器只需要调用step
                    scheduler.step()

                # 调度器更新后再次清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ==================== 记录训练指标 ====================
        avg_loss_cls = avg_loss_cls + loss_cls.item()
        
        # 在主进程中记录学习率到TensorBoard
        if accelerator.is_main_process:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('./LR/train', lr, nb_iter//args.gradient_accumulation_steps)

        nb_iter += 1
        
        # ==================== 梯度累积控制 ====================
        # 如果不是累积的最后一步，跳过打印和保存
        if (nb_iter-1) % args.gradient_accumulation_steps != 0:
            continue
        
        # 计算实际的有效迭代次数（考虑梯度累积）
        actual_nb_iter = (nb_iter-1)//args.gradient_accumulation_steps + 1
        
        # ==================== 打印训练进度 ====================
        if actual_nb_iter % args.print_iter ==  0 :
            if accelerator.is_main_process: 
                avg_loss_cls = avg_loss_cls / args.print_iter
                avg_acc = right_num * 100 / nb_sample_train
                # 记录到TensorBoard
                writer.add_scalar('./Loss/train', avg_loss_cls, actual_nb_iter)
                writer.add_scalar('./ACC/train', avg_acc, actual_nb_iter)
                # 打印日志
                msg = f"Train. Iter {actual_nb_iter} : LR. {lr:.6f}, Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
                logger.info(msg)
            # 重置统计变量
            avg_loss_cls = 0.
            right_num = 0
            nb_sample_train = 0
        
        # ==================== 同步所有进程 ====================
        accelerator.wait_for_everyone()
        
        # ==================== 保存检查点 ====================
        # 定期保存带有编号的检查点
        if actual_nb_iter % args.save_iter == 0 and accelerator.is_main_process:
            save_dict = {
                'trans' : trans_encoder.state_dict(),  # Transformer模型参数
                'optimizer' : optimizer.state_dict(),  # 优化器状态（用于恢复训练）
                'scheduler' : scheduler.state_dict(),  # 学习率调度器状态
                'nb_iter' : nb_iter,  # 当前迭代次数
                'actual_nb_iter' : actual_nb_iter  # 有效迭代次数
            }
            torch.save(save_dict, os.path.join(args.out_dir, f'net_{actual_nb_iter}.pth'))

        # 保存最新的检查点（每次覆盖）
        if actual_nb_iter % args.save_iter_last == 0 and accelerator.is_main_process:
            save_dict = {
                'trans' : trans_encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'nb_iter' : nb_iter,
                'actual_nb_iter' : actual_nb_iter
            }
            torch.save(save_dict, os.path.join(args.out_dir, f'net_last.pth'))
            


if __name__ == '__main__':
    main()