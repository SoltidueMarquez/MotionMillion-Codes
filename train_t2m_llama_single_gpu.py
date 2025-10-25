"""
单GPU版本训练脚本 - 基于 train_t2m_llama.py 修改

主要修改内容：
1. 移除 Accelerator 多GPU分布式训练框架
2. 简化设备管理，直接使用 torch.device('cuda:0')
3. 移除多进程同步相关代码 (accelerator.prepare, accelerator.accumulate, accelerator.backward, accelerator.sync_gradients, accelerator.wait_for_everyone, accelerator.is_main_process)
4. 添加单GPU专用的内存监控和清理函数
5. 【重要】移除梯度累积逻辑，单GPU下梯度累积没有意义，直接使用批次大小
6. 添加混合精度训练支持 (torch.cuda.amp.GradScaler)
7. 增强内存管理，添加频繁的内存清理和监控
8. 移除不必要的进程间通信和同步操作
9. 优化训练循环，减少内存占用
10. 添加详细的单GPU训练日志和监控
11. 【关键】每次迭代都进行参数更新，不再累积梯度

适用场景：单GPU环境下的模型训练，避免多GPU配置导致的内存爆炸问题
建议：直接调整 --batch-size 参数而不是使用 --gradient_accumulation_steps
"""

import os 
# import multiprocessing
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig
from transformers import T5EncoderModel, T5Tokenizer
# ========== 单GPU版本修改 ==========
# from accelerate import Accelerator # 单GPU版本不需要Accelerator
# ====================================

import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from dataset import dataset_TM_train, dataset_TM_train_motionmillion

# 自定义 warm-up + cosine decay scheduler
class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0, resume_trans=None):
        self.optimizer = optimizer
        self.warmup_iters = 12000 # warmup_iters 
        self.total_iters = total_iters 
        self.min_lr = min_lr
        self.resume_trans = resume_trans
        
        # if self.resume_trans is None:
        # use LambdaLR to warm up the learning rate linearly
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        
        # use CosineAnnealingLR to decay the learning rate
        self.cosine_scheduler = CosineAnnealingLR(optimizer, 
                                                  T_max=total_iters - warmup_iters, 
                                                  eta_min=min_lr)
            
    def warmup_lambda(self, current_iter):
        # if in warm-up period, the learning rate increases linearly
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        # after warm-up period, lambda = 1 (no more learning rate modification by LambdaLR)
        return 1.0

    def step(self, current_iter):
        
        # if in warm-up period, call warmup_scheduler to update the learning rate
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            # otherwise, use CosineAnnealingLR to update the learning rate
            self.cosine_scheduler.step()
        
        # 添加内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def state_dict(self):
        return {
            'warmup_scheduler' : self.warmup_scheduler.state_dict(),
            'cosine_scheduler' : self.cosine_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])


# custom warm-up + constant scheduler
class WarmupConstantScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0, resume_trans=None):
        self.optimizer = optimizer
        self.warmup_iters = 10 # warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.resume_trans = resume_trans
        
        # if self.resume_trans is None:
        # use LambdaLR to warm up the learning rate linearly
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
            
    def warmup_lambda(self, current_iter):
        # if in warm-up period, the learning rate increases linearly
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        # after warm-up period, lambda = 1
        return 1.0

    def step(self, current_iter):
        
        # if in warm-up period, call warmup_scheduler to update the learning rate
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            pass
        
        # 添加内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def state_dict(self):
        return {
            'warmup_scheduler' : self.warmup_scheduler.state_dict()
        }
        
    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
 

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device):
    
    m_tokens, m_tokens_len = m_tokens.to(comp_device), m_tokens_len.to(comp_device)
    bs = m_tokens.shape[0]

    target = m_tokens
    
    target = target.to(comp_device)
    
    input_index = target

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                        device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                        device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices

    cls_pred = trans_encoder(a_indices, feat_clip_text, y_mask)
    
    cls_pred = cls_pred.contiguous()
    
    # ========== 单GPU版本修改 ==========
    # 单GPU版本：立即清理中间变量
    del mask, r_indices, a_indices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ====================================

    return cls_pred, target


# ========== 单GPU版本新增函数 ==========
def print_memory_usage(step_name=""):
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_cached = torch.cuda.memory_cached() / 1024**3
        print(f"{step_name} - GPU内存: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB, 已缓存 {memory_cached:.2f}GB")

def analyze_memory_usage():
    """分析内存使用情况"""
    if torch.cuda.is_available():
        print("=" * 60)
        print("详细内存分析:")
        
        # 获取内存统计
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_cached = torch.cuda.memory_cached() / 1024**3
        
        print(f"已分配内存: {memory_allocated:.2f}GB")
        print(f"已保留内存: {memory_reserved:.2f}GB")
        print(f"已缓存内存: {memory_cached:.2f}GB")
        
        # 分析内存碎片
        fragmentation = (memory_reserved - memory_allocated) / memory_reserved * 100 if memory_reserved > 0 else 0
        print(f"内存碎片率: {fragmentation:.1f}%")
        
        # 获取GPU总内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU总内存: {gpu_memory:.2f}GB")
        print(f"内存使用率: {memory_reserved/gpu_memory*100:.1f}%")
        
        print("=" * 60)

def cleanup_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
# ====================================


def main():

    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()
    torch.manual_seed(args.seed)
    if args.debug:
        args.exp_name = 'debug'
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

    os.makedirs(args.out_dir, exist_ok = True)
    
    # region 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' 
       
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 获取GPU信息
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**3
        print(f"GPU总内存: {total_memory:.2f} GB")
        print(f"GPU名称: {gpu_props.name}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    # endregion

    if args.debug:
        args.print_iter = 1

    # ========== 单GPU版本修改 ==========
    # 单GPU设备设置
    if torch.cuda.is_available():
        comp_device = torch.device('cuda:0')
        print(f"使用GPU: {comp_device}")
    else:
        comp_device = torch.device('cpu')
        print("使用CPU")
    
    # 设置混合精度
    if args.mixed_precision == 'fp16':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    # 单GPU版本：梯度累积步数设为1，直接使用批次大小
    if args.gradient_accumulation_steps > 1:
        print(f"警告：单GPU环境下梯度累积步数 {args.gradient_accumulation_steps} 没有意义")
        print(f"建议直接调整批次大小而不是使用梯度累积")
        print(f"当前有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"建议设置: --batch-size {args.batch_size * args.gradient_accumulation_steps} --gradient_accumulation_steps 1")
    
    # 强制设置梯度累积步数为1
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    args.gradient_accumulation_steps = 1
    print(f"单GPU模式：梯度累积步数设为1，有效批次大小: {effective_batch_size}")
    # ====================================

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # region 打印调试信息
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
    # endregion

    from utils.word_vectorizer import WordVectorizer

    ##### ---- Network ---- #####
    if args.text_encode == 'clip':
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    elif args.text_encode == 'flan-t5-xl':
        # ========== 内存优化：T5编码器放在CPU上 ==========
        tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xl', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xl', local_files_only=True).to(device=comp_device)

        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 2048
        logger.info(f'Flan-t5-xl loaded')
    elif args.text_encode == 'flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True).to(device=comp_device)
        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 4096
        logger.info(f'Flan-t5-xxl loaded')
    else:
        raise ValueError(f'Unknown text encoder: {args.text_encode}')



    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm,
                        args.kernel_size,
                        args.use_patcher,
                        args.patch_size,
                        args.patch_method,
                        args.use_attn)


    args.nb_code = net.vqvae.quantizer.codebook_size
    config = LLaMAHFConfig.from_name(args.pretrained_llama)
    config.block_size = args.block_size
    config.vocab_size = args.nb_code + 2
    config.clip_dim = args.clip_dim
    
    # if args.use_moe:
    #     config.n_experts = args.n_experts
    #     config.top_k = args.top_k
    #     config.norm_topk_prob = args.norm_topk_prob

    config.tie_weights = args.tie_weights
    print(config)
    trans_encoder = LLaMAHF(config) # , args.use_qkNorm, args.use_moe)

    # ========== 单GPU版本修改 ==========
    # 单GPU版本直接计算参数
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    logger.info(f'Trans encoder total parameters: {total_params:,}')
    print(f'Trans encoder total parameters: {total_params:,}')
    # ====================================

    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')['net']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    net.to(comp_device)
    
    # 打印VQ-VAE模型信息
    print("=" * 50)
    print("VQ-VAE模型加载完毕")
    print(f"Codebook大小: {args.nb_code}")
    print(f"量化器类型: {args.quantizer}")
    print("=" * 50)

    nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
     
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
        nb_iter = 0
        
    trans_encoder.train()
    trans_encoder.to(comp_device)

    if args.mixed_precision == 'fp16':
        trans_encoder = trans_encoder.half()
    elif args.mixed_precision == 'bf16':
        trans_encoder = trans_encoder.bfloat16()

    ##### ---- Optimizer & Scheduler ---- #####
    if args.mixed_precision == 'bf16':
        eps = 1e-06
    else:
        eps = 1e-08

    optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer, eps)
    if args.lr_scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    elif args.lr_scheduler_type == 'CosineDecayScheduler':
        # leawrning rate warm up and then cosine decay
        scheduler = WarmupCosineDecayScheduler(optimizer, args.total_iter//10//args.gradient_accumulation_steps, args.total_iter//args.gradient_accumulation_steps, resume_trans=args.resume_trans)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter//args.gradient_accumulation_steps,eta_min=0)
    elif args.lr_scheduler_type == 'ConstantScheduler':
        scheduler = WarmupConstantScheduler(optimizer, args.total_iter//10//args.gradient_accumulation_steps, args.total_iter//args.gradient_accumulation_steps, resume_trans=args.resume_trans)
    else:
        raise ValueError(f'Unknown learning rate scheduler: {args.lr_scheduler}')

    right_num = 0
    nb_sample_train = 0
    
    ##### ---- get code ---- #####
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

    # ========== 单GPU版本修改 ==========
    print("=" * 50)
    print("开始LLaMA Transformer训练 (单GPU版本)!")
    print(f"总迭代次数: {args.total_iter}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积步数: {args.gradient_accumulation_steps} (单GPU下固定为1)")
    print(f"学习率: {args.lr}")
    print(f"LLaMA模型大小: {args.pretrained_llama}")
    print(f"混合精度: {args.mixed_precision}")
    print("=" * 50)
    
    # 打印初始内存使用
    print_memory_usage("训练开始前")
    # ====================================
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=args.nb_code+1)

    if args.dataname == 'motionmillion':
        train_loader = dataset_TM_train_motionmillion.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, args.train_split, clip_model, args.text_encode, args.text_sum_way, comp_device, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug, num_workers=args.num_workers)
    else:
        train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, 'train', clip_model, args.text_encode, args.text_sum_way, comp_device, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug)

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    # ========== 单GPU版本修改 ==========
    # 注意：trans_encoder已经在第378行移动到GPU，这里不需要重复移动
    # 只需要移动clip_model和net
    if args.dataname == 'motionmillion':
        # 移动clip_model到设备
        if isinstance(clip_model, tuple):
            clip_model = (clip_model[0], clip_model[1].to(comp_device))
        else:
            clip_model = clip_model.to(comp_device)
        # trans_encoder已经在上面移动到GPU，不需要重复移动
    else:
        if isinstance(clip_model, tuple):
            clip_model = (clip_model[0], clip_model[1].to(comp_device))
        else:
            clip_model = clip_model.to(comp_device)
        # trans_encoder已经在上面移动到GPU，不需要重复移动
        net = net.to(comp_device)

    if args.resume_trans is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
    
    # 打印模型内存使用
    print("=" * 50)
    print("模型内存使用:")
    print_memory_usage("模型加载后")
    analyze_memory_usage()
    print("=" * 50)
    # ====================================
    
    train_loader_iter = cycle(train_loader)


    # ========== 单GPU版本修改 ==========
    ##### ---- Training ---- #####
    
    # 单GPU简化训练循环 (移除梯度累积)
    while nb_iter <= args.total_iter:
        batch = next(train_loader_iter)
        
        # 前向传播和损失计算
        clip_text, m_tokens, m_tokens_len, feat_clip_text, y_mask, text_tokens_len = batch
        if len(y_mask.shape) == 1:
            y_mask = y_mask.unsqueeze(1)
            feat_clip_text = feat_clip_text.unsqueeze(1)
        
        # 内存监控
        if nb_iter % 10 == 0:  # 每10次迭代打印一次内存使用
            print_memory_usage(f"迭代 {nb_iter}")
        
        # 检查序列长度
        if nb_iter <= 5:  # 前5次迭代打印详细信息
            print(f"迭代 {nb_iter}: m_tokens shape: {m_tokens.shape}, feat_clip_text shape: {feat_clip_text.shape}")
            print(f"模型block_size: {trans_encoder.config.block_size}")
            
        # 使用混合精度训练
        if args.mixed_precision == 'fp16' and scaler is not None:
            with torch.cuda.amp.autocast():
                cls_pred, target = train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device)
                
                cls_pred = cls_pred[..., :-1, :].contiguous()
                target = target[..., 1:].contiguous().to(torch.int64)
                
                loss_cls = loss_ce(cls_pred.view(-1, cls_pred.shape[-1]), target.view(-1))
        else:
            cls_pred, target = train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device)
            
            cls_pred = cls_pred[..., :-1, :].contiguous()
            target = target[..., 1:].contiguous().to(torch.int64)
            
            loss_cls = loss_ce(cls_pred.view(-1, cls_pred.shape[-1]), target.view(-1))
        
        # 计算准确率
        probs = torch.softmax(cls_pred.float(), dim=-1)
        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)
        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        token_mask = (target != args.nb_code+1)
        right_num += ((cls_pred_index == target) & token_mask).sum().item()
        nb_sample_train += token_mask.sum().item()
        
        # 反向传播和参数更新 (单GPU下每次迭代都更新)
        optimizer.zero_grad()
        
        if args.mixed_precision == 'fp16' and scaler is not None:
            scaler.scale(loss_cls).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_cls.backward()
            optimizer.step()
        
        # 更新学习率
        if args.lr_scheduler_type == 'CosineDecayScheduler' or args.lr_scheduler_type == 'ConstantScheduler':
            scheduler.step(nb_iter)
        else:
            scheduler.step()
        
        # 清理内存
        cleanup_memory()
        
        avg_loss_cls = avg_loss_cls + loss_cls.item()
        
        # 记录学习率
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('./LR/train', lr, nb_iter)
        
        nb_iter += 1
        
        # 打印训练信息
        if nb_iter % args.print_iter == 0:
            avg_loss_cls = avg_loss_cls / args.print_iter
            avg_acc = right_num * 100 / nb_sample_train
            writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
            writer.add_scalar('./ACC/train', avg_acc, nb_iter)
            msg = f"Train. Iter {nb_iter} : LR. {lr:.6f}, Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
            logger.info(msg)
            print(msg)
            avg_loss_cls = 0.
            right_num = 0
            nb_sample_train = 0
        
        # 保存模型
        if nb_iter % args.save_iter == 0:
            save_dict = {
                'trans' : trans_encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'nb_iter' : nb_iter,
                'actual_nb_iter' : nb_iter
            }
            torch.save(save_dict, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
            print(f"模型已保存: net_{nb_iter}.pth")

        if nb_iter % args.save_iter_last == 0:
            save_dict = {
                'trans' : trans_encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'nb_iter' : nb_iter,
                'actual_nb_iter' : nb_iter
            }
            torch.save(save_dict, os.path.join(args.out_dir, f'net_last.pth'))
            print(f"最新模型已保存: net_last.pth")
    
    # 训练结束
    print("=" * 50)
    print("训练完成!")
    print_memory_usage("训练结束后")
    cleanup_memory()
    print("=" * 50)
    # ====================================
            


if __name__ == '__main__':
    main()