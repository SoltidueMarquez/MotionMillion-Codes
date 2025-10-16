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
from accelerate import Accelerator

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
        self.warmup_iters = 12000 # warmup_iters
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
    

    return cls_pred, target


def main():

    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()
    torch.manual_seed(args.seed)
    if args.debug:
        args.exp_name = 'debug'
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

    os.makedirs(args.out_dir, exist_ok = True)
    
    # region 内存优化设置
    # 设置CUDA内存分配策略 - 不要有虚拟内存
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False' 
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' 
       
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 获取GPU信息
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**3
        print(f"GPU总内存: {total_memory:.2f} GB")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    # endregion

    if args.debug:
        args.print_iter = 1

    # accelerate
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    comp_device = accelerator.device

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
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

    from utils.word_vectorizer import WordVectorizer

    ##### ---- Network ---- #####
    if args.text_encode == 'clip':
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    elif args.text_encode == 'flan-t5-xl':
        # tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True)
        # text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True).to(device=comp_device)
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

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info(f'Trans encoder total parameters: {total_params:,}')

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

    print("=" * 50)
    print("开始LLaMA Transformer训练!")
    print(f"总迭代次数: {args.total_iter}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"LLaMA模型大小: {args.pretrained_llama}")
    print("=" * 50)
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=args.nb_code+1)

    if args.dataname == 'motionmillion':
        train_loader = dataset_TM_train_motionmillion.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, args.train_split, clip_model, args.text_encode, args.text_sum_way, comp_device, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug, num_workers=args.num_workers)
    else:
        train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, 'train', clip_model, args.text_encode, args.text_sum_way, comp_device, motion_type=args.motion_type, text_type=args.text_type, version=args.version, unit_length=2**args.down_t, debug=args.debug)

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    if args.dataname == 'motionmillion':
        clip_model, trans_encoder, optimizer, train_loader = accelerator.prepare(clip_model, trans_encoder, optimizer, train_loader)
    else:
        clip_model, trans_encoder, net, optimizer, train_loader= accelerator.prepare(clip_model, trans_encoder, net, optimizer, train_loader)

    if args.resume_trans is not None:
        unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        unwrapped_scheduler = accelerator.unwrap_model(scheduler)
        unwrapped_optimizer.load_state_dict(state_dict['optimizer'])
        unwrapped_scheduler.load_state_dict(state_dict['scheduler'])
    
    train_loader_iter = cycle(train_loader)


    ##### ---- Training ---- #####
    
    while nb_iter <= args.total_iter:

        batch = next(train_loader_iter)

        with accelerator.accumulate(trans_encoder):
            # forward pass and loss calculation
            clip_text, m_tokens, m_tokens_len, feat_clip_text, y_mask, text_tokens_len = batch
            if len(y_mask.shape) == 1:
                y_mask = y_mask.unsqueeze(1)
                feat_clip_text = feat_clip_text.unsqueeze(1)
            
            # 内存监控和序列长度检查
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"迭代 {nb_iter}: GPU内存使用 {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            
            # 检查序列长度
            if nb_iter <= 10:  # 前几次迭代打印详细信息
                print(f"迭代 {nb_iter}: m_tokens shape: {m_tokens.shape}, feat_clip_text shape: {feat_clip_text.shape}")
                print(f"模型block_size: {trans_encoder.config.block_size}")
                
            cls_pred, target = train_one_iter(feat_clip_text, m_tokens, m_tokens_len, y_mask, trans_encoder, args, comp_device)
            bs = target.shape[0]  

            loss_cls = 0.0
            
            cls_pred = cls_pred[..., :-1, :].contiguous()
            target = target[..., 1:].contiguous().to(torch.int64)

            loss_cls = loss_ce(cls_pred.view(-1, cls_pred.shape[-1]), target.view(-1))

            probs = torch.softmax(cls_pred.float(), dim=-1)
            if args.if_maxtest:
                _, cls_pred_index = torch.max(probs, dim=-1)
            else:
                dist = Categorical(probs)
                cls_pred_index = dist.sample()
            token_mask = (target != args.nb_code+1)
            right_num += ((cls_pred_index == target) & token_mask).sum().item()
            nb_sample_train += token_mask.sum().item()

            optimizer.zero_grad()
            accelerator.backward(loss_cls)

            # Memory问题出在这里
            # only on the last gradient accumulation step, execute the optimizer step
            if accelerator.sync_gradients:
                optimizer.step()

                # 立即清理优化器相关的临时内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 更新学习率
                if args.lr_scheduler_type == 'CosineDecayScheduler' or args.lr_scheduler_type == 'ConstantScheduler':
                    scheduler.step(nb_iter//args.gradient_accumulation_steps)
                else:
                    scheduler.step()

                # 调度器更新后再次清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss_cls = avg_loss_cls + loss_cls.item()
        
        if accelerator.is_main_process:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('./LR/train', lr, nb_iter//args.gradient_accumulation_steps)

        nb_iter += 1
        
        if (nb_iter-1) % args.gradient_accumulation_steps != 0:
            continue
        
        actual_nb_iter = (nb_iter-1)//args.gradient_accumulation_steps + 1
        if actual_nb_iter % args.print_iter ==  0 :
            if accelerator.is_main_process: 
                avg_loss_cls = avg_loss_cls / args.print_iter
                avg_acc = right_num * 100 / nb_sample_train
                writer.add_scalar('./Loss/train', avg_loss_cls, actual_nb_iter)
                writer.add_scalar('./ACC/train', avg_acc, actual_nb_iter)
                msg = f"Train. Iter {actual_nb_iter} : LR. {lr:.6f}, Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
                logger.info(msg)
            avg_loss_cls = 0.
            right_num = 0
            nb_sample_train = 0
        
        accelerator.wait_for_everyone()
        if actual_nb_iter % args.save_iter == 0 and accelerator.is_main_process:
            save_dict = {
                'trans' : trans_encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'nb_iter' : nb_iter,
                'actual_nb_iter' : actual_nb_iter
            }
            torch.save(save_dict, os.path.join(args.out_dir, f'net_{actual_nb_iter}.pth'))

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