# region 导入模块
import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
# endregion

# region 工具函数
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr
# endregion


def main():
    accelerator = Accelerator()

    # region 参数解析与随机种子设置
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)
    # endregion

    # region Windows 路径净化与过长路径处理，这边是我加的
    def _sanitize_name(name):
        invalid = '<>:"/\\|?*'
        name = ''.join('_' if c in invalid else c for c in str(name))
        return name.rstrip(' .') or 'exp'

    safe_exp_name = _sanitize_name(args.exp_name)
    proposed_out_dir = os.path.join(args.out_dir, safe_exp_name)

    # 过长路径自动缩短（避免 MAX_PATH 问题）
    abs_path = os.path.abspath(proposed_out_dir)
    if len(abs_path) > 200:  # 留余量，避免后续 event 文件名再加长
        import hashlib
        h = hashlib.md5(abs_path.encode('utf-8')).hexdigest()[:8]
        short_name = (safe_exp_name[:64] + '_' + h) if len(safe_exp_name) > 70 else (safe_exp_name + '_' + h)
        short_name = _sanitize_name(short_name)
        proposed_out_dir = os.path.join(args.out_dir, short_name)
        # 记录原始实验名，方便追踪
        os.makedirs(proposed_out_dir, exist_ok=True)
        try:
            with open(os.path.join(proposed_out_dir, 'EXP_NAME.txt'), 'w', encoding='utf-8') as f:
                f.write(str(args.exp_name))
        except Exception:
            pass
    # endregion
    
    # region 设置输出目录
    args.out_dir = proposed_out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    # endregion

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    
    # region 尝试创建SummaryWriter，如果失败则使用临时目录
    try:
        writer = SummaryWriter(args.out_dir)
        logger.info(f"TensorBoard logs will be saved to: {args.out_dir}")
    except (OSError, FileNotFoundError) as e:
        logger.warning(f"Failed to create SummaryWriter in {args.out_dir}: {e}")
        logger.info("Using temporary directory for TensorBoard logs...")
        import tempfile
        temp_log_dir = tempfile.mkdtemp(prefix="tensorboard_")
        writer = SummaryWriter(temp_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {temp_log_dir}")
    
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    # endregion
    
    # 初始化词向量化器
    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    # region 数据集配置（dataset_opt_path从来没用过？）
    if args.dataname == 'kit' : 
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
        args.nb_joints = 21
    elif args.dataname == 't2m':
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22
    elif args.dataname == 'motionmillion':
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')
    comp_device = accelerator.device
    # endregion

    # region 数据加载器初始化
    train_loader, train_mean, train_std = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        args.motion_type, 
                                        args.text_type,
                                        args.version, 
                                        'train', 
                                        args.debug,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        num_workers=args.num_workers)

    # val_loader, test_mean, test_std = dataset_TM_eval.MotionMillionFSQDATALoader(args.dataname, True,
    #                                     32,
    #                                     w_vectorizer,
    #                                     unit_length=2**args.down_t,
    #                                     version=args.version)
    # 验证集在 Windows/调试下使用单进程加载，避免多进程导致的 None 批次
    val_loader, test_mean, test_std = dataset_TM_eval.MotionMillionFSQDATALoader(
                                        args.dataname,
                                        True,
                                        32,
                                        w_vectorizer,
                                        num_workers=0,
                                        unit_length=2**args.down_t,
                                        version=args.version)
    # endregion

    # region 模型初始化
    #模型组件：
    # - Encoder (models/encdec.py) - 编码器
    # - Decoder (models/encdec.py) - 解码器
    # - Quantizer (models/quantize_cnn.py) - 量化器
    # - FSQ (models/FSQ.py) - 有限标量量化器
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

    if args.resume_pth : 
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        state_dict = torch.load(args.resume_pth, map_location='cpu')
        ckpt = state_dict["net"]
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        net.load_state_dict(ckpt, strict=True)
    net.train()
    net.to(comp_device)
    
    # 打印信息
    logger.info("=" * 50)
    logger.info("HumanVQVAE 模型加载完毕")
    logger.info("=" * 50)
    # endregion
    

    # region 优化器和调度器设置
    # 创建AdamW优化器，用于模型参数优化
    # AdamW是Adam优化器的改进版本，具有更好的权重衰减处理
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    
    # 创建多步学习率调度器
    # MultiStepLR在指定的milestones处将学习率乘以gamma
    # 用于在训练过程中动态调整学习率，提高训练效果
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    # 如果提供了预训练模型路径，则加载优化器和调度器的状态
    if args.resume_pth:
        # 加载优化器的状态（包括动量、学习率等）
        optimizer.load_state_dict(state_dict["optimizer"])
        # 加载学习率调度器的状态（包括当前步数、学习率等）
        scheduler.load_state_dict(state_dict["scheduler"])

    # 使用Accelerator准备所有组件，实现分布式训练支持
    # Accelerator会自动处理GPU分配、数据并行、混合精度等
    net, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            net, optimizer, train_loader, val_loader, scheduler
        )

    # 创建无限循环的训练数据迭代器
    # cycle函数确保训练数据可以无限循环使用，当数据集遍历完毕后自动重新开始
    train_loader_iter = cycle(train_loader)

    # 创建重构损失函数
    # ReConsLoss用于计算预测运动与真实运动之间的差异
    # args.recons_loss指定损失类型（如L1、L2、SmoothL1等）
    # args.nb_joints指定关节数量，用于损失计算
    Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)
    # endregion

    print("------ warm-up -------")

    # region Warm-up训练阶段
    # 初始化平均损失和指标统计变量
    # avg_recons: 平均重构损失
    # avg_perplexity: 平均困惑度（量化器的多样性指标）
    # avg_commit: 平均commit损失（量化器的稳定性指标）
    # avg_activate: 平均激活率（量化器的利用率指标）
    avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.

    # 如果不是从预训练模型恢复，则进行warm-up训练
    if not args.resume_pth:
        # Warm-up训练循环：使用较小的学习率进行预热训练
        for nb_iter in range(1, args.warm_up_iter):
            print("nb_iter: ", nb_iter)
            
            # 更新学习率：在warm-up阶段逐渐增加学习率
            # 这有助于模型稳定地开始训练，避免初始阶段的不稳定
            optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
            
            # 获取下一个训练批次的数据
            gt_motion = next(train_loader_iter) # 获取下一个训练批次的数据
            # 将数据转移到计算设备（GPU）并转换为float32类型
            gt_motion = gt_motion.to(comp_device).float() # 数据预处理和设备转移(bs, 64, dim)

            # 前向传播：通过VQ-VAE网络处理运动数据
            # 不同量化器使用相同的网络结构，但内部处理方式不同
            if args.quantizer == "FSQ":
                # FSQ量化器：有限标量量化
                pred_motion, loss_commit, perplexity, activate, _ = net(gt_motion)
            else:
                # 其他量化器：VQ-VAE、LFQ、BSQ等
                pred_motion, loss_commit, perplexity, activate, _ = net(gt_motion)

            # 计算重构损失：预测运动与真实运动之间的差异
            loss_motion = Loss(pred_motion, gt_motion)
            # 计算速度损失：预测速度与真实速度之间的差异
            # 速度损失有助于保持运动的连续性和自然性
            loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        
            # 根据不同的量化器类型组合总损失
            if args.quantizer in ["LFQ", "BSQ"]:
                # LFQ和BSQ量化器：包含commit损失
                loss = loss_motion + loss_commit + args.loss_vel * loss_vel
            elif args.quantizer == "FSQ":
                # FSQ量化器：不需要commit损失，因为FSQ是无梯度的
                loss = loss_motion + args.loss_vel * loss_vel
            else:
                # 传统VQ-VAE：包含加权的commit损失
                loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        
            # 可选的额外损失项：根据配置决定是否使用
            if args.use_acc_loss:
                # 加速度损失：计算预测加速度与真实加速度的差异
                # 有助于保持运动的平滑性和物理合理性
                loss_acc = Loss.forward_acc(pred_motion, gt_motion)
                loss = loss + args.acc_loss * loss_acc
            if args.use_acc_vel_loss:
                # 加速度速度损失：加速度的导数损失
                # 进一步约束运动的平滑性
                loss_acc_vel = Loss.forward_acc_vel(pred_motion, gt_motion)
                loss = loss + args.acc_vel_loss * loss_acc_vel
            if args.use_root_loss:
                # 根关节损失：专门针对根关节的损失
                # 根关节控制整体运动，需要特别关注
                loss_root = Loss.forward_root(pred_motion, gt_motion)
                loss = loss + args.root_loss * loss_root
            
            # 反向传播和参数更新
            optimizer.zero_grad()  # 清零梯度
            accelerator.backward(loss)  # 计算梯度（支持分布式训练）
            optimizer.step()  # 更新模型参数

            # 累积统计信息用于监控训练进度
            avg_recons += loss_motion.item()  # 累积重构损失
            avg_perplexity += perplexity.item()  # 累积困惑度
            avg_commit += loss_commit.item()  # 累积commit损失
            avg_activate += activate.item()  # 累积激活率

            # 定期打印训练信息
            if nb_iter % args.print_iter ==  0 :
                # 只在主进程中打印，避免多进程重复输出
                if accelerator.is_main_process:
                    # 计算平均损失和指标
                    avg_recons /= args.print_iter
                    avg_perplexity /= args.print_iter
                    avg_commit /= args.print_iter
                    avg_activate /= args.print_iter
                    
                    # 打印详细的训练信息
                    logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Activate. {avg_activate:.2f}")
                
                # 重置统计变量，准备下一轮的统计
                avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.
    
    print("准备开始训练")
    # endregion

    # region 主训练循环
    # 初始化训练统计变量
    # avg_recons: 平均重构损失，衡量模型重构运动的质量
    # avg_perplexity: 平均困惑度，衡量量化器的多样性
    # avg_commit: 平均commit损失，衡量量化器的稳定性
    # avg_activate: 平均激活率，衡量量化器的利用率
    avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.

    # 等待所有进程同步，确保分布式训练的一致性
    accelerator.wait_for_everyone()
    
    # 初始评估：在训练开始前进行一次评估，建立基准性能
    # 评估指标包括MPJPE(Mean Per Joint Position Error)等
    # 使用MotionMillion数据集进行分布式评估
    best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion(args.out_dir, train_loader, val_loader, net, logger, writer, 0, best_mpjpe=1000, comp_device=comp_device, codebook_size=accelerator.unwrap_model(net).vqvae.quantizer.codebook_size, accelerator=accelerator)
    # 注释掉的单GPU评估版本，用于调试或单GPU环境
    # best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion_1gpu(args.out_dir, train_loader, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_mpjpe=1000, comp_device=comp_device, draw=True, save=True, savegif=False, savenpy=False, fps=60, cal_acceleration=False)

    # 确定训练起始迭代次数
    if args.resume_pth:
        # 从预训练模型恢复：从上次保存的迭代次数继续
        start_iter = state_dict["nb_iter"] + 1
    else:
        # 从头开始训练：从第1次迭代开始
        start_iter = 1

    # 主训练循环：从start_iter到total_iter
    for nb_iter in range(start_iter, args.total_iter + 1):
        print("nb_iter: ", nb_iter)
        
        # 获取下一个训练批次的数据
        # train_loader_iter是无限循环的迭代器，确保训练数据可以持续使用
        gt_motion = next(train_loader_iter)
    
        # 前向传播：通过VQ-VAE网络处理运动数据
        # pred_motion: 预测的运动数据
        # loss_commit: commit损失（量化器稳定性）
        # perplexity: 困惑度（量化器多样性）
        # activate: 激活率（量化器利用率）
        pred_motion, loss_commit, perplexity, activate, _ = net(gt_motion)
        
        # 计算重构损失：预测运动与真实运动之间的差异
        loss_motion = Loss(pred_motion, gt_motion)
        # 计算速度损失：预测速度与真实速度之间的差异
        # 速度损失有助于保持运动的连续性和自然性
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    
        # 根据量化器类型组合损失函数
        if args.quantizer == "LFQ":
            # LFQ量化器：使用标准权重组合损失
            loss = loss_motion + loss_commit + args.loss_vel * loss_vel
        else:
            # 其他量化器（VQ-VAE等）：使用可配置的commit权重
            loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
        # 可选的额外损失项：根据配置决定是否使用
        if args.use_acc_loss:
            # 加速度损失：计算预测加速度与真实加速度的差异
            # 有助于保持运动的平滑性和物理合理性
            loss_acc = Loss.forward_acc(pred_motion, gt_motion)
            loss = loss + args.acc_loss * loss_acc
        if args.use_acc_vel_loss:
            # 加速度速度损失：加速度的导数损失
            # 进一步约束运动的平滑性
            loss_acc_vel = Loss.forward_acc_vel(pred_motion, gt_motion)
            loss = loss + args.acc_vel_loss * loss_acc_vel
    
        # 反向传播和参数更新
        optimizer.zero_grad()  # 清零梯度，准备计算新的梯度
        accelerator.backward(loss)  # 计算梯度（支持分布式训练）
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新学习率（根据调度器策略）
    
        # 累积统计信息用于监控训练进度
        avg_recons += loss_motion.item()  # 累积重构损失
        avg_perplexity += perplexity.item()  # 累积困惑度
        avg_commit += loss_commit.item()  # 累积commit损失
        avg_activate += activate.item()  # 累积激活率

        # 定期打印训练信息和记录到TensorBoard
        if nb_iter % args.print_iter ==  0 :
            # 只在主进程中打印和记录，避免多进程重复输出
            if accelerator.is_main_process:
                # 计算平均损失和指标
                avg_recons /= args.print_iter
                avg_perplexity /= args.print_iter
                avg_commit /= args.print_iter
                avg_activate /= args.print_iter
                
                # 记录到TensorBoard用于可视化训练过程
                writer.add_scalar('./Train/L1', avg_recons, nb_iter)  # 重构损失
                writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)  # 困惑度
                writer.add_scalar('./Train/Commit', avg_commit, nb_iter)  # Commit损失
                writer.add_scalar('./Train/Activate', avg_activate, nb_iter)  # 激活率
                
                # 打印详细的训练信息到日志
                logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Activate {avg_activate:.2f}")
            
            # 重置统计变量，准备下一轮的统计
            avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.
    
        # 定期评估：在验证集上评估模型性能
        if nb_iter % args.eval_iter==0 :
            # 等待所有进程同步，确保评估的一致性
            accelerator.wait_for_everyone()
            # 在MotionMillion数据集上进行分布式评估
            # 更新最佳MPJPE指标
            best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion(args.out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_mpjpe, comp_device=comp_device, codebook_size=accelerator.unwrap_model(net).vqvae.quantizer.codebook_size, accelerator=accelerator)
            # 注释掉的单GPU评估版本
            # best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion_1gpu(args.out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_mpjpe=best_mpjpe, comp_device=comp_device, draw=True, save=True, savegif=False, savenpy=False, fps=60, cal_acceleration=False)

        # 定期保存模型检查点
        accelerator.wait_for_everyone()
        if nb_iter % args.save_iter == 0 and accelerator.is_main_process:
            # 保存带迭代次数的检查点：包含模型、优化器、调度器状态
            # 用于恢复训练或分析特定迭代的模型
            torch.save({'net' : net.state_dict(), 
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'nb_iter' : nb_iter}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
        if nb_iter % args.save_latest == 0 and accelerator.is_main_process:
            # 保存最新检查点：始终保存最新的模型状态
            # 用于快速恢复训练或部署最新模型
            torch.save({'net' : net.state_dict(), 
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'nb_iter' : nb_iter}, os.path.join(args.out_dir, f'net_latest.pth'))
    
    # 训练结束，等待所有进程同步
    accelerator.wait_for_everyone()
    # endregion

if __name__ == '__main__':
    main()