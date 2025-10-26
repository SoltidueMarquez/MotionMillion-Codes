# 导入必要的库和模块
import os  # 操作系统接口，用于文件路径操作
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库

from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录器
from os.path import join as pjoin  # 路径拼接工具
import json  # JSON数据处理
import options.option_transformer as option_trans  # Transformer模型配置选项
import models.vqvae as vqvae  # VQ-VAE模型实现
import utils.utils_model as utils_model  # 模型工具函数
from dataset import dataset_tokenize  # 数据集token化处理
from tqdm import tqdm  # 进度条显示
from accelerate import Accelerator  # 分布式训练加速器
from tqdm import tqdm  # 进度条显示（重复导入）
import pickle  # Python对象序列化

def merge_into_pickle(root_dir, split_file_path):
    """
    将文本数据和VQ代码数据合并到一个pickle文件中
    
    Args:
        root_dir: 根目录路径
        split_file_path: 包含所有文件名列表的文件路径
    """
    # 读取所有文件名列表
    all_files = open(split_file_path, "r").readlines()
    print(f"开始合并数据，共{len(all_files)}个文件")

    # 创建数据字典结构，用于存储文本数据、代码数据和文件名
    data_dict = {
        "text_data": {},    # 存储每个文件的文本数据
        "code_data": {},    # 存储每个文件的VQ代码数据
        "file_names": []    # 存储所有文件名列表
    }

    # 遍历所有文件，加载文本和代码数据
    for files in tqdm(all_files):
        name = files.strip()  # 去除文件名中的换行符
        data_dict["file_names"].append(name)
        
        # 构建文本文件和代码文件的完整路径
        text_file_path = os.path.join(root_dir, "texts", name + ".txt")
        code_file_path = os.path.join(root_dir, "VQVAE_codebook_65536_FSQ_all", name + ".npy")
        
        # 读取文本数据（每行作为一个元素）
        text_data = open(text_file_path, "r").readlines()
        data_dict["text_data"][name] = text_data  # 存储处理后的文本数据
        
        # 加载VQ代码数据（numpy数组格式）
        code_data = np.load(code_file_path)
        data_dict["code_data"][name] = code_data

    # 将合并后的数据保存为pickle文件
    with open(os.path.join(root_dir, "all_data.pkl"), "wb") as f:
        pickle.dump(data_dict, f)
    
    print(f"数据合并完成，共处理{len(data_dict['file_names'])}个文件")
        
def main():
    """
    主函数：用于生成运动数据的VQ代码和概率分布
    主要流程：
    1. 初始化实验环境和参数
    2. 加载预训练的VQ-VAE模型
    3. 生成运动数据的VQ代码
    4. 计算代码的概率分布
    5. 合并数据到pickle文件
    """
    ##### ---- 实验目录和参数设置 ---- #####
    # 解析命令行参数
    args = option_trans.get_args_parser()
    # 设置随机种子，确保实验可重复
    torch.manual_seed(args.seed)
    # 如果是调试模式，设置实验名称为debug
    if args.debug:
        args.exp_name = 'debug'
    # 构建输出目录路径
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

    # 创建输出目录（如果不存在）
    os.makedirs(args.out_dir, exist_ok = True)

    # 调试模式下设置打印间隔
    if args.debug:
        args.print_iter = 1

    # 初始化分布式训练加速器
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    # 获取计算设备（GPU或CPU）
    comp_device = accelerator.device

    ##### ---- 日志记录器设置 ---- #####
    # 创建日志记录器
    logger = utils_model.get_logger(args.out_dir)
    # 创建TensorBoard写入器
    writer = SummaryWriter(args.out_dir)
    # 记录所有参数到日志
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    # endregion

    # 打印调试信息
    print("=" * 50)
    print("开始生成运动代码")
    print(f"实验名称: {args.exp_name}")
    print(f"输出目录: {args.out_dir}")
    print(f"设备: {comp_device}")
    print(f"数据集: {args.dataname}")
    print("=" * 50)

    # 初始化VQ-VAE模型
    # 使用参数定义不同量化器的不同参数
    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,              # 代码本大小
                        args.code_dim,             # 代码维度
                        args.output_emb_width,     # 输出嵌入宽度
                        args.down_t,               # 时间下采样倍数
                        args.stride_t,             # 时间步长
                        args.width,                # 网络宽度
                        args.depth,                # 网络深度
                        args.dilation_growth_rate, # 膨胀增长率
                        args.vq_act,               # VQ激活函数
                        args.vq_norm,              # VQ归一化
                        args.kernel_size,          # 卷积核大小
                        args.use_patcher,          # 是否使用补丁器
                        args.patch_size,           # 补丁大小
                        args.patch_method,         # 补丁方法
                        args.use_attn)             # 是否使用注意力机制

    # 更新代码本大小为实际模型中的大小
    args.nb_code = net.vqvae.quantizer.codebook_size

    # 加载预训练模型检查点
    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')['net']
    # 移除模块名前缀（用于分布式训练）
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    # 加载模型权重
    net.load_state_dict(ckpt, strict=True)
    # 设置为评估模式
    net.eval()
    # 将模型移动到指定设备
    net.to(comp_device)
    
    # 打印VQ-VAE模型信息
    print("=" * 50)
    print("VQ-VAE模型加载完毕")
    print(f"Codebook大小: {args.nb_code}")
    print(f"量化器类型: {args.quantizer}")
    print("=" * 50)

    ##### ---- 获取VQ代码 ---- #####
    # 根据数据集名称设置相应的根目录和输出路径
    if args.dataname == 'motionmillion':
        root_dir = "./dataset/MotionMillion"
        args.vq_dir = os.path.join(root_dir, f'{args.vq_name}')      # VQ代码输出目录
        args.prob_dir = os.path.join(root_dir, f'{args.vq_name}' + '_prob.npy')  # 概率分布文件路径
    elif args.dataname == 'kit':
        root_dir = "./dataset/KIT-ML"
        args.vq_dir = os.path.join(root_dir, f'{args.vq_name}')
        args.prob_dir = os.path.join(root_dir, f'{args.vq_name}' + '_prob.npy')
    elif args.dataname == 't2m':
        root_dir = "./dataset/HumanML3D"
        args.vq_dir = os.path.join(root_dir, f'{args.vq_name}')
        args.prob_dir = os.path.join(root_dir, f'{args.vq_name}' + '_prob.npy')
    
    # 分隔线 --------
    # 只在主进程中执行代码生成（避免多进程重复执行）
    if accelerator.is_main_process:

        # 检查VQ代码目录和概率文件是否已存在
        if not os.path.exists(args.vq_dir) or not os.path.exists(args.prob_dir):
            # ========== 1. 生成VQ代码文件 ==========
            logger.info(f"Start to get code from the {args.dataname}!")
            print("=" * 50)
            print("开始生成运动代码")
            print(f"VQ目录: {args.vq_dir}")
            print(f"概率文件: {args.prob_dir}")
            print("=" * 50)
            
            # 创建数据加载器，用于加载token化的运动数据
            train_loader_token, _, _ = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t, motion_type=args.motion_type, text_type=args.text_type, version=args.version)
            # 创建VQ代码输出目录
            os.makedirs(args.vq_dir, exist_ok = True)
            
            # 初始化代码计数数组（+2是为了包含特殊token）
            code_counts = torch.zeros(args.nb_code + 2, dtype=torch.long)
            total_tokens = 0  # 总token数量
            
            # 第一个循环：统计每个代码的频率
            for batch in tqdm(train_loader_token):
                pose, name = batch  # pose: 运动数据, name: 文件名
                bs, seq = pose.shape[0], pose.shape[1]  # batch_size, sequence_length
                pose = pose.to(comp_device).float()  # 将数据移动到设备并转换为float类型
                
                # 使用torch.no_grad()禁用梯度计算，节省内存
                with torch.no_grad():
                    # 使用VQ-VAE编码器将运动数据编码为离散代码
                    target = net.encode(pose)
                    # 在代码序列末尾添加结束token（值为nb_code）
                    target_with_end = torch.cat([target, torch.ones(target.shape[0], 1).to(target.device) * args.nb_code], dim=1)
                    # 统计每个代码的出现频率
                    unique_codes, counts = torch.unique(target_with_end, return_counts=True)
                    for code, count in zip(unique_codes, counts):
                        code_counts[code.long()] += count.item()
                    total_tokens += target_with_end.numel()
                    
                    # 保存代码结果到numpy文件
                    target = target.cpu().numpy()  # 转换为numpy数组并移到CPU
                    output_path = pjoin(args.vq_dir, name[0] +'.npy')  # 构建输出文件路径
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建目录
                    np.save(output_path, target)  # 保存numpy数组
            
            # ========== 2. 计算代码概率分布 ==========
            # 计算每个代码的概率分布（频率除以总token数）
            code_probs = code_counts.float() / total_tokens
            
            # ========== 3. 保存概率文件 ==========
            # 将概率分布保存到文件
            torch.save(code_probs, args.prob_dir)
            
            logger.info(f"Code distribution saved to {args.prob_dir}")
            print("=" * 50)
            print("运动代码生成完成!")
            print(f"代码分布已保存到: {args.prob_dir}")
            print(f"总token数: {total_tokens}")
            print("=" * 50)
        else:
            # 如果VQ代码和概率文件已存在，跳过生成过程
            if accelerator.is_main_process:
                logger.info(f"The code has been saved in {args.vq_dir} before!")
                print("=" * 50)
                print("运动代码已存在，跳过生成过程")
                print(f"VQ目录: {args.vq_dir}")
                print("=" * 50)
    
    # ========== 数据合并阶段 ==========
    # 将文本数据和VQ代码数据合并到一个pickle文件中
    merge_into_pickle(root_dir, pjoin(root_dir, "split/version1/t2m_60_300/all.txt"))
    
    print("=" * 50)
    print("all_data.pkl文件生成完成!")
    print(f"文件路径: {os.path.join(root_dir, 'all_data.pkl')}")
    print("=" * 50)


if __name__ == '__main__':
    main()
