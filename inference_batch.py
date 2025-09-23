import torch
import options.option_transformer as option_trans  # 导入Transformer模型的配置选项
import clip  # 导入CLIP模型相关
import numpy as np
import random
import models.vqvae as vqvae  # 导入VQ-VAE模型
import os
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig  # 导入LLaMA模型
from transformers import T5EncoderModel, T5Tokenizer  # 导入T5文本编码器
from utils.quaternion import *  # 导入四元数相关工具函数
from visualize.plot_3d_global import plot_3d_motion  # 导入3D动作可视化函数
from visualize.smplx2joints import process, save_mesh, process_smplx_data  # 导入SMPLX模型处理函数

import imageio  # 用于生成GIF
import sys
from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle  # 导入旋转表示转换工具
from transformers import pipeline  # 导入HuggingFace管道
import re  # 正则表达式
from tqdm import tqdm  # 进度条
import moviepy.editor as mp  # 视频处理
import sys


def rotations_matrix_to_smplx85(rotations_matrix, translation):
    """
    将旋转矩阵和位移转换为SMPLX-85格式的数据
    输入:
        rotations_matrix: (nfrm, njoint, 3, 3) 旋转矩阵
        translation: (nfrm, 3) 位移向量
    输出:
        smplx_85: (nfrm, 85) SMPLX格式数据
    """
    nfrm, njoint, _, _ = rotations_matrix.shape
    # 将旋转矩阵转换为轴角表示
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    # 拼接轴角、零填充、位移和零填充得到SMPLX-85格式
    smplx_85 = np.concatenate([axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1)
    return smplx_85


def inv_transform(data, mean, std):
    """
    逆标准化操作，将标准化后的数据还原为原始数据
    
    输入:
        data: numpy数组, 标准化后的数据
        mean: numpy数组, 均值
        std: numpy数组, 标准差
    
    输出:
        numpy数组, 还原后的原始数据
    """
    return data * std + mean


def accumulate_rotations(relative_rotations):
    """
    累积旋转矩阵，将连续的相对旋转转换为绝对旋转
    
    输入:
        relative_rotations: numpy数组, 形状为(nfrm, 3, 3)的相对旋转矩阵序列
    
    输出:
        numpy数组, 形状为(nfrm, 3, 3)的累积旋转矩阵序列
    """
    # 初始化总旋转为第一个相对旋转
    R_total = [relative_rotations[0]]
    # 迭代累积所有相对旋转
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    
    return np.array(R_total)


# add hip height to translation when recoverring from rotation
def recover_from_local_rotation(final_x, njoint):
    """
    从局部旋转中恢复全局的SMPLX数据
    
    输入:
        final_x: numpy数组, 形状为(nfrm, 8+12*njoint)的局部旋转数据
        njoint: 整数, 关节数量
    
    输出:
        smplx_85: numpy数组, 形状为(nfrm, 85)的SMPLX格式数据
    """
    nfrm, _ = final_x.shape
    # 将6D旋转表示转换为旋转矩阵
    rotations_matrix = rotation_6d_to_matrix(torch.from_numpy(final_x[:,8+6*njoint:8+12*njoint]).reshape(nfrm, -1, 6)).numpy()
    global_heading_diff_rot = final_x[:,2:8]  # 全局朝向差分的6D表示
    velocities_root_xy_no_heading = final_x[:,:2]  # 根节点XY速度（无朝向）
    positions_no_heading = final_x[:, 8:8+3*njoint].reshape(nfrm, -1, 3)  # 关节位置（无朝向）
    height = positions_no_heading[:, 0, 1]  # 根节点高度
    # 累积全局朝向旋转
    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # 应用逆全局旋转到根关节
    rotations_matrix[:,0,...] = np.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])
    # 处理速度并积分得到位移
    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)  # 积分得到位移
    root_translation[:, 1] = height  # 设置高度
    # 转换为SMPLX-85格式
    smplx_85 = rotations_matrix_to_smplx85(rotations_matrix, root_translation)
    return smplx_85


def smplx85_2_smplx322(smplx_no_shape_data):
    """
    将85维的SMPLX数据转换为322维的SMPLX数据（填充零）
    
    输入:
        smplx_no_shape_data: numpy数组, 形状为(nfrm, 85)的SMPLX数据
    
    输出:
        result: numpy数组, 形状为(nfrm, 322)的完整SMPLX数据
    """
    result = np.concatenate((smplx_no_shape_data[:,:66], np.zeros((smplx_no_shape_data.shape[0], 90)), np.zeros((smplx_no_shape_data.shape[0], 3)), np.zeros((smplx_no_shape_data.shape[0], 50)), np.zeros((smplx_no_shape_data.shape[0], 100)), smplx_no_shape_data[:,72:72+3], smplx_no_shape_data[:,75:]), axis=-1)
    return result


def visualize_smplx_85(data, title=None, output_path='./recon_272/0_14_rot_new3.mp4', fps=60):
    """
    可视化SMPLX数据，生成动作视频
    
    输入:
        data: numpy数组, SMPLX格式的动作数据
        title: 字符串, 可选的标题
        output_path: 字符串, 输出视频文件路径
        fps: 整数, 帧率
    
    输出:
        无直接返回值，但会保存视频文件到指定路径
    """
    smplx_85_data = data
    if len(smplx_85_data.shape) == 3:
       smplx_85_data = np.squeeze(smplx_85_data, axis=0)
    # 转换为SMPLX-322格式
    smplx_85_data = smplx85_2_smplx322(smplx_85_data)
    # 处理SMPLX数据得到关节位置
    vert, joints, motion, faces = process_smplx_data(smplx_85_data, norm_global_orient=False, transform=False)
    xyz = joints[:, :22, :].reshape(-1, 22, 3).detach().cpu().numpy()  # 提取22个关节的3D位置
    # 创建输出目录并生成动画
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = plot_3d_motion([xyz, None, None])
    imageio.mimsave(output_path, np.array(img), fps=fps) 
    # 转换为MP4格式
    out_video = mp.VideoFileClip(output_path)
    out_video.write_videofile(output_path.replace('.gif', '.mp4'))
    

def construct_llama(rewrite_model_path):
    """
    构建LLaMA模型用于文本重写
    
    输入:
        rewrite_model_path: 字符串, LLaMA模型路径
    
    输出:
        pipe: transformers的pipeline对象, 用于文本生成
    """
    model_id = rewrite_model_path
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    return pipe


def call_llama_rewrite(pipe, input_text):
    """
    调用LLaMA模型对输入文本进行重写
    
    输入:
        pipe: transformers的pipeline对象, 用于文本生成
        input_text: 字符串, 需要重写的原始文本
    
    输出:
        response_list[0]: 字符串, 重写后的文本
    """
    prompt = f"You are a helpful assistant. You can rewrite the following sentence and do not change the meaning of the sentence. \
        You can change the subject and the way of description, but do not alter the original meaning. \
        Requirements: 1. Do not change the meaning of the sentence. 2. Does not change the logical relationship of the sentences. 3. The rewritten sentence should be more concise and clear. \
        The rewritten sentence should be written in a pair of brackets. \
        Example: Given the sentence: A man is walking forward for a few steps. \
        The rewritten sentence is: {{A man is walking starightly.}} \
        Now rewrite the following sentence: {input_text}"
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe( 
        messages,
        max_new_tokens=2560,
        do_sample=True,
    )
    
    response = outputs[0]["generated_text"][-1]["content"]
    # 使用正则表达式提取重写后的文本
    response_list = re.findall(r'\[(.*?)\]', response)
    response_list += re.findall(r'\{(.*?)\}', response)
    response_list = [item.strip() for item in response_list]
    
    return response_list[0]


@torch.no_grad()
def plot(pred_pose_denorm, dataname):
    """
    绘制预测动作（已弃用？）
    输入:
        pred_pose_denorm: 去归一化后的预测姿态
        dataname: 数据名称
    输出:
        pred_xyz: 恢复的SMPLX数据
        img: 可视化图像
    """
    pred_xyz = recover_from_local_rotation(pred_pose_denorm.squeeze(0).cpu().numpy(), njoint=22)
    img  = visualize_smplx_85(pred_xyz)
    return pred_xyz, img


if __name__ == '__main__':

    comp_device = torch.device('cuda:0')  # 计算设备
    args = option_trans.get_args_parser()  # 获取参数

    # pipe = construct_llama(args.rewrite_model_path)
    
    # 加载文本编码模型（clip）
    if args.text_encode == 'clip':
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)  # 转换为float16， Actually this line is unnecessary since clip by default already on float16
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    elif args.text_encode == 'flan-t5-xl':
        # 这里是我修改的，旧的模型路径不一样了
        # tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True)
        # text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True).to(device=comp_device)
        tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xl', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xl', local_files_only=True).to(device=comp_device)
        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 2048  # 设置编码维度
        print(f'Flan-t5-xl loaded')
    elif args.text_encode == 'flan-t5-xxl':
        # 加载Flan-T5-XXL模型
        tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True).to(device=comp_device)
        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 4096  # 设置编码维度
        print(f'Flan-t5-xxl loaded')
    else:
        raise ValueError(f'Unknown text encoder: {args.text_encode}')

    # 加载VQ-VAE模型
    net = vqvae.HumanVQVAE(args, # 使用参数定义量化器 # use args to define different parameters in different quantizers
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

    ckpt = torch.load(args.resume_pth, map_location='cpu')["net"]  # 加载检查点
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}  # 移除模块前缀
    net.load_state_dict(ckpt, strict=True)  # 加载权重
    net.eval()
    net.to(comp_device)
    print(f'Load VQVAE model successfully! from{args.resume_pth}')

    # Transformer模型配置与加载
    args.nb_code = net.vqvae.quantizer.codebook_size  # 更新代码本大小
    config = LLaMAHFConfig.from_name(args.pretrained_llama)  # 加载LLaMA配置
    config.block_size = args.block_size
    config.vocab_size = args.nb_code + 2  # 词汇表大小（代码本大小+2）
    config.clip_dim = args.clip_dim
    config.tie_weights = args.tie_weights
    print(config)
    trans_encoder = LLaMAHF(config)  # 创建LLaMA模型
    
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    ckpt = {k.replace('module.', ''): v for k, v in ckpt['trans'].items()}  # 移除模块前缀
    trans_encoder.load_state_dict(ckpt, strict=True)  # 加载权重
    trans_encoder.eval()
    trans_encoder.to(comp_device)
    print(f'Load transformer model successfully!, from {args.resume_trans}')

    basic_root = os.path.join(args.out_dir, args.exp_name)  # 输出根目录
    os.makedirs(basic_root, exist_ok=True)
        
    # 批量推理流程
    input_text_list = open(args.infer_batch_prompt, 'r').readlines()  # 读取输入文本列表
    
    for ori_input_text in tqdm(input_text_list):  # 遍历每个输入文本
        sub_dir_list = os.listdir(basic_root)
        sub_dir_list_prefix = [-1]
        sub_dir_list = sub_dir_list_prefix + [int(s) for s in sub_dir_list if os.path.isdir(os.path.join(basic_root,s))]
        sub_dir_list.sort()
        
        # 创建新的输出目录
        output_root = os.path.join(basic_root, str(sub_dir_list[-1] + 1))
        os.makedirs(output_root, exist_ok=True)

        # 文本重写（如果启用）
        if args.use_rewrite_model:
            try:
                rewrite_text = call_llama_rewrite(pipe, ori_input_text)
                input_text_list = [rewrite_text, ori_input_text]
                print(f"Rewrite text is: {rewrite_text}")
            except Exception as e:
                print(f"Error: {e}")
                rewrite_text = ori_input_text
                input_text_list = [ori_input_text]
        else:
            input_text_list = [ori_input_text]
        
        flag = 0
        for input_text in input_text_list: # 遍历原始文本和重写文本
            flag = 1-flag
            
            clip_text = input_text.strip()
            # 文本编码 # load clip model
            if args.text_encode == 'clip':
                text = clip.tokenize(clip_text, truncate=True).to(comp_device)
                feat_clip_text = clip_model.encode_text(text).float() # (bs, 512)
                feat_clip_text = feat_clip_text.unsqueeze(1)
                y_mask = torch.ones((feat_clip_text.shape[0], feat_clip_text.shape[1])).to(comp_device)
                assert args.text_sum_way is None
            elif args.text_encode == 'flan-t5-xxl':
                tokenizer, text_encoder = clip_model
                cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
                y_mask = cap_inputs.attention_mask.to(device=comp_device)
                feat_clip_text = text_encoder(
                    input_ids=cap_inputs.input_ids.to(comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(comp_device), output_hidden_states=False
                ).last_hidden_state
            elif args.text_encode == 'flan-t5-xl':
                tokenizer, text_encoder = clip_model
                cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
                y_mask = cap_inputs.attention_mask.to(device=comp_device)
                feat_clip_text = text_encoder(
                    input_ids=cap_inputs.input_ids.to(comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(comp_device), output_hidden_states=False
                ).last_hidden_state #(bs, word_nb, 2048)
            else: 
                raise NotImplementedError

            # 截断过长的文本
            if feat_clip_text.shape[1] > 150:
                feat_clip_text = feat_clip_text[:, :150, :]
                y_mask = y_mask[:, :150]

            # 文本特征汇总方式
            if args.text_sum_way == 'cls':
                feat_clip_text = feat_clip_text[:, 0, :] # 使用CLS令牌
            elif args.text_sum_way == 'mean':
                feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1) / y_mask.sum(dim=1, keepdim=True)  # 平均池化
            elif args.text_sum_way == 'sum':
                feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1)  # 求和池化

            # 使用Transformer生成动作索引
            index_motion = trans_encoder.sample(feat_clip_text, y_mask, if_categorial=False)
            print(index_motion)

            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
            # 使用VQ-VAE解码器从索引生成动作
            pred_pose = net.forward_decoder(index_motion)

            # 加载均值和标准差进行反标准化
            mean = np.load('dataset/MotionMillion/mean_std/vector_272/mean.npy')
            std = np.load('dataset/MotionMillion/mean_std/vector_272/std.npy')

            pred_pose = inv_transform(pred_pose.detach().cpu().numpy(), mean, std)
            
             # 保存预测结果
            np.save(f'{output_root}/{flag}_predict.npy', pred_pose[0])
            with open(f'{output_root}/{flag}_text.txt', 'w') as f:
                f.write(f'{input_text}\n')
                
            print('save pose!')
            short_name = clip_text[:50].strip() + '...' if len(clip_text) > 50 else clip_text
            
            # 从局部旋转恢复全局表示并可视化
            positions_with_heading = recover_from_local_rotation(pred_pose.squeeze(), 22)
            output_path = os.path.join(output_root, f'{flag}_{short_name}.gif')
            visualize_smplx_85(positions_with_heading, title=short_name, output_path=output_path, fps=args.fps)
            
            print("Inference done!")