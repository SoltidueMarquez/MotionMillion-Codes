import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
import clip
import os
import pickle

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


# write a collate function which can pad the feat_clip_text
def collate_fn(batch):
    result_caption = []
    result_m_tokens = []
    result_m_tokens_len = []
    result_feat_clip_text = []
    result_y_mask = []
    result_text_tokens_len = []
    for item in batch:
        result_caption.append(item[0])
        result_m_tokens.append(item[1])
        result_m_tokens_len.append(item[2])
        result_feat_clip_text.append(item[3])
        result_y_mask.append(item[4])
        result_text_tokens_len.append(item[5])

    return result_caption, torch.stack(result_m_tokens),torch.stack(result_m_tokens_len), collate_tensors(result_feat_clip_text), collate_tensors(result_y_mask), torch.stack(result_text_tokens_len)


class Text2MotionDataset_motionmillion(data.Dataset):
    def __init__(self, dataset_name, split, clip_model, text_encode, text_sum_way, comp_device, motion_type, text_type, version, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, debug=False):
        """
        文本到动作数据集加载器 - MotionMillion版本
        
        参数:
            dataset_name: 数据集名称 ('motionmillion', 't2m', 'kit')
            split: 数据划分 ('train', 'val', 'test')
            clip_model: 文本编码器模型
            text_encode: 文本编码方式 ('clip', 'flan-t5-xl', 'flan-t5-xxl')
            text_sum_way: 文本特征汇总方式 ('cls', 'mean', 'sum')
            comp_device: 计算设备 (CPU/GPU)
            motion_type: 动作数据类型 (如'vector_263')
            text_type: 文本数据类型 (如'texts')
            version: 数据版本 (如'version1')
            feat_bias: 特征偏置，默认为5
            unit_length: 单位长度，默认为4
            codebook_size: VQ码本大小，默认为1024
            tokenizer_name: tokenizer名称，默认为None
            debug: 调试模式，默认为False
        """
        
        self.pointer = 0  # 数据指针，用于顺序访问
        self.dataset_name = dataset_name  # 数据集名称
        self.motion_type = motion_type    # 动作数据类型
        self.text_type = text_type        # 文本数据类型
        self.version = version            # 数据版本

        self.unit_length = unit_length    # 单位长度（用于下采样计算）
        self.mot_end_idx = codebook_size  # 动作结束token索引 = codebook_size (512)
        self.mot_pad_idx = codebook_size + 1  # 动作填充token索引 = codebook_size + 1 (513)
        
        self.tokenizer_name = tokenizer_name  # tokenizer名称
        
        # ==================== 数据集配置 ====================
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')  # 动作数据目录
            self.text_dir = pjoin(self.data_root, 'texts')             # 文本数据目录
            self.joints_num = 22        # 关节点数量
            radius = 4                  # 半径参数
            fps = 20                    # 帧率
            self.max_motion_length = 201  # 最大动作序列长度
            self.max_text_length = 150    # 最大文本长度
            dim_pose = 263              # 姿态维度
            kinematic_chain = paramUtil.t2m_kinematic_chain  # 运动学链
            split_file = pjoin(self.data_root, f'{split}.txt')  # 划分文件
            
        elif dataset_name == 'kit':
            # KIT-ML数据集配置
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain
            split_file = pjoin(self.data_root, f'{split}.txt')
            
        elif dataset_name == 'motionmillion':
            # MotionMillion数据集配置（主要使用这个）
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)  # 动作数据子目录
            self.text_dir = pjoin(self.data_root, self.text_type)                     # 文本数据子目录
            self.joints_num = 22        # SMPL模型的22个关节点
            radius = 4                  # 用于归一化的半径
            fps = 20                    # 20帧/秒
            self.max_motion_length = 301  # 最大动作序列长度（301个token）
            self.max_text_length = 150    # 最大文本长度
            dim_pose = 272              # 原始动作数据维度（272维向量）
            kinematic_chain = paramUtil.t2m_kinematic_chain  # HumanML3D的运动学链
            split_file = pjoin(self.data_root, 'split', self.version, f'{split}.txt')  # 版本特定的划分文件
        
        # ==================== 加载数据索引 ====================
        # 从all_data.pkl加载所有预处理数据
        with open(os.path.join(self.data_root, "all_data.pkl"), "rb") as f:
            all_data = pickle.load(f)        
        id_list = []
        # 读取划分文件，获取当前split的所有样本ID
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # 调试模式：只使用前1000个样本
        # if debug:
        #     id_list = id_list[:1000]

        # ==================== 构建数据字典 ====================
        new_name_list = []    # 有效的样本名称列表
        data_dict = {}        # 数据字典：{样本名: {动作token列表, 文本列表}}
        
        # 遍历所有样本ID，构建数据字典
        for name in tqdm(id_list):
            code_data_ref = all_data["code_data"][name]   # 该样本的所有动作token序列
            text_data_ref = all_data["text_data"][name]   # 该样本的所有文本描述
            
            # 跳过没有文本描述的样本
            if len(text_data_ref) == 0:
                continue
                
            # 处理文本数据：清理空文本并构建文本字典列表
            text_data = []
            for line in text_data_ref:
                text_dict = {}
                caption = line.strip()  # 清理文本
                if caption == '':       # 跳过空文本
                    continue
                    
                text_dict['caption'] = caption
                text_data.append(text_dict)
            
            # 存储该样本的数据：动作token列表和文本列表
            data_dict[name] = {
                'm_token_list': code_data_ref,  # 动作token序列列表
                'text': text_data               # 文本描述列表
            }
            new_name_list.append(name)  # 添加到有效样本列表

        # 保存数据字典和样本名称列表
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(len(self.data_dict))  # 打印有效样本数量
    
        # ==================== 保存配置参数 ====================
        self.text_encode = text_encode    # 文本编码方式
        self.text_sum_way = text_sum_way  # 文本特征汇总方式
        self.comp_device = comp_device    # 计算设备
        self.clip_model = clip_model      # 文本编码器模型

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        """
        获取单个训练样本
        
        参数:
            item: 样本索引
            
        返回:
            caption: 原始文本描述 (字符串)
            m_tokens: 填充后的动作token序列 [max_length]
            m_tokens_len: 原始动作序列长度 (标量)
            feat_clip_text: 文本特征 [1, clip_dim] 或 [seq_len, clip_dim]
            y_mask: 文本mask [seq_len]
            text_tokens_len: 文本token长度 [1]
        """
        
        # 1. 获取样本数据
        data = self.data_dict[self.name_list[item]]  # 根据索引获取数据
        m_token_list, text_list = data['m_token_list'], data['text']  # 动作token列表和文本列表
        
        # 2. 随机选择动作序列和文本描述（数据增强）
        m_tokens = random.choice(m_token_list)  # 从多个动作序列中随机选择一个
        m_tokens = torch.tensor(m_tokens).to(self.comp_device)  # 转换为tensor并移到设备
        text_data = random.choice(text_list)  # 从多个文本描述中随机选择一个
        caption= text_data['caption']  # 获取文本内容

        # 3. 文本编码处理
        if self.text_encode == 'clip':
            # CLIP编码器处理
            text = clip.tokenize(caption, truncate=True).to(self.comp_device)  # tokenize文本
            feat_clip_text = self.clip_model.encode_text(text).float()  # 编码为特征向量
            feat_clip_text = feat_clip_text.unsqueeze(1)  # 增加序列维度 [batch, 1, dim]
            y_mask = torch.ones((feat_clip_text.shape[0], feat_clip_text.shape[1])).to(self.comp_device)  # 全1mask
            
        elif self.text_encode in ['flan-t5-xxl', 'flan-t5-xl']:
            # T5编码器处理
            cap_inputs = self.clip_model[0](caption, padding=True, truncation=True, return_tensors="pt")  # tokenize
            print("cap_inputs.shape:", cap_inputs.shape)

            y_mask = cap_inputs.attention_mask.to(device=self.comp_device) # 1,9  # 注意力mask [batch, seq_len]
            
            # 检查T5编码器是否在CPU上
            if next(self.clip_model[1].parameters()).device.type == 'cpu':
                # T5编码器在CPU上，需要临时移动到GPU进行推理
                with torch.no_grad():
                    feat_clip_text = self.clip_model[1](
                        input_ids=cap_inputs.input_ids.to(self.comp_device), 
                        attention_mask=cap_inputs.attention_mask.to(self.comp_device), 
                        output_hidden_states=False
                    ).last_hidden_state  # 获取最后一层隐藏状态
            else:
                # T5编码器在GPU上，正常处理
                feat_clip_text = self.clip_model[1](
                    input_ids=cap_inputs.input_ids.to(self.comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(self.comp_device), 
                    output_hidden_states=False
                ).last_hidden_state
        else:
            raise ValueError(f'Unknown text encoder: {self.text_encode}')
        
        # 4. 统一特征数据类型为bfloat16（节省内存）
        feat_clip_text = feat_clip_text.to(dtype=torch.bfloat16)
        
        # 5. 文本特征汇总策略
        if self.text_sum_way == 'cls':
            # 取CLS token作为整个文本的代表
            feat_clip_text = feat_clip_text[:, 0, :]  # 取第一个token [batch, dim]
            feat_clip_text = feat_clip_text.unsqueeze(1)  # 恢复序列维度 [batch, 1, dim]
            
        elif self.text_sum_way == 'mean':
            # 均值池化：对所有token特征求平均
            feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1) / y_mask.sum(dim=1, keepdim=True)
            feat_clip_text = feat_clip_text.unsqueeze(1)
            
        elif self.text_sum_way == 'sum':
            # 求和池化：对所有token特征求和
            feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1)
            feat_clip_text = feat_clip_text.unsqueeze(1)

        # 6. 动作序列数据增强：随机丢弃头尾token（1/3概率）
        coin = np.random.choice([False, False, True])  # 2/3概率不丢弃，1/3概率丢弃
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])  # 随机选择丢弃头部或尾部
            if coin2:
                m_tokens = m_tokens[:-1]  # 丢弃尾部token
            else:
                m_tokens = m_tokens[1:]   # 丢弃头部token
        m_tokens_len = m_tokens.shape[0]  # 记录原始动作序列长度

        # 7. 处理文本序列长度限制
        text_tokens_len = feat_clip_text.shape[1]  # 当前文本token长度
        if text_tokens_len > self.max_text_length:
            # 如果文本过长，进行截断
            feat_clip_text = feat_clip_text[:, :self.max_text_length, :]
            text_tokens_len = self.max_text_length
            y_mask = y_mask[:, :self.max_text_length]
        
        # 8. 计算需要填充的长度
        padding_length = self.max_motion_length - text_tokens_len  # 总填充长度

        # 9. 动作序列填充策略
        # 格式: [文本长度个PAD] + [动作token] + [1个END] + [剩余PAD]
        if m_tokens_len+1 < padding_length:
            # 如果动作序列较短，需要额外填充
            m_tokens = torch.cat([
                torch.ones((text_tokens_len), dtype=torch.int32).to(self.comp_device) * self.mot_pad_idx,  # 文本部分PAD
                m_tokens,  # 原始动作序列
                torch.ones((1), dtype=torch.int32).to(self.comp_device) * self.mot_end_idx,  # 结束token
                torch.ones((padding_length-1-m_tokens_len), dtype=torch.int32).to(self.comp_device) * self.mot_pad_idx  # 剩余PAD
            ], axis=0)
        else:
            # 如果动作序列较长，只添加结束token
            m_tokens = torch.cat([
                torch.ones((text_tokens_len), dtype=torch.int32).to(self.comp_device) * self.mot_pad_idx,  # 文本部分PAD
                m_tokens,  # 原始动作序列
                torch.ones((1), dtype=torch.int32).to(self.comp_device) * self.mot_end_idx  # 结束token
            ], axis=0)
        
        # 10. 返回处理后的数据
        return caption, m_tokens.reshape(-1), torch.tensor(m_tokens_len).to(self.comp_device), feat_clip_text.squeeze(), y_mask.squeeze(), torch.tensor(text_tokens_len).unsqueeze(0).to(self.comp_device)


def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, split, clip_model, text_encode, text_sum_way, comp_device, motion_type=None, text_type=None, version=None, unit_length=4,
                num_workers = 0, debug=False) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset_motionmillion(dataset_name, clip_model = clip_model, text_encode = text_encode, text_sum_way = text_sum_way, comp_device = comp_device, split = split, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, debug=debug, motion_type=motion_type, text_type=text_type, version=version),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

