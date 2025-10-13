import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate

"""
文本到运动生成模型的评估数据集加载模块
用于训练和评估文本到运动的生成模型
主要功能：
1. 支持多种数据集格式(t2m, kit, motionmillion)
2. 实现运动数据的长度过滤和动态调整
3. 支持FSQ(Finite Scalar Quantization)量化
4. 提供评估时的数据加载策略
"""

# 批处理函数：将多个样本组合成一个批次，目前使用默认的collate函数，可以在这里添加自定义的批处理逻辑
def collate_fn(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)  # 可以按长度排序
    return default_collate(batch)


class MotionMillionFSQDataset(data.Dataset):
    """
    用于文本到运动生成模型评估的数据集类
    特点：
    1. 支持动态长度调整，可以根据需要调整最大长度
    2. 实现长度过滤，确保数据质量
    3. 支持FSQ量化，用于向量量化训练
    4. 提供评估和测试两种模式
    """
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, version = "version1/tokenizer_no_mirror"):
        # region 基本参数设置
        self.max_length = 20  # 初始最大长度，可以通过reset_max_len动态调整
        self.pointer = 0  # 长度指针，用于长度过滤
        self.dataset_name = dataset_name  # 数据集名称
        self.is_test = is_test  # 是否为测试模式
        self.max_text_len = max_text_len  # 最大文本长度
        self.unit_length = unit_length  # 单位长度，用于运动长度对齐
        self.w_vectorizer = w_vectorizer  # 词向量化器，用于文本处理
        self.version = version  # 数据集版本
        # endregion
        
        # 数据集配置：根据不同数据集设置相应的参数
        # region HumanML3D数据集配置
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')  # 运动数据目录
            self.text_dir = pjoin(self.data_root, 'texts')  # 文本描述目录
            self.joints_num = 22  # 关节数量
            radius = 4  # 半径参数，用于运动处理
            fps = 20  # 帧率
            self.max_motion_length = 196  # 最大运动长度
            dim_pose = 263  # 姿态维度
            kinematic_chain = paramUtil.t2m_kinematic_chain  # 运动学链
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'  # 元数据目录
            # 根据测试/验证模式选择不同的分割文件
            if is_test:
                split_file = pjoin(self.data_root, 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'val.txt')
        # endregion
        
        # region KIT-ML数据集配置
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21  # KIT-ML使用21个关节
            radius = 240 * 8  # KIT-ML的特殊半径设置
            fps = 12.5  # KIT-ML的帧率
            dim_pose = 251  # KIT-ML的姿态维度
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain  # KIT-ML的运动学链
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            # 根据测试/验证模式选择分割文件
            if is_test:
                split_file = pjoin(self.data_root, 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'val.txt')
        # endregion
            
        # region MotionMillion数据集配置
        elif dataset_name == 'motionmillion':
            # MotionMillion数据集配置（大规模数据集）
            # 注释掉的代码是使用HumanML3D作为MotionMillion的替代方案
            # self.data_root = './dataset/HumanML3D'
            # self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            # self.text_dir = pjoin(self.data_root, 'texts')
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', "vector_272")  # 使用272维向量
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            radius = 4
            fps = 60  # MotionMillion使用更高的帧率
            self.max_motion_length = 600  # 支持更长的运动序列
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = pjoin(self.data_root, 'mean_std', "vector_272")  # 按向量类型存储统计信息
            # 支持版本化的分割文件
            if is_test:
                split_file = pjoin(self.data_root, 'split', self.version, 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'split', self.version, 'val.txt')
        # endregion
        
        # region 加载预计算的统计信息，用于数据标准化
        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        # endregion
        
        # region 设置最小运动长度，用于过滤过短的运动序列
        # 不同数据集有不同的最小长度要求
        if self.dataset_name == 'motionmillion':
            min_motion_len = 120  # MotionMillion要求更长的运动序列
        elif self.dataset_name == 't2m':
            min_motion_len = 40  # HumanML3D的最小长度
        else:
            min_motion_len = 24  # KIT-ML的最小长度
        # endregion

        # region 读取分割文件，获取样本ID列表
        id_list = []
        self.id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # endregion
        
        # region 过滤运动数据：只保留长度在合理范围内的运动
        # 过滤运动数据：只保留长度在合理范围内的运动
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                # 过滤条件：长度不能太短也不能太长
                if (len(motion)) < min_motion_len or (len(motion) > self.max_motion_length):
                    continue
                self.id_list.append(name)
                length_list.append(len(motion))
                
            except Exception as e:
                print(e)  # 处理加载失败的情况
        # endregion
        
        # region 排序ID列表，确保数据的一致性
        self.id_list.sort()
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)  # 转换为numpy数组，便于后续处理
        self.reset_max_len(self.max_length)  # 初始化长度指针
        print(len(self.id_list))  # 输出有效样本数量 
        # endregion
        
    # 动态调整最大长度，用于不同的实验需求
    def reset_max_len(self, length):
        assert length <= self.max_motion_length  # 确保不超过数据集的最大长度
        # 使用二分搜索找到长度指针位置，只返回长度大于等于指定长度的样本
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    # 逆标准化：将标准化后的数据还原为原始数据
    def inv_transform(self, data):
        return data * self.std + self.mean

    # 标准化：将原始数据转换为标准化数据
    def transform(self, data):
        return (data - self.mean) / self.std

    # 返回数据集大小（考虑长度指针）
    def __len__(self):
        return len(self.id_list) - self.pointer

    # 获取单个数据样本
    def __getitem__(self, item):
        idx = self.pointer + item  # 考虑长度指针的偏移
        name = self.id_list[idx]
        motion = np.load(pjoin(self.motion_dir, name + '.npy'))  # 动态加载运动数据

        m_length = len(motion)
        # 长度处理策略：根据unit_length决定使用single还是double模式
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])  # 2/3概率选择single
        else:
            coin2 = 'single'

        if coin2 == 'double':
            # double模式：减少一个单位长度，增加数据多样性
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            # single模式：保持单位长度的整数倍
            m_length = (m_length // self.unit_length) * self.unit_length
        
        # 随机选择起始位置，实现数据增强
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        # Z-score标准化：消除量纲差异，提高训练稳定性
        motion = (motion - self.mean) / self.std

        # 零填充：将短序列填充到统一长度
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        motion = motion.astype(np.float32)  # 转换为float32，节省内存

        print("评估数据处理后 motion.shape: ", motion.shape)
        return motion, m_length, name



def MotionMillionFSQDATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4, version = "version1/tokenizer_no_mirror") : 
    """
    创建MotionMillionFSQ数据集的数据加载器
    用于文本到运动生成模型的评估
    特点：
    1. 支持动态长度调整
    2. 实现长度过滤
    3. 支持FSQ量化
    """
    val_dataset = MotionMillionFSQDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length, version=version)
    val_loader = torch.utils.data.DataLoader( val_dataset, 
                                              batch_size,
                                              shuffle = True,  # 随机打乱数据
                                              num_workers=num_workers,  # 多进程并行加载
                                              collate_fn=collate_fn,  # 使用自定义的批处理函数
                                            #   drop_last = True,  # 注释掉的选项
                                            #   prefetch_factor=2)  # 预取因子
                                              drop_last = False,  # 不丢弃最后一个不完整的batch
                                              pin_memory=True)  # 将数据固定在内存中，加速GPU传输
    return val_loader, val_dataset.mean, val_dataset.std

def cycle(iterable):
    """
    无限循环迭代器
    用于训练时无限循环遍历数据集
    当数据集遍历完毕后自动重新开始
    """
    while True:
        for x in iterable:
            yield x
