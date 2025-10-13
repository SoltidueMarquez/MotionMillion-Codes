import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

"""
向量量化(VQ)运动数据集加载模块
用于训练和评估运动数据的向量量化模型
主要功能：
1. 支持多种数据集格式(t2m, kit, motionmillion)
2. 提供训练和评估两种不同的数据加载策略
3. 实现运动数据的标准化和窗口化处理
4. 支持批量数据加载和多进程处理
"""

class VQMotionDatasetEval(data.Dataset):
    """
    用于评估的向量量化运动数据集类
    与训练数据集的主要区别：
    1. 预加载所有运动数据到内存中，提高评估时的访问速度
    2. 支持更灵活的长度处理策略（single/double模式）
    3. 返回运动长度信息，便于后续分析
    """
    def __init__(self, dataset_name,  motion_type, text_type, version, split, debug, window_size = 64, unit_length = 4):
        # 基本参数设置
        self.window_size = window_size  # 滑动窗口大小，用于截取固定长度的运动片段
        self.unit_length = unit_length  # 单位长度，用于运动长度的对齐处理
        self.dataset_name = dataset_name  # 数据集名称
        self.motion_type = motion_type  # 运动类型（如joint, smpl等）
        self.text_type = text_type  # 文本类型
        self.version = version  # 数据集版本

        # 根据数据集类型配置相应的路径和参数
        # region t2m
        if dataset_name == 't2m':
            # HumanML3D数据集配置
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')  # 运动数据目录
            self.text_dir = pjoin(self.data_root, 'texts')  # 文本描述目录
            self.joints_num = 22  # 关节数量，HumanML3D使用22个关节
            self.max_motion_length = 196  # 最大运动长度，用于零填充
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))  # 加载预计算的均值
            std = np.load(pjoin(self.meta_dir, 'std.npy'))  # 加载预计算的标准差
            split_file = pjoin(self.data_root, 'train.txt')  # 训练集分割文件
        # endregion
        
        # region kit
        elif dataset_name == 'kit':
            # KIT-ML数据集配置
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21  # KIT-ML使用21个关节，比HumanML3D少一个
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'train.txt')
        # endregion
        
        # region motionmillion
        elif dataset_name == 'motionmillion':
            # MotionMillion数据集配置（大规模数据集）
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)  # 支持多种运动类型
            self.text_dir = pjoin(self.data_root, self.text_type)  # 支持多种文本类型
            self.joints_num = 22
            self.max_motion_length = 300  # MotionMillion支持更长的运动序列，用于统一运动序列的长度
            # MotionMillion的统计信息按运动类型分别存储
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            # 支持版本化的分割文件
            split_file = pjoin(self.data_root, 'split', self.version, split + '.txt')
        # endregion
        else:
            raise KeyError('Dataset Does not Exists')
        
        # 初始化数据存储列表
        joints_num = self.joints_num
        id_list = []
        
        self.data = []  # 存储所有运动数据（评估时预加载到内存）
        self.lengths = []  # 存储每个运动的长度信息
        self.id_list = []  # 存储有效的运动ID列表
        
        # 读取分割文件，获取所有运动样本的ID
        # 分割文件包含训练/验证/测试集的样本ID列表
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # 调试模式：只使用前1000个样本，加快调试速度
        if debug:
            id_list = id_list[:1000]
            
        # 预加载所有运动数据到内存中
        # 这样做的好处：评估时访问速度快，但会占用更多内存
        for name in tqdm(id_list):
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            # 过滤掉长度小于窗口大小的运动，确保可以截取有效片段
            if motion.shape[0] < self.window_size:
                continue
            self.id_list.append(name)
            # 记录可用长度（总长度减去窗口大小）
            self.lengths.append(motion.shape[0] - self.window_size)
            self.data.append(motion)
        
        # 保存标准化参数
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.id_list)))

    # 逆标准化函数：将标准化后的数据还原为原始数据，用于模型输出后还原到原始运动数据空间
    def inv_transform(self, data):
        return data * self.std + self.mean
    
    # 标准化函数：将原始数据转换为标准化数据，用于模型输入前转换到标准化空间，使用Z-score标准化：(x - mean) / std，消除不同关节和维度间的量纲差异，提高训练稳定性
    def transform(self, data):
        return (data - self.mean) / self.std
    
    # 返回数据集大小
    def __len__(self):
        return len(self.id_list)

    # 获取单个数据样本
    def __getitem__(self, item):
        name = self.id_list[item]
        motion = self.data[item]  # 从预加载的数据中获取
        
        m_length = len(motion)
        # 长度处理策略：根据unit_length决定使用single还是double模式
        # single模式：保持原始长度
        # double模式：减少一个unit_length，用于数据增强
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
        # 目的：支持批量处理，所有样本具有相同的维度
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        return motion, m_length, name  # 返回运动数据、长度和名称
    

class VQMotionDataset(data.Dataset):
    """
    用于训练的向量量化运动数据集类
    与评估数据集的主要区别：
    1. 不预加载数据，每次动态加载（节省内存）
    2. 使用固定窗口大小截取运动片段
    3. 只返回运动数据，不返回长度信息
    4. 适合大规模训练，内存占用更少
    """
    def __init__(self, dataset_name,  motion_type, text_type, version, split, debug, window_size = 64, unit_length = 4):
        # 基本参数设置（与评估版本相同）
        self.window_size = window_size  # 固定窗口大小，用于截取运动片段
        self.unit_length = unit_length  # 单位长度
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version

        # 数据集配置（与评估版本相同的配置逻辑）
        # region t2m
        if dataset_name == 't2m':
            # HumanML3D数据集配置
            self.data_root = './dataset/HumanML3D'  # 数据集根目录
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')  # 运动数据目录，存储关节向量
            self.text_dir = pjoin(self.data_root, 'texts')  # 文本描述目录，存储动作描述
            self.joints_num = 22  # 关节数量，HumanML3D使用22个关节点
            # 元数据目录：存储预计算的统计信息（均值、标准差）
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))  # 加载预计算的均值，用于标准化
            std = np.load(pjoin(self.meta_dir, 'std.npy'))  # 加载预计算的标准差，用于标准化
            split_file = pjoin(self.data_root, 'train.txt')  # 训练集分割文件，包含样本ID列表
        # endregion
        # region kit
        elif dataset_name == 'kit':
            # KIT-ML数据集配置
            self.data_root = './dataset/KIT-ML'  # KIT-ML数据集根目录
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')  # 运动数据目录
            self.text_dir = pjoin(self.data_root, 'texts')  # 文本描述目录
            self.joints_num = 21  # KIT-ML使用21个关节，比HumanML3D少一个关节点
            # KIT-ML的元数据目录
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))  # 加载KIT-ML的统计信息
            std = np.load(pjoin(self.meta_dir, 'std.npy'))  # 加载KIT-ML的统计信息
            split_file = pjoin(self.data_root, 'train.txt')  # KIT-ML的训练集分割文件
        # endregion
        # region motionmillion
        elif dataset_name == 'motionmillion':
            # MotionMillion数据集配置（大规模数据集）
            self.data_root = './dataset/MotionMillion'  # MotionMillion数据集根目录
            # 支持多种运动类型：joint, smpl, smplx等
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            # 支持多种文本类型：clip, t5等
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22  # MotionMillion使用22个关节点
            # MotionMillion的统计信息按运动类型分别存储，确保不同运动类型的标准化正确
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            # 支持版本化的分割文件，便于数据集版本管理
            split_file = pjoin(self.data_root, 'split', self.version, split + '.txt')
        # endregion
        else:
            raise KeyError('Dataset Does not Exists')
        
        # 训练版本：只存储ID列表，不预加载数据（节省内存）
        id_list = []
        self.id_list = []
        
        # 读取分割文件
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # 调试模式
        if debug:
            id_list = id_list[:1000]
            
        # 只存储ID，不加载实际数据（与评估版本的主要区别）
        for name in tqdm(id_list):
            self.id_list.append(name)
        
        # 保存标准化参数
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.id_list)))

    # 逆标准化函数：还原到原始数据空间
    def inv_transform(self, data):
        return data * self.std + self.mean
    
    # 标准化函数：转换到标准化空间
    def transform(self, data):
        return (data - self.mean) / self.std
    
    # 返回数据集大小
    def __len__(self):
        return len(self.id_list)

    # 获取单个数据样本
    def __getitem__(self, item):
        name = self.id_list[item]
        # 动态加载运动数据（与评估版本的主要区别）
        motion = np.load(pjoin(self.motion_dir, name + '.npy'))
        print("训练数据 motion.shape: ", motion.shape)
        
        # 随机选择起始位置，截取固定长度的运动片段
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx+self.window_size]
        
        # Z-score标准化
        motion = (motion - self.mean) / self.std
        motion = motion.astype(np.float32)  # 转换为float32，节省内存
        
        return motion  # 只返回运动数据

    
    
def DATALoader(dataset_name,
               batch_size,
               motion_type,
                text_type,
                version, 
                split, 
                debug,
               num_workers = 64, #8,
               window_size = 64,
               unit_length = 4):
    """
    创建训练用的数据加载器
    使用VQMotionDataset（训练版本），特点：
    1. 动态加载数据，节省内存
    2. 适合大规模训练
    3. 支持多进程并行加载
    """
    print("num_workers: ", num_workers)
    trainSet = VQMotionDataset(dataset_name, motion_type, text_type, version, split, debug, window_size=window_size, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size=batch_size,
                                              shuffle=True,  # 随机打乱数据，提高训练效果
                                              #sampler=sampler,
                                              num_workers=num_workers,  # 多进程并行加载，提高效率
                                              #collate_fn=collate_fn,
                                              drop_last = True,  # 丢弃最后一个不完整的batch
                                              pin_memory=True) #,  # 将数据固定在内存中，加速GPU传输
                                            #   prefetch_factor=4)
    
    return train_loader, trainSet.mean, trainSet.std


def DATALoaderEvalVQ(dataset_name,
               batch_size,
               motion_type,
                text_type,
                version, 
                split, 
                debug,
               num_workers = 64, #8,
               window_size = 64,
               unit_length = 4):
    """
    创建评估用的数据加载器
    使用VQMotionDatasetEval（评估版本），特点：
    1. 预加载数据到内存，访问速度快
    2. 支持灵活的长度处理
    3. 返回运动长度信息，便于分析
    """
    print("num_workers: ", num_workers)
    trainSet = VQMotionDatasetEval(dataset_name, motion_type, text_type, version, split, debug, window_size=window_size, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size=batch_size,
                                              shuffle=True,  # 评估时也可以打乱数据
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              pin_memory=True) #,
                                            #   prefetch_factor=4)
    
    return train_loader, trainSet.mean, trainSet.std


def cycle(iterable):
    """
    无限循环迭代器
    用于训练时无限循环遍历数据集
    当数据集遍历完毕后自动重新开始
    """
    while True:
        for x in iterable:
            yield x
