import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8, motion_type=None, text_type=None, version=None):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version
        self.dataset_name = dataset_name
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'all.txt')
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'all.txt')
        elif dataset_name == 'motionmillion':
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22
            radius = 4
            fps = 30
            self.max_motion_length = 300
            dim_pose = 272
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            split_file = pjoin(self.data_root, 'split', self.version, 'all.txt')
        else:
            raise KeyError('Dataset Does not Exists')
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        for name in tqdm(id_list):
            new_name_list.append(name)

        self.mean = mean
        self.std = std
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        motion = np.load(pjoin(self.motion_dir, name + '.npy'))
        m_length = len(motion)

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, unit_length = 4, motion_type=None, text_type=None, version=None) : 
    
    dataset = VQMotionDataset(dataset_name, unit_length=unit_length, motion_type=motion_type, text_type=text_type, version=version)
    train_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader, dataset.mean, dataset.std

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
