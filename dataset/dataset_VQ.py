import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class VQMotionDatasetEval(data.Dataset):
    def __init__(self, dataset_name,  motion_type, text_type, version, split, debug, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'train.txt')

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'train.txt')

        elif dataset_name == 'motionmillion':
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22
            self.max_motion_length = 300
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            split_file = pjoin(self.data_root, 'split', self.version, split + '.txt')
            
        else:
            raise KeyError('Dataset Does not Exists')
        
        joints_num = self.joints_num
        id_list = []
        
        self.data = []
        self.lengths = []
        self.id_list = []
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if debug:
            id_list = id_list[:1000]
            
        for name in tqdm(id_list):
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            if motion.shape[0] < self.window_size:
                continue
            self.id_list.append(name)
            self.lengths.append(motion.shape[0] - self.window_size)
            self.data.append(motion)
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.id_list)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        name = self.id_list[item]
        motion = self.data[item]
        
        m_length = len(motion)
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        return motion, m_length, name
    
class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name,  motion_type, text_type, version, split, debug, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'train.txt')
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
            split_file = pjoin(self.data_root, 'train.txt')  
        elif dataset_name == 'motionmillion':
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            split_file = pjoin(self.data_root, 'split', self.version, split + '.txt')
        else:
            raise KeyError('Dataset Does not Exists')
        
        id_list = []
        
        self.id_list = []
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if debug:
            id_list = id_list[:1000]
            
        for name in tqdm(id_list):
            self.id_list.append(name)
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.id_list)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        name = self.id_list[item]
        motion = np.load(pjoin(self.motion_dir, name + '.npy'))
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        motion = motion.astype(np.float32)
        return motion

    
    
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
    print("num_workers: ", num_workers)
    trainSet = VQMotionDataset(dataset_name, motion_type, text_type, version, split, debug, window_size=window_size, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              pin_memory=True) #,
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
    print("num_workers: ", num_workers)
    trainSet = VQMotionDatasetEval(dataset_name, motion_type, text_type, version, split, debug, window_size=window_size, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              pin_memory=True) #,
                                            #   prefetch_factor=4)
    
    return train_loader, trainSet.mean, trainSet.std


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
