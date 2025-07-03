import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, split="val"):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            if is_test:
                split_file = pjoin(self.data_root, 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'val.txt')
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            if is_test:
                split_file = pjoin(self.data_root, 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'val.txt')
        elif dataset_name == 'motionmillion':
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', "vector_272")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            radius = 4
            fps = 30
            self.max_motion_length = 300
            dim_pose = 272
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = pjoin(self.data_root, 'mean_std', "vector_272")
            split_file = pjoin(self.data_root, 'split', 'version1', 't2m_60_300', split+".txt")

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        min_motion_len = 60

        data_dict = {}
        id_list = []
        self.id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) > 200):
                    continue
                self.id_list.append(name)
                length_list.append(len(motion))
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip() # .split('#')
                        caption = line_split # [0]
                        
                        text_dict['caption'] = caption
                        flag = True
                        text_data.append(text_dict)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                print(e)
        
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        print(len(self.id_list)) # 
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        # return len(self.data_dict) - self.pointer
        return len(self.id_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']

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

        return "_", "_", caption, 0, motion, m_length, '_', name



def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4, split="val") : 
    
    val_dataset = Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length, split=split)
    val_loader = torch.utils.data.DataLoader( val_dataset, 
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return val_loader, val_dataset.mean, val_dataset.std


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
