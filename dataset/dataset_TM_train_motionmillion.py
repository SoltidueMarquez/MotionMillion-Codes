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
        
        self.pointer = 0
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version

        self.unit_length = unit_length
        self.mot_end_idx = codebook_size # 512
        self.mot_pad_idx = codebook_size + 1 # 513
        
        self.tokenizer_name = tokenizer_name
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            # self.max_motion_length = 26 if unit_length == 8 else 51
            self.max_motion_length = 201
            self.max_text_length = 150
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            split_file = pjoin(self.data_root, f'{split}.txt')
        elif dataset_name == 'kit':
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
            self.data_root = './dataset/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 301
            self.max_text_length = 150
            dim_pose = 272
            kinematic_chain = paramUtil.t2m_kinematic_chain
            split_file = pjoin(self.data_root, 'split', self.version, f'{split}.txt')
        
        with open(os.path.join(self.data_root, "all_data.pkl"), "rb") as f:
            all_data = pickle.load(f)        
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # if debug:
        #     id_list = id_list[:1000]

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            code_data_ref = all_data["code_data"][name]
            text_data_ref = all_data["text_data"][name]
            
            if len(text_data_ref) == 0:
                continue
                
            text_data = []
            for line in text_data_ref:
                text_dict = {}
                caption = line.strip()
                if caption == '':
                    continue
                    
                text_dict['caption'] = caption
                text_data.append(text_dict)
            
            data_dict[name] = {
                'm_token_list': code_data_ref,
                'text': text_data
            }
            new_name_list.append(name)

        self.data_dict = data_dict
        self.name_list = new_name_list
        print(len(self.data_dict))
    
        self.text_encode = text_encode
        self.text_sum_way = text_sum_way
        self.comp_device = comp_device
        self.clip_model = clip_model

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        
        m_tokens = random.choice(m_token_list)
        m_tokens = torch.tensor(m_tokens).to(self.comp_device)
        text_data = random.choice(text_list)
        caption= text_data['caption']

        if self.text_encode == 'clip':
            text = clip.tokenize(caption, truncate=True).to(self.comp_device)
            feat_clip_text = self.clip_model.encode_text(text).float()
            feat_clip_text = feat_clip_text.unsqueeze(1)
            y_mask = torch.ones((feat_clip_text.shape[0], feat_clip_text.shape[1])).to(self.comp_device)
        elif self.text_encode in ['flan-t5-xxl', 'flan-t5-xl']:
            cap_inputs = self.clip_model[0](caption, padding=True, truncation=True, return_tensors="pt")
            y_mask = cap_inputs.attention_mask.to(device=self.comp_device) # 1,9
            
            # 检查T5编码器是否在CPU上
            if next(self.clip_model[1].parameters()).device.type == 'cpu':
                # T5编码器在CPU上，需要临时移动到GPU进行推理
                with torch.no_grad():
                    feat_clip_text = self.clip_model[1](
                        input_ids=cap_inputs.input_ids.to(self.comp_device), 
                        attention_mask=cap_inputs.attention_mask.to(self.comp_device), 
                        output_hidden_states=False
                    ).last_hidden_state
            else:
                # T5编码器在GPU上，正常处理
                feat_clip_text = self.clip_model[1](
                    input_ids=cap_inputs.input_ids.to(self.comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(self.comp_device), 
                    output_hidden_states=False
                ).last_hidden_state
        else:
            raise ValueError(f'Unknown text encoder: {self.text_encode}')
        
        feat_clip_text = feat_clip_text.to(dtype=torch.bfloat16)
        
        if self.text_sum_way == 'cls':
            feat_clip_text = feat_clip_text[:, 0, :]
            feat_clip_text = feat_clip_text.unsqueeze(1)
        elif self.text_sum_way == 'mean':
            feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1) / y_mask.sum(dim=1, keepdim=True)
            feat_clip_text = feat_clip_text.unsqueeze(1)
        elif self.text_sum_way == 'sum':
            feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1)
            feat_clip_text = feat_clip_text.unsqueeze(1)

        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        text_tokens_len = feat_clip_text.shape[1]
        if text_tokens_len > self.max_text_length:
            feat_clip_text = feat_clip_text[:, :self.max_text_length, :]
            text_tokens_len = self.max_text_length
            y_mask = y_mask[:, :self.max_text_length]
        padding_length = self.max_motion_length - text_tokens_len

        
        if m_tokens_len+1 < padding_length:
            m_tokens = torch.cat([torch.ones((text_tokens_len), dtype=torch.int32).to(self.comp_device) * self.mot_pad_idx, m_tokens, torch.ones((1), dtype=torch.int32).to(self.comp_device) * self.mot_end_idx, torch.ones((padding_length-1-m_tokens_len), dtype=torch.int32).to(self.comp_device) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = torch.cat([torch.ones((text_tokens_len), dtype=torch.int32).to(self.comp_device) * self.mot_pad_idx, m_tokens, torch.ones((1), dtype=torch.int32).to(self.comp_device) * self.mot_end_idx], axis=0)
        

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

