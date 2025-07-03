import os 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval_motionmillion

from models.evaluator_wrapper_motionmillion_rpr272 import EvaluatorModelWrapper272RPR

import warnings

from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig
from transformers import T5EncoderModel, T5Tokenizer
from utils.quaternion import *

warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader, val_mean, val_std = dataset_TM_eval_motionmillion.DATALoader(args.dataname, True, 32, w_vectorizer, split=args.split)

comp_device = torch.device('cuda')

eval_wrapper = EvaluatorModelWrapper272RPR(args, comp_device)

##### ---- Network ---- #####

if args.text_encode == 'clip':
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
elif args.text_encode == 'flan-t5-xl':
    tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True).to(device=comp_device)
    clip_model = (tokenizer, text_encoder)
    clip_model[1].eval()
    for p in clip_model[1].parameters():
        p.requires_grad = False
    args.clip_dim = 2048
    logger.info(f'Flan-t5-xl loaded')
elif args.text_encode == 'flan-t5-xxl':
    tokenizer = T5Tokenizer.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained('checkpoints/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True).to(device=comp_device)
    clip_model = (tokenizer, text_encoder)
    clip_model[1].eval()
    for p in clip_model[1].parameters():
        p.requires_grad = False
    args.clip_dim = 4096
    logger.info(f'Flan-t5-xxl loaded')
else:
    raise ValueError(f'Unknown text encoder: {args.text_encode}')

ori_nb_code = args.nb_code
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


# trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
#                                 embed_dim=args.embed_dim_gpt, 
#                                 clip_dim=args.clip_dim, 
#                                 block_size=args.block_size, 
#                                 num_layers=args.num_layers, 
#                                 n_head=args.n_head_gpt, 
#                                 drop_out_rate=args.drop_out_rate, 
#                                 fc_rate=args.ff_rate)

ckpt = torch.load(args.resume_pth, map_location='cpu')['net']
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
net.load_state_dict(ckpt, strict=True)
net.eval()
net.to(comp_device)
print('Load VQVAE model successfully!')

args.nb_code = net.vqvae.quantizer.codebook_size
config = LLaMAHFConfig.from_name(args.pretrained_llama)
config.block_size = args.block_size
config.vocab_size = args.nb_code + 2
config.clip_dim = args.clip_dim

config.tie_weights = args.tie_weights
print(config)
trans_encoder = LLaMAHF(config) 

ckpt = torch.load(args.resume_trans, map_location='cpu')

# filter name
ckpt = {k.replace('module.', ''): v for k, v in ckpt['trans'].items()}
trans_encoder.load_state_dict(ckpt, strict=True)
trans_encoder.eval()
trans_encoder.to(comp_device)
print('Load transformer model successfully!')


fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = 1

        
for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer_motionmillion(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper, comp_device=comp_device, text_encode=args.text_encode, text_sum_way=args.text_sum_way, accelerator=None, draw=False, save=False, savegif=False)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)

top1 = [t.item() for t in top1]
top2 = [t.item() for t in top2]
top3 = [t.item() for t in top3]
matching = [m.item() for m in matching]

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)