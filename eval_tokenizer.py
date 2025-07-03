import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ
import utils.eval_trans as eval_trans
import warnings
warnings.filterwarnings('ignore')
import numpy as np
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' :
    args.nb_joints = 21
elif args.dataname == 't2m':
    args.nb_joints = 22
elif args.dataname == 'motionmillion':
    args.nb_joints = 22

train_loader, train_mean, train_std = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        args.motion_type, 
                                        args.text_type,
                                        args.version, 
                                        'train',
                                        True,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

##### ---- Dataloader ---- #####

val_loader, test_mean, test_std = dataset_VQ.DATALoaderEvalVQ(args.dataname,
                                        args.batch_size,
                                        args.motion_type, 
                                        args.text_type,
                                        args.version, 
                                        'val_debug',
                                        args.debug,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        num_workers=args.num_workers)

##### ---- Network ---- #####
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

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu') # ["net"]
    if 'net' in ckpt:
        ckpt = ckpt['net']
    else:
        ckpt = ckpt["trans"]
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
net.train()
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
mpjpe = []
if args.cal_acceleration:
    pred_mean_acceleration_seq = []
    pred_max_acceleration_seq = []
    gt_mean_acceleration_seq = []
    gt_max_acceleration_seq = []
repeat_time = 1
for i in range(repeat_time):
    if not args.cal_acceleration:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion_1gpu(args.out_dir, train_loader, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_mpjpe=10000, draw=args.draw, save=args.save, savegif=args.savegif, savenpy=args.savenpy, comp_device=torch.device('cuda'), fps=args.fps, cal_acceleration=args.cal_acceleration)
    else:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, writer, logger, best_pred_mean_acceleration_seq, best_pred_max_acceleration_seq, best_gt_mean_acceleration_seq, best_gt_max_acceleration_seq = eval_trans.evaluation_vqvae_motionmillion_1gpu(args.out_dir, train_loader, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, best_mpjpe=10000, draw=args.draw, save=args.save, savegif=args.savegif, savenpy=args.savenpy, comp_device=torch.device('cuda'), fps=args.fps, cal_acceleration=args.cal_acceleration)
    mpjpe.append(best_mpjpe)
    if args.cal_acceleration:
        pred_mean_acceleration_seq.append(best_pred_mean_acceleration_seq)
        pred_max_acceleration_seq.append(best_pred_max_acceleration_seq)
        gt_mean_acceleration_seq.append(best_gt_mean_acceleration_seq)
        gt_max_acceleration_seq.append(best_gt_max_acceleration_seq)
print('final result:')
print('mpjpe: ', sum(mpjpe)/repeat_time)
if args.cal_acceleration:
    print('pred_mean_acceleration_seq: ', sum(pred_mean_acceleration_seq)/repeat_time)
    print('pred_max_acceleration_seq: ', sum(pred_max_acceleration_seq)/repeat_time)
    print('gt_mean_acceleration_seq: ', sum(gt_mean_acceleration_seq)/repeat_time)
    print('gt_max_acceleration_seq: ', sum(gt_max_acceleration_seq)/repeat_time)
mpjpe = np.array((sum(mpjpe)/repeat_time).detach().cpu())

if args.cal_acceleration:
    pred_mean_acceleration_seq = np.array((sum(pred_mean_acceleration_seq)/repeat_time).detach().cpu())
    pred_max_acceleration_seq = np.array((sum(pred_max_acceleration_seq)/repeat_time).detach().cpu())
    gt_mean_acceleration_seq = np.array((sum(gt_mean_acceleration_seq)/repeat_time).detach().cpu())
    gt_max_acceleration_seq = np.array((sum(gt_max_acceleration_seq)/repeat_time).detach().cpu())
msg_final = f"MPJPE. {mpjpe:.3f}"
if args.cal_acceleration:
    msg_final += f", Pred_mean_acceleration_seq. {pred_mean_acceleration_seq:.3f}, Pred_max_acceleration_seq. {pred_max_acceleration_seq:.3f}, Gt_mean_acceleration_seq. {gt_mean_acceleration_seq:.3f}, Gt_max_acceleration_seq. {gt_max_acceleration_seq:.3f}"
logger.info(msg_final)

