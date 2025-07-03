import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu', 'swiGLU'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    parser.add_argument('--causal', action='store_true', help='whether use causal conv')
    parser.add_argument('--model', type=str, default='vqvae', choices=['tae', 'vqvae'], help='model type')
    parser.add_argument('--z-latent', type=int, default=1, help='z latent')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset', 'LFQ', "FSQ", "BSQ"], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=3000, type=int, help='evaluation frequency')
    parser.add_argument('--save-iter', default=10000, type=int, help='save frequency')
    parser.add_argument('--save-latest', default=1000, type=int, help='whether save the latest model')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')


    ## debug
    parser.add_argument('--debug', action='store_true', help='debug mode')

    ## motionx
    parser.add_argument('--motion_type', type=str, default='vector_263', help='motion type')
    parser.add_argument('--text_type', type=str, default='texts', help='text type')
    parser.add_argument('--version', type=str, default='version1', help='version')
    parser.add_argument('--num-workers', type=int, default=40, help='number of workers')
    
    # visualization
    parser.add_argument('--savegif', type=bool, default=False, help='save gif')
    parser.add_argument('--savenpy', type=bool, default=False, help='save npy')
    parser.add_argument('--draw', type=bool, default=False, help='draw')
    parser.add_argument('--save', type=bool, default=False, help='save')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    
    parser.add_argument('--use_acc_loss', type=bool, default=False, help='use acc loss')
    parser.add_argument('--use_acc_vel_loss', type=bool, default=False, help='use acc vel loss')
    
    parser.add_argument('--use_root_loss', type=bool, default=False, help='use root loss')
    parser.add_argument('--root_loss', type=float, default=0.5, help='root loss')
    
    parser.add_argument('--acc_loss', type=float, default=0.5, help='acc loss')
    parser.add_argument('--acc_vel_loss', type=float, default=0.5, help='acc vel loss')
    parser.add_argument('--kernel-size', type=int, default=3, help='kernel size')
    
    parser.add_argument('--use_patcher', action='store_true', help='use patcher')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--patch_method', type=str, default='haar', help='patch method')
    parser.add_argument('--use_attn', type=bool, default=False, help='use attn')
    parser.add_argument('--fps', type=int, default=60, help='fps')
    parser.add_argument('--cal_acceleration', action='store_true', help='cal acceleration')
    
    return parser.parse_args()