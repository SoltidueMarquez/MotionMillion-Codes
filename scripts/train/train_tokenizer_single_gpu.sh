export NCCL_TIMEOUT=1200
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29701 train_tokenizer.py \
--batch-size 64 \
--lr 5e-5 \
--total-iter 6000000 \
--lr-scheduler 300000 \
--down-t 1 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir results/output/FSQ_96len \
--dataname motionmillion \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name train_VQVAE_FSQ_bz64_lr5e-5_totaliter6000000_codebook65536_motionmillion_numworkers1_4gpu_online_96window-size_1down-t_3depth_3kernelsize_48000warmup_wavelet_1patch_layernorm \
--quantizer FSQ \
--nb-code 65536 \
--motion_type vector_272 \
--version version1/tokenizer_96 \
--warm-up-iter 48000 \
--num-workers 64 \
--window-size 96 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--vq-norm LN

# --print-iter 1 \
# --eval-iter 10 \
# --save-iter 10 \
# --save-latest 1
# --resume-pth 