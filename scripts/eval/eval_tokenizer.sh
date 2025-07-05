python eval_tokenizer.py \
--batch-size 1 \
--lr 4e-4 \
--total-iter 480000 \
--lr-scheduler 200000 \
--down-t 1 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir results/output/eval_fsq/600w_iter/wavelet \
--dataname motionmillion \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE_FSQ_bz64_lr5e-5_totaliter6000000_codebook65536_motionmillion_numworkers1_4gpu_online_96window-size_1down-t_3depth_3kernelsize_48000warmup_net_600w \
--quantizer FSQ \
--nb-code 65536 \
--motion_type vector_272 \
--version version1/tokenizer_96 \
--warm-up-iter 4800 \
--num-workers 16 \
--resume-pth results/output/FSQ_96len/train_VQVAE_FSQ_bz64_lr5e-5_totaliter6000000_codebook65536_motionmillion_numworkers1_4gpu_online_96window-size_1down-t_3depth_3kernelsize_48000warmup_wavelet_1patch_layernorm/net_6000000.pth \
--fps 30 \
--window-size 96 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--vq-norm LN 
# --cal_acceleration True 
# --savegif True \
# --draw True 