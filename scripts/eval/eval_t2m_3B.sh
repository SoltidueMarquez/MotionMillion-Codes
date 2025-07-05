python eval_t2m_llama.py \
--exp-name 3B_val_24w \
--batch-size 16 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 65536 \
--n-head-gpt 16 \
--block-size 301 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth results/output/FSQ_96len/train_VQVAE_FSQ_bz64_lr5e-5_totaliter6000000_codebook65536_motionmillion_numworkers1_4gpu_online_96window-size_1down-t_3depth_3kernelsize_48000warmup_wavelet_1patch_layernorm/net_6000000.pth \
--vq-name VQVAE_codebook_65536_FSQ_all \
--out-dir results/output/eval_T2M/ \
--total-iter 240000 \
--lr-scheduler-type CosineDecayScheduler \
--lr 0.0002 \
--dataname motionmillion \
--down-t 1 \
--depth 3 \
--quantizer FSQ \
--dilation-growth-rate 3 \
--vq-act relu \
--vq-norm LN \
--fps 30 \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--text_encode flan-t5-xl \
--pretrained_llama 3B \
--pkeep 1 \
--motion_type vector_272 \
--text_type texts \
--version version1/t2m_60_300 \
--mixed_precision bf16 \
--split val \
--resume-trans results/output/T2M/600iterFSQ/train_motionmillion_GPT_vqvae_65536_FSQ_pkeep_1_llama_3B_t5_xl_each_word_unigram_bf16_iter24w_bs16_8gpu_5machine_numworkers0_lr4e-4_gradacc1_cos-sched/net_240000.pth

# --print-iter 1 \
# --eval-metric-iter 1 \
# --eval-loss-iter 1 

# --eval-metric-iter 5000 \
# --eval-loss-iter 2000 \