export NCCL_TIMEOUT=1200
accelerate launch --config_file configs/accelerate_configs/mnode8gpu.yaml --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} --machine_rank=${RANK} --num_machines=${WORLD_SIZE} --num_processes=40 train_t2m_llama.py \
--exp-name train_motionmillion_GPT_vqvae_65536_FSQ_pkeep_1_llama_3B_t5_xl_each_word_unigram_bf16_iter24w_bs16_8gpu_5machine_numworkers0_lr4e-4_gradacc1_cos-sched \
--batch-size 16 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 65536 \
--n-head-gpt 16 \
--block-size 301 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth ./checkpoints/pretrained_models/fsq_net_6000000.pth \
--vq-name VQVAE_codebook_65536_FSQ_all \
--out-dir results/output/T2M/600iterFSQ \
--total-iter 240000 \
--lr-scheduler-type CosineDecayScheduler \
--lr 0.0004 \
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
--save-iter-last 1000 \
--gradient_accumulation_steps 1 \
--save-iter 10000 \
--train_split train
# --resume-trans results/output/T2M_60_300/600iterFSQ/train_motionmillion_GPT_vqvae_65536_FSQ_pkeep_1_llama_3B_t5_xl_each_word_unigram_bf16_iter24w_bs16_8gpu_5machine_numworkers0_lr4e-4_gradacc1_cos-sched/net_last.pth
# --print-iter 1 \
# --eval-metric-iter 1 \
# --eval-loss-iter 1 