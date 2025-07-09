python inference_batch.py \
--exp-name 7B_600wFSQ_24wIter \
--batch-size 32 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 65536 \
--n-head-gpt 16 \
--block-size 301 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth ./checkpoints/pretrained_models/fsq_net_6000000.pth \
--vq-name VQVAE_codebook_65536_FSQ_all \
--out-dir results/output/inference/batch_inference/ \
--total-iter 120000 \
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
--pretrained_llama 7B \
--pkeep 1 \
--motion_type vector_272 \
--text_type texts \
--version version1/t2m_60_300 \
--mixed_precision bf16 \
--use_rewrite_model \
--rewrite_model_path ./checkpoints/rewrite_models/Meta-Llama-3.1-8B-Instruct \
--infer_batch_prompt ./assets/infer_batch_prompt.txt \
--resume-trans ./checkpoints/pretrained_models/motionmillion_7B_all.pth

# --print-iter 1 \
# --eval-metric-iter 1 \
# --eval-loss-iter 1 

# --eval-metric-iter 5000 \
# --eval-loss-iter 2000 \