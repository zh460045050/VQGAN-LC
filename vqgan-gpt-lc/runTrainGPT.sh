#imagenet_path=""
imagenet_path="/mnt/nasv2/MILab/zhulei/nasv1_dmp/datas/imagenet"
codebook_path="codebook-100K.pth"
vq_path="vqgan-lc-100K-f16-dim8.pth"

####Training V2L Tokenizer Stage2 (32-V100, 32 batches per GPU)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=14297 training_gpt.py \
    --batch_size 1024 \
    --image_size 256 \
    --epochs 100 \
    --lr 4.5e-4 \
    --n_class 4 \
    --imagenet_path $imagenet_path \
    --num_workers 16 \
    --vq_config_path vqgan_configs/vq-f16.yaml \
    --local_embedding_path $codebook_path \
    --stage_1_ckpt $vq_path   \
    --n_vision_words 100000 \
    --tuning_codebook 0 \
    --use_cblinear 1 \
    --embed_dim 8 \
    --output_dir "train_logs_gpt/gpt_lc_100K" \
    --deepspeed \
    --deepspeed_config "config/deepspeed_gpt_zero2_small.json" \
    --gpt_type "small"
