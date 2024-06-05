imagenet_path=""
codebook_path="codebook-100K.pth"
vq_path="vqgan-lc-100K-f16-dim8.pth"
gpt_path="gpt-lc-100K-f16.pth"

###Eval Generation
torchrun --nproc_per_node 1 --master_port=13207 eval_generation_imagenet.py \
    --batch_size 50 \
    --image_size 256 \
    --epochs 100 \
    --lr 4.5e-4 \
    --n_class 1000 \
    --imagenet_path $imagenet_path \
    --num_workers 8 \
    --vq_config_path vqgan_configs/vq-f16.yaml \
    --output_dir "log_eval_gpt/gpt_lc_100K_f16" \
    --local_embedding_path $imagenet_path \
    --stage_1_ckpt $vq_path \
    --stage_2_ckpt $gpt_path \
    --n_vision_words 100000 \
    --tuning_codebook 0 \
    --use_cblinear 1 \
    --embed_dim 8 \
    --top_k 100000 \
    --dataset "imagenet" \
    --gpt_type "small"