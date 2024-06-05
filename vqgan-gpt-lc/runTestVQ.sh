imagenet_path=""
codebook_path="codebook-100K.pth"
vq_path="vqgan-lc-100K-f16-dim8.pth"

###Eval Reconstruction
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=15301 eval_reconstruction.py \
        --batch_size 8 \
        --image_size 256 \
        --lr 9e-3 \
        --n_class 1000 \
        --imagenet_path $imagenet_path \
        --vq_config_path vqgan_configs/vq-f16.yaml \
        --output_dir "log_eval_recons/vqgan_lc_100K_f16" \
        --log_dir "log_eval_recons/vqgan_lc_100K_f16" \
        --quantizer_type "org" \
        --local_embedding_path $codebook_path \
        --stage_1_ckpt $vq_path \
        --tuning_codebook 0 \
        --embed_dim 8 \
        --n_vision_words 100000 \
        --use_cblinear 1 \
        --dataset "imagenet"
