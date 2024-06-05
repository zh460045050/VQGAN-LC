##Training SiT with VQGAN-LC-100K
imagenet_train_path="/home/zl/.cache/autoencoders/data/train"
codebook_path="codebook-100K.pth"
vq_path="vqgan-lc-100K-f16-dim8.pth"
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 train.py --model SiT-XL/2 \
                                                --data-path $imagenet_train_path  \
                                                --embed_dim 8 \
                                                --n_vision_words 100000 \
                                                --tuning_codebook 0 \
                                                --use_cblinear 1 \
                                                --vq_config_path "vqgan_configs/vq-f16.yaml" \
                                                --local_embedding_path $codebook_path \
                                                --ckpt_path $vq_path \
                                                --global-batch-size 256 \
                                                --latent_size 16 \
                                                --results-dir "results/sit_lc_100K_f16"
