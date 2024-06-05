##Generating images for visualization
imagenet_train_path="/home/zl/.cache/autoencoders/data/train"
codebook_path="codebook-100K.pth"
vq_path="vqgan-lc-100K-f16-dim8.pth"
sit_path="sit-lc-100K-f16.pth"
CUDA_VISIBLE_DEVICES=0 python visualization.py ODE --model SiT-XL/2 \
                    --image-size 256 \
                    --num-sampling-steps 250 \
                    --times 1 \
                    --n_sample 10 \
                    --seed 0 \
                    --embed_dim 8 \
                    --n_vision_words 100000 \
                    --local_embedding_path $codebook_path \
                    --vq_ckpt_path $vq_path \
                    --ckpt $sit_path \
                    --tuning_codebook 0 \
                    --use_cblinear 1 \
                    --cfg-scale 8 \
                    --save_dir "visualizations/sit_lc_100K_f16"
