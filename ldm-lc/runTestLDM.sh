##Set checkpoint path of the trained LDM with VQGAN-LC
imagenet_path="/home/zl/.cache/autoencoders/data/"

##Evaluation-f16
ldm_lc_path="ldm-lc-f16.pth"
CUDA_VISIBLE_DEVICES=0 python run_evaluations.py \
            --times 2 \
            --n_sample 25 \
            --scale 1.6 \
            --edim 8 \
            --resolution 16 \
            --imagenet_path $imagenet_path \
            --ckpt_path $ldm_lc_path \
            --config_path "configs/imagenet-f16-vqgan-lc-100K.yaml" \
            --save_path "test_logs/ldm_lc_100K_f16"

##Evaluation-f8
ldm_lc_path="ldm-lc-f8.pth"
CUDA_VISIBLE_DEVICES=0 python run_evaluations.py \
            --times 2 \
            --n_sample 25 \
            --scale 1.4 \
            --edim 8 \
            --resolution 16 \
            --imagenet_path $imagenet_path \
            --ckpt_path $ldm_lc_path \
            --config_path "configs/imagenet-f8-vqgan-lc-100K.yaml" \
            --save_path "test_logs/ldm_lc_100K_f8"
