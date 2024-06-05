##Run codes in vqgan-lc-master to generate initialized codebook and VQGAN-LC.

#Set "config/imagenet-f16-vqgan-lc-100K.yaml and config/imagenet-f8-vqgan-lc-100K.yaml":
#### $IMAGENET_PATH: the path of imagenet dataset
#### $VQGAN_LC_PATH: ckpt of VQGAN-LC ("vqgan-lc-100K-f16-dim8.pth" or "vqgan-lc-100K-f8-dim4.pth")
#### $CODEBOOK_PATH_100K: Initialized codebook ("codebook-100K.pth")

##Training LDM-f16 with VQGAN-LC-100K
python main.py \
    --base configs/imagenet-f16-ours-100K.yaml \
    -t --gpus 0,1,2,3,4,5,6,7, --logdir "./train_logs/ldm_lc_100K_f16"

##Training LDM-f8 with VQGAN-LC-100K
python main.py \
    --base configs/imagenet-f8-ours-100K.yaml \
    -t --gpus 0,1,2,3,4,5,6,7, --logdir "./train_logs/ldm_lc_100K_f8"    