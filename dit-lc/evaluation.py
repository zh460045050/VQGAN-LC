# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from tokenizer.models_v2l import VQModel_LLaMA
from download import find_model
from models import DiT_models
import yaml
import os
import numpy as np
import cv2
import argparse


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    #latent_size = args.image_size // 8
    latent_size = args.latent_size #args.image_size // 16
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.embed_dim
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    ###
    config = load_config(args.vq_config_path, display=True)
    vae = VQModel_LLaMA(**config.model.params, tuning_codebook=args.tuning_codebook, n_vision_words=args.n_vision_words, local_embedding_path=args.local_embedding_path, use_cblinear=args.use_cblinear)
    vae = vae.to(device)
    sd = torch.load(os.path.join(args.vq_ckpt_path), map_location="cpu")["state_dict"]
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    vae = vae.eval()
    print(missing, unexpected)
    ###

    # Labels to condition the model with (feel free to change):
    #class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    count = 0
    n_sample = args.num_sample
    if not os.path.exists(os.path.join(args.save_dir, "vis_dir")):
        os.makedirs(os.path.join(args.save_dir, "vis_dir"))
    if not os.path.exists(os.path.join(args.save_dir, "save_dir")):
        os.makedirs(os.path.join(args.save_dir, "save_dir"))

    token_freq = torch.zeros(vae.n_vision_words).to(device)

    while count < 1000:

        if latent_size == 16:
            #class_labels = [count, count+1, count+2, count+3]
            #count = count + 4
            class_labels = [count, count+1]
            count = count + 2
        else:
            print("!!!")
            class_labels = [count]
            count = count + 1
        n = len(class_labels)
        z = torch.randn(n, n_sample, args.embed_dim, latent_size, latent_size, device=device)
        z = z.view(n*n_sample, args.embed_dim, latent_size, latent_size)

        y = torch.tensor(class_labels, device=device).unsqueeze(-1).expand(n, n_sample).contiguous()
        y = y.view(-1)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n * n_sample, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        #samples = vae.decode(samples / 0.18215).sample

        ###
        #with torch.no_grad():
        #    _, _, [_, _, tk_labels] = vae.quantize(samples)
        #    tk_index_one_hot = torch.nn.functional.one_hot(tk_labels.view(-1), num_classes=vae.n_vision_words)
        #    tk_index_num = torch.sum(tk_index_one_hot, dim=0)
        #    token_freq += tk_index_num
        #    ###

        samples = vae.decode(samples)

        # Save and display images:
        save_image(samples, os.path.join(args.save_dir, "vis_dir", "sample_%s.jpg")%(str(count)), nrow=n_sample, normalize=True, value_range=(-1, 1))
        
        
        #samples[samples < -1] = -1
        #samples[samples > 1] = 1
        #samples = (samples + 1) / 2
        for i in range(0, samples.shape[0]):
            save_image(samples[i].unsqueeze(0), os.path.join(args.save_dir, "save_dir", "%s_%s.jpg")%(str(count), str(i)), nrow=1, normalize=True, value_range=(-1, 1))
            #cv2.imwrite(os.path.join(args.save_dir, "save_dir", "%s_%s.jpg")%(str(count), str(i)), np.uint8(np.array(samples[i].permute(1, 2, 0).cpu().data)*255.0))

    np.save(os.path.join(args.save_dir, "token_freq.npy"), np.array(token_freq.cpu().data))
    from cleanfid import fid
    score = fid.compute_fid(os.path.join(args.save_dir, "save_dir"), dataset_name="imagenet_train", mode="clean", dataset_split="custom")

    efficient_token = np.sum(np.array(token_freq.cpu().data) != 0)
    #from metrics.inception_score.inception_score import IgnoreLabelDataset, inception_score, read_data
    #data = read_data(os.path.join(args.save_path, "save_dirs"))
    #ite_data = IgnoreLabelDataset(data)
    #print ("Calculating Inception Score...")
    #is_value, _ = inception_score(ite_data, cuda=True, batch_size=10, resize=True, splits=10)


    #with open(os.path.join(args.save_dir, "recons.csv"), 'a') as f:
    #    f.write("FID, IS, Effective_Tokens \n")
    #    f.write("%.4f, %.4f, %d \n"%(fid_value, is_value, efficient_token))

    with open(os.path.join(args.save_dir, "recons.csv"), 'a') as f:
        f.write("FID, Effective_Tokens \n")
        f.write("%.4f, %d \n"%(score, efficient_token))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    #####
    parser.add_argument("--embed_dim", type=int, default=8)
    parser.add_argument("--n_vision_words", type=int, default=16384)
    parser.add_argument("--vq_ckpt_path", type=str, default="")
    parser.add_argument("--local_embedding_path", type=str, default="cluster_codebook_1000cls_100000.pth")
    parser.add_argument("--tuning_codebook", type=int, default=0)
    parser.add_argument("--use_cblinear", type=int, default=1)
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/vq-f16.yaml")
    parser.add_argument("--save_dir", type=str, default="test_logs/debug")

    parser.add_argument("--num_sample", type=int, default=10)
    #####

    args = parser.parse_args()
    main(args)
