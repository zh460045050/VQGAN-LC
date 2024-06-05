#@title loading utils
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import argparse
import os 
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
#import matplotlib.pyplot as plt
import cv2

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(args):

    config = OmegaConf.load(args.config_path)  
    model = load_model_from_config(config, args.ckpt_path)

    return model



###
parser = argparse.ArgumentParser()
parser.add_argument("--step", default=200, type=int)
parser.add_argument("--times", default=1, type=int)
parser.add_argument("--n_sample", default=10, type=int)
parser.add_argument("--edim", default=4, type=int)
parser.add_argument("--resolution", default=32, type=int)
parser.add_argument("--eta", default=1.0, type=float)
parser.add_argument("--scale", default=4.0, type=float)
parser.add_argument("--save_path", type=str, default="test_logs", help="")
parser.add_argument("--ckpt_path", type=str, default="", help="")
parser.add_argument("--config_path", type=str, default="", help="")
parser.add_argument("--imagenet_path", type=str, default="", help="")
args = parser.parse_args()
###

model = get_model(args)
sampler = DDIMSampler(model)

n_samples_per_class = args.n_sample #每类每次采样几个
times = args.times #每类采样多少次
ddim_steps = args.step #difussion step数
scale = args.scale #4.0
ddim_eta = args.eta #1.0

count = 0
if not os.path.exists(os.path.join(args.save_path, "vis_dirs")):
    os.makedirs(os.path.join(args.save_path, "vis_dirs"))
if not os.path.exists(os.path.join(args.save_path, "save_dirs")):
    os.makedirs(os.path.join(args.save_path, "save_dirs"))


token_freq = torch.zeros(model.first_stage_model.n_vision_words).to(model.device)


while count < 1000:
    print(count)
    classes = [count, count+1, count+2, count+3, count+4, count+5, count+6, count+7]
    count = count + 8
    #classes = [count, count+1, count+2, count+3]
    #count = count + 4
    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )

            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                for time in range(0, times):
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples_per_class,
                                                    shape=[args.edim, args.resolution, args.resolution],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                     eta=ddim_eta)
                    
                    ###
                    with torch.no_grad():
                        _, _, [_, _, tk_labels] = model.first_stage_model.quantize(samples_ddim)
                        tk_index_one_hot = torch.nn.functional.one_hot(tk_labels.view(-1), num_classes=model.first_stage_model.n_vision_words)
                        tk_index_num = torch.sum(tk_index_one_hot, dim=0)
                        token_freq += tk_index_num
                        ###

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                                    min=0.0, max=1.0)
                    for num in range(0, n_samples_per_class):
                        cv2.imwrite(os.path.join(args.save_path, "save_dirs", "%d_%d_%d.jpg"%(class_label, time, num)), cv2.cvtColor(np.uint8(np.array(x_samples_ddim[num].permute(1, 2, 0).cpu().data * 255.0)), cv2.COLOR_BGR2RGB))
                all_samples.append(x_samples_ddim[:10])

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    cv2.imwrite(os.path.join(args.save_path, "vis_dirs/%d.jpg"%(count)), cv2.cvtColor(np.uint8(grid), cv2.COLOR_BGR2RGB))
    Image.fromarray(grid.astype(np.uint8))

###
np.save(os.path.join(args.save_path, "token_freq.npy"), np.array(token_freq.cpu().data))

from cleanfid import fid
fid_value = fid.compute_fid(os.path.join(args.save_path, "save_dirs"), imagenet_path + "/train", mode="clean")

efficient_token = np.sum(np.array(token_freq.cpu().data) != 0)
with open(os.path.join(args.save_path, "recons.csv"), 'a') as f:
    f.write("FID, IS, Effective_Tokens \n")
    f.write("%.4f, %.4f, %d \n"%(fid_value, is_value, efficient_token))