# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import util.misc as misc

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tokenizer.models_v2l import VQModel_LLaMA
from omegaconf import OmegaConf
import yaml
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    experiment_dir = f"{args.results_dir}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    #latent_size = args.image_size // 8
    latent_size = args.latent_size
    model = DiT_models[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        in_channels=args.embed_dim
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank % torch.cuda.device_count()])
    model_without_ddp = model.module

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    ###
    config = load_config(args.vq_config_path, display=True)
    vae = VQModel_LLaMA(**config.model.params, tuning_codebook=args.tuning_codebook, n_vision_words=args.n_vision_words, local_embedding_path=args.local_embedding_path, use_cblinear=args.use_cblinear)
    vae = vae.to(device)
    sd = torch.load(os.path.join(args.ckpt_path), map_location="cpu")["state_dict"]
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    vae = vae.eval()
    print(missing, unexpected)
    ###

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    ###
    if os.path.exists(os.path.join(checkpoint_dir, "last.pt")): ##Auto Resume
        print("Resume from: %s"%(os.path.join(checkpoint_dir, "last.pt")))
        ckpt = torch.load(os.path.join(checkpoint_dir, "last.pt"), map_location="cpu")
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        ema.load_state_dict(ckpt["ema"], strict=True)
        args = ckpt["args"]
        start_epoch = args.start_epoch
    else:
        start_epoch = 0

    if rank == 0 and experiment_dir is not None:
        log_writer = SummaryWriter(log_dir=experiment_dir)
    else:
        log_writer = None

    # Variables for monitoring/logging purposes:
    train_steps = len(loader) * start_epoch
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    print_freq = 10
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")

        sampler.set_epoch(epoch)
        header = "Training Epoch: [{}]".format(epoch)
        metric_logger = misc.MetricLogger(delimiter="  ")
        #for x, y in loader:
        for _, [x, y] in enumerate(metric_logger.log_every(loader, print_freq, header)):
            x = x.to(device)
            y = y.to(device)
            x = x*2.0 - 1
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x)
                #print(dec.shape)
                
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            ###
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)
            misc.all_reduce_mean(loss_value)
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None:# and cur_iter % 1000 == 0:
                log_writer.add_scalar("Iter/Loss", loss_value_reduce, train_steps)
            ###


            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
        
            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:09d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        metric_logger.synchronize_between_processes()
        print("Training Averaged stats:", metric_logger)
        if rank == 0:
            args.start_epoch = epoch + 1
            checkpoint = {
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            checkpoint_path = f"{checkpoint_dir}/last.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        if log_writer is not None:
            log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    #####
    parser.add_argument("--embed_dim", type=int, default=8)
    parser.add_argument("--n_vision_words", type=int, default=16384)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--local_embedding_path", type=str, default="cluster_codebook_1000cls_100000.pth")
    parser.add_argument("--tuning_codebook", type=int, default=0)
    parser.add_argument("--use_cblinear", type=int, default=1)
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/vq-f16.yaml")
    parser.add_argument("--latent_size", type=int, default=16)
    #####

    args = parser.parse_args()
    main(args)
