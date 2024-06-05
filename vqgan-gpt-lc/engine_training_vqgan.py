import math
import sys
from typing import Iterable

import torch
import util.lr_sched as lr_sched
import util.misc as misc
import copy
import numpy as np
import mlflow
from einops import rearrange
import matplotlib.pyplot as plt
import os
#import pyiqa
from scipy import linalg


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    ##
    #metric_logger.add_meter("acc", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    ##
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    opt_ae, opt_disc = optimizer
    loss_scaler_ae, loss_scaler_disc = loss_scaler
    #optimizer.zero_grad()
    token_freq = torch.zeros(args.n_vision_words).to(device)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, [image_ids, images, clip_images, label_cls] in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
    #for data_iter_step, images in enumerate(data_loader):

        cur_iter = len(data_loader) * epoch + data_iter_step

        ####Tokenizer with VQ-GAN
        b = images.shape[0]
        x = images.to(device)
        clip_images = clip_images.to(device)
        label_cls = label_cls.to(device)

        #with  torch.cuda.amp.autocast():
        loss, rec_loss, qloss, p_loss, g_loss, tk_labels, xrec = model(x, clip_images, cur_iter, step=0)
        
        tk_index_one_hot = torch.nn.functional.one_hot(tk_labels.view(-1), num_classes=args.n_vision_words)
        tk_index_num = torch.sum(tk_index_one_hot, dim=0)
        token_freq += tk_index_num
        
        opt_ae.zero_grad()
        lr_sched.adjust_learning_rate(opt_ae, data_iter_step / len(data_loader) + epoch, args)

        if args.use_cblinear != 0:
            loss_scaler_ae(loss, opt_ae, parameters=list(model.module.encoder.parameters())+
                                    list(model.module.decoder.parameters())+
                                    list(model.module.quant_conv.parameters())+
                                    list(model.module.tok_embeddings.parameters())+
                                    list(model.module.codebook_projection.parameters()) + 
                                    list(model.module.post_quant_conv.parameters()), update_grad=(data_iter_step + 1) % accum_iter == 0)
        else:
            loss_scaler_ae(loss, opt_ae, parameters=list(model.module.encoder.parameters())+
                                    list(model.module.decoder.parameters())+
                                    list(model.module.quant_conv.parameters())+
                                    list(model.module.tok_embeddings.parameters())+
                                    list(model.module.post_quant_conv.parameters()), update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if cur_iter > args.disc_start and args.rate_d != 0:
            #with  torch.cuda.amp.autocast():
            d_loss, _, _, _, _, _, _ = model(x, clip_images, cur_iter, step=1)
            opt_disc.zero_grad()
            lr_sched.adjust_learning_rate(opt_disc, data_iter_step / len(data_loader) + epoch, args)
            loss_scaler_disc(d_loss, opt_disc, parameters=model.module.discriminator.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)

        torch.cuda.synchronize()
        
        lr = opt_ae.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        misc.all_reduce_mean(loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)


        recloss_value = rec_loss.item()
        metric_logger.update(recloss=recloss_value)
        misc.all_reduce_mean(recloss_value)
        recloss_value_reduce = misc.all_reduce_mean(recloss_value)

        gloss_value = g_loss.item()
        metric_logger.update(gloss=gloss_value)
        misc.all_reduce_mean(gloss_value)
        gloss_value_reduce = misc.all_reduce_mean(gloss_value)

        if cur_iter > args.disc_start and args.rate_d != 0:
            dloss_value = d_loss.item()
            metric_logger.update(dloss=dloss_value)
            misc.all_reduce_mean(dloss_value)
            dloss_value_reduce = misc.all_reduce_mean(dloss_value)

        p_loss_value = p_loss.item()
        metric_logger.update(p_loss=p_loss_value)
        misc.all_reduce_mean(p_loss_value)
        p_loss_value_reduce = misc.all_reduce_mean(p_loss_value)

        qloss_value = qloss.item()
        metric_logger.update(qloss=qloss_value)
        misc.all_reduce_mean(qloss_value)
        qloss_value_reduce = misc.all_reduce_mean(qloss_value)


        """We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        if log_writer is not None and cur_iter % 1000 == 0:
            epoch_1000x = int(cur_iter)
            log_writer.add_scalar("Iter/lr", lr, epoch_1000x)
            log_writer.add_scalar("Iter/Loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/REC Loss", recloss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/Q Loss", qloss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/VGG Loss", p_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("Iter/GAN Loss", gloss_value_reduce, epoch_1000x)
            if cur_iter > args.disc_start and args.rate_d != 0:
                log_writer.add_scalar("Iter/Discriminator Loss", dloss_value_reduce, epoch_1000x)
    
    efficient_token = np.sum(np.array(token_freq.cpu().data) != 0)
    #metric_logger.update(efficient_token=efficient_token.float())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Efficient Tokens:", efficient_token)
    if log_writer is not None:
        log_writer.add_scalar("Epoch/Loss", loss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/REC Loss", recloss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/Q Loss", qloss_value_reduce, epoch)
        log_writer.add_scalar("Epoch/VGG Loss", p_loss_value_reduce, epoch)

        log_writer.add_scalar("Epoch/GAN Loss", gloss_value_reduce, epoch)
        if cur_iter > args.disc_start and args.rate_d != 0:
            log_writer.add_scalar("Epoch/Discriminator Loss", dloss_value_reduce, epoch)
        log_writer.add_scalar("Efficient Token", efficient_token, epoch)


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
