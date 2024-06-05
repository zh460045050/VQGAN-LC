import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mingpt import GPT
from models.models_vq import VQModel 
from omegaconf import OmegaConf
import yaml
import os
from models.llama import LLaMA

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token
        args.stage = 2
        self.args = args
        self.vqgan = self.load_vqgan(args)

        self.loss_computer = LabelSmoothing(smoothing=0.1)

        ####GPT-small
        transformer_config = {
            "vocab_size": args.n_vision_words + args.n_class,
            "block_size": 257,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024
        }
        self.transformer = GPT(**transformer_config)
        self.pkeep = args.pkeep

    @staticmethod
    def load_vqgan(args):
        #model = VQGAN(args)
        config = load_config(args.vq_config_path, display=True)
        model = VQModel(args=args, **config.model.params)
        if "last" in args.stage_1_ckpt:
            sd = torch.load(os.path.join(args.stage_1_ckpt), map_location="cpu")["model"]
        else:
            sd = torch.load(os.path.join(args.stage_1_ckpt), map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        #model.load_checkpoint(args.stage_1_ckpt)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        #quant_z, indices, _ = self.vqgan.encode(x)
        quant_z, _, [_, _, indices] = self.vqgan.encode(x)
        #print(quant_z.shape)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
    
    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):

        ###
        if self.args.use_cblinear == 1:
            vision_tok_embeddings_weight = self.vqgan.codebook_projection(self.vqgan.tok_embeddings.weight)
        else:
            vision_tok_embeddings_weight = self.vqgan.tok_embeddings.weight
        ix_to_vectors = F.embedding(indices, vision_tok_embeddings_weight).reshape(indices.shape[0], p1, p2, self.args.embed_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)

        return image

    def forward(self, x, c_indices=None):

        with torch.no_grad():
            _, indices = self.encode_to_z(x)
    
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
            mask = mask.round().to(dtype=torch.int64)
            random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
            new_indices = mask * indices + (1 - mask) * random_indices
        else:
            new_indices = indices

        if not c_indices is None:
            new_indices = torch.cat((c_indices, new_indices), dim=1)
            logits, _ = self.transformer(new_indices[:, :-1])
        else:
            new_indices = torch.cat((sos_tokens, new_indices), dim=1)
            logits, _ = self.transformer(new_indices[:, :-1])
        target = indices

        if self.args.label_smooth == 1:
            loss = self.loss_computer(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        
        return loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    #cleanFID
    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        if x is not None:
            x = torch.cat((c, x), dim=1)
        else:
            x = c
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :self.args.n_vision_words] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        #self.transformer.train()
        return x
    
    @torch.no_grad()
    def log_images(self, x, c_indices=None, num=None):
        log = dict()
        if num is None:
            num = x.shape[0]
            
        z_q, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(num, 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :indices.shape[1] // 2]
        if not c_indices is None:
            sample_indices = self.sample(start_indices, c_indices, steps=indices.shape[1] - start_indices.shape[1])
        else:
            sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        if not c_indices is None:
            sample_indices = self.sample(start_indices, c_indices, steps=indices.shape[1])
        else:
            sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))
















