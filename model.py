import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import json
import ml_collections

# from deepspeed.ops.adam import FusedAdam
from flow_matching.model import Flow
from dit.fused_add_dropout_scale import bias_dropout_add_scale_fused_train
from dit.rotary import Rotary, apply_rotary_pos_emb
from dit.transformer import DDiTBlock, DDitFinalLayer, LayerNorm, TimestepEmbedder
from tqdm import tqdm
from einops import rearrange
from setvae.models.networks import SetVAE
from argparse import Namespace
from setvae.args import get_parser

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.encoder.num_heads

        self.norm1 = LayerNorm(config.encoder.hidden_dim)
        self.norm2 = LayerNorm(config.encoder.hidden_dim)

        self.attn_qkv = nn.Linear(
            config.encoder.hidden_dim,
            3 * config.encoder.hidden_dim, 
            bias=False
        )
        self.attn_out = nn.Linear(
            config.encoder.hidden_dim, 
            config.encoder.hidden_dim,
            bias=False
        )
        self.ff = nn.Sequential(
            nn.Linear(config.encoder.hidden_dim, config.encoder.hidden_dim  * 4),
            nn.GELU(),
            nn.Linear(config.encoder.hidden_dim*4, config.encoder.hidden_dim),
        )

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]

        x_res = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x) # shape B x L X H
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        q, k, v = rearrange(qkv, 'b s three h d -> b h three s d', three=3, h=self.n_heads).unbind(2)
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask[:,None,None,:] if attn_mask is not None else None
        )

        x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)
        x = self.attn_out(x) + x_res

        x_res = x
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = x_res + ff_out

        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_proj = nn.Linear(
            config.encoder.input_dim,
            config.encoder.hidden_dim
        )

        self.blocks = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.encoder.depth)
        ])

        self.mu_proj = nn.Linear(
            config.encoder.hidden_dim,
            config.encoder.hidden_dim
        )

        self.logvar_proj = nn.Linear(
            config.encoder.hidden_dim,
            config.encoder.hidden_dim
        )

    def forward(self, x, attn_mask=None):
        x = self.input_proj(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, attn_mask)
        
        # mean pooling
        x = x * attn_mask.unsqueeze(-1)
        x = x.sum(dim=1) / attn_mask.sum(dim=1).unsqueeze(-1)

        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_scales = config.encoder.z_scales
        self.input_proj = nn.Linear(
            config.encoder.input_dim,
            config.decoder.hidden_dim
        )

        self.emb_proj_blocks = nn.ModuleList([
            nn.Linear(
                config.encoder.z_dim * z_scale,
                config.decoder.conditioning_dim
            ) for z_scale in config.encoder.z_scales
        ])

        self.timestep_emb = TimestepEmbedder(config.decoder.conditioning_dim)
        self.rotary_emb = Rotary(config.decoder.hidden_dim // config.decoder.num_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(
                config.decoder.hidden_dim,
                config.decoder.num_heads,
                config.decoder.conditioning_dim, 
                dropout=config.decoder.dropout
            ) for _ in range(config.decoder.depth)
        ])
        self.output_layer = DDitFinalLayer(
            config.decoder.hidden_dim,
            config.encoder.input_dim,
            config.decoder.conditioning_dim
        )

    def forward(self, x, t, model_extras=None):
        latents, attn_mask = model_extras

        B, N, _ = x.shape
        z = latents[0].view(B, -1)

        x = self.input_proj(x)
        t_emb = self.timestep_emb(t)
        rotary_cos_sin = self.rotary_emb(x)

        for i in range(len(self.blocks)):
            set_emb = self.emb_proj_blocks[i](latents[i])
            c = F.silu(t_emb + set_emb)
            x = self.blocks[i](x, rotary_cos_sin, c, attn_mask=attn_mask)
        mu = self.output_layer(x, c)

        return mu
    
class SetFlowModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = Decoder(config)

        if config.training.type == "mnist":
            input_dim=2
            max_outputs=400
            init_dim=32
            n_mixtures=4
            z_dim=16
            hidden_dim=64
            num_heads=4
            lr=1e-3
            beta=1e-2
            epochs=200
            kl_warmup_epochs=50
            scheduler="linear"
            dataset_type="mnist"
            log_name="test"
            mnist_data_dir="cache/mnist"

            args = Namespace(
                kl_warmup_epochs=kl_warmup_epochs,
                input_dim=input_dim,
                max_outputs=max_outputs,
                init_dim=init_dim,
                n_mixtures=n_mixtures,
                z_dim=z_dim,
                z_scales=[2, 4, 8, 16, 32],
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                lr=lr,
                beta=beta,
                epochs=epochs,
                dataset_type=dataset_type,
                log_name=log_name,
                mnist_data_dir=mnist_data_dir,
                resume_optimizer=True,
                save_freq=10,
                viz_freq=10,
                log_freq=10,
                val_freq=1000,
                scheduler=scheduler,
                slot_att=True,
                ln=True,
                seed=42,
                distributed=True,
            )
        elif config.training.type == "shapenet":
            input_dim=3
            max_outputs=2500
            init_dim=32
            n_mixtures=4
            z_dim=16
            hidden_dim=64
            num_heads=4
            dataset_type="shapenet15k"
            log_name="gen/shapenet15k-airplane/camera-ready"

            args = Namespace(
                input_dim=input_dim,
                max_outputs=max_outputs,
                init_dim=init_dim,
                n_mixtures=n_mixtures,
                z_dim=z_dim,
                z_scales=[1, 1, 2, 4, 8, 16, 32],
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dataset_type=dataset_type,
                log_name=log_name,
                resume_optimizer=True,
                save_freq=10,
                viz_freq=10,
                log_freq=10,
                val_freq=1000,
                slot_att=True,
                ln=True,
                seed=42,
                distributed=True,
            )

        parser = get_parser()
        args   = parser.parse_args([], namespace=args)
        self.encoder = SetVAE(args)
        if self.config.training.type == "mnist":
            ckpt = torch.load(f"setvae/checkpoints/gen/mnist/camera-ready/checkpoint-199.pt")
        elif self.config.training.type == "shapenet":
            ckpt = torch.load(f"setvae/checkpoints/gen/shapenet15k-airplane/camera-ready/checkpoint-7999.pt")

        self.encoder.load_state_dict(ckpt['model'])
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.model = Flow(config, self.decoder)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def on_train_epoch_start(self):
        # torch.cuda.empty_cache()

        self.model.train()
        
    def training_step(self, batch, batch_idx):
        if self.config.training.type == "mnist":
            x_0, x_1, mask = batch
        elif self.config.training.type == "shapenet":
            x_0, x_1, mask = batch["x_0"], batch["set"], batch["set_mask"]
            # x_0 = torch.randn_like(x_1).to(x_1.device)
        
        with torch.no_grad():
            out = self.encoder(x_1, ~mask)
            latents = [z[0].view(x_1.shape[0], -1) for z in out["posteriors"][1:]]
        # print(latents)
        
        loss = self.model.get_loss(x_0, x_1, latents, attn_mask=mask)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss
    
    @torch.no_grad()
    def reconstruct(self, sample, batch_size, timesteps, attn_mask=None, mode="vfm"):
        with torch.no_grad():
            out = self.encoder(sample, ~attn_mask)
            latents = [z[0].view(sample.shape[0], -1) for z in out["posteriors"][1:]]
            
        return self.model.sample(
            batch_size=batch_size,
            set_emb=latents,
            steps=timesteps,
            device="cuda:3",
            attn_mask=attn_mask,
            mode=mode
        )
    
    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(
            # self.parameters(),
            trainable_params,
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.beta1, self.config.training.beta2),
            fused=True
        )

        return {
            "optimizer": optimizer,
        }