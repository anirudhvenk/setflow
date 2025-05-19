import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.nn.attention import SDPBackend, sdpa_kernel

from .rotary import Rotary, apply_rotary_pos_emb
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6*dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        # self.adaLN_modulation = nn.Sequential(
        #     nn.Linear(cond_dim, 4 * dim, bias=True),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(4 * dim, 6 * dim, bias=True)   # index -1
        # )

        # nn.init.zeros_(self.adaLN_modulation[-1].weight)
        # nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, attn_mask=None):
        with torch.amp.autocast('cuda', dtype=torch.float32, enabled=True):
            batch_size, seq_len = x.shape[0], x.shape[1]
            bias_dropout_scale_fn = self._get_bias_dropout_scale()

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

            # attention operation
            x_skip = x
            x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

            qkv = self.attn_qkv(x) # shape B x L X H
            q, k, v = rearrange(qkv, 'b s (three h d) -> b h three s d', three=3, h=self.n_heads).unbind(2)
            # qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

            # q, k, v = rearrange(qkv, 'b s three h d -> b h three s d', three=3, h=self.n_heads).unbind(2)
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask[:,None,None,:] if attn_mask is not None else None
            )

            x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)
            x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

            # mlp operation
            x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x,
                                    self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Linear(vocab_dim, dim)

    def forward(self, x):
        return self.embedding(x)


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        # nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # x1, x2 = self.linear(x).chunk(2, dim=-1)
        # return x1, x2
        return x