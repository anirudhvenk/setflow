import time
import torch
import numpy as np

from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, VPScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from scipy.optimize import linear_sum_assignment

from torchdiffeq import odeint
# from chamferdist import ChamferDistance

# visualization
# import matplotlib.pyplot as plt

from matplotlib import cm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

class Flow(nn.Module):
    def __init__(self, config, decoder):
        super().__init__()
        self.decoder = decoder
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.solver = ODESolver(velocity_model=self.decoder)
        self.eps = 1e-5

    def velocity(self, x, t, set_emb=None, attn_mask=None):
        mu = self.decoder(x, t, model_extras=(set_emb, attn_mask))
        return (mu - x) / (1. - t.unsqueeze(-1) + self.eps)

    def forward(self, pt, t, set_emb, attn_mask=None):
        return self.decoder(pt, t, model_extras=(set_emb, attn_mask))
    
    def get_loss(self, x_0, x_1, set_emb, attn_mask=None):
        t = torch.rand(x_1.shape[0]).to(x_1.device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        mu = self(path_sample.x_t, path_sample.t, set_emb, attn_mask)
        mask = attn_mask.float().unsqueeze(-1)

        # mask = attn_mask.float().unsqueeze(-1)
        # sse    = 0.5 * ((mu - path_sample.dx_t)**2 * mask)
        # loss   = sse.sum((1, 2)) / mask.sum((1, 2))
        # loss   = loss.mean()      
        # return loss
        # loss = 0.5 * ((mu - x_1)**2 * mask).sum((1,2))

        # loss = ps_loss / attn_mask.sum(-1)
        loss = ((mu - path_sample.dx_t)**2 * mask).sum((1,2)) / attn_mask.sum(-1)
        # loss = ((mu-path_sample.dx_t)**2 * mask)

        return loss.mean()

    def sample(self, batch_size, device, steps=50, set_emb=None, attn_mask=None, mode="vfm"):
        x_0 = torch.randn(1, batch_size, 2).to(device)
        device = x_0.device
        time_grid = torch.linspace(0., 1., steps, device=device)

        def ode_func(t, x):
            t_batch = t * torch.ones(x.shape[0], device=device)
            if mode == "vfm":
                v = self.velocity(x, t_batch, set_emb, attn_mask)
            elif mode == "cfm":
                v = self.forward(x, t_batch, set_emb, attn_mask)
            return v

        sol = odeint(
            ode_func,
            x_0,
            time_grid,
            atol=1e-9,
            rtol=1e-9,
            method="euler"
        )

        return sol[-1]