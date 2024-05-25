from typing import Any, Tuple
import math
import torch
from torch import nn
import numpy as np


class PosEncodingBase(nn.Module):
    def __init__(self, in_dim, num_frequencies: int, **kwargs):
        super(PosEncodingBase, self).__init__()
        self.in_dim = in_dim
        self.out_dim = num_frequencies


class PosEncodingNone(PosEncodingBase):
    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        return x


class PosEncodingNeRF(PosEncodingBase):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020], BUT WITHOUT LOOPS :) '''
    LUT_NAME = "nerf"

    def __init__(self, in_dim, num_frequencies: int, **kwargs):
        super(PosEncodingNeRF, self).__init__(in_dim, num_frequencies, **kwargs)
        self.num_frequencies = num_frequencies // 2
        self.freq_scale = kwargs.get("coords_freq_scale", [1.0])
        if not isinstance(self.freq_scale, float):
            self.freq_scale = torch.as_tensor(self.freq_scale)[:, None]
        self.exp_i_pi = torch.tile(2**torch.arange(self.num_frequencies, dtype=torch.float32, requires_grad=False)[None], (in_dim, 1))

    def __repr__(self):
        d = "xyzt"
        return f"NeRF ({d[:self.in_dim]}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        _coords = torch.stack([torch.tile(coords[..., j:j + 1], (1, self.num_frequencies)) for j in range(coords.shape[-1])], dim=-1)
        prod = _coords * self.exp_i_pi.transpose(1, 0).to(coords.device)
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-2)
        emb = out.mean(-1)
        out = x + emb
        return out


class PosEncodingFourier(PosEncodingBase):
    ''' https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb '''
    LUT_NAME = "fourier"

    def __init__(self, in_dim, num_frequencies: int, **kwargs):
        super(PosEncodingFourier, self).__init__(in_dim, num_frequencies)
        self.num_frequencies = num_frequencies // 2
        self.freq_scale = kwargs.get("coords_freq_scale", [1.0])
        if not isinstance(self.freq_scale, float):
            self.freq_scale = torch.as_tensor(self.freq_scale)[:, None]
        self.B_gauss = torch.normal(0.0, 1.0, size=(self.in_dim, self.num_frequencies), requires_grad=False) * self.freq_scale
        self.B_gauss_pi = 2. * np.pi * self.B_gauss

    def __repr__(self):
        d = "xyzt"
        return f"Gaussian ({d[:self.in_dim]}   Freqs: {self.num_frequencies}, Scales: {self.freq_scale}, Out-dim: {self.out_dim})"

    def get_extra_state(self) -> Any:
        return {"B_gauss_pi": self.B_gauss_pi}  # Required to store gaussian array into network state dict

    def set_extra_state(self, state: Any):
        self.B_gauss_pi = state["B_gauss_pi"]  # Required to store gaussian array into network state dict

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        prod = coords @ self.B_gauss_pi.to(coords.device)
        emb = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        out = x + emb
        return out


class CAPE2D(PosEncodingBase):
    def __init__(self, in_dim, num_frequencies: int, **kwargs):
        super().__init__(in_dim, num_frequencies)
        self.big_k = num_frequencies

    def forward(self, x: torch.Tensor, coord: torch.Tensor):
        big_k = self.big_k
        k = torch.arange(0, big_k//2, device=coord.device)[None]
        wx = torch.pow(10, 2 * k / big_k) * torch.cos(k)
        wy = torch.pow(10, 2 * k / big_k) * torch.sin(k)
        cos = torch.cos(torch.pi * (coord[..., :1] @ wx + coord[..., 1:] @ wy))
        sin = torch.sin(torch.pi * (coord[..., :1] @ wx + coord[..., 1:] @ wy))
        emb = torch.cat((cos, sin), -1)
        out = x + emb
        return out


class CAPE1D(PosEncodingBase):
    def __init__(self, in_dim, num_frequencies: int, **kwargs):
        super().__init__(in_dim, num_frequencies)
        self.big_k = num_frequencies

    def forward(self, x: torch.Tensor, coord: torch.Tensor):
        big_k = self.big_k
        freq = 2.0 / 192.0 * torch.exp(-2.0 * torch.floor(torch.arange(big_k, device=coord.device) / 2) * math.log(1e4) / big_k)
        cos_shifts = torch.pi / 2.0 * (torch.arange(big_k, device=coord.device) % 2)
        emb = torch.sin(coord @ freq[None] + cos_shifts)
        out = x + emb
        return out


class CAPE2D1D(PosEncodingBase):
    def __init__(self, in_dim, num_frequencies: int, **kwargs):
        super().__init__(in_dim, num_frequencies)
        self.cape2d = CAPE2D(in_dim, num_frequencies)
        self.cape1d = CAPE1D(in_dim, num_frequencies)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, coord: torch.Tensor):
        out = self.cape2d(x, coord[..., :2])
        out = self.cape1d(out, coord[..., 2:])
        return out
