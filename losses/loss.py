import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import Optional, Tuple
import math

class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )


class NMI(_Loss):
    """Normalized mutual information metric.

    As presented in the work by `De Vos 2020: <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113130R/Mutual-information-for-unsupervised-deep-learning-image-registration/10.1117/12.2549729.full?SSO=1>`_

    """

    def __init__(
        self,
        intensity_range: Optional[Tuple[float, float]] = None,
        nbins: int = 32,
        sigma: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -self.nmi_gauss(
            fixed, warped, bins_fixed, bins_warped, sigma=self.sigma
        ).mean()

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -self.nmi_gauss_mask(
            fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma
        )

    def nmi_gauss(self, x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of similar shape"

        def gaussian_window(x, bins, sigma):
            assert x.ndim == 2, "Input tensor should be 2-dimensional."
            return torch.exp(
                -((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma ** 2)
            ) / (math.sqrt(2 * math.pi) * sigma)

        x1_windowed = gaussian_window(x1.flatten(1), x1_bins, sigma)
        x2_windowed = gaussian_window(x2.flatten(1), x2_bins, sigma)
        p_XY = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
        p_XY = p_XY + e  # deal with numerical instability

        p_XY = p_XY / p_XY.sum((1, 2))[:, None, None]

        p_X = p_XY.sum(1)
        p_Y = p_XY.sum(2)

        I = (p_XY * torch.log(p_XY / (p_X[:, None] * p_Y[:, :, None]))).sum((1, 2))

        marg_ent_0 = (p_X * torch.log(p_X)).sum(1)
        marg_ent_1 = (p_Y * torch.log(p_Y)).sum(1)

        normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

        return normalized

    def nmi_gauss_mask(self, x1, x2, x1_bins, x2_bins, mask, sigma=1e-3, e=1e-10):
        def gaussian_window_mask(x, bins, sigma):
            assert x.ndim == 1, "Input tensor should be 2-dimensional."
            return torch.exp(-((x[None, :] - bins[:, None]) ** 2) / (2 * sigma ** 2)) / (
                    math.sqrt(2 * math.pi) * sigma
            )

        x1_windowed = gaussian_window_mask(torch.masked_select(x1, mask), x1_bins, sigma)
        x2_windowed = gaussian_window_mask(torch.masked_select(x2, mask), x2_bins, sigma)
        p_XY = torch.mm(x1_windowed, x2_windowed.transpose(0, 1))
        p_XY = p_XY + e  # deal with numerical instability

        p_XY = p_XY / p_XY.sum()

        p_X = p_XY.sum(0)
        p_Y = p_XY.sum(1)

        I = (p_XY * torch.log(p_XY / (p_X[None] * p_Y[:, None]))).sum()

        marg_ent_0 = (p_X * torch.log(p_X)).sum()
        marg_ent_1 = (p_Y * torch.log(p_Y)).sum()

        normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

        return normalized



class NCC(_Loss):
    def __init__(self, use_mask: bool = False):
        super().__init__()
        self.stablestd = StableStd.apply
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return -self.ncc(fixed, warped)

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        return -self.ncc_mask(fixed, warped, mask)

    def ncc_mask(self, x1, x2, mask, e=1e-10):  # TODO: calculate ncc per sample
        assert x1.shape == x2.shape, "Inputs are not of equal shape"
        x1 = torch.masked_select(x1, mask)
        x2 = torch.masked_select(x2, mask)
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = self.stablestd(x1) * self.stablestd(x2)
        ncc = cc / (std + e)
        return ncc

    def ncc(self, x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of equal shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = self.stablestd(x1) * self.stablestd(x2)
        ncc = cc / (std + e)
        return ncc