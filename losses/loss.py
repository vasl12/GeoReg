import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import Optional, Tuple
import math
import torchvision
from torch import nn
import numpy as np
from torch.nn import functional as F


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

    def metric(self, warped: Tensor, fixed: Tensor) -> Tensor:
        return -self.ncc(fixed, warped)

    def masked_metric(self, warped: Tensor, fixed: Tensor, mask: Tensor) -> Tensor:
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


class CosineSimLoss(_Loss):
    def __init__(self, dim=1):
        super().__init__()
        self.cossim = torch.nn.CosineSimilarity(dim=dim)

    def forward(self, fixed: Tensor, warped: Tensor) -> Tensor:
        cossim = self.cossim(warped, fixed)
        loss = 1.0 - cossim
        return loss.mean()


class PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class L4(nn.Module):

    def forward(self, pred, tgt):
        diff = pred - tgt
        l4 = diff * diff * diff * diff
        return l4.mean()


class LNCCLoss(nn.Module):
    """
    Local Normalized Cross Correlation loss
    Adapted from VoxelMorph implementation:
    https://github.com/voxelmorph/voxelmorph/blob/5273132227c4a41f793903f1ae7e27c5829485c8/voxelmorph/torch/losses.py#L7
    """
    def __init__(self, window_size=7):
        super(LNCCLoss, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        # products and squares
        xsq = x * x
        ysq = y * y
        xy = x * y

        # set window size
        ndim = x.dim() - 2
        # window_size = param_ndim_setup(self.window_size, ndim)
        window_size = (self.window_size,) * ndim

        # summation filter for convolution
        sum_filt = torch.ones(1, 1, *window_size).type_as(x)

        # set stride and padding
        stride = (1,) * ndim
        padding = tuple([math.floor(window_size[i]/2) for i in range(ndim)])

        # get convolution function of the correct dimension
        conv_fn = getattr(F, f'conv{ndim}d')

        # summing over window by convolution
        x_sum = conv_fn(x, sum_filt, stride=stride, padding=padding)
        y_sum = conv_fn(y, sum_filt, stride=stride, padding=padding)
        xsq_sum = conv_fn(xsq, sum_filt, stride=stride, padding=padding)
        ysq_sum = conv_fn(ysq, sum_filt, stride=stride, padding=padding)
        xy_sum = conv_fn(xy, sum_filt, stride=stride, padding=padding)

        window_num_points = np.prod(window_size)
        x_mu = x_sum / window_num_points
        y_mu = y_sum / window_num_points

        cov = xy_sum - y_mu * x_sum - x_mu * y_sum + x_mu * y_mu * window_num_points
        x_var = xsq_sum - 2 * x_mu * x_sum + x_mu * x_mu * window_num_points
        y_var = ysq_sum - 2 * y_mu * y_sum + y_mu * y_mu * window_num_points

        lncc = cov * cov / (x_var * y_var + 1e-5)

        return -torch.mean(lncc)
