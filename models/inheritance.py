from typing import Optional, Tuple, Union

import torch
from torch import nn

from models.layers import CrossAttentionLayer, AttentionWeightedLayer
from utils.bpline_utils import set_kernel, bspline_interpolation
from utils.pos_encoding import PosEncodingBase
from utils.unfoldNd import unfold3D, unfold2D


def naive_interpolated_coord_inheritance(m_p_coord: torch.Tensor, m_c_coord: torch.Tensor,
                                         m_p_start_coord: torch.Tensor, interp_type: str) -> torch.Tensor:
    """
    For each element in the child volume, compute the parent coordinates and features around its parent.
    We do this by unfolding parent coordinates and features to get each parent's neighbourhood and use
    repeat_interleave to obtain neighbourhoods in the shape of the child volume.
    Args:
        m_p_coord: Current parent coordinates of shape (B, D, H, W, 3) or (B, H, W, 2)
        m_c_coord: Child coordinates of shape (B, d, h, w, 3) or (B, h, w, 2)
        m_p_start_coord: Starting parent coordinates of shape (B, D, H, W, 3) or (B, H, W, 2).
                        Only necessary when inheriting coordinates relative to parent's starting position.
        interp_type:
    Returns:
        The inherited coordinates of each child. Shape: (B, d, h, w, 3) or (B, h, w, 2)
    """
    if m_p_start_coord.shape[-1] == 3:
        corner = m_p_start_coord[:, 0, 0, 0]
        scaling = torch.full_like(corner, -1., device=corner.device) / corner
        scaling = scaling[:, None, None, None]
    else:
        corner = m_p_start_coord[:, 0, 0]
        scaling = torch.full_like(corner, -1., device=corner.device) / corner
        scaling = scaling[:, None, None]

    m_p_coord_scaled = m_p_coord * scaling
    m_c_coord_scaled = m_c_coord * scaling
    m_inh_coord_scaled = nn.functional.grid_sample(m_p_coord_scaled.moveaxis(-1, 1),
                                                   m_c_coord_scaled.flip(-1),
                                                   mode=interp_type,
                                                   padding_mode="border",
                                                   align_corners=True,
                                                   )
    m_inh_coord_scaled = m_inh_coord_scaled.moveaxis(1, -1)
    m_inh_coord = m_inh_coord_scaled / scaling
    return m_inh_coord


def nn_interpolation_repeat_interleave(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """ This function creates a nearest neighbour interpolation by using repeat_interleave spatially on
    the parent tensor. This assumes the parent array was pooled by an integer factor from the child array,
    which should result on integer spatial_size_ratio (or close to integer if child tensor shape
    was not a multiple of the pooling size.
      In the case child tensor shape is an exact multiple of the parent tensor shape, a 'view' of the parent tensor
    is provided leading to reduced memory usage. """
    p_spatial_sizes = p.shape[1:-1]
    c_spatial_sizes = c.shape[1:-1]
    if p_spatial_sizes == c_spatial_sizes:
        return p
    spatial_size_ratio = torch.ceil(torch.tensor(c_spatial_sizes) / torch.tensor(p_spatial_sizes)).long().tolist()
    if len(c_spatial_sizes) == 3:
        p_upsampled = (p.repeat_interleave(spatial_size_ratio[2], dim=3)
                       .repeat_interleave(spatial_size_ratio[1], dim=2)
                       .repeat_interleave(spatial_size_ratio[0], dim=1))
        if p_upsampled.shape[1] > c.shape[1] or p_upsampled.shape[2] > c.shape[2] or p_upsampled.shape[3] > c.shape[3]:
            # If children size is not divisible by parent size, we trim the edge of the repeated volume
            p_upsampled = p_upsampled[:, :c_spatial_sizes[-3], :c_spatial_sizes[-2], :c_spatial_sizes[-1]]
    else:
        p_upsampled = (p.repeat_interleave(spatial_size_ratio[1], dim=2)
                       .repeat_interleave(spatial_size_ratio[0], dim=1))
        if p_upsampled.shape[1] > c.shape[1] or p_upsampled.shape[2] > c.shape[2]:
            # If children size is not divisible by parent size, we trim the edge of the repeated volume
            p_upsampled = p_upsampled[:, :c_spatial_sizes[0], :c_spatial_sizes[1]]
    return p_upsampled


def parent_unfold_interleave(m_c_feat: torch.Tensor, m_p_coord: torch.Tensor, m_p_feat: torch.Tensor, k: int, m_p_start_coord: Optional[torch.Tensor] = None):
    batch_size = m_p_feat.shape[0]
    p_spatial_sizes = m_p_coord.shape[1:-1]
    c_spatial_sizes = m_c_feat.shape[1:-1]
    spatial_size_ratio = torch.ceil(torch.tensor(c_spatial_sizes) / torch.tensor(p_spatial_sizes)).long().tolist()

    unfold = unfold3D if len(c_spatial_sizes) == 3 else unfold2D
    m_p_neigh_coord = unfold(m_p_coord, kernel_size=k, pad_value=0, mode="replicate")  # (batch, D,H,W, K*K*K, c)
    m_p_neigh_feat = unfold(m_p_feat, kernel_size=k, pad_value=0)  # (batch, D,H,W, K*K*K, c)
    m_p_pad_mask = torch.zeros(p_spatial_sizes, dtype=torch.bool, device=m_p_feat.device)
    m_p_neigh_pad_mask = unfold(m_p_pad_mask[None, ..., None], k, pad_value=1)  # (1, D,H,W, K*K*K, 1)
    m_p_neigh_pad_mask = m_p_neigh_pad_mask.squeeze(-1)

    if len(c_spatial_sizes) == 3:  # We chose to treat the 3d case with an if statement for readability :)
        m_p_neigh_coord = (m_p_neigh_coord.repeat_interleave(spatial_size_ratio[2], dim=3)
                           .repeat_interleave(spatial_size_ratio[1], dim=2)
                           .repeat_interleave(spatial_size_ratio[0], dim=1))
        m_neigh_feat = (m_p_neigh_feat.repeat_interleave(spatial_size_ratio[2], dim=3).
                        repeat_interleave(spatial_size_ratio[1], dim=2)
                        .repeat_interleave(spatial_size_ratio[0], dim=1))
        m_neigh_pad_mask = (m_p_neigh_pad_mask.repeat_interleave(spatial_size_ratio[2], dim=3)
                            .repeat_interleave(spatial_size_ratio[1], dim=2)
                            .repeat_interleave(spatial_size_ratio[0], dim=1))
        if (m_neigh_feat.shape[1] > m_c_feat.shape[1]
                or m_neigh_feat.shape[2] > m_c_feat.shape[2] or m_neigh_feat.shape[3] > m_c_feat.shape[3]):
            # If children size is not divisible by parent size, we trim the edge of the repeated volume
            m_p_neigh_coord = m_p_neigh_coord[:, :c_spatial_sizes[-3], :c_spatial_sizes[-2], :c_spatial_sizes[-1]]
            m_neigh_feat = m_neigh_feat[:, :c_spatial_sizes[-3], :c_spatial_sizes[-2], :c_spatial_sizes[-1]]
            m_neigh_pad_mask = m_neigh_pad_mask[:, :c_spatial_sizes[-3], :c_spatial_sizes[-2], :c_spatial_sizes[-1]]
    else:
        m_p_neigh_coord = (m_p_neigh_coord.repeat_interleave(spatial_size_ratio[1], dim=2)
                           .repeat_interleave(spatial_size_ratio[0], dim=1))
        m_neigh_feat = (m_p_neigh_feat.repeat_interleave(spatial_size_ratio[1], dim=2).
                        repeat_interleave(spatial_size_ratio[0], dim=1))
        m_neigh_pad_mask = (m_p_neigh_pad_mask.repeat_interleave(spatial_size_ratio[1], dim=2)
                            .repeat_interleave(spatial_size_ratio[0], dim=1))
        if m_neigh_feat.shape[1] > m_c_feat.shape[1] or m_neigh_feat.shape[2] > m_c_feat.shape[2]:
            # If children size is not divisible by parent size, we trim the edge of the repeated volume
            m_p_neigh_coord = m_p_neigh_coord[:, :c_spatial_sizes[-2], :c_spatial_sizes[-1]]
            m_neigh_feat = m_neigh_feat[:, :c_spatial_sizes[-2], :c_spatial_sizes[-1]]
            m_neigh_pad_mask = m_neigh_pad_mask[:, :c_spatial_sizes[-2], :c_spatial_sizes[-1]]

    m_neigh_pad_mask = m_neigh_pad_mask.tile((batch_size, *[1]*len(m_neigh_pad_mask.shape[1:])))  # Shape: (batch, d, h, w, K*K*K, 1)

    # This should ideally not be needed, but for MHA seems to suffer from inaccuracies otherwise
    m_neigh_feat[m_neigh_pad_mask] = 0.0
    # m_p_neigh_coord[m_neigh_pad_mask] = 0.0
    m_neigh_pad_mask = m_neigh_pad_mask.unsqueeze(-2)                                    # Shape: (batch, d, h, w, 1, K*K*K)
    m_neigh_pad_mask = m_neigh_pad_mask.float()  # Numerical stability in att layers

    return m_neigh_feat, m_p_neigh_coord, m_neigh_pad_mask


class InheritanceHandler(nn.Module):
    def __init__(self, num_spatial_dims: int,
                 coord_inh_type: str = "none",
                 feature_inh_type: str = "none",
                 par_map_layer: Optional[Union[nn.Conv2d, nn.Conv3d]] = None,
                 coord_inh_kernel_size: Optional[int] = None,
                 feat_kernel_size: Optional[int] = None,
                 pos_encoder: Optional[PosEncodingBase] = None,
                 coord_inh_layer: Optional[AttentionWeightedLayer] = None,
                 feature_inh_layer: Optional[CrossAttentionLayer] = None,
                 disp_refinement_layer: Optional[CrossAttentionLayer] = None):
        super().__init__()
        self.num_spatial_dims = num_spatial_dims
        self.coord_inh_type = coord_inh_type
        self.feature_inh_type = feature_inh_type
        self.par_mapping_layer = par_map_layer
        self.coord_inh_k = coord_inh_kernel_size
        self.feat_k = feat_kernel_size
        self.pos_encoder = pos_encoder
        self.coord_inh_layer = coord_inh_layer
        self.feature_inh_layer = feature_inh_layer
        self.perform_coord_refine = disp_refinement_layer is not None
        self.disp_refinement_layer = disp_refinement_layer
        self.p_requires_mapping = (coord_inh_type == "mha" or feature_inh_type != "none" or self.perform_coord_refine)

    def forward(self, c_feat: torch.Tensor, c_coord: torch.Tensor,
                p_feat: torch.Tensor, p_coord: torch.Tensor, p_start_coord: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.p_requires_mapping:
            p_feat = self.par_mapping_layer(p_feat.moveaxis(-1, 1)).moveaxis(1, -1)
        c_coord_inh, unfolded_items = self._inherit_coords(c_feat, c_coord, p_feat, p_coord, p_start_coord)
        c_feat_new__ = self._inherit_features(c_feat, c_coord_inh, p_feat, p_coord, p_start_coord, unfolded_items)
        c_coord_new = self._refine_coords(c_feat, c_coord_inh, p_feat, p_coord, c_feat_new__, unfolded_items)
        return c_feat_new__, c_coord_new, c_coord_inh

    def _inherit_coords(self, c_feat: torch.Tensor, c_coord: torch.Tensor,
                        p_feat: Optional[torch.Tensor], p_coord: torch.Tensor,
                        p_start_coord: Optional[torch.Tensor] = None):
        unfolded_items = None
        if self.coord_inh_type == "nearest":
            c_coord_new = nn_interpolation_repeat_interleave(p_coord, c_coord)
        elif self.coord_inh_type in {"bilinear", "bicubic"}:
            c_coord_new = naive_interpolated_coord_inheritance(p_coord, c_coord, p_start_coord, self.coord_inh_type)
        elif self.coord_inh_type == "nearest_relative":
            c_nn_p_coord = nn_interpolation_repeat_interleave(p_coord, c_coord)
            c_nn_p_start_coord = nn_interpolation_repeat_interleave(p_start_coord, c_coord)
            c_p_start_relative = c_coord - c_nn_p_start_coord
            c_coord_new = c_nn_p_coord + c_p_start_relative
        elif self.coord_inh_type == "mha":
            p_neigh_feat, p_neigh_coord, pad_mask = parent_unfold_interleave(c_coord, p_coord, p_feat, self.coord_inh_k)
            p_neigh_relative_coord = p_neigh_coord - c_coord.unsqueeze(-2)
            p_neigh_feat = self.pos_encoder(p_neigh_feat, p_neigh_relative_coord)
            c_coord_new = self.coord_inh_layer(c_feat.reshape((-1, 1, c_feat.shape[-1])),
                                               p_neigh_feat.view((-1, *p_neigh_feat.shape[-2:])),
                                               p_neigh_coord.reshape((-1, *p_neigh_coord.shape[-2:])),
                                               pad_mask.view((-1, pad_mask.shape[-2], pad_mask.shape[-1])))
            c_coord_new = c_coord_new.view(c_coord.shape)
            unfolded_items = (p_neigh_feat, p_neigh_coord, pad_mask)
        elif self.coord_inh_type == "bspline":
            c_spatial_dims = c_coord.shape[1:-1]
            p_spatial_dims = p_coord.shape[1:-1]
            spatial_aspect_ratio = torch.ceil(
                torch.tensor(c_spatial_dims[0]) / torch.tensor(p_spatial_dims[1])).long().item()
            assert self.coord_inh_k >= spatial_aspect_ratio  # Kernel size must be larger than the aspect ratio

            kernel_size = [self.coord_inh_k, ] * len(c_spatial_dims)
            kernel = set_kernel(stride=kernel_size)
            parent_disp = p_coord - p_start_coord
            # TODO: let the user set it or define it in consistency with the kernel size?
            stride = spatial_aspect_ratio + 1
            m_inh_disp = bspline_interpolation(parent_disp.moveaxis(-1, 1), shape=c_coord.shape[1:-1], kernel=kernel,
                                               transpose=True, stride=stride).moveaxis(1, -1).contiguous()

            c_coord_new = c_coord + m_inh_disp
        else:
            raise ValueError("Unknown coord inheritance type")
        return c_coord_new, unfolded_items

    def _inherit_features(self, c_feat: torch.Tensor, c_coord: torch.Tensor,
                          p_feat: torch.Tensor, p_coord: torch.Tensor, p_start_coord: torch.Tensor,
                          unfolded_items: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        if self.feature_inh_type in {"None", "none"}:
            c_feat__ = c_feat.reshape((-1, 1, c_feat.shape[-1]))
            return c_feat__
        elif self.feature_inh_type == "nearest":
            features = nn_interpolation_repeat_interleave(p_feat, c_coord)
            c_feat = c_feat + features
            c_feat__ = c_feat.reshape((-1, 1, c_feat.shape[-1]))
            return c_feat__
        elif self.feature_inh_type in {"bilinear", "bicubic"}:
            p_spatial_sizes = p_coord.shape[1:-1]
            c_spatial_sizes = c_feat.shape[1:-1]
            spatial_size_ratio = torch.ceil(torch.tensor(c_spatial_sizes) / torch.tensor(p_spatial_sizes)).long().tolist()
            mode = self.feature_inh_type if len(p_spatial_sizes) == 2 else "trilinear"
            features = nn.functional.interpolate(p_feat.moveaxis(-1, 1),
                                                 scale_factor=spatial_size_ratio,
                                                 mode=mode,
                                                 align_corners=True)
            features = features.moveaxis(1, -1)
            features = features[:, :c_spatial_sizes[0], :c_spatial_sizes[1], :c_spatial_sizes[2], :] \
                if len(c_spatial_sizes) == 3 else features[:, :c_spatial_sizes[0], :c_spatial_sizes[1], :]
            c_feat = c_feat + features
            c_feat__ = c_feat.reshape((-1, 1, c_feat.shape[-1]))
            return c_feat__
        elif self.feature_inh_type == "mha":
            if unfolded_items is None:
                unfolded_items = parent_unfold_interleave(c_coord, p_coord, p_feat, self.feat_k)
            p_neigh_feat, p_neigh_coord, pad_mask = unfolded_items
            p_neigh_relative_coord = p_neigh_coord - c_coord.unsqueeze(-2)
            p_neigh_feat = self.pos_encoder(p_neigh_feat, p_neigh_relative_coord)
            c_feat__ = c_feat.reshape((-1, 1, c_feat.shape[-1]))
            features__ = self.feature_inh_layer(c_feat__,
                                                p_neigh_feat.view((-1, *p_neigh_feat.shape[-2:])),
                                                pad_mask.view((-1, pad_mask.shape[-2], pad_mask.shape[-1])))
            return c_feat__ + features__
        else:
            raise ValueError("Unknown coord inheritance type")

    def _refine_coords(self, c_feat: torch.Tensor, c_coord: torch.Tensor,
                       p_feat: torch.Tensor, p_coord: torch.Tensor,
                       features__: Optional[torch.Tensor] = None,
                       unfolded_items: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        if not self.perform_coord_refine:
            return c_coord
        if features__ is None:
            if unfolded_items is None:
                unfolded_items = parent_unfold_interleave(c_coord, p_coord, p_feat, self.k)
            p_neigh_feat, p_neigh_coord, pad_mask = unfolded_items
            p_neigh_relative_coord = p_neigh_coord - c_coord.unsqueeze(-2)
            p_neigh_feat = self.pos_encoder(p_neigh_feat, p_neigh_relative_coord)
            features__ = self.feature_inh_layer(c_feat.view((-1, 1, c_feat.shape[-1])),
                                                p_neigh_feat.view((-1, *p_neigh_feat.shape[-2:])),
                                                pad_mask.view((-1, pad_mask.shape[2], pad_mask.shape[3])))
        c_coord_new = c_coord + self.disp_refinement_layer(features__).reshape(c_coord.shape)
        return c_coord_new
