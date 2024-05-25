from typing import Optional, List, Tuple
import torch
torch.manual_seed(42)
from torch import nn

from models.layers import SelfAttentionLayer, CrossAttentionLayer, CustomOut, DoubleConv, AttentionWeightedLayer
from models.inheritance import InheritanceHandler
from utils.unfoldNd import unfoldNd, unfold3D, unfold2D
from utils.utils import make_coordinate_tensor, calculate_closest_indices, make_coordinate_tensor_like
from utils.pos_encoding import CAPE2D, CAPE2D1D, PosEncodingFourier, PosEncodingNeRF, PosEncodingNone


class ConvFeatures(nn.Module):
    def __init__(self, in_channels, channels, pool_sizes, dropout=0.0, **kwargs):
        super().__init__()
        self.embed = DoubleConv(in_channels, channels[0], dropout=dropout, **kwargs)
        self.embed2 = DoubleConv(channels[0], channels[0], dropout=dropout, **kwargs)
        self.downs = nn.ModuleList([DoubleConv(channels[i], channels[i + 1], dropout=dropout, **kwargs)
                                    for i in range(len(channels) - 1)])
        self.downs2 = nn.ModuleList([DoubleConv(channels[i + 1], channels[i + 1], dropout=dropout, **kwargs)
                                    for i in range(len(channels) - 1)])
        self.pool_sizes = pool_sizes
        assert len(pool_sizes) == len(self.downs)
        out_ch = [channels[i] for i in range(len(channels) - 1, -1, -1)]
        self.out_sizes = out_ch

    def forward(self, x: torch.Tensor, coord_batch: torch.Tensor = None):
        spatial_dims = len(x.shape[2:])
        pooling_func = nn.functional.avg_pool2d if spatial_dims == 2 else nn.functional.avg_pool3d
        out = self.embed(x)
        out = self.embed2(out) + out
        coords = [coord_batch]
        coord = coord_batch
        outs = [out]
        for down, down2, pool_size in zip(self.downs, self.downs2, self.pool_sizes):
            # Conv layer and pooling
            out = pooling_func(out, pool_size, ceil_mode=True)
            out = down(out)
            out = down2(out) + out
            outs.append(out)

            # Coords of each element in the layer
            if coord_batch is not None:
                coord = pooling_func(coord.moveaxis(-1, 1), pool_size, ceil_mode=True).moveaxis(1, -1)
            coords.append(coord)

        return outs[::-1], coords[::-1]


class MultiScaleTransformer(nn.Module):
    def __init__(self, channels, att_nhead=1, dropout=0.0, num_spatial_dims=2, **kwargs):
        super().__init__()
        # Kernel dimensions
        self.parent_k = kwargs["k_size_parent"][::-1]  # This decoder operates coarse->fine
        self.cross_k = kwargs["k_size_cross"][::-1]  # This decoder operates coarse->fine
        self.sibling_k = kwargs["k_size_sibling"][::-1]  # This decoder operates coarse->fine
        self.inheritance_k = kwargs["k_size_inh"][::-1]  # This decoder operates coarse->fine
        self.layer_iters = kwargs["iters"][::-1]  # This decoder operates coarse->fine

        pos_enc = kwargs["pos_econding"]
        if pos_enc == "none":
            pos_enc_class = PosEncodingNone
        elif pos_enc == "fourier":
            pos_enc_class = PosEncodingFourier
        elif pos_enc == "cape":
            pos_enc_class = CAPE2D if num_spatial_dims == 2 else CAPE2D1D
        elif pos_enc == "nerf":
            pos_enc_class = PosEncodingNeRF
        else:
            raise ValueError('What pos encoding is this?')
        self.pos_enc = [pos_enc_class(num_spatial_dims, c) for c in channels]

        self.inh_coord_interp_type = kwargs["inh_coord_interp_type"]
        self.inh_feat_interp_type = kwargs["inh_feat_interp_type"]
        self.inh_refinement = kwargs["inh_refinement"]

        # Self-to-parent parameters
        par_layers_needed = self.inh_refinement or self.inh_feat_interp_type != "none"
        self.par_layers = [CrossAttentionLayer(c, att_nhead, dim_feedforward=c*2, dropout=dropout) if par_layers_needed else None
                           for c, k in zip(channels[1:], self.parent_k)]
        self.par_out_layers = [CustomOut(c, num_spatial_dims) if self.inh_refinement else None
                               for c, k in zip(channels[1:], self.parent_k)]
        conv_class = nn.Conv3d if num_spatial_dims == 3 else nn.Conv2d
        mapping_layers_needed = self.inh_refinement or self.inh_coord_interp_type == "mha" or self.inh_feat_interp_type != "none"
        self.par_map_layers = [conv_class(c_in, c_out, kernel_size=1) if mapping_layers_needed else None
                               for c_in, c_out in zip(channels[:-1], channels[1:])]
        self.coord_inh_layers = [AttentionWeightedLayer(c, c) if self.inh_coord_interp_type == "mha" else None
                                 for c, k in zip(channels[1:], self.inheritance_k)]
        self.inh_handler = nn.ModuleList([InheritanceHandler(num_spatial_dims,
                                                             coord_inh_type=self.inh_coord_interp_type,
                                                             feature_inh_type=self.inh_feat_interp_type,
                                                             par_map_layer=self.par_map_layers[i],
                                                             coord_inh_kernel_size=self.inheritance_k[i],
                                                             feat_kernel_size=self.parent_k[i],
                                                             pos_encoder=self.pos_enc[i+1],
                                                             coord_inh_layer=self.coord_inh_layers[i],
                                                             feature_inh_layer=self.par_layers[i],
                                                             disp_refinement_layer=self.par_out_layers[i])
                                          for i in range(len(channels[1:]))])

        self.global_bottleneck = bool(kwargs["global_bottleneck"])
        self.global_deform = None
        self.global_deform_out = None
        if self.global_bottleneck:
            # Global attention (superglue)
            self.global_deform = nn.TransformerDecoderLayer(channels[0], att_nhead, dim_feedforward=channels[0]*2,
                                                            dropout=dropout, batch_first=True)
            self.global_deform_out = nn.Sequential(nn.Linear(channels[0], out_features=channels[0]),
                                                   nn.LeakyReLU(negative_slope=0.1), nn.Linear(channels[0], num_spatial_dims))

        # Self-to-sibling parameters
        self.sib_layers = nn.ModuleList([CrossAttentionLayer(c, att_nhead, dim_feedforward=c*2, dropout=dropout) if k > 0 else None
                                           for c, k in zip(channels, self.cross_k)])
        # self.sib_layers = nn.ModuleList([SelfAttentionLayer(c, att_nhead, dim_feedforward=c*2,
        #                                                     kernel_size=k, num_spatial_dims=num_spatial_dims) if k > 0 else None
        #                                 for c, k in zip(channels, self.sibling_k)])
        self.sib_out_layers = nn.ModuleList([CustomOut(c, num_spatial_dims) if k > 0 else None
                                             for c, k in zip(channels, self.sibling_k)])
        # Self-to-crossImage parameters
        self.cross_layers = nn.ModuleList([CrossAttentionLayer(c, att_nhead, dim_feedforward=c*2, dropout=dropout) if k > 0 else None
                                           for c, k in zip(channels, self.cross_k)])
        self.cross_out_layers = nn.ModuleList([CustomOut(c, num_spatial_dims) if k > 0 else None
                                               for c, k in zip(channels, self.cross_k)])

        self.out_feat = channels

    def forward(self,
                m_feat_list: List[torch.Tensor], f_feat_list: List[torch.Tensor],
                m_coord_list: List[torch.Tensor], f_coord_list: List[torch.Tensor], factor=1.0):
        # We move channel dimension to last dimension because that is what transformers like.
        m_feat_start = m_feat_list[0].moveaxis(1, -1)
        out__ = m_feat_start.reshape((-1, 1, m_feat_start.shape[-1]))
        m_coord = m_coord_list[0]
        batch_size, coord_size = m_coord.shape[0], m_coord.shape[-1]

        inh_coords = []
        par_coords = []
        sib_coords = []
        cross_coords = []
        final_coords = []
        out_features = []
        for i in range(len(m_feat_list)):
            cross_c_list = []
            sib_c_list = []

            if i == 0:
                # At the first iteration there are no parents' so we don't inherit anything
                inh_coords.append(m_coord.clone())
                par_coords.append(m_coord.clone())
            else:
                # -----------  Displacement inheritance  ---------------
                # When moving to finer layer, the child coordinates take into consideration the parent's current coordinates
                m_c_feat = m_feat_list[i].moveaxis(1, -1)
                m_p_feat = m_feat_list[i-1].moveaxis(1, -1)
                m_c_start_coord = m_coord_list[i]
                m_p_start_coord = m_coord_list[i-1]
                m_c_spatial_sizes, c_channels = m_c_feat.shape[1:-1], m_c_feat.shape[-1]
                m_p_spatial_sizes, p_channels = m_p_feat.shape[1:-1], m_p_feat.shape[-1]

                m_p_feat = out__.reshape((batch_size, *m_p_spatial_sizes, p_channels))
                out__, m_coord, m_coord_inh = self.inh_handler[i-1](m_c_feat, m_c_start_coord, m_p_feat, m_coord, m_p_start_coord)
                inh_coords.append(m_coord_inh)
                par_coords.append(m_coord.clone())

            # We move channel dimension to last dimension because that is what transformers expect
            f_feat = f_feat_list[i].moveaxis(1, -1)
            f_coord = f_coord_list[i]
            spatial_sizes, channel_size = f_feat.shape[1:-1], f_feat.shape[-1]

            par_out__ = out__
            for _ in range(self.layer_iters[i]):  # We can repeat cross-image attention several times
                out__ = par_out__

                if i == 0 and self.global_bottleneck:
                    # ----------- Global deformation  ----------------
                    # Apply coordinate encoding to starting features
                    m_feat_ = out__.reshape((batch_size, -1, channel_size))
                    f_feat_ = f_feat.reshape((batch_size, -1, channel_size))
                    m_coord_ = m_coord.reshape((batch_size, -1, coord_size))
                    f_coord_ = f_coord.reshape((batch_size, -1, coord_size))
                    m_feat_ = self.pos_enc[i](m_feat_, m_coord_.detach())
                    f_feat_ = self.pos_enc[i](f_feat_, f_coord_.detach())

                    # Global deformation (self-attention on moving points -> cross-attention to fixed points)
                    cross_out_ = self.global_deform(m_feat_, f_feat_)
                    cross_out__ = cross_out_.reshape((-1, 1, channel_size))
                    cross_displ = self.global_deform_out(cross_out__.squeeze(1)).reshape((batch_size, *spatial_sizes, coord_size))
                    m_coord = m_coord + cross_displ
                    cross_c_list.append(m_coord.clone())
                    sib_c_list.append(m_coord.clone())
                else:
                    # -----------  Cross-Image Attention  ------------------
                    # Define fixed image neighbourhoods for every moving image feature
                    if self.cross_k[i] > 0:  # We can use kernel size 0 to imply we don't want a layer
                        f_neigh_feat, f_relative_coord, pad_mask_ = self.compute_cross_image_neighbours(f_feat, f_coord, m_coord, self.cross_k[i])
                        f_neigh_feat = self.pos_enc[i](f_neigh_feat, f_relative_coord)
                        # Every moving image feature does cross attention to every surrounding fixed image neighbour
                        cross_out__ = self.cross_layers[i](out__,
                                                           f_neigh_feat.view((-1, f_neigh_feat.shape[2], channel_size)),
                                                           pad_mask_.view((-1, pad_mask_.shape[2], pad_mask_.shape[3])))
                        # out__ = cross_out__ + out__  # Skip-connection
                        cross_displ = self.cross_out_layers[i](cross_out__.squeeze(1)).reshape((batch_size, *spatial_sizes, coord_size))
                        cross_displ = cross_displ * factor
                        m_coord = m_coord + cross_displ
                    cross_c_list.append(m_coord.clone())
                    # -----------  Sibling Attention  ------------------
                    # Define moving image neighbourhoods for every moving image feature
                    if self.sibling_k[i] > 0:  # We can use kernel size 0 to imply we don't want a layer
                        # out = out__.reshape((batch_size, *spatial_sizes, channel_size))
                        # out = self.pos_enc[i](out, m_coord)
                        # sib_out = self.sib_layers[i](out)
                        # sib_out__ = sib_out.reshape((-1, 1, channel_size))
                        out = out__.reshape((batch_size, *spatial_sizes, channel_size))
                        m_neigh_feat_, m_relative_coord_, pad_mask_ = self.compute_moving_neighbours(out, m_coord, self.sibling_k[i])
                        m_neigh_feat_ += self.pos_enc[i](m_neigh_feat_, m_relative_coord_)
                        # Every moving image feature does cross attention to every surrounding moving image neighbour
                        sib_out__ = self.sib_layers[i](out__,  # 1 x neigh_size cross-att
                                                       m_neigh_feat_.reshape((-1, m_neigh_feat_.shape[2], channel_size)),
                                                       pad_mask_.reshape((-1, pad_mask_.shape[2], pad_mask_.shape[3])))
                        # out__ = sib_out__ + out__  # Skip-connection
                        sib_displ = self.sib_out_layers[i](sib_out__.squeeze(1)).reshape((batch_size, *spatial_sizes, coord_size))
                        # sib_displ = sib_displ * factor
                        m_coord = m_coord + sib_displ
                    sib_c_list.append(m_coord.clone())
            cross_coords.append(cross_c_list)
            sib_coords.append(sib_c_list)
            final_coords.append(m_coord)

        coord_components = {"start": m_coord_list, "inh": inh_coords, "par": par_coords,
                            "cross": cross_coords, "sib": sib_coords, "final": final_coords}
        return coord_components, out_features

    def compute_moving_neighbours(self, m_feat: torch.Tensor, m_coord: torch.Tensor, k: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each element in the child volume, compute the coordinates and features of neighbours around a K-sized kernel.
        We do this by unfolding coordinates and features to get each element's neighbourhood and use
        repeat_interleave to obtain neighbourhoods in the shape of the volume.
        Args:
            m_feat: Features of shape (B, D, H, W, C) or (B, H, W, C)
            m_coord: Coordinates of shape (B, D, H, W, 3) or (B, H, W, 2)
            k: Neighbourhood kernel size
        Returns:
            For each child, its parent's neighbourhood's features. Shape: (B, d*h*w, K*K*K, C) or (B, h*w, K*K, C)
            For each child, its parent's relative coordinates to every
                neighbourhood parent. Shape: (B, d*h*w, K*K*K, 3) or (B, h*w, K*K, 2)
            For each child, binary mask indicating which neighbours
                are padding. Shape: (B, d*h*w, 1, K*K*K) or (B, h*w, 1, K*K)
        """

        batch_size, spatial_sizes, channel_size, coord_size = \
            m_feat.shape[0], m_feat.shape[1:-1], m_feat.shape[-1], m_coord.shape[-1]
        spatial_1d_size = torch.prod(torch.tensor(m_coord.shape[1:-1])).item()

        unfold = unfold3D if len(spatial_sizes) == 3 else unfold2D
        pad_value = -1
        m_neigh_coord = unfold(m_coord, k, pad_value=pad_value)  # (batch, d, h, w, K*K*K, c)
        m_neigh_coord_ = m_neigh_coord.view((batch_size, spatial_1d_size, -1, coord_size))  # (batch, d*h*w, K*K*K, c)
        m_neigh_feat = unfold(m_feat, k, pad_value=pad_value)  # (batch, d, h, w, K*K*K, c)
        m_neigh_feat_ = m_neigh_feat.view((batch_size, spatial_1d_size, -1, channel_size))  # (batch, d*h*w, K*K*K, c)
        pad_mask = torch.zeros(spatial_sizes, dtype=torch.bool, device=m_feat.device)
        pad_mask = unfold(pad_mask[None, ..., None], k, pad_value=1).squeeze(-1)  # (1, d*h*w, K*K*K)
        pad_mask_ = pad_mask.view((1, spatial_1d_size, -1))  # (d*h*w, K*K*K)
        pad_mask_ = pad_mask_.tile((batch_size, 1, 1))                                      # Shape: (batch, d*h*w, K*K*K)

        m_center_ = m_coord.view(batch_size, spatial_1d_size, 1, coord_size)
        m_relative_coord_ = m_center_ - m_neigh_coord_  # (batch, d*h*w, K*K*K, c)

        # This should not be needed if att layers correctly mask out padded areas, but for some reason it does not
        m_neigh_feat_[pad_mask_] = 0.0
        m_relative_coord_[pad_mask_] = 0.0
        pad_mask_ = pad_mask_.unsqueeze(2)                               # Shape: (batch, d*h*w, 1, K*K*K)
        pad_mask_ = pad_mask_.float()  # Numerical stability in att layers

        return m_neigh_feat_, m_relative_coord_, pad_mask_

    def compute_cross_image_neighbours(self, f_feat: torch.Tensor, f_coord: torch.Tensor, m_coord: torch.Tensor,
                                       k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each element in the moving volume, compute the coordinates and features of neighbours in the
        fixed volume around a K-sized kernel.
        We do this by unfolding coordinates and features to get each element's neighbourhood and use
        repeat_interleave to obtain neighbourhoods in the shape of the volume.
        Args:
            f_feat: Features of shape (B, D, H, W, C) or (B, H, W, C)
            f_coord: Coordinates of shape (B, D, H, W, 3) or (B, H, W, 2)
            m_coord: Coordinates of shape (B, D, H, W, 3) or (B, H, W, 2)
            k: size of neighbourhood kernel

        Returns:
            For each element in the moving image, its fixed image
                neighbourhood's features. Shape: (B, d*h*w, K*K*K, C) or (B, h*w, K*K, C)
            For each element in the moving image, its neighbourhood's relative coordinates to every
                fixed image neighbour. Shape: (B, d*h*w, K*K*K, 3) or (B, h*w, K*K, 2)
            For each element in the moving image, binary mask indicating which neighbours
                are padding. Shape: (B, d*h*w, 1, K*K*K) or (B, h*w, 1, K*K)
        """
        batch_size, m_spatial_sizes, channel_size, coord_size = \
            m_coord.shape[0], m_coord.shape[1:-1], f_feat.shape[-1], m_coord.shape[-1]
        f_spatial_sizes = f_coord.shape[1:-1]
        f_spatial_1d_size = torch.prod(torch.tensor(m_spatial_sizes)).item()
        m_spatial_1d_size = torch.prod(torch.tensor(m_coord.shape[1:-1])).item()

        unfold = unfold3D if len(f_spatial_sizes) == 3 else unfold2D
        pad_value = 0
        f_neigh_coord = unfold(f_coord, k, pad_value=pad_value)  # (batch, d, h, w, K*K*K, c)
        f_neigh_feat = unfold(f_feat, k, pad_value=pad_value)  # (batch, d, h, w, K*K*K, c)
        pad_mask = torch.zeros(m_spatial_sizes, dtype=torch.bool, device=f_feat.device)
        pad_mask = unfold(pad_mask[None, ..., None], k, pad_value=1).squeeze(-1)  # (1, d, h, w, K*K*K)
        pad_mask = pad_mask.expand(batch_size, *[-1]*len(m_spatial_sizes), -1)  # (batch, d, h, w, K*K*K)

        # For each moving point, find the closest fixed point indices
        closest_f_nd_idx = calculate_closest_indices(m_coord, f_coord, round_to_closest=(k % 2) == 1)  # (batch, d, h, w, c)
        # Each moving point should index whichever fixed point neighbourhood they are closest to
        b, _ = torch.meshgrid(torch.arange(0, batch_size, device=m_coord.device),
                              torch.arange(0, f_spatial_1d_size, device=m_coord.device), indexing="ij")
        b__ = b.reshape(-1)
        closest_f_nd_idx_x__ = closest_f_nd_idx[..., 0].reshape(-1)
        closest_f_nd_idx_y__ = closest_f_nd_idx[..., 1].reshape(-1)
        if len(m_spatial_sizes) == 2:
            closest_f_neigh_coord__ = f_neigh_coord[b__, closest_f_nd_idx_x__, closest_f_nd_idx_y__]
            closest_f_neigh_feat__ = f_neigh_feat[b__, closest_f_nd_idx_x__, closest_f_nd_idx_y__]
            pad_mask__ = pad_mask[b__, closest_f_nd_idx_x__, closest_f_nd_idx_y__]
        else:
            closest_f_nd_idx_z__ = closest_f_nd_idx[..., 2].reshape(-1)
            closest_f_neigh_coord__ = f_neigh_coord[b__, closest_f_nd_idx_x__, closest_f_nd_idx_y__, closest_f_nd_idx_z__]
            closest_f_neigh_feat__ = f_neigh_feat[b__, closest_f_nd_idx_x__, closest_f_nd_idx_y__, closest_f_nd_idx_z__]
            pad_mask__ = pad_mask[b__, closest_f_nd_idx_x__, closest_f_nd_idx_y__, closest_f_nd_idx_z__]

        closest_f_neigh_coord_ = closest_f_neigh_coord__.view((batch_size, f_spatial_1d_size, -1, coord_size))  # (batch, d*h*w, K*K*K, c)
        closest_f_neigh_feat_ = closest_f_neigh_feat__.view((batch_size, f_spatial_1d_size, -1, channel_size))  # (batch, d*h*w, K*K*K, c)
        closest_f_neigh_pad_mask_ = pad_mask__.view((batch_size, f_spatial_1d_size, -1))  # (batch, d*h*w, K*K*K)

        m_center_ = m_coord.view((batch_size, m_spatial_1d_size, 1, coord_size))      # (batch, d*h*w, 1, c)
        m_neigh_relative_f_coord = m_center_ - closest_f_neigh_coord_                 # (batch, d*h*w, K*K*K, c)

        # This should not be needed if att layers correctly mask out padded areas, but for some reason it does not
        closest_f_neigh_feat_[closest_f_neigh_pad_mask_] = 0.0
        m_neigh_relative_f_coord[closest_f_neigh_pad_mask_] = 0.0
        closest_f_neigh_pad_mask_ = closest_f_neigh_pad_mask_.unsqueeze(2)            # Shape: (batch, d*h*w, 1, K*K*K)
        closest_f_neigh_pad_mask_ = closest_f_neigh_pad_mask_.float()  # Numerical stability in att layers

        return closest_f_neigh_feat_, m_neigh_relative_f_coord, closest_f_neigh_pad_mask_


class RegistrationTransformer(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.encoder = ConvFeatures(in_channels, **kwargs)
        del kwargs["channels"]
        self.decoder = MultiScaleTransformer(self.encoder.out_sizes, **kwargs)

    def forward(self, moving, fixed):
        coord = make_coordinate_tensor_like(moving)
        m_feat, m_coord = self.encoder(moving, coord)
        f_feat, f_coord = self.encoder(fixed, coord)
        displacements, *add = self.decoder(m_feat, f_feat, m_coord, f_coord)
        return displacements, (f_feat, m_feat)
