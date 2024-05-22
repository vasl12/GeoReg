from typing import Optional, List, Tuple
import torch
from torch import nn
from utils.utils import make_coordinate_tensor, CAPE, calculate_closest_indices, calculate_distances


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.double_conv(x)


class UnetFeatures(nn.Module):
    def __init__(self, in_channels, channels, dropout=0.0, pool_size=2):
        super().__init__()
        self.embed = DoubleConv(in_channels, channels[0], dropout=dropout)
        self.pool_size = pool_size
        self.pool = nn.AvgPool2d(self.pool_size)
        self.downs = nn.ModuleList([DoubleConv(channels[i], channels[i+1], dropout=dropout)
                                    for i in range(len(channels) - 1)])
        self.upsample = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True)
        in_ch = [channels[i+1] + channels[i] for i in range(len(channels) - 2, -1, -1)]
        out_ch = [channels[i] for i in range(len(channels) - 1, -1, -1)]
        self.ups = nn.ModuleList([DoubleConv(i, o, dropout=dropout) for i, o in zip(in_ch, out_ch[1:])])
        self.out_sizes = out_ch

    def forward(self, x: torch.Tensor, coord_batch: torch.Tensor):
        device = x.device
        out = self.embed(x)
        skips = []
        skips_coord = [coord_batch]
        coord = coord_batch[0]
        parent_idxs = []
        for down in self.downs:
            skips.append(out)
            out = self.pool(out)
            out = down(out)
            coord_step = coord[:, self.pool_size - 1, self.pool_size - 1] - coord[:, 0, 0]
            coord = torch.meshgrid(torch.arange((coord[0, 0, 0] + coord_step[0] / 2).item(),
                                                coord[0, 0, 0].item() + coord_step[0]*out.shape[-2],
                                                coord_step[0].item(),
                                                dtype=torch.float32, device=device),
                                   torch.arange((coord[1, 0, 0] + coord_step[1] / 2).item(),
                                                coord[1, 0, 0].item() + coord_step[1]*out.shape[-1],
                                                coord_step[1].item(),
                                                dtype=torch.float32, device=device))
            coord = torch.stack(coord, dim=0)
            skips_coord.append(coord.tile((x.shape[0], 1, 1, 1)))
            # Create 2D array with the 1D indices of each node
            parent_idx_ = torch.arange(0, torch.prod(torch.tensor(out.shape[2:])).item(), dtype=torch.long, device=device)
            parent_idx = parent_idx_.reshape(out.shape[2:])
            child_idx = parent_idx.repeat_interleave(self.pool_size, dim=-2).repeat_interleave(self.pool_size, dim=-1)
            parent_idxs.append(child_idx)
        parent_idxs.append(None)
        outs = [out]
        for up, skip in zip(self.ups, skips[::-1]):
            out = self.upsample(out)
            out = torch.cat((out, skip), dim=1)
            out = up(out)
            outs.append(out)
        return outs, skips_coord[::-1], parent_idxs[::-1]


class CrossAttentionLayer(nn.Module):
    r"""CrossAttentionLayer is made up of multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).

    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                                    **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x), memory, memory_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x, memory, memory_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(self, x: torch.Tensor, mem: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=True)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.dropout2(x)


class MultiScaleTransformer(nn.Module):
    def __init__(self, channels, nhead=1, dropout=0.0, num_neigh=5):
        super().__init__()
        self.sa_layers = nn.ModuleList([CrossAttentionLayer(c, nhead, dim_feedforward=c, dropout=dropout)
                                        for c in channels])
        self.ca_layers = nn.ModuleList([CrossAttentionLayer(c, nhead, dim_feedforward=c, dropout=dropout)
                                        for c in channels])
        self.out_layers = nn.ModuleList([nn.Linear(c, out_features=2) for c in channels])
        # self.out_layers = nn.ModuleList([i for c in channels for i in [nn.Linear(c, 2), nn.Tanh()]])
        # TODO: is this controlling how large the neightbourhood is?

        self.sa_kernel = num_neigh
        self.pos_enc = CAPE()

    def forward(self, 
                m_feat_list: List[torch.Tensor], f_feat_list: List[torch.Tensor],
                m_coord_list: List[torch.Tensor], f_coord_list: List[torch.Tensor],
                m_pidx_list: List[torch.Tensor], f_pidx_list: List[torch.Tensor]):

        # We move channel dimension to last dimension because that is what transformers like.
        m_feat_start = m_feat_list[0]
        m_feat_start = m_feat_start.moveaxis(1, -1)
        f_feat_start = f_feat_list[0]
        f_feat_start = f_feat_start.moveaxis(1, -1)
        f_coord_start = f_coord_list[0]
        f_coord_start = f_coord_start.moveaxis(1, -1)
        m_coord_start = m_coord_list[0]
        m_coord_start = m_coord_start.moveaxis(1, -1)

        # ---------- Align feature spaces (avoid affine registration)  -----
        # Compute cosine similarity
        # m_feat_start_ = m_feat_start.reshape((m_feat_start.shape[0], -1, 1, m_feat_start.shape[-1]))
        # f_feat_start_ = f_feat_start.reshape((f_feat_start.shape[0], 1, -1, f_feat_start.shape[-1]))
        # sim = torch.nn.functional.cosine_similarity(m_feat_start_, f_feat_start_, dim=-1)
        # most_sim = sim.argmax(-1)
        # # The starting coordinate of a moving image feature are the coords of the fixed image feature most similar to it
        # f_coord_start_ = f_coord_start.reshape((f_coord_start.shape[0], -1, f_coord_start.shape[-1]))
        # b, _ = torch.meshgrid(torch.arange(0, f_coord_start_.shape[0], device=f_coord_start.device),
        #                       torch.arange(0, f_coord_start_.shape[1], device=f_coord_start.device))
        # m_coord_moved = f_coord_start_[b.reshape(-1), most_sim.reshape(-1)].reshape(f_coord_start.shape)
        # displ = m_coord_start - m_coord_moved

        displ = torch.zeros_like(m_coord_start)
        displacements = [displ.clone()]
        for m_feat, f_feat, m_coord, f_coord, m_pidx, sa_layer, ca_layer, feat2displ \
                in zip(m_feat_list, f_feat_list, m_coord_list, f_coord_list, m_pidx_list,
                       self.sa_layers, self.ca_layers, self.out_layers):
            # We move channel dimension to last dimension because that is what transformers like
            m_feat = m_feat.moveaxis(1, -1)
            f_feat = f_feat.moveaxis(1, -1)
            m_coord = m_coord.moveaxis(1, -1)
            f_coord = f_coord.moveaxis(1, -1)
            batch_size, spatial_sizes, channel_size, coord_size = \
                m_feat.shape[0], m_feat.shape[1:-1], m_feat.shape[-1], f_coord.shape[-1]

            # -----------  Deformation inheritance  ---------------
            if m_pidx is not None:
                # When moving onto a finer layer, the child coordinates are their original coordinates
                # plus their parent's current displacement
                m_pidx_ = m_pidx.reshape(-1)
                displ_ = displ.reshape((batch_size, -1, displ.shape[-1]))
                b, m_pidx_tile_ = torch.meshgrid(torch.arange(0, batch_size, device=displ.device), m_pidx_)
                child_displ_ = displ_[b.reshape(-1), m_pidx_tile_.reshape(-1)]
                child_displ = child_displ_.reshape((batch_size, *spatial_sizes, coord_size))
                displ = child_displ

            m_coord = m_coord + displ

            # -----------  Moving Image Self-Attention  ------------------
            # Define moving image neighbourhoods for every moving image feature
            m_self_feats_, m_neigh_feat_, m_relative_coord_, pad_mask_ = self.compute_moving_neighbours(m_feat, m_coord)
            m_enc_coord_ = self.pos_enc(m_relative_coord_.detach(), channel_size)
            m_neigh_feat_ += m_enc_coord_
            # Every moving image feature does cross attention to every surrounding moving image neighbour
            out__ = sa_layer(m_self_feats_.reshape((-1, m_self_feats_.shape[2], channel_size)),  # 1 x neigh_size cross-att
                             m_neigh_feat_.reshape((-1, m_neigh_feat_.shape[2], channel_size)),
                             pad_mask_.reshape((-1, pad_mask_.shape[2], pad_mask_.shape[3])))

            # -----------  Fixed Image Cross-Attention  ------------------
            # Define fixed image neighbourhoods for every moving image feature
            f_neigh_feat, f_relative_coord, pad_mask_ = self.compute_fixed_neighbours(f_feat, f_coord, m_coord)
            f_enc_coord = self.pos_enc(f_relative_coord.detach(), channel_size)
            f_neigh_feat += f_enc_coord
            # Every moving image feature does cross attention to every surrounding fixed image neighbour
            out__ = ca_layer(out__,
                             f_neigh_feat.reshape((-1, f_neigh_feat.shape[2], channel_size)),
                             pad_mask_.reshape((-1, pad_mask_.shape[2], pad_mask_.shape[3])))
            out__ = out__.squeeze(1)  # (batch * H * W, channels)

            # ---------  Feature to deformation vector  -----------
            # Convert features into deformation vectors
            local_displ__ = feat2displ(out__)
            local_displ = local_displ__.reshape((batch_size, *spatial_sizes, coord_size))
            # local_displ = nn.functional.tanh(local_displ) * .2
            # TODO: regilarise per level?
            displ += local_displ
            displacements.append(displ.clone())
        return displacements

    def compute_moving_neighbours(self, m_feat: torch.Tensor, m_coord: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, spatial_sizes, channel_size, coord_size = \
            m_feat.shape[0], m_feat.shape[1:-1], m_feat.shape[-1], m_coord.shape[-1]
        spatial_1d_size = torch.prod(torch.tensor(m_coord.shape[1:-1])).item()

        # Create an indexing array where for each element in the feature array, the unrolled 1D index is given
        idx_ = torch.arange(0, spatial_1d_size, dtype=torch.float32, device=m_coord.device)  # Floats because F.unfold only takes floats
        idx = idx_.reshape(spatial_sizes)
        # Pad the indexing array such that all points have the same number of neighbours
        pad_value = -1
        idx_pad = nn.functional.pad(idx,
                                    (self.sa_kernel // 2, self.sa_kernel // 2,  # Left pad size, Right pad size
                                     self.sa_kernel // 2, self.sa_kernel // 2),  # Top pad size, Bottom pad size
                                    value=pad_value)
        # For each feature, get its neighbours indices. Shape: (K*K, H*W)
        neigh_ = nn.functional.unfold(idx_pad[None], kernel_size=self.sa_kernel).long()
        pad_mask = (neigh_ == pad_value)

        # Get the neighbouring points coordinates for each point in the feature array (and for each batch element)
        b, neigh_tile = torch.meshgrid(torch.arange(0, batch_size, device=m_coord.device),
                                       neigh_.reshape(-1))
        neigh_tile__ = neigh_tile.reshape(-1)
        b__ = b.reshape(-1)
        m_coord_ = m_coord.reshape((batch_size, spatial_1d_size, coord_size))  # Shape: (batch, H*W, coord)
        m_neigh_coord__ = m_coord_[b__, neigh_tile__]
        # This reshape causes the ordering of the neighbourhoods to be correct. Carefull about changing it!
        m_neigh_coord_ = m_neigh_coord__.reshape((batch_size, neigh_.shape[0],
                                                  spatial_1d_size, coord_size))  # Shape: (batch, K*K, H*W, coord)
        # For each neighbourhood kernel, get the distance from the center to each kernel element.
        m_center_ = m_coord_.unsqueeze(1)  # (batch, 1, H*W, coord)
        m_relative_coord_ = m_center_ - m_neigh_coord_  # (batch, K*K, H*W, coord)

        # For each neighbourhood kernel, get the feature of each neighbour.
        m_feat_ = m_feat.reshape((batch_size, spatial_1d_size, channel_size))
        m_neigh_feat_ = m_feat_[b__, neigh_tile__].reshape((batch_size, neigh_.shape[0], spatial_1d_size, channel_size))

        # Pytorch transformers like to have inputs in the shape (batch, seq_len, channels)
        m_relative_coord_ = m_relative_coord_.moveaxis(2, 1)                        # Shape: (batch, H*W, K*K, coord)
        m_neigh_feat_ = m_neigh_feat_.moveaxis(2, 1)                                # Shape: (batch, H*W, K*K, coord)
        m_self_feat_ = m_feat_.unsqueeze(2)                                        # Shape: (batch, H*W, 1, coord)
        pad_mask_ = pad_mask.moveaxis(1, 0).unsqueeze(1).tile((batch_size, 1, 1, 1))  # Shape: (batch, H*W, 1, K*K)
        return m_self_feat_, m_neigh_feat_, m_relative_coord_, pad_mask_

    def compute_fixed_neighbours(self, f_feat: torch.Tensor, f_coord: torch.Tensor, m_coord: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, f_spatial_sizes, channel_size, coord_size = \
            f_feat.shape[0], f_feat.shape[1:-1], f_feat.shape[-1], f_coord.shape[-1]
        f_spatial_1d_size = torch.prod(torch.tensor(f_feat.shape[1:-1])).item()
        m_spatial_1d_size = torch.prod(torch.tensor(m_coord.shape[1:-1])).item()

        # Create an indexing array where for each element in the feature array, the unrolled 1D index is given
        idx_ = torch.arange(0, f_spatial_1d_size, dtype=torch.float32, device=f_coord.device)  # Floats because F.unfold only takes floats
        f_idx = idx_.reshape(f_spatial_sizes)
        # Pad the indexing array such that all points have the same number of neighbours
        pad_value = -1
        idx_pad = nn.functional.pad(f_idx,
                                    (self.sa_kernel // 2, self.sa_kernel // 2,  # Left pad size, Right pad size
                                     self.sa_kernel // 2, self.sa_kernel // 2),  # Top pad size, Bottom pad size
                                    value=pad_value)
        # For each feature, get its neighbours indices. Shape: (K*K, H*W)
        neigh = nn.functional.unfold(idx_pad[None], kernel_size=self.sa_kernel).long()

        # ---------------  Match moving points to fixed points  ---------------------
        # Find the closest fixed image points to each moving image point
        closest_f_2d_idx = calculate_closest_indices(m_coord, f_coord)
        # Get the neighbouring points coordinates for each point in the feature array (and for each batch element)
        b, _ = torch.meshgrid(torch.arange(0, batch_size, device=m_coord.device),
                              torch.arange(0, f_spatial_1d_size, device=m_coord.device))
        b__ = b.reshape(-1)
        closest_f_2d_idx_x__ = closest_f_2d_idx[..., 0].reshape(-1)
        closest_f_2d_idx_y__ = closest_f_2d_idx[..., 1].reshape(-1)
        closest_f_1d_idx__ = f_idx.to(torch.long).tile((batch_size, 1, 1))[b__, closest_f_2d_idx_x__, closest_f_2d_idx_y__]
        # Make sure the calculate_closest_indices actually gives you the closest coords,
        # direct distance calculation is very memory expensive
        # a = calculate_distances(m_coord.reshape((m_coord.shape[0], -1, m_coord.shape[-1]))[:, :m_coord.shape[1]],
        #                                          f_coord.reshape((f_coord.shape[0], -1, f_coord.shape[-1]))).argmin(-1)
        # assert (closest_f_1d_idx_[:, :m_coord.shape[1]] == a).all().item()

        neigh_idx_ = neigh.tile((batch_size, 1, 1))[b__, :, closest_f_1d_idx__]
        neigh_pad_mask__ = (neigh_idx_ == pad_value)
        f_feat_ = f_feat.reshape((batch_size, f_spatial_1d_size, channel_size))
        f_coord_ = f_coord.reshape((batch_size, f_spatial_1d_size, coord_size))

        # Get the neighbouring points coordinates for each point in the feature array (and for each batch element)
        b, _ = torch.meshgrid(torch.arange(0, batch_size, device=m_coord.device),
                              torch.arange(0, neigh.shape[0] * neigh.shape[1], device=m_coord.device))
        b__ = b.reshape(-1)
        neigh_idx__ = neigh_idx_.reshape(-1)

        m_neigh_f_feat__ = f_feat_[b__, neigh_idx__]
        m_neigh_f_feat_ = m_neigh_f_feat__.reshape((batch_size, m_spatial_1d_size, -1, channel_size))
        m_neigh_f_coord__ = f_coord_[b__, neigh_idx__]
        # This reshape causes the ordering of the neighbourhoods to be correct. Carefull about changing it!
        m_neigh_f_coord_ = m_neigh_f_coord__.reshape((batch_size, m_spatial_1d_size, -1, coord_size))  # (batch, H*W, K*K, coord)

        m_coord_ = m_coord.reshape((batch_size, m_spatial_1d_size, coord_size))
        m_center_ = m_coord_.unsqueeze(2)  # (batch, H*W, 1, coord)
        m_relative_f_coord_ = m_center_ - m_neigh_f_coord_  # (batch, H*W, K*K, coord)

        neigh_pad_mask_ = neigh_pad_mask__.reshape((batch_size, m_spatial_1d_size, 1, -1))  # (batch, H*W, 1, K*K)

        return m_neigh_f_feat_, m_relative_f_coord_, neigh_pad_mask_


class RegistrationTransformer(nn.Module):
    def __init__(self, in_channels, channels, dropout=0.0, nhead=1, pool_size=4, num_neigh=5):
        super().__init__()
        self.encoder = UnetFeatures(in_channels, channels, dropout=dropout, pool_size=pool_size)
        self.decoder = MultiScaleTransformer(self.encoder.out_sizes, nhead=nhead, dropout=dropout, num_neigh=num_neigh)

    def forward(self, moving, fixed):
        coord = make_coordinate_tensor(dims=tuple(moving.shape)[-2:], gpu=True) # TODO: Allow for batch size of differing shapes
        coord = coord.tile((moving.shape[0], 1, 1, 1))
        m_feat, m_coord, m_pidx = self.encoder(moving, coord)
        f_feat, f_coord, f_pidx = self.encoder(fixed, coord)
        displacement = self.decoder(m_feat, f_feat, m_coord, f_coord, m_pidx, f_pidx)
        return displacement
