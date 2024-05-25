import math
from typing import Optional

# import natten
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0, stride=1, dilation=1, num_spatial_dims=2, out_activation=True, **kwargs):
        super().__init__()
        conv_class = nn.Conv2d if num_spatial_dims == 2 else nn.Conv3d
        hid_channels = out_channels if out_channels > in_channels else in_channels
        layers = [
            conv_class(in_channels, hid_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            conv_class(hid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride, dilation=dilation, bias=True),
        ]
        if out_activation:
            layers += [nn.ReLU(inplace=True), nn.Dropout(dropout),]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class CustomOut(nn.Module):
    def __init__(self, c, out):
        super().__init__()
        self.lin1 = nn.Linear(c, c*2)
        self.lin2 = nn.Linear(c*2, out, bias=False)
        nn.init.xavier_uniform_(self.lin2.weight, 0.01)

    def forward(self, x):
        y1 = self.lin1(x)
        out = self.lin2(torch.nn.functional.relu(y1))
        return out


class AttentionWeightedLayer(nn.Module):
    """ Based on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.k_proj.bias.data.fill_(0)
        self.q_proj.bias.data.fill_(0)

    @staticmethod
    def expand_mask(mask):
        assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
        return mask

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.bmm(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.bool(), -9e15)
        attention = torch.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, q, k, v, mask=None, return_attention=False):
        batch_size, seq_length, _ = q.size()
        q = self.q_proj(q)
        k = self.k_proj(k)

        # # Separate Q, K, V from linear output
        # qk = qk.reshape(batch_size, seq_length, 1, 2*self.head_dim)
        # qk = qk.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        # q, k = qk.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)

        if return_attention:
            return values, attention
        else:
            return values


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

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.0,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False, **kwargs) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        # -------- THE FOLLOWING IS NOT IN THE ORIGINAL PYTORCH CODE -----------
        # Pytorch expects us to repeat_interleave the mask for every att_head outside this layer,
        # but that makes things confusing so we do it here
        memory_mask = memory_mask.repeat_interleave(self.multihead_attn.num_heads, dim=0)
        # ----------------------------------------------------------------------
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


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, num_spatial_dims: int, kernel_size: int, dim_feedforward: int = 128, dropout: float = 0.0,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False, **kwargs) -> None:
        super().__init__()
        att_class = natten.NeighborhoodAttention3D if num_spatial_dims == 3 else natten.NeighborhoodAttention2D
        try:
            self.multihead_attn = att_class(d_model, nhead,
                                            rel_pos_bias=False,  # We don't want positional bias based on grid location
                                            kernel_size=kernel_size, attn_drop=dropout, proj_drop=dropout)
        except TypeError:
            self.multihead_attn = att_class(d_model, nhead,
                                            bias=False,  # We don't want positional bias based on grid location
                                            kernel_size=kernel_size, attn_drop=dropout, proj_drop=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        if self.norm_first:
            x = x + self._mha_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._mha_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    # multihead attention block
    def _mha_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.multihead_attn(x)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.dropout2(x)
