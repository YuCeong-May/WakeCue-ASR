"""Building blocks of the cue encoder and the acoustic encoder.

Adapted from DAE-TSE / KCE (https://github.com/GnafiY/DAE-TSE,
``kce/src/kce/model/NetModules.py``), which itself derives from WeNet
(https://github.com/wenet-e2e/wenet). Only the code paths used at inference are
kept; attribute names are unchanged so the checkpoint loads strictly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _attend(q, k, v, mask):
    """Scaled dot-product attention. ``mask`` is True at valid positions."""
    batch, _, tq, d = q.size()
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if mask.dim() < q.dim():
        mask = mask.unsqueeze(1)
    mask = mask.eq(0)
    score = score.masked_fill(mask, -float("inf"))
    weight = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
    context = torch.matmul(weight, v)
    return context.transpose(1, 2).contiguous().view(batch, tq, -1)


class TokenEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.emb(x)


class PositionalEncoding(nn.Module):
    """Scale by sqrt(d) and add a sinusoidal position encoding."""

    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.xscale = math.sqrt(dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x * self.xscale + self.pe[: x.size(1)]


class MultiHeadAtt(nn.Module):
    def __init__(self, n_head, n_feats):
        super().__init__()
        self.d_k = n_feats // n_head
        self.n_head = n_head
        self.q = nn.Linear(n_feats, n_feats)
        self.k = nn.Linear(n_feats, n_feats)
        self.v = nn.Linear(n_feats, n_feats)
        self.linear_out = nn.Linear(n_feats, n_feats)

    def forward(self, x, mask):
        batch = x.size(0)
        q, k, v = (
            proj(x).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
            for proj in (self.q, self.k, self.v)
        )
        return self.linear_out(_attend(q, k, v, mask))


class MultiHeadCrossAtt(nn.Module):
    """Cross-attention with LayerNorm before each of the q/k/v projections."""

    def __init__(self, n_head, n_feats, kv_feats=None):
        super().__init__()
        kv_feats = n_feats if kv_feats is None else int(kv_feats)
        self.q = nn.Sequential(nn.LayerNorm(n_feats), nn.Linear(n_feats, n_feats))
        self.k = nn.Sequential(nn.LayerNorm(kv_feats), nn.Linear(kv_feats, n_feats))
        self.v = nn.Sequential(nn.LayerNorm(kv_feats), nn.Linear(kv_feats, n_feats))
        self.linear_out = nn.Linear(n_feats, n_feats)
        self.n_head = n_head

    def forward(self, x, memory, mask):
        batch = x.size(0)
        q = self.q(x).view(batch, x.size(1), self.n_head, -1).transpose(1, 2)
        k = self.k(memory).view(batch, memory.size(1), self.n_head, -1).transpose(1, 2)
        v = self.v(memory).view(batch, memory.size(1), self.n_head, -1).transpose(1, 2)
        return self.linear_out(_attend(q, k, v, mask))


class DepthWiseConv(nn.Module):
    """Conformer convolution module: pointwise, GLU, depthwise, BatchNorm, pointwise."""

    def __init__(self, channels, kernel_size=15):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.pointwise_conv1(x.transpose(1, 2))
        x = F.glu(x, dim=1)
        x = self.activation(self.norm(self.depthwise_conv(x)))
        return self.pointwise_conv2(x).transpose(1, 2)


class FNNBlock(nn.Module):
    """Linear -> ReLU -> Linear. ``dim`` is ``d``, ``[in, out]`` or ``[in, hidden, out]``."""

    def __init__(self, dim):
        super().__init__()
        if isinstance(dim, int):
            idim = hdim = odim = dim
        elif len(dim) == 2:
            idim, odim = dim
            hdim = idim
        else:
            idim, hdim, odim = dim
        self.w1 = nn.Linear(idim, hdim)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(hdim, odim)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class TransformerLayer(nn.Module):
    """Pre-norm layer: [macaron FFN] -> self-att -> [cross-att] -> [conv] -> FFN."""

    def __init__(
        self,
        self_att,
        feed_forward,
        macaron_layer=None,
        conv_layer=None,
        cross_att=None,
        size=256,
    ):
        super().__init__()
        self.self_att = self_att
        self.feed_forward = feed_forward
        self.macaron_layer = macaron_layer
        self.cross_att = cross_att
        self.conv_layer = conv_layer
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        if conv_layer is not None:
            self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        if macaron_layer is not None:
            self.macaron_norm = nn.LayerNorm(size, eps=1e-5)

    def forward(self, x, mask, memory=None, memory_mask=None):
        if self.macaron_layer is not None:
            x = x + 0.5 * self.macaron_layer(self.macaron_norm(x))
        x = x + self.self_att(self.input_norm(x), mask)
        if self.cross_att is not None:
            x = self.cross_att(x, memory, memory_mask) + x
        if self.conv_layer is not None:
            x = self.conv_layer(self.conv_norm(x)) + x
        return self.feed_forward(self.fnn_norm(x)) + x


def valid_mask(lengths, max_len):
    """(B,) lengths -> (B, max_len) bool, True at valid positions."""
    pos = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    return pos[None, :] < lengths[:, None]
