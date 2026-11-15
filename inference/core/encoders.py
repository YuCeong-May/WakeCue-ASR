"""Cue encoder and cue-conditioned acoustic encoder.

Adapted from DAE-TSE / KCE (https://github.com/GnafiY/DAE-TSE,
``kce/src/kce/model/AEDKWSASRPhone.py``). Attribute names are unchanged so the
checkpoint loads strictly.
"""

import torch
import torch.nn as nn

from .layers import (
    DepthWiseConv,
    FNNBlock,
    MultiHeadAtt,
    MultiHeadCrossAtt,
    PositionalEncoding,
    TokenEmbedding,
    TransformerLayer,
    valid_mask,
)


class CueEncoder(nn.Module):
    """Text cue tokens -> contextual features (token embedding + N self-attention layers)."""

    def __init__(
        self,
        num_token,
        dim=256,
        num_blocks=4,
        input_trans_dim=256,
        self_att_heads=4,
        ffn_dim=(256, 2048, 256),
        padding_idx=0,
    ):
        super().__init__()
        self.phn_emb = TokenEmbedding(num_token, dim, padding_idx=padding_idx)
        self.keyword_input_projection = FNNBlock(input_trans_dim)
        self.keyword_pe_module = PositionalEncoding(dim)
        self.keyword_transformer = nn.ModuleList(
            [
                TransformerLayer(
                    size=dim,
                    self_att=MultiHeadAtt(self_att_heads, dim),
                    feed_forward=FNNBlock(list(ffn_dim)),
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, tokens, lengths):
        """(B, L) token ids, (B,) lengths -> (B, L, D) features, (B, L) valid mask."""
        valid = valid_mask(lengths.view(-1), tokens.size(1))
        x = self.keyword_pe_module(self.keyword_input_projection(self.phn_emb(tokens)))
        for layer in self.keyword_transformer:
            x = layer(x, valid.unsqueeze(1))
        return x, valid


class AcousticEncoder(nn.Module):
    """Conv2d x2 subsampling + N Conformer layers, each cross-attending to the cue."""

    def __init__(
        self,
        au_hidden_dim=256,
        num_mel_bins=80,
        num_blocks=9,
        input_trans_dim=(256, 256),
        self_att_heads=4,
        cross_att_heads=1,
        ffn_dim=(256, 2048, 256),
        conv_kernel_size=15,
    ):
        super().__init__()
        dim = au_hidden_dim
        self.au_conv = nn.Sequential(
            nn.Conv2d(1, dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 2),
            nn.ReLU(),
        )
        freq_dim = ((num_mel_bins - 1) // 2 - 1) // 2
        self.au_conv_trans = nn.Linear(dim * freq_dim, dim)
        self.speech_input_projection = FNNBlock(list(input_trans_dim))
        self.speech_pe_module = PositionalEncoding(dim)
        self.speech_transformer = nn.ModuleList(
            [
                TransformerLayer(
                    size=dim,
                    self_att=MultiHeadAtt(self_att_heads, dim),
                    cross_att=MultiHeadCrossAtt(cross_att_heads, dim),
                    feed_forward=FNNBlock(list(ffn_dim)),
                    macaron_layer=FNNBlock(list(ffn_dim)),
                    conv_layer=DepthWiseConv(dim, conv_kernel_size),
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_dim = dim

    @staticmethod
    def subsampled_lengths(lengths):
        """Frame count after two kernel-3, stride-2, unpadded convolutions."""
        for _ in range(2):
            lengths = torch.div(lengths - 3, 2, rounding_mode="floor") + 1
        return lengths

    def forward(self, speech, speech_lengths, cue_feat, cue_valid):
        """(B, T, 80) fbank -> (B, T', D) states and (B,) lengths."""
        lengths = self.subsampled_lengths(speech_lengths.view(-1))
        x = self.au_conv(speech.unsqueeze(1))
        b, c, t, f = x.size()
        x = self.au_conv_trans(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = self.speech_pe_module(self.speech_input_projection(x))

        lengths = torch.clamp(lengths, max=t)
        valid = valid_mask(lengths, t)
        cross_mask = (valid[:, :, None] & cue_valid[:, None, :]).unsqueeze(1)
        padding = ~valid.unsqueeze(-1)
        for layer in self.speech_transformer:
            # Zero the padding frames so the depthwise conv cannot leak them.
            x = x.masked_fill(padding, 0.0)
            x = layer(x, valid.unsqueeze(1), cue_feat, cross_mask)
        return x, lengths
