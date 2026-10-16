"""
Implements additive (Bahdanau-style) soft attention as described in Xu et al. 2015
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
        encoder_dim   : depth of each encoder location (2048 for ResNet-50).
        decoder_dim   : LSTM hidden size.
        attention_dim : size of the joint attention space.
    """

    def __init__(
        self,
        encoder_dim   : int,
        decoder_dim   : int,
        attention_dim : int,
    ):
        super().__init__()

        self.enc_attn = nn.Linear(encoder_dim, attention_dim)
        self.dec_attn = nn.Linear(decoder_dim, attention_dim)
        self.score    = nn.Linear(attention_dim, 1, bias=False)

        self.relu     = nn.ReLU()
        self.softmax  = nn.Softmax(dim=1)

    def forward(
        self,
        features: torch.Tensor,
        h_prev  : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context : (B, encoder_dim)
            alpha   : (B, L) - attention weights.
        """
        energy  = self.enc_attn(features) + self.dec_attn(h_prev).unsqueeze(1)
        energy  = self.score(self.relu(energy)).squeeze(2)

        alpha   = self.softmax(energy)
        context = (alpha.unsqueeze(2) * features).sum(dim=1)

        return context, alpha
