"""
Attention LSTM Decoder (Show, Attend and Tell).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocabulary import START_TOKEN, END_TOKEN
from .attention import Attention


class AttentionDecoder(nn.Module):
    """
    Single-layer LSTM decoder with soft attention.
    """

    def __init__(
        self,
        vocab_size    : int,
        encoder_dim   : int   = 2048,
        embed_dim     : int   = 256,
        hidden_dim    : int   = 512,
        attention_dim : int   = 512,
        dropout       : float = 0.5,
    ):
        super().__init__()

        self.encoder_dim   = encoder_dim
        self.embed_dim     = embed_dim
        self.hidden_dim    = hidden_dim
        self.attention_dim = attention_dim
        self.vocab_size    = vocab_size

        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.h_init = nn.Linear(encoder_dim, hidden_dim)
        self.c_init = nn.Linear(encoder_dim, hidden_dim)

        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    # Weight init

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight,        -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.0)

    # Hidden state init

    def _init_hidden(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, L, encoder_dim)
        Returns:
            h, c: each (B, hidden_dim)
        """
        mean_feat = features.mean(dim=1)
        h = torch.tanh(self.h_init(mean_feat))
        c = torch.tanh(self.c_init(mean_feat))
        return h, c

    # Training forward

    def forward(
        self,
        features : torch.Tensor,
        captions : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing forward pass.
        """
        B          = features.size(0)
        seq_len    = captions.size(1) - 1

        embeddings = self.embedding(captions[:, :-1])

        h, c    = self._init_hidden(features)
        logits  = []
        alphas  = []

        for t in range(seq_len):
            context, alpha = self.attention(features, h)
            lstm_input     = torch.cat([embeddings[:, t], context], dim=1)
            h, c           = self.lstm_cell(lstm_input, (h, c))

            logits.append(self.fc(self.dropout(h)))
            alphas.append(alpha)

        logits = torch.stack(logits, dim=1)
        alphas = torch.stack(alphas, dim=1)

        return logits, alphas

    # Greedy inference 

    @torch.no_grad()
    def generate_greedy(
        self,
        features   : torch.Tensor,
        vocab,
        max_length : int = 50,
    ) -> list[int]:

        h, c   = self._init_hidden(features)
        inp    = torch.tensor([[vocab[START_TOKEN]]], device=features.device)
        result = []

        for _ in range(max_length):
            emb            = self.embedding(inp).squeeze(1)
            context, _     = self.attention(features, h)
            lstm_input     = torch.cat([emb, context], dim=1)
            h, c           = self.lstm_cell(lstm_input, (h, c))

            logits = self.fc(h)
            pred   = logits.argmax(dim=-1).item()

            if pred == vocab[END_TOKEN]:
                break
            result.append(pred)
            inp = torch.tensor([[pred]], device=features.device)

        return result

    # Beam search inference

    @torch.no_grad()
    def generate_beam(
        self,
        features   : torch.Tensor,
        vocab,
        beam_width : int = 5,
        max_length : int = 50,
    ) -> list[int]:

        device  = features.device
        h, c    = self._init_hidden(features)
        start   = vocab[START_TOKEN]
        end_idx = vocab[END_TOKEN]

        beams: list[tuple[float, list[int], torch.Tensor, torch.Tensor]] = [
            (0.0, [start], h, c)
        ]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_length):
            candidates: list[tuple[float, list[int], torch.Tensor, torch.Tensor]] = []

            for score, tokens, bh, bc in beams:
                if tokens[-1] == end_idx:
                    completed.append((score, tokens[1:]))
                    continue

                inp        = torch.tensor([[tokens[-1]]], device=device)
                emb        = self.embedding(inp).squeeze(1)
                context, _ = self.attention(features, bh)
                lstm_input = torch.cat([emb, context], dim=1)
                nh, nc     = self.lstm_cell(lstm_input, (bh, bc))

                log_probs        = F.log_softmax(self.fc(nh), dim=-1)
                top_probs, top_idx = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    candidates.append((
                        score + top_probs[0, k].item(),
                        tokens + [top_idx[0, k].item()],
                        nh, nc,
                    ))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        if completed:
            completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            return completed[0][1]

        best = max(beams, key=lambda x: x[0])
        return best[1][1:]
