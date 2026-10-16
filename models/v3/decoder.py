"""
Transformer Caption Decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocabulary import START_TOKEN, END_TOKEN


class TransformerDecoder(nn.Module):
    """
    Transformer caption decoder with cross-attention over CNN spatial features.
    """

    def __init__(
        self,
        vocab_size      : int,
        d_model         : int   = 512,
        n_heads         : int   = 8,
        n_layers        : int   = 3,
        dim_feedforward : int   = 2048,
        dropout         : float = 0.1,
        max_seq_len     : int   = 50,
    ):
        super().__init__()

        self.d_model    = d_model
        self.vocab_size = vocab_size

        # Token embedding + learned positional encoding
        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Stack of Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers    = n_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, vocab_size)

        self._init_weights()

    # Weight init

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.02)

    # Helpers

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embedding scaled by √d_model + positional encoding."""
        T   = tokens.size(1)
        emb = self.embedding(tokens) * math.sqrt(self.d_model)
        emb = self.dropout(emb + self.pos_encoding[:, :T, :])
        return emb

    # Training forward (teacher-forcing)

    def forward(
        self,
        memory   : torch.Tensor,   # (B, 49, d_model) - encoder output
        captions : torch.Tensor,
    ) -> torch.Tensor:
        tgt = captions[:, :-1]
        T   = tgt.size(1)

        tgt_emb  = self._embed(tgt)
        causal   = self._causal_mask(T, tgt.device)
        pad_mask = (tgt == 0)

        out = self.transformer_decoder(
            tgt                  = tgt_emb,
            memory               = memory,
            tgt_mask             = causal,
            tgt_key_padding_mask = pad_mask,
        )

        return self.fc(out)

    # Greedy inference

    @torch.no_grad()
    def generate_greedy(
        self,
        memory     : torch.Tensor,
        vocab,
        max_length : int = 50,
    ) -> list[int]:

        device  = memory.device
        tokens  = [vocab[START_TOKEN]]
        result  = []

        for _ in range(max_length):
            inp    = torch.tensor([tokens], device=device)
            T      = inp.size(1)
            emb    = self._embed(inp)
            causal = self._causal_mask(T, device)

            out    = self.transformer_decoder(
                tgt    = emb,
                memory = memory,
                tgt_mask = causal,
            )

            logits = self.fc(out[:, -1, :])
            pred   = logits.argmax(dim=-1).item()

            if pred == vocab[END_TOKEN]:
                break
            result.append(pred)
            tokens.append(pred)

        return result

    # Beam search inference

    @torch.no_grad()
    def generate_beam(
        self,
        memory     : torch.Tensor,
        vocab,
        beam_width : int = 5,
        max_length : int = 50,
    ) -> list[int]:

        device  = memory.device
        start   = vocab[START_TOKEN]
        end_idx = vocab[END_TOKEN]

        # Each beam: (log_prob, token_list)
        beams: list[tuple[float, list[int]]] = [(0.0, [start])]
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_length):
            candidates: list[tuple[float, list[int]]] = []

            for score, tokens in beams:
                if tokens[-1] == end_idx:
                    completed.append((score, tokens[1:]))
                    continue

                inp    = torch.tensor([tokens], device=device)
                T      = inp.size(1)
                emb    = self._embed(inp)
                causal = self._causal_mask(T, device)

                out = self.transformer_decoder(
                    tgt      = emb,
                    memory   = memory,
                    tgt_mask = causal,
                )

                log_probs          = F.log_softmax(self.fc(out[:, -1, :]), dim=-1)
                top_probs, top_idx = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    candidates.append((
                        score + top_probs[0, k].item(),
                        tokens + [top_idx[0, k].item()],
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
