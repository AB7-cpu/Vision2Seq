"""
models/v1/decoder.py — LSTM Caption Decoder (Show and Tell, V1).

Architecture
============
The next step in the architeture is to take the vector embedding generated from the CNN Encoder
and feed it to the LSTM Decoder. 
It takes the embedding of the START token at the begining to start the sequence generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabulary import START_TOKEN, END_TOKEN


class LSTMDecoder(nn.Module):
    """
    Single-layer LSTM caption decoder.
    """

    def __init__(
        self,
        vocab_size  : int,
        embed_dim   : int   = 256,     # Same as the output from the CNNEncoder.
        hidden_dim  : int   = 512,     # Hidden dimension of the LSTM.
        num_layers  : int   = 1,       # Number of LSTM layers.
        dropout     : float = 0.5,     # Dropout probability.
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size


        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Project image embedding → initial (h_0, c_0) for each LSTM layer
        self.h_init = nn.Linear(embed_dim, num_layers * hidden_dim)
        self.c_init = nn.Linear(embed_dim, num_layers * hidden_dim)


        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    #     Weight initialisation

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight,        -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.0)

    #     Hidden state initialisation

    def _init_hidden(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        batch = features.size(0)
        h = torch.tanh(self.h_init(features))  # (batch, num_layers * hidden_dim)
        c = torch.tanh(self.c_init(features))
        # (num_layers, batch, hidden_dim)
        h = h.view(batch, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(batch, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h, c


    def forward(
        self,
        features  : torch.Tensor,
        captions  : torch.Tensor,
    ) -> torch.Tensor:

        # Input  = <START> tok1 tok2 … tokN-1   (all tokens except the last)
        # Target = tok1   tok2 … tokN-1 <END>   (all tokens except <START>)
        h, c = self._init_hidden(features)

        embeddings = self.embedding(captions[:, :-1])   # (batch, seq-1, embed_dim)
        hiddens, _ = self.lstm(embeddings, (h, c))      # (batch, seq-1, hidden_dim)
        logits     = self.fc(self.dropout(hiddens))     # (batch, seq-1, vocab_size)
        return logits


    def generate_greedy(
        self,
        features   : torch.Tensor,
        vocab,
        max_length : int = 50,
    ) -> list[int]:
        
        h, c    = self._init_hidden(features)
        inp     = torch.tensor([[vocab[START_TOKEN]]], device=features.device)
        result  = []

        for _ in range(max_length):
            emb         = self.embedding(inp)          # (1, 1, embed_dim)
            out, (h, c) = self.lstm(emb, (h, c))       # (1, 1, hidden_dim)
            logits      = self.fc(out.squeeze(1))       # (1, vocab_size)
            pred        = logits.argmax(dim=-1).item()

            if pred == vocab[END_TOKEN]:
                break
            result.append(pred)
            inp = torch.tensor([[pred]], device=features.device)

        return result

    #     Beam search inference

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

        # Each beam: (cumulative_log_prob, token_list, h, c)
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

                inp         = torch.tensor([[tokens[-1]]], device=device)
                emb         = self.embedding(inp)
                out, (nh, nc) = self.lstm(emb, (bh, bc))
                log_probs   = F.log_softmax(self.fc(out.squeeze(1)), dim=-1)
                top_probs, top_idx = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    new_score  = score + top_probs[0, k].item()
                    new_tokens = tokens + [top_idx[0, k].item()]
                    candidates.append((new_score, new_tokens, nh, nc))

            if not candidates:
                break

            
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        
        if completed:
            completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            return completed[0][1]

        best = max(beams, key=lambda x: x[0])
        return best[1][1:]
