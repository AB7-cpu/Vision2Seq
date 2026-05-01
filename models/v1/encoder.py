"""
models/v1/encoder.py — CNN Encoder (Show and Tell, V1).

Architecture
============
Backbone : ResNet-50 pretrained.
           The final classification layer is removed and will be trained on captions.
Pooling  : AdaptiveAvgPool2d → 2048-dim global average vector.
Projection: Linear(2048 → embed_dim) + ReLU + Dropout.

Fine-tuning strategy:

All ResNet-50 layers start frozen. Only layer3 and layer4 weights will be trained initially for transfer learning.

"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CNNEncoder(nn.Module):
    """
    ResNet-50 image encoder that produces a fixed-size embedding vector (256: same as embed_dim for LSTMDecoder).
    """

    def __init__(
        self,
        embed_dim : int   = 256,    # output embed_dim
        dropout   : float = 0.5,    # dropout probability
        fine_tune : bool  = True,   # True: unfreezes the last 2 layers of the ResNet50 model
    ):
        super().__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.embed_dim = embed_dim
        self._set_fine_tune(fine_tune)

    #     Fine-tuning

    def _set_fine_tune(self, enable: bool):
        """Freeze all weights, then optionally unfreeze the last 2 blocks."""
        for p in self.backbone.parameters():
            p.requires_grad = False

        if enable:
            # children()[6] = layer3,  children()[7] = layer4
            for block in list(self.backbone.children())[6:]:
                for p in block.parameters():
                    p.requires_grad = True

    def fine_tune(self, enable: bool = True):
        self._set_fine_tune(enable)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    #     Forward pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        x = self.backbone(images)   # (batch, 2048, 7, 7)
        x = self.pool(x)            # (batch, 2048, 1, 1)
        x = self.project(x)         # (batch, embed_dim)
        return x
