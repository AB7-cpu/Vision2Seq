"""
CNN Encoder (Transformer Captioning).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class TransformerEncoder(nn.Module):
    """
    ResNet-50 spatial encoder with a linear projection into Transformer space.

    Projects the (B, 2048, 7, 7) feature map to (B, 49, d_model) and adds a learned position embedding so the decoder can distinguish spatial locations.
    """

    def __init__(
        self,
        d_model    : int   = 512,    # Transformer model dimension
        dropout    : float = 0.1,
        fine_tune  : bool  = True,
    ):
        super().__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # 7×7 spatial output from layer4
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Project CNN channels (2048) → Transformer model dimension
        self.projection = nn.Sequential(
            nn.Linear(2048, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)

        self.dropout   = nn.Dropout(dropout)
        self.d_model   = d_model

        self._set_fine_tune(fine_tune)

    # Fine-tuning

    def _set_fine_tune(self, enable: bool):
        """Unfreeze only layer3 + layer4."""
        for p in self.backbone.parameters():
            p.requires_grad = False

        if enable:
            for block in list(self.backbone.children())[6:]:
                for p in block.parameters():
                    p.requires_grad = True

    def fine_tune(self, enable: bool = True):
        self._set_fine_tune(enable)

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    # Forward

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            memory: (B, 49, d_model)  — encoder memory for cross-attention.
        """
        x = self.backbone(images)              # (B, 2048, 7, 7)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, 49, 2048)
        x = self.projection(x)                 # (B, 49, d_model)
        x = self.dropout(x + self.pos_embed)
        return x
