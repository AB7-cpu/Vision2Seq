"""
CNN Encoder (Show, Attend and Tell).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class AttentionEncoder(nn.Module):
    """
    ResNet-50 encoder that outputs a spatial feature grid.
    """

    def __init__(
        self,
        encoder_dim : int  = 2048,
        fine_tune   : bool = True,
    ):
        super().__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.encoder_dim = encoder_dim
        self._set_fine_tune(fine_tune)

    # Fine-tuning

    def _set_fine_tune(self, enable: bool):
        """unfreeze only layer3 + layer4."""
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
        x = self.backbone(images)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, H * W, C)
        return x
