"""
Dual-Head Multi-Task Brain Tumor Network:
Jointly solves:
1. Semantic Segmentation: Delineating the exact tumor boundary mask
2. Diagnostic Classification: Subtyping tumor as Glioma, Meningioma, or Pituitary
"""
import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and GELU activations."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class DualHeadBrainTumorNet(nn.Module):
    """
    Dual-Head Architecture:
    - Backbone: Pretrained ResNet50 (or customized feature extractor)
    - Segmentation Branch: Decoder with lateral skip-connections for spatial mask recovery
    - Classification Branch: Adaptive pooling + projection head for multi-class diagnostics
    """

    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes

        # Encoder Backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Multi-scale feature stages
        self.encoder_stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [B, 64, H/2, W/2]
        self.encoder_maxpool = resnet.maxpool                                      # [B, 64, H/4, W/4]
        self.encoder_stage1 = resnet.layer1                                        # [B, 256, H/4, W/4]
        self.encoder_stage2 = resnet.layer2                                        # [B, 512, H/8, W/8]
        self.encoder_stage3 = resnet.layer3                                        # [B, 1024, H/16, W/16]
        self.encoder_stage4 = resnet.layer4                                        # [B, 2048, H/32, W/32]

        # Classification Branch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Segmentation Decoder (U-Net style with skip connections)
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(2048, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(128, 64)

        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = DoubleConvBlock(32, 32)

        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)  # Binary mask logits

    def forward(self, x):
        # Multi-scale Encoder
        s0 = self.encoder_stage0(x)         # [B, 64, H/2, W/2]
        s1 = self.encoder_maxpool(s0)       # [B, 64, H/4, W/4]
        s1 = self.encoder_stage1(s1)        # [B, 256, H/4, W/4]
        s2 = self.encoder_stage2(s1)        # [B, 512, H/8, W/8]
        s3 = self.encoder_stage3(s2)        # [B, 1024, H/16, W/16]
        s4 = self.encoder_stage4(s3)        # [B, 2048, H/32, W/32]

        # 1. Classification Forward Pass
        embeddings = self.global_pool(s4).flatten(1)
        logits = self.classifier(embeddings)

        # 2. Segmentation Forward Pass
        u4 = self.up4(s4)
        d4 = self.dec4(torch.cat([u4, s3], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, s2], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, s1], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, s0], dim=1))

        u0 = self.up0(d1)
        d0 = self.dec0(u0)

        seg_logits = self.seg_head(d0)

        return {
            "logits": logits,
            "seg_mask": seg_logits,
            "features": embeddings,
            "last_conv": s4,
        }
