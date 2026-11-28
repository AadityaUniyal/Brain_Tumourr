"""
Dual-Head Multi-Task Brain Tumor Network:
Jointly solves:
1. Semantic Segmentation: Lightweight attention-gated U-Net decoder with bilinear upsampling
2. Diagnostic Classification: Subtyping tumor as Meningioma, Glioma, or Pituitary
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


class UpsampleBlock(nn.Module):
    """Bilinear upsampling + 1x1 projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate for U-Net skip connections to focus on tumor regions."""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        g = self.W_g(gate)
        x = self.W_x(skip)
        attn = self.psi(g + x)
        return skip * attn


class DualHeadBrainTumorNet(nn.Module):
    """
    Dual-Head Architecture:
    - Backbone: Pretrained ResNet-34
    - Segmentation Branch: Compact attention-gated U-Net decoder (128->64->32->16)
    - Classification Branch: Adaptive pooling + projection head
    """

    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Multi-scale feature stages (ResNet-34 channels: 64, 64, 128, 256, 512)
        self.encoder_stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [B, 64, H/2, W/2]
        self.encoder_maxpool = resnet.maxpool                                      # [B, 64, H/4, W/4]
        self.encoder_stage1 = resnet.layer1                                        # [B, 64, H/4, W/4]
        self.encoder_stage2 = resnet.layer2                                        # [B, 128, H/8, W/8]
        self.encoder_stage3 = resnet.layer3                                        # [B, 256, H/16, W/16]
        self.encoder_stage4 = resnet.layer4                                        # [B, 512, H/32, W/32]

        # Classification Branch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Compact Attention-Gated Segmentation Decoder (fast on CPU, high spatial precision)
        self.up4 = UpsampleBlock(512, 128)
        self.att4 = AttentionGate(128, 256, 64)
        self.dec4 = DoubleConvBlock(128 + 256, 128)

        self.up3 = UpsampleBlock(128, 64)
        self.att3 = AttentionGate(64, 128, 32)
        self.dec3 = DoubleConvBlock(64 + 128, 64)

        self.up2 = UpsampleBlock(64, 48)
        self.att2 = AttentionGate(48, 64, 24)
        self.dec2 = DoubleConvBlock(48 + 64, 48)

        self.up1 = UpsampleBlock(48, 32)
        self.att1 = AttentionGate(32, 64, 16)
        self.dec1 = DoubleConvBlock(32 + 64, 32)

        self.up0 = UpsampleBlock(32, 24)
        self.dec0 = DoubleConvBlock(24, 24)

        self.seg_head = nn.Conv2d(24, 1, kernel_size=1)

    def forward(self, x):
        s0 = self.encoder_stage0(x)         # [B, 64, H/2, W/2]
        s1 = self.encoder_maxpool(s0)       # [B, 64, H/4, W/4]
        s1 = self.encoder_stage1(s1)        # [B, 64, H/4, W/4]
        s2 = self.encoder_stage2(s1)        # [B, 128, H/8, W/8]
        s3 = self.encoder_stage3(s2)        # [B, 256, H/16, W/16]
        s4 = self.encoder_stage4(s3)        # [B, 512, H/32, W/32]

        # 1. Classification Forward Pass
        embeddings = self.global_pool(s4).flatten(1)
        logits = self.classifier(embeddings)

        # 2. Attention-Gated Segmentation Forward Pass
        u4 = self.up4(s4)
        d4 = self.dec4(torch.cat([u4, self.att4(u4, s3)], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, self.att3(u3, s2)], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.att2(u2, s1)], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.att1(u1, s0)], dim=1))

        u0 = self.up0(d1)
        d0 = self.dec0(u0)

        seg_logits = self.seg_head(d0)

        return {
            "logits": logits,
            "seg_mask": seg_logits,
            "features": embeddings,
            "last_conv": s4,
        }
