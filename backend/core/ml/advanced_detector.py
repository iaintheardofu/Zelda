"""
Advanced Signal Detection Architecture
Combines state-of-the-art techniques for RF signal detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SqueezeExcitation1d(nn.Module):
    """Squeeze-and-Excitation block for 1D signals"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, L)
        # Global average pooling
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)

        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)

        # Scale
        return x * y.expand_as(x)


class MultiHeadSelfAttention1D(nn.Module):
    """Multi-head self-attention for 1D signals"""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape

        # Reshape to (B, L, C) for attention
        x = x.transpose(1, 2)

        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        # Output projection
        x = self.proj(x)

        # Back to (B, C, L)
        return x.transpose(1, 2)


class DilatedConvBlock(nn.Module):
    """Dilated convolution block with multiple dilation rates"""

    def __init__(self, in_channels: int, out_channels: int, dilations=[1, 2, 4, 8]):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(dilations),
                      kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Apply each dilated conv and concatenate
        outputs = [conv(x) for conv in self.convs]
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        return F.relu(x)


class ResidualBlock(nn.Module):
    """Residual block with SE and optional attention"""

    def __init__(self, channels: int, use_attention: bool = False):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = SqueezeExcitation1d(channels)

        self.use_attention = use_attention
        if use_attention:
            self.attention = MultiHeadSelfAttention1D(channels, num_heads=8)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze-Excitation
        out = self.se(out)

        # Optional attention
        if self.use_attention:
            out = self.attention(out)

        out += residual
        out = F.relu(out)

        return out


class UltraDetector(nn.Module):
    """
    Ultra-advanced signal detector combining:
    - Multi-scale dilated convolutions
    - Squeeze-Excitation blocks
    - Multi-head self-attention
    - Residual connections
    - Multi-task learning
    """

    def __init__(
        self,
        input_length: int = 4096,
        num_classes: int = 1,  # Binary detection
        base_channels: int = 64,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_length = input_length

        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        # Multi-scale dilated convolutions
        self.dilated_block1 = DilatedConvBlock(base_channels, base_channels * 2)

        # Residual blocks with increasing channels
        self.res1 = ResidualBlock(base_channels * 2, use_attention=False)
        self.pool1 = nn.MaxPool1d(2)

        self.res2 = ResidualBlock(base_channels * 2, use_attention=False)
        self.pool2 = nn.MaxPool1d(2)

        # Dilated block at intermediate scale
        self.dilated_block2 = DilatedConvBlock(base_channels * 2, base_channels * 4)

        # Deep residual blocks with attention
        self.res3 = ResidualBlock(base_channels * 4, use_attention=use_attention)
        self.res4 = ResidualBlock(base_channels * 4, use_attention=use_attention)
        self.pool3 = nn.MaxPool1d(2)

        # Final dilated block
        self.dilated_block3 = DilatedConvBlock(base_channels * 4, base_channels * 8)

        self.res5 = ResidualBlock(base_channels * 8, use_attention=use_attention)
        self.res6 = ResidualBlock(base_channels * 8, use_attention=use_attention)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8 * 2, 512),  # *2 for avg+max pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Auxiliary task: Signal strength estimation
        self.strength_head = nn.Linear(base_channels * 8 * 2, 1)

    def forward(self, x, return_features: bool = False):
        # Stem
        x = self.stem(x)  # -> (B, 64, L/2)

        # Multi-scale features
        x = self.dilated_block1(x)  # -> (B, 128, L/2)

        # Progressive depth
        x = self.res1(x)
        x = self.pool1(x)  # -> (B, 128, L/4)

        x = self.res2(x)
        x = self.pool2(x)  # -> (B, 128, L/8)

        x = self.dilated_block2(x)  # -> (B, 256, L/8)

        x = self.res3(x)
        x = self.res4(x)
        x = self.pool3(x)  # -> (B, 256, L/16)

        x = self.dilated_block3(x)  # -> (B, 512, L/16)

        x = self.res5(x)
        x = self.res6(x)

        # Global pooling (both avg and max)
        avg_pool = self.global_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        features = torch.cat([avg_pool, max_pool], dim=1)

        # Main output
        out = self.fc(features)

        # Auxiliary output
        strength = self.strength_head(features)

        if return_features:
            return out, strength, features
        else:
            return out, strength


class EnsembleDetector(nn.Module):
    """Ensemble of multiple detectors for robustness"""

    def __init__(self, num_models: int = 3, **kwargs):
        super().__init__()

        self.models = nn.ModuleList([
            UltraDetector(**kwargs)
            for _ in range(num_models)
        ])

    def forward(self, x):
        outputs = []
        strengths = []

        for model in self.models:
            out, strength = model(x)
            outputs.append(out)
            strengths.append(strength)

        # Average predictions
        final_out = torch.stack(outputs).mean(dim=0)
        final_strength = torch.stack(strengths).mean(dim=0)

        return final_out, final_strength


class TransformerDetector(nn.Module):
    """Transformer-based signal detector"""

    def __init__(
        self,
        input_length: int = 4096,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        num_classes: int = 1,
    ):
        super().__init__()

        # Patch embedding
        self.patch_size = 16
        self.num_patches = input_length // self.patch_size

        self.patch_embed = nn.Conv1d(2, d_model, kernel_size=self.patch_size, stride=self.patch_size)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 2, L)

        # Patch embedding
        x = self.patch_embed(x)  # -> (B, d_model, num_patches)
        x = x.transpose(1, 2)  # -> (B, num_patches, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        out = self.fc(x)

        return out, torch.zeros_like(out)  # Dummy strength for compatibility


def create_model(model_type: str = "ultra", **kwargs):
    """Factory function to create models"""

    if model_type == "ultra":
        return UltraDetector(**kwargs)
    elif model_type == "ensemble":
        return EnsembleDetector(**kwargs)
    elif model_type == "transformer":
        return TransformerDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    batch_size = 4
    input_length = 4096

    x = torch.randn(batch_size, 2, input_length)

    print("Testing UltraDetector...")
    model = UltraDetector(input_length=input_length)
    out, strength = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}, Strength: {strength.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting TransformerDetector...")
    model_tf = TransformerDetector(input_length=input_length)
    out_tf, _ = model_tf(x)
    print(f"Output: {out_tf.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_tf.parameters()):,}")
