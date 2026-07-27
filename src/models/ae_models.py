# src/models/ae_models.py
"""
3D autoencoder architectures for cardiac MRI representation learning.
"""

import torch
import torch.nn as nn

# Blocks original 

class Conv3DBlock(nn.Module):
    """
    3D convolutional block:
    Conv3D -> InstanceNorm3D -> ReLU -> Conv3D -> InstanceNorm3D -> ReLU
    Optionally followed by MaxPool3D for downsampling.
    """

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        if self.downsample:
            x = self.pool(x)

        return x


class UpConv3DBlock(nn.Module):
    """
    3D decoder block:
    ConvTranspose3D (upsampling) -> Conv3D -> InstanceNorm3D -> ReLU -> Conv3D -> InstanceNorm3D -> ReLU
    No skip connections.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv1 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return x

# Blocks AI Agent

class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D convolutions.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AttentionConv3DBlock(nn.Module):
    """
    3D convolutional block with Squeeze-and-Excitation attention.
    """
    def __init__(self, in_channels, out_channels, downsample=True, reduction=16):
        super().__init__()
        self.conv_block = Conv3DBlock(in_channels, out_channels, downsample=downsample)
        self.se = SEBlock3D(out_channels, reduction)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.se(x)
        return x

class DilatedConv3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, downsample=True):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        if self.downsample:
            x = self.pool(x)

        return x

class DilatedAttentionConv3DBlock(nn.Module):
    """
    3D dilated convolutional block with Squeeze-and-Excitation attention.
    """
    def __init__(self, in_channels, out_channels, dilation=1, downsample=True, reduction=16):
        super().__init__()
        self.conv_block = DilatedConv3DBlock(in_channels, out_channels, dilation=dilation, downsample=downsample)
        self.se = SEBlock3D(out_channels, reduction)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.se(x)
        return x


class SeparableConv3DBlock(nn.Module):
    """
    3D Depthwise Separable Convolutional block:
    (Depthwise -> Pointwise) -> Norm -> ReLU -> (Depthwise -> Pointwise) -> Norm -> ReLU
    Optionally followed by MaxPool3D for downsampling.
    """
    def __init__(self, in_channels, out_channels, dilation=1, downsample=True):
        super().__init__()

        # Layer 1
        self.depthwise1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=in_channels
        )
        self.pointwise1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2
        self.depthwise2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=out_channels
        )
        self.pointwise2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        if self.downsample:
            x = self.pool(x)

        return x

# class ResidualSeparableConv3DBlock(nn.Module):
#     """
#     3D Residual Depthwise Separable Convolutional block:
#     Residual connection: x = x + Conv(x)
#     """
#     def __init__(self, in_channels, out_channels, dilation=1, downsample=True):
#         super().__init__()
#         self.downsample = downsample
#         self.conv = SeparableConv3DBlock(in_channels, out_channels, dilation=dilation, downsample=downsample)
        
#         # Projection for residual to match dimensions
#         if in_channels != out_channels or downsample:
#             stride = 2 if downsample else 1
#             self.residual = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.InstanceNorm3d(out_channels)
#             )
#         else:
#             self.residual = nn.Identity()

#     def forward(self, x):
#         return self.conv(x) + self.residual(x)

# class MultiScaleDilatedConv3DBlock(nn.Module):
#     """
#     Multi-scale Depthwise Separable Convolutional block:
#     Parallel paths with different dilations fused together.
#     """
#     def __init__(self, in_channels, out_channels, downsample=True):
#         super().__init__()
#         mid_channels = out_channels // 3
#         self.path1 = SeparableConv3DBlock(in_channels, mid_channels, dilation=1, downsample=False)
#         self.path2 = SeparableConv3DBlock(in_channels, mid_channels, dilation=2, downsample=False)
#         self.path3 = SeparableConv3DBlock(in_channels, out_channels - 2*mid_channels, dilation=4, downsample=False)
        
#         self.fusion = nn.Conv3d(out_channels, out_channels, kernel_size=1)
#         self.norm = nn.InstanceNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#         self.downsample = downsample
#         if self.downsample:
#             self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

#     def forward(self, x):
#         p1 = self.path1(x)
#         p2 = self.path2(x)
#         p3 = self.path3(x)
        
#         x = torch.cat([p1, p2, p3], dim=1)
#         x = self.fusion(x)
#         x = self.norm(x)
#         x = self.relu(x)

#         if self.downsample:
#             x = self.pool(x)
#         return x

# Classes original

class AutoEncoder3D_Current(nn.Module):
    """
    Original AE model:
    (1,32,128,128)
    -> (8,16,64,64)
    -> (16,8,32,32)
    -> (32,4,16,16)
    -> bottleneck conv -> (64,4,16,16)
    -> flatten 65536 -> latent_dim
    """

    def __init__(self, latent_dim=10, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder
        self.enc1 = Conv3DBlock(in_channels=1, out_channels=8, downsample=True)
        self.enc2 = Conv3DBlock(in_channels=8, out_channels=16, downsample=True)
        self.enc3 = Conv3DBlock(in_channels=16, out_channels=32, downsample=True)

        self.bottleneck_conv = Conv3DBlock(in_channels=32, out_channels=64, downsample=False)

        self.feature_shape = (64, 4, 16, 16)
        flattened_size = 64 * 4 * 16 * 16  # 65536

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.dec1 = UpConv3DBlock(in_channels=64, out_channels=32)
        self.dec2 = UpConv3DBlock(in_channels=32, out_channels=16)
        self.dec3 = UpConv3DBlock(in_channels=16, out_channels=8)

        self.final_conv = nn.Conv3d(
            in_channels=8,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)              # -> (B, 8, 16, 64, 64)
        x = self.enc2(x)              # -> (B, 16, 8, 32, 32)
        x = self.enc3(x)              # -> (B, 32, 4, 16, 16)
        x = self.bottleneck_conv(x)   # -> (B, 64, 4, 16, 16)

        x = self.flatten(x)           # -> (B, 65536)
        x = self.dropout(x)
        z = self.fc_enc(x)            # -> (B, latent_dim)
        return z

    def decode(self, z):
        x = self.fc_dec(z)            # -> (B, 65536)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)

        x = self.dec1(x)              # -> (B, 32, 8, 32, 32)
        x = self.dec2(x)              # -> (B, 16, 16, 64, 64)
        x = self.dec3(x)              # -> (B, 8, 32, 128, 128)

        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoder3D_FCDeep(nn.Module):
    """
    Model A:
    Progressive compression down to (128,1,4,4),
    then flatten -> latent vector -> linear decode.
    """

    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder
        self.enc1 = Conv3DBlock(1, 8, downsample=True)      # -> (8,16,64,64)
        self.enc2 = Conv3DBlock(8, 16, downsample=True)     # -> (16,8,32,32)
        self.enc3 = Conv3DBlock(16, 32, downsample=True)    # -> (32,4,16,16)
        self.enc4 = Conv3DBlock(32, 64, downsample=True)    # -> (64,2,8,8)

        # Last compression without isotropic pooling because depth=2
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )                                                   # -> (128,2,8,8)

        self.final_down = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )                                                   # -> (128,1,4,4)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)   # ← nouveau
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.initial_up = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )                                                   # -> (128,2,8,8)

        self.dec1 = UpConv3DBlock(128, 64)   # -> (64,4,16,16)
        self.dec2 = UpConv3DBlock(64, 32)    # -> (32,8,32,32)
        self.dec3 = UpConv3DBlock(32, 16)    # -> (16,16,64,64)
        self.dec4 = UpConv3DBlock(16, 8)     # -> (8,32,128,128)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)  
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_Conv(nn.Module):
    """
    Model B:
    Fully convolutional bottleneck with shape (C,1,2,2),
    where latent_dim = 4 * C.
    No linear layers.
    """

    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128)):
        super().__init__()

        if latent_dim % 4 != 0:
            raise ValueError("For AutoEncoder3D_Conv, latent_dim must be a multiple of 4.")

        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.latent_channels = latent_dim // 4

        # Encoder
        self.enc1 = Conv3DBlock(1, 8, downsample=True)      # -> (8,16,64,64)
        self.enc2 = Conv3DBlock(8, 16, downsample=True)     # -> (16,8,32,32)
        self.enc3 = Conv3DBlock(16, 32, downsample=True)    # -> (32,4,16,16)
        self.enc4 = Conv3DBlock(32, 64, downsample=True)    # -> (64,2,8,8)

        # Bottleneck reduction:
        # (64,2,8,8) -> (128,2,8,8) -> (128,1,4,4) -> (C,1,2,2)
        self.pre_latent = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.reduce_to_1x4x4 = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2)
        )   # (128,2,8,8) -> (128,1,4,4)

        self.reduce_to_latent = nn.Conv3d(
            in_channels=128,
            out_channels=self.latent_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )   # (128,1,4,4) -> (C,1,2,2)

        # Decoder bottleneck inverse:
        # (C,1,2,2) -> (128,1,4,4) -> (128,2,8,8)
        self.expand_from_latent = nn.Sequential(
            nn.ConvTranspose3d(
                self.latent_channels,
                128,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
            ),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )   # -> (128,1,4,4)

        self.expand_to_2x8x8 = nn.Sequential(
            nn.ConvTranspose3d(
                128,
                128,
                kernel_size=(2, 2, 2),
                stride=(2, 2, 2)
            ),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )   # -> (128,2,8,8)

        self.dec1 = UpConv3DBlock(128, 64)   # -> (64,4,16,16)
        self.dec2 = UpConv3DBlock(64, 32)    # -> (32,8,32,32)
        self.dec3 = UpConv3DBlock(32, 16)    # -> (16,16,64,64)
        self.dec4 = UpConv3DBlock(16, 8)     # -> (8,32,128,128)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode_tensor(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.pre_latent(x)           # -> (128,2,8,8)
        x = self.reduce_to_1x4x4(x)      # -> (128,1,4,4)
        z_tensor = self.reduce_to_latent(x)  # -> (B,C,1,2,2)

        return z_tensor

    def encode(self, x):
        z_tensor = self.encode_tensor(x)
        z = z_tensor.flatten(start_dim=1)  # -> (B, latent_dim)
        return z

    def decode_from_tensor(self, z_tensor):
        x = self.expand_from_latent(z_tensor)   # -> (128,1,4,4)
        x = self.expand_to_2x8x8(x)             # -> (128,2,8,8)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def decode(self, z):
        z_tensor = z.view(-1, self.latent_channels, 1, 2, 2)
        x = self.decode_from_tensor(z_tensor)
        return x

    def forward(self, x):
        z_tensor = self.encode_tensor(x)
        z = z_tensor.flatten(start_dim=1)
        x_recon = self.decode_from_tensor(z_tensor)
        return x_recon, z

class AutoEncoder3D_Linear(nn.Module):
    """
    Purely linear autoencoder — no convolutions, no activations.
 
    Encoder: flatten -> nn.Linear(input_size, latent_dim)
    Decoder: nn.Linear(latent_dim, input_size) -> reshape
 
    With MSE loss and no non-linearities, this is theoretically equivalent
    to PCA: the learned subspace should converge to the top-k principal
    components (up to rotation within the subspace).
 
    Input shape : (B, 1, 32, 128, 128)
    Latent shape : (B, latent_dim)
    """
 
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128)):
        super().__init__()
 
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.input_size = 1
        for dim in input_shape:
            self.input_size *= dim  # 1 * 32 * 128 * 128 = 524288
 
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(self.input_size, latent_dim, bias=True)
        self.fc_dec = nn.Linear(latent_dim, self.input_size, bias=True)
 
    def encode(self, x):
        x = self.flatten(x)       # -> (B, 524288)
        z = self.fc_enc(x)        # -> (B, latent_dim)
        return z
 
    def decode(self, z):
        x = self.fc_dec(z)                     # -> (B, 524288)
        x = x.view(-1, *self.input_shape)      # -> (B, 1, 32, 128, 128)
        return x
 
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_FCDeep_VAE(nn.Module):
    """
    Variational Autoencoder based on AE3dFCDeep architecture.
    Identical encoder/decoder conv blocks — only the bottleneck differs:
    fc_enc is replaced by fc_mu + fc_logvar, with reparameterization trick.

    During training : z is sampled from N(mu, sigma²)
    During eval     : z = mu (deterministic, no sampling noise)
    """

    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # ── Encoder (identique à AE3dFCDeep) ─────────────────────────────────
        self.enc1 = Conv3DBlock(1, 8, downsample=True)
        self.enc2 = Conv3DBlock(8, 16, downsample=True)
        self.enc3 = Conv3DBlock(16, 32, downsample=True)
        self.enc4 = Conv3DBlock(32, 64, downsample=True)

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.final_down = nn.Conv3d(128, 128, kernel_size=2, stride=2)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)

        # ── Bottleneck VAE : 2 instead of 1 ─────────────────────────
        self.fc_mu     = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

        # ── Decoder (identical to  AE3dFCDeep) ─────────────────────────────────
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        self.dec1 = UpConv3DBlock(128, 64)
        self.dec2 = UpConv3DBlock(64, 32)
        self.dec3 = UpConv3DBlock(32, 16)
        self.dec4 = UpConv3DBlock(16, 8)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick : z = mu + eps * sigma
        During eval (model.eval()), sampling is disabled → z = mu
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu   # déterministe pendant val/test

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar

# Classes AI Agent

class AutoEncoder3D_Attention(nn.Module):
    """
    Attention-guided 3D AE (sibling of AE3dFCDeep with SE blocks).
    Same progressive compression, same depth, same channels.
    All Conv3DBlocks replaced by AttentionConv3DBlocks.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0, reduction=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder (Attention blocks)
        self.enc1 = AttentionConv3DBlock(1, 8, downsample=True, reduction=reduction)
        self.enc2 = AttentionConv3DBlock(8, 16, downsample=True, reduction=reduction)
        self.enc3 = AttentionConv3DBlock(16, 32, downsample=True, reduction=reduction)
        self.enc4 = AttentionConv3DBlock(32, 64, downsample=True, reduction=reduction)

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.final_down = nn.Conv3d(128, 128, kernel_size=2, stride=2)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        self.dec1 = UpConv3DBlock(128, 64)
        self.dec2 = UpConv3DBlock(64, 32)
        self.dec3 = UpConv3DBlock(32, 16)
        self.dec4 = UpConv3DBlock(16, 8)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_Dilated(nn.Module):
    """
    Dilated Convolutional Autoencoder.
    Uses dilated convolutions in the encoder to increase receptive field.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder (Dilated)
        # Layer 1: dilation=1
        self.enc1 = DilatedConv3DBlock(1, 8, dilation=1, downsample=True)
        # Layer 2: dilation=2
        self.enc2 = DilatedConv3DBlock(8, 16, dilation=2, downsample=True)
        # Layer 3: dilation=4
        self.enc3 = DilatedConv3DBlock(16, 32, dilation=4, downsample=True)
        # Layer 4: dilation=1 (to stabilize)
        self.enc4 = DilatedConv3DBlock(32, 64, dilation=1, downsample=True)

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.final_down = nn.Conv3d(128, 128, kernel_size=2, stride=2)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        self.dec1 = UpConv3DBlock(128, 64)
        self.dec2 = UpConv3DBlock(64, 32)
        self.dec3 = UpConv3DBlock(32, 16)
        self.dec4 = UpConv3DBlock(16, 8)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_DilatedAttention(nn.Module):
    """
    Dilated Convolutional Autoencoder with Squeeze-and-Excitation attention.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0, reduction=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder (Dilated + Attention)
        self.enc1 = DilatedAttentionConv3DBlock(1, 8, dilation=1, downsample=True, reduction=reduction)
        self.enc2 = DilatedAttentionConv3DBlock(8, 16, dilation=2, downsample=True, reduction=reduction)
        self.enc3 = DilatedAttentionConv3DBlock(16, 32, dilation=4, downsample=True, reduction=reduction)
        self.enc4 = DilatedAttentionConv3DBlock(32, 64, dilation=1, downsample=True, reduction=reduction)

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.final_down = nn.Conv3d(128, 128, kernel_size=2, stride=2)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        self.dec1 = UpConv3DBlock(128, 64)
        self.dec2 = UpConv3DBlock(64, 32)
        self.dec3 = UpConv3DBlock(32, 16)
        self.dec4 = UpConv3DBlock(16, 8)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_SeparableDilated(nn.Module):
    """
    Separable Dilated Convolutional Autoencoder.
    Uses Depthwise Separable Dilated convolutions in the encoder to reduce
    parameter count while maintaining large receptive fields.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder (Separable Dilated)
        self.enc1 = SeparableConv3DBlock(1, 8, dilation=1, downsample=True)
        self.enc2 = SeparableConv3DBlock(8, 16, dilation=2, downsample=True)
        self.enc3 = SeparableConv3DBlock(16, 32, dilation=4, downsample=True)
        self.enc4 = SeparableConv3DBlock(32, 64, dilation=1, downsample=True)

        self.bottleneck_conv = nn.Sequential(
            SeparableConv3DBlock(64, 128, dilation=1, downsample=False),
            SeparableConv3DBlock(128, 128, dilation=1, downsample=False),
        )

        self.final_down = nn.Conv3d(128, 128, kernel_size=2, stride=2)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        self.dec1 = UpConv3DBlock(128, 64)
        self.dec2 = UpConv3DBlock(64, 32)
        self.dec3 = UpConv3DBlock(32, 16)
        self.dec4 = UpConv3DBlock(16, 8)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class AutoEncoder3D_FCDeepAsym(nn.Module):
    """
    AE3dFCDeep with anisotropic stage-1 downsampling.
    Stage 1 uses (1,2,2) pooling to preserve z-depth (32 slices → 32);
    stages 2–4 use isotropic (2,2,2) pooling. FC bottleneck identical to champion.
    Rationale: cardiac MRI z-axis (32) is 4x smaller than y/x (128); standard
    (2,2,2) pooling in stage 1 immediately discards half the z-slices, losing
    early 3D context. Preserving z through stage 1 gives later blocks richer
    volumetric features before isotropic compression begins.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder: stage 1 uses anisotropic (1,2,2) pool; stages 2-4 isotropic
        self.enc1 = Conv3DBlock(1, 8, downsample=False)                  # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 8×32×64×64
        self.enc2 = Conv3DBlock(8, 16, downsample=True)                  # 16×16×32×32
        self.enc3 = Conv3DBlock(16, 32, downsample=True)                 # 32×8×16×16
        self.enc4 = Conv3DBlock(32, 64, downsample=True)                 # 64×4×8×8

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )                                                                # 128×4×8×8

        # Asymmetric final compression: (4,8,8) → (1,4,4) via kernel/stride (4,2,2)
        self.final_down = nn.Conv3d(128, 128, kernel_size=(4, 2, 2), stride=(4, 2, 2))

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder: symmetric asymmetric upsample
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=(4, 2, 2), stride=(4, 2, 2))

        self.dec1 = UpConv3DBlock(128, 64)   # 64×8×16×16
        self.dec2 = UpConv3DBlock(64, 32)    # 32×16×32×32
        self.dec3 = UpConv3DBlock(32, 16)    # 16×32×64×64

        # Anisotropic final decode: (1,2,2) upsample only y/x
        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = Conv3DBlock(16, 8, downsample=False)            # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoder3D_FCDeepAsymV2(nn.Module):
    """
    AE3dFCDeepAsym with two-step z-collapse before FC.
    Champion uses final_down(kernel=(4,2,2)) to collapse z 4→1 in one step.
    This model inserts MaxPool3d(2,1,1) between bottleneck_conv and final_down,
    then replaces final_down with Conv3d(k=2,s=2) (identical to AE3dFCDeep).
    Two-step z-collapse (4→2→1) helps dim=8 by spreading the z-compression
    burden; retains the anisotropic stage-1 benefit for large dims.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder: anisotropic stage-1 pool identical to AE3dFCDeepAsym
        self.enc1 = Conv3DBlock(1, 8, downsample=False)                  # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 8×32×64×64
        self.enc2 = Conv3DBlock(8, 16, downsample=True)                  # 16×16×32×32
        self.enc3 = Conv3DBlock(16, 32, downsample=True)                 # 32×8×16×16
        self.enc4 = Conv3DBlock(32, 64, downsample=True)                 # 64×4×8×8

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )                                                                # 128×4×8×8

        # Two-step z-collapse: first halve z via MaxPool (4→2), then compress via Conv3d (2→1)
        self.z_pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))  # 128×2×8×8
        self.final_down = nn.Conv3d(128, 128, kernel_size=2, stride=2)        # 128×1×4×4

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder: reverse the two-step z-collapse
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)  # 128×2×8×8
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)  # 128×4×8×8

        self.dec1 = UpConv3DBlock(128, 64)   # 64×8×16×16
        self.dec2 = UpConv3DBlock(64, 32)    # 32×16×32×32
        self.dec3 = UpConv3DBlock(32, 16)    # 16×32×64×64

        # Anisotropic final decode: (1,2,2) upsample only y/x (mirrors pool1)
        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = Conv3DBlock(16, 8, downsample=False)            # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.z_pool(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoder3D_FCDeepAsymV4(nn.Module):
    """
    AE3dFCDeepAsymV2 with MaxPool(2,1,1) z-halving moved earlier:
    between enc3 and enc4 instead of between bottleneck_conv and final_down.
    After the z-pool enc4 processes z=4 features, and bottleneck_conv operates
    on 128×2×8×8 — the same z-context as AE3dFCDeep (which achieves 0.772 at dim=8).
    The asymmetric stage-1 encoding (pool1=(1,2,2)) still preserves z-diversity
    through enc2 and enc3, maintaining the benefit for large dims. Decoder mirrors V2.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder: anisotropic stage-1 pool then extra z-halving before enc4
        self.enc1 = Conv3DBlock(1, 8, downsample=False)                   # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))   # 8×32×64×64
        self.enc2 = Conv3DBlock(8, 16, downsample=True)                   # 16×16×32×32
        self.enc3 = Conv3DBlock(16, 32, downsample=True)                  # 32×8×16×16
        # z-halving here: z 8→4 before enc4, then enc4's MaxPool(2,2,2) gives z 4→2
        self.z_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))  # 32×4×16×16
        self.enc4 = Conv3DBlock(32, 64, downsample=True)                  # 64×2×8×8

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )                                                                 # 128×2×8×8

        # Same final compression as AE3dFCDeep: 2→1 in z, 8→4 in spatial
        self.final_down = nn.Conv3d(128, 128, kernel_size=2, stride=2)   # 128×1×4×4

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder: mirrors V2 (initial_up + z_up restores 128×1×4×4 → 128×4×8×8)
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)  # 128×2×8×8
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)  # 128×4×8×8

        self.dec1 = UpConv3DBlock(128, 64)   # 64×8×16×16
        self.dec2 = UpConv3DBlock(64, 32)    # 32×16×32×32
        self.dec3 = UpConv3DBlock(32, 16)    # 16×32×64×64

        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = Conv3DBlock(16, 8, downsample=False)            # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.z_pool3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ResConv3DBlock(nn.Module):
    """
    Residual 3D encoder block: output = ReLU(F(x) + shortcut(x)).
    F(x) = Conv3d->IN->ReLU->Conv3d->IN. Shortcut is 1x1x1 conv when channels differ.
    MaxPool3d downsampling is applied after the residual sum (not inside the residual path).
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu_out = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu_out(out + residual)
        if self.downsample:
            out = self.pool(out)
        return out


class ResUpConv3DBlock(nn.Module):
    """
    Residual 3D decoder block: ConvTranspose3d upsampling, then residual conv block.
    After upsampling, channels are fixed so shortcut is Identity.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu_out(out + residual)
        return out


class AutoEncoder3D_AsymResidual(nn.Module):
    """
    AE3dFCDeepAsymV4 with residual blocks in encoder and decoder.
    Pooling structure identical to V4: anisotropic pool1=(1,2,2) at stage 1,
    z_pool3=(2,1,1) between enc3 and enc4. Encoder Conv3DBlocks replaced
    by ResConv3DBlocks; decoder UpConv3DBlocks replaced by ResUpConv3DBlocks.
    No encoder-to-decoder skip connections.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder: identical pooling to V4, residual conv blocks
        self.enc1 = ResConv3DBlock(1, 8, downsample=False)                    # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))    # 8×32×64×64
        self.enc2 = ResConv3DBlock(8, 16, downsample=True)                    # 16×16×32×32
        self.enc3 = ResConv3DBlock(16, 32, downsample=True)                   # 32×8×16×16
        self.z_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)) # 32×4×16×16
        self.enc4 = ResConv3DBlock(32, 64, downsample=True)                   # 64×2×8×8

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )                                                                      # 128×2×8×8

        self.final_down = nn.Conv3d(128, 128, 2, 2)                           # 128×1×4×4

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder: mirrors V4 structure with residual blocks
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, 2, 2)                  # 128×2×8×8
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)  # 128×4×8×8

        self.dec1 = ResUpConv3DBlock(128, 64)   # 64×8×16×16
        self.dec2 = ResUpConv3DBlock(64, 32)    # 32×16×32×32
        self.dec3 = ResUpConv3DBlock(32, 16)    # 16×32×64×64

        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = ResConv3DBlock(16, 8, downsample=False)              # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.z_pool3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoder3D_AsymResidualV4(nn.Module):
    """
    AE3dAsymResidual (trial 12) with enc4 changed from ResConv3DBlock to plain Conv3DBlock.
    enc1-enc3 keep residual shortcuts; enc4 (post-z_pool3) uses a standard conv block
    to allow freer transformation of z-compressed features. Decoder and bottleneck identical to trial 12.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        self.enc1 = ResConv3DBlock(1, 8, downsample=False)                     # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))    # 8×32×64×64
        self.enc2 = ResConv3DBlock(8, 16, downsample=True)                    # 16×16×32×32
        self.enc3 = ResConv3DBlock(16, 32, downsample=True)                   # 32×8×16×16
        self.z_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)) # 32×4×16×16
        self.enc4 = Conv3DBlock(32, 64, downsample=True)                      # 64×2×8×8 — plain conv

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )                                                                       # 128×2×8×8

        self.final_down = nn.Conv3d(128, 128, 2, 2)                            # 128×1×4×4

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, 2, 2)                  # 128×2×8×8
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)  # 128×4×8×8

        self.dec1 = ResUpConv3DBlock(128, 64)   # 64×8×16×16
        self.dec2 = ResUpConv3DBlock(64, 32)    # 32×16×32×32
        self.dec3 = ResUpConv3DBlock(32, 16)    # 16×32×64×64

        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = ResConv3DBlock(16, 8, downsample=False)               # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.z_pool3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ResSeparableConv3DBlock(nn.Module):
    """
    Residual 3D block with depthwise separable convolutions.
    output = ReLU(DW+PW path(x) + shortcut(x)). Shortcut = 1×1×1 conv when channels differ.
    MaxPool applied after residual sum (not inside the residual path).
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.dw1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.pw1 = nn.Conv3d(in_channels, out_channels, 1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dw2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels)
        self.pw2 = nn.Conv3d(out_channels, out_channels, 1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu_out = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dw1(x)
        out = self.pw1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dw2(out)
        out = self.pw2(out)
        out = self.norm2(out)
        out = self.relu_out(out + residual)
        if self.downsample:
            out = self.pool(out)
        return out


class AutoEncoder3D_AsymSeparable(nn.Module):
    """
    Depthwise separable convolutions with V4's anisotropic pooling.
    Trial 19 Exploration: SeparableConv3DBlock encoder + V4 pooling (pool1=(1,2,2), z_pool3=(2,1,1)).
    Bottleneck/FC identical to V4 (flattened_size=2048). Plain UpConv3DBlock decoder.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        self.enc1 = SeparableConv3DBlock(1, 8, downsample=False)                  # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))       # 8×32×64×64
        self.enc2 = SeparableConv3DBlock(8, 16, downsample=True)                  # 16×16×32×32
        self.enc3 = SeparableConv3DBlock(16, 32, downsample=True)                 # 32×8×16×16
        self.z_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))    # 32×4×16×16
        self.enc4 = SeparableConv3DBlock(32, 64, downsample=True)                 # 64×2×8×8

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )                                                                           # 128×2×8×8

        self.final_down = nn.Conv3d(128, 128, 2, 2)                               # 128×1×4×4

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, 2, 2)                     # 128×2×8×8
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)  # 128×4×8×8

        self.dec1 = UpConv3DBlock(128, 64)                                        # 64×8×16×16
        self.dec2 = UpConv3DBlock(64, 32)                                         # 32×16×32×32
        self.dec3 = UpConv3DBlock(32, 16)                                         # 16×32×64×64

        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = Conv3DBlock(16, 8, downsample=False)                     # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.z_pool3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoder3D_AsymResSeparable(nn.Module):
    """
    Trial 20 Exploitation: ResSeparableConv3DBlock enc1-enc3, plain SeparableConv3DBlock enc4.
    V4 pooling. ResUpConv3DBlock decoder. Combines separable dim=60 strength with residual
    gradient flow for dim=240; enc4 plain to allow free transformation post-z_pool3.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        self.enc1 = ResSeparableConv3DBlock(1, 8, downsample=False)               # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))       # 8×32×64×64
        self.enc2 = ResSeparableConv3DBlock(8, 16, downsample=True)               # 16×16×32×32
        self.enc3 = ResSeparableConv3DBlock(16, 32, downsample=True)              # 32×8×16×16
        self.z_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))    # 32×4×16×16
        self.enc4 = SeparableConv3DBlock(32, 64, downsample=True)                 # 64×2×8×8 — plain sep

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )                                                                           # 128×2×8×8

        self.final_down = nn.Conv3d(128, 128, 2, 2)                               # 128×1×4×4

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, 2, 2)                     # 128×2×8×8
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)  # 128×4×8×8

        self.dec1 = ResUpConv3DBlock(128, 64)                                     # 64×8×16×16
        self.dec2 = ResUpConv3DBlock(64, 32)                                      # 32×16×32×32
        self.dec3 = ResUpConv3DBlock(32, 16)                                      # 16×32×64×64

        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = ResConv3DBlock(16, 8, downsample=False)                  # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.z_pool3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ResUpSeparableConv3DBlock(nn.Module):
    """
    Residual 3D decoder block with depthwise separable convolutions.
    ConvTranspose3d upsampling, then residual DW+PW block with identity shortcut.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 2, 2)
        self.dw1 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels)
        self.pw1 = nn.Conv3d(out_channels, out_channels, 1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dw2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels)
        self.pw2 = nn.Conv3d(out_channels, out_channels, 1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        residual = x
        out = self.dw1(x)
        out = self.pw1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dw2(out)
        out = self.pw2(out)
        out = self.norm2(out)
        out = self.relu_out(out + residual)
        return out


class AutoEncoder3D_AsymResSeparableV2(nn.Module):
    """
    Trial 21: Full separable architecture. ResSeparableConv3DBlock enc1-enc3, plain SeparableConv3DBlock
    enc4, ResUpSeparableConv3DBlock decoder. V4 anisotropic pooling. Entirely separable throughout.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        self.enc1 = ResSeparableConv3DBlock(1, 8, downsample=False)               # 8×32×128×128
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))       # 8×32×64×64
        self.enc2 = ResSeparableConv3DBlock(8, 16, downsample=True)               # 16×16×32×32
        self.enc3 = ResSeparableConv3DBlock(16, 32, downsample=True)              # 32×8×16×16
        self.z_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))    # 32×4×16×16
        self.enc4 = SeparableConv3DBlock(32, 64, downsample=True)                 # 64×2×8×8

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )
        self.final_down = nn.Conv3d(128, 128, 2, 2)                               # 128×1×4×4

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, 2, 2)                     # 128×2×8×8
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)

        self.dec1 = ResUpSeparableConv3DBlock(128, 64)                            # 64×8×16×16
        self.dec2 = ResUpSeparableConv3DBlock(64, 32)                             # 32×16×32×32
        self.dec3 = ResUpSeparableConv3DBlock(32, 16)                             # 16×32×64×64

        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = ResSeparableConv3DBlock(16, 8, downsample=False)         # 8×32×128×128

        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.z_pool3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoder3D_AsymDilatedAttention(nn.Module):
    """
    Trial 8: Dilated attention encoder with V4 asymmetric pooling.
    Replaces the champion's ResSeparableConv3DBlock encoder blocks with
    DilatedAttentionConv3DBlock (dilated conv + SE attention) to capture
    multi-scale cardiac context. Dilation rates increase through stages
    (1, 2, 4) then reset to 1 at enc4. Decoder identical to champion.
    No skip connections.
    """
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0, reduction=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder: dilated attention blocks with increasing dilation
        self.enc1 = DilatedAttentionConv3DBlock(1, 8, dilation=1, downsample=False, reduction=reduction)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2 = DilatedAttentionConv3DBlock(8, 16, dilation=2, downsample=True, reduction=reduction)
        self.enc3 = DilatedAttentionConv3DBlock(16, 32, dilation=4, downsample=True, reduction=reduction)
        self.z_pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.enc4 = DilatedConv3DBlock(32, 64, dilation=1, downsample=True)

        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128), nn.ReLU(inplace=True),
        )

        self.final_down = nn.Conv3d(128, 128, 2, 2)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.initial_up = nn.ConvTranspose3d(128, 128, 2, 2)
        self.z_up = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)

        # Decoder: identical to champion (ResUpSeparableConv3DBlock)
        self.dec1 = ResUpSeparableConv3DBlock(128, 64)
        self.dec2 = ResUpSeparableConv3DBlock(64, 32)
        self.dec3 = ResUpSeparableConv3DBlock(32, 16)

        self.dec4_up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec4_conv = ResSeparableConv3DBlock(16, 8, downsample=False)

        self.final_conv = nn.Conv3d(8, 1, 3, 1, 1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.z_pool3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.z_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4_up(x)
        x = self.dec4_conv(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# Building

def build_autoencoder(model_name, latent_dimensions, dropout_rate=0.0):
    """
    Build one of the available AE models.
    """

    # BASE MODELS 
    
    if model_name == "AE3dCurrent": # 0.713189
        return AutoEncoder3D_Current(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dFCDeep": # 0.751709
        return AutoEncoder3D_FCDeep(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dConv": # 0.744460
        return AutoEncoder3D_Conv(latent_dim=latent_dimensions)

    elif model_name == "AE3dLinear": # O.590524
        return AutoEncoder3D_Linear(latent_dim=latent_dimensions)
    
    elif model_name == "AE3dFCDeep_VAE": # 0.741339
        return AutoEncoder3D_FCDeep_VAE(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    # AI Agent Pi

    elif model_name == "AE3dAttention": # 0.745631
        return AutoEncoder3D_Attention(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dDilated": # 0.748160
        return AutoEncoder3D_Dilated(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dDilatedAttention": # 0.733706
        return AutoEncoder3D_DilatedAttention(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dSeparableDilated": # 0.737950
        return AutoEncoder3D_SeparableDilated(latent_dim=latent_dimensions, dropout_rate=dropout_rate)
    
    # AI Agent Claude

    elif model_name == "AE3dFCDeepAsym": # 0.762770
        return AutoEncoder3D_FCDeepAsym(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dFCDeepAsymV2": # 0.766463
        return AutoEncoder3D_FCDeepAsymV2(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dFCDeepAsymV4": # 0.766648
        return AutoEncoder3D_FCDeepAsymV4(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dAsymResidual": # 0.762058
        return AutoEncoder3D_AsymResidual(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dAsymResidualV4": # 0.773806 -> TO TRAIN ALL DIMS?
        return AutoEncoder3D_AsymResidualV4(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dAsymSeparable": # 0.755189
        return AutoEncoder3D_AsymSeparable(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dAsymResSeparable": # 0.795874 -> TO TRAIN ALL DIMS?
        return AutoEncoder3D_AsymResSeparable(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dAsymResSeparableV2": # 0.811539 -> TO TRAIN ALL DIMS?
        return AutoEncoder3D_AsymResSeparableV2(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dAsymDilatedAttention":
        return AutoEncoder3D_AsymDilatedAttention(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    # ELSE

    else:
        raise ValueError(f"Unknown model_name: {model_name}")