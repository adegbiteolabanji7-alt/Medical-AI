"""
unet_model.py
-------------
U-Net architecture for 3D medical image segmentation.

U-Net was published by Ronneberger et al. in 2015 specifically
for biomedical image segmentation. It solves a problem standard
CNNs cannot: producing a full-resolution output mask where every
voxel is labelled, not just a single classification.

Architecture:
- Encoder: compresses the image, learning what features exist
- Bottleneck: most compressed representation
- Decoder: expands back to original size using learned features
- Skip connections: preserve spatial detail across the U

Why this matters clinically:
A radiologist does not just say there is a spleen in this scan.
They outline exactly where it is. U-Net produces that outline
automatically, for every voxel, in 3D.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    The building block of U-Net.

    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU

    First conv learns simple features at one scale.
    Second conv learns combinations of those features.
    BatchNorm keeps values in a stable range.
    ReLU removes negatives and makes depth meaningful.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    """
    The down path of the U-Net.

    At each level: DoubleConv then MaxPool3d.
    Spatial size halves, channels double.
    Each level output saved for skip connections before pooling.
    """

    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        skip_connections = []

        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        return x, skip_connections


class Decoder(nn.Module):
    """
    The up path of the U-Net.

    At each level: upsample -> concatenate skip connection -> DoubleConv.

    Skip connection restores spatial detail lost during pooling.
    ConvTranspose3d is learnable upsampling for sharper boundaries.
    """

    def __init__(self, features=[512, 256, 128, 64]):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for feature in features:
            self.upsample.append(
                nn.ConvTranspose3d(
                    feature * 2, feature,
                    kernel_size=2, stride=2
                )
            )
            self.decoder_blocks.append(
                DoubleConv(feature * 2, feature)
            )

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]

        for i in range(len(self.decoder_blocks)):
            x = self.upsample[i](x)
            skip = skip_connections[i]
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[i](x)

        return x


class UNet3D(nn.Module):
    """
    Full 3D U-Net for medical image segmentation.

    Input:  (batch, 1, D, H, W) greyscale CT volume
    Output: (batch, 1, D, H, W) probability per voxel
    """

    def __init__(self, in_channels=1, out_channels=1,
                 features=[64, 128, 256, 512]):
        super().__init__()

        self.encoder = Encoder(in_channels, features)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.decoder = Decoder(features[::-1])

        self.final_conv = nn.Conv3d(features[0], out_channels,
                                    kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return self.sigmoid(self.final_conv(x))


if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=1)

    test_input = torch.zeros(1, 1, 64, 64, 32)
    print(f"Input shape:  {test_input.shape}")

    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output min:   {output.min():.4f}")
    print(f"Output max:   {output.max():.4f}")
    print("\nValues between 0 and 1 — voxel probabilities.")
    print("Close to 1 = confident this voxel is spleen.")
    print("Close to 0 = confident this voxel is background.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters: {total_params:,}")