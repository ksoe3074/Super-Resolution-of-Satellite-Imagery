"""
PanNet Model Architecture

Usage:
    from pannet import PanNet
    from utils.data_loader import SuperResolutionDataset
    from pannet_trainer import run_pannet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class PanNet(nn.Module):
    """
    PanNet - Residual pan-sharpening network
    
    Architecture:
        InputConv: (6 or 7) → num_filters, kernel=3, padding=1
        ResidualBlocks: 7 × (num_filters → num_filters), kernel=3, padding=1
        OutputConv: num_filters → 6, kernel=3, padding=1
        Skip: reconstructed + lrms_upsampled (PanNet-style skip connection)
    
    Input: lrms_upsampled (6, H, W) [+ pan (1, H, W) when use_pan=True]
    Output: Super-resolved MS image (6, H, W)
    """
    def __init__(self, num_filters=64, use_pan=True):  # Add use_pan parameter
        super(PanNet, self).__init__()
        self.use_pan = use_pan
        
        # Conditionally set input channels based on use_pan flag
        input_channels = 7 if use_pan else 6
        self.input_conv = nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(7)])
        self.output_conv = nn.Conv2d(num_filters, 6, kernel_size=3, padding=1)

    def forward(self, lrms_upsampled, pan=None):
        # Conditionally concatenate PAN based on use_pan flag
        if self.use_pan:
            # lrms_upsampled: (batch, 6, H, W), pan: (batch, 1, H, W)
            x = torch.cat([lrms_upsampled, pan], dim=1)  # (batch, 7, H, W)
        else:
            # lrms_upsampled: (batch, 6, H, W) - no PAN concatenation
            x = lrms_upsampled  # (batch, 6, H, W)
            
        out = self.input_conv(x)
        out = self.relu(out)
        out = self.res_blocks(out)
        out = self.output_conv(out)
        # PanNet: add skip connection from upsampled LRMS
        out = out + lrms_upsampled
        return out  # (batch, 6, H, W) 