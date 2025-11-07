"""
VDSR Model Architecture

Usage:
    from vdsr import VDSR
    from utils.data_loader import SuperResolutionDataset
    from vdsr_trainer import run_vdsr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Module):
    """
    Conv3x3 + ReLU block (VDSR-style)
    Simple building block without residual connection
    """
    def __init__(self, channels):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class VDSR(nn.Module):
    """
    VDSR - Very Deep SR with residual learning
    
    Architecture:
        Conv1: 7 → 64, kernel=3, ReLU (input layer)
        18x ConvReLU: 64 → 64, kernel=3, ReLU (deep feature extraction)
        ConvLast: 64 → 6, kernel=3 (output layer, predicts residual)
        Skip: predicted_residual + MS↑ (residual learning)
    
    Input: lrms (6, 96, 96) + pan (1, 96, 96) = 7 channels
    Output: 6 channels (MS residual) + MS input = SR MS
    """
    
    def __init__(self, num_layers=20, num_filters=64, use_pan=True):  # Add use_pan parameter
        super(VDSR, self).__init__()
        self.use_pan = use_pan
        
        # Conditionally set input channels based on use_pan flag
        input_channels = 7 if use_pan else 6
        
        # Conv1: Input layer (7 or 6 → 64 channels) - adapted for multispectral
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Deep Conv3x3 + ReLU layers for feature extraction
        trunk = []
        for _ in range(num_layers - 2):  # Total: 1 input + (num_layers-2) middle + 1 output
            trunk.append(ConvReLU(num_filters))
        self.trunk = nn.Sequential(*trunk)
        
        # ConvLast: Output layer (64 → 6 channels, predicts residual)
        self.conv_last = nn.Conv2d(num_filters, 6, kernel_size=3, padding=1, bias=False)
        
        # Initialize weights like original VDSR
        self._initialize_weights()
        
        print(f"VDSR initialized!")
        print(f"  Architecture: {num_layers} layers (very deep)")
        print(f"  Input: {input_channels} channels ({'6 MS + 1 PAN' if use_pan else '6 MS only'})")
        print(f"  Output: 6 channels (MS residual)")
        print(f"  Filters: {num_filters}")
        print(f"  Residual learning: YES")
        print(f"  Skip connection: predicted_residual + MS↑")
    
    def _initialize_weights(self):
        """Initialize weights like original VDSR"""
        from math import sqrt
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))

    def forward(self, lrms_upsampled, pan=None):
        """
        Forward pass - VDSR with residual learning
        
        Args:
            lrms_upsampled: Bicubic upsampled MS (batch, 6, 96, 96)
            pan: PAN band (batch, 1, 96, 96) - optional if use_pan=False
            
        Returns:
            out: Super-resolved MS (batch, 6, 96, 96)
        """
        # Conditionally concatenate MS + PAN based on use_pan flag
        if self.use_pan:
            x = torch.cat([lrms_upsampled, pan], dim=1)  # (batch, 7, 96, 96)
        else:
            x = lrms_upsampled  # (batch, 6, 96, 96)
        
        # Save MS input for residual connection (PanNet-style skip)
        # Only use MS channels (first 6), not PAN
        identity = lrms_upsampled  # (batch, 6, 96, 96)
        
        # Deep feature extraction
        out = self.conv1(x)          # (batch, 64, 96, 96)
        out = self.trunk(out)        # (batch, 64, 96, 96) - 18 layers deep!
        out = self.conv_last(out)    # (batch, 6, 96, 96) - predicts residual
        
        # Residual learning: Output = predicted_residual + MS↑
        # This is the key difference from PanNet!
        out = torch.add(out, identity)  # (batch, 6, 96, 96)
        
        return out
