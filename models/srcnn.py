"""
SRCNN Model Architecture

Usage:
    from srcnn import SRCNN
    from utils.data_loader import SuperResolutionDataset
    from srcnn_trainer import run_srcnn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    SRCNN - Shallow CNN baseline built on PanNet foundation
    
    Architecture:
        Conv1: 7 → 64 channels, kernel=9, ReLU
        Conv2: 64 → 32 channels, kernel=5, ReLU  
        Conv3: 32 → 6 channels, kernel=5
        Skip: out + lrms (like PanNet)
    
    Input: lrms (6, 96, 96) + pan (1, 96, 96) = 7 channels
    Output: 6 channels (MS only)
    """
    
    def __init__(self, num_filters=64, use_pan=True):  # Add use_pan parameter
        super(SRCNN, self).__init__()
        self.use_pan = use_pan
        
        # Conditionally set input channels based on use_pan flag
        input_channels = 7 if use_pan else 6
        
        # Conv1: Extract features (7 or 6 → 64 channels, large kernel for receptive field)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Conv2: Non-linear mapping (64 → 32 channels)
        self.conv2 = nn.Conv2d(num_filters, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Conv3: Reconstruction (32 → 6 channels for MS output)
        self.conv3 = nn.Conv2d(32, 6, kernel_size=5, padding=2)
        
        print(f"SRCNN initialized!")
        print(f"  Architecture: 3 conv layers (shallow)")
        print(f"  Input: {input_channels} channels ({'6 MS + 1 PAN' if use_pan else '6 MS only'})")
        print(f"  Output: 6 channels (MS)")
        print(f"  Filters: {num_filters}")
        print(f"  Skip connection: YES (like PanNet)")

    def forward(self, lrms_upsampled, pan=None):
        """
        Forward pass - SRCNN with PanNet-style skip connection
        
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
        
        # Feature extraction
        out = self.conv1(x)      # (batch, 64, 96, 96)
        out = self.relu1(out)
        
        # Non-linear mapping
        out = self.conv2(out)    # (batch, 32, 96, 96)
        out = self.relu2(out)
        
        # Reconstruction
        out = self.conv3(out)    # (batch, 6, 96, 96)
        
        # Skip connection from bicubic input (like PanNet)
        out = out + lrms_upsampled  # (batch, 6, 96, 96)
        
        return out
