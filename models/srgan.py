"""
SRGAN Model Architecture

Usage:
    from srgan import Generator, Discriminator
    from utils.data_loader import SuperResolutionDataset
    from srgan_trainer import run_srgan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    SRGAN Generator - Simple concatenation + basic conv blocks
    
    Architecture:
    - Simple concatenation: MS(6) + PAN(1) → 7 channels
    - Basic conv blocks: 7 → 64 → 64 → 64
    - Simple output: 64 → 6 channels
    - No complex ResBlocks or feature extractors
    """
    
    def __init__(self, ms_channels=6, pan_channels=1, num_filters=64, use_pan=True):  # Add use_pan parameter
        super(Generator, self).__init__()
        
        self.ms_channels = ms_channels
        self.pan_channels = pan_channels
        self.num_filters = num_filters
        self.use_pan = use_pan
        
        # Conditionally set input channels based on use_pan flag
        input_channels = ms_channels + pan_channels if use_pan else ms_channels
        
        # Simple input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=9, padding=4, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction blocks (simple, no residual)
        self.feature_blocks = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Output layer (no activation!)
        self.output_conv = nn.Conv2d(num_filters, ms_channels, kernel_size=9, padding=4, bias=False)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"SRGAN Generator initialized!")
        print(f"Architecture: Simple concatenation + basic conv blocks")
        print(f"Input: {input_channels} channels ({'MS({}) + PAN({})'.format(ms_channels, pan_channels) if use_pan else 'MS({}) only'.format(ms_channels)}) → Output: {ms_channels}")
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, ms_input, pan_input=None):
        """
        Simple forward pass - conditionally concatenate and process
        
        Args:
            ms_input (torch.Tensor): Multispectral input (batch, 6, 96, 96)
            pan_input (torch.Tensor): Panchromatic input (batch, 1, 96, 96) - optional if use_pan=False
            
        Returns:
            torch.Tensor: Super-resolved MS image (batch, 6, 96, 96)
        """
        # Conditionally concatenate based on use_pan flag
        if self.use_pan:
            combined_input = torch.cat([ms_input, pan_input], dim=1)  # (batch, 7, 96, 96)
        else:
            combined_input = ms_input  # (batch, 6, 96, 96)
        
        # Process through conv blocks
        features = self.input_conv(combined_input)      # (batch, 64, 96, 96)
        features = self.feature_blocks(features)        # (batch, 64, 96, 96)
        
        # Generate output (no activation!)
        output = self.output_conv(features)             # (batch, 6, 96, 96)
        
        # Clamp output to [0, 1] range for stability
        output = torch.clamp(output, min=0.0, max=1.0)
        
        return output
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Discriminator(nn.Module):
    """
    SRGAN Discriminator - Progressive downsampling
    Based on SRGAN paper but streamlined to prevent mode collapse
    """
    
    def __init__(self, input_channels=6):
        super(Discriminator, self).__init__()
        
        self.input_channels = input_channels
        
        # Feature extraction with progressive downsampling
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 6
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 7
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),             # Flatten to (batch, 512)
            nn.Linear(512, 1024),     # 512 → 1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),       # 1024 → 1
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"SRGAN Discriminator initialized!")
        print(f"Architecture: {input_channels} → 64 → 128 → 256 → 512 → 1024 → 1")
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through Discriminator
        
        Args:
            x (torch.Tensor): Input image (batch, channels, 96, 96)
            
        Returns:
            torch.Tensor: Probability of being real (batch, 1)
        """
        # Extract features
        features = self.features(x)
        
        # Use classifier (includes AdaptiveAvgPool2d and flattening)
        output = self.classifier(features)
        
        return output
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
