#!/usr/bin/env python3
"""
Visualization Module for Super-Resolution Models
Shared visualization functions used across all model trainers.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


def create_side_by_side_comparison(lrms_upsampled, target, reconstructed, model_name, save_path):
    """
    Create a side-by-side comparison of bicubic upsampled input, target, and reconstructed images.
    
    Args:
        lrms_upsampled: Bicubic upsampled MS input (6, 96, 96) - what's fed to the model
        target: Target high resolution image (6, 96, 96) 
        reconstructed: Reconstructed high resolution image (6, 96, 96)
        model_name: Name of the model for the title
        save_path: Path to save the comparison image
    """
    # Convert to numpy if tensors
    if torch.is_tensor(lrms_upsampled):
        lrms_upsampled = lrms_upsampled.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.cpu().numpy()
    
    # Create RGB composite using bands 4, 3, 2 (B04, B03, B02) for visualization
    def create_rgb_composite(img):
        if len(img.shape) == 3 and img.shape[0] >= 4:
            # Multi-band image - use bands 4, 3, 2 (B04, B03, B02)
            rgb_bands = img[[3, 2, 1]]  # B04, B03, B02
            
            # Use 2nd and 98th percentiles for each band to avoid outliers
            rgb_normalized = np.zeros_like(rgb_bands)
            for i in range(3):
                band_data = rgb_bands[i]
                p2 = np.percentile(band_data, 2)
                p98 = np.percentile(band_data, 98)
                
                if p98 > p2:
                    rgb_normalized[i] = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                else:
                    rgb_normalized[i] = np.zeros_like(band_data)
            
            # Transpose to (H, W, 3) for matplotlib
            rgb = np.transpose(rgb_normalized, (1, 2, 0))
        else:
            # Single band image - convert to grayscale RGB
            single_band = img.squeeze()
            band_min = np.min(single_band)
            band_max = np.max(single_band)
            if band_max > band_min:
                normalized_band = (single_band - band_min) / (band_max - band_min)
            else:
                normalized_band = single_band
            rgb = np.stack([normalized_band, normalized_band, normalized_band], axis=-1)
        
        return rgb
    
    # Create RGB composites
    lrms_rgb = create_rgb_composite(lrms_upsampled)
    target_rgb = create_rgb_composite(target)
    reconstructed_rgb = create_rgb_composite(reconstructed)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot bicubic upsampled input (what's fed to the model)
    axes[0].imshow(lrms_rgb)
    axes[0].set_title(f'Bicubic Input\n{lrms_rgb.shape[0]}x{lrms_rgb.shape[1]}', fontsize=12)
    axes[0].axis('off')
    
    # Plot target image
    axes[1].imshow(target_rgb)
    axes[1].set_title(f'Target (Ground Truth)\n{target_rgb.shape[0]}x{target_rgb.shape[1]}', fontsize=12)
    axes[1].axis('off')
    
    # Plot reconstructed image
    axes[2].imshow(reconstructed_rgb)
    axes[2].set_title(f'{model_name} Reconstruction\n{reconstructed_rgb.shape[0]}x{reconstructed_rgb.shape[1]}', fontsize=12)
    axes[2].axis('off')
    
    # Add overall title
    fig.suptitle(f'{model_name} - Side by Side Comparison', fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Comparison image saved: {save_path}")


def get_random_sydney_sample(test_dataset, random_seed=None, max_attempts=10):
    """
    Get a random sample from the Sydney test dataset that is not blank/black.
    Returns both the upsampled data (for model input) and original downsampled data (for visualization).
    
    Args:
        test_dataset: The test dataset
        random_seed: Optional random seed for reproducibility
        max_attempts: Maximum number of attempts to find a non-blank image
    
    Returns:
        tuple: (lrms_upsampled, pan, hrms, lrms_original, lrms_upsampled_original) tensors for the random sample
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    for attempt in range(max_attempts):
        # Get a random index
        random_idx = random.randint(0, len(test_dataset) - 1)
        
        # Get the sample
        sample = test_dataset[random_idx]
        lrms_upsampled = sample['lrms']
        pan = sample['pan']
        hrms = sample['hrms']
        
        # Get the original downsampled data (48x48) for visualization
        loc, tile_idx = test_dataset.samples[random_idx]
        ms_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
        
        # Load original downsampled data without upsampling
        lrms_original = []
        for band in ms_bands:
            arr = np.load(os.path.join(test_dataset.data_root, loc, band, 'downsampled', f"{loc}_{band}_downsampled_{tile_idx}.npy"))
            lrms_original.append(arr)
        lrms_original = np.stack(lrms_original, axis=0)  # (bands, 48, 48)
        
        # For SRGAN, we also need the original upsampled data for visualization
        # since the dataset returns high-pass filtered data
        if hasattr(test_dataset, '__class__') and 'SRGAN' in test_dataset.__class__.__name__:
            # Load original upsampled data (before high-pass filtering) for visualization
            lrms_upsampled_original = []
            for band in ms_bands:
                arr = np.load(os.path.join(test_dataset.data_root, loc, band, 'downsampled', f"{loc}_{band}_downsampled_{tile_idx}.npy"))
                if arr.shape != (96, 96):
                    from scipy.ndimage import zoom
                    arr = zoom(arr, (96/arr.shape[0], 96/arr.shape[1]), order=3)
                lrms_upsampled_original.append(arr)
            lrms_upsampled_original = np.stack(lrms_upsampled_original, axis=0)  # (bands, 96, 96)
        else:
            lrms_upsampled_original = lrms_upsampled
        
        # Check if the image is not blank/black
        # Convert to numpy if tensor
        if torch.is_tensor(hrms):
            hrms_np = hrms.cpu().numpy()
        else:
            hrms_np = hrms
        
        # Check if the image has meaningful content (not all zeros or very low values)
        # Check the mean value and standard deviation
        mean_val = np.mean(hrms_np)
        std_val = np.std(hrms_np)
        
        # If mean is very low (< 0.01) or std is very low (< 0.01), it's likely blank
        if mean_val > 0.01 and std_val > 0.01:
            print(f"  ✅ Selected valid sample (attempt {attempt + 1}): mean={mean_val:.4f}, std={std_val:.4f}")
            return lrms_upsampled, pan, hrms, lrms_original, lrms_upsampled_original
        else:
            print(f"  ⚠️  Sample {random_idx} appears blank (mean={mean_val:.4f}, std={std_val:.4f}), trying again...")
    
    # If we couldn't find a good sample after max_attempts, just return the last one
    print(f"  ⚠️  Could not find non-blank sample after {max_attempts} attempts, using last sample")
    return lrms_upsampled, pan, hrms, lrms_original, lrms_upsampled_original


if __name__ == "__main__":
    """
    Test the visualization functions.
    """
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("  - create_side_by_side_comparison()")
    print("  - get_random_sydney_sample()")
