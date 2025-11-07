"""
Universal Data Loader for Super-Resolution Models

This dataset works for all super-resolution models:
- PanNet
- SRCNN  
- VDSR
- SRGAN

All models use the same data structure: MS bands + PAN band for input, HRMS for target.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


class SuperResolutionDataset(Dataset):
    """
    Universal Dataset for all Super-Resolution models.
    """
    def __init__(self, data_root, locations, ms_bands=['B02', 'B03', 'B04', 'B05', 'B06', 'B07'], pan_band='PAN_SIM', split='train', max_samples_per_location=500):
        self.data_root = data_root
        self.locations = locations
        self.ms_bands = ms_bands
        self.pan_band = pan_band
        self.split = split  # For future use (train/val/test split)
        self.max_samples_per_location = max_samples_per_location
        self.tile_indices = []
        self.samples = []  # List of (location, tile_index)
        self._gather_samples()

    def _gather_samples(self):
        # For each location, find all tile indices that exist for all bands and PAN
        for loc in self.locations:
            location_samples = []
            # Get all tile indices for the first MS band as reference
            ms_dir = os.path.join(self.data_root, loc, self.ms_bands[0], 'ground_truth')
            ms_files = [f for f in os.listdir(ms_dir) if f.endswith('.npy')]
            indices = set([f.split('_')[-1].replace('.npy', '') for f in ms_files])
            # Check that each index exists for all bands and PAN
            for idx in indices:
                ms_exists = all(os.path.exists(os.path.join(self.data_root, loc, band, 'ground_truth', f"{loc}_{band}_groundtruth_{idx}.npy")) for band in self.ms_bands)
                ms_ds_exists = all(os.path.exists(os.path.join(self.data_root, loc, band, 'downsampled', f"{loc}_{band}_downsampled_{idx}.npy")) for band in self.ms_bands)
                pan_exists = os.path.exists(os.path.join(self.data_root, loc, self.pan_band, 'ground_truth', f"{loc}_{self.pan_band}_groundtruth_{idx}.npy"))
                if ms_exists and ms_ds_exists and pan_exists:
                    location_samples.append((loc, idx))
            
            # Take samples up to the specified limit
            location_samples = location_samples[:self.max_samples_per_location]
            self.samples.extend(location_samples)
            print(f"Location {loc}: Selected {len(location_samples)} tiles")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        loc, tile_idx = self.samples[idx]
        # Load and upsample LRMS (downsampled MS bands) to (96, 96)
        lrms = []
        for band in self.ms_bands:
            arr = np.load(os.path.join(self.data_root, loc, band, 'downsampled', f"{loc}_{band}_downsampled_{tile_idx}.npy"))
            if arr.shape != (96, 96):
                arr = resize(arr, (96, 96), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
            lrms.append(arr)
        lrms = np.stack(lrms, axis=0)  # (bands, 96, 96)
        # Load HRMS (ground truth MS bands), upsample B05, B06, B07 to (96, 96)
        hrms_bands = []
        for band in self.ms_bands:
            arr = np.load(os.path.join(self.data_root, loc, band, 'ground_truth', f"{loc}_{band}_groundtruth_{tile_idx}.npy"))
            if band in ['B05', 'B06', 'B07'] and arr.shape != (96, 96):
                arr = resize(arr, (96, 96), order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
            hrms_bands.append(arr)
        hrms = np.stack(hrms_bands, axis=0)  # (bands, 96, 96)
        # Load PAN (simulated)
        pan = np.load(os.path.join(self.data_root, loc, self.pan_band, 'ground_truth', f"{loc}_{self.pan_band}_groundtruth_{tile_idx}.npy"))  # (H, W)
        
        return {
            'lrms': torch.from_numpy(lrms).float(),  # Original LRMS values
            'hrms': torch.from_numpy(hrms).float(),  # Original HRMS values
            'pan': torch.from_numpy(pan).unsqueeze(0).float(),  # Original PAN values
            'location': loc,
            'tile_idx': tile_idx
        }

