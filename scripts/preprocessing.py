import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import glob
from scipy.ndimage import gaussian_filter


def get_band_paths(base_dir, location):
    """Construct paths for the bands of interest by searching for the correct filenames."""
    granule_dir = os.path.join(base_dir, location, 'GRANULE')
    # Filter out .DS_Store and assume the first subfolder is the tile granule folder
    tile_granule = [d for d in os.listdir(granule_dir) if not d.startswith('.')][0]
    img_data_dir = os.path.join(granule_dir, tile_granule, 'IMG_DATA')
    
    r10m_dir = os.path.join(img_data_dir, 'R10m')
    r20m_dir = os.path.join(img_data_dir, 'R20m')
    
    # Helper to find the first file matching a pattern
    def find_band_file(directory, pattern):
        matches = glob.glob(os.path.join(directory, pattern))
        if matches:
            return matches[0]
        else:
            return None
    
    bands = {
        'B02': find_band_file(r10m_dir, '*_B02_10m.jp2'),
        'B03': find_band_file(r10m_dir, '*_B03_10m.jp2'),
        'B04': find_band_file(r10m_dir, '*_B04_10m.jp2'),
        'B05': find_band_file(r20m_dir, '*_B05_20m.jp2'),
        'B06': find_band_file(r20m_dir, '*_B06_20m.jp2'),
        'B07': find_band_file(r20m_dir, '*_B07_20m.jp2')
    }
    return bands


def read_band(band_path):
    """Read a band file."""
    with rasterio.open(band_path) as src:
        return src.read(1)


def tile_array(arr, tile_size):
    """Split the array into tiles of size tile_size x tile_size, skipping partial tiles."""
    h, w = arr.shape
    tiles = []
    for i in range(0, h - tile_size + 1, tile_size):
        for j in range(0, w - tile_size + 1, tile_size):
            # Extract a tile
            tile = arr[i:i + tile_size, j:j + tile_size]
            tiles.append(tile)
    return tiles


def downsample_tile(tile, target_size=32):
    """Downsample a tile to the target size (32x32) using Gaussian blur."""
    # Apply Gaussian blur
    blurred_tile = gaussian_filter(tile, sigma=1.0)
    h, w = blurred_tile.shape
    # Calculate the resampling factor
    resampling_factor = h / target_size
    # Resample the tile
    resampled = np.zeros((target_size, target_size), dtype=tile.dtype)
    for i in range(target_size):
        for j in range(target_size):
            # Simple nearest neighbor downsampling
            resampled[i, j] = blurred_tile[int(i * resampling_factor), int(j * resampling_factor)]
    return resampled


def main():
    base_dir = 'raw_data/SENTINEL-2'
    locations = ['SYDNEY']  # Only process the new SYDNEY scene
    
    location_tile_counts = {loc: 0 for loc in locations}
    for location in locations:
        print(f"Processing {location}...")
        granule_dir = os.path.join(base_dir, location, 'GRANULE')
        if not os.path.isdir(granule_dir):
            print(f"Warning: {granule_dir} does not exist. Skipping {location}.")
            continue
        # Get band paths
        band_paths = get_band_paths(base_dir, location)
        
        # Define the output directory structure
        output_base = 'processed_data'
        location_dir = os.path.join(output_base, location)
        
        # Create the directory structure
        os.makedirs(location_dir, exist_ok=True)
        
        # Process each band
        for band_name, band_path in band_paths.items():
            if band_path is None:
                print(f"Warning: {band_name} not found for {location}. Skipping this band.")
                continue
            print(f"Processing {band_name}...")
            # Read the band
            band_data = read_band(band_path)
            # Normalize the band data to [0, 1] and convert to float64
            band_data_normalized = band_data.astype(np.float64) / 65535.0  # Normalize to [0, 1]
            # Determine tile size based on band resolution
            tile_size = 96 if band_name in ['B02', 'B03', 'B04'] else 48
            # Tile the band
            tiles = tile_array(band_data_normalized, tile_size=tile_size)
            # Create directories for ground truth and downsampled tiles
            band_dir = os.path.join(location_dir, band_name)
            gt_dir = os.path.join(band_dir, 'ground_truth')
            ds_dir = os.path.join(band_dir, 'downsampled')
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(ds_dir, exist_ok=True)
            # Save the original (ground truth) tiles
            for idx, tile in enumerate(tiles):
                if np.all(tile == 0):
                    continue  # Skip fully black tiles
                gt_filename = f'{location}_{band_name}_groundtruth_{idx}.npy'
                output_path = os.path.join(gt_dir, gt_filename)
                np.save(output_path, tile)
                location_tile_counts[location] += 1
            # Downsample each tile to 32x32 and save
            for idx, tile in enumerate(tiles):
                if np.all(tile == 0):
                    continue  # Skip fully black tiles
                downsampled_tile = downsample_tile(tile, target_size=32)
                ds_filename = f'{location}_{band_name}_downsampled_{idx}.npy'
                output_path = os.path.join(ds_dir, ds_filename)
                np.save(output_path, downsampled_tile)
            print(f"Saved {len(tiles)} ground truth and downsampled tiles for {band_name}.")

        # --- Simulate PAN band (PAN_SIM) ---
        # Only if all three bands are available
        if all(b in band_paths and band_paths[b] is not None for b in ['B02', 'B03', 'B04']):
            print(f"Simulating PAN band for {location}...")
            # Read and normalize the three bands
            b2 = read_band(band_paths['B02']).astype(np.float64) / 65535.0
            b3 = read_band(band_paths['B03']).astype(np.float64) / 65535.0
            b4 = read_band(band_paths['B04']).astype(np.float64) / 65535.0
            # Tile all three bands
            tiles_b2 = tile_array(b2, tile_size=96)
            tiles_b3 = tile_array(b3, tile_size=96)
            tiles_b4 = tile_array(b4, tile_size=96)
            # Create PAN_SIM directories
            pan_dir = os.path.join(location_dir, 'PAN_SIM')
            pan_gt_dir = os.path.join(pan_dir, 'ground_truth')
            pan_ds_dir = os.path.join(pan_dir, 'downsampled')
            os.makedirs(pan_gt_dir, exist_ok=True)
            os.makedirs(pan_ds_dir, exist_ok=True)
            # Weightings from simulate_pan.py
            w2, w3, w4 = 0.251, 0.404, 0.345
            for idx, (t2, t3, t4) in enumerate(zip(tiles_b2, tiles_b3, tiles_b4)):
                pan_tile = w2 * t2 + w3 * t3 + w4 * t4
                if np.all(pan_tile == 0):
                    continue  # Skip fully black PAN tiles
                # Save ground truth PAN tile (96x96)
                pan_gt_filename = f'{location}_PAN_SIM_groundtruth_{idx}.npy'
                np.save(os.path.join(pan_gt_dir, pan_gt_filename), pan_tile)
                # Downsample to 64x64 for 15m equivalent
                pan_tile_ds = downsample_tile(pan_tile, target_size=64)
                pan_ds_filename = f'{location}_PAN_SIM_downsampled_{idx}.npy'
                np.save(os.path.join(pan_ds_dir, pan_ds_filename), pan_tile_ds)

    print("\nTotal number of saved tiles for each location:")
    for loc, count in location_tile_counts.items():
        print(f"{loc}: {count}")


if __name__ == '__main__':
    main() 