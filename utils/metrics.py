import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize


def calculate_psnr(gt, ds):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between ground truth and downsampled images."""
    # Ensure inputs are float32
    gt = gt.astype(np.float32)
    ds = ds.astype(np.float32)
    
    # Add small epsilon to prevent division by zero
    mse = np.mean((gt - ds) ** 2)
    if mse < 1e-10:  # Use a small threshold instead of exact zero
        return float('inf')  # Return infinity for perfect match
    max_pixel = max(gt.max(), ds.max())
    return 10 * np.log10((max_pixel ** 2) / mse)


def calculate_ssim(gt, ds):
    """Calculate the Structural Similarity Index (SSIM) between ground truth and downsampled images."""
    # Ensure inputs are float32
    gt = gt.astype(np.float32)
    ds = ds.astype(np.float32)
    
    data_range = max(gt.max() - gt.min(), ds.max() - ds.min())
    if data_range < 1e-10:  # Use a small threshold instead of exact zero
        return 0.0  # Return 0.0 for constant images
    return ssim(gt, ds, data_range=data_range)


def calculate_sam(gt, ds):
    """Calculate the Spectral Angle Mapper (SAM) between ground truth and downsampled images."""
    # Ensure inputs are float32
    gt = gt.astype(np.float32)
    ds = ds.astype(np.float32)
    
    # For single-band images, calculate angle between vectors
    gt_flat = gt.flatten()
    ds_flat = ds.flatten()
    
    # Calculate the dot product and the magnitudes
    dot_product = np.sum(gt_flat * ds_flat)
    gt_magnitude = np.sqrt(np.sum(gt_flat ** 2))
    ds_magnitude = np.sqrt(np.sum(ds_flat ** 2))
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    gt_magnitude = max(gt_magnitude, epsilon)
    ds_magnitude = max(ds_magnitude, epsilon)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (gt_magnitude * ds_magnitude)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to avoid numerical errors
    
    # Calculate the angle in radians and convert to degrees
    angle = np.arccos(cos_angle)
    return angle * 180 / np.pi


def calculate_sam_multiband(gt, ds):
    """Calculate the Spectral Angle Mapper (SAM) for multi-band images.
    
    Args:
        gt: Ground truth image with shape (bands, height, width)
        ds: Downsampled/predicted image with shape (bands, height, width)
    
    Returns:
        Average SAM value across all pixels
    """
    # Ensure inputs are float32
    gt = gt.astype(np.float32)
    ds = ds.astype(np.float32)
    
    # Reshape to (pixels, bands)
    gt_reshaped = gt.reshape(gt.shape[0], -1).T  # (pixels, bands)
    ds_reshaped = ds.reshape(ds.shape[0], -1).T  # (pixels, bands)
    
    # Calculate dot product for each pixel
    dot_product = np.sum(gt_reshaped * ds_reshaped, axis=1)
    
    # Calculate magnitudes for each pixel
    gt_magnitude = np.sqrt(np.sum(gt_reshaped ** 2, axis=1))
    ds_magnitude = np.sqrt(np.sum(ds_reshaped ** 2, axis=1))
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    gt_magnitude = np.maximum(gt_magnitude, epsilon)
    ds_magnitude = np.maximum(ds_magnitude, epsilon)
    
    # Calculate cosine of angle for each pixel
    cos_angle = dot_product / (gt_magnitude * ds_magnitude)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to avoid numerical errors
    
    # Calculate angle in radians and convert to degrees
    angle = np.arccos(cos_angle)
    sam_degrees = angle * 180 / np.pi
    
    # Return average SAM across all pixels
    return np.mean(sam_degrees)


def calculate_ergas(gt, ds, scale_factor=3):
    """Calculate the ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse) between ground truth and downsampled images."""
    # Ensure inputs are float32
    gt = gt.astype(np.float32)
    ds = ds.astype(np.float32)
    
    # Calculate the mean squared error
    mse = np.mean((gt - ds) ** 2)
    # Calculate the mean of the ground truth
    mean_gt = np.mean(gt)
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    mean_gt = max(mean_gt, epsilon)
    
    # Calculate ERGAS
    ergas = 100 * np.sqrt(mse / mean_gt ** 2) / scale_factor
    return ergas


# Example
if __name__ == '__main__':
    # Load a ground truth and downsampled tile
    gt_tile = np.load('processed_data/AGRICULTURAL/KANSAS/B02/ground_truth/AGRICULTURAL_KANSAS_B02_groundtruth_0.npy')
    ds_tile = np.load('processed_data/AGRICULTURAL/KANSAS/B02/downsampled/AGRICULTURAL_KANSAS_B02_downsampled_0.npy')
    
    # Interpolate downsampled tile to match ground truth dimensions
    ds_tile_resized = resize(ds_tile, gt_tile.shape, order=1, anti_aliasing=True)
    
    print('GT shape:', gt_tile.shape, 'DS shape:', ds_tile_resized.shape)
    print('GT min/max:', gt_tile.min(), gt_tile.max(), 'DS min/max:', ds_tile_resized.min(), ds_tile_resized.max())
    print('GT dtype:', gt_tile.dtype, 'DS dtype:', ds_tile_resized.dtype)
    
    # Calculate metrics
    psnr_value = calculate_psnr(gt_tile, ds_tile_resized)
    ssim_value = calculate_ssim(gt_tile, ds_tile_resized)
    # sam_value = calculate_sam(gt_tile, ds_tile_resized)  # Commented out as it requires all bands
    ergas_value = calculate_ergas(gt_tile, ds_tile_resized)
    
    print(f'PSNR: {psnr_value:.2f}')
    print(f'SSIM: {ssim_value:.2f}')
    # print(f'SAM: {sam_value:.2f}')  # Commented out as it requires all bands
    print(f'ERGAS: {ergas_value:.2f}') 