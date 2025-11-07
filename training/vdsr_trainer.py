#!/usr/bin/env python3
"""
VDSR Trainer Module
Modular training implementation for VDSR super-resolution model.
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vdsr import VDSR
from utils.data_loader import SuperResolutionDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam_multiband, calculate_ergas


class VDSRTrainer:
    """
    VDSR Trainer class for super-resolution training and evaluation.
    
    This class encapsulates the complete training pipeline for VDSR including:
    - Model initialization and configuration
    - Training loop with loss tracking and gradient clipping
    - Evaluation with comprehensive metrics
    - Visualization generation
    """
    
    def __init__(self, device, config):
        """
        Initialize VDSR trainer.
        
        Args:
            device: Training device (CPU/GPU)
            config (dict): Training configuration containing:
                - epochs: Number of training epochs
                - lr: Learning rate
                - filters: Number of filters in the model
                - batch_size: Batch size for training
                - layers: Number of layers in VDSR
        """
        self.device = device
        self.config = config
        
        # Initialize model components
        self.model = None
        self.criterion = None
        self.optimizer = None
        
        print(f"VDSR Trainer initialized!")
        print(f"Configuration: {config['epochs']} epochs, LR={config['lr']}, filters={config['filters']}, layers={config.get('layers', 20)}")
        use_pan = config.get('use_pan', True)
        print(f"PAN usage: {'Enabled' if use_pan else 'Disabled'}")
    
    def create_model(self):
        """Create and initialize the VDSR model."""
        num_layers = self.config.get('layers', 20)
        use_pan = self.config.get('use_pan', True)  # Default to True for backward compatibility
        self.model = VDSR(num_layers=num_layers, num_filters=self.config['filters'], use_pan=use_pan).to(self.device)
        
        # Use MSE with reduction="sum" like original VDSR
        self.criterion = torch.nn.MSELoss(reduction="sum")
        
        # Use AdamW like PanNet and SRCNN (more stable for deep networks)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=1e-4
        )
        
        print(f"  ✅ Model created with {self.config['filters']} filters, {num_layers} layers")
        print(f"  ✅ Optimizer: AdamW (lr={self.config['lr']}, weight_decay=1e-4)")
        print(f"  ✅ Loss function: MSE Loss (reduction='sum')")
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """
        Train for one epoch with VDSR-specific features.
        
        Args:
            train_loader: Training data loader
            epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            lrms = batch['lrms'].to(self.device)
            pan = batch['pan'].to(self.device)
            hrms = batch['hrms'].to(self.device)
            
            # VDSR-specific: Proper normalization to [0,1] range
            batch_min = min(lrms.min(), pan.min(), hrms.min())
            batch_max = max(lrms.max(), pan.max(), hrms.max())
            
            # Normalize to [0,1] range
            lrms_norm = (lrms - batch_min) / (batch_max - batch_min + 1e-8)
            pan_norm = (pan - batch_min) / (batch_max - batch_min + 1e-8)
            hrms_norm = (hrms - batch_min) / (batch_max - batch_min + 1e-8)
            
            self.optimizer.zero_grad()
            output = self.model(lrms_norm, pan_norm)
            loss = self.criterion(output, hrms_norm)
            loss.backward()
            
            # VDSR-specific: Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=0.01 / self.config['lr'],  # Adjustable clipping
                norm_type=2.0
            )
            
            self.optimizer.step()
            running_loss += loss.item()
            
            # DEBUG: Check if residual is learning (can be removed later)
            if batch_idx % 50 == 0:  # Print every 50 batches
                with torch.no_grad():
                    # Get the residual that was learned (using normalized data)
                    # Use the model's forward method to handle concatenation properly
                    if self.model.use_pan:
                        output_norm = self.model(lrms_norm, pan_norm)
                    else:
                        output_norm = self.model(lrms_norm, None)
                    
                    # Calculate residual magnitude
                    residual = output_norm - lrms_norm
                    residual_mag = torch.mean(torch.abs(residual)).item()
                    lrms_mean = torch.mean(torch.abs(lrms_norm)).item()
                    hrms_mean = torch.mean(torch.abs(hrms_norm)).item()
                    output_mean = torch.mean(torch.abs(output_norm)).item()
                    
                    print(f"  Batch {batch_idx}: Residual magnitude = {residual_mag:.6f}")
                    print(f"    LRMS mean: {lrms_mean:.6f}, HRMS mean: {hrms_mean:.6f}, Output mean: {output_mean:.6f}")
                    print(f"    Loss: {loss.item():.8f}")
                    
                    if residual_mag < 0.01:  # Adjusted threshold for normalized data
                        print(f"    ⚠️  WARNING: Residual is small! May need more training")
        
        avg_loss = running_loss / len(train_loader)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{total_epochs}, Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def evaluate(self, test_loader):
        """
        Evaluate the trained model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            dict: Evaluation metrics (psnr, ssim, ergas, sam)
        """
        self.model.eval()
        psnr_list, ssim_list, ergas_list, sam_list = [], [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                lrms = batch['lrms'].to(self.device)
                pan = batch['pan'].to(self.device)
                hrms = batch['hrms'].to(self.device)
                
                # Apply same normalization as training
                batch_min = min(lrms.min(), pan.min(), hrms.min())
                batch_max = max(lrms.max(), pan.max(), hrms.max())
                
                lrms_norm = (lrms - batch_min) / (batch_max - batch_min + 1e-8)
                pan_norm = (pan - batch_min) / (batch_max - batch_min + 1e-8)
                hrms_norm = (hrms - batch_min) / (batch_max - batch_min + 1e-8)
                
                output = self.model(lrms_norm, pan_norm)
                
                # Convert to numpy for metrics
                pred_np = output.cpu().numpy()[0]
                target_np = hrms_norm.cpu().numpy()[0]
                
                # Compute per-band metrics
                psnr_bands = [calculate_psnr(target_np[b], pred_np[b]) for b in range(6)]
                ssim_bands = [calculate_ssim(target_np[b], pred_np[b]) for b in range(6)]
                ergas_bands = [calculate_ergas(target_np[b], pred_np[b]) for b in range(6)]
                
                # Compute multi-band SAM
                sam_value = calculate_sam_multiband(target_np, pred_np)
                
                psnr_list.append(np.mean(psnr_bands))
                ssim_list.append(np.mean(ssim_bands))
                ergas_list.append(np.mean(ergas_bands))
                sam_list.append(sam_value)
        
        # Calculate average metrics
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_ergas = np.mean(ergas_list)
        avg_sam = np.mean(sam_list)
        
        metrics = {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'ergas': avg_ergas,
            'sam': avg_sam
        }
        
        print(f"  Test Results - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, ERGAS: {avg_ergas:.4f}, SAM: {avg_sam:.4f}")
        
        return metrics
    
    def train(self, data_root, train_locations, test_locations, samples_per_location):
        """
        Complete training pipeline for VDSR.
        
        Args:
            data_root (str): Root directory containing the data
            train_locations (list): List of training location names
            test_locations (list): List of test location names
            samples_per_location (int): Maximum samples per location
            
        Returns:
            dict: Final evaluation metrics
        """
        print(f"Training VDSR: {self.config['epochs']} epochs, LR={self.config['lr']}, filters={self.config['filters']}")
        
        # Create datasets
        train_dataset = SuperResolutionDataset(data_root, train_locations, max_samples_per_location=samples_per_location)
        test_dataset = SuperResolutionDataset(data_root, test_locations, max_samples_per_location=samples_per_location)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"  📊 Training samples: {len(train_dataset)}")
        print(f"  📊 Test samples: {len(test_dataset)}")
        
        # Create model
        self.create_model()
        
        # Training loop
        print("  🚀 Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            self.train_epoch(train_loader, epoch, self.config['epochs'])
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.config['epochs']} - Time: {epoch_time:.1f}s")
        
        training_time = time.time() - start_time
        print(f"  ✅ Training completed! Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        
        # Evaluation
        print("  📊 Evaluating model...")
        metrics = self.evaluate(test_loader)
        
        return metrics
    
    def generate_comparison_image(self, test_dataset, visualization_func, random_seed=42):
        """
        Generate a side-by-side comparison image for visualization.
        
        Args:
            test_dataset: Test dataset to sample from
            visualization_func: Function to create the comparison image
            random_seed (int): Random seed for reproducible sampling
            
        Returns:
            str: Path to the saved comparison image, or None if failed
        """
        print("  🖼️  Generating side-by-side comparison...")
        try:
            # Get a random sample from the test dataset
            lrms_upsampled, pan_sample, hrms_sample, lrms_original, lrms_upsampled_original = self._get_random_sample(test_dataset, random_seed)
            
            # Add batch dimension and move to device
            lrms_batch = lrms_upsampled.unsqueeze(0).to(self.device)
            pan_batch = pan_sample.unsqueeze(0).to(self.device)
            
            # Apply normalization
            batch_min = min(lrms_batch.min(), pan_batch.min(), hrms_sample.min())
            batch_max = max(lrms_batch.max(), pan_batch.max(), hrms_sample.max())
            
            lrms_norm = (lrms_batch - batch_min) / (batch_max - batch_min + 1e-8)
            pan_norm = (pan_batch - batch_min) / (batch_max - batch_min + 1e-8)
            
            # Get model prediction
            self.model.eval()
            with torch.no_grad():
                reconstructed_sample = self.model(lrms_norm, pan_norm).squeeze(0)
            
            # Create output directory if it doesn't exist
            os.makedirs("comparison_images", exist_ok=True)
            
            # Save comparison image
            save_path = f"comparison_images/VDSR_comparison_{int(time.time())}.png"
            visualization_func(
                lrms_original, 
                hrms_sample, 
                reconstructed_sample, 
                "VDSR", 
                save_path
            )
            
            return save_path
            
        except Exception as e:
            print(f"  ⚠️  Could not generate comparison image: {e}")
            return None
    
    def _get_random_sample(self, test_dataset, random_seed=None, max_attempts=10):
        """
        Get a random sample from the test dataset that is not blank/black.
        
        Args:
            test_dataset: The test dataset
            random_seed: Optional random seed for reproducibility
            max_attempts: Maximum number of attempts to find a non-blank image
        
        Returns:
            tuple: (lrms_upsampled, pan, hrms, lrms_original, lrms_upsampled_original) tensors
        """
        import random
        
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
            lrms_upsampled_original = lrms_upsampled
            
            # Check if the image is not blank/black
            if torch.is_tensor(hrms):
                hrms_np = hrms.cpu().numpy()
            else:
                hrms_np = hrms
            
            # Check if the image has meaningful content
            mean_val = np.mean(hrms_np)
            std_val = np.std(hrms_np)
            
            if mean_val > 0.01 and std_val > 0.01:
                print(f"  ✅ Selected valid sample (attempt {attempt + 1}): mean={mean_val:.4f}, std={std_val:.4f}")
                return lrms_upsampled, pan, hrms, lrms_original, lrms_upsampled_original
            else:
                print(f"  ⚠️  Sample {random_idx} appears blank (mean={mean_val:.4f}, std={std_val:.4f}), trying again...")
        
        # If we couldn't find a good sample after max_attempts, just return the last one
        print(f"  ⚠️  Could not find non-blank sample after {max_attempts} attempts, using last sample")
        return lrms_upsampled, pan, hrms, lrms_original, lrms_upsampled_original
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path where to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state dict and config
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_class': 'VDSR'
        }, model_path)
        
        print(f"  💾 Model saved to: {model_path}")
        return model_path


def run_vdsr(data_root, train_locations, test_locations, samples_per_location, device, config):
    """
    Convenience function to run VDSR training using the modular trainer.
    
    This function maintains compatibility with the original run_all_models_imported.py interface.
    
    Args:
        data_root (str): Root directory containing the data
        train_locations (list): List of training location names
        test_locations (list): List of test location names
        samples_per_location (int): Maximum samples per location
        device: Training device (CPU/GPU)
        config (dict): Training configuration
        
    Returns:
        dict: Evaluation metrics
    """
    # Create trainer
    trainer = VDSRTrainer(device, config)
    
    # Run training
    metrics = trainer.train(data_root, train_locations, test_locations, samples_per_location)
    
    # Generate comparison image (optional - can be imported from main script)
    try:
        from utils.visualization import create_side_by_side_comparison, get_random_sydney_sample
        test_dataset = SuperResolutionDataset(data_root, test_locations, max_samples_per_location=samples_per_location)
        trainer.generate_comparison_image(test_dataset, create_side_by_side_comparison, random_seed=44)
    except ImportError:
        print("  ⚠️  Could not import visualization functions - skipping comparison image generation")
    
    # Return both metrics and trainer for model saving
    return metrics, trainer


if __name__ == "__main__":
    """
    Standalone testing of the VDSR trainer.
    """
    import torch
    import sys
    
    # Check for quick test flag
    QUICK_TEST = '--quick' in sys.argv or '--test' in sys.argv
    MEDIUM_TEST = '--medium' in sys.argv
    
    # Configuration for testing
    if QUICK_TEST:
        config = {
            'epochs': 3,
            'lr': 1e-3,
            'filters': 64,
            'batch_size': 16,
            'layers': 6
        }
        samples_per_location = 100
    elif MEDIUM_TEST:
        config = {
            'epochs': 15,
            'lr': 1e-4,
            'filters': 64,
            'batch_size': 12,
            'layers': 16
        }
        samples_per_location = 200
    else:
        config = {
            'epochs': 30,
            'lr': 1e-4,
            'filters': 64,
            'batch_size': 8,
            'layers': 20
        }
        samples_per_location = 2000
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Test parameters
    data_root = "processed_data"
    train_locations = ["KANSAS", "PARIS", "PNG"]
    test_locations = ["SYDNEY"]
    
    print("="*60)
    if QUICK_TEST:
        print("🧪 TESTING VDSR TRAINER MODULE - QUICK TEST")
    elif MEDIUM_TEST:
        print("🧪 TESTING VDSR TRAINER MODULE - MEDIUM TEST")
    else:
        print("🧪 TESTING VDSR TRAINER MODULE - FULL TRAINING")
    print("="*60)
    
    # Run training
    metrics = run_vdsr(data_root, train_locations, test_locations, samples_per_location, device, config)
    
    print("\n" + "="*60)
    print("🏁 TESTING COMPLETE")
    print("="*60)
    print(f"Final metrics: {metrics}")
