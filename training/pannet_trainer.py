#!/usr/bin/env python3
"""
PanNet Trainer Module
Modular training implementation for PanNet super-resolution model.
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pannet import PanNet
from utils.data_loader import SuperResolutionDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam_multiband, calculate_ergas


class PanNetTrainer:
    """
    PanNet Trainer class for super-resolution training and evaluation.
    
    This class encapsulates the complete training pipeline for PanNet including:
    - Model initialization and configuration
    - Training loop with loss tracking
    - Evaluation with comprehensive metrics
    - Visualization generation
    """
    
    def __init__(self, device, config):
        """
        Initialize PanNet trainer.
        
        Args:
            device: Training device (CPU/GPU)
            config (dict): Training configuration containing:
                - epochs: Number of training epochs
                - lr: Learning rate
                - filters: Number of filters in the model
                - batch_size: Batch size for training
        """
        self.device = device
        self.config = config
        
        # Initialize model components
        self.model = None
        self.criterion = None
        self.optimizer = None
        
        print(f"PanNet Trainer initialized!")
        print(f"Configuration: {config['epochs']} epochs, LR={config['lr']}, filters={config['filters']}")
        use_pan = config.get('use_pan', True)
        print(f"PAN usage: {'Enabled' if use_pan else 'Disabled'}")
    
    def create_model(self):
        """Create and initialize the PanNet model."""
        use_pan = self.config.get('use_pan', True)  # Default to True for backward compatibility
        self.model = PanNet(num_filters=self.config['filters'], use_pan=use_pan).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4, 
            amsgrad=True
        )
        
        print(f"  ✅ Model created with {self.config['filters']} filters")
        print(f"  ✅ Optimizer: AdamW (lr=1e-3, weight_decay=1e-4, amsgrad=True)")
        print(f"  ✅ Loss function: MSE Loss")
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            lrms = batch['lrms'].to(self.device)
            pan = batch['pan'].to(self.device)
            hrms = batch['hrms'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(lrms, pan)
            loss = self.criterion(output, hrms)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
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
                
                output = self.model(lrms, pan)
                
                # Convert to numpy for metrics
                pred_np = output.cpu().numpy()[0]
                target_np = hrms.cpu().numpy()[0]
                
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
        Complete training pipeline for PanNet.
        
        Args:
            data_root (str): Root directory containing the data
            train_locations (list): List of training location names
            test_locations (list): List of test location names
            samples_per_location (int): Maximum samples per location
            
        Returns:
            dict: Final evaluation metrics
        """
        print(f"Training PanNet: {self.config['epochs']} epochs, LR={self.config['lr']}, filters={self.config['filters']}")
        
        # Create datasets
        train_dataset = SuperResolutionDataset(data_root, train_locations, max_samples_per_location=samples_per_location)
        test_dataset = SuperResolutionDataset(data_root, test_locations, max_samples_per_location=samples_per_location)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        print(f"  📊 Training samples: {len(train_dataset)}")
        print(f"  📊 Test samples: {len(test_dataset)}")
        
        # Create model
        self.create_model()
        
        # Training loop
        print("  🚀 Starting training...")
        for epoch in range(self.config['epochs']):
            self.train_epoch(train_loader, epoch, self.config['epochs'])
        
        print("  ✅ Training completed!")
        
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
            
            # Get model prediction
            self.model.eval()
            with torch.no_grad():
                reconstructed_sample = self.model(lrms_batch, pan_batch).squeeze(0)
            
            # Create output directory if it doesn't exist
            os.makedirs("comparison_images", exist_ok=True)
            
            # Save comparison image
            save_path = f"comparison_images/PanNet_comparison_{int(time.time())}.png"
            visualization_func(
                lrms_original, 
                hrms_sample, 
                reconstructed_sample, 
                "PanNet", 
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
            'model_class': 'PanNet'
        }, model_path)
        
        print(f"  💾 Model saved to: {model_path}")
        return model_path


def run_pannet(data_root, train_locations, test_locations, samples_per_location, device, config):
    """
    Convenience function to run PanNet training using the modular trainer.
    
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
    trainer = PanNetTrainer(device, config)
    
    # Run training
    metrics = trainer.train(data_root, train_locations, test_locations, samples_per_location)
    
    # Generate comparison image (optional - can be imported from main script)
    try:
        from utils.visualization import create_side_by_side_comparison, get_random_sydney_sample
        test_dataset = SuperResolutionDataset(data_root, test_locations, max_samples_per_location=samples_per_location)
        trainer.generate_comparison_image(test_dataset, create_side_by_side_comparison, random_seed=42)
    except ImportError:
        print("  ⚠️  Could not import visualization functions - skipping comparison image generation")
    
    # Return both metrics and trainer for model saving
    return metrics, trainer


if __name__ == "__main__":
    """
    Standalone testing of the PanNet trainer.
    """
    import torch
    
    # Configuration for testing
    config = {
        'epochs': 5,
        'lr': 1e-4,
        'filters': 48,
        'batch_size': 16
    }
    
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
    samples_per_location = 100  # Small number for testing
    
    print("="*60)
    print("🧪 TESTING PANNET TRAINER MODULE")
    print("="*60)
    
    # Run training
    metrics = run_pannet(data_root, train_locations, test_locations, samples_per_location, device, config)
    
    print("\n" + "="*60)
    print("🏁 TESTING COMPLETE")
    print("="*60)
    print(f"Final metrics: {metrics}")
