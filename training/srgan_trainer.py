#!/usr/bin/env python3
"""
SRGAN Trainer Module
Modular training implementation for SRGAN super-resolution model.
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.srgan import Generator, Discriminator
from utils.data_loader import SuperResolutionDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam_multiband, calculate_ergas


class PanNetSRGANLoss:
    """
    PanNet-SRGAN Loss Function
    
    Combines content loss, gradient loss, and adversarial loss
    """
    
    def __init__(self, content_weight=1.0, gradient_weight=0.1, adversarial_weight=0.0):
        self.content_weight = content_weight
        self.gradient_weight = gradient_weight
        self.adversarial_weight = adversarial_weight
        
        self.mse_loss = torch.nn.MSELoss()
        self.gradient_loss = torch.nn.L1Loss()
        self.bce_loss = torch.nn.BCELoss()
        
        print(f"PanNet-SRGAN Loss initialized!")
        print(f"Content weight: {content_weight}")
        print(f"Gradient weight: {gradient_weight}")
        print(f"Adversarial weight: {adversarial_weight}")
    
    def compute_gradients(self, img):
        """Compute gradients using Sobel filters"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        
        # Apply Sobel filters to each channel separately
        grad_x = F.conv2d(img, sobel_x.expand(img.size(1), 1, 3, 3), groups=img.size(1), padding=1)
        grad_y = F.conv2d(img, sobel_y.expand(img.size(1), 1, 3, 3), groups=img.size(1), padding=1)
        
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    def content_loss(self, sr_output, hr_target):
        """Content loss (MSE)"""
        return self.mse_loss(sr_output, hr_target)
    
    def adversarial_loss(self, fake_output):
        """Adversarial loss for generator"""
        real_labels = torch.ones_like(fake_output)
        return self.bce_loss(fake_output, real_labels)
    
    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss"""
        real_labels = torch.ones_like(real_output)
        fake_labels = torch.zeros_like(fake_output)
        
        real_loss = self.bce_loss(real_output, real_labels)
        fake_loss = self.bce_loss(fake_output, fake_labels)
        
        return (real_loss + fake_loss) * 0.5
    
    def generator_loss(self, sr_output, hr_target, fake_output):
        """Combined generator loss"""
        # Content loss
        content_loss = self.content_loss(sr_output, hr_target)
        
        # Gradient loss (Mixed Gradient Loss)
        sr_gradients = self.compute_gradients(sr_output)
        hr_gradients = self.compute_gradients(hr_target)
        gradient_loss = self.gradient_loss(sr_gradients, hr_gradients)
        
        # Adversarial loss
        adv_loss = self.adversarial_loss(fake_output)
        
        # Combined loss
        total_loss = (self.content_weight * content_loss + 
                     self.gradient_weight * gradient_loss + 
                     self.adversarial_weight * adv_loss)
        
        return total_loss, content_loss, adv_loss


class SRGANTrainer:
    """
    SRGAN Trainer class for super-resolution training and evaluation.
    
    This class encapsulates the complete training pipeline for SRGAN including:
    - Generator and Discriminator initialization
    - Adversarial training with content-only pre-training
    - Evaluation with comprehensive metrics
    - Visualization generation
    """
    
    def __init__(self, device, config):
        """
        Initialize SRGAN trainer.
        
        Args:
            device: Training device (CPU/GPU)
            config (dict): Training configuration containing:
                - epochs: Number of training epochs
                - lr: Learning rate
                - batch_size: Batch size for training
                - content_weight: Weight for content loss
                - gradient_weight: Weight for gradient loss
                - adversarial_weight: Weight for adversarial loss
                - content_only_epochs: Number of content-only pre-training epochs
        """
        self.device = device
        self.config = config
        
        # Initialize model components
        self.generator = None
        self.discriminator = None
        self.loss_fn = None
        self.g_optimizer = None
        self.d_optimizer = None
        
        print(f"SRGAN Trainer initialized!")
        print(f"Configuration: {config['epochs']} epochs, LR={config['lr']}, batch_size={config['batch_size']}")
        print(f"Loss weights: content={config.get('content_weight', 1.0)}, gradient={config.get('gradient_weight', 0.1)}, adversarial={config.get('adversarial_weight', 0.0)}")
        use_pan = config.get('use_pan', True)
        print(f"PAN usage: {'Enabled' if use_pan else 'Disabled'}")
    
    def create_model(self):
        """Create and initialize the SRGAN Generator and Discriminator."""
        # Create models
        use_pan = self.config.get('use_pan', True)  # Default to True for backward compatibility
        self.generator = Generator(
            ms_channels=6, 
            pan_channels=1, 
            num_filters=64,
            use_pan=use_pan
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_channels=6
        ).to(self.device)
        
        # Create loss function
        self.loss_fn = PanNetSRGANLoss(
            content_weight=self.config.get('content_weight', 1.0),
            gradient_weight=self.config.get('gradient_weight', 0.1),
            adversarial_weight=self.config.get('adversarial_weight', 0.0)
        )
        
        # Create optimizers (conservative learning rates)
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=self.config['lr'], 
            betas=(0.9, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=self.config['lr'] * 0.1, 
            betas=(0.9, 0.999)
        )
        
        print(f"  ✅ Generator created with {self.generator.count_parameters()} parameters")
        print(f"  ✅ Discriminator created with {self.discriminator.count_parameters()} parameters")
        print(f"  ✅ Generator optimizer: Adam (lr={self.config['lr']})")
        print(f"  ✅ Discriminator optimizer: Adam (lr={self.config['lr'] * 0.1})")
    
    def train_epoch(self, train_loader, epoch, total_epochs, content_only_epochs=2):
        """
        Train for one epoch with alternating Generator/Discriminator updates.
        
        Args:
            train_loader: Training data loader
            epoch (int): Current epoch number
            total_epochs (int): Total number of epochs
            content_only_epochs (int): Number of content-only pre-training epochs
            
        Returns:
            dict: Average losses for the epoch
        """
        self.generator.train()
        self.discriminator.train()
        
        g_losses, d_losses, content_losses, adv_losses = [], [], [], []
        
        # Content-only pre-training for first few epochs
        # If adversarial_weight is 0, stay in content-only mode for ALL epochs
        content_only_mode = epoch < content_only_epochs or self.config.get('adversarial_weight', 0.0) == 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            lrms = batch['lrms'].to(self.device)
            pan = batch['pan'].to(self.device)
            hrms = batch['hrms'].to(self.device)
            
            batch_size = lrms.size(0)
            
            # ---------------------
            # Train Discriminator (LESS FREQUENTLY)
            # ---------------------
            if not content_only_mode and batch_idx % 4 == 0:
                self.d_optimizer.zero_grad()
                
                # Generate fake images
                with torch.no_grad():
                    fake_sr = self.generator(lrms, pan)
                
                # Discriminator predictions
                real_output = self.discriminator(hrms)
                fake_output = self.discriminator(fake_sr.detach())
                
                # Discriminator loss
                d_loss = self.loss_fn.discriminator_loss(real_output, fake_output)
                
                d_loss.backward()
                self.d_optimizer.step()
            else:
                # Skip discriminator training this batch
                d_loss = torch.tensor(0.0, device=self.device)
                real_output = torch.tensor(0.0, device=self.device)
                fake_output = torch.tensor(0.0, device=self.device)
            
            # ---------------------
            # Train Generator (EVERY BATCH)
            # ---------------------
            self.g_optimizer.zero_grad()
            
            # Generate fake images again
            fake_sr = self.generator(lrms, pan)
            
            # Content-only loss during pre-training phase
            if content_only_mode:
                # Only use simple L1 loss during pre-training (like srgan_fixed.py)
                content_loss = F.l1_loss(fake_sr, hrms)
                adv_loss = torch.tensor(0.0, device=self.device)
                g_loss = content_loss
                fake_output = torch.tensor(0.0, device=self.device)  # No discriminator needed
            else:
                # Full SRGAN loss after pre-training
                fake_output = self.discriminator(fake_sr)  # Call discriminator for adversarial training
                g_loss, content_loss, adv_loss = self.loss_fn.generator_loss(fake_sr, hrms, fake_output)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Store losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            content_losses.append(content_loss.item())
            adv_losses.append(adv_loss.item())
        
        # Calculate average losses
        avg_metrics = {
            'g_loss': np.mean(g_losses),
            'd_loss': np.mean(d_losses),
            'content_loss': np.mean(content_losses),
            'adv_loss': np.mean(adv_losses)
        }
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            mode_str = "Content-only" if content_only_mode else "Adversarial"
            print(f"  Epoch {epoch+1}/{total_epochs} ({mode_str}) - G: {avg_metrics['g_loss']:.6f}, D: {avg_metrics['d_loss']:.6f}, Content: {avg_metrics['content_loss']:.6f}, Adv: {avg_metrics['adv_loss']:.6f}")
        
        return avg_metrics
    
    def evaluate(self, test_loader):
        """
        Evaluate the trained models on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            dict: Evaluation metrics (psnr, ssim, ergas, sam)
        """
        self.generator.eval()
        self.discriminator.eval()
        
        psnr_list, ssim_list, sam_list, ergas_list = [], [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                lrms = batch['lrms'].to(self.device)
                pan = batch['pan'].to(self.device)
                hrms_target = batch['hrms'].to(self.device)
                
                # Generate super-resolution image
                sr_output = self.generator(lrms, pan)
                
                # Convert to numpy for metrics
                pred_np = sr_output.cpu().numpy()[0]
                target_np = hrms_target.cpu().numpy()[0]
                
                # Clamp to [0, 1] range
                pred_np = np.clip(pred_np, 0, 1)
                target_np = np.clip(target_np, 0, 1)
                
                # Compute metrics for each band and average
                psnr_bands = []
                ssim_bands = []
                for b in range(pred_np.shape[0]):
                    psnr_bands.append(calculate_psnr(target_np[b], pred_np[b]))
                    ssim_bands.append(calculate_ssim(target_np[b], pred_np[b]))
                
                psnr_list.append(np.mean(psnr_bands))
                ssim_list.append(np.mean(ssim_bands))
                
                # For SAM and ERGAS, use the full stack
                sam_list.append(calculate_sam_multiband(target_np, pred_np))
                ergas_list.append(calculate_ergas(target_np, pred_np))
        
        # Calculate average metrics
        metrics = {
            'psnr': np.mean(psnr_list),
            'ssim': np.mean(ssim_list),
            'sam': np.mean(sam_list),
            'ergas': np.mean(ergas_list)
        }
        
        print(f"  Test Results - PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}, ERGAS: {metrics['ergas']:.4f}, SAM: {metrics['sam']:.4f}")
        
        return metrics
    
    def train(self, data_root, train_locations, test_locations, samples_per_location):
        """
        Complete training pipeline for SRGAN.
        
        Args:
            data_root (str): Root directory containing the data
            train_locations (list): List of training location names
            test_locations (list): List of test location names
            samples_per_location (int): Maximum samples per location
            
        Returns:
            dict: Final evaluation metrics
        """
        print(f"Training SRGAN: {self.config['epochs']} epochs, LR={self.config['lr']}, batch_size={self.config['batch_size']}")
        
        # Create datasets
        train_dataset = SuperResolutionDataset(data_root, train_locations, max_samples_per_location=samples_per_location)
        test_dataset = SuperResolutionDataset(data_root, test_locations, max_samples_per_location=samples_per_location)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"  📊 Training samples: {len(train_dataset)}")
        print(f"  📊 Test samples: {len(test_dataset)}")
        
        # Create models
        self.create_model()
        
        # Training loop
        print("  🚀 Starting training...")
        content_only_epochs = self.config.get('content_only_epochs', 2)
        
        for epoch in range(self.config['epochs']):
            self.train_epoch(train_loader, epoch, self.config['epochs'], content_only_epochs)
        
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
            self.generator.eval()
            with torch.no_grad():
                reconstructed_sample = self.generator(lrms_batch, pan_batch).squeeze(0)
            
            # Create output directory if it doesn't exist
            os.makedirs("comparison_images", exist_ok=True)
            
            # Save comparison image
            save_path = f"comparison_images/SRGAN_comparison_{int(time.time())}.png"
            visualization_func(
                lrms_original, 
                hrms_sample, 
                reconstructed_sample, 
                "SRGAN", 
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
        
        # Save both generator and discriminator state dicts
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'model_class': 'SRGAN'
        }, model_path)
        
        print(f"  💾 Model saved to: {model_path}")
        return model_path


def run_srgan(data_root, train_locations, test_locations, samples_per_location, device, config):
    """
    Convenience function to run SRGAN training using the modular trainer.
    
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
    trainer = SRGANTrainer(device, config)
    
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
    Standalone testing of the SRGAN trainer.
    """
    import torch
    import sys
    
    # Check for quick test flag
    QUICK_TEST = '--quick' in sys.argv or '--test' in sys.argv
    MEDIUM_TEST = '--medium' in sys.argv
    
    # Configuration for testing
    if QUICK_TEST:
        config = {
            'epochs': 5,
            'lr': 1e-4,
            'batch_size': 8,
            'content_weight': 1.0,
            'gradient_weight': 0.1,
            'adversarial_weight': 0.0,
            'content_only_epochs': 4
        }
        samples_per_location = 100
    elif MEDIUM_TEST:
        config = {
            'epochs': 15,
            'lr': 1e-4,
            'batch_size': 6,
            'content_weight': 5.0,       # Balanced content weight
            'gradient_weight': 0.5,      # Balanced gradient weight
            'adversarial_weight': 5e-5,  # Moderate adversarial weight
            'content_only_epochs': 8     # More content-only pre-training
        }
        samples_per_location = 200
    else:
        config = {
            'epochs': 60,
            'lr': 1e-4,
            'batch_size': 8,
            'content_weight': 10.0,      # Best performing content weight
            'gradient_weight': 1.0,      # Best performing gradient weight
            'adversarial_weight': 1e-4,  # True GAN with discriminator
            'content_only_epochs': 20    # Good pre-training period
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
        print("🧪 TESTING SRGAN TRAINER MODULE - QUICK TEST")
    elif MEDIUM_TEST:
        print("🧪 TESTING SRGAN TRAINER MODULE - MEDIUM TEST")
    else:
        print("🧪 TESTING SRGAN TRAINER MODULE - FULL TRAINING")
    print("="*60)
    
    # Run training
    metrics = run_srgan(data_root, train_locations, test_locations, samples_per_location, device, config)
    
    print("\n" + "="*60)
    print("🏁 TESTING COMPLETE")
    print("="*60)
    print(f"Final metrics: {metrics}")
