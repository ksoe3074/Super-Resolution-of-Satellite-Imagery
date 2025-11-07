#!/usr/bin/env python3
"""
Super-Resolution Model Testing Script - Modular Trainer Versions
Runs all 4 models with and without PAN by importing from their modular trainer files.
Uses the clean, tested implementations with proper separation of concerns.


Architecture:
- pannet.py + pannet_trainer.py → PanNet model
- srcnn.py + srcnn_trainer.py → SRCNN model
- vdsr.py + vdsr_trainer.py → VDSR model
- srgan.py + srgan_trainer.py → SRGAN model


Usage:
    Quick Test (~40 minutes):
        python run_all_models_imported.py --quick
        or
        python run_all_models_imported.py --test
    
    Full Training (~16 hours):
        python run_all_models_imported.py

Quick Test (8 runs total - 4 models × 2 PAN configurations):
    - Total: ~40 minutes
    - Purpose: Verify all models work and compare PAN impact

Full Training (8 runs total - 4 models × 2 PAN configurations):
    - Total: ~21 hours
"""

import os
import time
import torch
import numpy as np

# Import the modular trainer implementations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import SuperResolutionDataset
from training.pannet_trainer import run_pannet as run_pannet_trainer
from training.srcnn_trainer import run_srcnn
from training.vdsr_trainer import run_vdsr
from training.srgan_trainer import run_srgan
from utils.visualization import create_side_by_side_comparison, get_random_sydney_sample
from torch.utils.data import DataLoader

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    import sys
    
    # Check for quick test flag
    QUICK_TEST = '--quick' in sys.argv or '--test' in sys.argv
    
    if QUICK_TEST:
        print("="*80)
        print(" 🧪 QUICK TEST MODE - All Models (~40 minutes)")
        print("="*80)
        print("Running all 4 models with both PAN and no-PAN configurations")
        print("Purpose: Verify all models work and compare PAN impact")
        print()
        
        SAMPLES_PER_LOCATION = 500
        data_root = "processed_data"
        train_locations = ["KANSAS", "PARIS", "PNG"]
        test_locations = ["SYDNEY"]
        
        # Quick test configurations - ~20 minutes total
        configs = {
            'PanNet': {'epochs': 10, 'lr': 1e-4, 'filters': 32, 'batch_size': 16, 'use_pan': True},      # ~3min
            'SRCNN': {'epochs': 3, 'lr': 1e-3, 'filters': 32, 'batch_size': 16, 'use_pan': True},         # ~3min
            'VDSR': {'epochs': 3, 'lr': 1e-3, 'filters': 64, 'layers': 6, 'batch_size': 16, 'use_pan': True},  # ~3min
            'SRGAN': {'epochs': 5, 'lr': 1e-4, 'batch_size': 8, 'content_weight': 1.0, 'gradient_weight': 0.1, 'adversarial_weight': 0.0, 'content_only_epochs': 4, 'use_pan': True}  # ~10min
        }
    else:
        print("="*80)
        print(" 🌙 OVERNIGHT SUPER-RESOLUTION MODEL TESTING (IMPORTED VERSIONS)")
        print("="*80)
        print("Running all 4 models using their modular trainer implementations")
        print("Each model runs TWICE: once with PAN, once without PAN")
        print("All models use 8000 samples per location (24000 total training samples)")
        print("Target runtime: ~20-24 hours (8 total runs: 4 models × 2 PAN configurations)")
        print()
        
        # Fixed parameters for extended runtime - MAXIMUM SAMPLES
        SAMPLES_PER_LOCATION = 6000
        data_root = "processed_data"
        train_locations = ["KANSAS", "PARIS", "PNG"]
        test_locations = ["SYDNEY"]
        
        # Model configurations - IMPROVED for better performance
        configs = {
            'PanNet': {'epochs': 40, 'lr': 1e-4, 'filters': 64, 'batch_size': 16, 'use_pan': True},      # ~1.5h
            'SRCNN': {'epochs': 50, 'lr': 1e-4, 'filters': 64, 'batch_size': 16, 'use_pan': True},        # ~1h
            'VDSR': {'epochs': 60, 'lr': 5e-5, 'filters': 64, 'layers': 20, 'batch_size': 8, 'use_pan': True},  # ~1.5h
            'SRGAN': {'epochs': 60, 'lr': 1e-4, 'batch_size': 8, 'content_weight': 1.0, 'gradient_weight': 0.1, 'adversarial_weight': 1e-4, 'content_only_epochs': 30, 'use_pan': True}  # ~4h
        }
    
    # Try CUDA first (NVIDIA GPU), then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print(f"Device: {device}")
    print()
    
    results = {}
    total_start_time = time.time()
    
    # Run each model with both PAN and no-PAN configurations
    for model_name in ['PanNet', 'SRCNN', 'VDSR', 'SRGAN']:
        base_config = configs[model_name]
        
        # Run with PAN enabled
        print(f"\n{'='*80}")
        print(f" 🚀 RUNNING {model_name} WITH PAN")
        print(f"{'='*80}")
        
        start_time = time.time()
        config_with_pan = base_config.copy()
        config_with_pan['use_pan'] = True
        
        try:
            if model_name == 'PanNet':
                result, trainer = run_pannet_trainer(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_with_pan)
            elif model_name == 'SRCNN':
                result, trainer = run_srcnn(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_with_pan)
            elif model_name == 'VDSR':
                result, trainer = run_vdsr(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_with_pan)
            elif model_name == 'SRGAN':
                result, trainer = run_srgan(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_with_pan)
            
            # Save the trained model
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"models/{model_name.lower()}_with_pan_{timestamp}.pth"
            trainer.save_model(model_filename)
            
            runtime = time.time() - start_time
            results[f"{model_name}_with_PAN"] = {'status': 'success', 'runtime': runtime, 'result': result, 'model_path': model_filename}
            print(f"✅ {model_name} with PAN completed in {runtime/3600:.2f} hours")
            
        except Exception as e:
            runtime = time.time() - start_time
            results[f"{model_name}_with_PAN"] = {'status': 'failed', 'runtime': runtime, 'error': str(e)}
            print(f"❌ {model_name} with PAN failed after {runtime/3600:.2f} hours: {str(e)}")
        
        # Run without PAN
        print(f"\n{'='*80}")
        print(f" 🚀 RUNNING {model_name} WITHOUT PAN")
        print(f"{'='*80}")
        
        start_time = time.time()
        config_no_pan = base_config.copy()
        config_no_pan['use_pan'] = False
        
        try:
            if model_name == 'PanNet':
                result, trainer = run_pannet_trainer(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_no_pan)
            elif model_name == 'SRCNN':
                result, trainer = run_srcnn(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_no_pan)
            elif model_name == 'VDSR':
                result, trainer = run_vdsr(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_no_pan)
            elif model_name == 'SRGAN':
                result, trainer = run_srgan(data_root, train_locations, test_locations, SAMPLES_PER_LOCATION, device, config_no_pan)
            
            # Save the trained model
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"models/{model_name.lower()}_no_pan_{timestamp}.pth"
            trainer.save_model(model_filename)
            
            runtime = time.time() - start_time
            results[f"{model_name}_no_PAN"] = {'status': 'success', 'runtime': runtime, 'result': result, 'model_path': model_filename}
            print(f"✅ {model_name} without PAN completed in {runtime/3600:.2f} hours")
            
        except Exception as e:
            runtime = time.time() - start_time
            results[f"{model_name}_no_PAN"] = {'status': 'failed', 'runtime': runtime, 'error': str(e)}
            print(f"❌ {model_name} without PAN failed after {runtime/3600:.2f} hours: {str(e)}")
        
        # Show progress
        elapsed_total = (time.time() - total_start_time) / 3600
        print(f"⏰ Total time: {elapsed_total:.1f}h")
    
    # Final results
    total_runtime = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(" 🏁 TESTING COMPLETE")
    print(f"{'='*80}")
    
    for model_name, result in results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"{status} {model_name}: {result['status']} ({result['runtime']/3600:.2f}h)")
    
    print(f"\n⏱️  Total runtime: {total_runtime/3600:.2f} hours")
    
    # Save results to text file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{timestamp}.txt"
    
    with open(results_filename, "w") as f:
        f.write(f"Super-Resolution Model Results - {timestamp}\n")
        f.write(f"Mode: {'Quick Test' if QUICK_TEST else 'Full Training'}\n")
        f.write(f"Total runtime: {total_runtime/3600:.2f} hours\n")
        f.write(f"Device: {device}\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Runtime: {result['runtime']/3600:.2f} hours\n")
            
            if result['status'] == 'success' and 'result' in result and result['result']:
                metrics = result['result']
                f.write(f"  PSNR: {metrics.get('psnr', 'N/A'):.4f}\n")
                f.write(f"  SSIM: {metrics.get('ssim', 'N/A'):.4f}\n")
                f.write(f"  ERGAS: {metrics.get('ergas', 'N/A'):.4f}\n")
                f.write(f"  SAM: {metrics.get('sam', 'N/A'):.4f}\n")
                if 'model_path' in result:
                    f.write(f"  Model saved: {result['model_path']}\n")
            elif result['status'] == 'failed':
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
            
            f.write("\n")
    
    print(f"\n📄 Results saved to {results_filename}")
    print(f"🖼️  Comparison images saved to comparison_images/ directory")
    print(f"💾 Trained models saved to models/ directory")


if __name__ == "__main__":
    main()
