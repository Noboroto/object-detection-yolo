# -*- coding: utf-8 -*-
"""
Quick Windows Fix for YOLO8 DataLoader
Run this to test the DataLoader with Windows-safe settings
"""

if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path
    
    # Add current directory to path
    sys.path.append('.')
    
    try:
        from yolo8 import Dataset, DATASET_PATH
        from torch.utils.data import DataLoader
        import torch
        
        print("QUICK WINDOWS DATALOADER TEST")
        print("=" * 50)
        
        # Dataset setup
        train_dir = os.path.join(DATASET_PATH, "train", "images")
        
        if not os.path.exists(train_dir):
            print(f"Dataset not found: {train_dir}")
            print("Please check your dataset path")
            exit(1)
        
        # Quick file listing
        train_path = Path(train_dir)
        filenames_train = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
        filenames_train = [str(p) for p in filenames_train[:100]]  # Take first 100 for testing
        
        print(f"Testing with {len(filenames_train)} images")
        
        # Simple parameters
        params = {
            'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
            'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
            'degrees': 0.0, 'translate': 0.0, 'scale': 1.0,
            'shear': 0.0, 'flip_ud': 0.0, 'flip_lr': 0.0,
            'mosaic': 0.0, 'mix_up': 0.0,
            'nc': 5, 'names': ['Elephant', 'Giraffe', 'Leopard', 'Lion', 'Zebra']
        }
        
        # Create dataset
        dataset = Dataset(filenames_train, 640, params, augment=False)
        print(f"Dataset created with {len(dataset)} samples")
        
        # Windows-safe DataLoader (single-threaded)
        print("Creating Windows-safe DataLoader (num_workers=0)...")
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=0,  # Safe for Windows
            pin_memory=False,
            shuffle=False,
            collate_fn=Dataset.collate_fn
        )
        
        print("Testing batch loading...")
        batch = next(iter(dataloader))
        
        print("SUCCESS! Batch loaded:")
        print(f"  Images shape: {batch[0].shape}")
        print(f"  Targets keys: {list(batch[1].keys())}")
        print(f"  Classes shape: {batch[1]['cls'].shape}")
        print(f"  Boxes shape: {batch[1]['box'].shape}")
        
        print("\n" + "=" * 50)
        print("Windows DataLoader fix is working!")
        print("You can now use the DataLoader safely.")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
