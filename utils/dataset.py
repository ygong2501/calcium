"""
Dataset management for calcium simulation.
"""
import os
import json
import numpy as np
import random
import glob
from pathlib import Path
from datetime import datetime


class DatasetManager:
    """
    Manages the creation and organization of simulation datasets.
    """
    
    def __init__(self, output_dir='./output'):
        """
        Initialize the dataset manager.
        
        Args:
            output_dir (str): Base output directory.
        """
        self.output_dir = output_dir
        self.dataset_dir = None
        
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def create_dataset(self, name=None, split_ratios=(0.7, 0.15, 0.15)):
        """
        Create a new dataset with train/val/test splits.
        
        Args:
            name (str, optional): Name of the dataset. If None, uses timestamp.
            split_ratios (tuple): Train/validation/test split ratios.
        
        Returns:
            dict: Dataset directory structure.
        """
        # Validate split ratios
        if sum(split_ratios) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Generate dataset name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"dataset_{timestamp}"
        
        # Create dataset directory
        dataset_path = os.path.join(self.output_dir, name)
        
        # Create subdirectories for train/val/test
        train_dir = os.path.join(dataset_path, 'train')
        val_dir = os.path.join(dataset_path, 'val')
        test_dir = os.path.join(dataset_path, 'test')
        
        # Create image and label directories
        for split_dir in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
        
        # Create dataset info file
        dataset_info = {
            'name': name,
            'created': datetime.now().isoformat(),
            'split_ratios': split_ratios,
            'train_dir': train_dir,
            'val_dir': val_dir,
            'test_dir': test_dir
        }
        
        with open(os.path.join(dataset_path, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        self.dataset_dir = dataset_path
        return dataset_info
    
    def add_simulation_to_dataset(self, images, labels, simulation_info, split='random'):
        """
        Add a simulation's images and labels to the dataset.
        
        Args:
            images (list): List of image file paths.
            labels (list): List of label file paths.
            simulation_info (dict): Simulation parameters and metadata.
            split (str): Which split to add to ('train', 'val', 'test', or 'random').
        
        Returns:
            dict: Information about the added simulation.
        """
        if self.dataset_dir is None:
            raise ValueError("No dataset created. Call create_dataset first.")
        
        # If random split, randomly assign to train/val/test based on dataset ratios
        if split == 'random':
            with open(os.path.join(self.dataset_dir, 'dataset_info.json'), 'r') as f:
                dataset_info = json.load(f)
            
            ratios = dataset_info['split_ratios']
            # Choose split based on probabilities
            choices = ['train', 'val', 'test']
            split = random.choices(choices, weights=ratios, k=1)[0]
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be 'train', 'val', 'test', or 'random'")
        
        # Prepare destination directories
        dest_img_dir = os.path.join(self.dataset_dir, split, 'images')
        dest_label_dir = os.path.join(self.dataset_dir, split, 'labels')
        
        # Copy or move images and labels
        import shutil
        for img_path in images:
            dest_path = os.path.join(dest_img_dir, os.path.basename(img_path))
            shutil.copy2(img_path, dest_path)
        
        for label_path in labels:
            dest_path = os.path.join(dest_label_dir, os.path.basename(label_path))
            shutil.copy2(label_path, dest_path)
        
        # Return info about the added simulation
        return {
            'simulation_id': simulation_info.get('simulation_id', 'unknown'),
            'split': split,
            'image_count': len(images),
            'label_count': len(labels)
        }
    
    def split_existing_dataset(self, input_dir, split_ratios=(0.7, 0.15, 0.15)):
        """
        Split an existing dataset into train/val/test.
        
        Args:
            input_dir (str): Directory containing images and labels subdirectories.
            split_ratios (tuple): Train/validation/test split ratios.
        
        Returns:
            dict: Information about the split dataset.
        """
        # Validate input directory
        img_dir = os.path.join(input_dir, 'images')
        label_dir = os.path.join(input_dir, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            raise ValueError("Input directory must contain 'images' and 'labels' subdirectories")
        
        # Create a new dataset
        dataset_info = self.create_dataset(
            name=os.path.basename(input_dir) + '_split',
            split_ratios=split_ratios
        )
        
        # Get all image files
        image_files = glob.glob(os.path.join(img_dir, '*.png'))
        
        # Shuffle the files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_files = len(image_files)
        n_train = int(n_files * split_ratios[0])
        n_val = int(n_files * split_ratios[1])
        
        # Split the files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Process each split
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, img_files in splits.items():
            # Get corresponding label files
            label_files = []
            for img_file in img_files:
                img_basename = os.path.basename(img_file)
                label_basename = os.path.splitext(img_basename)[0] + '.json'
                label_path = os.path.join(label_dir, label_basename)
                if os.path.exists(label_path):
                    label_files.append(label_path)
            
            # Add to dataset
            self.add_simulation_to_dataset(
                img_files, label_files, 
                {'simulation_id': f'split_{split_name}'}, 
                split=split_name
            )
        
        return {
            'dataset_info': dataset_info,
            'stats': {
                'total_images': n_files,
                'train_images': len(train_files),
                'val_images': len(val_files),
                'test_images': len(test_files)
            }
        }
    
    def load_dataset(self, dataset_path):
        """
        Load an existing dataset.
        
        Args:
            dataset_path (str): Path to dataset directory.
        
        Returns:
            dict: Dataset information.
        """
        # Validate dataset path
        info_file = os.path.join(dataset_path, 'dataset_info.json')
        if not os.path.exists(info_file):
            raise ValueError(f"No dataset info found at {dataset_path}")
        
        # Load dataset info
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        
        self.dataset_dir = dataset_path
        return dataset_info
    
    def generate_dataset_stats(self):
        """
        Generate statistics for the current dataset.
        
        Returns:
            dict: Dataset statistics.
        """
        if self.dataset_dir is None:
            raise ValueError("No dataset loaded or created.")
        
        stats = {
            'num_images': {},
            'num_labels': {},
            'simulation_types': {},
            'parameter_distributions': {}
        }
        
        # Count images and labels in each split
        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(self.dataset_dir, split, 'images')
            label_dir = os.path.join(self.dataset_dir, split, 'labels')
            
            stats['num_images'][split] = len(glob.glob(os.path.join(img_dir, '*.png')))
            stats['num_labels'][split] = len(glob.glob(os.path.join(label_dir, '*.json')))
        
        # Analyze labels to get simulation types and parameter distributions
        all_labels = []
        for split in ['train', 'val', 'test']:
            label_dir = os.path.join(self.dataset_dir, split, 'labels')
            label_files = glob.glob(os.path.join(label_dir, '*.json'))
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                    all_labels.append(label_data)
        
        # Extract simulation types
        sim_types = {}
        for label in all_labels:
            sim_type = label.get('simulation_type', 'unknown')
            sim_types[sim_type] = sim_types.get(sim_type, 0) + 1
        
        stats['simulation_types'] = sim_types
        
        # Save stats to file
        stats_file = os.path.join(self.dataset_dir, 'dataset_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        return stats