
# Create shared utility modules first

# 1. dataset.py - Custom Dataset with class balancing

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from collections import Counter

class FacialParalysisDataset(Dataset):
    """Custom Dataset for Facial Paralysis Detection"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NonStroke', 'Stroke']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        self.targets = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.exists(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append(os.path.join(cls_dir, img_name))
                        self.targets.append(self.class_to_idx[cls])
        
        print(f"Loaded {len(self.samples)} images")
        print(f"Class distribution: {dict(Counter(self.targets))}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(train=True, img_size=224):
    """Get data transformations"""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def get_weighted_sampler(dataset):

    # ✅ FIX: handle Subset and normal dataset
    if hasattr(dataset, 'indices'):  # Subset case
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    # Count class distribution
    class_counts = Counter(targets)

    # Compute class weights (inverse frequency)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in targets]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

print("Dataset utilities created successfully!")
