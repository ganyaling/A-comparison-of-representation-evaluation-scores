"""
TinyImageNet Dataset Utilities
Provides custom Dataset class for loading TinyImageNet validation set
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetVal(Dataset):
    """
    Custom Dataset for TinyImageNet validation set
    
    Val set structure:
    - All images in: val/images/
    - Labels in: val/val_annotations.txt
    
    val_annotations.txt format:
    val_0.JPEG	n03444034	0	32	44	62
    (filename, class_id, bbox coordinates)
    """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): TinyImageNet root directory path
            transform (callable, optional): Transform to apply to images
        """
        self.root = root
        self.transform = transform
        
        # Read val_annotations.txt
        anno_file = os.path.join(root, 'val', 'val_annotations.txt')
        
        if not os.path.exists(anno_file):
            raise FileNotFoundError(f"Annotation file not found: {anno_file}")
        
        # Build class name to integer label mapping
        # Read all class names from train directory
        train_dir = os.path.join(root, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        
        class_names = sorted([d for d in os.listdir(train_dir) 
                             if os.path.isdir(os.path.join(train_dir, d))])
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.classes = class_names
        
        # Parse val_annotations.txt
        self.samples = []
        self.targets = []  # For compatibility with some PyTorch functions
        
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]      # Image filename
                    class_id = parts[1]      # Class ID (e.g., n03444034)
                    
                    # Build full image path
                    img_path = os.path.join(root, 'val', 'images', img_name)
                    
                    # Convert class ID to integer label
                    if class_id in self.class_to_idx:
                        label = self.class_to_idx[class_id]
                        self.samples.append((img_path, label))
                        self.targets.append(label)
                    else:
                        print(f"Warning: Unknown class_id '{class_id}' for {img_name}")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {anno_file}")
        
        print(f"TinyImageNet Val Dataset initialized:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Number of classes: {len(self.classes)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, label) where label is the integer class index
        """
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


def get_tinyimagenet_dataset(root, split='train', transform=None):
    """
    Get TinyImageNet dataset for specified split
    
    Args:
        root (str): TinyImageNet root directory
        split (str): 'train' or 'val'
        transform (callable, optional): Transform to apply
        
    Returns:
        Dataset: PyTorch Dataset object
    """
    if split == 'train':
        # Train set uses standard ImageFolder structure
        from torchvision.datasets import ImageFolder
        train_dir = os.path.join(root, 'train')
        dataset = ImageFolder(train_dir, transform=transform)
    elif split == 'val':
        # Val set uses custom Dataset with txt annotations
        dataset = TinyImageNetVal(root, transform=transform)
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'")
    
    return dataset
