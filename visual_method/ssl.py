import os
import numpy as np
import argparse
import sys
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import umap.umap_ as umap
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader.tinyimagenet_dataset import get_tinyimagenet_dataset
from torch.utils.data import DataLoader

def transform(use_grayscale3=False):
        base = [transforms.Grayscale(3)] if use_grayscale3 else []
        return transforms.Compose(base + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def transform_base(use_grayscale3=False):
        base = [transforms.Grayscale(3)] if use_grayscale3 else []
        return transforms.Compose(base + [
                transforms.ToTensor()
            ])

# Create argument parser
parser = argparse.ArgumentParser(description='UMAP and t-SNE dimensionality reduction with SSL metrics')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='Dataset to use')
parser.add_argument('--data', type=str, default='./data',
                    help='Path to dataset')
parser.add_argument('--batch_size', type=int, default=24,
                    help='Batch size for data loading')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU id to use')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--save_dir', type=str, default='./results_visual',
                    help='Directory to save results')
parser.add_argument('--method', type=str, default='both', 
                    choices=['umap','pca', 'both'],
                    help='Dimensionality reduction method to use (umap/pca/both)')

args = parser.parse_args()
    
    # Get dataset based on args
use_grayscale = (args.dataset == 'mnist')
if args.dataset == 'cifar10':
        # dataset1: 原始图像 (无增强) - features1
        #train = datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_base(False))
        test = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_base(False))
        #dataset1 = torch.utils.data.ConcatDataset([train, test])
        
        # dataset2: 增强图像 (有变换) - features2
        #train2 = datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform(False))
        test2 = datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform(False))
        #dataset2 = torch.utils.data.ConcatDataset([train2, test2])
        num_classes = 10
elif args.dataset == 'mnist':
        # dataset1: 原始图像 (无增强) - features1
        #train = datasets.MNIST(root=args.data, train=True, download=True, transform=transform_base(True))
        test = datasets.MNIST(root=args.data, train=False, download=True, transform=transform_base(True))
       # dataset1 = torch.utils.data.ConcatDataset([train, test])

        # dataset2: 增强图像 (有变换) - features2
        #train2 = datasets.MNIST(root=args.data, train=True, download=True, transform=transform(True))
        test2 = datasets.MNIST(root=args.data, train=False, download=True, transform=transform(True))
        #dataset2 = torch.utils.data.ConcatDataset([train2, test2])
        num_classes = 10
elif args.dataset == 'tinyimagenet':
        # TinyImageNet requires custom loading  
        # dataset1: 原始图像 (无增强) - features1
        test_full =  get_tinyimagenet_dataset(args.data, split='val',
                                          transform=transform_base(False))
        test2_full = get_tinyimagenet_dataset(args.data, split='val',
                                         transform=transform(False))

        test_len = len(test_full)
        test = torch.utils.data.Subset(test_full, range(test_len // 2))
        test2 = torch.utils.data.Subset(test2_full, range(test_len // 2))
        num_classes = 200
else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

data1_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
data2_loader = DataLoader(test2, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Set device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

    # Set random seed if specified
if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

# Load all data into memory for UMAP and t-SNE
print("Loading data into memory...")
print(f"Loading training data... (this may take a minute)")
data1_list = []
labels1_list = []
for images, labels in data1_loader:
    data1_list.append(images.numpy())
    labels1_list.append(labels.numpy())
data1_array = np.concatenate(data1_list, axis=0)
labels1_array = np.concatenate(labels1_list, axis=0)

# Flatten images: (N, C, H, W) -> (N, C*H*W)
data1_flat = data1_array.reshape(data1_array.shape[0], -1)
print(f"✓ Data1 shape: {data1_flat.shape}, Labels1 shape: {labels1_array.shape}")

print(f"Loading test data...")
data2_list = []
labels2_list = []
for images, labels in data2_loader:
    data2_list.append(images.numpy())
    labels2_list.append(labels.numpy())
data2_array = np.concatenate(data2_list, axis=0)
labels2_array = np.concatenate(labels2_list, axis=0)

# Flatten images: (N, C, H, W) -> (N, C*H*W)
data2_flat = data2_array.reshape(data2_array.shape[0], -1)
print(f"✓ Data2 shape: {data2_flat.shape}, Labels2 shape: {labels2_array.shape}")

# Clear memory
del data1_loader, data2_loader, data1_list, data2_list
del data1_array, data2_array
import gc
gc.collect()
print("✓ Memory cleared, ready for dimensionality reduction...")

print(f"Method: {args.method.upper()}")
print(f"Starting processing...\n")

candidate_dims = [32, 64, 128, 256, 384, 512, 768, 1024, 2048]
print(f"\nTotal dimensions to process: {len(candidate_dims)}")        

for dim in candidate_dims:
    print(f"\n{'='*60}")
    print(f"Processing dimension: {dim}")
    print(f"{'='*60}")

    # PCA
    if args.method in ['pca', 'both']:
        print(f"\n--- PCA ---")
        pca = PCA(n_components=dim, random_state=42)
        print(f"Fitting PCA on data1...")
        embedding_pca1 = pca.fit_transform(data1_flat)
        print(f"Transforming data2 with PCA...")

        pca2 = PCA(n_components=dim, random_state=42)
        embedding_pca2 = pca2.fit_transform(data2_flat)
        
        # Save PCA features
        pca_features1_dir = os.path.join(args.save_dir, 'pca', f'{args.dataset}_test1')
        pca_features2_dir = os.path.join(args.save_dir, 'pca', f'{args.dataset}_test2')
        os.makedirs(pca_features1_dir, exist_ok=True)
        os.makedirs(pca_features2_dir, exist_ok=True)
        
        np.save(os.path.join(pca_features1_dir, f'{args.dataset}_dim{dim}.npy'), embedding_pca1)
        np.save(os.path.join(pca_features2_dir, f'{args.dataset}_dim{dim}.npy'), embedding_pca2)
        print(f"✅ Saved PCA features for dimension {dim}")

    # UMAP
    if args.method in ['umap', 'both']:
        print(f"\n--- UMAP ---")
        reducer = umap.UMAP(n_components=dim, n_neighbors=15, min_dist=0.1, random_state=42)
        print(f"Fitting UMAP on data1...")
        embedding_umap1 = reducer.fit_transform(data1_flat)
        print(f"Transforming data2 with UMAP...")
        embedding_umap2 = reducer.fit_transform(data2_flat)
        
        # Save UMAP features
        umap_features1_dir = os.path.join(args.save_dir, 'umap', f'{args.dataset}_test1')
        umap_features2_dir = os.path.join(args.save_dir, 'umap', f'{args.dataset}_test2')
        os.makedirs(umap_features1_dir, exist_ok=True)
        os.makedirs(umap_features2_dir, exist_ok=True)
        
        np.save(os.path.join(umap_features1_dir, f'{args.dataset}_dim{dim}.npy'), embedding_umap1)
        np.save(os.path.join(umap_features2_dir, f'{args.dataset}_dim{dim}.npy'), embedding_umap2)
        print(f"✅ Saved UMAP features for dimension {dim}")

print(f"\n{'='*60}")
print("All dimensions processed successfully!")
print(f"Results saved to: {args.save_dir}")
print(f"{'='*60}")

   
   
