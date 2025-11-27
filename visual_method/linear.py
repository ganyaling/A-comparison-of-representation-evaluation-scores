import os
import sys
import numpy as np

import argparse


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

def transform_train(use_grayscale3=False):
        base = [transforms.Grayscale(3)] if use_grayscale3 else []
        return transforms.Compose(base + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def transform_test(use_grayscale3=False):
        base = [transforms.Grayscale(3)] if use_grayscale3 else []
        return transforms.Compose(base + [
                transforms.ToTensor()
            ])

# Create argument parser
parser = argparse.ArgumentParser(description='UMAP and t-SNE dimensionality reduction with SSL metrics')
parser.add_argument('--dataset_name', type=str, default='cifar10', 
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='Dataset to use')
parser.add_argument('--data_path', type=str, default='./data',
                    help='Path to dataset')
parser.add_argument('--batch_size', type=int, default=24,
                    help='Batch size for data loading')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU id to use')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--dim', type=int, nargs='+', default=list(range(50, 201, 10)),
                    help='Dimensions to evaluate (e.g., 50 100 150 or range)')
parser.add_argument('--save_dir', type=str, default='./results_visual',
                    help='Directory to save results')
parser.add_argument('--method', type=str, default='both', 
                    choices=['umap', 'pca', 'both'],
                    help='Dimensionality reduction method to use (umap/pca/both)')

args = parser.parse_args()
    
    # Get dataset based on args
use_grayscale = (args.dataset_name == 'mnist')
if args.dataset_name == 'cifar10':
    train = datasets.CIFAR10( root=args.data_path, train=True, download=True, transform=transform_train(False))
    test = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test(False))
    num_classes = 10
elif args.dataset_name == 'mnist':
    train = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform_train(True))
    test = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform_test(True))
    num_classes = 10
elif args.dataset_name == 'tinyimagenet':
    train_full = get_tinyimagenet_dataset(args.data_path, split='train', transform=transform_train(False))
    test_full = get_tinyimagenet_dataset(args.data_path, split='val', transform=transform_test(False))
    # 只用前50%数据
    train_len = len(train_full)
    test_len = len(test_full)
    train = torch.utils.data.Subset(train_full, range(train_len // 2))
    test = torch.utils.data.Subset(test_full, range(test_len // 2))
    num_classes = 200
else:
    raise ValueError(f"Unknown dataset: {args.dataset_name}")

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

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

# Load all data into memory for UMAP and PCA
print("Loading data into memory...")
print(f"Loading training data... (this may take a minute)")

train_list = []
train_labels_list = []
for images, labels in train_loader:
    # 手动将 PIL.Image.Image 转为 tensor
    if isinstance(images, torch.Tensor):
        images_tensor = images
    else:
        images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
    train_list.append(images_tensor.numpy())
    train_labels_list.append(labels.numpy())
train_array = np.concatenate(train_list, axis=0)
train_labels_array = np.concatenate(train_labels_list, axis=0)

# Flatten images: (N, C, H, W) -> (N, C*H*W)
train_flat = train_array.reshape(train_array.shape[0], -1)
print(f"✓ Train shape: {train_flat.shape}, Labels1 shape: {train_labels_array.shape}")
print(f"Loading test data...")

test_list = []
test_labels_list = []
for images, labels in test_loader:
    if isinstance(images, torch.Tensor):
        images_tensor = images
    else:
        images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
    test_list.append(images_tensor.numpy())
    test_labels_list.append(labels.numpy())
test_array = np.concatenate(test_list, axis=0)
test_labels_array = np.concatenate(test_labels_list, axis=0)

# Flatten images: (N, C, H, W) -> (N, C*H*W)
test_flat = test_array.reshape(test_array.shape[0], -1)
print(f"✓ Test shape: {test_flat.shape}, Labels2 shape: {test_labels_array.shape}")

# Clear memory
del train_loader, test_loader, train_list, test_list
del train_array, test_array
import gc
gc.collect()
print("✓ Memory cleared, ready for dimensionality reduction...")
print(f"\n{'='*60}")
print(f"Dataset: {args.dataset_name.upper()} with {num_classes} classes")
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
        print(f"Fitting PCA on train data...")
        embedding_pca1 = pca.fit_transform(train_flat)
        print(f"Transforming test data with PCA...")
        embedding_pca2 = pca.transform(test_flat)
        
        # Save PCA features
        pca_features1_dir = os.path.join(args.save_dir, 'pca', 'train_features')
        pca_features2_dir = os.path.join(args.save_dir, 'pca', 'test_features')
        os.makedirs(pca_features1_dir, exist_ok=True)
        os.makedirs(pca_features2_dir, exist_ok=True)
        
        np.save(os.path.join(pca_features1_dir, f'{args.dataset_name}_dim{dim}.npy'), embedding_pca1)
        np.save(os.path.join(pca_features2_dir, f'{args.dataset_name}_dim{dim}.npy'), embedding_pca2)
        print(f"✅ Saved PCA features for dimension {dim}")

    # UMAP
    if args.method in ['umap', 'both']:
        print(f"\n--- UMAP ---")
        reducer = umap.UMAP(n_components=dim, n_neighbors=15, min_dist=0.1, random_state=42)
        print(f"Fitting UMAP on train data...")
        embedding_umap1 = reducer.fit_transform(train_flat)
        print(f"Transforming test data with UMAP...")
        embedding_umap2 = reducer.transform(test_flat)
        
        # Save UMAP features
        umap_features1_dir = os.path.join(args.save_dir, 'umap', 'train_features')
        umap_features2_dir = os.path.join(args.save_dir, 'umap', 'test_features')
        os.makedirs(umap_features1_dir, exist_ok=True)
        os.makedirs(umap_features2_dir, exist_ok=True)
        
        np.save(os.path.join(umap_features1_dir, f'{args.dataset_name}_dim{dim}.npy'), embedding_umap1)
        np.save(os.path.join(umap_features2_dir, f'{args.dataset_name}_dim{dim}.npy'), embedding_umap2)
        print(f"✅ Saved UMAP features for dimension {dim}")


print(f"\n{'='*60}")
print("All dimensions processed successfully!")
print(f"Results saved to: {args.save_dir}")
print(f"{'='*60}")

   
   
