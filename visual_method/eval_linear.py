import os
import sys
import numpy as np
from pydash import transform
from sklearn.linear_model import LogisticRegression
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader.tinyimagenet_dataset import get_tinyimagenet_dataset
from torch.utils.data import DataLoader


# Create argument parser
parser = argparse.ArgumentParser(description='Evaluate dimensionality reduction features with SSL metrics')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='Dataset name')
parser.add_argument('--method', type=str, default='both', 
                    choices=['umap', 'pca', 'both', 'all'],
                    help='Dimensionality reduction method to evaluate')
parser.add_argument('--train_dir', type=str, default='./results_visual',
                    help='Directory where train features are saved')
parser.add_argument('--test_dir', type=str, default='./results_visual',
                    help='Directory where test features are saved')
parser.add_argument('--output_dir', type=str, default='./result_loss',
                    help='Directory to save evaluation results')
parser.add_argument('--labels_path', type=str, required=True,
                    help='Path to .npy file containing training labels')
parser.add_argument('--batch_size', type=int, default=24,
                    help='Batch size for data loading')

args = parser.parse_args()
use_grayscale = (args.dataset == 'mnist')

if args.dataset == 'cifar10':
    train = datasets.CIFAR10(root=args.labels_path, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.CIFAR10(root=args.labels_path, train=False, download=True, transform=transforms.ToTensor())
elif args.dataset == 'mnist':      
    train = datasets.MNIST(root=args.labels_path, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root=args.labels_path, train=False, download=True, transform=transforms.ToTensor())
elif args.dataset == 'tinyimagenet':
    train_full = get_tinyimagenet_dataset(args.labels_path, split='train', transform=transforms.ToTensor())
    test_full = get_tinyimagenet_dataset(args.labels_path, split='val', transform=transforms.ToTensor()) 
    train_len = len(train_full)
    test_len = len(test_full)
    train = torch.utils.data.Subset(train_full, range(train_len // 2))
    test = torch.utils.data.Subset(test_full, range(test_len // 2))
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

# Load labels once (outside the loop)
print("Loading train labels...")
train_labels_list = []
for images, labels in tqdm(train_loader, desc="Loading train labels"):
    train_labels_list.append(labels.numpy())
train_labels_array = np.concatenate(train_labels_list, axis=0)
print(f"Train labels shape: {train_labels_array.shape}")

print("Loading test labels...")
test_labels_list = []
for images, labels in tqdm(test_loader, desc="Loading test labels"):
    test_labels_list.append(labels.numpy())
test_labels_array = np.concatenate(test_labels_list, axis=0)
print(f"Test labels shape: {test_labels_array.shape}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Determine which methods to evaluate
if args.method == 'all':
    methods = ['umap', 'pca']
elif args.method == 'both':
    methods = ['umap', 'pca']
else:
    methods = [args.method]

# Process each method
for method in methods:
    print(f"\n{'='*60}")
    print(f"Evaluating {method.upper()} features")
    print(f"{'='*60}")
    
    train_dir = os.path.join(args.train_dir)
    test_dir = os.path.join(args.test_dir)
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Warning: {train_dir} does not exist, skipping {method}")
        continue
    if not os.path.exists(test_dir):
        print(f"Warning: {test_dir} does not exist, skipping {method}")
        continue
    
    # Get list of files
    files1 = sorted([f for f in os.listdir(train_dir) if f.endswith('.npy')])
    if len(files1) == 0:
        print(f"Warning: No .npy files found in {train_dir}, skipping {method}")
        continue

    print(f"Found {len(files1)} feature files to evaluate")

    # Create DataFrame to store results
    df_results = pd.DataFrame()
    
    # Process each file
    for filename in tqdm(files1, desc=f"Processing {method}"):
        # Extract dimension from filename
        try:
            dim = int(filename.split('dim')[1].split('.')[0])
        except:
            print(f"Warning: Could not extract dimension from filename {filename}, skipping")
            continue
        
        # Load features
        train_path = os.path.join(train_dir, filename)
        test_path = os.path.join(test_dir, filename)
        
        if not os.path.exists(test_path):
            print(f"Warning: Corresponding test file not found: {test_path}, skipping")
            continue
        
        train_features = np.load(train_path)
        test_features = np.load(test_path)
        
        print(f"\nDim={dim}: Train features shape: {train_features.shape}, Test features shape: {test_features.shape}")
        print(f"Dim={dim}: Train labels shape: {train_labels_array.shape}, Test labels shape: {test_labels_array.shape}")
        
        # Convert to torch tensors and normalize
        train_torch = torch.from_numpy(train_features).float()
        test_torch = torch.from_numpy(test_features).float()
        
        train_norm = torch.nn.functional.normalize(train_torch, p=2, dim=1)
        test_norm = torch.nn.functional.normalize(test_torch, p=2, dim=1)
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_norm.numpy(), train_labels_array)
        
        # Test accuracy
        test_accuracy = clf.score(test_norm.numpy(), test_labels_array)
        print(f"  {method.upper()} dim={dim}: validation accuracy={test_accuracy:.4f}")
        
        # Store results
        df_results.loc[dim, f'{method}_accuracy'] = test_accuracy
    
    # Sort and save results
    df_results = df_results.sort_index()
    csv_path = os.path.join(args.output_dir, f"{args.dataset}_{method}_linear_results.csv")
    df_results.to_csv(csv_path)
    
    print(f"\n✅ Saved {method.upper()} results to: {csv_path}")
    print(f"DataFrame shape: {df_results.shape}")
    print(f"\nFirst few rows:")
    print(df_results.head())

print(f"\n{'='*60}")
print("All evaluations completed!")
print(f"Results saved to: {args.output_dir}")
print(f"{'='*60}")