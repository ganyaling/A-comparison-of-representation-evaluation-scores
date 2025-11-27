import os
import sys
from matplotlib import transforms
import numpy as np
from torchvision import datasets, transforms
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add parent directory to path to import eval module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader.tinyimagenet_dataset import get_tinyimagenet_dataset
from torch.utils.data import DataLoader




# Create argument parser
parser = argparse.ArgumentParser(description='Evaluate dimensionality reduction features with SSL metrics')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='Dataset name')
parser.add_argument('--method', type=str, default='both', 
                    choices=['umap','pca', 'both', 'all'],
                    help='Dimensionality reduction method to evaluate')
parser.add_argument('--output_dir', type=str, default='./result_loss',
                    help='Directory to save evaluation results')
parser.add_argument('--train_features_dir', type=str, 
                    default=None,
                    help='Directory containing training features (default: base_dir/train_features)')
parser.add_argument('--test_subsets_dir', type=str, 
                    default=None,
                    help='Directory containing test subsets (default: base_dir/test_subsets)')
parser.add_argument("--batch_size", type=int, default=24,
                    help="Batch size for data loading")
parser.add_argument('--base_dir', type=str, default='./results_visual',
                    help='Base directory for features')
parser.add_argument('--labels_path', type=str, required=True,
                    help='Path to .npy file containing training labels')

args = parser.parse_args()


use_grayscale = (args.dataset == 'mnist')

if args.dataset == 'cifar10':
    train = datasets.CIFAR10(root=args.labels_path, train=True, download=True, transform=transforms.ToTensor())

elif args.dataset == 'mnist':      
    train = datasets.MNIST(root=args.labels_path, train=True, download=True, transform=transforms.ToTensor())
   
elif args.dataset == 'tinyimagenet':
    train_full = get_tinyimagenet_dataset(args.labels_path, split='train', transform=transforms.ToTensor())
    train_len = len(train_full)
    train = torch.utils.data.Subset(train_full, range(train_len // 2))
  
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
# Load labels once (outside the loop)
print("Loading train labels...")
train_labels_list = []
for images, labels in tqdm(train_loader, desc="Loading train labels"):
    train_labels_list.append(labels.numpy())
train_labels_array = np.concatenate(train_labels_list, axis=0)
print(f"Train labels shape: {train_labels_array.shape}")


# Set default paths if not provided
if args.train_features_dir is None:
    args.train_features_dir = os.path.join(args.base_dir, 'train_features')
if args.test_subsets_dir is None:
    args.test_subsets_dir = os.path.join(args.base_dir, 'test_subsets')

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Determine which methods to evaluate
if args.method == 'all':
    methods = ['umap', 'tsne', 'pca']
elif args.method == 'both':
    methods = ['umap', 'tsne']
else:
    methods = [args.method]

# Process each method
for method in methods:
    print(f"\n{'='*60}")
    print(f"Evaluating {method.upper()} features")
    print(f"{'='*60}")

    # Get list of dimensions from train_features directory
    train_features_dir = args.train_features_dir
    test_subsets_dir = args.test_subsets_dir
    
    if not os.path.exists(train_features_dir):
        print(f"Warning: {train_features_dir} does not exist, skipping {method}")
        continue
    if not os.path.exists(test_subsets_dir):
        print(f"Warning: {test_subsets_dir} does not exist, skipping {method}")
        continue
    
    # Get all training feature files
    train_files = sorted([f for f in os.listdir(train_features_dir) if f.endswith('.npy') and '_dim' in f])   
    print(f"Found {len(train_files)} training feature files")
    
    # Get all dimension directories from test_subsets
    dim_dirs = sorted([d for d in os.listdir(test_subsets_dir) if os.path.isdir(os.path.join(test_subsets_dir, d)) and d.startswith('dim')])
    print(f"Found {len(dim_dirs)} dimension directories in test_subsets: {dim_dirs}")
    
    # Process each dimension
    for dim_dir in tqdm(dim_dirs, desc=f"Processing {method} dimensions"):
        # Extract dimension number
        try:
            dim = int(dim_dir.replace('dim', ''))
        except:
            print(f"Warning: Could not extract dimension from {dim_dir}, skipping")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing Dimension: {dim}")
        print(f"{'='*50}")
        
        # Path for training features in train_features directory
        train_features_path = os.path.join(train_features_dir, f'{args.dataset}_dim{dim}.npy')
        test_dim_dir = os.path.join(test_subsets_dir, dim_dir)
        
        # Check if training file exists
        if not os.path.exists(train_features_path):
            print(f"Warning: Training features not found at {train_features_path}, skipping")
            continue
        
        # Load training features
        train_features = np.load(train_features_path)
        print(f"  ✓ Loaded train features: {train_features.shape}")
        
        
        # Train Logistic Regression
        print(f"  Training Logistic Regression...")
        lr_model = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            multi_class='multinomial'
        )
        lr_model.fit(train_features, train_labels_array)
        print(f"  ✓ Training completed")
        
        # Get list of test subsets
        test_files = sorted([f for f in os.listdir(test_dim_dir) if f.endswith('_features.npy')])
        print(f"  Found {len(test_files)} test subset files")
        
        # Store accuracies for this dimension
        test_accuracies = []
        
        # Evaluate on each test subset
        for test_file in test_files:
            # Extract subset name
            subset_name = test_file.replace('_features.npy', '')
            
            # Load test features and labels
            test_features_path = os.path.join(test_dim_dir, test_file)
            test_labels_file = test_file.replace('_features.npy', '_labels.npy')
            test_labels_path = os.path.join(test_dim_dir, test_labels_file)
            
            if not os.path.exists(test_labels_path):
                print(f"    Warning: Labels not found for {test_file}, skipping")
                continue
            
            test_features = np.load(test_features_path)
            test_labels = np.load(test_labels_path)
            
            # Predict
            predictions = lr_model.predict(test_features)
            accuracy = accuracy_score(test_labels, predictions)
            
            test_accuracies.append({
                'Subset': subset_name,
                'Accuracy': accuracy
            })
            
            print(f"    {subset_name}: Accuracy = {accuracy:.4f}")
    
     # Create DataFrame for this dimension
        if len(test_accuracies) > 0:
            df_dim_results = pd.DataFrame(test_accuracies)
            
            # Add dimension column
            df_dim_results.insert(0, 'Dimension', dim)
            
            # Save to CSV
            csv_path = os.path.join(args.output_dir, f"{args.dataset}_{method}_dim{dim}_lr_accuracies.csv")
            df_dim_results.to_csv(csv_path, index=False)
            print(f"\n  ✅ Saved dim{dim} results to: {csv_path}")
            print(f"  Mean Accuracy: {df_dim_results['Accuracy'].mean():.4f}")
            print(f"  Std Accuracy: {df_dim_results['Accuracy'].std():.4f}")     

print(f"\n{'='*60}")
print("All linear accuracy completed!")
print(f"Results saved to: {args.output_dir}")
print(f"{'='*60}")