import os
import sys
import numpy as np
import argparse

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader.tinyimagenet_dataset import get_tinyimagenet_dataset


def create_subsets_from_saved_features(base_dir, dataset_name, n_subsets=30, subset_ratio=0.1, seed=42):
    """
    Create subsets from saved full test features
    
    Args:
        base_dir: Base directory (e.g., 'E:/master thesis/A_CODE/results_visual/umap/linear_tinyimagenet')
        dataset_name: dataset name
        n_subsets: number of subsets
        subset_ratio: sampling ratio for each subset
        seed: random seed
    """
    
    test_features_dir = os.path.join(base_dir, 'test_features')
    
    # Check if directory exists
    if not os.path.exists(test_features_dir):
        raise ValueError(f"Directory not found: {test_features_dir}")
    
    # Get all feature files for all dimensions
    feature_files = [f for f in os.listdir(test_features_dir) if f.endswith('.npy')]
    print(f"Found {len(feature_files)} feature files in {test_features_dir}")
    
    # Extract dimension information
    dims = []
    for file in feature_files:
        # filename format: tinyimagenet_dim512.npy
        if 'dim' in file:
            dim_str = file.split('dim')[1].split('.')[0]
            dims.append(int(dim_str))
    
    dims = sorted(dims)
    print(f"Dimensions found: {dims}")
    
    # Load the first file to determine the number of samples
    first_file = os.path.join(test_features_dir, f'{dataset_name}_dim{dims[0]}.npy')
    first_features = np.load(first_file)
    n_test_samples = len(first_features)
    subset_size = int(n_test_samples * subset_ratio)
    
    print(f"\nTest samples: {n_test_samples}")
    print(f"Subset size: {subset_size} ({subset_ratio*100}%)")
    print(f"Number of subsets: {n_subsets}")
    
    # Generate subset indices (same indices for all dimensions)
    print(f"\nGenerating {n_subsets} subset indices...")
    test_subset_indices = []
    for i in range(n_subsets):
        np.random.seed(seed + i)
        indices = np.random.choice(n_test_samples, size=subset_size, replace=False)
        test_subset_indices.append(indices)
    
    # Save subset indices
    indices_dir = os.path.join(base_dir, 'subset_indices')
    os.makedirs(indices_dir, exist_ok=True)
    np.save(os.path.join(indices_dir, f'{dataset_name}_subset_indices.npy'), 
            np.array(test_subset_indices))
    print(f"✓ Saved subset indices to {indices_dir}")
    
    # Load labels file (assumed to be in the same directory)
    labels_file = os.path.join(test_features_dir, f'{dataset_name}_labels.npy')
    if os.path.exists(labels_file):
        test_labels = np.load(labels_file)
        print(f"✓ Loaded labels from {labels_file}")
    else:
        print(f"⚠ Warning: Labels file not found at {labels_file}")
        print(f"  Labels will not be saved with subsets")
        test_labels = None
    
    # Create subsets for each dimension
    print(f"\nProcessing dimensions...")
    for dim in dims:
        print(f"\n{'='*60}")
        print(f"Processing dimension: {dim}")
        print(f"{'='*60}")
        
        # Load full test features
        feature_file = os.path.join(test_features_dir, f'{dataset_name}_dim{dim}.npy')
        embedding_test_full = np.load(feature_file)
        print(f"Loaded features shape: {embedding_test_full.shape}")
        
        # Create subset directories
        subsets_dir = os.path.join(base_dir, 'test_subsets', f'dim{dim}')
        os.makedirs(subsets_dir, exist_ok=True)
        
        # Create and save each subset
        for subset_idx, indices in enumerate(test_subset_indices):
            subset_features = embedding_test_full[indices]
            subset_labels = test_labels_array[indices]
           
            # Save features
            np.save(os.path.join(subsets_dir, 
                                f'{dataset_name}_subset{subset_idx:02d}_features.npy'), 
                   subset_features)
            
            # Save labels
            np.save(os.path.join(subsets_dir, 
                                f'{args.dataset_name}_subset{subset_idx:02d}_labels.npy'), 
                   subset_labels)
            
            if (subset_idx + 1) % 10 == 0:
                print(f"  Created {subset_idx + 1}/{n_subsets} subsets...")
        
        print(f"✓ Saved {n_subsets} subsets for dimension {dim}")
    
    print(f"\n{'='*60}")
    print("All subsets created successfully!")
    print(f"{'='*60}")
    print(f"\nDirectory structure:")
    print(f"{base_dir}/")
    print(f"  ├── test_features/")
    print(f"  │   ├── {dataset_name}_dim{dims[0]}.npy")
    print(f"  │   └── ...")
    print(f"  ├── subset_indices/")
    print(f"  │   └── {dataset_name}_subset_indices.npy")
    print(f"  └── test_subsets/")
    for dim in dims[:3]:  # only show first 3 dimensions for brevity
        print(f"      ├── dim{dim}/")
        print(f"      │   ├── {dataset_name}_subset00_features.npy")
        print(f"      │   ├── {dataset_name}_subset00_labels.npy")
        print(f"      │   └── ... ({n_subsets} subsets)")
    if len(dims) > 3:
        print(f"      └── ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create test subsets from saved features')
    parser.add_argument('--base_dir', type=str, 
                        default=r'E:\master thesis\A_CODE\results_visual\umap\linear_tinyimagenet',
                        help='Base directory containing test_features folder')
    parser.add_argument('--dataset_name', type=str, default='tinyimagenet',
                        help='Dataset name')
    parser.add_argument('--n_subsets', type=int, default=30,
                        help='Number of subsets to create')
    parser.add_argument('--subset_ratio', type=float, default=0.3,
                        help='Ratio of test data for each subset (0.3 = 30%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_path', type=str, default='./data',
                    help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=24,
                    help='Batch size for data loading')
    args = parser.parse_args()
    
    use_grayscale = (args.dataset_name == 'mnist')
    
    if args.dataset_name == 'cifar10':
       test = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())    
    elif args.dataset_name == 'mnist':
       test = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transforms.ToTensor())
    elif args.dataset_name == 'tinyimagenet':
       test_full = get_tinyimagenet_dataset(args.data_path, split='val', transform=transforms.ToTensor())
      # Only use the first 50% of the data
       test_len = len(test_full)
       test = torch.utils.data.Subset(test_full, range(test_len // 2))
    else:
       raise ValueError(f"Unknown dataset: {args.dataset_name}")

    #train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

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


    print(f"✓ Test Labels shape: {test_labels_array.shape}")

    create_subsets_from_saved_features(
        base_dir=args.base_dir,
        dataset_name=args.dataset_name,
        n_subsets=args.n_subsets,
        subset_ratio=args.subset_ratio,
        seed=args.seed
    )