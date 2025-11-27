import os
import sys
import numpy as np
import argparse

# Get dataset based on args

def create_subsets_from_saved_features(base_dir, dataset_name, n_subsets=30, subset_ratio=0.1, seed=42):
    """
    create subsets from saved full test features.
    
    Args:
        base_dir: Base directory (e.g., 'E:/master thesis/A_CODE/results_visual/umap/linear_tinyimagenet')
        dataset_name: Dataset name
        n_subsets: Number of subsets
        subset_ratio: Sampling ratio for each subset
        seed: Random seed
    """
    
    test1_features_dir = os.path.join(base_dir,f"{dataset_name}_test1")
    test2_features_dir = os.path.join(base_dir,f"{dataset_name}_test2")
    
    # Check if directories exist
    if not os.path.exists(test1_features_dir):
        raise ValueError(f"Directory not found: {test1_features_dir}")
    if not os.path.exists(test2_features_dir):
        raise ValueError(f"Directory not found: {test2_features_dir}")
    # Get all feature files for all dimensions
    feature1_files = [f for f in os.listdir(test1_features_dir) if f.endswith('.npy')]
    print(f"Found {len(feature1_files)} feature files in {test1_features_dir}")
    feature2_files = [f for f in os.listdir(test2_features_dir) if f.endswith('.npy')]
    print(f"Found {len(feature2_files)} feature files in {test2_features_dir}")

    # Extract dimension information
    dims = []
    for file in feature1_files:
        # Assume filename format: tinyimagenet_dim512.npy
        if 'dim' in file:
            dim_str = file.split('dim')[1].split('.')[0]
            dims.append(int(dim_str))
    
    dims = sorted(dims)
    print(f"Dimensions found: {dims}")
    
    # Load the first file to determine the number of samples
    first_file = os.path.join(test1_features_dir, f'{dataset_name}_dim{dims[0]}.npy')
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
    #labels_file = os.path.join(test_features_dir, f'{dataset_name}_labels.npy')
    #if os.path.exists(labels_file):
    #    test_labels = np.load(labels_file)
    #    print(f"✓ Loaded labels from {labels_file}")
    #else:
    #    print(f"⚠ Warning: Labels file not found at {labels_file}")
    #    print(f"  Labels will not be saved with subsets")
        #test_labels = None
    
    # Create subsets for each dimension
    print(f"\nProcessing dimensions...")
    for dim in dims:
        print(f"\n{'='*60}")
        print(f"Processing dimension: {dim}")
        print(f"{'='*60}")
        
        # Load full test features
        feature1_file = os.path.join(test1_features_dir, f'{dataset_name}_dim{dim}.npy')
        embedding_test1_full = np.load(feature1_file)
        print(f"Loaded features shape: {embedding_test1_full.shape}")

        feature2_file = os.path.join(test2_features_dir, f'{dataset_name}_dim{dim}.npy')
        embedding_test2_full = np.load(feature2_file)
        print(f"Loaded features shape: {embedding_test2_full.shape}")
        
        # Create subset directories
        subsets1_dir = os.path.join(base_dir, 'test1_subsets', f'dim{dim}')
        os.makedirs(subsets1_dir, exist_ok=True)
        subsets2_dir = os.path.join(base_dir, 'test2_subsets', f'dim{dim}')
        os.makedirs(subsets2_dir, exist_ok=True)
        
        # Create and save each subset
        for subset_idx, indices in enumerate(test_subset_indices):
            subset_features1 = embedding_test1_full[indices]
            subset_features2 = embedding_test2_full[indices]
            #subset_labels = test_labels_array[indices]
           
            # Save features
            np.save(os.path.join(subsets1_dir, 
                                f'{dataset_name}_subset{subset_idx:02d}_features.npy'), 
                   subset_features1)
            np.save(os.path.join(subsets2_dir, 
                                f'{dataset_name}_subset{subset_idx:02d}_features.npy'), 
                   subset_features2)
            
            # Save labels
            #np.save(os.path.join(subsets_dir, 
                               # f'{args.dataset_name}_subset{subset_idx:02d}_labels.npy'), 
                   #subset_labels)
            
            if (subset_idx + 1) % 10 == 0:
                print(f"  Created {subset_idx + 1}/{n_subsets} subsets...")
        
        print(f"✓ Saved {n_subsets} subsets for dimension {dim}")
    
    print(f"\n{'='*60}")
    print("All subsets created successfully!")
   

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
    args = parser.parse_args()
    


    create_subsets_from_saved_features(
        base_dir=args.base_dir,
        dataset_name=args.dataset_name,
        n_subsets=args.n_subsets,
        subset_ratio=args.subset_ratio,
        seed=args.seed
    )