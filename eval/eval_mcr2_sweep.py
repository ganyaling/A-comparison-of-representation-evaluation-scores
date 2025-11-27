"""
Sweep MCR2 metric across different numbers of clusters (K values)
This script computes MCR2 loss for various K values to analyze the impact of cluster count
on the Maximal Coding Rate Reduction.
"""

import argparse
from sklearn.cluster import KMeans
import torch
import eval_methods as loss
import pandas as pd
import os
import numpy as np
from datetime import datetime


def compute_mcr2_for_k(features, k, gam1, gam2, eps, device='cuda'):
    """
    Compute MCR2 loss for a given number of clusters K
    
    Args:
        features: numpy array of shape (N, D) - feature vectors
        k: int - number of clusters
        gam1: float - gamma1 parameter for MCR2
        gam2: float - gamma2 parameter for MCR2
        eps: float - epsilon parameter for MCR2
        device: str - 'cuda' or 'cpu'
    
    Returns:
        dict with MCR2 components
    """
    # KMeans clustering
    print(f"  Running KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    # Normalize features (L2 normalization)
    features_tensor_norm = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
    
    # Compute MCR2
    print(f"  Computing MCR2...")
    mcr2_calculator = loss.MaximalCodingRateReduction(gam1, gam2, eps)
    mcr2_result = mcr2_calculator.forward(x=features_tensor_norm, y=labels_tensor)
    
    # Extract results
    # mcr2_result = (total_loss_empi, [discrimn_loss_empi, compress_loss], [discrimn_loss_theo, compress_loss])
    result = {
        'k': k,
        'total_loss': mcr2_result[0].item() if isinstance(mcr2_result, tuple) else float('nan'),
        'discrimn_empirical': mcr2_result[1][0] if len(mcr2_result) > 1 else float('nan'),
        'compress': mcr2_result[1][1] if len(mcr2_result) > 1 else float('nan'),
        'discrimn_theoretical': mcr2_result[2][0] if len(mcr2_result) > 2 else float('nan'),
    }
    
    return result


def sweep_mcr2_k_values(args):
    """
    Main function to sweep MCR2 across different K values
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load features
    print(f"\nLoading features from: {args.features}")
    features = np.load(args.features)
    print(f"Loaded features shape: {features.shape}")
    n_samples = features.shape[0]
    
    # Limit samples if needed
    if args.max_samples > 0 and n_samples > args.max_samples:
        print(f"⚠️  Limiting samples from {n_samples} to {args.max_samples}")
        features = features[:args.max_samples]
        n_samples = args.max_samples
        print(f"New features shape: {features.shape}")
    
    # Determine K values to test
    if args.k_list is not None and len(args.k_list) > 0:
        k_values = sorted(args.k_list)
        print(f"\nUsing custom K values: {k_values}")
    else:
        # Default: from k_min to k_max with step k_step
        k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
        print(f"\nUsing K range: {args.k_min} to {args.k_max} (step={args.k_step})")
        print(f"K values to test: {k_values}")
    
    # Validate K values
    k_values = [k for k in k_values if 2 <= k <= n_samples]
    if len(k_values) == 0:
        raise ValueError(f"No valid K values! Must be between 2 and {n_samples}")
    
    print(f"\n{'='*70}")
    print(f"Starting MCR2 sweep for {len(k_values)} K values")
    print(f"{'='*70}\n")
    
    # Run MCR2 for each K
    all_results = []
    for idx, k in enumerate(k_values, 1):
        print(f"[{idx}/{len(k_values)}] Computing MCR2 for K={k}")
        result = compute_mcr2_for_k(
            features=features,
            k=k,
            gam1=args.gam1,
            gam2=args.gam2,
            eps=args.eps,
            device=device
        )
        
        # Add metadata
        result['dataset'] = args.dataset_name
        result['method'] = args.method
        result['checkpoint'] = args.checkpoint
        result['n_samples'] = n_samples
        result['feature_dim'] = features.shape[1]
        
        all_results.append(result)
        
        # Print current result
        print(f"  ✓ K={k}: total_loss={result['total_loss']:.4f}, "
              f"discrimn_emp={result['discrimn_empirical']:.4f}, "
              f"compress={result['compress']:.4f}")
        print()
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result_loss')
    method_dir = os.path.join(base_result_dir, args.method)
    result_dir = os.path.join(method_dir, args.dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save results
    filename = f"mcr2_k_sweep_{args.dataset_name}_{args.method}_{args.checkpoint}_{timestamp}.csv"
    csv_path = os.path.join(result_dir, filename)
    df_results.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"MCR2 K-SWEEP COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Results saved to: {csv_path}")
    print(f"\nSummary:")
    print(f"  Tested {len(k_values)} K values: {k_values}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Method: {args.method}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {n_samples}")
    print(f"  Feature dim: {features.shape[1]}")
    
    # Print top 5 K values by total loss (lower is better for MCR2 minimize objective)
    print(f"\n📊 Top 5 K values by Total Loss (lower = better):")
    top5 = df_results.nsmallest(5, 'total_loss')
    for idx, row in top5.iterrows():
        print(f"  K={int(row['k']):3d}: total_loss={row['total_loss']:.4f}, "
              f"discrimn_emp={row['discrimn_empirical']:.4f}, "
              f"compress={row['compress']:.4f}")
    
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sweep MCR2 metric across different K (number of clusters) values'
    )
    
    # Required arguments
    parser.add_argument('--features', required=True, 
                        help='Path to saved features .npy file')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset (e.g., cifar10, mnist, tinyimagenet)')
    parser.add_argument('--method', type=str, required=True,
                        help='Method name (e.g., moco, simclr, dino)')
    parser.add_argument('--checkpoint', type=str, default='unknown',
                        help='Checkpoint name (e.g., checkpoint_0200.pth.tar)')
    
    # K value configuration (two modes: range or list)
    parser.add_argument('--k-min', type=int, default=5,
                        help='Minimum K value (default: 5)')
    parser.add_argument('--k-max', type=int, default=50,
                        help='Maximum K value (default: 50)')
    parser.add_argument('--k-step', type=int, default=5,
                        help='Step size for K values (default: 5)')
    parser.add_argument('--k-list', type=int, nargs='+', default=None,
                        help='Custom list of K values to test (e.g., --k-list 2 5 10 20 50 100 200)')
    
    # MCR2 hyperparameters
    parser.add_argument('--gam1', type=float, default=1.0,
                        help='Gamma 1 for MCR2 (default: 1.0)')
    parser.add_argument('--gam2', type=float, default=1.0,
                        help='Gamma 2 for MCR2 (default: 1.0)')
    parser.add_argument('--eps', type=float, default=0.01,
                        help='Epsilon for MCR2 (default: 0.01)')
    
    # Other options
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum samples to use (default: 10000, -1 for all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run the sweep
    results_df = sweep_mcr2_k_values(args)
