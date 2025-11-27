import os
import sys
import numpy as np
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

# Add parent directory to path to import eval module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval.eval_methods as loss

# Create argument parser
parser = argparse.ArgumentParser(description='Evaluate dimensionality reduction features with SSL metrics on subsets')
parser.add_argument('--dataset', type=str, default='tinyimagenet', 
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='Dataset name')
parser.add_argument('--method', type=str, default='umap', 
                    choices=['umap', 'tsne', 'pca', 'both', 'all'],
                    help='Dimensionality reduction method to evaluate')
parser.add_argument('--base_dir', type=str, 
                    default='E:/master thesis/A_CODE/results_visual/umap/SSL_tinyimagenet',
                    help='Base directory containing test1_subsets and test2_subsets')
parser.add_argument('--output_dir', type=str, default='./result_loss',
                    help='Directory to save evaluation results')
parser.add_argument('--n_clusters', type=int, default=10,
                    help='Number of clusters for KMeans (for MCR2)')
parser.add_argument('--gam1', type=float, default=1.0,
                    help='MCR2 gamma1 parameter')
parser.add_argument('--gam2', type=float, default=1.0,
                    help='MCR2 gamma2 parameter')
parser.add_argument('--eps', type=float, default=0.5,
                    help='MCR2 epsilon parameter')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='Alignment weight parameter')
parser.add_argument('--t', type=float, default=2.0,
                    help='Uniformity temperature parameter')
parser.add_argument('--k_classifier', type=int, default=1,
                    help='Number of neighbors for KNN classifier in CLID')

args = parser.parse_args()

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
    print(f"Evaluating {method.upper()} features on subsets")
    print(f"{'='*60}")
    
    # Define directories for test1 and test2 subsets
    test1_dir = os.path.join(args.base_dir, 'test1_subsets')
    test2_dir = os.path.join(args.base_dir, 'test2_subsets')
    
    # Check if directories exist
    if not os.path.exists(test1_dir):
        print(f"Warning: {test1_dir} does not exist, skipping {method}")
        continue
    if not os.path.exists(test2_dir):
        print(f"Warning: {test2_dir} does not exist, skipping {method}")
        continue
    
    # Get all dimension directories from test1_subsets
    dim_dirs = sorted([d for d in os.listdir(test1_dir) 
                      if os.path.isdir(os.path.join(test1_dir, d)) and d.startswith('dim')])
    
    if len(dim_dirs) == 0:
        print(f"Warning: No dimension directories found in {test1_dir}, skipping {method}")
        continue
    
    print(f"Found {len(dim_dirs)} dimension directories: {dim_dirs}")
    
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
        
        # Paths for test1 and test2 for this dimension
        test1_dim_dir = os.path.join(test1_dir, dim_dir)
        test2_dim_dir = os.path.join(test2_dir, dim_dir)
        
        # Check if test2 dimension directory exists
        if not os.path.exists(test2_dim_dir):
            print(f"Warning: {test2_dim_dir} does not exist, skipping dim{dim}")
            continue
        
        # Get list of test subset feature files
        test_files = sorted([f for f in os.listdir(test1_dim_dir) 
                            if f.endswith('_features.npy')])
        
        if len(test_files) == 0:
            print(f"Warning: No feature files found in {test1_dim_dir}, skipping dim{dim}")
            continue
        
        print(f"  Found {len(test_files)} test subset files")
        
        # Create DataFrame to store results for this dimension
        df_results = pd.DataFrame()
        
        # Evaluate on each test subset
        for test_file in tqdm(test_files, desc=f"  Processing subsets", leave=False):
            # Extract subset name
            subset_name = test_file.replace('_features.npy', '')
            
            # Load features from test1 and test2
            features1_path = os.path.join(test1_dim_dir, test_file)
            features2_path = os.path.join(test2_dim_dir, test_file)
            
            if not os.path.exists(features2_path):
                print(f"    Warning: {features2_path} not found, skipping {subset_name}")
                continue
            
            # Load features
            features1 = np.load(features1_path)
            features2 = np.load(features2_path)
            
            # Convert to torch tensors
            features1_torch = torch.from_numpy(features1).float()
            features2_torch = torch.from_numpy(features2).float()
            
            # Normalize features before computing losses (L2 normalization)
            features1_norm = torch.nn.functional.normalize(features1_torch, p=2, dim=1)
            features2_norm = torch.nn.functional.normalize(features2_torch, p=2, dim=1)
            
            # Prepare labels for MCR2 using KMeans clustering on features1
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features1)
            labels_torch = torch.from_numpy(labels).long()
            
            # Compute losses with normalized features
            try:
                # MCR2
                mcr2_result = loss.MaximalCodingRateReduction(args.gam1, args.gam2, args.eps).forward(
                    x=features1_norm, y=labels_torch)
                
                # Alignment & Uniformity
                align_unif = loss.align_unif(args.alpha, args.t).forward(
                    x=features1_norm, y=features2_norm)
                
                # CLID
                clid_loss = loss.CLID(k_classifier=args.k_classifier).forward(features1_norm)
                
                # RankMe
                rankme_full_calc = loss.RankMe(eps=1e-12, use_simple=False)
                rankme_simple_calc = loss.RankMe(threshold=0.01, use_simple=True)
                rankme_full_score = rankme_full_calc(features1_norm)
                rankme_simple_score = rankme_simple_calc(features1_norm)
                
                # Extract MCR2 results
                mcr2_total_loss = mcr2_result[0].item() if isinstance(mcr2_result, tuple) and len(mcr2_result) > 0 else float('nan')
                mcr2_discrimn_empi = mcr2_result[1][0] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 1 and len(mcr2_result[1]) > 0 else float('nan')
                mcr2_compress_empi = mcr2_result[1][1] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 1 and len(mcr2_result[1]) > 1 else float('nan')
                mcr2_discrimn_theo = mcr2_result[2][0] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 2 and len(mcr2_result[2]) > 0 else float('nan')
                mcr2_compress_theo = mcr2_result[2][1] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 2 and len(mcr2_result[2]) > 1 else float('nan')
                
                # Extract Alignment & Uniformity values
                alignment_val = align_unif[1][0] if isinstance(align_unif, tuple) and len(align_unif) > 1 else float('nan')
                uniformity_val = align_unif[1][1] if isinstance(align_unif, tuple) and len(align_unif) > 1 and len(align_unif[1]) > 1 else float('nan')
                align_uniform_total_val = align_unif[0].item() if isinstance(align_unif, tuple) and hasattr(align_unif[0], 'item') else float('nan')
                
                # Extract CLID values
                cl_val = clid_loss.get('cl', float('nan')) if isinstance(clid_loss, dict) else float('nan')
                id_val = clid_loss.get('id', float('nan')) if isinstance(clid_loss, dict) else float('nan')
                clid_val = clid_loss.get('clid', float('nan')) if isinstance(clid_loss, dict) else float('nan')
                
                # Store results in DataFrame (each subset is a row)
                idx = len(df_results)
                df_results.loc[idx, 'Subset'] = subset_name
                df_results.loc[idx, 'Dimension'] = dim
                df_results.loc[idx, 'MCR2_Total_Loss'] = mcr2_total_loss
                df_results.loc[idx, 'MCR2_Discriminative_Empirical'] = mcr2_discrimn_empi
                df_results.loc[idx, 'MCR2_Compression_Empirical'] = mcr2_compress_empi
                df_results.loc[idx, 'MCR2_Discriminative_Theoretical'] = mcr2_discrimn_theo
                df_results.loc[idx, 'MCR2_Compression_Theoretical'] = mcr2_compress_theo
                df_results.loc[idx, 'Alignment'] = alignment_val
                df_results.loc[idx, 'Uniformity'] = uniformity_val
                df_results.loc[idx, 'Alignment_Uniformity_Total'] = align_uniform_total_val
                df_results.loc[idx, 'CL'] = cl_val
                df_results.loc[idx, 'ID'] = id_val
                df_results.loc[idx, 'CLID'] = clid_val
                df_results.loc[idx, 'RankMe_Full'] = rankme_full_score if isinstance(rankme_full_score, (int, float)) else rankme_full_score.item()
                df_results.loc[idx, 'RankMe_Simple'] = rankme_simple_score if isinstance(rankme_simple_score, (int, float)) else rankme_simple_score.item()
                
            except Exception as e:
                print(f"    ✗ Error processing {subset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Add row with NaN values for this subset
                idx = len(df_results)
                df_results.loc[idx, 'Subset'] = subset_name
                df_results.loc[idx, 'Dimension'] = dim
                for col in ['MCR2_Total_Loss', 'MCR2_Discriminative_Empirical', 'MCR2_Compression_Empirical',
                           'MCR2_Discriminative_Theoretical', 'MCR2_Compression_Theoretical',
                           'Alignment', 'Uniformity', 'Alignment_Uniformity_Total',
                           'CL', 'ID', 'CLID', 'RankMe_Full', 'RankMe_Simple']:
                    df_results.loc[idx, col] = float('nan')
        
        # Save DataFrame for this dimension to CSV
        if len(df_results) > 0:
            csv_path = os.path.join(args.output_dir, f"{args.dataset}_{method}_dim{dim}_ssl_results.csv")
            df_results.to_csv(csv_path, index=False)
            
            print(f"\n  ✅ Saved dim{dim} results to: {csv_path}")
            print(f"  DataFrame shape: {df_results.shape}")
            print(f"  Mean metrics across subsets:")
            print(f"    - MCR2 Total Loss: {df_results['MCR2_Total_Loss'].mean():.4f}")
            print(f"    - Alignment: {df_results['Alignment'].mean():.4f}")
            print(f"    - Uniformity: {df_results['Uniformity'].mean():.4f}")
            print(f"    - CLID: {df_results['CLID'].mean():.4f}")
            print(f"    - RankMe Full: {df_results['RankMe_Full'].mean():.4f}")
        else:
            print(f"  Warning: No results generated for dim{dim}")
    
    print(f"\n{'='*60}")
    print(f"✅ Completed evaluation for {method.upper()}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

print(f"\n{'='*60}")
print("All evaluations completed!")
print(f"Results saved to: {args.output_dir}")
print(f"{'='*60}")