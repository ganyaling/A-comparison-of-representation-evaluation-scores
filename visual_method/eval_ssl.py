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
parser = argparse.ArgumentParser(description='Evaluate dimensionality reduction features with SSL metrics')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='Dataset name')
parser.add_argument('--method', type=str, default='both', 
                    choices=['umap', 'tsne', 'pca', 'both', 'all'],
                    help='Dimensionality reduction method to evaluate')
parser.add_argument('--test1_dir', type=str, default='./results_visual',
                    help='Directory where features1 (original images) are saved')
parser.add_argument('--test2_dir', type=str, default='./results_visual',
                    help='Directory where features2 (augmented images) are saved')
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
    print(f"Evaluating {method.upper()} features")
    print(f"{'='*60}")
    
    # Define directories for features1 and features2
    test1_dir = os.path.join(args.test1_dir)
    test2_dir = os.path.join(args.test2_dir)
    
    # Check if directories exist
    if not os.path.exists(test1_dir):
        print(f"Warning: {test1_dir} does not exist, skipping {method}")
        continue
    if not os.path.exists(test2_dir):
        print(f"Warning: {test2_dir} does not exist, skipping {method}")
        continue
    
    # Get list of files in features1 directory
    files1 = sorted([f for f in os.listdir(test1_dir) if f.endswith('.npy')])
    if len(files1) == 0:
        print(f"Warning: No .npy files found in {test1_dir}, skipping {method}")
        continue

    print(f"Found {len(files1)} feature files to evaluate")

    # Create DataFrame to store results
    # Rows = dimensions, Columns = metric values
    df_results = pd.DataFrame()
    
    # Process each file
    for filename in tqdm(files1, desc=f"Processing {method}"):
        # Extract dimension from filename (e.g., cifar10_dim50.npy -> 50)
        try:
            dim = int(filename.split('dim')[1].split('.')[0])
        except:
            print(f"Warning: Could not extract dimension from filename {filename}, skipping")
            continue
        
        # Load features from both directories
        test1_path = os.path.join(test1_dir, filename)
        test2_path = os.path.join(test2_dir, filename)
        
        if not os.path.exists(test2_path):
            print(f"Warning: Corresponding features2 file not found: {test2_path}, skipping")
            continue
        
        # Load features
        features1 = np.load(test1_path)
        print(f"  ✓ features1 shape: {features1.shape}, dtype: {features1.dtype}")
        features2 = np.load(test2_path)
        print(f"  ✓ features2 shape: {features2.shape}, dtype: {features2.dtype}")
        
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
            print(f"  Computing metrics for dimension {dim}...")
            
            # MCR2
            print(f"    - Computing MCR2...")
            mcr2_result = loss.MaximalCodingRateReduction(args.gam1, args.gam2, args.eps).forward(
                x=features1_norm, y=labels_torch)
            
            # Alignment & Uniformity
            print(f"    - Computing Alignment & Uniformity...")
            align_unif = loss.align_unif(args.alpha, args.t).forward(
                x=features1_norm, y=features2_norm)
            
            # CLID
            print(f"    - Computing CLID...")
            clid_loss = loss.CLID(k_classifier=args.k_classifier).forward(features1_norm)
            
            # RankMe
            print(f"    - Computing RankMe...")
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
            
            # Store results in DataFrame (add a new row)
            df_results.loc[dim, 'Dimension'] = dim
            df_results.loc[dim, 'MCR2_Total_Loss'] = mcr2_total_loss
            df_results.loc[dim, 'MCR2_Discriminative_Empirical'] = mcr2_discrimn_empi
            df_results.loc[dim, 'MCR2_Compression_Empirical'] = mcr2_compress_empi
            df_results.loc[dim, 'MCR2_Discriminative_Theoretical'] = mcr2_discrimn_theo
            df_results.loc[dim, 'MCR2_Compression_Theoretical'] = mcr2_compress_theo
            df_results.loc[dim, 'Alignment'] = alignment_val
            df_results.loc[dim, 'Uniformity'] = uniformity_val
            df_results.loc[dim, 'Alignment_Uniformity_Total'] = align_uniform_total_val
            df_results.loc[dim, 'CL'] = cl_val
            df_results.loc[dim, 'ID'] = id_val
            df_results.loc[dim, 'CLID'] = clid_val
            # RankMe scores are already float values, no need to call .item()
            df_results.loc[dim, 'RankMe_Full'] = rankme_full_score if isinstance(rankme_full_score, (int, float)) else rankme_full_score.item()
            df_results.loc[dim, 'RankMe_Simple'] = rankme_simple_score if isinstance(rankme_simple_score, (int, float)) else rankme_simple_score.item()
            
        except Exception as e:
            print(f"❌ Error processing dimension {dim}: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
            # Add row with NaN values for this dimension
            df_results.loc[dim, 'Dimension'] = dim
            for col in ['MCR2_Total_Loss', 'MCR2_Discriminative_Empirical', 'MCR2_Compression_Empirical',
                       'MCR2_Discriminative_Theoretical', 'MCR2_Compression_Theoretical',
                       'Alignment', 'Uniformity', 'Alignment_Uniformity_Total',
                       'CL', 'ID', 'CLID', 'RankMe_Full', 'RankMe_Simple']:
                df_results.loc[dim, col] = float('nan')
    
    # Sort DataFrame by dimension
    df_results = df_results.sort_index()
    
    # Save DataFrame to CSV
    csv_path = os.path.join(args.output_dir, f"{args.dataset}_{method}_ssl_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    print(f"\n✅ Saved {method.upper()} results to: {csv_path}")
    print(f"DataFrame shape: {df_results.shape} (rows={len(df_results)}, columns={len(df_results.columns)})")
    print(f"\nFirst few rows:")
    print(df_results.head())

print(f"\n{'='*60}")
print("All evaluations completed!")
print(f"Results saved to: {args.output_dir}")
print(f"{'='*60}")
