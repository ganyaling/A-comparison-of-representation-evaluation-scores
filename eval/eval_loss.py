
import argparse
from sklearn.cluster import KMeans
import torch
import eval_methods as loss
import pandas as pd
import os
import numpy as np
from datetime import datetime


def gen_loss(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # compute losses for each feature set and append to csv
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(args.features)
    
    # Convert to tensors for loss computation
    features_tensor = torch.tensor(args.features, dtype=torch.float32).to(device)
    features2_tensor = torch.tensor(args.features2, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    # Normalize features before computing losses
    # L2 normalization for each sample
    features_tensor_norm = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
    features2_tensor_norm = torch.nn.functional.normalize(features2_tensor, p=2, dim=1)
    

    # Compute losses with normalized features
    #mcr2
    mcr2_result = loss.MaximalCodingRateReduction(args.gam1, args.gam2, args.eps).forward(x=features_tensor_norm, y=labels_tensor)
    align_unif = loss.align_unif(args.alpha, args.t).forward(x=features_tensor_norm, y=features2_tensor_norm)
    
    # CLID: n_clusters is now auto-computed as sqrt(N), k_classifier from args (paper default k=1)
    clid_loss = loss.CLID(k_classifier=args.k_classifier).forward(features_tensor_norm)
    
    # 计算两种 RankMe - 正确的调用方式
    rankme_full_calc = loss.RankMe(eps=1e-12, use_simple=False)
    rankme_simple_calc = loss.RankMe(threshold=0.01, use_simple=True)
    
    rankme_full_score = rankme_full_calc(features_tensor_norm)
    rankme_simple_score = rankme_simple_calc(features_tensor_norm)

    # Extract MCR2 results
    # mcr2_result = (total_loss_empi, [discrimn_loss_empi, compress_loss], [discrimn_loss_theo, compress_loss])
    # Note: compress_loss is the same for empirical and theoretical versions
    mcr2_total_loss = mcr2_result[0].item() if isinstance(mcr2_result, tuple) and len(mcr2_result) > 0 else float('nan')
    mcr2_discrimn_empi = mcr2_result[1][0] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 1 and len(mcr2_result[1]) > 0 else float('nan')
    mcr2_compress = mcr2_result[1][1] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 1 and len(mcr2_result[1]) > 1 else float('nan')
    mcr2_discrimn_theo = mcr2_result[2][0] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 2 and len(mcr2_result[2]) > 0 else float('nan')

    # Extract other loss values
    aligment_val = align_unif[1][0] if isinstance(align_unif, tuple) and len(align_unif) > 1 else float('nan')
    uniformity_val = align_unif[1][1] if isinstance(align_unif, tuple) and len(align_unif) > 1 and len(align_unif[1]) > 1 else float('nan')
    align_uniform_total_val = align_unif[0].item() if isinstance(align_unif, tuple) and hasattr(align_unif[0], 'item') else float('nan')
    cl_val = clid_loss.get('cl', float('nan')) if isinstance(clid_loss, dict) else float('nan')
    id_val = clid_loss.get('id', float('nan')) if isinstance(clid_loss, dict) else float('nan')
    clid_val = clid_loss.get('clid', float('nan')) if isinstance(clid_loss, dict) else float('nan')

    # Create results in vertical format
    results = []
    results.append({'metric': 'mcr2_total_loss', 'value': mcr2_total_loss})
    results.append({'metric': 'mcr2_discrimn_empirical', 'value': mcr2_discrimn_empi})
    results.append({'metric': 'mcr2_discrimn_theoretical', 'value': mcr2_discrimn_theo})
    results.append({'metric': 'mcr2_compress', 'value': mcr2_compress})
    results.append({'metric': 'alignment', 'value': aligment_val})
    results.append({'metric': 'uniformity', 'value': uniformity_val})
    results.append({'metric': 'align_uniform_total', 'value': align_uniform_total_val})
    results.append({'metric': 'cl', 'value': cl_val})
    results.append({'metric': 'id', 'value': id_val})
    results.append({'metric': 'clid', 'value': clid_val})
    results.append({'metric': 'rankme_full', 'value': rankme_full_score})
    results.append({'metric': 'rankme_simple', 'value': rankme_simple_score})

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add metadata to each result
    for result in results:
        result['dataset'] = args.dataset_name
        result['method'] = args.method
        result['timestamp'] = timestamp
        
    # Convert to DataFrame in vertical format
    df_results = pd.DataFrame(results)

    # Create result_loss directory structure: result_loss/{method}/{dataset}/
    base_result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result_loss')
    method_dir = os.path.join(base_result_dir, args.method)
    result_dir = os.path.join(method_dir, args.dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Generate filename based on dataset and method
    filename = f"{args.dataset_name}_{args.method}_{timestamp}.csv"
    
    csv_path = os.path.join(result_dir, filename)
    
    # Always create a new file with timestamp (no appending)
    df_results.to_csv(csv_path, mode='w', header=True, index=False)
    print(f'✅ Saved results to new file: {csv_path}')
    
    # Also create/update a summary file that contains all runs
    summary_filename = f"all_runs.csv"
    summary_path = os.path.join(result_dir, summary_filename)
    
    # Save to summary file (append mode) - use same df_results
    if os.path.exists(summary_path):
        df_results.to_csv(summary_path, mode='a', header=False, index=False)
        print(f'📊 Updated summary file: {summary_path}')
    else:
        df_results.to_csv(summary_path, mode='w', header=True, index=False)
        print(f'📊 Created summary file: {summary_path}')
    
    # Print results in the requested format
    print(f"\n🔬 Results for {args.dataset_name} using {args.method} (with normalized features)")
    print(f"⏰ Run timestamp: {timestamp}")
    print("=" * 70)
    for result in results:
        print(f"{result['metric']}: {result['value']}")
    
    print(f"\n📁 Files saved:")
    print(f"   Individual run: {csv_path}")
    print(f"   All runs summary: {summary_path}")


def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)) + '\n')
    return csv_path

parser = argparse.ArgumentParser(description='Generating files')
parser.add_argument('--features', required=True, help='Path to saved features file or feature array.')
parser.add_argument('--features2', required=True, help='Path to saved features2 file or feature array.')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (e.g., cifar10, mnist).')
parser.add_argument('--method', type=str, required=True, help='Method name (e.g., moco, dino, umap, tsne).')
parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for K-means (MCR2 only; CLID auto-computes as sqrt(N)).')
parser.add_argument('--k_classifier', type=int, default=1, help='Number of neighbors for KNN classifier in CLID (paper default: 1).')
parser.add_argument('--gam1', type=float, default=1.0, help='Gamma 1 for MCR2.')
parser.add_argument('--gam2', type=float, default=1.0, help='Gamma 2 for MCR2.')
parser.add_argument('--eps', type=float, default=0.01, help='Epsilon for MCR2.')
parser.add_argument('--alpha', type=float, default=2.0, help='Alpha for align-uniform.')
parser.add_argument('--t', type=float, default=2.0, help='T for align-uniform.')
parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use for loss computation to avoid memory issues. Set to -1 to use all samples.')
parser.add_argument('--aug_plus', action='store_true', help='Use MoCo v2 augmentation (stronger augmentation for MoCo).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--csv_path', default='./results_loss.csv', help='Path to the CSV file for losses.')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Load features from file paths
    print(f"Loading features from: {args.features}")
    features = np.load(args.features)
    print(f"Loaded features shape: {features.shape}")
    
    print(f"Loading features2 from: {args.features2}")
    features2 = np.load(args.features2)
    print(f"Loaded features2 shape: {features2.shape}")
    
    # Limit samples to max_samples to avoid memory issues (unless max_samples = -1)
    if args.max_samples > 0 and features.shape[0] > args.max_samples:
        print(f"⚠️  Limiting samples from {features.shape[0]} to {args.max_samples} to avoid memory issues")
        features = features[:args.max_samples]
        features2 = features2[:args.max_samples]
        print(f"New features shape: {features.shape}")
    elif args.max_samples == -1:
        print(f"✅ Using all {features.shape[0]} samples (memory-optimized MCR2)")
    
    # Replace the file paths with actual numpy arrays
    args.features = features
    args.features2 = features2
    
    # Call the loss generation function
    gen_loss(args)