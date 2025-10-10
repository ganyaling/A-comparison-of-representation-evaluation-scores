import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime


import torch
from torch.utils.data import DataLoader
import eval_methods as loss
from model import DINO_model, MoCo_model
from sklearn.cluster import KMeans
from torchvision import transforms
from torchvision import datasets

def transform(use_grayscale3=False, crop_size=32):
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []
    return transforms.Compose( base + [
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def gen_loss(args):

    # ============ preparing data ... ============
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Handle different dataset types with appropriate crop sizes
    if args.dataset_name.lower() == 'cifar10':
        # CIFAR-10 is 32x32, keep it at original size
        crop_size = 32
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform(args.use_grayscale3, crop_size))
        dataset2 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform(args.use_grayscale3, crop_size))
    elif args.dataset_name.lower() == 'mnist':
        # MNIST is 28x28, keep it at original size
        crop_size = 28
        dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform(args.use_grayscale3, crop_size))
        dataset2 = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform(args.use_grayscale3, crop_size))
    else:
        # For TinyImageNet (64x64) or other datasets
        crop_size = 64
        dataset = datasets.ImageFolder(args.data_path, transform=transform(args.use_grayscale3, crop_size))
        dataset2 = datasets.ImageFolder(args.data_path, transform=transform(args.use_grayscale3, crop_size))
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train_loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)

    print(f"Data loaded: there are {len(dataset)} images.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.method in ('dino', 'moco','umap','tsne'):
        # build model wrapper
        if args.method == 'dino':
            model = DINO_model(args.checkpoint_path, device=device)
            model.build_model()
            
            # Extract features for both augmentations (limited by max_samples)
            print(f"Extracting features for augmentation 1 (max {args.max_samples} samples)...")
            features_list = []
            samples_processed = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                if samples_processed >= args.max_samples:
                    break
                data = data.to(device)
                with torch.no_grad():
                    batch_features = model.model(data)
                features_list.append(batch_features.cpu().numpy())
                samples_processed += data.size(0)
                if batch_idx % 10 == 0:
                    print(f"Processed {samples_processed} samples...")
            features = np.concatenate(features_list, axis=0)[:args.max_samples]  # Ensure exact limit
            
            print(f"Extracting features for augmentation 2 (max {args.max_samples} samples)...")
            features2_list = []
            samples_processed = 0
            for batch_idx, (data, _) in enumerate(train_loader2):
                if samples_processed >= args.max_samples:
                    break
                data = data.to(device)
                with torch.no_grad():
                    batch_features = model.model(data)
                features2_list.append(batch_features.cpu().numpy())
                samples_processed += data.size(0)
                if batch_idx % 10 == 0:
                    print(f"Processed {samples_processed} samples...")
            features2 = np.concatenate(features2_list, axis=0)[:args.max_samples]  # Ensure exact limit
            
        elif args.method == 'moco':
            model = MoCo_model(args.checkpoint_path, device=device)
            model.build_model()
            
            # Extract features for both augmentations (limited by max_samples)
            print(f"Extracting features for augmentation 1 (max {args.max_samples} samples)...")
            features_list = []
            samples_processed = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                if samples_processed >= args.max_samples:
                    break
                data = data.to(device)
                with torch.no_grad():
                    batch_features = model.model(data)
                features_list.append(batch_features.cpu().numpy())
                samples_processed += data.size(0)
                if batch_idx % 10 == 0:
                    print(f"Processed {samples_processed} samples...")
            features = np.concatenate(features_list, axis=0)[:args.max_samples]  # Ensure exact limit
            
            print(f"Extracting features for augmentation 2 (max {args.max_samples} samples)...")
            features2_list = []
            samples_processed = 0
            for batch_idx, (data, _) in enumerate(train_loader2):
                if samples_processed >= args.max_samples:
                    break
                data = data.to(device)
                with torch.no_grad():
                    batch_features = model.model(data)
                features2_list.append(batch_features.cpu().numpy())
                samples_processed += data.size(0)
                if batch_idx % 10 == 0:
                    print(f"Processed {samples_processed} samples...")
            features2 = np.concatenate(features2_list, axis=0)[:args.max_samples]  # Ensure exact limit
            
        elif args.method == 'umap':
            features = np.load(args.umapnumpy_path, mmap_mode=None)
            features2 = np.load(args.umapnumpy_path2, mmap_mode=None)
            # Limit samples for memory efficiency
            features = features[:args.max_samples]
            features2 = features2[:args.max_samples]
        elif args.method == 'tsne':
            features = np.load(args.tsnenumpy_path, mmap_mode=None)
            features2 = np.load(args.tsnenumpy_path2, mmap_mode=None)
            # Limit samples for memory efficiency
            features = features[:args.max_samples]
            features2 = features2[:args.max_samples]
        else:       
            raise NotImplementedError(f"Method {args.method} not implemented.")      
    
    print(f"Using {features.shape[0]} samples for loss computation (features shape: {features.shape})")
    print(f"Features2 shape: {features2.shape}")
    
    # compute losses for each feature set and append to csv
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Convert to tensors for loss computation
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    features2_tensor = torch.tensor(features2, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    
    # Normalize features before computing losses
    # L2 normalization for each sample
    features_tensor_norm = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
    features2_tensor_norm = torch.nn.functional.normalize(features2_tensor, p=2, dim=1)
    

    # Compute losses with normalized features
    mcr2_result = loss.MaximalCodingRateReduction(args.gam1, args.gam2, args.eps).forward(x=features_tensor_norm, y=labels_tensor)
    align_unif = loss.align_unif(args.alpha, args.t).forward(x=features_tensor_norm, y=features2_tensor_norm)
    clid_loss = loss.CLID(args.n_clusters, args.k_classifier).forward(features_tensor_norm)
    
    # 计算两种 RankMe - 正确的调用方式
    rankme_full_calc = loss.RankMe(eps=1e-12, use_simple=False)
    rankme_simple_calc = loss.RankMe(threshold=0.01, use_simple=True)
    
    rankme_full_score = rankme_full_calc(features_tensor_norm)
    rankme_simple_score = rankme_simple_calc(features_tensor_norm)

    # Extract MCR2 results - save all components
    # mcr2_result = (total_loss_empi, [discrimn_loss_empi, compress_loss_empi], [discrimn_loss_theo, compress_loss_theo])
    mcr2_total_loss = mcr2_result[0].item() if isinstance(mcr2_result, tuple) and len(mcr2_result) > 0 else float('nan')
    mcr2_discrimn_empi = mcr2_result[1][0] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 1 and len(mcr2_result[1]) > 0 else float('nan')
    mcr2_compress_empi = mcr2_result[1][1] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 1 and len(mcr2_result[1]) > 1 else float('nan')
    mcr2_discrimn_theo = mcr2_result[2][0] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 2 and len(mcr2_result[2]) > 0 else float('nan')
    mcr2_compress_theo = mcr2_result[2][1] if isinstance(mcr2_result, tuple) and len(mcr2_result) > 2 and len(mcr2_result[2]) > 1 else float('nan')

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
    results.append({'metric': 'mcr2_compress_empirical', 'value': mcr2_compress_empi})
    results.append({'metric': 'mcr2_discrimn_theoretical', 'value': mcr2_discrimn_theo})
    results.append({'metric': 'mcr2_compress_theoretical', 'value': mcr2_compress_theo})
    results.append({'metric': 'alignment', 'value': aligment_val})
    results.append({'metric': 'uniformity', 'value': uniformity_val})
    results.append({'metric': 'align_uniform_total', 'value': align_uniform_total_val})
    results.append({'metric': 'cl', 'value': cl_val})
    results.append({'metric': 'id', 'value': id_val})
    results.append({'metric': 'clid', 'value': clid_val})
    results.append({'metric': 'rankme_full', 'value': rankme_full_score})
    results.append({'metric': 'rankme_simple', 'value': rankme_simple_score})

    # Add metadata to each result
    for result in results:
        result['dataset'] = args.dataset_name
        result['method'] = args.method
        
    # Convert to DataFrame in vertical format
    df_results = pd.DataFrame(results)

    # Create result_loss directory if it doesn't exist
    result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result_loss')
    os.makedirs(result_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filename based on method, architecture and dataset
    if args.method == 'dino' and 'model' in locals():
        # Try to get architecture info from the model
        arch_info = getattr(model, 'arch_name', 'unknown')
        filename = f"{args.method}_{arch_info}_{args.dataset_name}_{timestamp}.csv"
    elif args.method == 'moco' and 'model' in locals():
        # For MoCo, use the arch information
        arch_info = getattr(model, 'arch', 'unknown')
        filename = f"{args.method}_{arch_info}_{args.dataset_name}_{timestamp}.csv"
    else:
        # For UMAP/t-SNE or when model info is not available
        filename = f"{args.method}_{args.dataset_name}_{timestamp}.csv"
    
    csv_path = os.path.join(result_dir, filename)
    
    # Always create a new file with timestamp (no appending)
    df_results.to_csv(csv_path, mode='w', header=True, index=False)
    print(f'✅ Saved results to new file: {csv_path}')
    
    # Also create/update a summary file that contains all runs
    summary_filename = f"{args.method}_{args.dataset_name}_all_runs.csv"
    summary_path = os.path.join(result_dir, summary_filename)
    
    # Add timestamp to each result for the summary file
    for result in results:
        result['timestamp'] = timestamp
        result['run_id'] = timestamp  # Alternative identifier
    
    df_summary = pd.DataFrame(results)
    
    # Save to summary file (append mode)
    if os.path.exists(summary_path):
        df_summary.to_csv(summary_path, mode='a', header=False, index=False)
        print(f'📊 Updated summary file: {summary_path}')
    else:
        df_summary.to_csv(summary_path, mode='w', header=True, index=False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating files')
    parser.add_argument('--checkpoint_path', default='path/to/checkpoint', help='Path to the model checkpoint.')
    parser.add_argument('--csv_path', default='./results_loss.csv', help='Path to the CSV file for losses.')
    parser.add_argument('--data_path', default='path/to/data', help='Path to the dataset.')
    parser.add_argument('--dataset_name', default='dataset', help='Name of the dataset.')
    parser.add_argument('--method', default='dino', choices=['dino', 'moco', 'umap', 'tsne'], help='Method to use.')
    parser.add_argument('--umapnumpy_path', default='path/to/umap.npy', help='Path to UMAP numpy file.')
    parser.add_argument('--umapnumpy_path2', default='', help='Path to second UMAP numpy file (augmentation 2).')
    parser.add_argument('--tsnenumpy_path', default='path/to/tsne.npy', help='Path to t-SNE numpy file.')
    parser.add_argument('--tsnenumpy_path2', default='', help='Path to second t-SNE numpy file (augmentation 2).')
    parser.add_argument('--use_grayscale3', action='store_true', help='Use 3-channel grayscale conversion.')


    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for K-means.')
    parser.add_argument('--k_classifier', type=int, default=5, help='Number of neighbors for KNN classifier.')
    parser.add_argument('--gam1', type=float, default=1.0, help='Gamma 1 for MCR2.')
    parser.add_argument('--gam2', type=float, default=1.0, help='Gamma 2 for MCR2.')
    parser.add_argument('--eps', type=float, default=0.2, help='Epsilon for MCR2.')
    parser.add_argument('--alpha', type=float, default=2.0, help='Alpha for align-uniform.')
    parser.add_argument('--t', type=float, default=2.0, help='T for align-uniform.')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to use for loss computation to avoid memory issues.')
    
    parser.add_argument('--test', action='store_true', help='Run test loss generation.')
    
    args = parser.parse_args()

    # Run loss generation if test flag is set
    if args.test:
        gen_loss(args)
