"""
Evaluation Pipeline for Self-Supervised Representation Learning

Resolution Strategy Note:
All datasets are resized to 224×224 resolution to ensure fair comparison with 
ImageNet-pretrained models. While this changes the native characteristics of 
low-resolution datasets (MNIST: 28×28, CIFAR-10: 32×32), it prevents distribution 
shift that would otherwise confound evaluation results. This follows standard 
practice in the SSL literature.

Methods evaluated:
- MCR2 (Maximal Coding Rate Reduction)
- Alignment-Uniformity 
- CLID (Cluster Learnability + Intrinsic Dimension)
- RankMe (Effective Rank via SVD)
"""


import os
import numpy as np
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from model import DINO_model, MoCo_model, SimCLR_model
from loader.blur_solar import GaussianBlur_moco, GaussianBlur_dino, Solarization
from loader.tinyimagenet_dataset import get_tinyimagenet_dataset


def transform_moco(use_grayscale3=False, aug_plus=False):
    """
    MoCo-specific transform with 224x224 resolution
    
    Args:
        use_grayscale3: Convert grayscale to 3-channel (for MNIST)
        aug_plus: Use MoCo v2 augmentation (SimCLR-style)
    """
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = base + [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur_moco([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = base + [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    return transforms.Compose(augmentation)


def transform_base(use_grayscale3=False):
    """
    Basic transform with resize to 224x224 and normalization, but no augmentation
    This ensures consistent input size for pretrained models while keeping the original data distribution
    
    Args:
        use_grayscale3: Convert grayscale to 3-channel (for MNIST)
    """
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    # Resize to 224x224 without random cropping (center crop for consistency)
    basic_transform = base + [
        #transforms.Resize(256),  # Resize shorter side to 256
        #transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),
        #normalize,
    ]
    
    return transforms.Compose(basic_transform)


def transform_dino(use_grayscale3=False):
    """
    DINO-specific transform with standard augmentation
    
    Args:
        use_grayscale3: Convert grayscale to 3-channel (for MNIST)
    """  
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # DINO augmentation pipeline
    augmentation = base + [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(augmentation)

def transform_simclr(use_grayscale3=False, s=1, size=224):
    """
    SimCLR-style augmentation pipeline.

    Args:
        use_grayscale3 (bool): convert input to 3-channel grayscale
        s (float): strength multiplier for color jitter
        size (int): output crop size (default 224)
    """
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    augmentation = base + [transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor()
    ]
    return transforms.Compose(augmentation)


def gen_features(args):
    
    # ============ Set random seeds for reproducibility ============
    import random
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to: {args.seed}")
 
    # ============ preparing data ... ============
    
    
    # Select appropriate transform based on the method
    # dataset: original images (no augmentation)
    # dataset2: augmented images (with augmentation)
    if args.method == 'moco':
        transform_aug = lambda grayscale: transform_moco(grayscale, aug_plus=getattr(args, 'aug_plus', False))
    elif args.method == 'dino':
        transform_aug = transform_dino
    elif args.method == 'simclr':
        # SimCLR uses args.size for custom input size
        transform_aug = lambda grayscale: transform_simclr(grayscale, s=1, size=args.size)
    else:
        # For other methods, use SimCLR transform as default with args.size
        transform_aug = lambda grayscale: transform_simclr(grayscale, s=1, size=args.size)
    
    # Handle different dataset types with standard 224x224 resolution
    # dataset: basic transform (resize + center crop + normalize, no augmentation)
    # dataset2: augmented images (full augmentation pipeline)
    if args.dataset_name.lower() == 'cifar10':
        # CIFAR-10: Native 32x32 -> resized to 224x224
        #train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_base(False))
        test = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_base(False))
        #dataset = torch.utils.data.ConcatDataset([train, test])
        
        #train2 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_aug(False))
        test2 = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_aug(False))
        #dataset2 = torch.utils.data.ConcatDataset([train2, test2])
    elif args.dataset_name.lower() == 'mnist':
        # MNIST: Native 28x28 -> resized to 224x224, converted to 3-channel
        #train = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform_base(True))
        test = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform_base(True))
        #dataset = torch.utils.data.ConcatDataset([train, test])

        #train2 = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform_aug(True))
        test2 = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform_aug(True))
        #dataset2 = torch.utils.data.ConcatDataset([train2, test2])
    else:
        # For TinyImageNet (64x64) or other datasets -> resized to 224x224
        
        test =  get_tinyimagenet_dataset(args.data_path, split='val',
                                          transform=transform_base(False))
        test2 = get_tinyimagenet_dataset(args.data_path, split='val',
                                         transform=transform_aug(False))

    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)
    test_loader2 = DataLoader(test2, batch_size=args.batch_size, shuffle=True)

    print(f"Data loaded: there are {len(test)} images.")
    print(f"Test 1: Basic transform (no augmentation)")
    print(f"Test 2: Full augmentation ({args.method} style)")
    print(f"Will extract max {args.max_samples} samples from each")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.method in ('dino', 'moco', 'simclr'):
        # build model wrapper
        if args.method == 'dino':
            model = DINO_model(args.checkpoint_path, device=device)
            model.build_model()
            
            # Extract features for both augmentations (limited by max_samples)
            print(f"Extracting features for augmentation 1 (max {args.max_samples} samples)...")
            features_list = []
            samples_processed = 0
            for batch_idx, (data, _) in enumerate(test_loader):
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
            for batch_idx, (data, _) in enumerate(test_loader2):
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
            for batch_idx, (data, _) in enumerate(test_loader):
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
            for batch_idx, (data, _) in enumerate(test_loader2):
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

        elif args.method == 'simclr':
            model= SimCLR_model(args.checkpoint_path, device=device)
            model.build_model()

            # Extract features for both augmentations (limited by max_samples)
            print(f"Extracting features for augmentation 1 (max {args.max_samples} samples)...")
            features_list = []  
            samples_processed = 0
            for batch_idx, (data, _) in enumerate(test_loader):
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
            for batch_idx, (data, _) in enumerate(test_loader2):
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

        elif args.method == 'simclr':
            model= SimCLR_model(args.checkpoint_path, device=device)
            model.build_model()

            # Extract features for both augmentations (limited by max_samples)
            print(f"Extracting features for augmentation 1 (max {args.max_samples} samples)...")
            features_list = []  
            samples_processed = 0
            for batch_idx, (data, _) in enumerate(test_loader):
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
            for batch_idx, (data, _) in enumerate(test_loader2):
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

        else:       
            raise NotImplementedError(f"Method {args.method} not implemented.")      
    
    print(f"Using {features.shape[0]} samples for loss computation (features shape: {features.shape})")
    print(f"Features2 shape: {features2.shape}")

    # Save features to disk
    # Use csv_path as the base directory for saving features, or default to 'saved_features'
    if args.csv_path and args.csv_path != './saved_features':
        # User specified a custom directory
        save_dir = args.csv_path
    else:
        # Default directory
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_features')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename based on dataset name and checkpoint name
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]  # Remove extension
    dataset_name = args.dataset_name
    
    # Create subdirectory structure: {save_dir}/{method}/{dataset}/
    method_dir = os.path.join(save_dir, args.method)
    dataset_dir = os.path.join(method_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Add timestamp to filename to avoid overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save features with timestamp
    features_filename = f"{dataset_name}_{checkpoint_name}_{timestamp}_features.npy"
    features2_filename = f"{dataset_name}_{checkpoint_name}_{timestamp}_features2.npy"
    
    features_path = os.path.join(dataset_dir, features_filename)
    features2_path = os.path.join(dataset_dir, features2_filename)
    
    np.save(features_path, features)
    np.save(features2_path, features2)
    
    print(f"\n{'='*70}")
    print(f"✅ Features saved successfully!")
    print(f"{'='*70}")
    print(f"Features (aug1):  {features_path}")
    print(f"Features (aug2):  {features2_path}")
    print(f"Shape: {features.shape}")
    print(f"{'='*70}\n")
    
    return features, features2
    
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating files')
    parser.add_argument('--checkpoint_path', default='path/to/checkpoint', help='Path to the model checkpoint.')
    parser.add_argument('--csv_path', default='./saved_features', help='Directory to save feature files (default: ./saved_features).')
    parser.add_argument('--data_path', '--dataset', default='path/to/data', help='Path to the dataset.')
    parser.add_argument('--dataset_name', default='dataset', help='Name of the dataset.')
    parser.add_argument('--method', default='dino', choices=['dino', 'moco', 'simclr'], help='Method to use.')


    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use for loss computation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--aug_plus', action='store_true', help='Use MoCo v2 augmentation (stronger augmentation for MoCo).')
    parser.add_argument('--size', type=int, default=34, help='Input image size for SimCLR transform (default: 34).')
    args = parser.parse_args()

    # Generate features and save to disk 
    gen_features(args)
