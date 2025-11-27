#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import sys
import random
import warnings

# Add parent directory to path to import DINO module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import DINO.vision_transformer as vits
from loader.tinyimagenet_dataset import get_tinyimagenet_dataset

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Linear MoCo Evaluation")
parser.add_argument('--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='dataset name (cifar10, mnist, tinyimagenet)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to dino checkpoint')
parser.add_argument('--batch_size', default=24, type=int, metavar='N',
                    help='mini-batch size for training')
parser.add_argument('--val_batch_size', default=48, type=int, metavar='N',
                    help='mini-batch size for validation')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use')


def train_t(use_grayscale3=False):
    """Training transforms - same augmentation for all datasets"""
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    augmentation = base + [
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize,
    ]
    return transforms.Compose(augmentation)

def val_t(use_grayscale3=False):
    """Test transforms - same augmentation for all datasets"""
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    augmentation = base + [
          transforms.Resize(256, interpolation=3),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
    ]
    return transforms.Compose(augmentation)

#load model
class DINO_model(nn.Module):
    """Feature extractor that loads a DINO checkpoint and builds the backbone
    """

    def __init__(self, 
                 checkpoint_path, 
                 device="cpu"):

        super(DINO_model, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path

        # --- load checkpoint ---
        ck = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(ck, dict):
            raise RuntimeError("Checkpoint must be a dict-like object")

        print(f"Checkpoint keys: {list(ck.keys())}")

        # Handle different checkpoint formats
        if 'teacher' in ck:
            model_sd = ck['teacher']
            print("Using 'teacher' weights from checkpoint")
        elif 'state_dict' in ck:
            model_sd = ck['state_dict']
            print("Using 'state_dict' weights from checkpoint")
        elif 'model' in ck:
            model_sd = ck['model']
            print("Using 'model' weights from checkpoint")
        else:
            model_sd = ck
            print("Using checkpoint as direct state dict")
    
        # Try to get args from checkpoint
        self.args = ck.get('args', None)

        # remove common prefixes
        self.state_dict = {k.replace('module.', '').replace('backbone.', '').replace('encoder.', ''): v
                           for k, v in model_sd.items()}
        
        # Auto-detect patch_size and arch from weights
        self.detected_patch_size = None
        self.detected_arch = None
        
        if 'patch_embed.proj.weight' in self.state_dict:
            patch_weight_shape = self.state_dict['patch_embed.proj.weight'].shape
            self.detected_patch_size = patch_weight_shape[2]
            embed_dim = patch_weight_shape[0]
            
            if embed_dim == 192:
                self.detected_arch = 'vit_tiny'
            elif embed_dim == 384:
                self.detected_arch = 'vit_small'
            elif embed_dim == 768:
                self.detected_arch = 'vit_base'
            
            print(f"Auto-detected: arch={self.detected_arch}, patch_size={self.detected_patch_size}, embed_dim={embed_dim}")
        
        
    def build_model(self):
        # Use auto-detected values first, then args, then defaults
        arch = self.detected_arch or 'vit_small'
        patch_size = self.detected_patch_size or 16
        
        # Override with args if available (only if not auto-detected)
        if self.args is not None:
            if isinstance(self.args, dict):
                if self.detected_arch is None:
                    arch = self.args.get('arch', arch)
                if self.detected_patch_size is None:
                    patch_size = self.args.get('patch_size', patch_size)
            else:
                if self.detected_arch is None:
                    arch = getattr(self.args, 'arch', arch)
                if self.detected_patch_size is None:
                    patch_size = getattr(self.args, 'patch_size', patch_size)

        print(f"Building model: arch={arch}, patch_size={patch_size}")

        # Update arch_name after potential override
        self.arch_name = arch

        if arch in vits.__dict__:
            try:
                model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            except Exception as e:
                print(f"Failed to create model with patch_size, trying without: {e}")
                model = vits.__dict__[arch](num_classes=0)
        elif 'xcit' in arch:
            model = torch.hub.load('facebookresearch/xcit:main', arch, num_classes=0)
        else:
            # fallback to vit_small
            model = vits.__dict__['vit_small'](patch_size=patch_size, num_classes=0)

        self.model = model.to(self.device)

        # load teacher weights
        msg = self.model.load_state_dict(self.state_dict, strict=False)
        print(f"Loaded teacher weights from '{self.checkpoint_path}' with msg: {msg}")

        # ensure head is identity (so forward returns backbone features)
        for head in ('head', 'heads', 'classifier', 'fc'):
            if hasattr(self.model, head):
                try:
                    setattr(self.model, head, nn.Identity())
                except Exception:
                    pass

        # freeze all params
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.feature_dim = getattr(self.model, 'embed_dim', None) or getattr(self.model, 'num_features', None) or 768

        return self.model


def flatten_feats(feats):
    if feats.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(feats, 1).view(feats.size(0), -1)

    if feats.ndim == 3:
        return feats.mean(dim=1)
    return feats

    
def main():
    args = parser.parse_args()
    
    # Get dataset based on args
    if args.dataset == 'cifar10':
        train_ds = datasets.CIFAR10(args.data, train=True, download=True, 
                                    transform=train_t(use_grayscale3=False))
        val_ds = datasets.CIFAR10(args.data, train=False, download=True, 
                                  transform=val_t(use_grayscale3=False))
       
    elif args.dataset == 'mnist':
        train_ds = datasets.MNIST(args.data, train=True, download=True, 
                                  transform=train_t(use_grayscale3=True))
        val_ds = datasets.MNIST(args.data, train=False, download=True, 
                                transform=val_t(use_grayscale3=True))
    
    elif args.dataset == 'tinyimagenet':
        # TinyImageNet: train uses ImageFolder, val uses custom loader
        train_ds = get_tinyimagenet_dataset(args.data, split='train',
                                            transform=train_t(use_grayscale3=False))
        val_ds = get_tinyimagenet_dataset(args.data, split='val',
                                          transform=val_t(use_grayscale3=False))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=0, pin_memory=True)


    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    backbone = DINO_model(checkpoint_path=args.checkpoint, device=device)
    backbone_model = backbone.build_model()

    # Create save directory structure: saved_features/linear_dino/{dataset}/{checkpoint_name}/
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_dir = os.path.join('saved_features', 'linear_dino', args.dataset, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*50)
    print("EXTRACTING AND SAVING FEATURES")
    print("="*50)
    
    # Extract ALL train features
    print(f"Extracting train features...")
    train_features_list = []
    train_labels_list = []
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            feats = backbone_model(imgs)
            feats = flatten_feats(feats)  # Flatten to (B, D)
            
            train_features_list.append(feats.cpu())
            train_labels_list.append(labels)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
    
    train_features = torch.cat(train_features_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)
    
    # Save train features
    train_save_path = os.path.join(save_dir, 'train_features.pt')
    torch.save({
        'features': train_features,
        'labels': train_labels
    }, train_save_path)
    print(f"✓ Saved train features: {train_features.shape} to {train_save_path}")
    
    # Extract ALL val features
    print(f"\nExtracting val features...")
    val_features_list = []
    val_labels_list = []
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.to(device)
            feats = backbone_model(imgs)
            feats = flatten_feats(feats)  # Flatten to (B, D)
            
            val_features_list.append(feats.cpu())
            val_labels_list.append(labels)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    val_features = torch.cat(val_features_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)
    
    # Save val features
    val_save_path = os.path.join(save_dir, 'val_features.pt')
    torch.save({
        'features': val_features,
        'labels': val_labels
    }, val_save_path)
    print(f"✓ Saved val features: {val_features.shape} to {val_save_path}")



    # Print summary
    print("\n" + "="*50)
    print("FEATURE EXTRACTION COMPLETED")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"Train samples: {len(train_labels)}, Feature dim: {train_features.shape[1]}")
    print(f"Val samples: {len(val_labels)}, Feature dim: {val_features.shape[1]}")
    print(f"Saved to: {save_dir}")
    print("="*50)


    

if __name__ == '__main__':
    main()

