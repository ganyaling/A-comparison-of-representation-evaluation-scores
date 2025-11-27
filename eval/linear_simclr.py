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

parser = argparse.ArgumentParser(description="PyTorch Linear SimCLR Evaluation")
parser.add_argument('--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'mnist', 'tinyimagenet'],
                    help='dataset name (cifar10, mnist, tinyimagenet)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to simclr checkpoint')
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
    """Validation transforms - same augmentation for all datasets"""
    base = [transforms.Grayscale(num_output_channels=3)] if use_grayscale3 else []
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    augmentation = base + [
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
    ]
    return transforms.Compose(augmentation)

def get_feature_dim(arch_name, model=None):
    """
    Get feature dimension based on architecture name.
    
    Args:
        arch_name: Architecture name (e.g., 'resnet18', 'resnet50')
        model: Optional model instance to check attributes
    
    Returns:
        int: Feature dimension
    """
    arch_lower = arch_name.lower() if arch_name else ''
    
    # ResNet architectures
    if 'resnet18' in arch_lower or 'resnet34' in arch_lower:
        return 512
    elif 'resnet50' in arch_lower or 'resnet101' in arch_lower or 'resnet152' in arch_lower:
        return 2048
    elif 'resnext' in arch_lower or 'wide_resnet' in arch_lower:
        return 2048
    
    # Try to get from model attributes
    if model is not None:
        if hasattr(model, 'embed_dim'):
            return model.embed_dim
        if hasattr(model, 'num_features'):
            return model.num_features
    
    # Default fallback
    return 2048

class SimCLR_model(nn.Module):
    """Feature extractor that loads a SimCLR checkpoint and builds the backbone
    """

    def __init__(self, 
                 checkpoint_path, 
                 device="cpu"):

        super(SimCLR_model, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path  

        # --- Handle zip files (e.g., resnet18_100-epochs_cifar10.zip) ---
        import zipfile
        import tempfile
        
        if checkpoint_path.endswith('.zip'):
            print(f"Extracting checkpoint from zip file: {checkpoint_path}")
            with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
                # List all files in the zip
                file_list = zip_ref.namelist()
                print(f"Files in zip: {file_list}")
                
                # Find the .pth or .pth.tar file
                pth_files = [f for f in file_list if f.endswith('.pth') or f.endswith('.pth.tar')]
                if not pth_files:
                    raise RuntimeError(f"No .pth or .pth.tar file found in {checkpoint_path}")
                
                # Use the first .pth file found
                pth_file = pth_files[0]
                print(f"Loading checkpoint: {pth_file}")
                
                # Extract to temporary location and load
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                    tmp_file.write(zip_ref.read(pth_file))
                    tmp_path = tmp_file.name
                
                ck = torch.load(tmp_path, map_location="cpu")
                
                # Clean up temp file
                import os as os_module
                os_module.unlink(tmp_path)
        else:
            # Direct .pth or .pth.tar file
            ck = torch.load(checkpoint_path, map_location="cpu")
        
        if not isinstance(ck, dict):
            raise RuntimeError("Checkpoint must be a dict-like object")

        # Handle different checkpoint formats
        if 'state_dict' in ck:
            model_sd = ck['state_dict']
        else:
            model_sd = ck
            
        # For SimCLR: extract encoder weights and remove prefixes
       
        cleaned_state_dict = {}
        for k, v in model_sd.items():
            # Remove 'module.' prefix
            k = k.replace('module.', '')
            
            # For SimCLR: only keep encoder weights and remove the 'encoder.' prefix
            if k.startswith('encoder.'):
                new_key = k.replace('encoder.', '')
                cleaned_state_dict[new_key] = v
            else:
                # Remove other common prefixes
                new_key = k.replace('backbone.', '')
                cleaned_state_dict[new_key] = v
                
        self.state_dict = cleaned_state_dict
        self.arch = ck.get("arch", "resnet50")

    def build_model(self):
        # Build model with architecture from checkpoint, fallback to resnet50
        if self.arch is None or self.arch not in models.__dict__:
            print(f"Warning: Architecture '{self.arch}' not found, using resnet50 as fallback")
            self.arch = 'resnet50'
            
        print(f"Building {self.arch} model...")
        self.model = models.__dict__[self.arch]()
        self.model = self.model.to(self.device)

        # For SimCLR: Replace fc layer with Identity before loading weights
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()
        
        # Remove fc weights from state_dict if present (to avoid size mismatch)
        state_dict_no_fc = {k: v for k, v in self.state_dict.items() if not k.startswith('fc.')}
        
        # load encoder weights (without fc layer)
        msg = self.model.load_state_dict(state_dict_no_fc, strict=False)
        print(f"Loaded encoder weights from '{os.path.basename(self.checkpoint_path)}'")
        print(f"Load message: {msg}")
        
        # Print how many weights were loaded
        loaded_keys = set(msg.missing_keys) if hasattr(msg, 'missing_keys') else set()
        total_keys = set(state_dict_no_fc.keys())
        print(f"Successfully loaded {len(total_keys) - len(loaded_keys)}/{len(total_keys)} weight tensors")

        # ensure other heads are identity too (redundant but safe)
        for head in ('head', 'heads', 'classifier'):
            if hasattr(self.model, head):
                try:
                    setattr(self.model, head, nn.Identity())
                except Exception:
                    pass

        # freeze all params
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        
        # Get feature dimension based on architecture
        self.feature_dim = get_feature_dim(self.arch, self.model)
        print(f"Feature dimension set to: {self.feature_dim}")
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

    backbone = SimCLR_model(checkpoint_path=args.checkpoint, device=device)
    backbone_model = backbone.build_model()

    # Create save directory structure: saved_features/linear_simclr/{dataset}/{checkpoint_name}/
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_dir = os.path.join('saved_features', 'linear_simclr', args.dataset, checkpoint_name)
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

