import torch
import torch.nn as nn
from torchvision import models
from torchvision import models as torchvision_models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import DINO.vision_transformer as vits


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

# Helper function to get feature dimension based on architecture
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

# DINO model
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

    def forward(self, train_loader):
        with torch.no_grad():
            features = self.model(train_loader)
        return features

       

     
# MoCo model
class MoCo_model(nn.Module):
    """Feature extractor that loads a MoCo checkpoint and builds the backbone
    """

    def __init__(self, 
                 checkpoint_path, 
                 device="cpu"):

        super(MoCo_model, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path  

        # --- load checkpoint and use only 'state_dict' ---
        ck = torch.load(checkpoint_path, map_location="cpu")
        
        if not isinstance(ck, dict):
            raise RuntimeError("Checkpoint must be a dict-like object")

        # Handle different checkpoint formats
        if 'state_dict' in ck:
            model_sd = ck['state_dict']
        else:
            model_sd = ck
            
        # For MoCo: extract encoder_q weights and remove prefixes
       
        cleaned_state_dict = {}
        for k, v in model_sd.items():
            # Remove 'module.' prefix
            k = k.replace('module.', '')
            
            # For MoCo: only keep encoder_q weights and remove the 'encoder_q.' prefix
            if k.startswith('encoder_q.'):
                new_key = k.replace('encoder_q.', '')
                cleaned_state_dict[new_key] = v
            # Also handle case where keys don't have encoder_q prefix (other formats)
            elif not k.startswith('encoder_k.') and 'encoder_q' not in k and 'encoder_k' not in k:
                # Remove other common prefixes
                new_key = k.replace('backbone.', '').replace('encoder.', '')
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

        # For MoCo: Replace fc layer with Identity before loading weights
        # MoCo's fc is a projection head with different dimensions (usually 128)
        # We want to extract features from the backbone, not the projection head
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
 
    def forward(self, train_loader):
        with torch.no_grad():
            features = self.model(train_loader)
        return features


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

    def forward(self, train_loader):
        with torch.no_grad():
            features = self.model(train_loader)
        return features
    























