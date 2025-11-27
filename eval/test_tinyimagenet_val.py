"""
Test script to verify TinyImageNet val loading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from loader.tinyimagenet_dataset import get_tinyimagenet_dataset
import torchvision.transforms as transforms

# Configure paths
TINYIMAGENET_ROOT = r"C:\Users\kawayi_yaling\OneDrive\datasets\tiny\tiny-imagenet-200"

# Define transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

print("="*60)
print("Testing TinyImageNet Val Loading")
print("="*60)

# Load val dataset
print("\nLoading val dataset...")
val_dataset = get_tinyimagenet_dataset(TINYIMAGENET_ROOT, split='val', transform=transform)

print(f"\n✅ Val dataset loaded successfully!")
print(f"Total samples: {len(val_dataset)}")
print(f"Number of classes: {len(val_dataset.classes)}")

# Test loading a few samples
print("\nTesting sample loading:")
for i in range(5):
    img, label = val_dataset[i]
    print(f"  Sample {i}: image shape={img.shape}, label={label}")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
