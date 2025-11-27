import argparse
import os
from datetime import datetime


from torch import device
import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import DataLoader
import sys


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class FeatureDataset(torch.utils.data.Dataset):
    """Dataset for pre-extracted features"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



# Parse arguments
parser = argparse.ArgumentParser(description="Linear Evaluation with Pre-extracted Features")
parser.add_argument('--features-path', type=str, required=True,
                    help='path to saved features directory (e.g., saved_features/moco/cifar10/checkpoint_name/)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset name for saving results')
parser.add_argument('--checkpoint', default='', type=str,
                    help='checkpoint name for saving results')
parser.add_argument('--batch-size', default=34, type=int,
                    help='batch size for training')
parser.add_argument('--milestone_epochs', type=int, nargs='+', default=[20, 40, 60, 80, 100],
                    help='epochs to train and record (default: 20 40 60 80 100)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--lr_list', type=float, nargs='+', default=[0.0001, 0.001, 0.01, 0.1, 1.0],
                    help='List of learning rates to try (only used for dino). Default: [0.0001, 0.001, 0.01, 0.1, 1.0]')
parser.add_argument('--method', default='dino', type=str, choices=['moco', 'simclr', 'dino'],
                    help='SSL method used for feature extraction')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use')

args = parser.parse_args()

# Set device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-extracted features
print("="*50)
print("LOADING PRE-EXTRACTED FEATURES")
print("="*50)

train_features_path = os.path.join(args.features_path, 'train_features.pt')
val_features_path = os.path.join(args.features_path, 'val_features.pt')

if not os.path.exists(train_features_path) or not os.path.exists(val_features_path):
    raise FileNotFoundError(
        f"Features not found in {args.features_path}\n"
        f"Please run feature extraction first."
    )

# Load features
train_data = torch.load(train_features_path)
val_data = torch.load(val_features_path)

train_features = train_data['features']
train_labels = train_data['labels']
val_features = val_data['features']
val_labels = val_data['labels']

print(f"✓ Loaded train features: {train_features.shape}")
print(f"✓ Loaded val features: {val_features.shape}")
# print("Note: For CIFAR-10/MNIST, val set is also the test set (no separate test set)")

# Get feature dimension and number of classes
feat_dim = train_features.shape[1]
num_classes = len(torch.unique(train_labels))

print(f"Feature dimension: {feat_dim}")
print(f"Number of classes: {num_classes}")
print("="*50)

# Create datasets and dataloaders
train_dataset = FeatureDataset(train_features, train_labels)
val_dataset = FeatureDataset(val_features, val_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# ============================================================================
# DINO LR SWEEP MODE: If --lr-list provided, run multiple lr experiments
# ============================================================================
# This section runs when: method=dino AND --lr-list is provided
# Default lr-list for DINO: [0.0001, 0.001, 0.01, 0.1, 1.0]
# After completion, it exits (sys.exit) to avoid running the single-run code below
if args.method.lower() == 'dino' and args.lr_list is not None and len(args.lr_list) > 0:
    lr_values = args.lr_list
    milestone_epochs = sorted(args.milestone_epochs)
    total_epochs = max(milestone_epochs)
    results_dir = os.path.join('results_linear', args.method, args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for lr_run in lr_values:
        print('\n' + '='*60)
        print(f"Running linear eval for DINO lr={lr_run}")
        print('='*60 + '\n')

        head = nn.Linear(feat_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        actual_lr = lr_run
        optimizer = optim.SGD(head.parameters(), lr=actual_lr, momentum=0.9, weight_decay=0.0)
        print(f"DINO optimizer: lr={actual_lr}, momentum=0.9, weight_decay=0.0")

        best_val_acc1 = 0
        best_val_acc5 = 0
        best_epoch = 0
        milestone_results = {}

        for epoch in range(total_epochs):
            head.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for feats, labels in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = head(feats)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * feats.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += feats.size(0)

            train_acc = correct / total
            print(f"Epoch {epoch+1}/{total_epochs} train_loss={running_loss/total:.4f} train_acc={train_acc:.4f}")

            head.eval()
            val_top1 = 0
            val_top5 = 0
            val_total = 0
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats, labels = feats.to(device), labels.to(device)
                    logits = head(feats)
                    acc1, acc5 = accuracy(logits, labels, topk=(1,5))
                    val_top1 += acc1.item() * feats.size(0)
                    val_top5 += acc5.item() * feats.size(0)
                    val_total += feats.size(0)

            avg_val_top1 = val_top1 / val_total
            avg_val_top5 = val_top5 / val_total
            print(f"Validation - Top1 Acc: {avg_val_top1:.2f}%, Top5 Acc: {avg_val_top5:.2f}%")

            if avg_val_top1 > best_val_acc1:
                best_val_acc1 = avg_val_top1
                best_val_acc5 = avg_val_top5
                best_epoch = epoch + 1

            current_epoch = epoch + 1
            if current_epoch in milestone_epochs:
                milestone_results[current_epoch] = {'top1': avg_val_top1, 'top5': avg_val_top5, 'train_acc': train_acc}
                marker = '📍' if os.name != 'nt' else '[MILESTONE]'
                print(f"{marker} Milestone: Epoch {current_epoch} - Top1: {avg_val_top1:.2f}%, Top5: {avg_val_top5:.2f}%")

        run_result = {'lr': actual_lr, 'best_top1': best_val_acc1, 'best_top5': best_val_acc5, 'best_epoch': best_epoch, 'milestones': milestone_results}
        all_results.append(run_result)

        # save per-lr results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckname = args.checkpoint if args.checkpoint else 'unknown'
        lr_tag = str(actual_lr).replace('.', 'p')
        savefile = os.path.join(results_dir, f"linear_eval_{args.dataset}_{ckname}_lr{lr_tag}_ep{total_epochs}_{ts}.txt")
        with open(savefile, 'w') as f:
            f.write('='*50 + '\n')
            f.write(f'DINO linear eval\n')
            f.write('='*50 + '\n')
            f.write(f'LR: {actual_lr}\n')
            f.write(f'Total epochs: {total_epochs}\n')
            f.write(f'Best Top1: {best_val_acc1:.2f}% (epoch {best_epoch})\n')
            f.write(f'Best Top5: {best_val_acc5:.2f}%\n')
            f.write('Milestones:\n')
            for ep in sorted(milestone_results.keys()):
                m = milestone_results[ep]
                f.write(f'  Epoch {ep}: Top1={m["top1"]:.2f}%, Top5={m["top5"]:.2f}%, Train={m["train_acc"]*100:.2f}%\n')
        print(f"Saved per-lr results to {savefile}\n")

    print('\nSUMMARY:')
    for r in all_results:
        print(f"LR={r['lr']}: Best Top1={r['best_top1']:.2f}% (ep {r['best_epoch']})")

    sys.exit(0)

# ============================================================================
# SINGLE-RUN MODE: Standard training for all methods or DINO with single lr
# ============================================================================
# This section runs when:
# - method = moco (uses fixed lr=30.0)
# - method = simclr (uses fixed lr=1.6)
# Note: DINO always uses lr-sweep mode above and exits before reaching here

# 1) create linear head, loss, and optimizer (only optimize head params)
head = nn.Linear(feat_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()

if args.method == "moco":
    # MoCo uses high learning rate (30.0) for linear evaluation
    actual_lr = 30.0
    optimizer = optim.SGD(
        head.parameters(), lr=actual_lr, momentum=0.9, weight_decay=0.0)
elif args.method == "simclr":
    # SimCLR uses lr=1.6 with Nesterov momentum
    actual_lr = 1.6
    optimizer = optim.SGD(
        head.parameters(), lr=actual_lr, momentum=0.9, nesterov=True, weight_decay=0.0)
else:
    # This should never be reached (DINO exits above)
    raise ValueError(f"Unexpected method: {args.method}. DINO should use lr-sweep mode.")


# Track best accuracy (no early stopping for CIFAR-10/MNIST)
best_val_acc1 = 0
best_val_acc5 = 0
best_epoch = 0

# Use milestone_epochs to determine training schedule
milestone_epochs = sorted(args.milestone_epochs)
total_epochs = max(milestone_epochs)  # Train until the last milestone
milestone_results = {}

print("\nStarting training...")
print(f"Training for {total_epochs} epochs total")
print(f"Will record accuracy at epochs: {milestone_epochs}")
    
# 2) training loop (only update head)
for epoch in range(total_epochs):
    head.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)
        
        logits = head(feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += feats.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}/{total_epochs} train_loss={running_loss/total:.4f} train_acc={train_acc:.4f}")

    # Validation with top1 and top5 accuracy
    head.eval()
    val_top1 = 0
    val_top5 = 0
    val_total = 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = head(feats)
            
            # Calculate top1 and top5 accuracy
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            val_top1 += acc1.item() * feats.size(0)
            val_top5 += acc5.item() * feats.size(0)
            val_total += feats.size(0)
    
    avg_val_top1 = val_top1 / val_total
    avg_val_top5 = val_top5 / val_total
    
    print(f"Validation - Top1 Acc: {avg_val_top1:.2f}%, Top5 Acc: {avg_val_top5:.2f}%")
    
    # Track best accuracy (for reporting, not for early stopping)
    if avg_val_top1 > best_val_acc1:
        best_val_acc1 = avg_val_top1
        best_val_acc5 = avg_val_top5
        best_epoch = epoch + 1
    
    # Record accuracy at milestone epochs
    current_epoch = epoch + 1
    if current_epoch in milestone_epochs:
        milestone_results[current_epoch] = {
            'top1': avg_val_top1,
            'top5': avg_val_top5,
            'train_acc': train_acc
        }
        print(f"📍 Milestone: Epoch {current_epoch} - Top1: {avg_val_top1:.2f}%, Top5: {avg_val_top5:.2f}%")

# Final results (use best validation accuracy achieved during training)
print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)
print(f"Best Validation Top-1 Accuracy: {best_val_acc1:.2f}% (epoch {best_epoch})")
print(f"Best Validation Top-5 Accuracy: {best_val_acc5:.2f}% (epoch {best_epoch})")

# Print milestone results
if milestone_results:
    print("\n" + "-"*50)
    print("MILESTONE EPOCHS RESULTS:")
    print("-"*50)
    for ep in sorted(milestone_results.keys()):
        res = milestone_results[ep]
        print(f"Epoch {ep:3d} - Top1: {res['top1']:5.2f}%, Top5: {res['top5']:5.2f}%, Train: {res['train_acc']*100:5.2f}%")
    print("-"*50)

print("Note:  validation set = test set")

# Save results to file
results = {
    'method': args.method,
    'dataset': args.dataset,
    'checkpoint': args.checkpoint,
    'features_path': args.features_path,
    'total_epochs': total_epochs,
    'milestone_epochs': milestone_epochs,
    'best_epoch': best_epoch,
    'lr': actual_lr,  # Use actual learning rate
    'best_val_top1_acc': best_val_acc1,
    'best_val_top5_acc': best_val_acc5,
}

# Create results directory structure: results_linear/{method}/{dataset}/
results_dir = os.path.join('results_linear', args.method, args.dataset)
os.makedirs(results_dir, exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_name = args.checkpoint if args.checkpoint else 'unknown'
results_file = os.path.join(results_dir, f"linear_eval_{args.dataset}_{checkpoint_name}_ep{total_epochs}_{timestamp}.txt")

with open(results_file, 'w') as f:
    f.write("="*50 + "\n")
    f.write("Linear Evaluation Results\n")
    f.write("="*50 + "\n")
    f.write(f"Method: {results['method']}\n")
    f.write(f"Dataset: {results['dataset']}\n")
    f.write(f"Checkpoint: {results['checkpoint']}\n")
    f.write(f"Features Path: {results['features_path']}\n")
    f.write(f"Total Epochs: {results['total_epochs']}\n")
    f.write(f"Milestone Epochs: {results['milestone_epochs']}\n")
    f.write(f"Best Epoch: {results['best_epoch']}\n")
    f.write(f"Learning Rate: {results['lr']}\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write("-"*50 + "\n")
    f.write(f"Best Validation Top-1 Accuracy: {results['best_val_top1_acc']:.2f}%\n")
    f.write(f"Best Validation Top-5 Accuracy: {results['best_val_top5_acc']:.2f}%\n")
    f.write("-"*50 + "\n")
    
    # Write milestone results
    if milestone_results:
        f.write("Milestone Epochs Results:\n")
        for ep in sorted(milestone_results.keys()):
            res = milestone_results[ep]
            f.write(f"  Epoch {ep:3d} - Top1: {res['top1']:5.2f}%, Top5: {res['top5']:5.2f}%, Train: {res['train_acc']*100:5.2f}%\n")
        f.write("-"*50 + "\n")
    
    f.write("Note: validation set = test set\n")
    f.write("No early stopping applied (trained for full epochs)\n")
    f.write("="*50 + "\n")

print(f"\n{'='*50}")
print(f"Results saved to: {results_file}")
print(f"{'='*50}")

if __name__ == '__main__':
    pass