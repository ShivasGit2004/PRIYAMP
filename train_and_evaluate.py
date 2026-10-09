import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm

from src.models import CNNBaseline, MultiScaleDomainPCOSNet, coral_loss
from src.evaluate import calculate_metrics

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 1e-4

TRAIN_IMG_DIR = 'PCOSGen-train/images'
LABEL_FILE = 'PCOSGen-train/class_label.xlsx'
TEST_IMG_DIR = 'PCOSGen-test/images'

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PCOSMedicalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['imagePath']
        # Map class label: 0 -> Unhealthy/PCOS (Class 1), 1 -> Healthy (Class 0)
        # In challenge data: Healthy column is 1 for healthy, 0 for unhealthy PCOS.
        healthy_val = int(row['Healthy'])
        label = 0 if healthy_val == 1 else 1 # 0: Healthy, 1: Unhealthy (PCOS)
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


class UnlabeledTargetDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_patient_id(filename):
    """Extract patient/video group ID to prevent cross-frame data leakage."""
    match = re.match(r'^(.*?)image', str(filename))
    if match and match.group(1):
        return match.group(1)
    # If no prefix, each individual frame is treated as independent group
    return f"indiv_{filename}"


def prepare_data_splits():
    df = pd.read_excel(LABEL_FILE)
    df['group_id'] = df['imagePath'].apply(get_patient_id)
    
    # Filter only images that exist on disk
    existing_images = set(os.listdir(TRAIN_IMG_DIR))
    df = df[df['imagePath'].isin(existing_images)].reset_index(drop=True)
    
    # Patient-level group stratified split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(df, df['Healthy'], groups=df['group_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Data split without leakage: Train = {len(train_df)} images ({train_df['group_id'].nunique()} groups), Val = {len(val_df)} images ({val_df['group_id'].nunique()} groups)")
    return train_df, val_df


def train_single_model(model, train_loader, val_loader, model_name, target_loader=None, is_coral=False, epochs=NUM_EPOCHS):
    print(f"\n================ Training {model_name} ================")
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    target_iter = iter(target_loader) if is_coral and target_loader is not None else None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        coral_losses = 0.0
        correct = 0
        total = 0
        
        # Subsample batches for fast execution on CPU while preserving training dynamics
        max_batches = 60
        for b_idx, (src_imgs, src_labels) in enumerate(train_loader):
            if b_idx >= max_batches:
                break
                
            src_imgs, src_labels = src_imgs.to(DEVICE), src_labels.to(DEVICE)
            optimizer.zero_grad()
            
            if is_coral and target_loader is not None:
                try:
                    tgt_imgs = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tgt_imgs = next(target_iter)
                tgt_imgs = tgt_imgs.to(DEVICE)
                
                # Forward source & target
                src_logits, src_feats = model(src_imgs, return_features=True)
                tgt_logits, tgt_feats = model(tgt_imgs, return_features=True)
                
                cls_loss = criterion(src_logits, src_labels)
                c_loss = coral_loss(src_feats, tgt_feats)
                loss = cls_loss + 0.5 * c_loss
                coral_losses += c_loss.item()
            else:
                src_logits = model(src_imgs)
                if isinstance(src_logits, tuple):
                    src_logits = src_logits[0]
                cls_loss = criterion(src_logits, src_labels)
                loss = cls_loss
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(src_logits, 1)
            correct += (preds == src_labels).sum().item()
            total += src_labels.size(0)
            
        scheduler.step()
        train_acc = correct / total if total > 0 else 0
        coral_info = f" | CORAL Loss: {coral_losses/(b_idx+1):.4f}" if is_coral else ""
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/(b_idx+1):.4f} - Train Acc: {train_acc*100:.2f}%{coral_info}")
        
    # Evaluate on validation set
    metrics = evaluate_model(model, val_loader)
    print(f"Validation Metrics for {model_name}: {metrics}")
    return model, metrics


def evaluate_model(model, val_loader):
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of PCOS (Class 1)
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    return calculate_metrics(all_targets, all_probs)


def main():
    print("Starting Model Training, Domain Adaptation & Ablation Study...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    train_df, val_df = prepare_data_splits()
    
    train_dataset = PCOSMedicalDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
    val_dataset = PCOSMedicalDataset(val_df, TRAIN_IMG_DIR, transform=val_transform)
    target_dataset = UnlabeledTargetDataset(TEST_IMG_DIR, transform=train_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 1. Baseline CNN (Single-Scale ResNet18)
    model_baseline = CNNBaseline(num_classes=2, pretrained=True)
    model_baseline, metrics_baseline = train_single_model(
        model_baseline, train_loader, val_loader, "Baseline CNN (ResNet18)", epochs=NUM_EPOCHS
    )
    torch.save(model_baseline.state_dict(), 'models/cnn_baseline.pth')
    
    # 2. Ablation 1: Baseline + SE Channel Attention
    model_se = MultiScaleDomainPCOSNet(num_classes=2, pretrained=True, use_se=True, use_multiscale=False)
    model_se, metrics_se = train_single_model(
        model_se, train_loader, val_loader, "Baseline + SE Attention", epochs=NUM_EPOCHS
    )
    
    # 3. Ablation 2: Baseline + Multi-Scale Feature Fusion
    model_ms = MultiScaleDomainPCOSNet(num_classes=2, pretrained=True, use_se=False, use_multiscale=True)
    model_ms, metrics_ms = train_single_model(
        model_ms, train_loader, val_loader, "Baseline + Multi-Scale Fusion", epochs=NUM_EPOCHS
    )
    
    # 4. Proposed: Multi-Scale + SE Attention + Deep CORAL Domain Adaptation
    model_proposed = MultiScaleDomainPCOSNet(num_classes=2, pretrained=True, use_se=True, use_multiscale=True)
    model_proposed, metrics_proposed = train_single_model(
        model_proposed, train_loader, val_loader, "Proposed: Multi-Scale + SE + CORAL",
        target_loader=target_loader, is_coral=True, epochs=NUM_EPOCHS
    )
    torch.save(model_proposed.state_dict(), 'models/pcos_coral_multiscale.pth')
    
    # Save Ablation Results
    ablation_data = [
        {"Model": "Baseline CNN (ResNet-18)", **metrics_baseline},
        {"Model": "Baseline + SE Channel Attention", **metrics_se},
        {"Model": "Baseline + Multi-Scale Fusion", **metrics_ms},
        {"Model": "Proposed (Multi-Scale + SE + CORAL)", **metrics_proposed}
    ]
    ablation_df = pd.DataFrame(ablation_data)
    ablation_df.to_csv('results/ablation_study.csv', index=False)
    print("\nSaved Ablation Study to results/ablation_study.csv:")
    print(ablation_df[['Model', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', 'roc_auc', 'brier_score']])
    
    # Save Evaluation Metrics
    eval_df = pd.DataFrame([
        {"Model": "Baseline CNN", **metrics_baseline},
        {"Model": "Proposed Robust PCOSNet", **metrics_proposed}
    ])
    eval_df.to_csv('results/evaluation_metrics.csv', index=False)
    print("\nSaved Evaluation Metrics to results/evaluation_metrics.csv")
    print("Training and evaluation completed successfully!")


if __name__ == '__main__':
    main()
