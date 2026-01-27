

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score

import mlflow
import mlflow.pytorch
import dagshub

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")




CONFIG = {
    "DAGSHUB_USERNAME":os.getenv("DAGSHUB_USERNAME"),
    "DAGSHUB_TOKEN":os.getenv("DAGSHUB_TOKEN"),
    "DAGSHUB_REPO":"radae_mlops",
    "ENABLE_MLFLOW": os.getenv("ENABLE_MLFLOW", "false").lower() == "true",

    # Data
    "DATA_DIR": "data/raw",
    "OUTPUT_DIR": "data/processed",
    
    # Model
    "NUM_CLASSES": 9,
    "IMAGE_SIZE": 224,
    "BACKBONE": "efficientnet_b0",
    
    # Training
    "EPOCHS": 3,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    
    # Classes
    'CLASS_MAPPING': {
        '0': 'bicycle',
        '1': 'car',
        '2': '1_person',
        '3': '2_persons',
        '4': '3_persons',
        '5': 'mixed_car_bike_person',
        '6': 'person_bicycle',
        '7': 'person_car',
        '8': 'bicycle_car'
    }
    

}


print("COnfig loaded")

for k,v in CONFIG.items():
    if "TOKEN" not in k:
        print(f"{k}: {v}")




def init_mlflow():
    if not CONFIG["ENABLE_MLFLOW"]:
        print("🚫 MLflow disabled")
        return None

    if not CONFIG['DAGSHUB_USERNAME'] or not CONFIG['DAGSHUB_TOKEN']:
        raise RuntimeError("DAGSHUB credentials missing")

    os.environ['MLFLOW_TRACKING_USERNAME'] = CONFIG['DAGSHUB_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = CONFIG['DAGSHUB_TOKEN']

    dagshub.init(
        repo_owner=CONFIG['DAGSHUB_USERNAME'],
        repo_name=CONFIG['DAGSHUB_REPO'],
        mlflow=True
    )

    tracking_uri = f"https://dagshub.com/{CONFIG['DAGSHUB_USERNAME']}/{CONFIG['DAGSHUB_REPO']}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Radar_MLOps_Experiment")

    print(f"MLflow tracking at {tracking_uri}")
    return tracking_uri






class RadarDataset(Dataset):
    """Multimodal Radar Dataset (Image + Radar)"""
    
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __len__(self):
        # Return total number of samples in dataset
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Retrieve sample at given index
        sample = self.samples[idx]
        
        # Load image from file path
        image = cv2.imread(sample["image"])
        # Convert image from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image to configured size (224x224)
        image = cv2.resize(image, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        # Normalize image to [0, 1] range and convert to float32
        image = image.astype(np.float32) / 255.0
        
        # Load radar data from .mat file
        try:
            radar_data = loadmat(sample["radar"])
            # Extract radar cube, fallback to 'radar' key or zeros if not found
            radar = radar_data.get('data_cube', radar_data.get('radar', np.zeros((128, 255))))
            # If radar is 3D, extract first slice
            if radar.ndim > 2:
                radar = radar[:, :, 0]  
            # Resize radar to target dimensions
            radar = cv2.resize(radar.astype(np.float32), (255, 128))
            # Normalize radar to [0, 1] range
            radar = (radar - radar.min()) / (radar.max() - radar.min() + 1e-8)
        except:
            # Create empty radar array if loading fails
            radar = np.zeros((128, 255), dtype=np.float32)
            
        # Convert image from HWC to CHW format for PyTorch
        image = torch.from_numpy(image).permute(2, 0, 1)
        # Add channel dimension to radar (1, 128, 255)
        radar = torch.from_numpy(radar).unsqueeze(0)
        # Convert label to tensor
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {"image": image, 'radar': radar, 'label': label}
    
    
def load_dataset(dat_dir):
    # Initialize empty list to store sample metadata
    samples = []
    # Create mapping from class names to indices
    class_to_idx = {v: int(k) for k, v in CONFIG['CLASS_MAPPING'].items()}
    
    # Convert directory path to Path object
    data_path = Path(dat_dir)
    
    # Iterate through each class
    for class_name, class_idx in tqdm(class_to_idx.items(), desc="Loading classes"):
        # Construct path for current class directory
        class_path = data_path / class_name
        
        # Skip if class directory doesn't exist
        if not class_path.exists():
            continue
        
        # Iterate through scenario subdirectories within class
        for scenario in class_path.iterdir():
            if not scenario.is_dir():
                continue
            
            # Construct paths for images and radar subdirectories
            images_path = scenario / "images"
            radar_path = scenario / "radar"
            
            # Skip scenario if images directory doesn't exist
            if not images_path.exists():
                continue
            
            # Iterate through all jpg files in images directory
            for img_file in images_path.glob("*.jpg"):
                # Construct corresponding radar file path
                rad_file = radar_path / f"{img_file.stem}.mat"
                
                # Add sample record with image, radar, label, and class name
                samples.append({
                    'image': str(img_file),
                    "radar": str(rad_file) if rad_file.exists() else None,
                    "label": class_idx,
                    "class_name": class_name
                })
                

    # Initialize dictionary to count samples per class
    class_counts = {}

    # Count samples for each class
    for s in samples:
        class_counts[s['class_name']] = class_counts.get(s['class_name'], 0) + 1
    
    # Print class distribution statistics
    print("\nClass distribution")
    for c, n in sorted(class_counts.items()):
        print(f"   {c}: {n}")
        
    # Return samples list and class-to-index mapping        
    return samples, class_to_idx

# Load dataset from configured directory
samples, class_to_idx = load_dataset(CONFIG["DATA_DIR"])





"""Split data and create Pytorch Dataloaders"""


def create_dataloaders(samples, class_to_idx):
    """Create train/test dataloaders"""
    
    
    # Get labels from stratified split
    labels = [s['label'] for s in samples]
    indices = np.arange(len(samples))
    
    # Split 70% train 15 test and 15 val
    train_idx, temp_idx = train_test_split(indices, train_size=0.7, stratify=labels, random_state=42)
    
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, 
        stratify=[labels[i] for i in temp_idx], random_state=42
    )
    
    
    # Create Datasets
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    
    train_dataset = RadarDataset(train_samples, class_to_idx)
    val_dataset = RadarDataset(val_samples, class_to_idx)
    test_dataset = RadarDataset(test_samples, class_to_idx)
    
    
    # Create dataloaders
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, num_workers=4, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, num_workers=2, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, num_workers=2, pin_memory=False
    )
    
    
    print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"   Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    
    return train_loader, val_loader,test_loader

train_loader, val_loader, test_loader = create_dataloaders(samples, class_to_idx)





"""Multimodal CNN+LSTM Fusion Model"""

class MultimodalModel(nn.Module):
    """Multimodal fusion model for radar classification"""
    
    def __init__(self, num_classes=9):
        super().__init__()
        
        # Image encoder (EfficientNet)
        self.image_encoder = timm.create_model(
            CONFIG['BACKBONE'], pretrained=True, num_classes=0, global_pool="avg"
        )
        
        img_feat = 1200
        
        # Radar encoder (CNN)
        self.radar_encoder = nn.Sequential(
            nn.Conv2d(1,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        rad_feat = 256
        
        
        # Feature projection
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.rad_proj = nn.Sequential(
            nn.Linear(rad_feat, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        
        # LSTM feature
        self.lstm = nn.LSTM(1024, 512, num_layers=2, batch_first=True, dropout=0.4, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        
    def forward(self, image, radar):
        # Extract features
        img_feat = self.img_proj(self.image_encoder(image))
        rad_feat = self.rad_proj(self.radar_encoder(radar).flatten(1))
        
        # Fuse and LSTM
        fused = torch.cat([img_feat,rad_feat],dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(fused)
        
        
        return self.classifier(lstm_out.squeeze(1))
    
    
# Create model
model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)
print(f"Model created | Parameters: {sum(p.numel() for p in model.parameters()):,}")





"""Comprehensive metrics tracking"""

class MetricsTracker:
    """Track all classification metrics"""
    
    
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        
        # Initialize metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        self.precision = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.recall = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        self.precision_per_class = Precision(task='multiclass', num_classes=num_classes, average=None).to(device)
        self.recall_per_class = Recall(task='multiclass', num_classes=num_classes, average=None).to(device)
        self.f1_per_class = F1Score(task='multiclass', num_classes=num_classes, average=None).to(device)
        
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
        
    def update(self, preds, labels, probs=None):
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1.update(preds, labels)
        self.precision_per_class.update(preds, labels)
        self.recall_per_class.update(preds, labels)
        self.f1_per_class.update(preds, labels)
        
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())
        if probs is not None:
            self.all_probs.append(probs.cpu())
            
    def compute(self):
        metrics = {
            'accuracy': self.accuracy.compute().item(),
            'precision_macro': self.precision.compute().item(),
            'recall_macro': self.recall.compute().item(),
            'f1_macro': self.f1.compute().item(),
        }
        
        # Per-class metrics
        prec_pc = self.precision_per_class.compute()
        rec_pc = self.recall_per_class.compute()
        f1_pc = self.f1_per_class.compute()
        
        
        class_names = list(CONFIG["CLASS_MAPPING"].values())
        for i, name in enumerate(class_names):
            metrics[f'precision_{name}'] = prec_pc[i].item()
            metrics[f'recall_{name}'] = rec_pc[i].item()
            metrics[f'f1_{name}'] = f1_pc[i].item()
            
        # Additional metrics
        if self.all_preds:
            all_preds = torch.cat(self.all_preds).numpy()
            all_labels = torch.cat(self.all_labels).numpy()
            
            metrics['precision_weighted'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            metrics['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
            metrics['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
            metrics['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            
            # ROC-AUC
            if self.all_probs:
                try:
                    all_probs = torch.cat(self.all_probs).numpy()
                    metrics['roc_auc_ovr'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                    metrics['roc_auc_weighted'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
                except:
                    pass
        
        return metrics
    
    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.precision_per_class.reset()
        self.recall_per_class.reset()
        self.f1_per_class.reset()
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
        
print("Metrics tracker ready")




"""Training and Validation functions"""

def train_epoch(model, loader, criterion, optimizer, device, metrics):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device)
        radars = batch['radar'].to(device)
        labels = batch['label'].to(device)
        
        
        optimizer.zero_grad()
        outputs = model(images, radars)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        metrics.update(preds, labels, probs.detach())
        total_loss += loss.item() 
        
    return total_loss/len(loader), metrics.compute()

def validate_epoch(model, loader, criterion, device, metrics):
    """Validate one epoch"""
    model.eval()

    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device)
            radars = batch['radar'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, radars)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            metrics.update(preds, labels, probs.detach())
            total_loss += loss.item()
            
    return total_loss/len(loader), metrics.compute()

print("Training function defined")




"""Complete training loop with MLflow logging"""

def train_model():
    """Main training function"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True)
    
    train_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    val_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    
    best_val_acc = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        
        # Log config
        mlflow.log_params(CONFIG)
        
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Epochs: {CONFIG['EPOCHS']}")
        print("-" * 80)
        
        for epoch in range(CONFIG['EPOCHS']):
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
            print("-" * 80)
            
            # Train
            train_metrics.reset()
            train_loss, train_m = train_epoch(
                model, train_loader, criterion, optimizer, device, train_metrics)
            
            val_metrics.reset()
            val_loss, val_m = validate_epoch(
                model, val_loader, criterion, device, val_metrics)
            
            # Scheduler step
            scheduler.step(val_m['accuracy'])
            lr = optimizer.param_groups[0]['lr']
            
            # history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_m['accuracy'])
            history['val_acc'].append(val_m['accuracy'])
            
            # log in to mlflow
            mlflow.log_metric({
                
                'train_loss': train_loss,
                'train_accuracy': train_m['accuracy'],
                'train_precision': train_m['precision_macro'],
                'train_recall': train_m['recall_macro'],
                'train_f1': train_m['f1_macro'],
                'val_loss': val_loss,
                'val_accuracy': val_m['accuracy'],
                'val_precision': val_m['precision_macro'],
                'val_recall': val_m['recall_macro'],
                'val_f1': val_m['f1_macro'],
                'learning_rate': lr
            }, step=epoch)
            
            
            # log per class metrics
            for name in CONFIG['CLASS_MAPPING'].values():
                if f'precision_{name}' in val_m:
                    mlflow.log_metrics({
                        f'val_precision_{name}': val_m[f'precision_{name}'],
                        f'val_recall_{name}': val_m[f'recall_{name}'],
                        f'val_f1_{name}': val_m[f'f1_{name}']
                    }, step=epoch)
                    
            # Log weighted/micro metrics
            if 'precision_weighted' in val_m:
                mlflow.log_metrics({
                    'val_precision_weighted': val_m['precision_weighted'],
                    'val_recall_weighted': val_m['recall_weighted'],
                    'val_f1_weighted': val_m['f1_weighted'],
                    'val_precision_micro': val_m['precision_micro'],
                    'val_recall_micro': val_m['recall_micro'],
                    'val_f1_micro': val_m['f1_micro']
                }, step=epoch)
            
            # Log ROC-AUC
            if 'roc_auc_ovr' in val_m:
                mlflow.log_metrics({
                    'val_roc_auc': val_m['roc_auc_ovr'],
                    'val_roc_auc_weighted': val_m['roc_auc_weighted']
                }, step=epoch)
            
            # Print summary
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_m['accuracy']:.4f} | F1: {train_m['f1_macro']:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_m['accuracy']:.4f} | F1: {val_m['f1_macro']:.4f}")
            
            if val_m['accuracy'] > best_val_acc:
                best_val_acc = val_m['accuracy']
                torch.save(
                    {
                        'epoch':epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "val_accuracy":val_m['accuracy'],
                        "config":CONFIG
                    }, f"{CONFIG['OUTPUT_DIR']}/best_model.pth"
                )
                
                mlflow.pytorch.log_model(model,"best_model")
                
        mlflow.pytorch.log_model(model,"final_model")
    
    return history, best_val_acc

history, best_acc = train_model()





"""
Visualize training results
"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Val')
axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['OUTPUT_DIR']}/training_history.png", dpi=150)
plt.show()





"""Evaluate on test set"""


def evaluate_test():
    """Evaluation of model on test set """
    
    # load best model
    checkpoint = torch.load(f"{CONFIG['OUTPUT_DIR']}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            radars = batch['radar'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, radars)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            
            test_metrics.update(preds, labels, probs.detach())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    
    # Compute metrics
    metrics = test_metrics.compute()
    
    # Print results
    print("\n" + "-" * 50)
    print("OVERALL METRICS:")
    print("-" * 50)
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision (macro):{metrics['precision_macro']:.4f}")
    print(f"Recall (macro):   {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    if 'roc_auc_ovr' in metrics:
        print(f"ROC-AUC (OvR):    {metrics['roc_auc_ovr']:.4f}")
    
    # Classification report
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(
        all_labels, all_preds,
        target_names=list(CONFIG['CLASS_MAPPING'].values()),
        zero_division=0
    ))
    
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CONFIG['CLASS_MAPPING'].values()),
                yticklabels=list(CONFIG['CLASS_MAPPING'].values()))
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['OUTPUT_DIR']}/confusion_matrix.png", dpi=150)
    plt.show()
    
    
    with mlflow.start_run(run_name="test_evaluation"):
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(f"{CONFIG['OUTPUT_DIR']}/confusion_matrix.png")
    
    # Evaluation Completed
    return metrics





"""Production inference function"""


def predict_single(image_path, radar_path, model_path=None):
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Load and preprocess
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
    image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    try:
        radar_data = loadmat(radar_path)
        radar = radar_data.get('data_cube', np.zeros((128, 255)))
        if radar.ndim > 2:
            radar = radar[:, :, 0]
        radar = cv2.resize(radar.astype(np.float32), (255, 128))
        radar = (radar - radar.min()) / (radar.max() - radar.min() + 1e-8)
    except:
        radar = np.zeros((128, 255), dtype=np.float32)
    
    radar = torch.from_numpy(radar).unsqueeze(0).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image, radar)
        probs = F.softmax(output, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()
    
    class_name = CONFIG['CLASS_MAPPING'][str(pred)]
    
    return {'class': class_name, 'confidence': conf, 'probabilities': probs.cpu().numpy()[0]}


if __name__ == "__main__":
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    MLFLOW_URI = init_mlflow()

    samples, class_to_idx = load_dataset(CONFIG["DATA_DIR"])
    train_loader, val_loader, test_loader = create_dataloaders(samples, class_to_idx)

    model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)

    history, best_acc = train_model()
