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

# PRODUCTION: Enable cudnn benchmark for free GPU speedup
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# PRODUCTION: Load classes from JSON (data-driven, not hardcoded)
def load_class_mapping(json_path="data/class_mapping.json"):
    """
    Load class mapping from JSON file.
    Falls back to minimal mapping in test/CI environments.
    """
    if not os.path.exists(json_path):
        #  SAFE fallback for tests / CI
        print(f"WARNING: {json_path} not found. Using test fallback mapping.")
        return {
            "0": "bicycle",
            "1": "car",
            "2": "1_person"
        }

    with open(json_path, 'r') as f:
        class_mapping = json.load(f)

    print(f"\nLoaded {len(class_mapping)} classes from {json_path}")
    return class_mapping


# Load class mapping BEFORE config
CLASS_MAPPING = load_class_mapping()
NUM_CLASSES = len(CLASS_MAPPING)


CONFIG = {
    "DAGSHUB_USERNAME": os.getenv("DAGSHUB_USERNAME"),
    "DAGSHUB_TOKEN": os.getenv("DAGSHUB_TOKEN"),
    "DAGSHUB_REPO": "radae_mlops",
    "ENABLE_MLFLOW": os.getenv("ENABLE_MLFLOW", "false").lower() == "true",

    # Data
    "DATA_DIR": "data/raw",
    "OUTPUT_DIR": "data/processed",
    
    # Model - AUTO-DETECTED from JSON
    "NUM_CLASSES": NUM_CLASSES,
    "IMAGE_SIZE": 128,
    "BACKBONE": "mobilenetv2_100",
    
    # Training
    "EPOCHS": 2,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    
    # Classes - LOADED from JSON
    'CLASS_MAPPING': CLASS_MAPPING
}


print("Config loaded")
for k, v in CONFIG.items():
    if "TOKEN" not in k:
        print(f"{k}: {v}")


# PRODUCTION: Track MLflow state globally
MLFLOW_ACTIVE = False


def init_mlflow():
    """Initialize MLflow with protection - won't crash training if it fails"""
    global MLFLOW_ACTIVE
    
    if not CONFIG["ENABLE_MLFLOW"]:
        print("MLflow disabled")
        return None

    if not CONFIG["DAGSHUB_USERNAME"] or not CONFIG["DAGSHUB_TOKEN"]:
        print("WARNING: DAGSHUB credentials missing - training will continue without MLflow")
        return None

    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = CONFIG["DAGSHUB_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = CONFIG["DAGSHUB_TOKEN"]

        tracking_uri = (
            f"https://dagshub.com/"
            f"{CONFIG['DAGSHUB_USERNAME']}/"
            f"{CONFIG['DAGSHUB_REPO']}.mlflow"
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Radar_MLOps_3Class_Experiment")

        print(f"MLflow tracking at {tracking_uri}")
        MLFLOW_ACTIVE = True
        return tracking_uri
        
    except Exception as e:
        print(f"WARNING: MLflow initialization failed: {e}")
        print("Training will continue without MLflow logging")
        MLFLOW_ACTIVE = False
        return None


def safe_mlflow_log(func):
    """Decorator to protect MLflow logging - won't crash training"""
    def wrapper(*args, **kwargs):
        if MLFLOW_ACTIVE:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"MLflow logging failed (non-critical): {e}")
        return None
    return wrapper


@safe_mlflow_log
def log_params(params):
    mlflow.log_params(params)


@safe_mlflow_log
def log_metrics(metrics, step=None):
    mlflow.log_metrics(metrics, step=step)


@safe_mlflow_log
def log_model(model, name):
    mlflow.pytorch.log_model(model, name)


class RadarDataset(Dataset):
    """
    Production Multimodal Radar Dataset.
    Fails FAST if any file is missing or corrupted.
    """

    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_path = sample["image"]
        radar_path = sample.get("radar")

        # HARD FAIL if image file missing
        if not os.path.exists(image_path):
            raise RuntimeError(f"Missing image file: {image_path}")

        # ---------------- IMAGE ----------------

        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"Corrupted image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        image = image.astype(np.float32) / 255.0

        # ---------------- RADAR ----------------

        # If radar path is missing (allowed), generate a placeholder radar array
        # Do not spam per-sample warnings here — load_dataset aggregates missing counts.
        if radar_path is None or not os.path.exists(radar_path):
            radar = np.zeros((128, 255), dtype=np.float32)
        else:
            try:
                radar_data = loadmat(radar_path)

                # Try common radar field names
                radar = None
                for field in ['data_cube', 'radar', 'range_doppler', 'rd_map', 'data', 'cube']:
                    if field in radar_data:
                        radar = radar_data[field]
                        break
                
                # If still no radar found, try the first non-header field
                if radar is None:
                    # Skip matlab metadata fields
                    data_fields = [k for k in radar_data.keys() if not k.startswith('__')]
                    if data_fields:
                        radar = radar_data[data_fields[0]]

                if radar is None:
                    radar = np.zeros((128, 255), dtype=np.float32)
                else:
                    if radar.ndim > 2:
                        radar = radar[:, :, 0]

                    radar = cv2.resize(radar.astype(np.float32), (255, 128))
                    radar = (radar - radar.min()) / (radar.max() - radar.min() + 1e-8)

            except Exception as e:
                radar = np.zeros((128, 255), dtype=np.float32)

        # ---------------- TENSORS ----------------

        image = torch.from_numpy(image).permute(2, 0, 1)
        radar = torch.from_numpy(radar).unsqueeze(0)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return {
            "image": image,
            "radar": radar,
            "label": label
        }

    
def load_dataset(dat_dir):
    """
    Production dataset loader.
    Fails FAST if dataset is missing or corrupted.
    Only loads classes defined in class_mapping.json
    """

    samples = []
    class_to_idx = {v: int(k) for k, v in CONFIG['CLASS_MAPPING'].items()}
    data_path = Path(dat_dir)

    # HARD FAIL - no demo mode
    if not data_path.exists():
        raise RuntimeError(
            f"\nDATASET NOT FOUND at: {dat_dir}\n"
            "Fix:\n"
            "   1. Run: dvc pull\n"
            "   2. Verify data/raw exists\n"
        )

    print(f"\nLoading dataset from: {data_path.resolve()}")
    print(f"Loading classes: {list(class_to_idx.keys())}")

    missing_classes = []

    # Iterate classes
    for class_name, class_idx in tqdm(class_to_idx.items(), desc="Loading classes"):

        class_path = data_path / class_name

        # Do NOT silently skip classes
        if not class_path.exists():
            missing_classes.append(class_name)
            continue

        for scenario in class_path.iterdir():

            if not scenario.is_dir():
                continue

            # Auto-detect common image and radar folder names (some recordings use images_0 / radar_raw_frame)
            image_candidates = ["images", "images_0", "images_1"]
            radar_candidates = ["radar", "radar_raw_frame", "radar_raw", "radar_frame"]

            images_path = None
            for c in image_candidates:
                p = scenario / c
                if p.exists() and p.is_dir():
                    images_path = p
                    break

            if images_path is None:
                raise RuntimeError(f"Images folder missing in {scenario} (looked for {image_candidates})")

            radar_folder = None
            for c in radar_candidates:
                p = scenario / c
                if p.exists() and p.is_dir():
                    radar_folder = p
                    break

            # Iterate images; map each image to a radar .mat and optional csv if present
            missing_radar_local = 0
            corrupted_radar_local = 0
            csv_present_local = 0
            csv_candidates = ["csv", "annotations", "labels"]

            # helper: search for csv matching the image stem in likely locations
            def find_csv_for_image(img_path):
                # 1) same folder as image
                cand = img_path.with_suffix('.csv')
                if cand.exists():
                    return str(cand)

                # 2) scenario-level csv with same stem
                for c in csv_candidates:
                    p = scenario / c / f"{img_path.stem}.csv"
                    if p.exists():
                        return str(p)

                # 3) scenario root csv filename matching stem
                p = scenario / f"{img_path.stem}.csv"
                if p.exists():
                    return str(p)

                return None

            # helper: check if radar file contains valid data
            def check_radar_validity(radar_path):
                try:
                    radar_data = loadmat(radar_path)
                    # Try common radar field names
                    for field in ['data_cube', 'radar', 'range_doppler', 'rd_map', 'data', 'cube']:
                        if field in radar_data:
                            return True
                    # Check if any non-header field exists
                    data_fields = [k for k in radar_data.keys() if not k.startswith('__')]
                    return len(data_fields) > 0
                except:
                    return False

            for img_file in images_path.glob("*.jpg"):
                rad_file = None
                if radar_folder is not None:
                    # Try exact match first
                    candidate = radar_folder / f"{img_file.stem}.mat"
                    if candidate.exists():
                        if check_radar_validity(candidate):
                            rad_file = str(candidate)
                        else:
                            corrupted_radar_local += 1
                            rad_file = None
                    else:
                        # Handle numbering mismatch: images use 0000000003.jpg, radar uses 000003.mat
                        # Extract numeric part and try shorter format
                        try:
                            img_num = int(img_file.stem)
                            short_name = f"{img_num:06d}.mat"  # 6-digit format
                            candidate_short = radar_folder / short_name
                            if candidate_short.exists():
                                if check_radar_validity(candidate_short):
                                    rad_file = str(candidate_short)
                                else:
                                    corrupted_radar_local += 1
                                    rad_file = None
                            else:
                                missing_radar_local += 1
                        except ValueError:
                            missing_radar_local += 1  # Non-numeric filename, skip
                else:
                    missing_radar_local += 1

                csv_file = find_csv_for_image(img_file)
                if csv_file is not None:
                    csv_present_local += 1

                samples.append({
                    'image': str(img_file),
                    'radar': rad_file,
                    'csv': csv_file,
                    'label': class_idx,
                    'class_name': class_name
                })

            # Aggregate per-scenario missing counts into running counters
            if missing_radar_local > 0:
                if 'missing_radar_count' not in locals():
                    missing_radar_count = 0
                missing_radar_count += missing_radar_local

            if corrupted_radar_local > 0:
                if 'corrupted_radar_count' not in locals():
                    corrupted_radar_count = 0
                corrupted_radar_count += corrupted_radar_local

            if csv_present_local > 0:
                if 'csv_present_count' not in locals():
                    csv_present_count = 0
                csv_present_count += csv_present_local

    # Ensure all classes exist
    if missing_classes:
        raise RuntimeError(
            f"\nMissing class folders:\n{missing_classes}\n"
            "Dataset is incomplete."
        )

    # Ensure dataset is not empty
    if len(samples) == 0:
        raise RuntimeError(
            "Dataset loaded ZERO samples - check DVC remote."
        )

    # Print aggregated missing radar summary
    total_samples = len(samples)
    missing = locals().get('missing_radar_count', 0)
    corrupted = locals().get('corrupted_radar_count', 0)
    if missing > 0 or corrupted > 0:
        print(f"\nRadar file status: {missing} missing, {corrupted} corrupted / {total_samples} total (zero placeholders will be used)")

    # -------- Class distribution --------

    class_counts = {}
    for s in samples:
        class_counts[s['class_name']] = class_counts.get(s['class_name'], 0) + 1

    print("\nClass distribution:")
    for c, n in sorted(class_counts.items()):
        print(f"   {c}: {n}")

    # Guard against stratify crash
    if min(class_counts.values()) < 2:
        raise RuntimeError(
            "\nSome classes have <2 samples.\n"
            "Stratified split will crash.\n"
            "Fix dataset."
        )

    print(f"\nLoaded {len(samples)} samples successfully.\n")

    return samples, class_to_idx


from torch.utils.data import WeightedRandomSampler


def create_dataloaders(samples, class_to_idx):
    """
    Create train/val/test dataloaders with PRODUCTION settings
    IMPROVED: Better num_workers, pin_memory, prefetch_factor
    """
    
    # Get labels for stratified split
    labels = [s['label'] for s in samples]
    indices = np.arange(len(samples))
    
    # Split 70% train, 15% val, 15% test
    train_idx, temp_idx = train_test_split(
        indices, train_size=0.7, stratify=labels, random_state=42
    )
    
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
    
    # PRODUCTION: Better DataLoader settings
    # Optimized for GPU training
    num_workers = 4 if torch.cuda.is_available() else 2
    pin_memory = True if torch.cuda.is_available() else False
    prefetch_factor = 2 if num_workers > 0 else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers, 
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # Calculate class weights for imbalanced dataset handling
    class_counts = {}
    for sample in train_samples:
        class_name = sample['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Calculate inverse frequency weights
    total_samples = len(train_samples)
    class_weights = []
    for i in range(CONFIG['NUM_CLASSES']):
        class_name = CONFIG['CLASS_MAPPING'][str(i)]
        count = class_counts.get(class_name, 1)
        weight = total_samples / (CONFIG['NUM_CLASSES'] * count)
        class_weights.append(weight)
    
    print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"   Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    print(f"   DataLoader workers: {num_workers}, pin_memory: {pin_memory}")
    print(f"   Class weights: {dict(zip(CONFIG['CLASS_MAPPING'].values(), class_weights))}")
    
    return train_loader, val_loader, test_loader, class_weights


class MultimodalModel(nn.Module):
    """
    PRODUCTION: Auto-detects feature dimensions
    Works with ANY backbone without manual feature size updates
    """
    
    def __init__(self, num_classes, backbone_name=None):
        super().__init__()
        
        if backbone_name is None:
            backbone_name = CONFIG['BACKBONE']
        
        # Image encoder (EfficientNet or any timm model)
        self.image_encoder = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        
        # PRODUCTION: AUTO-DETECT feature size (works with any backbone)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])
            img_feat = self.image_encoder(dummy_input).shape[1]
        
        print(f"Auto-detected image feature size: {img_feat}")
        
        # Radar encoder (CNN)
        self.radar_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        rad_feat = 128  # Fixed by architecture
        
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
        self.lstm = nn.LSTM(
            1024, 512, num_layers=2, batch_first=True, 
            dropout=0.4, bidirectional=True
        )
        
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
        fused = torch.cat([img_feat, rad_feat], dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(fused)
        
        return self.classifier(lstm_out.squeeze(1))


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
            
            # ROC-AUC
            if self.all_probs:
                try:
                    all_probs = torch.cat(self.all_probs).numpy()
                    metrics['roc_auc_ovr'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
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


def train_epoch(model, loader, criterion, optimizer, device, metrics):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device, non_blocking=True)
        radars = batch['radar'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
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
            images = batch['image'].to(device, non_blocking=True)
            radars = batch['radar'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(images, radars)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            metrics.update(preds, labels, probs.detach())
            total_loss += loss.item()
            
    return total_loss/len(loader), metrics.compute()


print("Training functions defined")


def train_model(model, train_loader, val_loader, test_loader, class_weights=None):
    """
    PRODUCTION: Main training function with MLflow protection
    Training continues even if MLflow fails
    """
    
    # Ensure output directory exists
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Use class weights if provided (for imbalanced dataset)
    if class_weights is not None:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['LEARNING_RATE'], 
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )
    
    train_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    val_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    run_name = f"run_3class_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # PRODUCTION: MLflow protected - won't crash training if fails
    if MLFLOW_ACTIVE:
        try:
            mlflow.start_run(run_name=run_name)
            log_params(CONFIG)
            print(f"MLflow Run: {run_name}")
        except Exception as e:
            print(f"MLflow start failed (non-critical): {e}")
    
    print(f"Epochs: {CONFIG['EPOCHS']}")
    print("-" * 80)
    
    try:
        for epoch in range(CONFIG['EPOCHS']):
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
            print("-" * 80)
            
            # Train
            train_metrics.reset()
            train_loss, train_m = train_epoch(
                model, train_loader, criterion, optimizer, device, train_metrics
            )
            
            # Validate
            val_metrics.reset()
            val_loss, val_m = validate_epoch(
                model, val_loader, criterion, device, val_metrics
            )
            
            # Scheduler step
            scheduler.step(val_m['accuracy'])
            lr = optimizer.param_groups[0]['lr']
            
            # History
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_m['accuracy'])
            history['val_acc'].append(val_m['accuracy'])
            
            # PRODUCTION: Protected MLflow logging
            log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_m['accuracy'],
                'train_f1': train_m['f1_macro'],
                'val_loss': val_loss,
                'val_accuracy': val_m['accuracy'],
                'val_f1': val_m['f1_macro'],
                'learning_rate': lr
            }, step=epoch)
            
            # Print summary
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_m['accuracy']:.4f} | F1: {train_m['f1_macro']:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_m['accuracy']:.4f} | F1: {val_m['f1_macro']:.4f}")
            
            # Save best model
            if val_m['accuracy'] > best_val_acc:
                best_val_acc = val_m['accuracy']
                checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/best_model_3class.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "val_accuracy": val_m['accuracy'],
                    "config": CONFIG
                }, checkpoint_path)
                
                log_model(model, "best_model")
                print(f"  Saved best model: {best_val_acc:.4f}")
        
        log_model(model, "final_model")
        
    finally:
        # PRODUCTION: Always close MLflow run (even if training crashes)
        if MLFLOW_ACTIVE:
            try:
                mlflow.end_run()
            except:
                pass
    
    return history, best_val_acc


def evaluate_test(test_loader):
    """Evaluation on test set"""
    
    # Load best model
    model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)
    checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/best_model_3class.pth"
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Model not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device, non_blocking=True)
            radars = batch['radar'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(images, radars)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            test_metrics.update(preds, labels, probs.detach())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Compute metrics
    metrics = test_metrics.compute()
    
    # Concatenate tensors
    all_preds_concat = torch.cat(all_preds).numpy()
    all_labels_concat = torch.cat(all_labels).numpy()
    
    # Print results
    print("\n" + "-" * 50)
    print("TEST RESULTS:")
    print("-" * 50)
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision (macro):{metrics['precision_macro']:.4f}")
    print(f"Recall (macro):   {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    
    # Classification report
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(
        all_labels_concat, all_preds_concat,
        target_names=list(CONFIG['CLASS_MAPPING'].values()),
        zero_division=0
    ))
    
    # PRODUCTION: Protected MLflow logging
    if MLFLOW_ACTIVE:
        try:
            with mlflow.start_run(run_name="test_evaluation_3class"):
                log_metrics({f'test_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception as e:
            print(f"MLflow test logging failed (non-critical): {e}")
    
    return metrics


def predict_single(image_path, radar_path, model_path=None):
    """Production inference function"""
    
    model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)
    
    if model_path is None:
        model_path = f"{CONFIG['OUTPUT_DIR']}/best_model_3class.pth"
    
    if os.path.exists(model_path):
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
    
    return {
        'class': class_name, 
        'confidence': conf, 
        'probabilities': probs.cpu().numpy()[0]
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RADAR MLOPS - PRODUCTION PIPELINE")
    print("=" * 80)
    
    # Initialize MLflow (protected - won't crash if fails)
    MLFLOW_URI = init_mlflow()
    
    # Load dataset
    samples, class_to_idx = load_dataset(CONFIG["DATA_DIR"])
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        samples, class_to_idx
    )
    
    # Create model
    model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)
    
    # Train
    history, best_acc = train_model(
        model, train_loader, val_loader, test_loader, class_weights
    )
    
    print("\n" + "=" * 80)
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    print("=" * 80)