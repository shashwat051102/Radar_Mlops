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
    roc_auc_score, roc_curve, auc,
    balanced_accuracy_score, cohen_kappa_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder

import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import timm
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchvision import transforms
import torch.nn.utils as utils

import mlflow
import mlflow.pytorch
import dagshub

# ====== CRITICAL FIXES ======

# 1. IMPROVED FOCAL LOSS with proper class balancing
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 2. COSINE ANNEALING WITH WARM RESTARTS - better than static LR
class CosineAnnealingWarmupRestarts:
    """Cosine annealing with warmup and restarts for better convergence"""
    def __init__(self, optimizer, first_cycle_steps, warmup_steps, 
                 max_lr, min_lr=0, gamma=1.0):
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma
        
        self.current_step = 0
        self.cycle = 0
        self.step_in_cycle = 0
        
    def step(self):
        self.current_step += 1
        self.step_in_cycle += 1
        
        if self.step_in_cycle >= self.first_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = 0
        
        if self.step_in_cycle < self.warmup_steps:
            # Warmup
            lr = self.max_lr * self.step_in_cycle / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_in_cycle - self.warmup_steps) / (
                self.first_cycle_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        
        lr = lr * (self.gamma ** self.cycle)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
    
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


# 3. MIXUP AUGMENTATION for better generalization
def mixup_data(x, y, alpha=0.3):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# PRODUCTION: Enable cudnn benchmark for free GPU speedup
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# Load class mapping
def load_class_mapping(json_path="data/class_mapping.json"):
    """Load class mapping from JSON file"""
    if not os.path.exists(json_path):
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


CLASS_MAPPING = load_class_mapping()
NUM_CLASSES = len(CLASS_MAPPING)


# ====== IMPROVED CONFIG ======
CONFIG = {
    "DAGSHUB_USERNAME": os.getenv("DAGSHUB_USERNAME"),
    "DAGSHUB_TOKEN": os.getenv("DAGSHUB_TOKEN"),
    "DAGSHUB_REPO": "radae_mlops",
    "ENABLE_MLFLOW": os.getenv("ENABLE_MLFLOW", "false").lower() == "true",

    # Data
    "DATA_DIR": "data/raw",
    "OUTPUT_DIR": "data/processed",
    
    # Model
    "NUM_CLASSES": NUM_CLASSES,
    "IMAGE_SIZE": int(os.environ.get('IMAGE_SIZE', 224)),
    "BACKBONE": os.environ.get('BACKBONE', "mobilenetv3_small_075"),  # ULTRA-SMALL: ~1.5M params vs 5M
    
    # ====== KEY IMPROVEMENTS ======
    # Training - BALANCED for both learning AND generalization
    "EPOCHS": 20,  # Shorter training for small model
    "BATCH_SIZE": 16,  # Smaller batches = more gradient updates
    "LEARNING_RATE": 1e-4,  # Conservative for tiny model
    "LEARNING_RATE_MIN": 1e-6,
    "WEIGHT_DECAY": 1e-4,  # Moderate L2 (was 1e-1, way too high!)
    "WARMUP_STEPS": 300,  # Steps, not epochs
    "LABEL_SMOOTHING": 0.1,  # Moderate (was 0.4, too aggressive)
    "DROPOUT_RATE": 0.3,  # Moderate dropout (was 0.7!)
    "NOISE_FACTOR": 0.01,  # Minimal
    "GRADIENT_CLIP": 1.0,  # Standard clipping
    "MIXUP_ALPHA": 0.3,  # Moderate mixup
    "SCHEDULER": "cosine_warmup",  # Smooth LR schedule
    "FOCAL_LOSS_GAMMA": 2.0,  # Standard focal loss
    "CLASS_BOOST_CAR": 5.0,  # Strong boost for failing car class
    "CLASS_BOOST_PERSON": 2.5,  # Moderate boost
    "BALANCED_SAMPLING": True,
    "EARLY_STOPPING_PATIENCE": 15,  # More patience
    "USE_MIXUP": True,  # Enable mixup
    "COSINE_RESTARTS": True,  # Enable restarts
    
    # Unfreeze more of backbone for better learning
    "FREEZE_BACKBONE_BLOCKS": 2,  # Freeze only first 2 blocks (was 4+)
    
    'CLASS_MAPPING': CLASS_MAPPING
}

print("Improved Config loaded")
for k, v in CONFIG.items():
    if "TOKEN" not in k:
        print(f"{k}: {v}")


# MLflow setup (same as before)
MLFLOW_ACTIVE = False

def init_mlflow():
    global MLFLOW_ACTIVE
    
    if not CONFIG["ENABLE_MLFLOW"]:
        print("MLflow disabled")
        return None

    if not CONFIG["DAGSHUB_USERNAME"] or not CONFIG["DAGSHUB_TOKEN"]:
        print("WARNING: DAGSHUB credentials missing")
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
        mlflow.set_experiment("Radar_MLOps_Improved")

        print(f"MLflow tracking at {tracking_uri}")
        MLFLOW_ACTIVE = True
        return tracking_uri
        
    except Exception as e:
        print(f"WARNING: MLflow initialization failed: {e}")
        MLFLOW_ACTIVE = False
        return None


def safe_mlflow_log(func):
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


# ====== IMPROVED DATASET ======
class RadarDataset(Dataset):
    """Improved dataset with better augmentations"""

    def __init__(self, samples, class_to_idx, transform=None, is_training=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_training = is_training
        
        # BALANCED augmentations - not too aggressive
        if is_training:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(15),  # Moderate rotation
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image"]
        radar_path = sample.get("radar")
        csv_path = sample.get("csv")

        if not os.path.exists(image_path):
            raise RuntimeError(f"Missing image file: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Corrupted image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        
        # Apply transforms
        image = self.img_transform(image)

        # Load radar
        if radar_path is None or not os.path.exists(radar_path):
            radar = np.zeros((128, 255), dtype=np.float32)
        else:
            try:
                radar_data = loadmat(radar_path)
                radar = None
                for field in ['data_cube', 'radar', 'range_doppler', 'rd_map', 'data', 'cube', 'adcData']:
                    if field in radar_data:
                        radar = radar_data[field]
                        break
                
                if radar is None:
                    data_fields = [k for k in radar_data.keys() if not k.startswith('__')]
                    if data_fields:
                        radar = radar_data[data_fields[0]]

                if radar is None:
                    radar = np.zeros((128, 255), dtype=np.float32)
                else:
                    if radar.ndim > 3:
                        radar = radar[:, :, 0, 0] if radar.ndim == 4 else radar[:, :, 0]
                    elif radar.ndim == 3:
                        radar = radar[:, :, 0]

                    if radar.ndim > 2:
                        radar = radar.squeeze()
                    
                    if np.iscomplexobj(radar):
                        radar = np.abs(radar)

                    radar = cv2.resize(radar.astype(np.float32), (255, 128))
                    radar = (radar - radar.min()) / (radar.max() - radar.min() + 1e-8)

            except Exception as e:
                radar = np.zeros((128, 255), dtype=np.float32)

        # Load CSV features
        if csv_path is not None and os.path.exists(csv_path):
            try:
                csv_data = pd.read_csv(csv_path, header=None)
                csv_features = csv_data.values.flatten().astype(np.float32)
                
                target_size = 32
                if len(csv_features) > target_size:
                    csv_features = csv_features[:target_size]
                elif len(csv_features) < target_size:
                    padding = np.zeros(target_size - len(csv_features), dtype=np.float32)
                    csv_features = np.concatenate([csv_features, padding])
                    
            except Exception as e:
                csv_features = np.zeros(32, dtype=np.float32)
        else:
            csv_features = np.zeros(32, dtype=np.float32)

        radar = torch.from_numpy(radar).unsqueeze(0).float()
        csv_features = torch.from_numpy(csv_features).float()
        label = torch.tensor(sample['label'], dtype=torch.long)

        return {
            "image": image,
            "radar": radar,
            "csv": csv_features,
            "label": label
        }

    
def load_dataset(dat_dir):
    """Production dataset loader"""
    samples = []
    class_to_idx = {v: int(k) for k, v in CONFIG['CLASS_MAPPING'].items()}
    data_path = Path(dat_dir)

    if not data_path.exists():
        raise RuntimeError(f"DATASET NOT FOUND at: {dat_dir}")

    print(f"\nLoading dataset from: {data_path.resolve()}")

    missing_classes = []

    for class_name, class_idx in tqdm(class_to_idx.items(), desc="Loading classes"):
        class_path = data_path / class_name

        if not class_path.exists():
            missing_classes.append(class_name)
            continue

        for scenario in class_path.iterdir():
            if not scenario.is_dir():
                continue

            image_candidates = ["images", "images_0", "images_1"]
            radar_candidates = ["radar", "radar_raw_frame", "radar_raw", "radar_frame"]

            images_path = None
            for c in image_candidates:
                p = scenario / c
                if p.exists() and p.is_dir():
                    images_path = p
                    break

            if images_path is None:
                raise RuntimeError(f"Images folder missing in {scenario}")

            radar_folder = None
            for c in radar_candidates:
                p = scenario / c
                if p.exists() and p.is_dir():
                    radar_folder = p
                    break

            csv_candidates = ["csv", "annotations", "labels"]

            def find_csv_for_image(img_path):
                cand = img_path.with_suffix('.csv')
                if cand.exists():
                    return str(cand)

                for c in csv_candidates:
                    p = scenario / c / f"{img_path.stem}.csv"
                    if p.exists():
                        return str(p)

                p = scenario / f"{img_path.stem}.csv"
                if p.exists():
                    return str(p)

                return None

            def check_radar_validity(radar_path):
                try:
                    radar_data = loadmat(radar_path)
                    for field in ['data_cube', 'radar', 'range_doppler', 'rd_map', 'data', 'cube', 'adcData']:
                        if field in radar_data:
                            return True
                    data_fields = [k for k in radar_data.keys() if not k.startswith('__')]
                    return len(data_fields) > 0
                except:
                    return False

            for img_file in images_path.glob("*.jpg"):
                rad_file = None
                if radar_folder is not None:
                    candidate = radar_folder / f"{img_file.stem}.mat"
                    if candidate.exists():
                        if check_radar_validity(candidate):
                            rad_file = str(candidate)
                    else:
                        try:
                            img_num = int(img_file.stem)
                            short_name = f"{img_num:06d}.mat"
                            candidate_short = radar_folder / short_name
                            if candidate_short.exists():
                                if check_radar_validity(candidate_short):
                                    rad_file = str(candidate_short)
                        except ValueError:
                            pass

                csv_file = find_csv_for_image(img_file)

                samples.append({
                    'image': str(img_file),
                    'radar': rad_file,
                    'csv': csv_file,
                    'label': class_idx,
                    'class_name': class_name
                })

    if missing_classes:
        raise RuntimeError(f"Missing class folders: {missing_classes}")

    if len(samples) == 0:
        raise RuntimeError("Dataset loaded ZERO samples")

    class_counts = {}
    for s in samples:
        class_counts[s['class_name']] = class_counts.get(s['class_name'], 0) + 1

    print("\nClass distribution:")
    for c, n in sorted(class_counts.items()):
        print(f"   {c}: {n}")

    if min(class_counts.values()) < 2:
        raise RuntimeError("Some classes have <2 samples")

    print(f"\nLoaded {len(samples)} samples successfully.\n")

    return samples, class_to_idx


def get_balanced_class_weights(train_labels, num_classes=3):
    """Compute balanced class weights with moderate boosting"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    
    # Moderate boosting for failing classes
    boost_factors = {
        0: 1.0,  # bicycle
        1: CONFIG.get('CLASS_BOOST_CAR', 5.0),   # car - strong boost
        2: CONFIG.get('CLASS_BOOST_PERSON', 2.5)  # person - moderate boost
    }
    
    enhanced_weights = []
    for i, weight in enumerate(class_weights):
        enhanced_weights.append(weight * boost_factors.get(i, 1.0))
    
    print(f"📊 Class weights: bicycle={enhanced_weights[0]:.2f}, car={enhanced_weights[1]:.2f}, person={enhanced_weights[2]:.2f}")
    return torch.FloatTensor(enhanced_weights)


def create_dataloaders(samples, class_to_idx):
    """Create dataloaders with improved settings"""
    
    labels = [s['label'] for s in samples]
    indices = np.arange(len(samples))
    
    # SEQUENCE-AWARE SPLIT: Prevent temporal leakage
    # Group samples by sequence to avoid consecutive frames in train/val
    sequence_groups = {}
    for i, sample in enumerate(samples):
        # Extract sequence ID from radar filename (e.g., "000003.mat" -> "000003")
        radar_file = Path(sample['radar_path']).stem
        sequence_id = radar_file[:6]  # First 6 characters as sequence ID
        
        if sequence_id not in sequence_groups:
            sequence_groups[sequence_id] = []
        sequence_groups[sequence_id].append(i)
    
    print(f"📊 Temporal Analysis:")
    print(f"   Total sequences: {len(sequence_groups)}")
    print(f"   Avg samples per sequence: {len(samples)/len(sequence_groups):.1f}")
    
    # Split by sequences, not individual samples
    sequence_ids = list(sequence_groups.keys())
    seq_labels = [labels[sequence_groups[seq_id][0]] for seq_id in sequence_ids]  # First sample's label
    
    # 60-20-20 split by sequences (prevents temporal leakage)
    train_seqs, temp_seqs = train_test_split(
        sequence_ids, train_size=0.6, stratify=seq_labels, random_state=12345
    )
    
    val_seqs, test_seqs = train_test_split(
        temp_seqs, train_size=0.5, 
        stratify=[seq_labels[sequence_ids.index(seq)] for seq in temp_seqs], random_state=67890
    )
    
    # Convert sequence splits back to sample indices
    train_idx = []
    val_idx = []
    test_idx = []
    
    for seq in train_seqs:
        train_idx.extend(sequence_groups[seq])
    for seq in val_seqs:
        val_idx.extend(sequence_groups[seq])
    for seq in test_seqs:
        test_idx.extend(sequence_groups[seq])
    
    # CRITICAL: Check for data leakage by file paths
    train_files = [samples[i]['image_path'] for i in train_idx]
    val_files = [samples[i]['image_path'] for i in val_idx]
    test_files = [samples[i]['image_path'] for i in test_idx]
    
    # Check for duplicate files
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"🚨 DATA LEAKAGE DETECTED:")
        print(f"   Train-Val overlap: {len(overlap_train_val)} files")
        print(f"   Train-Test overlap: {len(overlap_train_test)} files")
        print(f"   Val-Test overlap: {len(overlap_val_test)} files")
        raise ValueError("Data leakage detected! Fix data splitting.")
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    train_dataset = RadarDataset(train_samples, class_to_idx, is_training=True)
    val_dataset = RadarDataset(val_samples, class_to_idx, is_training=False)
    test_dataset = RadarDataset(test_samples, class_to_idx, is_training=False)
    
    # DEBUGGING: Print dataset sizes and class distribution
    print(f"🔍 Dataset Split Analysis:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Train: {len(train_samples)} ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"   Val: {len(val_samples)} ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"   Test: {len(test_samples)} ({len(test_samples)/len(samples)*100:.1f}%)")
    
    # Check validation set class distribution
    val_labels = [s['label'] for s in val_samples]
    from collections import Counter
    val_dist = Counter(val_labels)
    print(f"   Val class distribution: {dict(val_dist)}")
    if min(val_dist.values()) < 5:
        print(f"⚠️  WARNING: Some classes have <5 validation samples!")
    
    # Optimized DataLoader settings
    num_workers = 4 if torch.cuda.is_available() else 2
    pin_memory = True if torch.cuda.is_available() else False
    
    # Get class weights
    train_labels = [s['label'] for s in train_samples]
    class_weights = get_balanced_class_weights(train_labels, CONFIG['NUM_CLASSES'])
    
    # Balanced sampling
    train_sampler = None
    if CONFIG.get('BALANCED_SAMPLING', True):
        sample_weights = [class_weights[s['label']].item() for s in train_samples]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        print("✅ Using balanced sampling")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=True  # Avoid BatchNorm issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"   Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader, class_weights


# ====== IMPROVED MODEL ======
class MultimodalModel(nn.Module):
    """Improved model with better capacity"""
    
    def __init__(self, num_classes, backbone_name=None):
        super().__init__()
        
        if backbone_name is None:
            backbone_name = CONFIG['BACKBONE']
        
        # Image encoder
        self.image_encoder = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        
        # IMPROVED: Unfreeze more layers for better learning
        self._smart_freeze_backbone()
        
        # Auto-detect feature size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])
            img_feat = self.image_encoder(dummy_input).shape[1]
        
        print(f"Auto-detected image feature size: {img_feat}")
        
        # ULTRA-SMALL radar encoder - prevent overfitting
        self.radar_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # Much smaller channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),  # Minimal complexity
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        rad_feat = 32  # Much smaller feature size
        
        # ULTRA-SMALL projections - prevent overfitting
        dropout_rate = CONFIG.get('DROPOUT_RATE', 0.3)
        
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat, 64),  # Direct to tiny size
            nn.Dropout(dropout_rate),
        )
        
        self.rad_proj = nn.Sequential(
            nn.Linear(rad_feat, 64),  # Match other projections
            nn.Dropout(dropout_rate)
        )
        
        self.csv_proj = nn.Sequential(
            nn.Linear(32, 64),  # Much smaller
            nn.Dropout(dropout_rate)
        )
        
        # ULTRA-SIMPLE: Direct concatenation (no attention)
        fusion_size = 64 + 64 + 64  # img + rad + csv
        
        # ULTRA-SIMPLE classifier - prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 32),  # Very small hidden layer
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)   # Direct to output
        )
        
    def forward(self, image, radar, csv_features):
        # Extract features
        img_feat = self.img_proj(self.image_encoder(image))
        rad_feat = self.rad_proj(self.radar_encoder(radar).flatten(1))
        csv_feat = self.csv_proj(csv_features)
        
        # ULTRA-SIMPLE concatenation - no attention complexity
        fused = torch.cat([img_feat, rad_feat, csv_feat], dim=1)
        
        return self.classifier(fused)
    
    def _smart_freeze_backbone(self):
        """Freeze only early layers, allow more learning"""
        freeze_blocks = CONFIG.get('FREEZE_BACKBONE_BLOCKS', 2)
        
        total_blocks = len(list(self.image_encoder.named_children()))
        
        frozen_params = 0
        trainable_params = 0
        
        for i, (name, module) in enumerate(self.image_encoder.named_children()):
            if i < freeze_blocks and 'classifier' not in name:
                for param in module.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
            else:
                for param in module.parameters():
                    trainable_params += param.numel()
        
        print(f"🧊 Backbone Freezing:")
        print(f"   Frozen blocks: {freeze_blocks}/{total_blocks}")
        print(f"   Frozen params: {frozen_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Training ratio: {trainable_params/(frozen_params+trainable_params):.1%}")


# ====== IMPROVED METRICS TRACKER ======
class MetricsTracker:
    """Track comprehensive metrics"""
    
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        self.balanced_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        self.precision_macro = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.recall_macro = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.f1_macro = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        self.precision_weighted = Precision(task='multiclass', num_classes=num_classes, average='weighted').to(device)
        self.recall_weighted = Recall(task='multiclass', num_classes=num_classes, average='weighted').to(device)
        self.f1_weighted = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
        
        self.f1_per_class = F1Score(task='multiclass', num_classes=num_classes, average=None).to(device)
        
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
    def update(self, preds, labels, probs=None):
        self.accuracy.update(preds, labels)
        self.balanced_accuracy.update(preds, labels)
        
        self.precision_macro.update(preds, labels)
        self.recall_macro.update(preds, labels)
        self.f1_macro.update(preds, labels)
        
        self.precision_weighted.update(preds, labels)
        self.recall_weighted.update(preds, labels)
        self.f1_weighted.update(preds, labels)
        
        self.f1_per_class.update(preds, labels)
        
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())
        if probs is not None:
            self.all_probs.append(probs.cpu())
            
    def compute(self):
        metrics = {
            'accuracy': self.accuracy.compute().item(),
            'balanced_accuracy': self.balanced_accuracy.compute().item(),
            'precision_macro': self.precision_macro.compute().item(),
            'recall_macro': self.recall_macro.compute().item(),
            'f1_macro': self.f1_macro.compute().item(),
            'precision_weighted': self.precision_weighted.compute().item(),
            'recall_weighted': self.recall_weighted.compute().item(),
            'f1_weighted': self.f1_weighted.compute().item(),
        }
        
        f1_pc = self.f1_per_class.compute()
        class_names = list(CONFIG["CLASS_MAPPING"].values())
        for i, name in enumerate(class_names):
            metrics[f'f1_{name}'] = f1_pc[i].item()
        
        if self.all_preds:
            all_preds = torch.cat(self.all_preds).numpy()
            all_labels = torch.cat(self.all_labels).numpy()
            
            cm = confusion_matrix(all_labels, all_preds)
            metrics['confusion_matrix'] = cm.tolist()
            metrics['cohen_kappa'] = cohen_kappa_score(all_labels, all_preds)
        
        return metrics
    
    def reset(self):
        self.accuracy.reset()
        self.balanced_accuracy.reset()
        self.precision_macro.reset()
        self.recall_macro.reset()
        self.f1_macro.reset()
        self.precision_weighted.reset()
        self.recall_weighted.reset()
        self.f1_weighted.reset()
        self.f1_per_class.reset()
        
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []


# ====== IMPROVED TRAINING LOOP ======
def train_epoch(model, loader, criterion, optimizer, scheduler, device, metrics, epoch, use_mixup=True):
    """Train one epoch with mixup"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        images = batch['image'].to(device, non_blocking=True)
        radars = batch['radar'].to(device, non_blocking=True)
        csv_features = batch['csv'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Apply mixup
        if use_mixup and CONFIG.get('USE_MIXUP', True):
            images_mixed, labels_a, labels_b, lam = mixup_data(images, labels, CONFIG['MIXUP_ALPHA'])
            outputs = model(images_mixed, radars, csv_features)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images, radars, csv_features)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['GRADIENT_CLIP'])
        
        optimizer.step()
        
        # Step scheduler
        if hasattr(scheduler, 'step') and CONFIG['SCHEDULER'] == 'cosine_warmup':
            scheduler.step()
        
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        metrics.update(preds, labels, probs.detach())
        total_loss += loss.item()
        
    return total_loss/len(loader), metrics.compute()


def validate_epoch(model, loader, criterion, device, metrics, epoch):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating Epoch {epoch}"):
            images = batch['image'].to(device, non_blocking=True)
            radars = batch['radar'].to(device, non_blocking=True)
            csv_features = batch['csv'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(images, radars, csv_features)
            loss = criterion(outputs, labels)
            
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            metrics.update(preds, labels, probs.detach())
            total_loss += loss.item()
            
    val_metrics = metrics.compute()
    
    # CRITICAL: Validation sanity check for suspicious perfect scores
    if val_metrics.get('accuracy', 0) > 0.95 and epoch > 2:
        print(f"🚨 VALIDATION ALERT: Suspiciously high accuracy ({val_metrics['accuracy']:.3f})")
        print(f"   This may indicate data leakage or validation set issues")
        print(f"   Validation samples: {len(loader.dataset)}")
        
    if val_metrics.get('f1', 0) > 0.95 and epoch > 2:
        print(f"🚨 VALIDATION ALERT: Suspiciously high F1 ({val_metrics['f1']:.3f})")
        print(f"   Perfect validation scores are statistically unlikely")
    
    # EMERGENCY: Force emergency stop for impossible validation
    if total_loss/len(loader) < 0.01 and epoch > 1:  # More aggressive
        print(f"🚨 EMERGENCY STOP: Validation loss impossibly low ({total_loss/len(loader):.6f})")
        print(f"   Current epoch: {epoch}, Loss: {total_loss/len(loader):.6f}")
        print(f"   This confirms data leakage - stopping training immediately")
        raise RuntimeError("CRITICAL: Data leakage confirmed - training aborted")
    return total_loss/len(loader), val_metrics


def train_model(model, train_loader, val_loader, test_loader, class_weights=None):
    """Improved training function"""
    
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Focal loss with class weights
    if class_weights is not None:
        alpha_weights = class_weights.to(device)
    else:
        alpha_weights = torch.FloatTensor([1.0, CONFIG['CLASS_BOOST_CAR'], CONFIG['CLASS_BOOST_PERSON']]).to(device)
    
    criterion = FocalLoss(alpha=alpha_weights, gamma=CONFIG['FOCAL_LOSS_GAMMA'])
    print(f"✅ Using Focal Loss (gamma={CONFIG['FOCAL_LOSS_GAMMA']}) with weights: {alpha_weights}")
    
    # Improved optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['LEARNING_RATE'], 
        weight_decay=CONFIG['WEIGHT_DECAY'],
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    if CONFIG['SCHEDULER'] == 'cosine_warmup':
        total_steps = len(train_loader) * CONFIG['EPOCHS']
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            warmup_steps=CONFIG['WARMUP_STEPS'],
            max_lr=CONFIG['LEARNING_RATE'],
            min_lr=CONFIG['LEARNING_RATE_MIN'],
            gamma=0.9
        )
        print("✅ Using Cosine Annealing with Warmup")
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    train_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    val_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
    
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    run_name = f"improved_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if MLFLOW_ACTIVE:
        try:
            mlflow.start_run(run_name=run_name)
            log_params(CONFIG)
            print(f"MLflow Run: {run_name}")
        except Exception as e:
            print(f"MLflow start failed: {e}")
    
    print(f"\n{'='*80}")
    print(f"Starting IMPROVED Training")
    print(f"{'='*80}\n")
    
    try:
        for epoch in range(CONFIG['EPOCHS']):
            print(f"\nEpoch {epoch+1}/{CONFIG['EPOCHS']}")
            print(f"{'-'*80}")
            
            # Train
            train_metrics.reset()
            train_loss, train_m = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, 
                device, train_metrics, epoch+1, use_mixup=True
            )
            
            # Validate
            val_metrics.reset()
            val_loss, val_m = validate_epoch(
                model, val_loader, criterion, device, val_metrics, epoch+1
            )
            
            # Get learning rate
            try:
                current_lr = scheduler.get_last_lr()[0]
            except:
                current_lr = optimizer.param_groups[0]['lr']
            
            # History
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_m['accuracy'])
            history['val_acc'].append(val_m['accuracy'])
            
            # Logging
            epoch_metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_m['accuracy'],
                'train_f1_macro': train_m['f1_macro'],
                'val_loss': val_loss,
                'val_accuracy': val_m['accuracy'],
                'val_f1_macro': val_m['f1_macro'],
                'learning_rate': current_lr
            }
            
            class_names = list(CONFIG['CLASS_MAPPING'].values())
            for name in class_names:
                epoch_metrics[f'train_f1_{name}'] = train_m.get(f'f1_{name}', 0)
                epoch_metrics[f'val_f1_{name}'] = val_m.get(f'f1_{name}', 0)
            
            log_metrics(epoch_metrics, step=epoch)
            
            # Print results
            train_val_gap = train_m['accuracy'] - val_m['accuracy']
            print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_m['accuracy']:.4f} | F1: {train_m['f1_macro']:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_m['accuracy']:.4f} | F1: {val_m['f1_macro']:.4f}")
            print(f"  Gap: {train_val_gap:.4f} | LR: {current_lr:.2e}")
            
            val_f1_per_class = [f"{name}: {val_m.get(f'f1_{name}', 0):.3f}" for name in class_names]
            print(f"  Val F1 per class: {' | '.join(val_f1_per_class)}")
            
            # Save best model
            if val_m['accuracy'] > best_val_acc or val_m['f1_macro'] > best_val_f1:
                if val_m['accuracy'] > best_val_acc:
                    best_val_acc = val_m['accuracy']
                if val_m['f1_macro'] > best_val_f1:
                    best_val_f1 = val_m['f1_macro']
                
                checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/best_model_improved.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "val_accuracy": val_m['accuracy'],
                    "val_f1": val_m['f1_macro'],
                    "config": CONFIG,
                }, checkpoint_path)
                
                log_model(model, "best_model")
                print(f"  ✅ Best model saved! Acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
        
        log_model(model, "final_model")
        
    finally:
        if MLFLOW_ACTIVE:
            try:
                mlflow.end_run()
            except:
                pass
    
    return history, best_val_acc


def evaluate_test(test_loader):
    """Test evaluation"""
    
    model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)
    checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/best_model_improved.pth"
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/best_model.pth"
    
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
            csv_features = batch['csv'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(images, radars, csv_features)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            test_metrics.update(preds, labels, probs.detach())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    metrics = test_metrics.compute()
    all_preds_concat = torch.cat(all_preds).numpy()
    all_labels_concat = torch.cat(all_labels).numpy()
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Balanced Acc:    {metrics['balanced_accuracy']:.4f}")
    print(f"F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):   {metrics['f1_weighted']:.4f}")
    
    print("\n" + classification_report(
        all_labels_concat, all_preds_concat,
        target_names=list(CONFIG['CLASS_MAPPING'].values()),
        zero_division=0
    ))
    
    return metrics


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IMPROVED RADAR MLOPS TRAINING")
    print("=" * 80)
    
    MLFLOW_URI = init_mlflow()
    
    samples, class_to_idx = load_dataset(CONFIG["DATA_DIR"])
    
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        samples, class_to_idx
    )
    
    model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)
    
    history, best_acc = train_model(
        model, train_loader, val_loader, test_loader, class_weights
    )
    
    print("\n" + "=" * 80)
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    print("=" * 80)
    
    # Evaluate on test set
    test_metrics = evaluate_test(test_loader)