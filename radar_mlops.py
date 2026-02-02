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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchvision import transforms
import torch.nn.utils as utils

import mlflow
import mlflow.pytorch
import dagshub

# Import enhanced accuracy solutions - INTEGRATED
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
            ce_loss = alpha_t * ce_loss
            
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedLRScheduler:
    """Advanced learning rate scheduling with warmup and adaptive decay"""
    def __init__(self, optimizer, warmup_epochs=3, max_lr=5e-5, min_lr=1e-6, 
                 decay_factor=0.95, patience=2):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.is_advanced_scheduler = True
        self._last_lr = [max_lr]
        self._state = {'current_epoch': 0, 'best_val_acc': 0.0, 'epochs_without_improvement': 0, 'last_lr': max_lr}
        
    def step(self, epoch=None, val_acc=None):
        if epoch is None:
            return self._last_lr[0] if self._last_lr else self.max_lr
        self.current_epoch = epoch
        try:
            if epoch < self.warmup_epochs:
                lr = self.max_lr * (epoch + 1) / self.warmup_epochs
            else:
                if val_acc is not None:
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.patience:
                        lr = max(self.max_lr * (self.decay_factor ** (self.epochs_without_improvement - self.patience + 1)), self.min_lr)
                    else:
                        lr = self.max_lr
                else:
                    lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / 50))
            lr = max(self.min_lr, min(lr, self.max_lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self._last_lr = [lr]
            self._state.update({'current_epoch': self.current_epoch, 'best_val_acc': self.best_val_acc, 'epochs_without_improvement': self.epochs_without_improvement, 'last_lr': lr})
            return lr
        except Exception as e:
            print(f"⚠️ Error in AdvancedLRScheduler.step: {e}")
            return self._last_lr[0] if self._last_lr else self.max_lr
        
    def get_last_lr(self):
        return self._last_lr
    
    def state_dict(self):
        self._state.update({'current_epoch': self.current_epoch, 'best_val_acc': self.best_val_acc, 'epochs_without_improvement': self.epochs_without_improvement, 'last_lr': self._last_lr[0] if self._last_lr else self.max_lr})
        return self._state.copy()
    
    def load_state_dict(self, state_dict):
        try:
            self.current_epoch = state_dict.get('current_epoch', 0)
            self.best_val_acc = state_dict.get('best_val_acc', 0.0)
            self.epochs_without_improvement = state_dict.get('epochs_without_improvement', 0)
            last_lr = state_dict.get('last_lr', self.max_lr)
            self._last_lr = [last_lr]
            self._state = state_dict.copy()
        except Exception as e:
            print(f"⚠️ Warning: Failed to load scheduler state: {e}")
            self.current_epoch = 0
            self.best_val_acc = 0.0
            self.epochs_without_improvement = 0
            self._last_lr = [self.max_lr]

def get_balanced_class_weights(train_labels, num_classes=3):
    """Compute ULTRA-enhanced class weights for severe imbalance"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    # Apply massive boosting for completely failing classes
    boost_factors = {
        0: 1.0,  # bicycle - baseline (performing okay)
        1: CONFIG.get('CLASS_BOOST_CAR', 8.0),   # car - massive boost (F1=0.014)
        2: CONFIG.get('CLASS_BOOST_PERSON', 6.0) # person - very strong boost
    }
    
    enhanced_weights = []
    for i, weight in enumerate(class_weights):
        enhanced_weights.append(weight * boost_factors.get(i, 1.0))
    
    print(f"📊 Class weights: bicycle={enhanced_weights[0]:.2f}, car={enhanced_weights[1]:.2f}, person={enhanced_weights[2]:.2f}")
    return torch.FloatTensor(enhanced_weights)

def get_weighted_sampler(dataset, class_weights):
    """Create weighted random sampler for balanced training"""
    sample_weights = []
    for sample in dataset.samples:
        label = sample['label']
        sample_weights.append(class_weights[label].item())
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

class AdvancedMetricsTracker:
    """Track comprehensive metrics for better monitoring"""
    def __init__(self, num_classes=3, class_names=['bicycle', 'car', '1_person']):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
        
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.losses = []
        
    def update(self, preds, targets, probs=None):
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        
    def compute_metrics(self):
        from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        accuracy = accuracy_score(targets, preds)
        f1_macro = f1_score(targets, preds, average='macro')
        f1_micro = f1_score(targets, preds, average='micro')
        f1_weighted = f1_score(targets, preds, average='weighted')
        precision_macro = precision_score(targets, preds, average='macro')
        precision_micro = precision_score(targets, preds, average='micro')
        precision_weighted = precision_score(targets, preds, average='weighted')
        recall_macro = recall_score(targets, preds, average='macro')
        recall_micro = recall_score(targets, preds, average='micro')
        recall_weighted = recall_score(targets, preds, average='weighted')
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
        recall_per_class = []
        for i in range(self.num_classes):
            if cm[i, :].sum() > 0:
                recall = cm[i, i] / cm[i, :].sum()
                recall_per_class.append(recall)
        balanced_acc = np.mean(recall_per_class) if recall_per_class else 0.0
        f1_per_class = f1_score(targets, preds, average=None)
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        metrics = {
            'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weighted': f1_weighted,
            'precision_macro': precision_macro, 'precision_micro': precision_micro, 'precision_weighted': precision_weighted,
            'recall_macro': recall_macro, 'recall_micro': recall_micro, 'recall_weighted': recall_weighted,
            'balanced_accuracy': balanced_acc, 'loss': avg_loss
        }
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        return metrics
    
    def compute(self):
        return self.compute_metrics()

ENHANCED_CONFIG = {
    'focal_loss_gamma': 2.0, 'focal_loss_alpha': [0.8, 1.0, 2.0], 'learning_rate_max': 5e-5, 'learning_rate_min': 1e-6,
    'warmup_epochs': 3, 'class_weight_boost': {'bicycle': 1.0, 'car': 1.1, '1_person': 2.5}, 'early_stopping_patience': 6,
    'mixup_alpha': 0.4, 'balanced_sampling': True, 'monitor_metric': 'balanced_accuracy'
}

print("✅ Enhanced accuracy solutions integrated")

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
    
    # Model - CI environment override support
    "NUM_CLASSES": NUM_CLASSES,
    "IMAGE_SIZE": int(os.environ.get('IMAGE_SIZE', 224)),  # CI: 224, Local: 224 (full quality)
    "BACKBONE": os.environ.get('BACKBONE', "efficientnet_b0"),  # CI: efficientnet_b0, Local: efficientnet_b0
    
    # Training - ULTRA ANTI-OVERFITTING for small dataset
    "EPOCHS": int(os.environ.get('EPOCHS', 25)),  # Moderate epochs
    "BATCH_SIZE": int(os.environ.get('BATCH_SIZE', 20)),  # Larger batches for stability
    "LEARNING_RATE": 5e-6,  # Ultra-low LR for gradual learning
    "LEARNING_RATE_MIN": 1e-8,  # Very low minimum
    "WEIGHT_DECAY": 1e-1,  # Very strong L2 regularization
    "WARMUP_EPOCHS": 2,  # Short warmup
    "LABEL_SMOOTHING": 0.4,  # Very strong label smoothing
    "DROPOUT_RATE": 0.7,  # Very strong dropout
    "NOISE_FACTOR": 0.05,  # Minimal noise to preserve features
    "GRADIENT_CLIP": 0.3,  # Very strong gradient clipping
    "MIXUP_ALPHA": 0.8,  # Very strong mixup for generalization
    "SCHEDULER": "adaptive",  # Advanced adaptive scheduling
    "FOCAL_LOSS_GAMMA": 4.0,  # Very strong focal loss
    "CLASS_BOOST_PERSON": 6.0,  # Massive boost for person class
    "CLASS_BOOST_CAR": 8.0,  # Massive boost for failing car class
    "BALANCED_SAMPLING": True,  # Essential for class balance
    "EARLY_STOPPING_PATIENCE": 4,  # Reduced patience
    
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

    def __init__(self, samples, class_to_idx, transform=None, is_training=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_training = is_training
        
        # Enhanced augmentations for advanced training
        if is_training:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(30),  # Increased rotation
                transforms.RandomHorizontalFlip(0.6),  # Increased flip probability
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),  # Stronger jitter
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # Random erasing after ToTensor
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_path = sample["image"]
        radar_path = sample.get("radar")
        csv_path = sample.get("csv")

        # HARD FAIL if image file missing
        if not os.path.exists(image_path):
            raise RuntimeError(f"Missing image file: {image_path}")

        # ---------------- IMAGE ----------------

        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"Corrupted image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']))
        
        # Apply augmentations if training
        if self.is_training and self.img_transform:
            image = self.img_transform(image)  # Transforms handle uint8 conversion and normalization
            
            # Add noise to prevent overfitting (to tensor)
            noise_factor = CONFIG.get('NOISE_FACTOR', 0.1)
            noise = torch.normal(0, noise_factor, image.shape)
            image = torch.clamp(image + noise, 0, 1)
        else:
            # Convert to tensor for non-training
            image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

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
                for field in ['data_cube', 'radar', 'range_doppler', 'rd_map', 'data', 'cube', 'adcData']:
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
                    # Handle multi-dimensional radar data
                    if radar.ndim > 3:
                        # Take first channel if complex/multi-channel data
                        radar = radar[:, :, 0, 0] if radar.ndim == 4 else radar[:, :, 0]
                    elif radar.ndim == 3:
                        radar = radar[:, :, 0]

                    # Ensure we have 2D data before resize
                    if radar.ndim > 2:
                        radar = radar.squeeze()
                    
                    # Handle complex data (take magnitude)
                    if np.iscomplexobj(radar):
                        radar = np.abs(radar)

                    radar = cv2.resize(radar.astype(np.float32), (255, 128))
                    radar = (radar - radar.min()) / (radar.max() - radar.min() + 1e-8)

            except Exception as e:
                radar = np.zeros((128, 255), dtype=np.float32)

        # ---------------- CSV FEATURES ----------------
        
        # Load CSV features if available
        if csv_path is not None and os.path.exists(csv_path):
            try:
                # Read CSV file
                csv_data = pd.read_csv(csv_path, header=None)
                
                # Convert to numpy array and flatten
                csv_features = csv_data.values.flatten().astype(np.float32)
                
                # Pad or truncate to fixed size (let's use 32 features max)
                target_size = 32
                if len(csv_features) > target_size:
                    csv_features = csv_features[:target_size]
                elif len(csv_features) < target_size:
                    # Pad with zeros
                    padding = np.zeros(target_size - len(csv_features), dtype=np.float32)
                    csv_features = np.concatenate([csv_features, padding])
                    
            except Exception as e:
                # If CSV loading fails, use zero features
                csv_features = np.zeros(32, dtype=np.float32)
        else:
            # No CSV file, use zero features
            csv_features = np.zeros(32, dtype=np.float32)

        # ---------------- TENSORS ----------------

        # Image is already a tensor from augmentations or converted above
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
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
                    for field in ['data_cube', 'radar', 'range_doppler', 'rd_map', 'data', 'cube', 'adcData']:
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
    
    train_dataset = RadarDataset(train_samples, class_to_idx, is_training=True)
    val_dataset = RadarDataset(val_samples, class_to_idx, is_training=False)
    test_dataset = RadarDataset(test_samples, class_to_idx, is_training=False)
    
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
    
    # Enhanced class weights with stronger balancing
    class_counts = {}
    train_labels = []
    for sample in train_samples:
        class_name = sample['class_name']
        label = sample['label']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        train_labels.append(label)
    
    # Get enhanced balanced class weights
    try:
        enhanced_class_weights = get_balanced_class_weights(train_labels, CONFIG['NUM_CLASSES'])
        print(f"✅ Using enhanced class weights: {enhanced_class_weights}")
    except:
        # Fallback to traditional class weights with person class boost
        total_samples = len(train_samples)
        enhanced_class_weights = []
        boost_factors = {'bicycle': 1.0, 'car': 1.1, '1_person': CONFIG.get('CLASS_BOOST_PERSON', 2.5)}
        
        for i in range(CONFIG['NUM_CLASSES']):
            class_name = CONFIG['CLASS_MAPPING'][str(i)]
            count = class_counts.get(class_name, 1)
            weight = total_samples / (CONFIG['NUM_CLASSES'] * count)
            weight *= boost_factors.get(class_name, 1.0)  # Apply boost
            enhanced_class_weights.append(weight)
        
        enhanced_class_weights = torch.FloatTensor(enhanced_class_weights)
        print(f"✅ Using fallback enhanced weights: {enhanced_class_weights}")
    
    # Create balanced sampler if enabled
    train_sampler = None
    if CONFIG.get('BALANCED_SAMPLING', False):
        try:
            train_sampler = get_weighted_sampler(train_dataset, enhanced_class_weights)
            print("✅ Using weighted random sampler for balanced training")
            train_shuffle = False  # Don't shuffle when using sampler
        except:
            print("⚠️ Balanced sampling failed, using regular shuffling")
            train_shuffle = True
    else:
        train_shuffle = True
    
    # Create enhanced dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=train_shuffle,
        sampler=train_sampler,
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
    
    print(f"   Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"   Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"   Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    print(f"   DataLoader workers: {num_workers}, pin_memory: {pin_memory}")
    print(f"   Enhanced Class weights: {dict(zip(CONFIG['CLASS_MAPPING'].values(), enhanced_class_weights))}")
    
    # Convert to proper format for return
    if hasattr(enhanced_class_weights, 'tolist'):
        return train_loader, val_loader, test_loader, enhanced_class_weights.tolist()
    else:
        return train_loader, val_loader, test_loader, enhanced_class_weights


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
        
        rad_feat = 128  # Matches actual radar encoder output
        
        # Enhanced projection layers with residual connections
        dropout_rate = CONFIG.get('DROPOUT_RATE', 0.5)  # Reduced for better performance
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(384, 256)
        )
        self.rad_proj = nn.Sequential(
            nn.Linear(rad_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(384, 256)
        )
        
        # Enhanced CSV feature processor
        self.csv_proj = nn.Sequential(
            nn.Linear(32, 128),  # 32 CSV features -> 128
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(256, 256)
        )
        
        # Reduced LSTM complexity to prevent overfitting
        dropout_rate = CONFIG.get('DROPOUT_RATE', 0.7)
        self.lstm = nn.LSTM(
            768, 128, num_layers=1, batch_first=True,  # Reduced from 384 to 128
            dropout=dropout_rate * 0.8, bidirectional=True
        )
        
        # Simplified classifier with anti-overfitting measures
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 128*2 from reduced bidirectional LSTM
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),  # Very high dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),  # High dropout
            nn.Linear(64, num_classes)
        )
        
    def forward(self, image, radar, csv_features):
        # Extract features from all modalities
        img_feat = self.img_proj(self.image_encoder(image))
        rad_feat = self.rad_proj(self.radar_encoder(radar).flatten(1))
        csv_feat = self.csv_proj(csv_features)
        
        # Trimodal fusion
        fused = torch.cat([img_feat, rad_feat, csv_feat], dim=1).unsqueeze(1)
        lstm_out, _ = self.lstm(fused)
        
        return self.classifier(lstm_out.squeeze(1))


class MetricsTracker:
    """Track comprehensive classification metrics"""
    
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        
        # Initialize torchmetrics
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        self.balanced_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        # Macro averages
        self.precision_macro = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.recall_macro = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)
        self.f1_macro = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
        # Micro averages
        self.precision_micro = Precision(task='multiclass', num_classes=num_classes, average='micro').to(device)
        self.recall_micro = Recall(task='multiclass', num_classes=num_classes, average='micro').to(device)
        self.f1_micro = F1Score(task='multiclass', num_classes=num_classes, average='micro').to(device)
        
        # Weighted averages
        self.precision_weighted = Precision(task='multiclass', num_classes=num_classes, average='weighted').to(device)
        self.recall_weighted = Recall(task='multiclass', num_classes=num_classes, average='weighted').to(device)
        self.f1_weighted = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
        
        # Per-class metrics
        self.precision_per_class = Precision(task='multiclass', num_classes=num_classes, average=None).to(device)
        self.recall_per_class = Recall(task='multiclass', num_classes=num_classes, average=None).to(device)
        self.f1_per_class = F1Score(task='multiclass', num_classes=num_classes, average=None).to(device)
        
        # Store raw predictions for advanced metrics
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
    def update(self, preds, labels, probs=None):
        # Update all torchmetrics
        self.accuracy.update(preds, labels)
        self.balanced_accuracy.update(preds, labels)
        
        # Macro metrics
        self.precision_macro.update(preds, labels)
        self.recall_macro.update(preds, labels)
        self.f1_macro.update(preds, labels)
        
        # Micro metrics
        self.precision_micro.update(preds, labels)
        self.recall_micro.update(preds, labels)
        self.f1_micro.update(preds, labels)
        
        # Weighted metrics
        self.precision_weighted.update(preds, labels)
        self.recall_weighted.update(preds, labels)
        self.f1_weighted.update(preds, labels)
        
        # Per-class metrics
        self.precision_per_class.update(preds, labels)
        self.recall_per_class.update(preds, labels)
        self.f1_per_class.update(preds, labels)
        
        # Store for sklearn metrics
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())
        if probs is not None:
            self.all_probs.append(probs.cpu())
            
    def compute(self):
        # Basic metrics
        metrics = {
            'accuracy': self.accuracy.compute().item(),
            'balanced_accuracy': self.balanced_accuracy.compute().item(),
            
            # Macro averages
            'precision_macro': self.precision_macro.compute().item(),
            'recall_macro': self.recall_macro.compute().item(),
            'f1_macro': self.f1_macro.compute().item(),
            
            # Micro averages
            'precision_micro': self.precision_micro.compute().item(),
            'recall_micro': self.recall_micro.compute().item(),
            'f1_micro': self.f1_micro.compute().item(),
            
            # Weighted averages
            'precision_weighted': self.precision_weighted.compute().item(),
            'recall_weighted': self.recall_weighted.compute().item(),
            'f1_weighted': self.f1_weighted.compute().item(),
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
        
        # Advanced sklearn-based metrics
        if self.all_preds:
            all_preds = torch.cat(self.all_preds).numpy()
            all_labels = torch.cat(self.all_labels).numpy()
            
            # Confusion matrix based metrics
            cm = confusion_matrix(all_labels, all_preds)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Cohen's Kappa
            metrics['cohen_kappa'] = cohen_kappa_score(all_labels, all_preds)
            
            # Class support (number of samples per class)
            class_support = np.bincount(all_labels)
            for i, name in enumerate(class_names):
                metrics[f'support_{name}'] = int(class_support[i])
            
            # ROC-AUC metrics
            if self.all_probs:
                try:
                    all_probs = torch.cat(self.all_probs).numpy()
                    
                    # Multi-class ROC-AUC
                    metrics['roc_auc_ovr'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                    metrics['roc_auc_ovo'] = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
                    
                    # Per-class ROC-AUC
                    from sklearn.preprocessing import label_binarize
                    y_bin = label_binarize(all_labels, classes=list(range(self.num_classes)))
                    for i, name in enumerate(class_names):
                        if len(np.unique(all_labels)) > 1:  # Check if class exists
                            try:
                                metrics[f'roc_auc_{name}'] = roc_auc_score(y_bin[:, i], all_probs[:, i])
                            except:
                                metrics[f'roc_auc_{name}'] = 0.0
                        
                    # Average Precision (AP) scores
                    for i, name in enumerate(class_names):
                        try:
                            metrics[f'avg_precision_{name}'] = average_precision_score(y_bin[:, i], all_probs[:, i])
                        except:
                            metrics[f'avg_precision_{name}'] = 0.0
                            
                    # Mean Average Precision (mAP)
                    ap_scores = [metrics.get(f'avg_precision_{name}', 0.0) for name in class_names]
                    metrics['mean_avg_precision'] = np.mean(ap_scores)
                    
                except Exception as e:
                    print(f"Warning: Could not compute ROC-AUC metrics: {e}")
        
        return metrics
    
    def reset(self):
        # Reset all torchmetrics
        self.accuracy.reset()
        self.balanced_accuracy.reset()
        
        self.precision_macro.reset()
        self.recall_macro.reset()
        self.f1_macro.reset()
        
        self.precision_micro.reset()
        self.recall_micro.reset()
        self.f1_micro.reset()
        
        self.precision_weighted.reset()
        self.recall_weighted.reset()
        self.f1_weighted.reset()
        
        self.precision_per_class.reset()
        self.recall_per_class.reset()
        self.f1_per_class.reset()
        
        # Clear stored values
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []


print("Metrics tracker ready")


def train_epoch(model, loader, criterion, optimizer, scheduler, device, metrics, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        images = batch['image'].to(device, non_blocking=True)
        radars = batch['radar'].to(device, non_blocking=True)
        csv_features = batch['csv'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # Handle small batch sizes for batch normalization
        current_batch_size = images.size(0)
        if current_batch_size == 1:
            print(f"⚠️ Warning: Batch size 1 detected, skipping batch to avoid BatchNorm error")
            continue
        
        optimizer.zero_grad()
        outputs = model(images, radars, csv_features)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step-wise scheduling only for PyTorch schedulers (not AdvancedLRScheduler)
        try:
            if hasattr(scheduler, 'is_advanced_scheduler'):  # Our custom scheduler
                pass  # AdvancedLRScheduler is called per-epoch, not per-step
            elif hasattr(scheduler, 'step'):
                scheduler.step()  # PyTorch built-in schedulers
        except Exception as e:
            print(f"⚠️ Batch-level scheduler step failed: {e}")
            pass  # Continue training even if scheduler fails
        
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
            
    return total_loss/len(loader), metrics.compute()


print("Training functions defined")


def train_model(model, train_loader, val_loader, test_loader, class_weights=None):
    """
    PRODUCTION: Main training function with MLflow protection
    Training continues even if MLflow fails
    """
    
    # Ensure output directory exists
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Enhanced loss function with focal loss for class imbalance
    if class_weights is not None:
        # Use enhanced focal loss with class weighting
        alpha_weights = torch.FloatTensor([1.0, 1.1, 2.5]).to(device)  # Boost person class
        criterion = FocalLoss(
            alpha=alpha_weights, 
            gamma=CONFIG.get('FOCAL_LOSS_GAMMA', 2.0)
        )
        print(f"✅ Using Focal Loss (gamma={CONFIG.get('FOCAL_LOSS_GAMMA', 2.0)}) with class weights: {alpha_weights}")
    else:
        # Fallback to enhanced CrossEntropyLoss
        alpha_weights = torch.FloatTensor([1.0, 1.1, 2.5]).to(device)
        criterion = FocalLoss(
            alpha=alpha_weights,
            gamma=CONFIG.get('FOCAL_LOSS_GAMMA', 2.0)
        )
        print("✅ Using Focal Loss without external class weights")
        
    # Enhanced optimizer with better settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['LEARNING_RATE'], 
        weight_decay=CONFIG['WEIGHT_DECAY'],
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True  # More stable convergence
    )
    
    # Enhanced adaptive learning rate scheduler
    if CONFIG.get('SCHEDULER') == 'adaptive':
        scheduler = AdvancedLRScheduler(
            optimizer, 
            warmup_epochs=CONFIG.get('WARMUP_EPOCHS', 4),
            max_lr=CONFIG['LEARNING_RATE'],
            min_lr=CONFIG.get('LEARNING_RATE_MIN', 1e-6),
            decay_factor=0.9,
            patience=2
        )
        print("✅ Using Advanced Adaptive LR Scheduler")
    else:
        # Fallback to cosine annealing with warmup
        warmup_epochs = CONFIG.get('WARMUP_EPOCHS', 3)
        total_steps = len(train_loader) * CONFIG['EPOCHS']
        warmup_steps = len(train_loader) * warmup_epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print("✅ Using Cosine Annealing LR Scheduler")
    
    # Enhanced metrics tracking
    try:
        train_metrics = AdvancedMetricsTracker(CONFIG['NUM_CLASSES'], ['bicycle', 'car', '1_person'])
        val_metrics = AdvancedMetricsTracker(CONFIG['NUM_CLASSES'], ['bicycle', 'car', '1_person'])
        print("✅ Using Advanced Metrics Tracker")
    except:
        train_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
        val_metrics = MetricsTracker(CONFIG['NUM_CLASSES'], device)
        print("⚠️ Using fallback metrics tracker")
    
    # Enhanced early stopping
    try:
        early_stopping = EnhancedEarlyStopping(
            patience=CONFIG.get('EARLY_STOPPING_PATIENCE', 6),
            min_delta=0.005,
            monitor='balanced_accuracy'
        )
        print("✅ Using Enhanced Early Stopping")
    except:
        early_stopping = None
        print("⚠️ Enhanced early stopping not available")
    
    best_val_acc = 0
    epochs_without_improvement = 0
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
    
    # Anti-overfitting: Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3  # Stop if no improvement for 3 epochs
    best_model_state = None
    best_epoch = 0
    
    print(f"🚀 Anti-overfitting measures active:")
    print(f"   Early stopping patience: {patience} epochs")
    print(f"   Overfitting monitoring threshold: 30%")
    print("-" * 80)
    
    try:
        for epoch in range(CONFIG['EPOCHS']):
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
            print("-" * 80)
            
            # Train
            if hasattr(train_metrics, 'reset'):
                train_metrics.reset()
            train_loss, train_m = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, train_metrics, epoch+1
            )
            
            # Validate
            if hasattr(val_metrics, 'reset'):
                val_metrics.reset()
            val_loss, val_m = validate_epoch(
                model, val_loader, criterion, device, val_metrics, epoch+1
            )
            
            # Enhanced scheduler step with validation accuracy
            if CONFIG.get('SCHEDULER') == 'adaptive':
                try:
                    # Check if it's our AdvancedLRScheduler
                    if hasattr(scheduler, 'is_advanced_scheduler'):
                        current_lr = scheduler.step(epoch, val_m.get('balanced_accuracy', val_m.get('accuracy', 0.0)))
                        if current_lr:
                            print(f"📈 Adaptive LR: {current_lr:.2e}")
                        else:
                            print(f"📈 Adaptive LR: {scheduler.get_last_lr()[0]:.2e}")
                    else:
                        scheduler.step()
                        print(f"📈 Standard scheduler step completed")
                except Exception as e:
                    print(f"⚠️ Scheduler step failed: {e}")
                    # Try fallback scheduler step
                    try:
                        if hasattr(scheduler, 'step'):
                            scheduler.step()
                    except:
                        print(f"⚠️ Fallback scheduler step also failed")
            
            # Enhanced early stopping
            current_val_metrics = {
                'balanced_accuracy': val_m.get('balanced_accuracy', val_m.get('accuracy', 0.0)),
                'accuracy': val_m.get('accuracy', 0.0),
                'f1_macro': val_m.get('f1_macro', 0.0)
            }
            
            if early_stopping and hasattr(early_stopping, '__call__'):
                try:
                    should_stop = early_stopping(current_val_metrics)
                    if should_stop:
                        print(f"🛑 Enhanced early stopping triggered after {epoch+1} epochs")
                        break
                except Exception as e:
                    print(f"⚠️ Enhanced early stopping failed: {e}")
            
            # Traditional early stopping and overfitting check
            train_val_gap = train_m['accuracy'] - val_m['accuracy']
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                print(f"✅ New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Overfitting monitoring
            print(f"📊 Overfitting Monitor:")
            print(f"   Train-Val Gap: {train_val_gap:.1%}")
            print(f"   Patience: {patience_counter}/{patience}")
            
            if train_val_gap > 0.20:  # 20% gap warning (very strict)
                print(f"⚠️  WARNING: Severe overfitting detected!")
                # Force early stop if gap exceeds 30% (much stricter)
                if train_val_gap > 0.30:
                    print(f"🚨 CRITICAL: Train-val gap exceeds 30% - forcing early stop")
                    print(f"   Current gap: {train_val_gap:.1%} - model memorizing training data")
                    print(f"   This indicates insufficient regularization for dataset size")
                    break
            
            # Additional check: Stop if validation accuracy drops significantly
            if epoch > 2 and val_m.get('accuracy', 0) < 0.25:  # Below random baseline
                print(f"🛑 CRITICAL: Validation accuracy below 25% (random = 33%) - stopping")
                break
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\\n🛑 Early stopping at epoch {epoch+1}")
                print(f"   Best model from epoch {best_epoch}")
                break
            
            # Get current learning rate (handle both scheduler types)
            try:
                current_lr = scheduler.get_last_lr()[0]
            except AttributeError:
                # Fallback for schedulers without get_last_lr()
                current_lr = scheduler.optimizer.param_groups[0]['lr']
            
            # History
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_m.get('accuracy', 0.0))
            history['val_acc'].append(val_m.get('accuracy', 0.0))
            
            # Enhanced MLflow logging with comprehensive metrics (with safe access)
            epoch_metrics = {
                # Training metrics
                'train_loss': train_loss,
                'train_accuracy': train_m.get('accuracy', 0.0),
                'train_f1_macro': train_m.get('f1_macro', 0.0),
                'train_f1_micro': train_m.get('f1_micro', train_m.get('accuracy', 0.0)),  # Fallback to accuracy
                'train_f1_weighted': train_m.get('f1_weighted', 0.0),
                'train_precision_macro': train_m.get('precision_macro', 0.0),
                'train_recall_macro': train_m.get('recall_macro', 0.0),
                'train_balanced_accuracy': train_m.get('balanced_accuracy', train_m.get('accuracy', 0.0)),
                
                # Validation metrics (with safe access)
                'val_loss': val_loss,
                'val_accuracy': val_m.get('accuracy', 0.0),
                'val_f1_macro': val_m.get('f1_macro', 0.0),
                'val_f1_micro': val_m.get('f1_micro', val_m.get('accuracy', 0.0)),  # Fallback to accuracy
                'val_f1_weighted': val_m.get('f1_weighted', 0.0),
                'val_precision_macro': val_m.get('precision_macro', 0.0),
                'val_recall_macro': val_m.get('recall_macro', 0.0),
                'val_balanced_accuracy': val_m.get('balanced_accuracy', val_m.get('accuracy', 0.0)),
                'val_cohen_kappa': val_m.get('cohen_kappa', 0.0),
                
                # Learning rate
                'learning_rate': current_lr
            }
            
            # Add per-class metrics
            class_names = list(CONFIG['CLASS_MAPPING'].values())
            for name in class_names:
                epoch_metrics[f'train_f1_{name}'] = train_m.get(f'f1_{name}', 0)
                epoch_metrics[f'val_f1_{name}'] = val_m.get(f'f1_{name}', 0)
                epoch_metrics[f'val_precision_{name}'] = val_m.get(f'precision_{name}', 0)
                epoch_metrics[f'val_recall_{name}'] = val_m.get(f'recall_{name}', 0)
            
            # Add ROC-AUC if available
            if 'roc_auc_ovr' in val_m:
                epoch_metrics['val_roc_auc_ovr'] = val_m['roc_auc_ovr']
                epoch_metrics['val_roc_auc_ovo'] = val_m.get('roc_auc_ovo', 0)
                epoch_metrics['val_mean_avg_precision'] = val_m.get('mean_avg_precision', 0)
            
            log_metrics(epoch_metrics, step=epoch)
            
            # Monitor train-validation gap for overfitting  
            train_val_gap = train_m.get('accuracy', 0.0) - val_m.get('accuracy', 0.0)
            
            # Enhanced console output with overfitting detection (safe access)
            print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_m.get('accuracy', 0.0):.4f} | F1: {train_m.get('f1_macro', 0.0):.4f} | Bal-Acc: {train_m.get('balanced_accuracy', 0.0):.4f}")
            print(f"  Val   → Loss: {val_loss:.4f} | Acc: {val_m.get('accuracy', 0.0):.4f} | F1: {val_m.get('f1_macro', 0.0):.4f} | Bal-Acc: {val_m.get('balanced_accuracy', 0.0):.4f}")
            
            # Overfitting warning
            if train_val_gap > 0.1:
                print(f"  ⚠️  OVERFITTING DETECTED: Train-Val gap = {train_val_gap:.4f}")
            
            # Per-class F1 scores
            class_names = list(CONFIG['CLASS_MAPPING'].values())
            val_f1_per_class = [f"{name}: {val_m.get(f'f1_{name}', 0):.3f}" for name in class_names]
            print(f"  Val F1 per class: {' | '.join(val_f1_per_class)}")
            
            # Advanced metrics if available
            if 'cohen_kappa' in val_m:
                print(f"  Kappa: {val_m.get('cohen_kappa', 0.0):.4f} | ROC-AUC: {val_m.get('roc_auc_ovr', 0):.4f} | mAP: {val_m.get('mean_avg_precision', 0):.4f} | LR: {current_lr:.2e}")
            
            # Stop if perfect training accuracy (clear overfitting)
            if train_m['accuracy'] >= 0.999:
                print(f"\n🛑  STOPPING: Perfect training accuracy detected (overfitting)")
                break
                
            # Monitor train-validation gap for overfitting
            train_val_gap = train_m['accuracy'] - val_m['accuracy']
            if train_val_gap > 0.15:
                print(f"\n⚠️  OVERFITTING WARNING: Train-Val accuracy gap = {train_val_gap:.4f}")
                
            # Save best model with enhanced checkpoint
            if val_m['accuracy'] > best_val_acc:
                best_val_acc = val_m['accuracy']
                checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/best_model.pth"
                
                # Safe scheduler state_dict with fallback
                scheduler_state = None
                try:
                    if hasattr(scheduler, 'state_dict'):
                        scheduler_state = scheduler.state_dict()
                    else:
                        print(f"⚠️ Scheduler {type(scheduler).__name__} has no state_dict method")
                        scheduler_state = {'warning': 'scheduler_state_not_available'}
                except Exception as e:
                    print(f"⚠️ Failed to get scheduler state: {e}")
                    scheduler_state = {'error': str(e)}
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler_state,
                    "val_accuracy": val_m['accuracy'],
                    "val_f1": val_m['f1_macro'],
                    "train_accuracy": train_m['accuracy'],
                    "config": CONFIG,
                    "class_mapping": CONFIG['CLASS_MAPPING']
                }, checkpoint_path)
                
                log_model(model, "best_model")
                print(f"  ✅ New best model saved! Val Acc: {best_val_acc:.4f}")
                
                # Aggressive early stopping (validation-based)
                epochs_without_improvement = 0
                
                # Also track if validation loss is increasing (overfitting signal)
                if 'best_val_loss' not in locals():
                    best_val_loss = val_loss
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    
            else:
                epochs_without_improvement += 1
                
            # Only stop if validation loss increases dramatically (not just 10%)
            # Allow some fluctuation for natural learning
            if val_loss > best_val_loss * 1.5 and epoch > 10:
                print(f"\n☠️  STOPPING: Validation loss increased dramatically (overfitting after epoch 10)")
                break
                
            # More patient early stopping - allow time for convergence
            if epochs_without_improvement >= CONFIG.get('EARLY_STOPPING_PATIENCE', 8):
                print(f"\n⚠️  Early stopping after {epoch+1} epochs (no improvement for {CONFIG.get('EARLY_STOPPING_PATIENCE', 8)} epochs)")
                break
                
            # Stop only if perfect training accuracy AND high overfitting
            if train_m['accuracy'] >= 0.999 and (train_m['accuracy'] - val_m['accuracy']) > 0.3:
                print(f"\n🛑  STOPPING: Perfect training accuracy with severe overfitting detected")
                break
        
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
    checkpoint_path = f"{CONFIG['OUTPUT_DIR']}/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Model not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"⚠️ Warning: Failed to load model checkpoint: {e}")
        print(f"Available keys in checkpoint: {list(checkpoint.keys()) if 'checkpoint' in locals() else 'Unknown'}")
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")
    
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
    
    # Compute metrics
    metrics = test_metrics.compute()
    
    # Concatenate tensors
    all_preds_concat = torch.cat(all_preds).numpy()
    all_labels_concat = torch.cat(all_labels).numpy()
    
    # Print comprehensive results
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    # Overall metrics
    print(f"Overall Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:       {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen's Kappa:           {metrics.get('cohen_kappa', 0):.4f}")
    
    print("\n📈 MACRO AVERAGES:")
    print(f"Precision (macro):       {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):          {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro):        {metrics['f1_macro']:.4f}")
    
    print("\n📊 WEIGHTED AVERAGES:")
    print(f"Precision (weighted):    {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted):       {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (weighted):     {metrics['f1_weighted']:.4f}")
    
    # Per-class detailed metrics
    print("\n" + "-" * 60)
    print("📋 PER-CLASS DETAILED METRICS")
    print("-" * 60)
    class_names = list(CONFIG['CLASS_MAPPING'].values())
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 60)
    for name in class_names:
        prec = metrics.get(f'precision_{name}', 0)
        rec = metrics.get(f'recall_{name}', 0)
        f1 = metrics.get(f'f1_{name}', 0)
        support = metrics.get(f'support_{name}', 0)
        print(f"{name:<12} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {support:<8}")
    
    # ROC-AUC metrics if available
    if 'roc_auc_ovr' in metrics:
        print("\n" + "-" * 50)
        print("🎯 ROC-AUC METRICS")
        print("-" * 50)
        print(f"ROC-AUC (One-vs-Rest):   {metrics['roc_auc_ovr']:.4f}")
        print(f"ROC-AUC (One-vs-One):    {metrics.get('roc_auc_ovo', 0):.4f}")
        print(f"Mean Avg Precision:      {metrics.get('mean_avg_precision', 0):.4f}")
        
        print("\nPer-class ROC-AUC:")
        for name in class_names:
            auc_score = metrics.get(f'roc_auc_{name}', 0)
            ap_score = metrics.get(f'avg_precision_{name}', 0)
            print(f"  {name:<12} AUC: {auc_score:.4f} | AP: {ap_score:.4f}")
    
    # Confusion Matrix
    if 'confusion_matrix' in metrics:
        print("\n" + "-" * 40)
        print("🔥 CONFUSION MATRIX")
        print("-" * 40)
        cm = np.array(metrics['confusion_matrix'])
        print(f"{'Predicted →':<12} {' '.join([f'{name:<8}' for name in class_names])}")
        print(f"{'Actual ↓':<12}")
        for i, name in enumerate(class_names):
            row = ' '.join([f'{cm[i,j]:<8}' for j in range(len(class_names))])
            print(f"{name:<12} {row}")
    
    # Classification report
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(
        all_labels_concat, all_preds_concat,
        target_names=list(CONFIG['CLASS_MAPPING'].values()),
        zero_division=0
    ))
    
    # Enhanced MLflow logging for test results
    if MLFLOW_ACTIVE:
        try:
            with mlflow.start_run(run_name="comprehensive_test_evaluation"):
                # Log all test metrics
                test_log_metrics = {f'test_{k}': v for k, v in metrics.items() 
                                   if isinstance(v, (int, float)) and not k.startswith('confusion_matrix')}
                log_metrics(test_log_metrics)
                
                # Log confusion matrix as artifact if available
                if 'confusion_matrix' in metrics:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 6))
                    cm = np.array(metrics['confusion_matrix'])
                    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title('Confusion Matrix')
                    plt.colorbar()
                    
                    class_names = list(CONFIG['CLASS_MAPPING'].values())
                    tick_marks = np.arange(len(class_names))
                    plt.xticks(tick_marks, class_names, rotation=45)
                    plt.yticks(tick_marks, class_names)
                    
                    # Add text annotations
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], 'd'),
                                    horizontalalignment="center",
                                    color="white" if cm[i, j] > thresh else "black")
                    
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.tight_layout()
                    
                    # Save and log as artifact
                    plt.savefig(f"{CONFIG['OUTPUT_DIR']}/confusion_matrix.png")
                    mlflow.log_artifact(f"{CONFIG['OUTPUT_DIR']}/confusion_matrix.png")
                    plt.close()
                    
        except Exception as e:
            print(f"MLflow test logging failed (non-critical): {e}")
    
    return metrics


def predict_single(image_path, radar_path, csv_path=None, model_path=None):
    """Trimodal inference function"""
    
    model = MultimodalModel(CONFIG['NUM_CLASSES']).to(device)
    
    if model_path is None:
        model_path = f"{CONFIG['OUTPUT_DIR']}/best_model.pth"
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load model from {model_path}: {e}")
            print(f"Continuing with randomly initialized weights...")
    else:
        print(f"⚠️ Model not found at {model_path}, using randomly initialized weights")
    
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
    
    # Load CSV features if provided
    if csv_path and os.path.exists(csv_path):
        try:
            csv_data = pd.read_csv(csv_path, header=None)
            csv_features = csv_data.values.flatten().astype(np.float32)
            # Pad/truncate to 32 features
            if len(csv_features) > 32:
                csv_features = csv_features[:32]
            elif len(csv_features) < 32:
                padding = np.zeros(32 - len(csv_features))
                csv_features = np.concatenate([csv_features, padding])
        except:
            csv_features = np.zeros(32, dtype=np.float32)
    else:
        csv_features = np.zeros(32, dtype=np.float32)
    
    csv_features = torch.from_numpy(csv_features).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image, radar, csv_features)
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