"""
Advanced Solutions for Accuracy and Class Imbalance Issues
==========================================================

This module implements comprehensive solutions to address:
1. Person class validation collapse (2.4% F1 → target >40%)
2. Low validation accuracy (35.64% → target >60%)
3. Train-val gap optimization (17.1% → target <15%)
4. Learning rate optimization for better convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses learning on hard examples and reduces easy example impact
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            ce_loss = alpha_t * ce_loss
            
        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdvancedLRScheduler:
    """
    Advanced learning rate scheduling with warmup and adaptive decay
    """
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
        
        # Identifier for compatibility checks
        self.is_advanced_scheduler = True
        self._last_lr = [max_lr]  # Track last learning rate for get_last_lr()
        
    def step(self, epoch=None, val_acc=None):
        # Handle calls without parameters (for compatibility)
        if epoch is None:
            return  # Skip step if no epoch provided
            
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup phase - linear increase
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Adaptive decay based on validation performance
            if val_acc is not None:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Decay if no improvement
                if self.epochs_without_improvement >= self.patience:
                    lr = max(self.max_lr * (self.decay_factor ** (self.epochs_without_improvement - self.patience + 1)), 
                             self.min_lr)
                else:
                    lr = self.max_lr
            else:
                # Fallback to cosine annealing
                lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / 50))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self._last_lr = [lr]  # Update last LR for get_last_lr()
        return lr
        
    def get_last_lr(self):
        """Compatibility method for PyTorch scheduler interface"""
        return self._last_lr


def get_balanced_class_weights(train_labels, num_classes=3):
    """
    Compute enhanced class weights to address severe class imbalance
    """
    # Compute standard class weights
    classes = np.arange(num_classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    
    # Apply additional boosting for severely underperforming classes
    # Based on validation performance, boost person class (class 2) significantly
    boost_factors = {0: 1.0, 1: 1.1, 2: 2.5}  # Strong boost for person class
    
    enhanced_weights = []
    for i, weight in enumerate(class_weights):
        enhanced_weights.append(weight * boost_factors.get(i, 1.0))
    
    return torch.FloatTensor(enhanced_weights)


def get_weighted_sampler(dataset, class_weights):
    """
    Create weighted random sampler for balanced training
    """
    # Get sample weights based on class
    sample_weights = []
    for sample in dataset.samples:
        label = sample['label']
        sample_weights.append(class_weights[label].item())
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


class EnhancedEarlyStopping:
    """
    Enhanced early stopping based on multiple metrics
    """
    def __init__(self, patience=5, min_delta=0.001, monitor='balanced_accuracy'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.epochs_without_improvement = 0
        self.should_stop = False
        
    def __call__(self, val_metrics):
        current_score = val_metrics.get(self.monitor, 0.0)
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            
        return self.should_stop


def calculate_balanced_accuracy(y_true, y_pred, num_classes=3):
    """
    Calculate balanced accuracy for multi-class classification
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recall_per_class = []
    
    for i in range(num_classes):
        if cm[i, :].sum() > 0:
            recall = cm[i, i] / cm[i, :].sum()
            recall_per_class.append(recall)
    
    return np.mean(recall_per_class) if recall_per_class else 0.0


def apply_mixup_advanced(x, y, alpha=0.4):
    """
    Advanced mixup with better lambda distribution
    """
    batch_size = x.size(0)
    
    # Sample lambda from beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    # Ensure lambda is not too extreme
    lam = max(0.1, min(0.9, lam))
    
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def enhanced_validation_strategy():
    """
    Configuration for enhanced validation strategy
    """
    return {
        'stratified_split': True,
        'cross_validation_folds': 3,
        'monitor_metrics': ['balanced_accuracy', 'f1_macro', 'f1_per_class'],
        'early_stopping_metric': 'balanced_accuracy',
        'save_best_model': True
    }


class AdvancedMetricsTracker:
    """
    Track comprehensive metrics for better monitoring
    """
    def __init__(self, num_classes=3, class_names=['bicycle', 'car', '1_person']):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
        
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.losses = []
        
    def update(self, preds, targets, probs=None):
        """Compatible with original MetricsTracker interface"""
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        # Note: loss is not available in this interface, will calculate later if needed
        
    def update_loss(self, loss_value):
        """Separate method to update loss if needed"""
        self.losses.append(loss_value)
        
    def compute_metrics(self):
        from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
        
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, preds)
        
        # F1 scores (macro, micro, weighted)
        f1_macro = f1_score(targets, preds, average='macro')
        f1_micro = f1_score(targets, preds, average='micro')
        f1_weighted = f1_score(targets, preds, average='weighted')
        
        # Precision scores (macro, micro, weighted)
        precision_macro = precision_score(targets, preds, average='macro')
        precision_micro = precision_score(targets, preds, average='micro')
        precision_weighted = precision_score(targets, preds, average='weighted')
        
        # Recall scores (macro, micro, weighted)
        recall_macro = recall_score(targets, preds, average='macro')
        recall_micro = recall_score(targets, preds, average='micro')
        recall_weighted = recall_score(targets, preds, average='weighted')
        
        # Balanced accuracy
        balanced_acc = calculate_balanced_accuracy(targets, preds, self.num_classes)
        
        # Per-class F1 scores
        f1_per_class = f1_score(targets, preds, average=None)
        
        # Average loss (if available)
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'recall_weighted': recall_weighted,
            'balanced_accuracy': balanced_acc,
            'loss': avg_loss
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
            
        return metrics
    
    def compute(self):
        """Compatibility method for training loop"""
        return self.compute_metrics()


# Configuration for enhanced training
ENHANCED_CONFIG = {
    'focal_loss_gamma': 2.0,
    'focal_loss_alpha': [0.8, 1.0, 2.0],  # Boost person class
    'learning_rate_max': 5e-5,  # Higher than current 2.85e-5
    'learning_rate_min': 1e-6,
    'warmup_epochs': 3,
    'class_weight_boost': {'bicycle': 1.0, 'car': 1.1, '1_person': 2.5},
    'early_stopping_patience': 6,
    'mixup_alpha': 0.4,  # Slightly stronger mixup
    'balanced_sampling': True,
    'monitor_metric': 'balanced_accuracy'
}

print("Advanced accuracy solutions loaded:")
print("✅ Focal Loss for class imbalance")
print("✅ Advanced LR scheduling with warmup")
print("✅ Enhanced class weighting (2.5x boost for person class)")
print("✅ Balanced sampling strategy")
print("✅ Comprehensive metrics tracking")
print("✅ Enhanced early stopping")