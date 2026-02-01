# Advanced Training Solutions for Radar Classification
# Solutions for: Car class performance drop, Train-Val gap reduction, Validation accuracy improvement

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation for advanced regularization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss computation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def advanced_train_epoch(model, train_loader, optimizer, criterion, device, config):
    """Enhanced training epoch with mixup, gradient clipping, and class balancing."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, radar_features, csv_features, labels) in enumerate(train_loader):
        images = images.to(device).float()
        radar_features = radar_features.to(device).float()
        csv_features = csv_features.to(device).float()
        labels = labels.to(device)
        
        # Apply mixup augmentation randomly (50% chance)
        use_mixup = np.random.random() < 0.5
        
        if use_mixup and config.get('MIXUP_ALPHA', 0) > 0:
            mixed_images, labels_a, labels_b, lam = mixup_data(
                images, labels, config['MIXUP_ALPHA']
            )
            outputs = model(mixed_images, radar_features, csv_features)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images, radar_features, csv_features)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if config.get('GRADIENT_CLIP', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['GRADIENT_CLIP'])
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        
        if not use_mixup:
            train_correct += (predicted == labels).sum().item()
        else:
            # Approximate accuracy for mixup
            train_correct += ((lam * (predicted == labels_a).float() + 
                             (1 - lam) * (predicted == labels_b).float()).sum().item())
    
    return train_loss / len(train_loader), 100.0 * train_correct / train_total

def get_advanced_optimizer(model, config):
    """Create optimizer with advanced settings."""
    return optim.AdamW(
        model.parameters(),
        lr=config['LEARNING_RATE'],
        weight_decay=config['WEIGHT_DECAY'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

def get_advanced_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    scheduler_type = config.get('SCHEDULER', 'cosine')
    
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer, T_max=config['EPOCHS'], eta_min=1e-7
        )
    elif scheduler_type == 'step':
        return StepLR(
            optimizer, step_size=7, gamma=0.7
        )
    else:
        return ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

def get_class_weights(dataset, device):
    """Calculate balanced class weights for loss function."""
    class_counts = {}
    for sample in dataset:
        label = sample['class_name']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_name, count in class_counts.items():
        class_weights[class_name] = total_samples / (len(class_counts) * count)
    
    # Convert to tensor (assuming bicycle=0, car=1, 1_person=2)
    weight_tensor = torch.tensor([
        class_weights.get('bicycle', 1.0),
        class_weights.get('car', 1.0),
        class_weights.get('1_person', 1.0)
    ], dtype=torch.float).to(device)
    
    return weight_tensor, class_weights

def advanced_training_loop(model, train_loader, val_loader, device, dataset, config):
    """Complete advanced training loop with all optimizations."""
    
    # Get class weights for balanced training
    weight_tensor, class_weights = get_class_weights(dataset, device)
    print(f"Class weights: {class_weights}")
    
    # Enhanced loss with class weighting and label smoothing
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['LABEL_SMOOTHING'],
        weight=weight_tensor
    )
    
    # Advanced optimizer and scheduler
    optimizer = get_advanced_optimizer(model, config)
    scheduler = get_advanced_scheduler(optimizer, config)
    
    # Early stopping variables
    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 5  # Increased patience for extended training
    
    print(f"Starting advanced training for {config['EPOCHS']} epochs...")
    print("Optimizations: Mixup, Gradient Clipping, Class Weighting, Enhanced Augmentation")
    
    for epoch in range(config['EPOCHS']):
        # Advanced training epoch
        train_loss, train_accuracy = advanced_train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        
        # Standard validation epoch
        val_loss, val_accuracy, val_f1_macro, val_f1_per_class = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calculate improvements
        train_val_gap = train_accuracy - val_accuracy
        
        print(f'Epoch {epoch+1}/{config["EPOCHS"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'  Train-Val Gap: {train_val_gap:.1f}% (Target: <15%)')
        print(f'  Val F1 Macro: {val_f1_macro:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Progress indicators with emojis
        if train_val_gap < 15:
            print(f"  ✅ EXCELLENT: Healthy generalization gap ({train_val_gap:.1f}%)")
        elif train_val_gap < 25:
            print(f"  🟡 GOOD: Improving overfitting control ({train_val_gap:.1f}%)")
        else:
            print(f"  🔴 NEEDS WORK: Still overfitting ({train_val_gap:.1f}%)")
        
        if val_accuracy > 60:
            print(f"  🎯 TARGET ACHIEVED: Validation accuracy above 60% ({val_accuracy:.1f}%)")
        elif val_accuracy > 50:
            print(f"  📈 PROGRESS: Good validation performance ({val_accuracy:.1f}%)")
        
        # Early stopping and best model tracking
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  🚀 NEW BEST: Validation accuracy {val_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        # Update learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_accuracy)
        else:
            scheduler.step()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"🏆 Restored best model: {best_val_accuracy:.2f}% validation accuracy")
    
    return model, best_val_accuracy

def validate_epoch(model, val_loader, criterion, device):
    """Standard validation epoch."""
    from sklearn.metrics import f1_score
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, radar_features, csv_features, labels in val_loader:
            images = images.to(device).float()
            radar_features = radar_features.to(device).float()
            csv_features = csv_features.to(device).float()
            labels = labels.to(device)
            
            outputs = model(images, radar_features, csv_features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_accuracy = 100.0 * val_correct / val_total
    val_f1_macro = f1_score(all_labels, all_preds, average='macro')
    val_f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    return val_loss / len(val_loader), val_accuracy, val_f1_macro, val_f1_per_class