
"""
Facial Paralysis Detection - ResNet50 + Custom CNN
Single script with K-Fold CV (K=5), 90-10 split, results in ± format
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, matthews_corrcoef,
                             cohen_kappa_score, balanced_accuracy_score)
from PIL import Image
import numpy as np
import json
import warnings
import time
from datetime import datetime
import copy
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# UPDATE THESE PATHS
DATA_DIR = r"C:\Users\Astha Paika\Desktop\facial_paralysis_detection\data\raw"
NON_STROKE_DIR = os.path.join(DATA_DIR, "NonStroke")
STROKE_DIR = os.path.join(DATA_DIR, "Stroke")

# Settings
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
PATIENCE = 2
K_FOLDS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.7
LABEL_SMOOTHING = 0.1

MODEL_TYPE = 'cnn'

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("\n" + "="*70)
print(f"FACIAL PARALYSIS DETECTION - {MODEL_TYPE.upper()}")
print("="*70)

# ==================== DATASET CLASS ====================
class FacialParalysisDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.failed_indices = set()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            if idx not in self.failed_indices:
                print(f"⚠️  Error loading {os.path.basename(img_path)}: {e}")
                self.failed_indices.add(idx)
            blank = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            return blank, torch.tensor(label, dtype=torch.long)

# ==================== DATA LOADING ====================
def load_data():
    image_paths = []
    labels = []

    if not os.path.exists(NON_STROKE_DIR):
        raise FileNotFoundError(f"Directory not found: {NON_STROKE_DIR}")
    if not os.path.exists(STROKE_DIR):
        raise FileNotFoundError(f"Directory not found: {STROKE_DIR}")

    # Load Non-Stroke (Class 0)
    non_stroke_files = [f for f in os.listdir(NON_STROKE_DIR) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    for f in non_stroke_files:
        image_paths.append(os.path.join(NON_STROKE_DIR, f))
        labels.append(0)

    # Load Stroke (Class 1)
    stroke_files = [f for f in os.listdir(STROKE_DIR) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    for f in stroke_files:
        image_paths.append(os.path.join(STROKE_DIR, f))
        labels.append(1)

    print(f"\nDataset Summary:")
    print(f"  Non-Stroke (0): {len(non_stroke_files)} | Stroke (1): {len(stroke_files)}")
    print(f"  Total: {len(image_paths)} images")

    # Verify images
    print("  Verifying images...")
    valid_paths = []
    valid_labels = []
    corrupted = []

    for path, label in zip(image_paths, labels):
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
            valid_labels.append(label)
        except Exception as e:
            corrupted.append(os.path.basename(path))

    if corrupted:
        print(f"  ⚠️  Found {len(corrupted)} corrupted files (excluded)")

    return np.array(valid_paths), np.array(valid_labels)

# ==================== TRANSFORMS ====================
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==================== CUSTOM CNN MODEL ====================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(CustomCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # Global Average Pooling + Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout/2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================== MODEL CREATION ====================
def create_model(model_type='resnet50', freeze_ratio=0.5):
    if model_type == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')

        # Freeze first N layers
        num_layers = len(list(model.children()))
        freeze_until = int(num_layers * freeze_ratio)

        for i, child in enumerate(model.children()):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False

        # Replace classifier
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT/2),
            nn.Linear(128, 2)
        )

    elif model_type == 'cnn':
        model = CustomCNN(num_classes=2, dropout=DROPOUT)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    print(f"\nModel ({model_type}): {trainable:,} trainable, {frozen:,} frozen ({100*trainable/total:.1f}% active)")

    return model.to(DEVICE)

# ==================== CLASS WEIGHTS ====================
def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    weights = weights / weights.sum() * len(class_counts)
    print(f"Class weights: [0]={weights[0]:.3f}, [1]={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ==================== LABEL SMOOTHING LOSS ====================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.weight = weight

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()

# ==================== TRAINING ====================
def train_epoch(model, dataloader, criterion, optimizer, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), 100 * correct / total

# ==================== VALIDATION ====================
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

# ==================== METRICS ====================
def calculate_metrics(y_true, y_pred, y_probs):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    try:
        if y_probs is not None and len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            metrics['roc_auc'] = 0.0
    except:
        metrics['roc_auc'] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    report = classification_report(y_true, y_pred, target_names=['Non-Stroke', 'Stroke'], 
                                   output_dict=True, zero_division=0)
    metrics['per_class'] = report

    return metrics

# ==================== K-FOLD CV ====================
def run_kfold_cv(X, y, class_weights, model_type='resnet50'):
    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    all_train_metrics = []
    all_val_metrics = []

    print("\n" + "="*70)
    print(f"STRATIFIED {K_FOLDS}-FOLD CROSS VALIDATION ({model_type.upper()})")
    print("="*70)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{K_FOLDS}")
        print(f"{'='*70}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {len(X_train)} (Class: {np.bincount(y_train)})")
        print(f"Val: {len(X_val)} (Class: {np.bincount(y_val)})")

        train_dataset = FacialParalysisDataset(X_train, y_train, transform=get_transforms(train=True))
        val_dataset = FacialParalysisDataset(X_val, y_val, transform=get_transforms(train=False))

        # Weighted sampler
        sample_weights = np.array([class_weights[int(label)].cpu().numpy() for label in y_train])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = create_model(model_type=model_type, freeze_ratio=0.5)
        criterion = LabelSmoothingCrossEntropy(weight=class_weights, smoothing=LABEL_SMOOTHING)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        early_stopping = EarlyStopping(patience=PATIENCE)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        fold_train_losses = []
        fold_train_accs = []
        fold_val_losses = []
        fold_val_accs = []

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion)

            scheduler.step(val_loss)

            fold_train_losses.append(train_loss)
            fold_train_accs.append(train_acc)
            fold_val_losses.append(val_loss)
            fold_val_accs.append(val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} - Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"✋ Early stopping at epoch {epoch + 1}")
                break

        # Store train/val metrics for this fold
        all_train_metrics.append({
            'loss': fold_train_losses,
            'acc': fold_train_accs,
            'final_loss': fold_train_losses[-1],
            'final_acc': fold_train_accs[-1]
        })
        all_val_metrics.append({
            'loss': fold_val_losses,
            'acc': fold_val_accs,
            'final_loss': fold_val_losses[-1],
            'final_acc': fold_val_accs[-1]
        })

        # Final validation
        model.load_state_dict(early_stopping.best_model_state)
        _, _, y_true, y_pred, y_probs = validate(model, val_loader, criterion)
        fold_metrics = calculate_metrics(y_true, y_pred, y_probs)
        fold_metrics['fold'] = fold + 1
        fold_results.append(fold_metrics)

        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {fold_metrics['accuracy']:.2%} | F1: {fold_metrics['f1']:.4f} | AUC: {fold_metrics['roc_auc']:.4f}")

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    return fold_results, all_train_metrics, all_val_metrics

# ==================== HOLDOUT TEST ====================
def run_holdout_test(X_train, y_train, X_test, y_test, class_weights, model_type='resnet50'):
    print("\n" + "="*70)
    print(f"FINAL HOLDOUT TEST - 90/10 Split ({model_type.upper()})")
    print("="*70)

    train_dataset = FacialParalysisDataset(X_train, y_train, transform=get_transforms(train=True))
    test_dataset = FacialParalysisDataset(X_test, y_test, transform=get_transforms(train=False))

    sample_weights = np.array([class_weights[int(label)].cpu().numpy() for label in y_train])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = create_model(model_type=model_type, freeze_ratio=0.5)
    criterion = LabelSmoothingCrossEntropy(weight=class_weights, smoothing=LABEL_SMOOTHING)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=PATIENCE)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        test_loss, test_acc, _, _, _ = validate(model, test_loader, criterion)

        scheduler.step(test_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Test: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print(f"✋ Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(early_stopping.best_model_state)
    _, _, y_true, y_pred, y_probs = validate(model, test_loader, criterion)
    final_metrics = calculate_metrics(y_true, y_pred, y_probs)

    # Save model
    model_path = os.path.join(DATA_DIR, f'facial_paralysis_{model_type}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': final_metrics,
        'config': {
            'model_type': model_type,
            'dropout': DROPOUT,
            'weight_decay': WEIGHT_DECAY,
            'label_smoothing': LABEL_SMOOTHING
        }
    }, model_path)
    print(f"\n💾 Model saved: {model_path}")

    return final_metrics, history, model

# ==================== FORMAT RESULTS WITH ± ====================
def format_with_pm(values, metric_name=""):
    """Format results as mean ± std"""
    mean_val = np.mean(values)
    std_val = np.std(values)
    return f"{mean_val:.4f} ± {std_val:.4f}", mean_val, std_val

# ==================== MAIN ====================
def main():
    start_time = time.time()

    print("Loading dataset...")
    X, y = load_data()

    if len(X) == 0:
        print("ERROR: No valid images found!")
        return

    class_weights = calculate_class_weights(y)

    # 90/10 Holdout split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"\nData Split: Train={len(X_train_full)}, Test={len(X_test)}")

    # Run K-Fold CV
    fold_results, all_train_metrics, all_val_metrics = run_kfold_cv(
        X_train_full, y_train_full, class_weights, model_type=MODEL_TYPE
    )

    # CV Summary with ± format
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY (Mean ± Std)")
    print("="*70)

    cv_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'specificity': [], 'roc_auc': [], 'mcc': [],
        'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []
    }

    for result in fold_results:
        for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc', 'mcc']:
            cv_metrics[key].append(result[key])

    # Add train/val metrics from each fold
    for tm, vm in zip(all_train_metrics, all_val_metrics):
        cv_metrics['train_acc'].append(tm['final_acc'] / 100.0)  # Convert to 0-1
        cv_metrics['train_loss'].append(tm['final_loss'])
        cv_metrics['val_acc'].append(vm['final_acc'] / 100.0)
        cv_metrics['val_loss'].append(vm['final_loss'])

    print(f"\n{'Metric':<20} {'Value (Mean ± Std)':<25}")
    print("-" * 50)

    formatted_results = {}
    for key, values in cv_metrics.items():
        formatted, mean_val, std_val = format_with_pm(values, key)
        formatted_results[key] = {'mean': float(mean_val), 'std': float(std_val), 'formatted': formatted}
        display_name = key.replace('_', ' ').title()
        print(f"{display_name:<20} {formatted}")

    # Final holdout test
    final_metrics, history, final_model = run_holdout_test(
        X_train_full, y_train_full, X_test, y_test, class_weights, model_type=MODEL_TYPE
    )

    # Final results with ± from CV
    print("\n" + "="*70)
    print("FINAL TEST SET RESULTS")
    print("="*70)

    # Format test metrics
    test_acc = final_metrics['accuracy']
    test_f1 = final_metrics['f1']
    test_auc = final_metrics['roc_auc']

    # Get CV ± for comparison
    cv_acc_mean = formatted_results['accuracy']['mean']
    cv_acc_std = formatted_results['accuracy']['std']
    cv_f1_mean = formatted_results['f1']['mean']
    cv_f1_std = formatted_results['f1']['std']
    cv_auc_mean = formatted_results['roc_auc']['mean']
    cv_auc_std = formatted_results['roc_auc']['std']

    print(f"\nTest Accuracy:        {test_acc:.4f} (CV: {cv_acc_mean:.4f} ± {cv_acc_std:.4f})")
    print(f"Test F1-Score:        {test_f1:.4f} (CV: {cv_f1_mean:.4f} ± {cv_f1_std:.4f})")
    print(f"Test ROC-AUC:         {test_auc:.4f} (CV: {cv_auc_mean:.4f} ± {cv_auc_std:.4f})")

    print(f"\nDetailed Test Metrics:")
    print(f"  Accuracy:          {final_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {final_metrics['precision']:.4f}")
    print(f"  Recall:            {final_metrics['recall']:.4f}")
    print(f"  F1-Score:          {final_metrics['f1']:.4f}")
    print(f"  Specificity:       {final_metrics['specificity']:.4f}")
    print(f"  MCC:               {final_metrics['mcc']:.4f}")
    print(f"  Cohen Kappa:       {final_metrics['cohen_kappa']:.4f}")
    print(f"  ROC-AUC:           {final_metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(final_metrics['confusion_matrix'])
    print(f"                 Predicted")
    print(f"                 Non-Stroke  Stroke")
    print(f"Actual Non-Stroke    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"       Stroke        {cm[1,0]:4d}      {cm[1,1]:4d}")

    # Save comprehensive results
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'model_type': MODEL_TYPE,
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'k_folds': K_FOLDS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'dropout': DROPOUT,
            'label_smoothing': LABEL_SMOOTHING,
            'device': str(DEVICE)
        },
        'dataset': {
            'total_images': len(X),
            'train_size': len(X_train_full),
            'test_size': len(X_test),
            'class_distribution': {
                'non_stroke': int(np.sum(y == 0)),
                'stroke': int(np.sum(y == 1))
            }
        },
        'cross_validation': {
            'fold_results': [
                {
                    'fold': r['fold'],
                    'accuracy': float(r['accuracy']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'f1': float(r['f1']),
                    'specificity': float(r['specificity']),
                    'mcc': float(r['mcc']),
                    'roc_auc': float(r['roc_auc'])
                } for r in fold_results
            ],
            'summary_mean_std': {
                k: {
                    'mean': float(formatted_results[k]['mean']),
                    'std': float(formatted_results[k]['std']),
                    'formatted': formatted_results[k]['formatted']
                }
                for k in formatted_results.keys()
            }
        },
        'final_test': {
            k: float(v) if isinstance(v, (np.floating, float)) else v 
            for k, v in final_metrics.items() if k not in ['confusion_matrix', 'per_class']
        },
        'comparison': {
            'test_vs_cv_accuracy': {
                'test': float(test_acc),
                'cv_mean': float(cv_acc_mean),
                'cv_std': float(cv_acc_std),
                'difference': float(test_acc - cv_acc_mean)
            },
            'test_vs_cv_f1': {
                'test': float(test_f1),
                'cv_mean': float(cv_f1_mean),
                'cv_std': float(cv_f1_std),
                'difference': float(test_f1 - cv_f1_mean)
            }
        }
    }

    results_path = os.path.join(DATA_DIR, f'training_results_{MODEL_TYPE}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved: {results_path}")

    # Also save a text summary
    summary_path = os.path.join(DATA_DIR, f'summary_{MODEL_TYPE}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"FACIAL PARALYSIS DETECTION - {MODEL_TYPE.upper()} RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write("CROSS-VALIDATION RESULTS (K=5):\n")
        f.write("-"*50 + "\n")
        for k, v in formatted_results.items():
            display_name = k.replace('_', ' ').title()
            f.write(f"{display_name:<20} {v['formatted']}\n")
        f.write("\nFINAL TEST RESULTS (90/10 Split):\n")
        f.write("-"*50 + "\n")
        f.write(f"Accuracy:          {final_metrics['accuracy']:.4f}\n")
        f.write(f"F1-Score:          {final_metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC:           {final_metrics['roc_auc']:.4f}\n")
        f.write(f"Precision:         {final_metrics['precision']:.4f}\n")
        f.write(f"Recall:            {final_metrics['recall']:.4f}\n")
        f.write(f"Specificity:       {final_metrics['specificity']:.4f}\n")
        f.write(f"MCC:               {final_metrics['mcc']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TN: {cm[0,0]}, FP: {cm[0,1]}\n")
        f.write(f"  FN: {cm[1,0]}, TP: {cm[1,1]}\n")

    print(f"💾 Text summary saved: {summary_path}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()