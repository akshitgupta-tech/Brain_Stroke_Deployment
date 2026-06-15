
# 2. metrics.py - Comprehensive metrics calculation

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    cohen_kappa_score, matthews_corrcoef
)
import json
import os

class MetricsCalculator:
    """Calculate and store all performance metrics"""
    
    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ['NonStroke', 'Stroke']
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.all_losses = []
    
    def update(self, preds, labels, probs=None, loss=None):
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.detach().cpu().numpy())
        if loss is not None:
            self.all_losses.append(loss)
    
    def compute(self):
        """Compute all metrics"""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_cc': matthews_corrcoef(y_true, y_pred),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, name in enumerate(self.class_names):
            metrics[f'precision_{name}'] = precision_per_class[i]
            metrics[f'recall_{name}'] = recall_per_class[i]
            metrics[f'f1_{name}'] = f1_per_class[i]
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Specificity (True Negative Rate)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROC-AUC (if probabilities available)
        if len(self.all_probs) > 0:
            y_probs = np.array(self.all_probs)
            try:
                if self.num_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
                    metrics['average_precision'] = average_precision_score(y_true, y_probs)
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            except:
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        # Average loss
        if self.all_losses:
            metrics['avg_loss'] = np.mean(self.all_losses)
        
        return metrics
    
    def get_classification_report(self):
        """Get detailed classification report"""
        return classification_report(
            self.all_labels, 
            self.all_preds, 
            target_names=self.class_names,
            output_dict=True
        )
    
    def save_metrics(self, save_path, fold=None):
        """Save metrics to JSON file"""
        metrics = self.compute()
        report = self.get_classification_report()
        
        output = {
            'fold': fold,
            'metrics': metrics,
            'classification_report': report,
            'timestamp': str(datetime.now())
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        return output

print("Metrics calculator created successfully!")
