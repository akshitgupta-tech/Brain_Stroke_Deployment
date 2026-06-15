import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import yaml
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime

from models.ensemble_model import FacialParalysisEnsemble
from training.losses import CombinedLoss
from utils.preprocessing import AsymmetryAnalyzer

import cv2
from PIL import Image, ImageDraw, ImageFont
class MetricsTracker:
    """Track and save all metrics"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train': {'loss': [], 'accuracy': [], 'precision': [], 
                     'recall': [], 'f1': [], 'auc': [], 'pos_rate': []},
            'val': {'loss': [], 'accuracy': [], 'precision': [], 
                   'recall': [], 'f1': [], 'auc': [], 'pos_rate': []}
        }
        
        self.best_metrics = {}
        self.test_metrics = {}
        
    def update(self, phase, metrics):
        """Update history for a phase (train/val)"""
        for key in self.history[phase].keys():
            if key in metrics:
                self.history[phase][key].append(metrics[key])
    
    def save_history(self):
        """Save training history to CSV and JSON"""
        # Save as CSV for easy plotting
        history_df = pd.DataFrame({
            'epoch': range(1, len(self.history['train']['loss']) + 1),
            'train_loss': self.history['train']['loss'],
            'train_accuracy': self.history['train']['accuracy'],
            'train_precision': self.history['train']['precision'],
            'train_recall': self.history['train']['recall'],
            'train_f1': self.history['train']['f1'],
            'train_pos_rate': self.history['train']['pos_rate'],
            'val_loss': self.history['val']['loss'],
            'val_accuracy': self.history['val']['accuracy'],
            'val_precision': self.history['val']['precision'],
            'val_recall': self.history['val']['recall'],
            'val_f1': self.history['val']['f1'],
            'val_auc': self.history['val']['auc'],
            'val_pos_rate': self.history['val']['pos_rate']
        })
        history_df.to_csv(self.save_dir / 'training_history.csv', index=False)
        
        # Save as JSON for detailed info
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def save_best_metrics(self, metrics):
        """Save best validation metrics"""
        self.best_metrics = metrics
        with open(self.save_dir / 'best_val_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def save_test_metrics(self, metrics):
        """Save final test metrics safely to JSON"""
        self.test_metrics = metrics

        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        serializable_metrics = convert(metrics)

        with open(self.save_dir / 'test_metrics.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        with open(self.save_dir / 'test_report.txt', 'w') as f:
            for k, v in serializable_metrics.items():
                f.write(f"{k}: {v}\n")

    # def save_test_metrics(self, metrics):
    #     """Save final test metrics"""
    #     self.test_metrics = metrics
    #     with open(self.save_dir / 'test_metrics.json', 'w') as f:
    #         json.dump(metrics, f, indent=4)
        
    #     # Also save as formatted text
    #     with open(self.save_dir / 'test_report.txt', 'w') as f:
    #         f.write("=" * 60 + "\n")
    #         f.write("FINAL TEST SET EVALUATION REPORT\n")
    #         f.write("=" * 60 + "\n\n")
    #         for key, value in metrics.items():
    #             if isinstance(value, float):
    #                 f.write(f"{key.upper()}: {value:.4f}\n")
    #             else:
    #                 f.write(f"{key.upper()}: {value}\n")

class GraphPlotter:
    """Plot and save all graphs"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')  # Updated for newer matplotlib
        
    def plot_training_curves(self, history):
        """Plot and save training/validation curves"""
        epochs = range(1, len(history['train']['loss']) + 1)
        
        # 1. Loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train']['loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, history['val']['loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Accuracy curves
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train']['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, history['val']['accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. F1 Score curves
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train']['f1'], 'b-', label='Training F1', linewidth=2)
        plt.plot(epochs, history['val']['f1'], 'r-', label='Validation F1', linewidth=2)
        plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'f1_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. All metrics combined
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
        titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 3, idx % 3]
            if metric == 'auc':
                # AUC only for validation
                ax.plot(epochs, history['val'][metric], 'r-', label='Validation', linewidth=2)
            else:
                ax.plot(epochs, history['train'][metric], 'b-', label='Training', linewidth=2)
                ax.plot(epochs, history['val'][metric], 'r-', label='Validation', linewidth=2)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Complete Training Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'all_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, classes, title, filename):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   annot_kws={'size': 14})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, fpr, tpr, auc_score, filename):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, precision, recall, filename):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_asymmetry_distribution(self, asymmetry_scores, labels, filename):
        """Plot asymmetry score distribution"""
        plt.figure(figsize=(10, 6))
        
        no_stroke_scores = [s for s, l in zip(asymmetry_scores, labels) if l == 0]
        stroke_scores = [s for s, l in zip(asymmetry_scores, labels) if l == 1]
        
        plt.hist(no_stroke_scores, bins=30, alpha=0.6, label='No Stroke', 
                color='green', density=True)
        plt.hist(stroke_scores, bins=30, alpha=0.6, label='Stroke', 
                color='red', density=True)
        plt.xlabel('Asymmetry Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Asymmetry Scores', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Create directories
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config['paths']['results']) / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trackers
        self.metrics_tracker = MetricsTracker(self.run_dir / 'metrics')
        self.graph_plotter = GraphPlotter(self.run_dir / 'figures')
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            fused=True if torch.cuda.is_available() else False
        )
        
        # Scheduler
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5
            )
        
        # Loss function
        self.criterion = CombinedLoss()
        
        # Mixed precision
        self.scaler = GradScaler(device='cuda') if config['training']['mixed_precision'] and torch.cuda.is_available() else None
        
        # Asymmetry analyzer
        self.asymmetry_analyzer = AsymmetryAnalyzer()
        
        # Tensorboard
        self.writer = SummaryWriter(self.run_dir / 'logs')
        
        self.best_val_f1 = 0
        self.patience_counter = 0
        
        print(f"Run directory: {self.run_dir}")
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, graphs, targets, landmarks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            graphs = graphs.to(self.device)
            targets = targets.float().unsqueeze(1).to(self.device)
            
            # Calculate true asymmetry scores
            asymmetry_scores = []
            for lm in landmarks:
                score, _ = self.asymmetry_analyzer.calculate_asymmetry_index(lm.numpy())
                asymmetry_scores.append(score)
            asymmetry_scores = torch.tensor(asymmetry_scores, dtype=torch.float).to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward
            if self.scaler:
                with autocast(device_type='cuda'):
                    logits, asym_pred, _, _ = self.model(images, graphs)
                    loss, cls_loss, reg_loss = self.criterion(
                        logits, asym_pred, targets, asymmetry_scores
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, asym_pred, _, _ = self.model(images, graphs)
                loss, cls_loss, reg_loss = self.criterion(
                    logits, asym_pred, targets, asymmetry_scores
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # all_preds.extend(preds.cpu().numpy())
            # all_targets.extend(targets.cpu().numpy())
            # all_probs.extend(probs.cpu().numpy())

            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
                        
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_preds, all_probs)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader, phase='val'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        all_asym_scores = []
        
        with torch.no_grad():
            for images, graphs, targets, landmarks in tqdm(val_loader, desc=f'{phase.capitalize()}'):
                images = images.to(self.device, non_blocking=True)
                graphs = graphs.to(self.device)
                targets = targets.float().unsqueeze(1).to(self.device)
                
                # Calculate asymmetry
                asymmetry_scores = []
                for lm in landmarks:
                    score, _ = self.asymmetry_analyzer.calculate_asymmetry_index(lm.numpy())
                    asymmetry_scores.append(score)
                asymmetry_scores_t = torch.tensor(asymmetry_scores, dtype=torch.float).to(self.device)
                
                if self.scaler:
                    with autocast(device_type='cuda'):
                        logits, asym_pred, _, _ = self.model(images, graphs)
                        loss, _, _ = self.criterion(logits, asym_pred, targets, asymmetry_scores_t)
                else:
                    logits, asym_pred, _, _ = self.model(images, graphs)
                    loss, _, _ = self.criterion(logits, asym_pred, targets, asymmetry_scores_t)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
                all_asym_scores.extend(asymmetry_scores)
        
        metrics = self._calculate_metrics(all_targets, all_preds, all_probs)
        metrics['loss'] = total_loss / len(val_loader)
        
        # Store asymmetry scores for plotting (only for test set)
        if phase == 'test':
            metrics['asymmetry_scores'] = all_asym_scores
            metrics['labels'] = all_targets
        
        return metrics
    
    def _calculate_metrics(self, targets, preds, probs):
        """Calculate all performance metrics"""
        # Convert to numpy arrays and flatten
        targets = np.array(targets).flatten()
        preds = np.array(preds).flatten()
        probs = np.array(probs).flatten()
        
        # Calculate AUC only if we have both classes
        unique_classes = len(set(targets))
        auc_score = 0.0
        if unique_classes > 1:
            try:
                auc_score = roc_auc_score(targets, probs)
            except ValueError:
                auc_score = 0.0
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0),
            'auc': auc_score,
            'pos_rate': float(preds.mean()) if len(preds) > 0 else 0.0
        }
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'pytorch_version': torch.__version__
        }
        
        path = self.run_dir / 'checkpoints'
        path.mkdir(exist_ok=True)
        
        torch.save(checkpoint, path / f'checkpoint_epoch_{epoch}.pt')
        
        if is_best:
            torch.save(checkpoint, path / 'best_model.pt')
            print(f"✓ Saved best model with F1: {metrics['f1']:.4f}")
    
    def log_metrics(self, metrics, epoch, prefix=''):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{prefix}/{key}', value, epoch)
    
    def train(self, train_loader, val_loader):
        """Complete training loop"""
        self.val_loader = val_loader
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print("-" * 60)
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            print(f"TRAIN | Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f} | "
                  f"Prec: {train_metrics['precision']:.4f} | "
                  f"Rec: {train_metrics['recall']:.4f} | "
                  f"F1: {train_metrics['f1']:.4f} | "
                  f"Pos%: {train_metrics['pos_rate']*100:.1f}")
            
            # Validation
            val_metrics = self.validate(val_loader, phase='val')
            print(f"VAL   | Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"Prec: {val_metrics['precision']:.4f} | "
                  f"Rec: {val_metrics['recall']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | "
                  f"Pos%: {val_metrics['pos_rate']*100:.1f}")
            
            # Update history
            self.metrics_tracker.update('train', train_metrics)
            self.metrics_tracker.update('val', val_metrics)
            
            # Log to tensorboard
            self.log_metrics(train_metrics, epoch, 'train')
            self.log_metrics(val_metrics, epoch, 'val')
            
            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.metrics_tracker.save_best_metrics(val_metrics)
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"{'='*60}")
                break
            
            # Step scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['f1'])
        
        # Save training history and plots
        self.metrics_tracker.save_history()
        self.graph_plotter.plot_training_curves(self.metrics_tracker.history)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")
        
        return self.run_dir

    def _find_best_threshold(self, loader):
        """Find probability threshold that maximizes F1 on the given loader"""
        self.model.eval()
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, graphs, targets, _ in loader:
                images = images.to(self.device)
                graphs = graphs.to(self.device)
                
                if self.scaler:
                    with autocast(device_type='cuda'):
                        logits, _, _, _ = self.model(images, graphs)
                else:
                    logits, _, _, _ = self.model(images, graphs)
                
                probs = torch.sigmoid(logits)
                all_targets.extend(targets.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
        
        targets = np.array(all_targets).flatten()
        probs = np.array(all_probs).flatten()
        
        best_thresh = 0.5
        best_f1 = -1.0
        best_acc = -1.0
        
        for t in np.linspace(0.05, 0.95, 91):
            preds = (probs > t).astype(np.float32)
            f1 = f1_score(targets, preds, zero_division=0)
            acc = accuracy_score(targets, preds)
            if (f1 > best_f1) or (f1 == best_f1 and acc > best_acc):
                best_f1 = f1
                best_acc = acc
                best_thresh = t
        
        return float(best_thresh), float(best_f1), float(best_acc)
    
    def evaluate_test_set(self, test_loader):
        """Final evaluation on test set"""
        print(f"\n{'='*60}")
        print(f"FINAL TEST SET EVALUATION")
        print(f"{'='*60}")
        
        # Load best model
        checkpoint_path = self.run_dir / 'checkpoints' / 'best_model.pt'
        if checkpoint_path.exists():
            # checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        # Evaluate
        test_metrics = self.validate(test_loader, phase='test')
        
        # Threshold tuning on validation set
        self.best_threshold, best_f1, best_acc = self._find_best_threshold(
            self.val_loader if hasattr(self, 'val_loader') else test_loader
        )
        print(f"\nBest threshold from validation: {self.best_threshold:.2f} | "
              f"F1: {best_f1:.4f} | Acc: {best_acc:.4f}")
        
        # Get detailed predictions for plots
        all_preds, all_targets, all_probs = self._get_predictions(
            test_loader, threshold=self.best_threshold
        )
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        self.graph_plotter.plot_confusion_matrix(
            cm, ['No Stroke', 'Stroke'], 
            'Test Set Confusion Matrix', 
            'test_confusion_matrix.png'
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        self.graph_plotter.plot_roc_curve(
            fpr, tpr, test_metrics['auc'], 
            'test_roc_curve.png'
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(all_targets, all_probs)
        self.graph_plotter.plot_precision_recall_curve(
            precision, recall, 
            'test_precision_recall_curve.png'
        )
        
        # Asymmetry distribution
        if 'asymmetry_scores' in test_metrics:
            self.graph_plotter.plot_asymmetry_distribution(
                test_metrics['asymmetry_scores'],
                test_metrics['labels'],
                'test_asymmetry_distribution.png'
            )
        
        # Classification report
        report = classification_report(
            all_targets, all_preds, 
            target_names=['No Stroke', 'Stroke'],
            output_dict=True
        )
        
        # Save detailed metrics
        detailed_metrics = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1'],
            'auc_roc': test_metrics['auc'],
            'specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
            'sensitivity': test_metrics['recall'],  # Same as recall
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'total_samples': len(all_targets),
            'stroke_samples': sum(all_targets),
            'no_stroke_samples': len(all_targets) - sum(all_targets)
        }
        
        self.metrics_tracker.save_test_metrics(detailed_metrics)
        
        # Print results
        print(f"\nTEST SET RESULTS:")
        print(f"  Accuracy:    {detailed_metrics['accuracy']:.4f}")
        print(f"  Precision:   {detailed_metrics['precision']:.4f}")
        print(f"  Recall:      {detailed_metrics['recall']:.4f}")
        print(f"  F1-Score:    {detailed_metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:     {detailed_metrics['auc_roc']:.4f}")
        print(f"  Specificity: {detailed_metrics['specificity']:.4f}")
        print(f"  Sensitivity: {detailed_metrics['sensitivity']:.4f}")
        print(f"  Threshold:   {self.best_threshold:.2f}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 No Stroke  Stroke")
        print(f"Actual No Stroke    {cm[0,0]:4d}     {cm[0,1]:4d}")
        print(f"       Stroke       {cm[1,0]:4d}     {cm[1,1]:4d}")
        
        print(f"\n{'='*60}")
        print(f"All results saved to: {self.run_dir}")
        print(f"{'='*60}")
        
        return detailed_metrics
    
    def _get_predictions(self, loader, threshold=0.5):
        """Get all predictions from a loader"""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, graphs, targets, _ in loader:
                images = images.to(self.device)
                graphs = graphs.to(self.device)
                
                if self.scaler:
                    with autocast(device_type='cuda'):
                        logits, _, _, _ = self.model(images, graphs)
                else:
                    logits, _, _, _ = self.model(images, graphs)
                
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
        
        return np.array(all_preds).flatten(), np.array(all_targets).flatten(), np.array(all_probs).flatten()
