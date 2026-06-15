"""
K-Fold Cross Validation utilities for robust model evaluation
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class CrossValidator:
    """
    Manages K-Fold Cross Validation for facial paralysis detection
    """
    def __init__(self, n_folds=5, stratified=True, shuffle=True, random_state=42):
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.fold_results = {}
        self.fold_histories = {}
        
    def get_splitter(self, labels):
        """Get appropriate splitter based on configuration"""
        if self.stratified:
            return StratifiedKFold(
                n_splits=self.n_folds, 
                shuffle=self.shuffle, 
                random_state=self.random_state
            )
        else:
            return KFold(
                n_splits=self.n_folds, 
                shuffle=self.shuffle, 
                random_state=self.random_state
            )
    
    def split_data(self, data_indices, labels):
        """Generate train/val indices for each fold"""
        splitter = self.get_splitter(labels)
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(data_indices, labels)):
            train_indices = [data_indices[i] for i in train_idx]
            val_indices = [data_indices[i] for i in val_idx]
            
            # Verify stratification
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            print(f"\nFold {fold_idx + 1}:")
            print(f"  Train: {len(train_indices)} samples "
                  f"(Stroke: {sum(train_labels)}/{len(train_labels)}, "
                  f"{sum(train_labels)/len(train_labels)*100:.1f}%)")
            print(f"  Val: {len(val_indices)} samples "
                  f"(Stroke: {sum(val_labels)}/{len(val_labels)}, "
                  f"{sum(val_labels)/len(val_labels)*100:.1f}%)")
            
            folds.append({
                'fold': fold_idx,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'train_labels': train_labels,
                'val_labels': val_labels
            })
            
        return folds
    
    def aggregate_fold_results(self, save_dir):
        """
        Aggregate results across all folds with statistical analysis
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect metrics from all folds
        metrics_by_fold = defaultdict(list)
        
        for fold_idx, results in self.fold_results.items():
            for metric_name, value in results.items():
                if isinstance(value, (int, float)):
                    metrics_by_fold[metric_name].append(value)
        
        # Calculate statistics
        summary = {}
        for metric_name, values in metrics_by_fold.items():
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values  # Keep individual values
            }
        
        # Save detailed results
        with open(save_dir / 'fold_results_detailed.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json_safe = {}
            for k, v in summary.items():
                json_safe[k] = {
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'min': float(v['min']),
                    'max': float(v['max']),
                    'median': float(v['median']),
                    'values': [float(x) for x in v['values']]
                }
            json.dump(json_safe, f, indent=2)
        
        # Create visualization
        self._plot_fold_comparison(summary, save_dir)
        self._plot_fold_distributions(summary, save_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("K-FOLD CROSS VALIDATION RESULTS")
        print("="*60)
        print(f"Number of Folds: {self.n_folds}")
        print(f"Stratified: {self.stratified}")
        print("-"*60)
        
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in key_metrics:
            if metric in summary:
                print(f"{metric.upper():12s}: "
                      f"{summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f} "
                      f"[{summary[metric]['min']:.4f} - {summary[metric]['max']:.4f}]")
        
        print("="*60)
        
        return summary
    
    def _plot_fold_comparison(self, summary, save_dir):
        """Bar plot comparing metrics across folds"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        available_metrics = [m for m in metrics if m in summary]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(20, 4))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            values = summary[metric]['values']
            mean_val = summary[metric]['mean']
            std_val = summary[metric]['std']
            
            axes[idx].bar(range(1, len(values) + 1), values, alpha=0.7, color='steelblue')
            axes[idx].axhline(y=mean_val, color='r', linestyle='--', 
                            label=f'Mean: {mean_val:.4f}')
            axes[idx].fill_between(range(1, len(values) + 1), 
                                  mean_val - std_val, mean_val + std_val,
                                  alpha=0.2, color='r', label=f'±1 Std: {std_val:.4f}')
            axes[idx].set_xlabel('Fold')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} across Folds')
            axes[idx].set_ylim([0, 1])
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'fold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fold_distributions(self, summary, save_dir):
        """Box plot showing distribution of metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        available_metrics = [m for m in metrics if m in summary]
        
        data = [summary[m]['values'] for m in available_metrics]
        
        plt.figure(figsize=(10, 6))
        bp = plt.boxplot(data, labels=[m.capitalize() for m in available_metrics], 
                        patch_artist=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(available_metrics)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.ylabel('Score')
        plt.title('Distribution of Metrics across K-Folds', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
        plt.ylim([0, 1])
        
        # Add mean values as text
        for idx, metric in enumerate(available_metrics):
            mean_val = summary[metric]['mean']
            plt.text(idx + 1, mean_val + 0.02, f'{mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'fold_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_ensemble_prediction(self, models, test_loader, device, method='voting'):
        """
        Ensemble prediction using all fold models
        """
        all_predictions = []
        all_probabilities = []
        
        for fold_idx, model in enumerate(models):
            model.eval()
            fold_probs = []
            
            with torch.no_grad():
                for images, graphs, targets, _ in test_loader:
                    images = images.to(device)
                    graphs = graphs.to(device)
                    
                    logits, _, _, _ = model(images, graphs)
                    probs = torch.sigmoid(logits)
                    fold_probs.extend(probs.cpu().numpy())
            
            all_probabilities.append(fold_probs)
            print(f"Fold {fold_idx + 1} predictions collected")
        
        # Aggregate predictions
        all_probabilities = np.array(all_probabilities)  # [n_folds, n_samples]
        
        if method == 'averaging':
            # Average probabilities
            final_probs = np.mean(all_probabilities, axis=0)
            final_preds = (final_probs > 0.5).astype(int)
        elif method == 'voting':
            # Majority voting
            fold_preds = (all_probabilities > 0.5).astype(int)
            final_preds = np.round(np.mean(fold_preds, axis=0)).astype(int)
            final_probs = np.mean(all_probabilities, axis=0)
        elif method == 'weighted':
            # Weight by validation performance (if available)
            weights = [self.fold_results[i].get('f1', 1.0) 
                      for i in range(len(models))]
            weights = np.array(weights) / sum(weights)
            final_probs = np.average(all_probabilities, axis=0, weights=weights)
            final_preds = (final_probs > 0.5).astype(int)
        
        return final_preds, final_probs, all_probabilities


class FoldTrainer:
    """
    Trainer for a single fold in K-Fold CV
    """
    def __init__(self, model, config, device, fold_idx):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.fold_idx = fold_idx
        
        # Same optimizer and scheduler setup as regular trainer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5
            )
        
        from training.losses import CombinedLoss
        from utils.preprocessing import AsymmetryAnalyzer
        
        self.criterion = CombinedLoss()
        self.scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        self.asymmetry_analyzer = AsymmetryAnalyzer()
        
        self.best_val_f1 = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
        
    def train_epoch(self, train_loader):
        """Same as regular trainer but with fold-specific logging"""
        self.model.train()
        total_loss = 0
        
        for images, graphs, targets, landmarks in train_loader:
            images = images.to(self.device)
            graphs = graphs.to(self.device)
            targets = targets.float().unsqueeze(1).to(self.device)
            
            # Calculate asymmetry scores
            asymmetry_scores = []
            for lm in landmarks:
                score, _ = self.asymmetry_analyzer.calculate_asymmetry_index(lm.numpy())
                asymmetry_scores.append(score)
            asymmetry_scores = torch.tensor(asymmetry_scores, dtype=torch.float).to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits, asym_pred, _, _ = self.model(images, graphs)
                    loss, _, _ = self.criterion(logits, asym_pred, targets, asymmetry_scores)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               self.config['training']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, asym_pred, _, _ = self.model(images, graphs)
                loss, _, _ = self.criterion(logits, asym_pred, targets, asymmetry_scores)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config['training']['grad_clip'])
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validation with metrics calculation"""
        self.model.eval()
        total_loss = 0
        all_preds, all_targets, all_probs = [], [], []
        
        with torch.no_grad():
            for images, graphs, targets, landmarks in val_loader:
                images = images.to(self.device)
                graphs = graphs.to(self.device)
                targets = targets.float().unsqueeze(1).to(self.device)
                
                asymmetry_scores = []
                for lm in landmarks:
                    score, _ = self.asymmetry_analyzer.calculate_asymmetry_index(lm.numpy())
                    asymmetry_scores.append(score)
                asymmetry_scores = torch.tensor(asymmetry_scores, dtype=torch.float).to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits, asym_pred, _, _ = self.model(images, graphs)
                        loss, _, _ = self.criterion(logits, asym_pred, targets, asymmetry_scores)
                else:
                    logits, asym_pred, _, _ = self.model(images, graphs)
                    loss, _, _ = self.criterion(logits, asym_pred, targets, asymmetry_scores)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'auc': roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, epochs, patience):
        """Training loop for one fold"""
        print(f"\n{'='*60}")
        print(f"Training Fold {self.fold_idx + 1}")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}, "
                      f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model for this fold
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['f1'])
        
        print(f"Fold {self.fold_idx + 1} Best F1: {self.best_val_f1:.4f}")
        
        return self.best_metrics, self.history