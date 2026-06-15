
# 3. early_stopping.py - Early stopping implementation

import torch
import numpy as np

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = 0
        else:
            if self.mode == 'min':
                improved = score < (self.best_score - self.min_delta)
            else:
                improved = score > (self.best_score + self.min_delta)
            
            if improved:
                self.best_score = score
                self.save_checkpoint(model)
                self.counter = 0
                self.best_epoch = 0
            else:
                self.counter += 1
                self.best_epoch += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                
                if self.counter >= self.patience:
                    self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, model):
        """Save model state"""
        self.best_model_state = model.state_dict().copy()
    
    def restore_best_model(self, model):
        """Restore best model"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print('Restored best model from early stopping')

print("Early stopping module created successfully!")
