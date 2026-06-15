"""
Custom loss functions including Focal Loss and Asymmetry-aware loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        # Apply alpha per-class (alpha for positive, 1-alpha for negative)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class AsymmetryAwareLoss(nn.Module):
    """
    NOVELTY: Loss function that incorporates asymmetry information
    Higher weight for samples with high asymmetry
    """
    def __init__(self, base_loss='focal', asymmetry_weight=0.5):
        super().__init__()
        self.asymmetry_weight = asymmetry_weight
        
        if base_loss == 'focal':
            self.base_criterion = FocalLoss(reduction='none')
        else:
            self.base_criterion = nn.BCEWithLogitsLoss(reduction='none')
            
    def forward(self, logits, targets, asymmetry_scores=None):
        base_loss = self.base_criterion(logits, targets)
        
        if asymmetry_scores is not None:
            # Weight loss by asymmetry (higher asymmetry = higher weight for positive class)
            scores = asymmetry_scores.squeeze()
            # Ensure non-negative and normalize to stabilize weighting
            scores = torch.clamp(scores, min=0.0)
            if scores.numel() > 1:
                denom = (scores.max() - scores.min()).clamp(min=1e-6)
                scores = (scores - scores.min()) / denom
            weights = 1.0 + self.asymmetry_weight * scores
            weights = weights * targets.squeeze() + (1.0 - targets.squeeze())  # Only weight positive class
            base_loss = base_loss.squeeze() * weights
        
        return base_loss.mean()

class CombinedLoss(nn.Module):
    """Combined classification and asymmetry regression loss"""
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for classification
        self.beta = beta    # Weight for asymmetry regression
        self.class_criterion = AsymmetryAwareLoss()
        self.reg_criterion = nn.MSELoss()
        
    def forward(self, logits, asymmetry_pred, targets, true_asymmetry=None):
        # Use true asymmetry for weighting to avoid negative/unstable weights
        cls_loss = self.class_criterion(logits, targets, true_asymmetry)
        
        if true_asymmetry is not None:
            reg_loss = self.reg_criterion(asymmetry_pred.squeeze(), true_asymmetry)
        else:
            reg_loss = 0
            
        return self.alpha * cls_loss + self.beta * reg_loss, cls_loss, reg_loss
