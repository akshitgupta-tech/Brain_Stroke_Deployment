"""
MobileNetV3-Small branch for global feature extraction
"""
import torch
import torch.nn as nn
import torchvision.models as models
from .attention_modules import CBAM

class MobileNetV3Branch(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Load MobileNetV3-Small
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Remove classifier
        self.features = self.backbone.features
        
        # NOVELTY: Add attention modules at correct positions
        # Based on actual architecture:
        # Layer 0: 16, Layer 1: 16, Layer 2: 24, Layer 3: 24
        # Layer 4: 40, Layer 5: 40, Layer 6: 40, Layer 7: 48, Layer 8: 48
        # Layer 9: 96, Layer 10: 96, Layer 11: 96, Layer 12: 576
        if use_attention:
            self.attention1 = CBAM(24)    # After layer 3 (channels: 24)
            self.attention2 = CBAM(48)    # After layer 8 (channels: 48)
            self.attention3 = CBAM(576)   # After layer 12 (channels: 576)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        # Process layers with attention at correct positions
        # Layers 0-3 (output: 24 channels)
        x = self.features[0:4](x)
        if self.use_attention:
            x = self.attention1(x)
        
        # Layers 4-8 (output: 48 channels)
        x = self.features[4:9](x)
        if self.use_attention:
            x = self.attention2(x)
        
        # Layers 9-12 (output: 576 channels)
        x = self.features[9:](x)
        if self.use_attention:
            x = self.attention3(x)
        
        # Global pooling and projection
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_proj(x)
        
        return x