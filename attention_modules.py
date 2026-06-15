"""
NOVELTY: Attention mechanisms for improved feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on asymmetric regions"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention (SE-block variant)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class CrossModalAttention(nn.Module):
    """NOVELTY: Cross-modal attention between image and graph features"""
    def __init__(self, img_dim=256, graph_dim=256, hidden_dim=128):
        super().__init__()
        self.img_query = nn.Linear(img_dim, hidden_dim)
        self.graph_key = nn.Linear(graph_dim, hidden_dim)
        self.graph_value = nn.Linear(graph_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        
        self.output_proj = nn.Linear(hidden_dim, img_dim)
        
    def forward(self, img_features, graph_features):
        """
        img_features: [B, img_dim]
        graph_features: [B, graph_dim]
        """
        Q = self.img_query(img_features)  # [B, hidden]
        K = self.graph_key(graph_features)  # [B, hidden]
        V = self.graph_value(graph_features)  # [B, hidden]
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = self.output_proj(out)
        
        # Residual connection
        return img_features + out