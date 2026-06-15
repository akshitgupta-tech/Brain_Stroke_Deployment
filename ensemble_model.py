"""
Ensemble model combining MobileNetV3 and GCNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mobilenetv3_branch import MobileNetV3Branch
from .gcn_branch import GCNNBranch
from .attention_modules import CrossModalAttention

class FacialParalysisEnsemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Branches
        self.mobilenet = MobileNetV3Branch(
            feature_dim=config['model']['mobilenetv3']['feature_dim'],
            pretrained=config['model']['mobilenetv3']['pretrained'],
            use_attention=config['model']['attention']['use_spatial_attention']
        )
        
        self.gcn = GCNNBranch(
            input_dim=2,
            hidden_dim=config['model']['gcn']['hidden_dim'],
            output_dim=config['model']['gcn']['output_dim'],
            num_layers=config['model']['gcn']['num_layers'],
            dropout=config['model']['gcn']['dropout']
        )
        
        # NOVELTY: Cross-modal attention
        self.cross_attn = CrossModalAttention(
            img_dim=config['model']['mobilenetv3']['feature_dim'],
            graph_dim=config['model']['gcn']['output_dim']
        )
        
        # Fusion dimensions
        img_dim = config['model']['mobilenetv3']['feature_dim']
        graph_dim = config['model']['gcn']['output_dim']
        fusion_dim = img_dim + graph_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config['model']['ensemble']['classifier_hidden']),
            nn.BatchNorm1d(config['model']['ensemble']['classifier_hidden']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['model']['ensemble']['dropout']),
            nn.Linear(config['model']['ensemble']['classifier_hidden'], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(config['model']['ensemble']['dropout'] / 2),
            nn.Linear(64, 1)
        )
        
        # NOVELTY: Asymmetry regression head (auxiliary task)
        self.asymmetry_head = nn.Sequential(
            nn.Linear(graph_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, img, graph_data):
        """
        img: [B, 3, H, W]
        graph_data: PyG Data object
        """
        # Extract features
        img_features = self.mobilenet(img)  # [B, 256]
        graph_features = self.gcn(graph_data)  # [B, 256]
        
        # NOVELTY: Cross-modal attention
        img_features = self.cross_attn(img_features, graph_features)
        
        # Concatenate features
        combined = torch.cat([img_features, graph_features], dim=1)  # [B, 512]
        
        # Classification
        logits = self.classifier(combined)
        
        # NOVELTY: Asymmetry score prediction (auxiliary)
        asymmetry_score = self.asymmetry_head(graph_features)
        
        return logits, asymmetry_score, img_features, graph_features
    
    def get_embeddings(self, img, graph_data):
        """Get feature embeddings for visualization"""
        with torch.no_grad():
            img_features = self.mobilenet(img)
            graph_features = self.gcn(graph_data)
            combined = torch.cat([img_features, graph_features], dim=1)
        return combined