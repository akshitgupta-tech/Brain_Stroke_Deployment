import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNNBranch(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=256, 
                 num_layers=2, dropout=0.3, use_edge_attr=False):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        # Last layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.bns.append(nn.BatchNorm1d(output_dim))
        
        self.dropout = dropout
        
        # NOVELTY: Graph attention for important landmarks (custom implementation)
        self.graph_attention = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, data):
        """
        data: PyTorch Geometric Data object
          - x: Node features [num_nodes, feature_dim]
          - edge_index: Graph connectivity [2, num_edges]
          - batch: Batch assignment [num_nodes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection if dimensions match
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        # NOVELTY: Attention-based pooling instead of mean pooling
        attn_weights = self.graph_attention(x)  # [num_nodes, 1]
        attn_weights = torch.exp(attn_weights)
        
        # Normalize within each graph
        batch_size = batch.max().item() + 1
        attn_sum = torch.zeros(batch_size, device=x.device)
        attn_sum.scatter_add_(0, batch, attn_weights.squeeze())
        attn_weights = attn_weights / (attn_sum[batch].unsqueeze(1) + 1e-8)
        
        # Weighted sum pooling
        x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)
        x_pooled.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(1)), 
                             x * attn_weights)
        
        return x_pooled