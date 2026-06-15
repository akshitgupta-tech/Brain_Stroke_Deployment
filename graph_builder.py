"""
Graph construction for GCN - PyTorch 2.7.1 Compatible
"""
import numpy as np
import torch
from torch_geometric.data import Data

class GraphBuilder:
    def __init__(self, use_delaunay=True):
        self.use_delaunay = use_delaunay
        
    def build_graph(self, landmarks, adj_matrix=None):
        """Convert landmarks to PyTorch Geometric Data object"""
        if landmarks is None:
            # Create dummy graph if no landmarks detected
            x = torch.zeros((68, 2), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index)
        
        # Node features: landmark coordinates (can be extended with features)
        x = torch.tensor(landmarks, dtype=torch.float32)  # [68, 2]
        
        if adj_matrix is not None:
            # Convert adjacency matrix to edge_index
            edge_index = self._adj_to_edge_index(adj_matrix)
        else:
            # Use k-NN graph as fallback
            edge_index = self._knn_graph(landmarks, k=8)
            
        return Data(x=x, edge_index=edge_index)
    
    def _adj_to_edge_index(self, adj):
        """Convert adjacency matrix to edge_index format"""
        rows, cols = np.where(adj > 0)
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        return edge_index
    
    def _knn_graph(self, landmarks, k=8):
        """Create k-nearest neighbors graph"""
        from sklearn.neighbors import kneighbors_graph
        
        adj = kneighbors_graph(landmarks, k, mode='connectivity', 
                              include_self=True)
        return self._adj_to_edge_index(adj.toarray())
    
    def add_edge_features(self, data, landmarks):
        """NOVELTY: Add edge features (distances, angles)"""
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        
        edge_attr = []
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            # Euclidean distance
            dist = np.linalg.norm(landmarks[src] - landmarks[dst])
            
            # Angle
            angle = np.arctan2(landmarks[dst, 1] - landmarks[src, 1],
                              landmarks[dst, 0] - landmarks[src, 0])
            
            edge_attr.append([dist, angle])
            
        data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        return data