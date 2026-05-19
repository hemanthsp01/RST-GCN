#!/usr/bin/env python3
"""
RST-GCN Model
Graph Convolutional Network for AI text detection using RST trees
Based on PRDetect architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

class GCNConv(nn.Module):
    """Single GCN layer"""
    
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        """
        x: node features [num_nodes, in_features]
        adj: adjacency matrix with self-loops [num_nodes, num_nodes]
        """
        # Normalize adjacency matrix
        # D^-0.5 * A * D^-0.5
        adj = adj + torch.eye(adj.size(0), device=adj.device)  # Add self-loops
        
        # Degree matrix
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Normalized adjacency
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Apply convolution
        out = adj_normalized @ x
        out = self.linear(out)
        
        return out

class RSTGCN(nn.Module):
    """
    RST-GCN Classifier
    2-layer GCN + fully connected + dropout
    """
    
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.5):
        super(RSTGCN, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        x: node features [num_nodes, input_dim]
        adj: adjacency matrix [num_nodes, num_nodes]
        """
        # GCN layer 1
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN layer 2
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling (mean over all nodes)
        x = torch.mean(x, dim=0)
        
        # Classifier
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x

def graph_to_tensors(graph, node_embeddings, node_spans, device):
    """
    Convert NetworkX graph + embeddings to tensors
    
    Returns:
        node_features: [num_nodes, embedding_dim]
        adj_matrix: [num_nodes, num_nodes]
    """
    # Get nodes in consistent order
    nodes = sorted(graph.nodes())
    num_nodes = len(nodes)
    
    # Create node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Build adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            adj[node_to_idx[u], node_to_idx[v]] = 1.0
    
    # Get node features in correct order
    features = []
    for node in nodes:
        # Find this node in our span list
        if node in node_spans:
            idx = node_spans.index(node)
            features.append(node_embeddings[idx])
        else:
            # Node not in embedding list - use zero vector
            features.append(np.zeros(node_embeddings.shape[1]))
    
    features = np.stack(features)
    
    # Convert to tensors
    node_features = torch.tensor(features, dtype=torch.float32, device=device)
    adj_matrix = torch.tensor(adj, dtype=torch.float32, device=device)
    
    return node_features, adj_matrix

def prepare_batch(graphs, embeddings, spans, labels, device):
    batch = []
    for graph, emb, span, label in zip(graphs, embeddings, spans, labels):
        node_feat, adj = graph_to_tensors(graph, emb, span, device)

        # ── Fix: handle both int (0/1) and string ('human'/'machine') ──
        if isinstance(label, str):
            y = torch.tensor([0.0], device=device) if label == 'human' \
                else torch.tensor([1.0], device=device)
        else:
            y = torch.tensor([float(label)], device=device)

        batch.append((node_feat, adj, y))

    return batch