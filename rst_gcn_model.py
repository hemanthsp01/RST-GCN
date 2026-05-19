#!/usr/bin/env python3
"""
RST-GCN Model — 3-Class Version
Graph Convolutional Network for AI text detection using RST trees

Classes:
    0 → human
    1 → ai_original
    2 → ai_paraphrased
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


# ============================================================
# GCN Layer
# ============================================================

class GCNConv(nn.Module):
    """Single GCN layer"""

    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x:   node features  [num_nodes, in_features]
        adj: adjacency matrix [num_nodes, num_nodes]
        """
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)

        # D^-0.5 * A * D^-0.5 normalisation
        d             = adj.sum(1)
        d_inv_sqrt    = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_norm      = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        out = adj_norm @ x
        out = self.linear(out)
        return out


# ============================================================
# RST-GCN Classifier  (3-class)
# ============================================================

class RSTGCN(nn.Module):
    """
    RST-GCN — 2-layer GCN + FC head
    Output: raw logits for 3 classes (no softmax — handled by CrossEntropyLoss)
    """

    def __init__(self, input_dim=768, hidden_dim=128, num_classes=3, dropout=0.5):
        super(RSTGCN, self).__init__()

        self.conv1   = GCNConv(input_dim,  hidden_dim)
        self.conv2   = GCNConv(hidden_dim, hidden_dim)
        self.fc      = nn.Linear(hidden_dim, num_classes)   # 3 output logits
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x:   node features  [num_nodes, input_dim]
        adj: adjacency matrix [num_nodes, num_nodes]
        Returns: logits [num_classes]  (no softmax)
        """
        # GCN layer 1
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)

        # GCN layer 2
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling over all nodes → graph-level representation
        x = torch.mean(x, dim=0)        # [hidden_dim]

        # Classification head — raw logits, NO softmax
        x = self.fc(x)                  # [num_classes]

        return x


# ============================================================
# Graph → Tensors
# ============================================================

def graph_to_tensors(graph, node_embeddings, node_spans, device):
    """
    Convert NetworkX graph + embeddings to tensors.

    Returns:
        node_features : [num_nodes, embedding_dim]
        adj_matrix    : [num_nodes, num_nodes]
    """
    nodes       = sorted(graph.nodes())
    num_nodes   = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            adj[node_to_idx[u], node_to_idx[v]] = 1.0

    # Node feature matrix
    features = []
    for node in nodes:
        if node in node_spans:
            idx = node_spans.index(node)
            features.append(node_embeddings[idx])
        else:
            features.append(np.zeros(node_embeddings.shape[1]))

    features = np.stack(features)

    node_features = torch.tensor(features, dtype=torch.float32, device=device)
    adj_matrix    = torch.tensor(adj,      dtype=torch.float32, device=device)

    return node_features, adj_matrix


# ============================================================
# Batch Preparation
# ============================================================

# Label mapping — update if your dataset uses different strings
LABEL_MAP = {
    # integer labels (already correct)
    0: 0,
    1: 1,
    2: 2,
    # string labels
    'human':          0,
    'ai_original':    1,
    'ai_generated':   1,   # alias
    'ai_paraphrased': 2,
    'paraphrased':    2,   # alias
    'machine':        1,   # fallback for binary datasets
}


def prepare_batch(graphs, embeddings, spans, labels, device):
    """
    Convert a list of (graph, embedding, span, label) into model-ready tensors.

    Returns:
        list of (node_features, adj_matrix, label_tensor)
        label_tensor is a scalar LongTensor for CrossEntropyLoss
    """
    batch = []
    for graph, emb, span, label in zip(graphs, embeddings, spans, labels):
        node_feat, adj = graph_to_tensors(graph, emb, span, device)

        # Resolve label → integer class index
        if isinstance(label, str):
            mapped = LABEL_MAP.get(label.lower().strip(), -1)
        else:
            mapped = LABEL_MAP.get(int(label), -1)

        if mapped == -1:
            raise ValueError(
                f"Unknown label '{label}'. "
                f"Expected one of: {list(LABEL_MAP.keys())}"
            )

        # LongTensor scalar — required by CrossEntropyLoss
        y = torch.tensor(mapped, dtype=torch.long, device=device)

        batch.append((node_feat, adj, y))

    return batch
