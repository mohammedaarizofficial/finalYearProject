#!/usr/bin/env python3
"""
FIXED: SEAL Link Prediction
Key fixes:
- Proper feature dimension handling
- Better subgraph extraction
- Fixed training loop with validation monitoring
- Improved negative sampling
"""

import os
import random
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_sort_pool
from sklearn.metrics import roc_auc_score, average_precision_score


class SEALLinkPredictor:
    def __init__(
        self,
        hop_k=2,
        hidden_dim=32,
        device="cpu",
        seed=42,
        checkpoint_dir="checkpoints",
        lr=0.001,
        epochs=50,
        k=30,
        in_channels=None,
    ):
        self.hop_k = hop_k
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.k = k
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.in_channels = in_channels
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        
        # Track training metrics
        self.training_history = []

    def prepare_link_prediction_data(self, G, features_df=None, test_ratio=0.2, max_samples=3000):
        """
        FIXED: Prepare SEAL training data with proper feature handling
        """
        print("\n🔧 Preparing SEAL link prediction data...")
        
        # Determine feature dimension
        if features_df is not None:
            numeric_cols = [c for c in features_df.columns 
                          if c not in ('person_id', 'node_id', 'true_role', 'predicted_role') 
                          and np.issubdtype(features_df[c].dtype, np.number)]
            
            if len(numeric_cols) == 0:
                print("  ⚠️  No numeric columns in features_df, using degree features")
                features_df = None
                feature_dim = 2
            else:
                feature_dim = len(numeric_cols)
                # Create feature map
                features_map = {}
                for _, row in features_df.iterrows():
                    pid = row.get('person_id')
                    if pid is not None:
                        features_map[pid] = row[numeric_cols].astype(np.float32).values
                
                print(f"  ✅ Using {feature_dim} features: {numeric_cols[:5]}...")
        else:
            feature_dim = 2
            features_map = None
            print("  ✅ Using degree-based features (2D)")

        # Initialize model if not exists
        if self.model is None:
            self.in_channels = feature_dim
            self.model = SEALNet(
                in_channels=self.in_channels, 
                hidden_channels=self.hidden_dim, 
                k=self.k
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            print(f"  ✅ Initialized SEAL model: in_channels={self.in_channels}")

        # Split edges
        all_edges = list(G.edges())
        random.shuffle(all_edges)
        
        num_test = min(int(len(all_edges) * test_ratio), len(all_edges) // 3)
        num_val = num_test // 2
        
        # Ensure we don't exceed max_samples
        train_end = min(len(all_edges) - num_test - num_val, max_samples)
        
        splits = {
            "train": all_edges[num_test + num_val : num_test + num_val + train_end],
            "val": all_edges[num_test : num_test + num_val],
            "test": all_edges[:num_test],
        }
        
        print(f"  Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        # Extract subgraphs with caching
        data_dict = {}
        for split_name, split_edges in splits.items():
            save_path = os.path.join(self.checkpoint_dir, f"{split_name}_subgraphs_v2.pkl")
            
            if os.path.exists(save_path):
                try:
                    with open(save_path, "rb") as f:
                        subgraphs = pickle.load(f)
                    print(f"  ✅ Loaded cached {split_name} subgraphs")
                    data_dict[split_name] = subgraphs
                    continue
                except:
                    print(f"  ⚠️  Failed to load cache, rebuilding...")

            print(f"  Extracting {split_name} subgraphs...")
            subgraphs = []
            all_nodes = list(G.nodes())
            
            for u, v in tqdm(split_edges, desc=f"  {split_name}"):
                pair = self._extract_pos_neg_pair(
                    G, u, v, all_nodes, features_map, feature_dim
                )
                subgraphs.append(pair)
            
            with open(save_path, "wb") as f:
                pickle.dump(subgraphs, f)
            print(f"  ✅ Saved {split_name} subgraphs")
            
            data_dict[split_name] = subgraphs

        return data_dict

    def _extract_pos_neg_pair(self, G, src, dst, all_nodes, features_map, feature_dim):
        """Extract positive and negative subgraph pair"""
        # Positive subgraph
        pos_data = self._extract_subgraph(G, src, dst, label=1.0, 
                                         features_map=features_map, 
                                         feature_dim=feature_dim)
        
        # Negative subgraph (sample non-edge)
        attempts = 0
        while attempts < 100:
            u, v = random.sample(all_nodes, 2)
            if not G.has_edge(u, v) and u != v:
                break
            attempts += 1
        
        if attempts >= 100:  # Fallback
            u, v = random.sample(all_nodes, 2)
        
        neg_data = self._extract_subgraph(G, u, v, label=0.0,
                                         features_map=features_map,
                                         feature_dim=feature_dim)
        
        return [pos_data, neg_data]

    def _extract_subgraph(self, G, src, dst, label, features_map, feature_dim):
        """Extract h-hop subgraph around an edge"""
        # Get nodes within h hops
        try:
            nodes_src = set(nx.single_source_shortest_path_length(G, src, cutoff=self.hop_k).keys())
        except:
            nodes_src = {src}
        
        try:
            nodes_dst = set(nx.single_source_shortest_path_length(G, dst, cutoff=self.hop_k).keys())
        except:
            nodes_dst = {dst}
        
        nodes = nodes_src.union(nodes_dst)
        
        if len(nodes) < 2:
            nodes = {src, dst}
        
        # Create subgraph
        subG = G.subgraph(nodes).copy()
        
        # Relabel nodes contiguously
        node_list = sorted(list(subG.nodes()))
        mapping = {n: i for i, n in enumerate(node_list)}
        subG_rel = nx.relabel_nodes(subG, mapping, copy=True)
        
        # Build edge_index
        edges = list(subG_rel.edges())
        if len(edges) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_list = []
            for u, v in edges:
                edge_list.extend([[u, v], [v, u]])  # Bidirectional
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Build node features
        if features_map is not None:
            X = []
            inv_map = {v: k for k, v in mapping.items()}
            
            for i in range(len(node_list)):
                orig_node = inv_map[i]
                vec = features_map.get(orig_node)
                
                if vec is None or len(vec) != feature_dim:
                    # Fallback to degree
                    deg = subG_rel.degree(i)
                    max_deg = max([d for _, d in subG_rel.degree()]) if subG_rel.number_of_nodes() > 0 else 1
                    norm = deg / (max_deg + 1e-6)
                    vec = np.zeros(feature_dim, dtype=np.float32)
                    vec[0] = deg
                    if feature_dim > 1:
                        vec[1] = norm
                
                X.append(vec)
            
            x = torch.tensor(np.vstack(X), dtype=torch.float)
        else:
            # Degree features fallback
            degrees = np.array([subG_rel.degree(n) for n in range(subG_rel.number_of_nodes())], dtype=np.float32)
            max_deg = degrees.max() if degrees.size > 0 else 1.0
            norm_deg = degrees / (max_deg + 1e-6)
            x = torch.tensor(np.column_stack([degrees, norm_deg]).astype(np.float32), dtype=torch.float)
        
        # Ensure feature dimension matches
        if x.size(1) != feature_dim:
            if x.size(1) < feature_dim:
                pad = torch.zeros(x.size(0), feature_dim - x.size(1))
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, :feature_dim]
        
        y = torch.tensor([label], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y)

    def train_seal(self, train_data, val_data=None, batch_size=32):
        """
        FIXED: Training with proper monitoring and early stopping
        """
        print("\n🚀 Training SEAL model...")
        
        # Flatten pairs
        train_list = [d for pair in train_data for d in pair]
        val_list = [d for pair in val_data for d in pair] if val_data else None
        
        train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False) if val_list else None
        
        best_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out.view(-1), batch.y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            
            # Validation
            if val_loader:
                val_loss, val_auc, val_ap = self.evaluate_with_metrics(val_loader)
                
                print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
                
                # Save best model
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_auc)
                    print(f"  ✅ New best AUC: {best_auc:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"  ⏹️  Early stopping at epoch {epoch}")
                    break
                
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_ap': val_ap
                })
            else:
                print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}")
        
        print(f"\n✅ Training complete! Best Val AUC: {best_auc:.4f} at epoch {best_epoch}")
        
        return {
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "training_history": self.training_history
        }

    def evaluate_with_metrics(self, loader):
        """Evaluate and compute metrics"""
        self.model.eval()
        total_loss = 0.0
        preds = []
        labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out.view(-1), batch.y)
                
                total_loss += loss.item()
                preds.extend(out.view(-1).cpu().numpy().tolist())
                labels.extend(batch.y.cpu().numpy().tolist())
        
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        
        try:
            auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else 0.5
            ap = average_precision_score(labels, preds) if len(set(labels)) > 1 else 0.5
        except:
            auc, ap = 0.5, 0.5
        
        return avg_loss, auc, ap

    def save_checkpoint(self, epoch, auc):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, "seal_best_model.pt")
        torch.save({
            "epoch": epoch,
            "auc": auc,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path=None):
        """Load model checkpoint"""
        if path is None:
            path = os.path.join(self.checkpoint_dir, "seal_best_model.pt")
        
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ck["model_state"])
            self.optimizer.load_state_dict(ck["optimizer_state"])
            print(f"✅ Loaded checkpoint (epoch {ck['epoch']}, AUC {ck.get('auc', 'N/A')})")
            return ck["epoch"]
        return 0


class SEALNet(nn.Module):
    """SEAL GNN Model"""
    def __init__(self, in_channels=2, hidden_channels=32, num_layers=2, k=30):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.k = k
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.linear = nn.Linear(hidden_channels * k, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        # GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        
        # Global pooling
        x = global_sort_pool(x, batch, k=self.k)
        
        # MLP
        x = self.linear(x)
        return torch.sigmoid(x.view(-1))


if __name__ == "__main__":
    # Test with dummy data
    print("Testing SEAL implementation...")
    
    # Create dummy graph
    G = nx.karate_club_graph()
    
    # Dummy features
    features_df = pd.DataFrame({
        'person_id': list(G.nodes()),
        'feature1': np.random.randn(G.number_of_nodes()),
        'feature2': np.random.randn(G.number_of_nodes())
    })
    
    # Initialize predictor
    predictor = SEALLinkPredictor(hop_k=2, hidden_dim=32, epochs=10)
    
    # Prepare data
    data_dict = predictor.prepare_link_prediction_data(
        G, features_df=features_df, max_samples=50
    )
    
    # Train
    results = predictor.train_seal(
        data_dict['train'], 
        data_dict['val'], 
        batch_size=16
    )
    
    print(f"\n✅ Test complete! AUC: {results['best_auc']:.4f}")