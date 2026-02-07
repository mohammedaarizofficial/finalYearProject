#!/usr/bin/env python3
"""
STAGE 4 - Enhanced ML Enhancement Layer (SEAL)
Purpose: Recover hidden/missing connections using ML

Key Features:
1. DRNL (Double Radius Node Labeling) for subgraph encoding
2. 2-hop enclosing subgraph extraction
3. GNN training (SEAL)
4. Link prediction with confidence scores
5. Graph augmentation with predicted edges (if confidence > threshold)
"""

import os
import random
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_sort_pool
from sklearn.metrics import roc_auc_score, average_precision_score


class EnhancedSEALLinkPredictor:
    """
    STAGE 4: Enhanced SEAL Link Predictor with DRNL
    
    Recovers hidden/missing connections using:
    - 2-hop enclosing subgraphs
    - DRNL node labeling
    - GNN-based link prediction
    - Confidence threshold filtering
    """
    
    def __init__(
        self,
        hop_k: int = 2,
        hidden_dim: int = 64,
        device: str = "cpu",
        seed: int = 42,
        checkpoint_dir: str = "checkpoints",
        lr: float = 0.001,
        epochs: int = 250,
        k: int = 30,
        confidence_threshold: float = 0.7,
        use_drnl: bool = True
    ):
        """
        Initialize enhanced SEAL predictor
        
        Args:
            hop_k: Number of hops for subgraph extraction (default: 2)
            hidden_dim: Hidden dimension for GNN
            device: Device for training ('cpu' or 'cuda')
            seed: Random seed
            checkpoint_dir: Directory for checkpoints
            lr: Learning rate
            epochs: Number of training epochs
            k: Size for global sort pool
            confidence_threshold: Threshold for adding predicted edges (0-1)
            use_drnl: Whether to use DRNL node labeling
        """
        self.hop_k = hop_k
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.k = k
        self.confidence_threshold = confidence_threshold
        self.use_drnl = use_drnl
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        self.training_history = []
        self.in_channels = None
        
        # Track predicted edges
        self.predicted_edges = []
    
    def drnl_node_labeling(self, G: nx.Graph, src: int, dst: int) -> Dict[int, int]:
        """
        DRNL (Double Radius Node Labeling)
        
        Labels nodes based on their distance to src and dst:
        Label = 1 + min(d(src, v), d(dst, v)) + (d(src, v) + d(dst, v) - d(src, dst)) // 2
        
        This encoding helps the GNN distinguish nodes' roles in the subgraph.
        
        Args:
            G: NetworkX graph
            src: Source node
            dst: Destination node
        
        Returns:
            Dictionary mapping node_id -> DRNL label
        """
        labels = {}
        
        # Compute shortest path distances
        try:
            dist_src = nx.single_source_shortest_path_length(G, src)
            dist_dst = nx.single_source_shortest_path_length(G, dst)
            dist_src_dst = nx.shortest_path_length(G, src, dst)
        except:
            # Fallback if graph is disconnected
            dist_src = {src: 0}
            dist_dst = {dst: 0}
            dist_src_dst = float('inf')
        
        # Label all nodes in subgraph
        all_nodes = set(dist_src.keys()) | set(dist_dst.keys())
        
        for node in all_nodes:
            d_src = dist_src.get(node, float('inf'))
            d_dst = dist_dst.get(node, float('inf'))
            
            if d_src == float('inf') or d_dst == float('inf'):
                label = 0  # Unreachable
            else:
                # DRNL formula
                min_dist = min(d_src, d_dst)
                diff = d_src + d_dst - dist_src_dst
                label = 1 + min_dist + (diff // 2)
            
            labels[node] = label
        
        return labels
    
    def extract_enclosing_subgraph(
        self,
        G: nx.Graph,
        src: int,
        dst: int
    ) -> Tuple[nx.Graph, Dict[int, int], int, int]:
        """
        Extract 2-hop enclosing subgraph around edge (src, dst)
        
        Returns:
            Tuple of (subgraph, node_mapping, src_mapped, dst_mapped)
        """
        # Get nodes within hop_k from both src and dst
        try:
            nodes_src = set(nx.single_source_shortest_path_length(G, src, cutoff=self.hop_k).keys())
        except:
            nodes_src = {src}
        
        try:
            nodes_dst = set(nx.single_source_shortest_path_length(G, dst, cutoff=self.hop_k).keys())
        except:
            nodes_dst = {dst}
        
        # Enclosing subgraph: union of neighborhoods
        nodes = nodes_src.union(nodes_dst)
        
        if len(nodes) < 2:
            nodes = {src, dst}
        
        # Create subgraph
        subG = G.subgraph(nodes).copy()
        
        # Relabel nodes contiguously
        node_list = sorted(list(subG.nodes()))
        mapping = {n: i for i, n in enumerate(node_list)}
        subG_rel = nx.relabel_nodes(subG, mapping, copy=True)
        
        # Map src and dst to new labels
        src_mapped = mapping.get(src, 0)
        dst_mapped = mapping.get(dst, 0)
        
        return subG_rel, mapping, src_mapped, dst_mapped
    
    def prepare_link_prediction_data(
        self,
        G: nx.Graph,
        features_df: Optional[pd.DataFrame] = None,
        test_ratio: float = 0.2,
        max_samples: int = 1200
    ) -> Dict[str, List]:
        """
        Prepare SEAL training data with DRNL labeling
        
        Args:
            G: NetworkX graph
            features_df: Optional DataFrame with node features
            test_ratio: Ratio of test edges
            max_samples: Maximum training samples
        
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        print("\n" + "="*70)
        print("STAGE 4: SEAL LINK PREDICTION DATA PREPARATION")
        print("="*70)
        
        # Determine feature dimension
        if features_df is not None:
            numeric_cols = [
                c for c in features_df.columns 
                if c not in ('person_id', 'node_id', 'true_role', 'predicted_role', 
                           'mo_cluster', 'mo_score', 'mo_importance_score')
                and pd.api.types.is_numeric_dtype(features_df[c])
            ]
            
            if len(numeric_cols) == 0:
                print("  ⚠️  No numeric columns, using degree features")
                features_df = None
                feature_dim = 2
            else:
                feature_dim = len(numeric_cols)
                features_map = {}
                for _, row in features_df.iterrows():
                    pid = row.get('person_id') or row.get('node_id')
                    if pid is not None:
                        features_map[pid] = row[numeric_cols].astype(np.float32).values
                
                print(f"  ✅ Using {feature_dim} features")
        else:
            feature_dim = 2
            features_map = None
            print("  ✅ Using degree-based features (2D)")
        
        # Add DRNL label dimension if using DRNL
        if self.use_drnl:
            self.in_channels = feature_dim + 1  # +1 for DRNL label
        else:
            self.in_channels = feature_dim
        
        # Initialize model
        if self.model is None:
            self.model = EnhancedSEALNet(
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
            save_path = os.path.join(self.checkpoint_dir, f"{split_name}_subgraphs_stage4.pkl")
            
            if os.path.exists(save_path):
                try:
                    with open(save_path, "rb") as f:
                        subgraphs = pickle.load(f)
                    print(f"  ✅ Loaded cached {split_name} subgraphs")
                    data_dict[split_name] = subgraphs
                    continue
                except:
                    print(f"  ⚠️  Failed to load cache, rebuilding...")
            
            print(f"  Extracting {split_name} subgraphs (with DRNL)...")
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
    
    def _extract_pos_neg_pair(
        self,
        G: nx.Graph,
        src: int,
        dst: int,
        all_nodes: List[int],
        features_map: Optional[Dict],
        feature_dim: int
    ) -> List[Data]:
        """Extract positive and negative subgraph pair"""
        # Positive subgraph
        pos_data = self._extract_subgraph_with_drnl(
            G, src, dst, label=1.0,
            features_map=features_map,
            feature_dim=feature_dim
        )
        
        # Negative subgraph (sample non-edge)
        attempts = 0
        while attempts < 100:
            u, v = random.sample(all_nodes, 2)
            if not G.has_edge(u, v) and u != v:
                break
            attempts += 1
        
        if attempts >= 100:
            u, v = random.sample(all_nodes, 2)
        
        neg_data = self._extract_subgraph_with_drnl(
            G, u, v, label=0.0,
            features_map=features_map,
            feature_dim=feature_dim
        )
        
        return [pos_data, neg_data]
    
    def _extract_subgraph_with_drnl(
        self,
        G: nx.Graph,
        src: int,
        dst: int,
        label: float,
        features_map: Optional[Dict],
        feature_dim: int
    ) -> Data:
        """
        Extract subgraph with DRNL node labeling
        
        Returns:
            PyTorch Geometric Data object
        """
        # Extract enclosing subgraph
        subG, mapping, src_mapped, dst_mapped = self.extract_enclosing_subgraph(G, src, dst)
        
        # Build edge_index
        edges = list(subG.edges())
        if len(edges) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_list = []
            for u, v in edges:
                edge_list.extend([[u, v], [v, u]])  # Bidirectional
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Build node features
        node_list = sorted(list(subG.nodes()))
        X = []
        inv_map = {v: k for k, v in mapping.items()}
        
        # Get DRNL labels
        if self.use_drnl:
            drnl_labels = self.drnl_node_labeling(G, src, dst)
        else:
            drnl_labels = {}
        
        for i in range(len(node_list)):
            orig_node = inv_map[i]
            
            # Base features
            if features_map is not None:
                vec = features_map.get(orig_node)
                if vec is None or len(vec) != feature_dim:
                    # Fallback to degree
                    deg = subG.degree(i)
                    max_deg = max([d for _, d in subG.degree()]) if subG.number_of_nodes() > 0 else 1
                    norm = deg / (max_deg + 1e-6)
                    vec = np.zeros(feature_dim, dtype=np.float32)
                    vec[0] = deg
                    if feature_dim > 1:
                        vec[1] = norm
            else:
                # Degree features
                deg = subG.degree(i)
                max_deg = max([d for _, d in subG.degree()]) if subG.number_of_nodes() > 0 else 1
                norm = deg / (max_deg + 1e-6)
                vec = np.zeros(feature_dim, dtype=np.float32)
                vec[0] = deg
                if feature_dim > 1:
                    vec[1] = norm
            
            # Add DRNL label as additional feature
            if self.use_drnl:
                drnl_label = drnl_labels.get(orig_node, 0)
                # Normalize label (max label is typically < 20)
                drnl_normalized = drnl_label / 20.0
                vec = np.append(vec, drnl_normalized)
            
            X.append(vec)
        
        x = torch.tensor(np.vstack(X), dtype=torch.float)
        
        # Ensure correct dimension
        if x.size(1) != self.in_channels:
            if x.size(1) < self.in_channels:
                pad = torch.zeros(x.size(0), self.in_channels - x.size(1))
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, :self.in_channels]
        
        y = torch.tensor([label], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def train_seal(
        self,
        train_data: List,
        val_data: Optional[List] = None,
        batch_size: int = 32
    ) -> Dict:
        """
        Train SEAL model
        
        Returns:
            Dictionary with training results
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
        patience = 15
        
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
                
                if epoch % 10 == 0 or epoch == 1:
                    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
                
                # Save best model
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_auc)
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
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}")
        
        print(f"\n✅ Training complete! Best Val AUC: {best_auc:.4f} at epoch {best_epoch}")
        
        return {
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "training_history": self.training_history
        }
    
    def evaluate_with_metrics(self, loader: DataLoader) -> Tuple[float, float, float]:
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
    
    def predict_missing_links(
        self,
        G: nx.Graph,
        features_df: Optional[pd.DataFrame] = None,
        candidate_pairs: Optional[List[Tuple[int, int]]] = None,
        max_candidates: int = 1000
    ) -> pd.DataFrame:
        """
        Predict missing links with confidence scores
        
        Args:
            G: Noisy graph
            features_df: Optional node features
            candidate_pairs: Optional list of (u, v) pairs to evaluate
            max_candidates: Maximum candidate pairs to evaluate
        
        Returns:
            DataFrame with columns: from_id, to_id, confidence, predicted
        """
        print("\n🔮 Predicting missing links...")
        
        if self.model is None:
            raise ValueError("Model not trained! Call train_seal() first.")
        
        # Get candidate pairs
        if candidate_pairs is None:
            # Sample non-edges as candidates
            all_nodes = list(G.nodes())
            existing_edges = set(G.edges())
            candidate_pairs = []
            
            attempts = 0
            while len(candidate_pairs) < max_candidates and attempts < max_candidates * 10:
                u, v = random.sample(all_nodes, 2)
                if u > v:
                    u, v = v, u
                
                if (u, v) not in existing_edges and (v, u) not in existing_edges:
                    candidate_pairs.append((u, v))
                
                attempts += 1
        
        print(f"  Evaluating {len(candidate_pairs)} candidate pairs...")
        
        # Prepare features
        features_map = None
        feature_dim = 2
        
        if features_df is not None:
            numeric_cols = [
                c for c in features_df.columns 
                if c not in ('person_id', 'node_id', 'true_role', 'predicted_role',
                           'mo_cluster', 'mo_score', 'mo_importance_score')
                and pd.api.types.is_numeric_dtype(features_df[c])
            ]
            
            if len(numeric_cols) > 0:
                feature_dim = len(numeric_cols)
                features_map = {}
                for _, row in features_df.iterrows():
                    pid = row.get('person_id') or row.get('node_id')
                    if pid is not None:
                        features_map[pid] = row[numeric_cols].astype(np.float32).values
        
        # Predict for each candidate
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for u, v in tqdm(candidate_pairs, desc="  Predicting"):
                # Extract subgraph
                data = self._extract_subgraph_with_drnl(
                    G, u, v, label=0.0,  # Label doesn't matter for prediction
                    features_map=features_map,
                    feature_dim=feature_dim
                )
                
                # Predict
                data = data.to(self.device)
                pred = self.model(
                    data.x.unsqueeze(0),
                    data.edge_index,
                    torch.zeros(data.x.size(0), dtype=torch.long).to(self.device)
                )
                confidence = pred.item()
                
                predictions.append({
                    'from_id': u,
                    'to_id': v,
                    'confidence': confidence,
                    'predicted': confidence >= self.confidence_threshold
                })
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('confidence', ascending=False)
        
        # Filter by threshold
        high_confidence = predictions_df[predictions_df['predicted']]
        print(f"  ✅ Found {len(high_confidence)} high-confidence predictions (≥{self.confidence_threshold})")
        
        return predictions_df
    
    def augment_graph(
        self,
        G: nx.Graph,
        predictions_df: pd.DataFrame,
        validate_against: Optional[nx.Graph] = None
    ) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Augment graph with predicted edges
        
        Args:
            G: Original noisy graph
            predictions_df: DataFrame with predictions (from predict_missing_links)
            validate_against: Optional ground truth graph for validation
        
        Returns:
            Tuple of (augmented_graph, validation_stats)
        """
        print("\n🔗 Augmenting graph with predicted edges...")
        
        G_aug = G.copy()
        
        # Filter by confidence threshold
        high_conf = predictions_df[predictions_df['predicted']]
        
        added_edges = []
        for _, row in high_conf.iterrows():
            u, v = row['from_id'], row['to_id']
            confidence = row['confidence']
            
            # Add edge if not already present
            if not G_aug.has_edge(u, v):
                G_aug.add_edge(u, v, weight=1.0, confidence=confidence, source='seal_predicted')
                added_edges.append({
                    'from_id': u,
                    'to_id': v,
                    'confidence': confidence
                })
        
        print(f"  ✅ Added {len(added_edges)} predicted edges")
        print(f"  Original edges: {G.number_of_edges()}")
        print(f"  Augmented edges: {G_aug.number_of_edges()}")
        print(f"  Recovery rate: {len(added_edges) / G.number_of_edges() * 100:.1f}%")
        
        # Validate against ground truth if available
        validation_stats = {}
        if validate_against is not None:
            true_edges = set(validate_against.edges())
            predicted_edges = set((row['from_id'], row['to_id']) for _, row in high_conf.iterrows())
            
            # Check how many predicted edges are actually in ground truth
            correct = len(predicted_edges & true_edges)
            precision = correct / len(predicted_edges) if len(predicted_edges) > 0 else 0.0
            
            validation_stats = {
                'predicted_count': len(predicted_edges),
                'correct_count': correct,
                'precision': precision
            }
            
            print(f"\n📊 Validation against ground truth:")
            print(f"  Predicted edges: {len(predicted_edges)}")
            print(f"  Correct predictions: {correct}")
            print(f"  Precision: {precision:.3f}")
        
        # Store predicted edges
        self.predicted_edges = added_edges
        
        return G_aug, validation_stats
    
    def save_checkpoint(self, epoch: int, auc: float):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, "seal_best_model_stage4.pt")
        torch.save({
            "epoch": epoch,
            "auc": auc,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "in_channels": self.in_channels
        }, path)
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load model checkpoint"""
        if path is None:
            path = os.path.join(self.checkpoint_dir, "seal_best_model_stage4.pt")
        
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device)
            if self.model is None:
                self.model = EnhancedSEALNet(
                    in_channels=ck.get('in_channels', self.in_channels),
                    hidden_channels=self.hidden_dim,
                    k=self.k
                ).to(self.device)
            self.model.load_state_dict(ck["model_state"])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(ck["optimizer_state"])
            print(f"✅ Loaded checkpoint (epoch {ck['epoch']}, AUC {ck.get('auc', 'N/A')})")
            return ck["epoch"]
        return 0


class EnhancedSEALNet(nn.Module):
    """Enhanced SEAL GNN Model with DRNL support"""
    
    def __init__(self, in_channels: int = 2, hidden_channels: int = 64, num_layers: int = 2, k: int = 30):
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
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
    # Test Stage 4
    print("Testing Stage 4: Enhanced SEAL with DRNL")
    
    import networkx as nx
    
    # Create test graph
    G = nx.karate_club_graph()
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Initialize predictor
    predictor = EnhancedSEALLinkPredictor(
        hop_k=2,
        hidden_dim=32,
        epochs=10,
        confidence_threshold=0.7,
        use_drnl=True
    )
    
    # Prepare data
    data_dict = predictor.prepare_link_prediction_data(G, max_samples=50)
    
    # Train
    results = predictor.train_seal(
        data_dict['train'],
        data_dict['val'],
        batch_size=16
    )
    
    print(f"\n✅ Training complete! AUC: {results['best_auc']:.4f}")
    
    # Predict missing links
    predictions = predictor.predict_missing_links(G, max_candidates=100)
    
    print(f"\n📊 Predictions:")
    print(predictions.head(10))
    
    # Augment graph
    G_aug, stats = predictor.augment_graph(G, predictions)
    
    print(f"\n✅ Graph augmentation complete!")
    print(f"   Original: {G.number_of_edges()} edges")
    print(f"   Augmented: {G_aug.number_of_edges()} edges")
