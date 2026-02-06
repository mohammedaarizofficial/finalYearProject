#!/usr/bin/env python3
"""
STAGE 2 - Enhanced Feature Engineering Layer
Purpose: Convert raw graphs into behaviorally meaningful feature vectors

Features:
1. Structural Features (centrality metrics)
2. Community/Ego Features (participation, ego density)
3. Embedding Features (Node2Vec with BFS-biased walks, q < 1)

Output: pandas DataFrame where each row is a node and columns are features
"""

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from collections import defaultdict
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """
    STAGE 2: Enhanced Feature Engineering
    
    Converts NetworkX graphs into comprehensive feature vectors for ML models.
    Focuses on behavioral patterns and structural importance.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,  # Return parameter (BFS when p > 1)
        q: float = 0.5,  # In-out parameter (BFS when q < 1)
        workers: int = 4
    ):
        """
        Initialize feature engineer
        
        Args:
            embedding_dim: Dimension of Node2Vec embeddings (default: 128)
            walk_length: Length of random walks (default: 30)
            num_walks: Number of walks per node (default: 200)
            p: Return parameter for Node2Vec (BFS when p > 1)
            q: In-out parameter for Node2Vec (BFS when q < 1)
            workers: Number of parallel workers
        """
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q  # q < 1 creates BFS-biased walks (explores local neighborhoods)
        self.workers = workers
    
    def extract_features(
        self,
        G: nx.Graph,
        community_labels: Optional[Dict[int, int]] = None,
        node_ids: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Extract comprehensive feature set from graph
        
        Args:
            G: NetworkX graph
            community_labels: Optional dict mapping node_id -> community_id
            node_ids: Optional list of node IDs to process (default: all nodes)
        
        Returns:
            pandas DataFrame with columns:
            - node_id: Node identifier
            - structural features (degree, betweenness, closeness, clustering)
            - community features (participation_coef, ego_density)
            - embedding features (emb_0, emb_1, ..., emb_127)
        """
        print("\n" + "="*70)
        print("STAGE 2: FEATURE ENGINEERING")
        print("="*70)
        
        if node_ids is None:
            node_ids = list(G.nodes())
        
        # Step 1: Structural Features
        print("\n📊 Step 1: Computing structural features...")
        struct_df = self._compute_structural_features(G, node_ids)
        
        # Step 2: Community/Ego Features
        print("🏘️  Step 2: Computing community/ego features...")
        community_df = self._compute_community_ego_features(G, node_ids, community_labels)
        
        # Step 3: Embedding Features (Node2Vec with BFS bias)
        print("🔢 Step 3: Computing embedding features (Node2Vec, BFS-biased)...")
        embedding_df = self._compute_embedding_features(G, node_ids)
        
        # Combine all features
        print("\n🔗 Combining features...")
        features_df = struct_df.copy()
        features_df = pd.merge(features_df, community_df, on='node_id', how='left')
        features_df = pd.merge(features_df, embedding_df, on='node_id', how='left')
        
        # Fill NaN values
        for col in features_df.columns:
            if col != 'node_id':
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        print(f"✅ Feature extraction complete: {len(features_df)} nodes, {len(features_df.columns)-1} features")
        print(f"   Feature columns: {list(features_df.columns)[:10]}... (+{len(features_df.columns)-11} more)")
        
        return features_df
    
    def _compute_structural_features(self, G: nx.Graph, node_ids: list) -> pd.DataFrame:
        """
        Compute structural centrality features
        
        Features:
        - degree: Node degree
        - degree_centrality: Normalized degree
        - betweenness_centrality: Betweenness centrality
        - closeness_centrality: Closeness centrality
        - clustering_coefficient: Local clustering coefficient
        """
        print("  - Computing centrality metrics...")
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        # Compute centralities
        degree_cent = nx.degree_centrality(G)
        
        # Betweenness (sample for large graphs)
        k_sample = min(100, n_nodes)
        betw_cent = nx.betweenness_centrality(G, k=k_sample)
        
        # Closeness
        close_cent = nx.closeness_centrality(G)
        
        # Clustering coefficient
        clustering = nx.clustering(G)
        
        # Build feature vectors
        features = []
        for node_id in node_ids:
            if node_id not in G:
                # Node not in graph
                features.append({
                    'node_id': node_id,
                    'degree': 0,
                    'degree_centrality': 0.0,
                    'betweenness_centrality': 0.0,
                    'closeness_centrality': 0.0,
                    'clustering_coefficient': 0.0
                })
                continue
            
            deg = G.degree(node_id)
            
            features.append({
                'node_id': node_id,
                'degree': deg,
                'degree_centrality': degree_cent.get(node_id, 0.0),
                'betweenness_centrality': betw_cent.get(node_id, 0.0),
                'closeness_centrality': close_cent.get(node_id, 0.0),
                'clustering_coefficient': clustering.get(node_id, 0.0)
            })
        
        return pd.DataFrame(features)
    
    def _compute_community_ego_features(
        self,
        G: nx.Graph,
        node_ids: list,
        community_labels: Optional[Dict[int, int]] = None
    ) -> pd.DataFrame:
        """
        Compute community and ego network features
        
        Features:
        - participation_coefficient: Diversity of community connections
        - ego_density: Density of ego network (1-hop neighborhood)
        - ego_size: Size of ego network
        """
        print("  - Computing participation coefficient and ego features...")
        
        features = []
        
        for node_id in node_ids:
            if node_id not in G:
                features.append({
                    'node_id': node_id,
                    'participation_coefficient': 0.0,
                    'ego_density': 0.0,
                    'ego_size': 0
                })
                continue
            
            # Ego network (1-hop neighborhood)
            ego = nx.ego_graph(G, node_id, radius=1)
            ego_size = ego.number_of_nodes()
            ego_density = nx.density(ego) if ego_size > 1 else 0.0
            
            # Participation coefficient
            participation_coef = 0.0
            if community_labels is not None:
                participation_coef = self._compute_participation_coefficient(
                    G, node_id, community_labels
                )
            
            features.append({
                'node_id': node_id,
                'participation_coefficient': participation_coef,
                'ego_density': ego_density,
                'ego_size': ego_size
            })
        
        return pd.DataFrame(features)
    
    def _compute_participation_coefficient(
        self,
        G: nx.Graph,
        node_id: int,
        community_labels: Dict[int, int]
    ) -> float:
        """
        Compute participation coefficient (diversity of community connections)
        
        Formula: 1 - sum((k_is / k_i)^2)
        where k_is = connections to community s, k_i = total connections
        """
        if node_id not in G:
            return 0.0
        
        neighbors = list(G.neighbors(node_id))
        if len(neighbors) == 0:
            return 0.0
        
        # Count connections per community
        node_comm = community_labels.get(node_id, -1)
        comm_degrees = defaultdict(int)
        
        for neighbor in neighbors:
            neighbor_comm = community_labels.get(neighbor, -1)
            comm_degrees[neighbor_comm] += 1
        
        # Compute participation coefficient
        k_total = len(neighbors)
        if k_total == 0:
            return 0.0
        
        participation = 1.0 - sum((k_s / k_total) ** 2 for k_s in comm_degrees.values())
        
        return participation
    
    def _compute_embedding_features(self, G: nx.Graph, node_ids: list) -> pd.DataFrame:
        """
        Compute Node2Vec embeddings with BFS-biased walks
        
        Uses q < 1 to create BFS-biased walks that explore local neighborhoods
        """
        print(f"  - Generating Node2Vec embeddings (dim={self.embedding_dim}, q={self.q} for BFS bias)...")
        
        try:
            # Create Node2Vec with BFS bias (q < 1)
            # q < 1: BFS behavior (explores local neighborhoods)
            # q > 1: DFS behavior (explores distant nodes)
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                p=self.p,  # Return parameter
                q=self.q,  # In-out parameter (BFS when q < 1)
                workers=self.workers,
                quiet=True
            )
            
            # Train model
            print("  - Training Node2Vec model...")
            model = node2vec.fit(
                window=10,
                min_count=1,
                batch_words=4
            )
            
            # Extract embeddings
            print("  - Extracting embeddings...")
            embeddings = []
            for node_id in node_ids:
                if str(node_id) in model.wv:
                    emb = model.wv[str(node_id)]
                    emb_dict = {'node_id': node_id}
                    emb_dict.update({f'emb_{i}': float(emb[i]) for i in range(len(emb))})
                    embeddings.append(emb_dict)
                else:
                    # Fallback: zero vector
                    emb_dict = {'node_id': node_id}
                    emb_dict.update({f'emb_{i}': 0.0 for i in range(self.embedding_dim)})
                    embeddings.append(emb_dict)
            
            return pd.DataFrame(embeddings)
            
        except Exception as e:
            print(f"  ⚠️  Node2Vec failed ({e}), using zero embeddings")
            # Return zero embeddings as fallback
            embeddings = []
            for node_id in node_ids:
                emb_dict = {'node_id': node_id}
                emb_dict.update({f'emb_{i}': 0.0 for i in range(self.embedding_dim)})
                embeddings.append(emb_dict)
            return pd.DataFrame(embeddings)
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of extracted features
        
        Returns:
            Dictionary with feature statistics
        """
        summary = {
            'n_nodes': len(features_df),
            'n_features': len(features_df.columns) - 1,  # Exclude node_id
            'feature_types': {
                'structural': 5,  # degree, degree_cent, betweenness, closeness, clustering
                'community_ego': 3,  # participation_coef, ego_density, ego_size
                'embedding': self.embedding_dim
            },
            'missing_values': features_df.isnull().sum().to_dict(),
            'feature_ranges': {}
        }
        
        # Compute ranges for numeric features
        numeric_cols = [c for c in features_df.columns if c != 'node_id']
        for col in numeric_cols:
            summary['feature_ranges'][col] = {
                'min': float(features_df[col].min()),
                'max': float(features_df[col].max()),
                'mean': float(features_df[col].mean()),
                'std': float(features_df[col].std())
            }
        
        return summary


def extract_features_from_persons(
    G: nx.Graph,
    persons_df: pd.DataFrame,
    embedding_dim: int = 128,
    q: float = 0.5
) -> pd.DataFrame:
    """
    Convenience function to extract features using persons DataFrame
    
    Args:
        G: NetworkX graph
        persons_df: DataFrame with 'person_id' and optionally 'community' column
        embedding_dim: Dimension of embeddings
        q: Node2Vec q parameter (BFS bias when q < 1)
    
    Returns:
        Feature DataFrame with 'person_id' as node identifier
    """
    # Extract community labels if available
    community_labels = None
    if 'community' in persons_df.columns:
        community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
    
    # Initialize feature engineer
    engineer = EnhancedFeatureEngineer(
        embedding_dim=embedding_dim,
        q=q  # BFS-biased walks
    )
    
    # Extract features
    features_df = engineer.extract_features(
        G,
        community_labels=community_labels,
        node_ids=persons_df['person_id'].tolist()
    )
    
    # Rename node_id to person_id for consistency
    if 'node_id' in features_df.columns:
        features_df = features_df.rename(columns={'node_id': 'person_id'})
    
    return features_df


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing Enhanced Feature Engineering (Stage 2)")
    
    # Create a test graph
    import networkx as nx
    
    G = nx.karate_club_graph()
    print(f"\nTest graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create community labels (using club attribute)
    community_labels = {}
    for node in G.nodes():
        club = G.nodes[node].get('club', 0)
        community_labels[node] = 0 if club == 'Mr. Hi' else 1
    
    # Initialize feature engineer
    engineer = EnhancedFeatureEngineer(
        embedding_dim=64,  # Smaller for testing
        walk_length=20,
        num_walks=50,
        q=0.5  # BFS bias
    )
    
    # Extract features
    features_df = engineer.extract_features(G, community_labels=community_labels)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Shape: {features_df.shape}")
    print(f"   Columns: {list(features_df.columns)[:10]}...")
    
    # Get summary
    summary = engineer.get_feature_summary(features_df)
    print(f"\n📊 Feature Summary:")
    print(f"   Nodes: {summary['n_nodes']}")
    print(f"   Features: {summary['n_features']}")
    print(f"   Feature types: {summary['feature_types']}")
    
    # Show sample
    print(f"\n📋 Sample features (first 5 rows, first 10 columns):")
    print(features_df.iloc[:5, :10].to_string())
