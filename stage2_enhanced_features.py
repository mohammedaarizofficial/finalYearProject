#!/usr/bin/env python3
"""
STAGE 2 - Enhanced Feature Engineering Layer
Purpose: Convert raw graphs into behaviorally meaningful feature vectors

Key Features:
1. Structural Features (Degree, Betweenness, Closeness, Clustering)
2. Community/Ego Features (Participation coefficient, Ego network density)
3. Embedding Features (Node2Vec 128-dim with BFS bias q < 1)
"""

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """
    STAGE 2: Enhanced feature engineering with BFS-biased Node2Vec
    
    Extracts:
    - Structural features (centrality metrics)
    - Community/Ego features (participation, density)
    - Embedding features (Node2Vec with q < 1 for BFS bias)
    """
    
    def __init__(
        self,
        embedding_dim=128,
        walk_length=30,
        num_walks=200,
        q=0.5,  # BFS bias (q < 1)
        p=1.0   # Return parameter
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.q = q  # BFS bias parameter
        self.p = p
    
    def extract_features(
        self,
        G,
        community_labels=None,
        node_ids=None
    ):
        """
        Extract comprehensive feature set
        
        Args:
            G: NetworkX graph
            community_labels: Optional dict mapping node_id -> community_id
            node_ids: Optional list of node IDs (if None, uses G.nodes())
        
        Returns:
            DataFrame with columns: node_id, structural_features, community_features, embedding_features
        """
        print("\n" + "="*70)
        print("STAGE 2: ENHANCED FEATURE ENGINEERING")
        print("="*70)
        
        if node_ids is None:
            node_ids = list(G.nodes())
        
        # 1. Structural Features
        print("\n1️⃣ Computing structural features...")
        structural_df = self._compute_structural_features(G, node_ids)
        print(f"   ✅ Extracted {len(structural_df.columns) - 1} structural features")
        
        # 2. Community/Ego Features
        print("\n2️⃣ Computing community/ego features...")
        community_df = self._compute_community_features(G, node_ids, community_labels)
        print(f"   ✅ Extracted {len(community_df.columns) - 1} community/ego features")
        
        # 3. Embedding Features (BFS-biased Node2Vec)
        print(f"\n3️⃣ Computing embedding features (Node2Vec with BFS bias q={self.q})...")
        embedding_df = self._compute_embeddings(G, node_ids)
        print(f"   ✅ Extracted {len(embedding_df.columns) - 1} embedding features")
        
        # Combine all features
        print("\n🔗 Combining all features...")
        features_df = structural_df.copy()
        features_df = pd.merge(features_df, community_df, on='node_id', how='left')
        features_df = pd.merge(features_df, embedding_df, on='node_id', how='left')
        
        # Fill NaN values
        for col in features_df.columns:
            if col != 'node_id':
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Total features: {len(features_df.columns) - 1}")
        print(f"   Nodes: {len(features_df)}")
        
        return features_df
    
    def _compute_structural_features(self, G, node_ids):
        """
        Compute structural features:
        - Degree centrality
        - Betweenness centrality
        - Closeness centrality
        - Clustering coefficient
        """
        print("   - Computing centrality metrics...")
        
        nodes = node_ids if node_ids else list(G.nodes())
        
        # Degree centrality
        degree_cent = nx.degree_centrality(G)
        
        # Betweenness centrality (sample for large graphs)
        k = min(100, len(nodes))
        betw_cent = nx.betweenness_centrality(G, k=k)
        
        # Closeness centrality
        try:
            close_cent = nx.closeness_centrality(G)
        except:
            close_cent = {n: 0 for n in nodes}
        
        # Clustering coefficient
        clustering = nx.clustering(G)
        
        # K-core (core number)
        try:
            core_number = nx.core_number(G)
        except:
            core_number = {n: 0 for n in nodes}
        
        # Load centrality (bottleneck detection)
        try:
            load_cent = nx.load_centrality(G)
        except:
            load_cent = {n: 0 for n in nodes}
        
        # Normalized betweenness (for comparison)
        max_betw = max(betw_cent.values()) if betw_cent.values() else 1.0
        betweenness_norm = {n: betw_cent.get(n, 0) / (max_betw + 1e-6) for n in nodes}
        
        features = []
        for node in nodes:
            features.append({
                'node_id': node,
                'degree_centrality': degree_cent.get(node, 0),
                'betweenness_centrality': betw_cent.get(node, 0),
                'closeness_centrality': close_cent.get(node, 0),
                'clustering_coefficient': clustering.get(node, 0),
                'core_number': core_number.get(node, 0),
                'load_centrality': load_cent.get(node, 0),
                'betweenness_norm': betweenness_norm.get(node, 0)
            })
        
        return pd.DataFrame(features)
    
    def _compute_community_features(self, G, node_ids, community_labels):
        """
        Compute community/ego features:
        - Participation coefficient
        - Ego network density
        - Ego network size
        - Cross-community edges
        - Within-module degree z-score
        """
        print("   - Computing community/ego features...")
        
        nodes = node_ids if node_ids else list(G.nodes())
        
        # If no community labels, try to detect communities
        if community_labels is None:
            print("     ⚠️  No community labels provided, detecting communities...")
            try:
                communities = nx.community.greedy_modularity_communities(G)
                community_labels = {}
                for comm_id, comm in enumerate(communities):
                    for node in comm:
                        community_labels[node] = comm_id
            except:
                # Fallback: assign all nodes to same community
                community_labels = {n: 0 for n in nodes}
        
        # Compute features
        features = []
        
        for node in nodes:
            # Ego network
            ego_nodes = list(G.neighbors(node)) + [node]
            ego_graph = G.subgraph(ego_nodes)
            
            # Ego network metrics
            ego_size = len(ego_nodes) - 1  # Exclude self
            ego_edges = ego_graph.number_of_edges()
            max_possible_edges = ego_size * (ego_size - 1) / 2 if ego_size > 1 else 1
            ego_density = ego_edges / max_possible_edges if max_possible_edges > 0 else 0
            
            # Community-based metrics
            node_community = community_labels.get(node, 0)
            
            # Count edges to each community
            community_degrees = defaultdict(int)
            for neighbor in G.neighbors(node):
                neighbor_comm = community_labels.get(neighbor, node_community)
                community_degrees[neighbor_comm] += 1
            
            # Participation coefficient
            # P = 1 - sum((k_is / k_i)^2
            # where k_is = degree to community s, k_i = total degree
            total_degree = G.degree(node)
            if total_degree > 0:
                participation = 1 - sum((deg / total_degree) ** 2 for deg in community_degrees.values())
            else:
                participation = 0
            
            # Cross-community edges
            cross_community_edges = sum(
                1 for neighbor in G.neighbors(node)
                if community_labels.get(neighbor, node_community) != node_community
            )
            
            # Within-module degree z-score
            # z_i = (k_is - k_bar_s) / sigma_s
            # where k_is = degree within module, k_bar_s = avg degree in module
            within_module_degree = community_degrees[node_community]
            
            # Compute average degree within module
            module_nodes = [n for n, comm in community_labels.items() if comm == node_community]
            if len(module_nodes) > 1:
                module_degrees = [G.degree(n) for n in module_nodes]
                k_bar_s = np.mean(module_degrees)
                sigma_s = np.std(module_degrees) if len(module_degrees) > 1 else 1.0
                within_module_z = (within_module_degree - k_bar_s) / (sigma_s + 1e-6) if sigma_s > 0 else 0
            else:
                within_module_z = 0
            
            features.append({
                'node_id': node,
                'participation_coefficient': participation,
                'ego_density': ego_density,
                'ego_size': ego_size,
                'cross_community_edges': cross_community_edges,
                'within_module_z': within_module_z
            })
        
        return pd.DataFrame(features)
    
    def _compute_embeddings(self, G, node_ids):
        """
        Compute Node2Vec embeddings with BFS bias (q < 1)
        
        BFS bias (q < 1) means:
        - Walks stay close to starting node
        - Better for capturing local neighborhood structure
        - Good for role-based classification
        """
        print(f"   - Computing Node2Vec embeddings (q={self.q}, BFS bias)...")
        
        nodes = node_ids if node_ids else list(G.nodes())
        
        try:
            # Node2Vec with BFS bias
            # p: return parameter (1.0 = no bias)
            # q: in-out parameter (q < 1 = BFS, q > 1 = DFS)
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                p=self.p,  # Return parameter
                q=self.q,  # BFS bias (q < 1)
                workers=4,
                quiet=True
            )
            
            # Fit model
            model = node2vec.fit(
                window=10,
                min_count=1,
                batch_words=4
            )
            
            # Extract embeddings
            embeddings = []
            for node in nodes:
                try:
                    emb = model.wv[str(node)]
                    emb_dict = {'node_id': node}
                    emb_dict.update({f'emb_{i}': float(emb[i]) for i in range(len(emb))})
                    embeddings.append(emb_dict)
                except KeyError:
                    # Fallback: zero embedding
                    emb_dict = {'node_id': node}
                    emb_dict.update({f'emb_{i}': 0.0 for i in range(self.embedding_dim)})
                    embeddings.append(emb_dict)
            
            return pd.DataFrame(embeddings)
            
        except Exception as e:
            print(f"     ⚠️  Embedding failed ({e}), using fallback")
            # Fallback: return zero embeddings
            embeddings = []
            for node in nodes:
                emb_dict = {'node_id': node}
                emb_dict.update({f'emb_{i}': 0.0 for i in range(self.embedding_dim)})
                embeddings.append(emb_dict)
            return pd.DataFrame(embeddings)


if __name__ == "__main__":
    # Test Stage 2
    print("Testing Stage 2: Enhanced Feature Engineering")
    
    import networkx as nx
    
    # Create test graph
    G = nx.karate_club_graph()
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create community labels (using ground truth from karate club)
    community_labels = {}
    for node in G.nodes():
        # Karate club has two communities
        community_labels[node] = 0 if node < 17 else 1
    
    # Initialize feature engineer
    engineer = EnhancedFeatureEngineer(
        embedding_dim=128,
        walk_length=30,
        num_walks=200,
        q=0.5  # BFS bias
    )
    
    # Extract features
    features_df = engineer.extract_features(
        G,
        community_labels=community_labels
    )
    
    print(f"\n✅ Stage 2 test complete!")
    print(f"   Features shape: {features_df.shape}")
    print(f"   Feature columns: {list(features_df.columns[:10])}...")
