#!/usr/bin/env python3
"""
STAGE 3 - Enhanced MO Inference Layer
Purpose: Infer Modus Operandi roles using behavior + structure

Key Features:
1. Feature normalization
2. HDBSCAN clustering (finds natural behavioral groups)
3. MO importance scoring: MOscore = 0.40*BC + 0.25*DC + 0.20*Participation + 0.15*EmbeddingInfluence
4. (Optional) Random Forest for role prediction
5. Map clusters → MO labels (for evaluation)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    classification_report
)
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EnhancedMOInference:
    """
    STAGE 3: Enhanced MO Inference with HDBSCAN and specific MO scoring formula
    
    MO Importance Score Formula:
    MOscore = 0.40*BC + 0.25*DC + 0.20*Participation + 0.15*EmbeddingInfluence
    
    Where:
    - BC = Betweenness Centrality
    - DC = Degree Centrality
    - Participation = Participation Coefficient
    - EmbeddingInfluence = derived from embedding features
    """
    
    def __init__(
        self,
        min_cluster_size=3,
        min_samples=1,
        cluster_selection_epsilon=0.01,
        random_state=42
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.random_state = random_state
        
        self.clusterer = None
        self.classifier = None
        self.feature_cols = None
        self.scaler = RobustScaler()
    
    def infer_mo_roles(
        self,
        features_df,
        true_roles=None,
        method='enhanced_clustering'
    ):
        """
        Infer MO roles using enhanced clustering or supervised methods
        
        Args:
            features_df: DataFrame with features (from Stage 2)
            true_roles: Optional ground truth roles for mapping/evaluation
            method: 'enhanced_clustering' or 'supervised'
        
        Returns:
            predicted_roles: List of predicted role labels
            mo_clusters: Array of cluster labels
        """
        print("\n" + "="*70)
        print("STAGE 3: ENHANCED MO INFERENCE")
        print("="*70)
        
        # Select features for clustering/classification
        print("\n📊 Selecting features for MO inference...")
        structural_features = [
            'degree_centrality', 'betweenness_centrality', 'closeness_centrality',
            'clustering_coefficient', 'core_number', 'load_centrality', 'betweenness_norm'
        ]
        community_features = [
            'ego_density', 'ego_size', 'participation_coefficient',
            'cross_community_edges', 'within_module_z'
        ]
        embedding_features = [f'emb_{i}' for i in range(features_df.filter(like='emb_').shape[1])]
        behavioral_features = [
            'incident_count', 'crime_diversity', 'avg_severity',
            'coordinator_interactions', 'broker_interactions', 'enabler_interactions',
            'peripheral_interactions'
        ]
        
        # Combine and filter for available features
        all_potential_features = (
            structural_features + community_features + 
            embedding_features + behavioral_features
        )
        self.feature_cols = [
            f for f in all_potential_features 
            if f in features_df.columns
        ]
        
        if not self.feature_cols:
            raise ValueError("No valid features found in DataFrame for MO inference.")
        
        print(f"   ✅ Selected {len(self.feature_cols)} features")
        
        # Prepare feature matrix
        X = features_df[self.feature_cols].fillna(0).values
        X_scaled = self._normalize_features(X)
        
        # Initialize cluster labels
        mo_clusters = np.array([-1] * len(features_df))  # Default to noise
        
        if method == 'enhanced_clustering':
            print("\n🔍 Performing HDBSCAN clustering...")
            mo_clusters = self._cluster_features(X_scaled)
            predicted_roles = self._map_clusters_to_roles(
                mo_clusters, features_df, true_roles
            )
        
        elif method == 'supervised' and true_roles is not None:
            print("\n🌳 Training Random Forest classifier...")
            predicted_roles = self._train_and_predict_rf(X_scaled, true_roles)
            # For supervised, assign clusters based on predicted roles
            unique_roles = sorted(list(set(predicted_roles)))
            role_to_cluster = {role: i for i, role in enumerate(unique_roles)}
            mo_clusters = np.array([role_to_cluster[role] for role in predicted_roles])
        
        else:
            print("\n⚠️  Invalid method or missing true_roles. Using rule-based mapping.")
            predicted_roles = self._rule_based_mapping(features_df)
            mo_clusters = np.array([0] * len(features_df))
        
        return predicted_roles, mo_clusters
    
    def _normalize_features(self, X):
        """Normalize features using RobustScaler"""
        print("   - Normalizing features...")
        return self.scaler.fit_transform(X)
    
    def _cluster_features(self, X_scaled):
        """
        Perform HDBSCAN clustering with KMeans fallback
        
        Returns:
            cluster_labels: Array of cluster assignments
        """
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method='leaf',
            metric='euclidean',
            allow_single_cluster=True,
            gen_min_span_tree=True
        )
        
        cluster_labels = self.clusterer.fit_predict(X_scaled)
        
        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"   ✅ HDBSCAN found {unique_clusters} clusters")
        
        # Fallback to KMeans if too few clusters
        if unique_clusters < 4:
            print(f"   ⚠️  Too few clusters ({unique_clusters}), falling back to KMeans with 4 clusters")
            kmeans = KMeans(n_clusters=4, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            print(f"   ✅ KMeans found {len(set(cluster_labels))} clusters")
        
        return cluster_labels
    
    def _map_clusters_to_roles(self, cluster_labels, features_df, true_roles):
        """
        Map numerical cluster labels to meaningful MO role labels
        
        Uses ground truth roles for mapping if available, otherwise uses rule-based approach
        """
        print("\n🏷️  Mapping clusters to MO roles...")
        
        if true_roles is None:
            print("   ⚠️  Ground truth roles not provided. Using rule-based mapping.")
            return self._rule_based_mapping(features_df)
        
        unique_clusters = sorted(set(cluster_labels))
        cluster_to_role = {}
        predicted_roles_list = [None] * len(features_df)
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            indices = np.where(mask)[0]
            
            if cluster_id == -1:  # Noise cluster
                cluster_to_role[cluster_id] = 'Peripheral'
                for idx in indices:
                    predicted_roles_list[idx] = 'Peripheral'
                continue
            
            true_roles_in_cluster = [true_roles[i] for i in indices]
            if not true_roles_in_cluster:
                cluster_to_role[cluster_id] = 'Peripheral'
                for idx in indices:
                    predicted_roles_list[idx] = 'Peripheral'
                continue
            
            # Majority voting for role assignment
            role_counts = defaultdict(int)
            for role in true_roles_in_cluster:
                role_counts[role] += 1
            
            most_common_role = max(role_counts.items(), key=lambda item: item[1])[0]
            cluster_to_role[cluster_id] = most_common_role
            
            for idx in indices:
                predicted_roles_list[idx] = most_common_role
        
        # Ensure all nodes have a predicted role
        for i, role in enumerate(predicted_roles_list):
            if role is None:
                predicted_roles_list[i] = 'Peripheral'
        
        print(f"   ✅ Mapped {len(unique_clusters)} clusters to roles")
        return predicted_roles_list
    
    def _train_and_predict_rf(self, X_scaled, true_roles):
        """Train RandomForestClassifier and predict roles"""
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',
            random_state=self.random_state
        )
        
        # Train on all data (in production, use train/test split)
        self.classifier.fit(X_scaled, true_roles)
        return self.classifier.predict(X_scaled)
    
    def _rule_based_mapping(self, features_df):
        """
        Fallback rule-based mapping for MO roles based on key features
        """
        print("   - Applying rule-based mapping...")
        roles = []
        
        for _, row in features_df.iterrows():
            # Coordinators: high betweenness + high participation
            if (row.get('betweenness_centrality', 0) > 0.05 and
                row.get('participation_coefficient', 0) > 0.6):
                roles.append('Coordinator')
            
            # Brokers: moderate betweenness + high ego density
            elif (row.get('betweenness_centrality', 0) > 0.02 and
                  row.get('ego_density', 0) > 0.5):
                roles.append('Broker')
            
            # Enablers: high degree + incident count
            elif (row.get('degree_centrality', 0) > 0.1 and
                  row.get('incident_count', 0) > 10):
                roles.append('Enabler')
            
            # Peripheral: rest
            else:
                roles.append('Peripheral')
        
        return roles
    
    def compute_mo_importance_scores(self, features_df):
        """
        Compute MO-based importance scores using the specified formula:
        
        MOscore = 0.40*BC + 0.25*DC + 0.20*Participation + 0.15*EmbeddingInfluence
        
        Where:
        - BC = Betweenness Centrality (normalized)
        - DC = Degree Centrality (normalized)
        - Participation = Participation Coefficient (normalized)
        - EmbeddingInfluence = derived from embedding features (normalized)
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            Array of MO importance scores (normalized to [0, 1])
        """
        print("\n💯 Computing MO importance scores...")
        print("   Formula: MOscore = 0.40*BC + 0.25*DC + 0.20*Participation + 0.15*EmbeddingInfluence")
        
        # Ensure required features are present
        required_features = [
            'betweenness_centrality',
            'degree_centrality',
            'participation_coefficient'
        ]
        
        for feat in required_features:
            if feat not in features_df.columns:
                print(f"   ⚠️  Missing feature '{feat}'. Filling with 0.")
                features_df[feat] = 0.0
        
        # Calculate EmbeddingInfluence
        # Use sum of absolute values of embeddings as proxy for 'influence'
        embedding_cols = features_df.filter(like='emb_').columns
        if not embedding_cols.empty:
            features_df['embedding_influence'] = features_df[embedding_cols].abs().sum(axis=1)
        else:
            print("   ⚠️  No embedding features found. 'EmbeddingInfluence' will be 0.")
            features_df['embedding_influence'] = 0.0
        
        # Normalize individual components before summing
        # This ensures fair weighting as specified in the formula
        scaler_bc = RobustScaler()
        scaler_dc = RobustScaler()
        scaler_part = RobustScaler()
        scaler_emb = RobustScaler()
        
        bc_scaled = scaler_bc.fit_transform(
            features_df[['betweenness_centrality']]
        ).flatten()
        dc_scaled = scaler_dc.fit_transform(
            features_df[['degree_centrality']]
        ).flatten()
        part_scaled = scaler_part.fit_transform(
            features_df[['participation_coefficient']]
        ).flatten()
        emb_scaled = scaler_emb.fit_transform(
            features_df[['embedding_influence']]
        ).flatten()
        
        # Apply formula: MOscore = 0.40*BC + 0.25*DC + 0.20*Participation + 0.15*EmbeddingInfluence
        scores = []
        for i in range(len(features_df)):
            mo_score = (
                0.40 * bc_scaled[i] +
                0.25 * dc_scaled[i] +
                0.20 * part_scaled[i] +
                0.15 * emb_scaled[i]
            )
            scores.append(mo_score)
        
        scores = np.array(scores)
        
        # Scale scores to [0, 1] for consistency
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)
        
        print(f"   ✅ Computed MO scores (range: [{scores.min():.3f}, {scores.max():.3f}])")
        
        return scores
    
    def evaluate_mo_inference(self, predicted_roles, true_roles):
        """
        Evaluate MO inference quality using ARI, F1, and NMI
        
        Args:
            predicted_roles: List of predicted role labels
            true_roles: List of true role labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n📈 Evaluating MO inference...")
        
        if true_roles is None or len(set(true_roles)) < 2:
            print("   ⚠️  Cannot evaluate: Ground truth roles not provided or too few unique roles.")
            return {'ARI': 0.0, 'F1': 0.0, 'NMI': 0.0}
        
        ari = adjusted_rand_score(true_roles, predicted_roles)
        f1 = f1_score(true_roles, predicted_roles, average='weighted', zero_division=0)
        nmi = normalized_mutual_info_score(true_roles, predicted_roles)
        
        print("\n" + classification_report(
            true_roles, predicted_roles,
            target_names=['Coordinator', 'Broker', 'Enabler', 'Peripheral'],
            zero_division=0
        ))
        
        results = {
            'ARI': ari,
            'F1': f1,
            'NMI': nmi
        }
        
        print(f"\n📊 MO Inference Evaluation:")
        print(f"   Adjusted Rand Index: {ari:.3f}")
        print(f"   F1-Score (weighted): {f1:.3f}")
        print(f"   Normalized Mutual Info: {nmi:.3f}")
        
        return results


if __name__ == "__main__":
    # Test Stage 3
    print("Testing Stage 3: Enhanced MO Inference")
    
    import pandas as pd
    import numpy as np
    
    # Create dummy features DataFrame
    n_nodes = 100
    features_df = pd.DataFrame({
        'node_id': range(n_nodes),
        'degree_centrality': np.random.rand(n_nodes),
        'betweenness_centrality': np.random.rand(n_nodes) * 0.1,
        'closeness_centrality': np.random.rand(n_nodes),
        'clustering_coefficient': np.random.rand(n_nodes),
        'participation_coefficient': np.random.rand(n_nodes),
        'ego_density': np.random.rand(n_nodes),
        'ego_size': np.random.randint(5, 20, n_nodes),
        'cross_community_edges': np.random.randint(0, 5, n_nodes),
        'within_module_z': np.random.randn(n_nodes),
        'incident_count': np.random.randint(0, 20, n_nodes),
        'crime_diversity': np.random.randint(1, 5, n_nodes),
        'avg_severity': np.random.rand(n_nodes) * 5
    })
    
    # Add embedding features
    for i in range(128):
        features_df[f'emb_{i}'] = np.random.randn(n_nodes)
    
    # Create dummy true roles
    true_roles = np.random.choice(
        ['Coordinator', 'Broker', 'Enabler', 'Peripheral'],
        size=n_nodes,
        p=[0.08, 0.14, 0.20, 0.58]
    )
    
    # Initialize inference
    inference = EnhancedMOInference(
        min_cluster_size=3,
        random_state=42
    )
    
    # Infer roles
    predicted_roles, mo_clusters = inference.infer_mo_roles(
        features_df,
        true_roles=true_roles,
        method='enhanced_clustering'
    )
    
    # Evaluate
    eval_results = inference.evaluate_mo_inference(predicted_roles, true_roles)
    
    # Compute MO scores
    mo_scores = inference.compute_mo_importance_scores(features_df)
    
    print(f"\n✅ Stage 3 test complete!")
    print(f"   Predicted roles: {len(set(predicted_roles))} unique")
    print(f"   MO clusters: {len(set(mo_clusters))} unique")
    print(f"   MO scores: {mo_scores.min():.3f} - {mo_scores.max():.3f}")
