# STAGE 2 - Enhanced Feature Engineering Layer

## Overview

Stage 2 upgrades the feature extraction system to produce behaviorally meaningful feature vectors from raw graphs. Focuses on structural patterns, community roles, and graph embeddings.

## Key Features

### 1. Structural Features

**Centrality Metrics:**
- `degree`: Node degree (number of connections)
- `degree_centrality`: Normalized degree (0-1)
- `betweenness_centrality`: Fraction of shortest paths passing through node
- `closeness_centrality`: Average distance to all other nodes
- `clustering_coefficient`: Local clustering (triangles)

**Purpose**: Identify structural importance and hub vs peripheral roles

### 2. Community/Ego Features

**Ego Network Features:**
- `participation_coefficient`: Diversity of community connections (0-1)
  - High: Connects across many communities (broker role)
  - Low: Mostly within one community (peripheral role)
- `ego_density`: Density of 1-hop neighborhood
- `ego_size`: Size of 1-hop neighborhood

**Purpose**: Capture community roles and local network structure

### 3. Embedding Features

**Node2Vec Embeddings:**
- 128-dimensional vectors (configurable)
- BFS-biased walks (q < 1) to explore local neighborhoods
- Captures structural similarity and role patterns

**Purpose**: Dense representations for ML models

## Usage

### Basic Usage

```python
from feature_engineering_v2 import EnhancedFeatureEngineer
import networkx as nx

# Create graph
G = nx.karate_club_graph()

# Optional: community labels
community_labels = {i: i % 2 for i in G.nodes()}

# Initialize feature engineer
engineer = EnhancedFeatureEngineer(
    embedding_dim=128,
    walk_length=30,
    num_walks=200,
    q=0.5  # BFS bias (q < 1)
)

# Extract features
features_df = engineer.extract_features(
    G,
    community_labels=community_labels
)

# Result: pandas DataFrame
# Columns: node_id, degree, degree_centrality, ..., emb_0, emb_1, ..., emb_127
```

### Using with Persons DataFrame

```python
from feature_engineering_v2 import extract_features_from_persons
import pandas as pd
import networkx as nx

# Load data
persons_df = pd.read_csv('persons.csv')
G = nx.read_edgelist('relations.csv')

# Extract features (automatically uses community column if present)
features_df = extract_features_from_persons(
    G,
    persons_df,
    embedding_dim=128,
    q=0.5  # BFS bias
)
```

### Using with Main Pipeline

The main pipeline automatically uses Stage 2 when enabled:

```python
# In main_pipeline.py CONFIG
CONFIG = {
    'USE_ENHANCED_FEATURES': True,  # Enable Stage 2
    'MO_EMBEDDING_DIM': 128,
    'MO_WALK_LENGTH': 30,
    'MO_NUM_WALKS': 200,
    'MO_NODE2VEC_Q': 0.5,  # BFS bias (q < 1)
    # ... other config
}
```

Run the pipeline:
```bash
python main_pipeline.py
```

## Feature Output

### DataFrame Structure

Each row represents a node with the following columns:

**Node Identifier:**
- `node_id` (or `person_id` if using convenience function)

**Structural Features (5):**
- `degree`
- `degree_centrality`
- `betweenness_centrality`
- `closeness_centrality`
- `clustering_coefficient`

**Community/Ego Features (3):**
- `participation_coefficient`
- `ego_density`
- `ego_size`

**Embedding Features (128):**
- `emb_0`, `emb_1`, ..., `emb_127`

**Total**: 136 features per node

## BFS-Biased Walks (q < 1)

### Why BFS Bias?

- **q < 1**: BFS behavior - explores local neighborhoods
  - Better for capturing community structure
  - Identifies role patterns within neighborhoods
  - Recommended for criminal networks

- **q > 1**: DFS behavior - explores distant nodes
  - Better for global structure
  - Less useful for role identification

### Configuration

```python
engineer = EnhancedFeatureEngineer(
    q=0.5  # BFS bias (recommended)
    # q=1.0  # Unbiased
    # q=2.0  # DFS bias
)
```

## Feature Summary

Get statistics about extracted features:

```python
summary = engineer.get_feature_summary(features_df)

print(f"Nodes: {summary['n_nodes']}")
print(f"Features: {summary['n_features']}")
print(f"Feature types: {summary['feature_types']}")
print(f"Ranges: {summary['feature_ranges']}")
```

## Comparison: Legacy vs Enhanced

| Feature | Legacy | Enhanced (Stage 2) |
|---------|--------|-------------------|
| Structural Features | 8 features | 5 core features (focused) |
| Community Features | Basic | Participation coefficient + ego |
| Embeddings | Node2Vec (default) | Node2Vec with BFS bias (q < 1) |
| Output Format | Mixed | Clean pandas DataFrame |
| Embedding Dim | 64 (default) | 128 (configurable) |
| BFS Bias | No | Yes (q < 1) |

## Integration with MO Inference

Stage 2 features work seamlessly with MO inference:

```python
from mo_feature_extraction import MOInference

# Extract features (Stage 2)
features_df = engineer.extract_features(G, community_labels)

# Infer MO roles
inference = MOInference()
predicted_roles = inference.infer_mo_roles(
    features_df,
    true_roles=true_roles,
    method='enhanced_clustering'
)
```

## Performance Considerations

### Computational Cost

1. **Structural Features**: O(n²) for betweenness (sampled for large graphs)
2. **Community Features**: O(n × k) where k is average degree
3. **Embeddings**: O(n × walks × walk_length) - most expensive

### Optimization Tips

- Use `k` sampling for betweenness on large graphs (automatic)
- Reduce `num_walks` for faster embedding (trade-off: quality)
- Use parallel workers for Node2Vec (`workers=4`)

## Expected Output

When Stage 2 runs successfully:

```
STAGE 2: FEATURE ENGINEERING
======================================================================

📊 Step 1: Computing structural features...
  - Computing centrality metrics...
🏘️  Step 2: Computing community/ego features...
  - Computing participation coefficient and ego features...
🔢 Step 3: Computing embedding features (Node2Vec, BFS-biased)...
  - Generating Node2Vec embeddings (dim=128, q=0.5 for BFS bias)...
  - Training Node2Vec model...
  - Extracting embeddings...

🔗 Combining features...
✅ Feature extraction complete: 400 nodes, 136 features
   Feature columns: ['node_id', 'degree', 'degree_centrality', ...]
```

## Troubleshooting

### Embeddings fail
- Check if Node2Vec is installed: `pip install node2vec`
- Reduce `embedding_dim` or `num_walks` for large graphs
- Check graph connectivity (should be mostly connected)

### Participation coefficient is 0
- Ensure `community_labels` are provided
- Check that nodes have neighbors
- Verify community labels match node IDs

### Features are all zeros
- Check if graph is empty or disconnected
- Verify node IDs match between graph and labels
- Check for NaN handling (should fill with 0)

## Next Steps

After Stage 2, proceed to:
- **Stage 3**: SEAL Link Prediction (uses Stage 2 features)
- **Stage 4**: Adaptive Graph Repair
- **Stage 5**: Interpretable Disruption Insights

## Example Output

```python
# Sample feature DataFrame
   node_id  degree  degree_centrality  betweenness_centrality  ...  emb_0    emb_1    ...
0        0       5               0.15                    0.02  ...  0.123  -0.456   ...
1        1       8               0.24                    0.15  ...  0.234   0.567    ...
2        2       3               0.09                    0.01  ... -0.123   0.234   ...
```

## Conclusion

Stage 2 provides a robust, focused feature engineering layer that:
- ✅ Captures structural importance
- ✅ Identifies community roles
- ✅ Produces dense embeddings for ML
- ✅ Outputs clean pandas DataFrames
- ✅ Integrates seamlessly with pipeline

**Status**: Ready for production use.
