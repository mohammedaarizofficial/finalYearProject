# Integration Guide - All Enhanced Stages

## Overview

This guide explains how all enhanced stages (1-4) work together and how to use the integrated pipeline.

## Stage Integration

### Data Flow

```
Stage 1 (Enhanced Generator)
    ↓
    graph, persons, incidents, graph_clean
    ↓
Stage 2 (Enhanced Features)
    ↓
    features_df (with node_id)
    ↓
Stage 3 (Enhanced MO Inference)
    ↓
    features_df (with predicted_role, mo_cluster, mo_score)
    ↓
Stage 4 (Enhanced SEAL)
    ↓
    G_aug, features_aug
```

### Key Interface Points

1. **Stage 1 → Stage 2**
   - Stage 1 outputs: `graph`, `persons`, `incidents`
   - Stage 2 needs: `graph`, `community_labels` (from persons), `node_ids`
   - ✅ Compatible

2. **Stage 2 → Stage 3**
   - Stage 2 outputs: DataFrame with `node_id` column
   - Stage 3 expects: DataFrame with features
   - ⚠️ **Fix**: Rename `node_id` to `person_id` or handle both
   - ✅ Handled in integrated pipeline

3. **Stage 3 → Stage 4**
   - Stage 3 outputs: `features_df` with `person_id`
   - Stage 4 needs: `graph`, optional `features_df`
   - ✅ Compatible

## Using the Integrated Pipeline

### Option 1: Use `pipeline_integrated.py`

```bash
python pipeline_integrated.py
```

This automatically:
- Detects which stages are available
- Uses enhanced stages if available, falls back to legacy
- Handles all interface conversions (node_id ↔ person_id)
- Maintains data flow between stages

### Option 2: Use Enhanced Stages Directly

```python
from stage1_enhanced_generator import EnhancedSyntheticCriminalNetwork
from stage2_enhanced_features import EnhancedFeatureEngineer
from stage3_enhanced_mo_inference import EnhancedMOInference
from stage4_enhanced_seal import EnhancedSEALLinkPredictor

# Stage 1
gen = EnhancedSyntheticCriminalNetwork(n_nodes=400, n_communities=3, core_fraction=0.22)
data_clean = gen.generate_network()
data_noisy = gen.add_noise(data_clean, missing_edge_rate=0.18, false_edge_rate=0.01)

# Stage 2
engineer = EnhancedFeatureEngineer(embedding_dim=128, q=0.5)
community_labels = dict(zip(data_noisy['persons']['person_id'], data_noisy['persons']['community']))
features_df = engineer.extract_features(
    data_noisy['graph'],
    community_labels=community_labels,
    node_ids=data_noisy['persons']['person_id'].tolist()
)

# Fix: Rename node_id to person_id
if 'node_id' in features_df.columns:
    features_df = features_df.rename(columns={'node_id': 'person_id'})

# Stage 3
inference = EnhancedMOInference()
true_roles = data_noisy['persons']['role'].values
predicted_roles, mo_clusters = inference.infer_mo_roles(features_df, true_roles=true_roles)
mo_scores = inference.compute_mo_importance_scores(features_df)

# Stage 4
predictor = EnhancedSEALLinkPredictor(confidence_threshold=0.7, use_drnl=True)
data_dict = predictor.prepare_link_prediction_data(data_noisy['graph'], features_df=features_df)
seal_results = predictor.train_seal(data_dict['train'], data_dict['val'])
predictions = predictor.predict_missing_links(data_noisy['graph'], features_df=features_df)
G_aug, stats = predictor.augment_graph(data_noisy['graph'], predictions)
```

## Common Issues and Fixes

### Issue 1: `node_id` vs `person_id`

**Problem**: Stage 2 outputs `node_id`, but Stage 3 expects `person_id`

**Fix**: Rename column
```python
if 'node_id' in features_df.columns:
    features_df = features_df.rename(columns={'node_id': 'person_id'})
```

### Issue 2: Missing Community Labels

**Problem**: Stage 2 needs community labels for participation coefficient

**Fix**: Extract from persons DataFrame
```python
if 'community' in persons_df.columns:
    community_labels = dict(zip(persons_df['person_id'], persons_df['community']))
else:
    community_labels = None  # Stage 2 will detect communities
```

### Issue 3: Missing Embedding Features

**Problem**: Stage 3 needs embedding features for MO scoring formula

**Fix**: Ensure Stage 2 extracts embeddings
```python
# Stage 2 should output emb_0, emb_1, ..., emb_127
# Check if present:
embedding_cols = features_df.filter(like='emb_').columns
assert len(embedding_cols) > 0, "No embedding features found!"
```

### Issue 4: Ground Truth Not Available

**Problem**: Stage 4 validation needs ground truth graph

**Fix**: Stage 1 saves it automatically
```python
# Stage 1 saves relations_ground_truth.csv if using enhanced generator
# Load it:
if os.path.exists('synthetic_criminal_network/relations_ground_truth.csv'):
    G_clean = load_graph_from_csv('synthetic_criminal_network/relations_ground_truth.csv')
```

## Testing Integration

Run the integration test:

```bash
python test_integration.py
```

This tests:
1. Module imports
2. Class definitions
3. Data flow between stages
4. Interface compatibility

## Configuration

Edit `CONFIG` in `pipeline_integrated.py`:

```python
CONFIG = {
    'USE_STAGE1': True,  # Enable Stage 1
    'USE_STAGE2': True,  # Enable Stage 2
    'USE_STAGE3': True,  # Enable Stage 3
    'USE_STAGE4': True,  # Enable Stage 4
    
    # Stage-specific parameters
    'CORE_FRACTION': 0.22,
    'MO_NODE2VEC_Q': 0.5,  # BFS bias
    'SEAL_CONFIDENCE_THRESHOLD': 0.7,
    'SEAL_USE_DRNL': True,
    # ...
}
```

## Troubleshooting

### All stages fail to import

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Stage 2 fails with "No embedding features"

**Solution**: Check Node2Vec installation
```bash
pip install node2vec
```

### Stage 4 fails with DRNL

**Solution**: DRNL is optional, set `use_drnl=False` if issues occur

### Data flow breaks between stages

**Solution**: Use `pipeline_integrated.py` which handles all conversions automatically

## Next Steps

1. Run integration test: `python test_integration.py`
2. Use integrated pipeline: `python pipeline_integrated.py`
3. Check outputs in `synthetic_criminal_network/` directory
