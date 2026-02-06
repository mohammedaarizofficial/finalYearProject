# STAGE 1 - Enhanced Synthetic Network Generation

## Overview

Stage 1 upgrades the synthetic network generator with dual-graph architecture, ground truth tracking, and controlled noise injection for quantitative ML validation.

## Key Improvements

### 1. Dual Graph Architecture

**Meetings Graph (Stochastic Block Model)**
- Community-based structure
- High probability of edges within communities
- Lower probability between communities
- Coordinators boost inter-community connectivity

**Communications Graph (Preferential Attachment)**
- Power-law structure (hub-and-spoke)
- High-degree nodes attract more connections
- Role-based attachment weights (Coordinators > Brokers > Enablers > Peripheral)
- Realistic growth patterns

**Combined Network**
- Union of both graphs
- Merged edge weights
- Preserves both structural patterns

### 2. Ground Truth Tracking

- `G_clean`: Complete ground-truth network
- `G_noisy`: Observed network with noise
- Tracks missing edges (recovery target for SEAL)
- Enables quantitative ML validation

### 3. Controlled Noise Injection

- **Missing edges**: 15-30% of original edges removed
- **False edges**: 5-10% of original edges added
- Preserves critical coordinator connections (85% retention)
- Validates density constraints

## Usage

### Basic Usage

```python
from synthetic_network_gen_v2 import EnhancedSyntheticCriminalNetwork

generator = EnhancedSyntheticCriminalNetwork(
    n_nodes=400,
    n_communities=3,
    core_fraction=0.22,  # Coordinators + Brokers
    seed=42
)

# Generate network
data_clean = generator.generate_network()

# Add noise
data_noisy = generator.add_noise(
    data_clean,
    missing_edge_rate=0.20,  # 20% missing
    false_edge_rate=0.05     # 5% false
)

# Save
generator.save_dataset(data_noisy, 'synthetic_criminal_network')
```

### Using with Main Pipeline

The main pipeline automatically uses the enhanced generator if available:

```python
# In main_pipeline.py CONFIG
CONFIG = {
    'USE_ENHANCED_GENERATOR': True,  # Enable Stage 1
    'N_NODES': 400,
    'N_COMMUNITIES': 3,
    'CORE_FRACTION': 0.22,
    'MISSING_EDGE_RATE': 0.20,
    'FALSE_EDGE_RATE': 0.05,
    # ... other config
}
```

Run the pipeline:
```bash
python main_pipeline.py
```

### Validation

Validate Stage 1 generation:

```bash
python validate_stage1.py --output-dir synthetic_criminal_network --plot
```

This will:
- Compare G_clean vs G_noisy
- Compute edge recovery metrics
- Measure structure preservation
- Generate validation plots

## Output Files

### Main Outputs
- `persons.csv`: Node attributes with MO roles
- `incidents.csv`: Crime incidents
- `relations.csv`: Noisy observed edges

### Ground Truth (New)
- `relations_ground_truth.csv`: Complete ground-truth edges
- Enables quantitative SEAL validation

## Metrics

### Edge Recovery Metrics
- **Missing edges**: Number of edges in G_clean but not in G_noisy
- **False edges**: Number of edges in G_noisy but not in G_clean
- **Precision**: Fraction of observed edges that are correct
- **Recall**: Fraction of true edges that were observed
- **F1**: Harmonic mean of precision and recall

### Structure Preservation
- **Density ratio**: How well network density is preserved
- **Clustering ratio**: How well clustering is preserved
- **LCC ratio**: Largest connected component preservation
- **Betweenness correlation**: Centrality structure preservation

## Integration with SEAL

Stage 1 provides the foundation for SEAL link prediction:

1. **Training Data**: SEAL trains on G_noisy to predict missing edges
2. **Validation**: Compare SEAL predictions against G_clean
3. **Recovery Target**: Number of missing edges to recover
4. **Success Metric**: Precision/Recall of recovered edges

## Configuration Parameters

### Network Generation
- `n_nodes`: Number of nodes (default: 400)
- `n_communities`: Number of communities (default: 3)
- `core_fraction`: Fraction of coordinators + brokers (default: 0.22)
- `seed`: Random seed for reproducibility

### Noise Injection
- `missing_edge_rate`: Fraction of edges to remove (0.15-0.30)
- `false_edge_rate`: Fraction of original edges to add as false (0.05-0.10)

### Role Distribution (default)
- Coordinator: 8%
- Broker: 14%
- Enabler: 20%
- Peripheral: 58%

## Comparison: Legacy vs Enhanced

| Feature | Legacy | Enhanced (Stage 1) |
|---------|--------|-------------------|
| Graph Model | Single custom model | Dual graphs (SBM + PA) |
| Ground Truth | Not tracked | Full tracking (G_clean) |
| Noise Control | Basic | Controlled (15-30% missing, 5-10% false) |
| Validation | Manual | Automated metrics |
| SEAL Integration | Limited | Full validation support |

## Next Steps

After Stage 1, proceed to:
- **Stage 2**: Enhanced MO feature extraction
- **Stage 3**: SEAL link prediction with ground truth validation
- **Stage 4**: Adaptive graph repair
- **Stage 5**: Interpretable disruption insights

## Troubleshooting

### Network too sparse
- Reduce `missing_edge_rate` (e.g., 0.20 → 0.15)
- Increase `p_in` in SBM (meetings graph)

### Network too dense
- Increase `missing_edge_rate` (e.g., 0.20 → 0.30)
- Reduce `p_in` in SBM

### Poor role separation
- Adjust `core_fraction`
- Try different `seed` values
- Modify role distribution

### Validation fails
- Ensure `relations_ground_truth.csv` exists
- Check that noise was added correctly
- Verify network generation completed

## Example Output

```
STAGE 1: ENHANCED SYNTHETIC NETWORK GENERATION
======================================================================

📊 Step 1: Assigning communities...
👥 Step 2: Assigning MO roles...
  ✅ Role distribution:
     Coordinator: 32 (8.0%)
     Broker: 56 (14.0%)
     Enabler: 80 (20.0%)
     Peripheral: 232 (58.0%)
🤝 Step 3: Generating Meetings graph (Stochastic Block Model)...
  ✅ Meetings graph: 400 nodes, 1245 edges
     Density: 0.0156
📡 Step 4: Generating Communications graph (Preferential Attachment)...
  ✅ Communications graph: 400 nodes, 795 edges
     Density: 0.0100
🔗 Step 5: Combining graphs into ground-truth network...
  ✅ Combined graph: 400 nodes, 1898 edges
     Density: 0.0238
📝 Step 6: Generating metadata...
✅ Step 7: Validating network quality...
  ✅ All quality metrics acceptable

🔇 Adding noise: 20.0% missing, 5.0% false
  Original edges: 1898
  Removed: 380 edges
  Added: 95 false edges
  Final edges: 1613
  Final density: 0.0202

📊 Ground Truth Tracking:
  Clean graph: 1898 edges
  Noisy graph: 1613 edges
  Missing edges: 285 (recovery target for SEAL)
```
