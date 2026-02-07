# Criminal Network Analysis - Enhanced ML Framework

A robust, ML-enhanced, self-healing network framework for analyzing criminal networks with missing/noisy data.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python run_pipeline.py
```

This will:
- ✅ Check all dependencies
- ✅ Verify all stage files
- ✅ Run the complete pipeline
- ✅ Show progress in real-time

### 3. Check Status

```bash
python check_status.py
```

Shows what stages have been completed and what files exist.

## 📋 What This System Does

### Stage 0: Baseline (Already Exists)
- Graph creation, centrality metrics, Node2Vec, HDBSCAN, disruption simulation

### Stage 1: Enhanced Synthetic Data Generation
- Creates controlled networks with known ground truth
- Dual graphs (Meetings + Communications)
- Tracks clean vs noisy graphs

### Stage 2: Enhanced Feature Engineering
- BFS-biased Node2Vec embeddings (q < 1)
- Structural + Community + Embedding features
- 128-dimensional feature vectors

### Stage 3: Enhanced MO Inference
- HDBSCAN clustering for natural groups
- MO scoring formula: `0.40*BC + 0.25*DC + 0.20*Participation + 0.15*EmbeddingInfluence`
- Role prediction and evaluation

### Stage 4: Enhanced SEAL Link Prediction
- DRNL node labeling
- 2-hop enclosing subgraphs
- Recovers missing links with confidence scores
- Graph augmentation

### Stage 5: Adaptive Rewiring Layer
- Bridge restoration
- Role substitution
- Triadic closure
- Preferential attachment
- Community healing

### Stage 6: Enhanced Visualization & Reporting
- PCA/UMAP dimensionality reduction
- Ego-graph visualizations
- Enhanced disruption curves
- Comprehensive final report

## 📁 Project Structure

```
finalYearProject/
├── stage1_enhanced_generator.py      # Stage 1: Dual graph generation
├── stage2_enhanced_features.py       # Stage 2: BFS-biased Node2Vec
├── stage3_enhanced_mo_inference.py   # Stage 3: HDBSCAN + MO scoring
├── stage4_enhanced_seal.py           # Stage 4: DRNL + link recovery
├── stage5_adaptive_rewiring.py       # Stage 5: Adaptive healing
├── stage6_enhanced_visualization.py  # Stage 6: Visualization & reports
├── pipeline_integrated.py            # Integrated pipeline (use this!)
├── run_pipeline.py                   # Simple runner with progress
├── check_status.py                   # Status checker
├── test_integration.py               # Integration tests
├── main_pipeline.py                  # Original pipeline (legacy)
├── requirements.txt                  # Dependencies
├── QUICK_START.md                    # Quick start guide
├── INTEGRATION_GUIDE.md              # Integration documentation
└── .gitignore                        # Git ignore (excludes venv, outputs)
```

## 🎯 How to See if It's Running

### Method 1: Console Output
The pipeline shows detailed progress:
```
STAGE 1: ENHANCED SYNTHETIC DATA GENERATION
Generating network: 400 nodes, 3 communities
📅 Generating Meetings graph...
📞 Generating Communications graph...
✅ Saved dataset
```

### Method 2: Status Checker
```bash
python check_status.py
```

Shows:
- ✅ Which stages are complete
- 📊 File counts and statistics
- ⏳ Overall progress

### Method 3: File Watching
```bash
# Watch output directory
watch -n 2 'ls -lht synthetic_criminal_network/ | head -10'
```

### Method 4: Process Check
```bash
# Check if Python is running
ps aux | grep python

# Check CPU usage (should be high if running)
top -p $(pgrep -f "python.*pipeline")
```

## ⏱️ Expected Runtime

**Full pipeline (400 nodes):**
- Stage 1: ~30 seconds
- Stage 2: ~5-10 minutes (Node2Vec is slow)
- Stage 3: ~1-2 minutes
- Stage 4: ~10-30 minutes (SEAL training)
- Stage 5: ~30 seconds
- Stage 6: ~1 minute

**Total: ~20-45 minutes**

**Quick test (50 nodes):**
- Change `N_NODES: 50` in `pipeline_integrated.py`
- Total: ~5-10 minutes

## 📊 Output Files

After running, check `synthetic_criminal_network/`:

**Data Files:**
- `persons.csv` - Person data with roles
- `incidents.csv` - Incident records
- `relations.csv` - Network edges (noisy)
- `relations_ground_truth.csv` - Ground truth (Stage 1)
- `node_features_with_mo.csv` - Features + MO predictions (Stage 2-3)
- `seal_predictions.csv` - Predicted links (Stage 4)
- `relations_seal_augmented.csv` - Augmented network (Stage 4)

**Results:**
- `disruption_summary.csv` - Disruption effectiveness
- `FINAL_REPORT.md` - Comprehensive report (Stage 6)

**Visualizations:**
- `embedding_pca_by_role.png` - PCA visualization
- `enhanced_disruption_curves.png` - Disruption analysis
- `network_comparison.png` - Original vs Augmented
- `mo_score_distribution.png` - MO score distribution
- `ego_graph_node_*.png` - Ego-graphs for key nodes

## 🔧 Configuration

Edit `pipeline_integrated.py` to adjust:

```python
CONFIG = {
    'N_NODES': 400,              # Network size
    'N_COMMUNITIES': 3,          # Number of communities
    'CORE_FRACTION': 0.22,       # Coordinators + Brokers
    'USE_STAGE1': True,          # Enable Stage 1
    'USE_STAGE2': True,          # Enable Stage 2
    'USE_STAGE3': True,          # Enable Stage 3
    'USE_STAGE4': True,          # Enable Stage 4
    'SEAL_CONFIDENCE_THRESHOLD': 0.7,
    # ... more options
}
```

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Pipeline seems stuck
- **Stage 2 (Node2Vec)** and **Stage 4 (SEAL)** are slow - this is normal!
- Check CPU usage - should be high if running
- Look at console for current stage

### Out of memory
- Reduce `N_NODES` (400 → 200)
- Reduce `SEAL_MAX_SAMPLES` (1200 → 600)

### No output files
- Check if pipeline is still running
- Verify write permissions
- Check for error messages

## 📚 Documentation

- **QUICK_START.md** - Step-by-step getting started guide
- **INTEGRATION_GUIDE.md** - How stages work together
- **FINAL_REPORT.md** - Generated after pipeline completes

## ✅ Success Indicators

**Pipeline is working if:**
- ✅ Progress messages appear for each stage
- ✅ Files are being created in output directory
- ✅ No error messages (warnings are OK)
- ✅ CPU usage is high during Stage 2 and Stage 4

**Pipeline completed successfully if:**
- ✅ `FINAL_REPORT.md` exists
- ✅ All visualization PNG files exist
- ✅ Console shows "PIPELINE COMPLETE!"
- ✅ Status checker shows all stages complete

## 🎓 Usage Examples

### Run Full Pipeline
```bash
python run_pipeline.py
```

### Check Current Status
```bash
python check_status.py
```

### Run from Specific Stage
```bash
python pipeline_integrated.py --start-from 3
```

### Force Rerun Everything
```bash
python pipeline_integrated.py --force-rerun
```

### Test Integration
```bash
python test_integration.py
```

## 📝 Notes

- **First run**: Will take 20-45 minutes (full pipeline)
- **Subsequent runs**: Faster if using checkpoints (skip completed stages)
- **Output directory**: `synthetic_criminal_network/` (excluded from git)
- **Checkpoints**: Saved after each stage for resuming

## 🔗 Integration

All stages are integrated and work together:
- Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
- Data flows automatically between stages
- Interface conversions handled automatically (node_id ↔ person_id)

See `INTEGRATION_GUIDE.md` for details.

## 📄 License

This project is part of a final year project for criminal network analysis.
