# Quick Start Guide

## How to Run the Enhanced Pipeline

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy, pandas, networkx
- scikit-learn, hdbscan
- node2vec
- torch, torch-geometric
- tqdm, matplotlib, seaborn
- (optional) umap-learn for UMAP visualization

### Step 2: Check Everything is Ready

```bash
python run_pipeline.py
```

This will:
- ✅ Check if all dependencies are installed
- ✅ Check if all stage files exist
- ✅ Show configuration
- ✅ Run the pipeline with progress tracking

### Step 3: Monitor Progress

The pipeline will show real-time progress:

```
======================================================================
  CRIMINAL NETWORK ANALYSIS - ENHANCED PIPELINE
======================================================================
  Started: 2024-01-15 10:30:00
======================================================================

🔍 Checking dependencies...
  ✅ numpy
  ✅ pandas
  ✅ networkx
  ...

📁 Checking stage files...
  ✅ Stage 1: stage1_enhanced_generator.py
  ✅ Stage 2: stage2_enhanced_features.py
  ...

🚀 Starting pipeline...

======================================================================
  STAGE 1: ENHANCED SYNTHETIC DATA GENERATION
======================================================================
Generating network: 400 nodes, 3 communities
...
```

### Step 4: Check Outputs

After completion, check the output directory:

```bash
ls -lh synthetic_criminal_network/
```

**Expected outputs:**
- `persons.csv` - Person data with roles
- `incidents.csv` - Incident records
- `relations.csv` - Network edges (noisy)
- `relations_ground_truth.csv` - Ground truth edges (Stage 1)
- `node_features_with_mo.csv` - Features with MO predictions (Stage 2-3)
- `seal_predictions.csv` - Predicted missing links (Stage 4)
- `relations_seal_augmented.csv` - Augmented network (Stage 4)
- `FINAL_REPORT.md` - Comprehensive report (Stage 6)
- `*.png` - Visualizations (Stage 6)

## Alternative: Run Individual Stages

### Option 1: Use Integrated Pipeline

```bash
python pipeline_integrated.py
```

### Option 2: Use Original Pipeline (Legacy)

```bash
python main_pipeline.py
```

### Option 3: Test Integration

```bash
python test_integration.py
```

## How to See if It's Running

### Visual Indicators:

1. **Console Output**: You'll see progress messages:
   ```
   STAGE 1: ENHANCED SYNTHETIC DATA GENERATION
   Generating network: 400 nodes, 3 communities
   📅 Generating Meetings graph...
   📞 Generating Communications graph...
   ```

2. **File Creation**: Watch for files being created:
   ```bash
   # In another terminal, watch the output directory
   watch -n 2 'ls -lht synthetic_criminal_network/ | head -10'
   ```

3. **Process Status**: Check if Python is running:
   ```bash
   ps aux | grep python
   ```

4. **CPU/Memory Usage**: Pipeline uses CPU during:
   - Stage 1: Network generation (fast)
   - Stage 2: Node2Vec embeddings (slow, ~5-10 min)
   - Stage 3: HDBSCAN clustering (moderate)
   - Stage 4: SEAL training (slow, ~10-30 min)
   - Stage 5: Rewiring (fast)
   - Stage 6: Visualization (fast)

## Troubleshooting

### Problem: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: Pipeline seems stuck

**Check:**
1. Which stage is it on? (look at console output)
2. Stage 2 (Node2Vec) and Stage 4 (SEAL) are slow - this is normal
3. Check CPU usage - should be high if running

**If truly stuck:**
- Press Ctrl+C to stop
- Check error messages
- Try running from a specific stage:
  ```bash
  python pipeline_integrated.py --start-from 3
  ```

### Problem: Out of memory

**Solution:**
- Reduce `N_NODES` in CONFIG (e.g., 400 → 200)
- Reduce `SEAL_MAX_SAMPLES` (e.g., 1200 → 600)
- Reduce `MO_EMBEDDING_DIM` (e.g., 128 → 64)

### Problem: No output files

**Check:**
1. Is the pipeline still running?
2. Check for errors in console
3. Verify write permissions:
   ```bash
   touch synthetic_criminal_network/test.txt
   ```

## Expected Runtime

**Full pipeline (400 nodes):**
- Stage 1: ~30 seconds
- Stage 2: ~5-10 minutes (Node2Vec is slow)
- Stage 3: ~1-2 minutes
- Stage 4: ~10-30 minutes (SEAL training)
- Stage 5: ~30 seconds
- Stage 6: ~1 minute

**Total: ~20-45 minutes** (depending on hardware)

**Quick test (50 nodes):**
- Change `N_NODES: 50` in CONFIG
- Total: ~5-10 minutes

## Monitoring Progress

### Method 1: Console Output
Just watch the terminal - it shows detailed progress for each stage.

### Method 2: File Watching
```bash
# Watch output directory
watch -n 5 'ls -lht synthetic_criminal_network/ | head -15'
```

### Method 3: Checkpoint Files
Pipeline saves checkpoints after each phase:
- `checkpoint_phase1.pkl`
- `checkpoint_phase2.pkl`
- `checkpoint_phase3.pkl`
- etc.

If these exist, the pipeline can resume from that point.

### Method 4: Log File (if you redirect)
```bash
python run_pipeline.py 2>&1 | tee pipeline.log
```

Then monitor:
```bash
tail -f pipeline.log
```

## Success Indicators

✅ **Pipeline is working if you see:**
1. Progress messages for each stage
2. Files being created in output directory
3. No error messages (warnings are OK)
4. Final summary with metrics

✅ **Pipeline completed successfully if:**
1. `FINAL_REPORT.md` exists
2. All visualization PNG files exist
3. CSV files with data exist
4. Console shows "PIPELINE COMPLETE!"

## Quick Test Run

To test quickly with smaller network:

1. Edit `pipeline_integrated.py`:
   ```python
   CONFIG = {
       'N_NODES': 50,  # Reduced from 400
       'SEAL_MAX_SAMPLES': 100,  # Reduced from 1200
       ...
   }
   ```

2. Run:
   ```bash
   python run_pipeline.py
   ```

This should complete in ~5-10 minutes.

## Need Help?

1. Check `INTEGRATION_GUIDE.md` for detailed integration info
2. Run `python test_integration.py` to diagnose issues
3. Check console output for specific error messages
