# Visual Guide - What You'll See When Running

## 🎬 Running the Pipeline

### Step 1: Start the Pipeline

```bash
python run_pipeline.py
```

### Step 2: What You'll See

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
  ✅ sklearn
  ✅ hdbscan
  ✅ node2vec
  ✅ torch
  ✅ torch_geometric
  ✅ tqdm

✅ All dependencies installed!

📁 Checking stage files...
  ✅ Stage 1: stage1_enhanced_generator.py
  ✅ Stage 2: stage2_enhanced_features.py
  ✅ Stage 3: stage3_enhanced_mo_inference.py
  ✅ Stage 4: stage4_enhanced_seal.py
  ✅ Stage 5: stage5_adaptive_rewiring.py
  ✅ Stage 6: stage6_enhanced_visualization.py

======================================================================
  Ready to run pipeline? (y/n): y
```

### Step 3: Stage 1 Progress

```
======================================================================
  STARTING PIPELINE EXECUTION
======================================================================

📋 Configuration:
   Nodes: 400
   Communities: 3
   Output: synthetic_criminal_network
   Stage 1: ✅
   Stage 2: ✅
   Stage 3: ✅
   Stage 4: ✅

🚀 Starting pipeline...

======================================================================
  STAGE 1: ENHANCED SYNTHETIC DATA GENERATION
======================================================================
Generating network: 400 nodes, 3 communities
Core fraction: 22.0% (Coordinators + Brokers)

📅 Generating Meetings graph (Stochastic Block Model)...
   Meetings graph: 1245 edges

📞 Generating Communications graph (Preferential Attachment)...
   Communications graph: 368 edges

🔗 Combining graphs...
   Combined graph: 1613 edges
   Density: 0.0202

🔇 Adding noise to create observed graph...
   Removed 290 edges (18.0%)
   Added 16 false edges (1.0%)

📊 Noise Statistics:
   Clean graph: 1613 edges
   Noisy graph: 1339 edges
   Missing: 290 edges

✅ Dataset saved to synthetic_criminal_network/
```

### Step 4: Stage 2 Progress (SLOW - 5-10 minutes)

```
======================================================================
  STAGE 2: ENHANCED FEATURE ENGINEERING
======================================================================

1️⃣ Computing structural features...
   - Computing centrality metrics...
   ✅ Extracted 7 structural features

2️⃣ Computing community/ego features...
   - Computing community/ego features...
   ✅ Extracted 5 community/ego features

3️⃣ Computing embedding features (Node2Vec with BFS bias q=0.5)...
   - Computing Node2Vec embeddings (q=0.5, BFS bias)...
   [THIS TAKES 5-10 MINUTES - BE PATIENT!]
   ✅ Extracted 128 embedding features

🔗 Combining all features...
✅ Feature extraction complete!
   Total features: 140
   Nodes: 400
```

**⏰ During this stage:**
- CPU usage: 80-100%
- Console: May appear "stuck" but it's working!
- Wait: 5-10 minutes

### Step 5: Stage 3 Progress

```
======================================================================
  STAGE 3: ENHANCED MO INFERENCE
======================================================================

📊 Selecting features for MO inference...
   ✅ Selected 140 features

🔍 Performing HDBSCAN clustering...
   ✅ HDBSCAN found 4 clusters

🏷️  Mapping clusters to MO roles...
   ✅ Mapped 4 clusters to roles

💯 Computing MO importance scores...
   Formula: MOscore = 0.40*BC + 0.25*DC + 0.20*Participation + 0.15*EmbeddingInfluence
   ✅ Computed MO scores (range: [0.000, 1.000])

📈 Evaluating MO inference...
              precision    recall  f1-score   support
   Coordinator       0.75      0.68      0.71        32
   Broker           0.65      0.72      0.68        56
   Enabler          0.58      0.55      0.56        80
   Peripheral       0.92      0.94      0.93       232

📊 MO Inference Evaluation:
   Adjusted Rand Index: 0.623
   F1-Score (weighted): 0.712
   Normalized Mutual Info: 0.689

✅ MO Inference F1: 0.712 (target: 0.60)
```

### Step 6: Stage 4 Progress (SLOW - 10-30 minutes)

```
======================================================================
  STAGE 4: SEAL LINK PREDICTION DATA PREPARATION
======================================================================
  ✅ Using 140 features
  ✅ Initialized SEAL model: in_channels=141 (with DRNL)
  Split: train=960, val=120, test=120

  Extracting train subgraphs (with DRNL)...
  [Progress bar shows: 960/960]
  ✅ Saved train subgraphs

🚀 Training SEAL model...
Epoch 001 | Train Loss: 0.6234 | Val Loss: 0.5891 | Val AUC: 0.7123 | Val AP: 0.6543
  ✅ New best AUC: 0.7123
Epoch 010 | Train Loss: 0.5891 | Val Loss: 0.5712 | Val AUC: 0.7234 | Val AP: 0.6789
  ✅ New best AUC: 0.7234
...
Epoch 045 | Train Loss: 0.5234 | Val Loss: 0.5123 | Val AUC: 0.7456 | Val AP: 0.7123
  ⏹️  Early stopping at epoch 45

✅ Training complete! Best Val AUC: 0.7456 at epoch 45

🔮 Predicting missing links...
  Evaluating 2000 candidate pairs...
  [Progress bar shows: 2000/2000]
  ✅ Found 342 high-confidence predictions (≥0.7)

🔗 Augmenting graph with predicted edges...
  ✅ Added 342 predicted edges
  Original edges: 1339
  Augmented edges: 1681
  Recovery rate: 25.5%

📊 Validation against ground truth:
  Predicted edges: 342
  Correct predictions: 285
  Precision: 0.833

✅ SEAL AUC: 0.746 (target: 0.65)
```

**⏰ During this stage:**
- CPU usage: 80-100%
- Console: Shows epoch updates
- Wait: 10-30 minutes

### Step 7: Stage 5 Progress

```
======================================================================
  STAGE 5: ADAPTIVE REWIRING LAYER
======================================================================

🔍 Detecting communities...
   Detected 3 communities

1️⃣ Bridge Restoration...
   ✅ Restored 45 bridge edges

2️⃣ Role Substitution...
   Found 0 removed coordinators
   ✅ Made 0 role substitution connections

3️⃣ Triadic Closure...
   ✅ Added 78 triadic closure edges

4️⃣ Preferential Attachment...
   ✅ Added 12 preferential attachment edges

5️⃣ Community Healing...
   ✅ Added 23 community healing edges

✅ Rewiring complete!
   Original edges: 1681
   Healed edges: 1839
   Edges added: 158
   Rewiring operations: 158
```

### Step 8: Stage 6 Progress

```
======================================================================
  STAGE 6: GENERATING ALL VISUALIZATIONS & REPORTS
======================================================================

1️⃣ Embedding Dimensionality Reduction (PCA)...
   Found 128 embedding dimensions
   Applying PCA reduction to 2D...
   Explained variance: 45.2%
   ✅ Saved to embedding_pca_by_role.png

2️⃣ Ego-Graph Visualizations...
   📊 Plotting ego-graph for node 42...
   ✅ Saved to ego_graph_node_42.png
   ...

3️⃣ Enhanced Disruption Curves...
   📈 Plotting enhanced disruption curves...
   ✅ Saved to enhanced_disruption_curves.png

4️⃣ Network Comparison...
   🔍 Plotting network comparison...
   ✅ Saved to network_comparison.png

5️⃣ MO Score Distribution...
   📊 Plotting MO score distribution...
   ✅ Saved to mo_score_distribution.png

6️⃣ Final Report...
   ✅ Final report saved to FINAL_REPORT.md

✅ All visualizations and reports generated!
   Output directory: synthetic_criminal_network/
```

### Step 9: Completion

```
======================================================================
  PIPELINE COMPLETE!
======================================================================
  Total time: 1245.3 seconds (20.8 minutes)
  Output directory: synthetic_criminal_network/
======================================================================

📂 Output Files:
======================================================================
  📄 persons.csv (45.2 KB)
  📄 incidents.csv (123.4 KB)
  📄 relations.csv (78.9 KB)
  📄 relations_ground_truth.csv (82.1 KB)
  📄 node_features_with_mo.csv (234.5 KB)
  📄 seal_predictions.csv (156.7 KB)
  📄 relations_seal_augmented.csv (89.3 KB)
  📄 disruption_summary.csv (2.1 KB)
  📄 FINAL_REPORT.md (8.5 KB)
  📄 embedding_pca_by_role.png (234.5 KB)
  📄 enhanced_disruption_curves.png (456.7 KB)
  📄 network_comparison.png (345.6 KB)
  ...

✅ Pipeline execution complete!

📊 Next steps:
   1. Check output files in synthetic_criminal_network/
   2. Review FINAL_REPORT.md for summary
   3. View generated visualizations (.png files)
```

## 🎯 Key Indicators It's Working

### ✅ Good Signs:
- ✅ Progress messages appearing
- ✅ Stage names showing
- ✅ "✅" checkmarks appearing
- ✅ Files being created
- ✅ CPU usage high (during Stage 2 & 4)
- ✅ No error messages

### ⚠️ Warning Signs:
- ⚠️ No output for 15+ minutes during Stage 2 or 4 (might be normal, but check CPU)
- ⚠️ Error messages (not warnings)
- ⚠️ Process stopped/crashed

### ❌ Problem Signs:
- ❌ "ModuleNotFoundError" → Install dependencies
- ❌ "FileNotFoundError" → Check file paths
- ❌ Process died → Check error messages

## 📊 Real-Time Monitoring

**Open 3 terminals:**

**Terminal 1: Run pipeline**
```bash
python run_pipeline.py
```

**Terminal 2: Watch files**
```bash
watch -n 2 'ls -lht synthetic_criminal_network/ | head -15'
```

**Terminal 3: Check status**
```bash
watch -n 10 'python check_status.py'
```

This gives you:
- Terminal 1: Detailed progress
- Terminal 2: Files being created
- Terminal 3: Overall status

## 🎉 Success!

When you see:
```
======================================================================
  PIPELINE COMPLETE!
======================================================================
```

And:
```bash
ls synthetic_criminal_network/FINAL_REPORT.md
# File exists!
```

**You're done!** 🎉
