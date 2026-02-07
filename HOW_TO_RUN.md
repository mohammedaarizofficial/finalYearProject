# How to Run and Monitor the Pipeline

## 🚀 Step-by-Step Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**This installs:**
- numpy, pandas, networkx
- scikit-learn, hdbscan
- node2vec
- torch, torch-geometric
- matplotlib, seaborn
- tqdm

**Time:** ~5-10 minutes (depending on internet speed)

### Step 2: Run the Pipeline

```bash
python run_pipeline.py
```

**What happens:**
1. Checks if dependencies are installed ✅
2. Checks if all stage files exist ✅
3. Asks if you want to proceed
4. Runs the complete pipeline
5. Shows progress for each stage

### Step 3: Monitor Progress

**You'll see output like this:**

```
======================================================================
  CRIMINAL NETWORK ANALYSIS - ENHANCED PIPELINE
======================================================================

🔍 Checking dependencies...
  ✅ numpy
  ✅ pandas
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
📅 Generating Meetings graph...
📞 Generating Communications graph...
✅ Saved dataset

======================================================================
  STAGE 2: ENHANCED FEATURE ENGINEERING
======================================================================
1️⃣ Computing structural features...
2️⃣ Computing community/ego features...
3️⃣ Computing embedding features (Node2Vec with BFS bias q=0.5)...
   [This takes 5-10 minutes - BE PATIENT!]
...
```

## 👀 How to See if It's Running

### Method 1: Watch the Console
**Look for:**
- ✅ Progress messages appearing
- ✅ Stage names (STAGE 1, STAGE 2, etc.)
- ✅ File creation messages ("✅ Saved...")
- ✅ No error messages

**What you'll see:**
- Stage 1: Fast (~30 seconds)
- Stage 2: **SLOW** (~5-10 min) - Node2Vec is computing embeddings
- Stage 3: Fast (~1-2 min)
- Stage 4: **SLOW** (~10-30 min) - SEAL is training
- Stage 5: Fast (~30 seconds)
- Stage 6: Fast (~1 min)

### Method 2: Check Status

**In a new terminal:**
```bash
python check_status.py
```

**Shows:**
```
📊 Stage 1: Synthetic Data Generation
  ✅ COMPLETE
     - 400 persons
     - 1613 relations

📊 Stage 2-3: Feature Engineering & MO Inference
  ✅ COMPLETE
     - 400 nodes with features
     - Role distribution: ...

📊 Stage 4: SEAL Link Prediction
  ⏳ IN PROGRESS (if files are being created)
  ✅ COMPLETE (if all files exist)
```

### Method 3: Watch Files Being Created

**In a new terminal:**
```bash
# Watch the output directory
watch -n 2 'ls -lht synthetic_criminal_network/ | head -15'
```

**You'll see files appear as stages complete:**
- `persons.csv` (Stage 1)
- `node_features_with_mo.csv` (Stage 2-3)
- `seal_predictions.csv` (Stage 4)
- `FINAL_REPORT.md` (Stage 6)

### Method 4: Check Process

```bash
# See if Python is running
ps aux | grep python

# See CPU usage (should be high if running)
top
```

**Look for:**
- High CPU usage (50-100%) = Pipeline is working
- Low CPU usage (<10%) = Might be stuck or waiting

## ⏱️ Expected Timeline

**Full Run (400 nodes):**
```
00:00 - Start
00:30 - Stage 1 complete ✅
05:00 - Stage 2 complete ✅ (Node2Vec takes time!)
07:00 - Stage 3 complete ✅
25:00 - Stage 4 complete ✅ (SEAL training takes time!)
26:00 - Stage 5 complete ✅
27:00 - Stage 6 complete ✅
27:00 - DONE! 🎉
```

**Quick Test (50 nodes):**
```
00:00 - Start
00:10 - Stage 1 complete ✅
02:00 - Stage 2 complete ✅
03:00 - Stage 3 complete ✅
08:00 - Stage 4 complete ✅
09:00 - Stage 5 complete ✅
10:00 - Stage 6 complete ✅
10:00 - DONE! 🎉
```

## 🎯 Success Indicators

### ✅ Pipeline is Running if:
1. **Console shows progress messages**
   ```
   STAGE 2: ENHANCED FEATURE ENGINEERING
   1️⃣ Computing structural features...
   ```

2. **Files are being created**
   ```bash
   ls -lht synthetic_criminal_network/
   # You'll see new files appearing
   ```

3. **CPU usage is high**
   - Stage 2: High CPU (Node2Vec)
   - Stage 4: High CPU (SEAL training)
   - Other stages: Lower CPU

4. **No error messages**
   - Warnings are OK
   - Errors will stop the pipeline

### ✅ Pipeline Completed if:
1. **Console shows:**
   ```
   ======================================================================
     PIPELINE COMPLETE!
   ======================================================================
   ```

2. **Final report exists:**
   ```bash
   ls synthetic_criminal_network/FINAL_REPORT.md
   ```

3. **All visualization files exist:**
   ```bash
   ls synthetic_criminal_network/*.png
   ```

4. **Status checker shows all complete:**
   ```bash
   python check_status.py
   # Shows: Overall Progress: 5/5 stages complete
   ```

## 🐛 Common Issues

### Issue: "ModuleNotFoundError"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Pipeline seems stuck at Stage 2
**This is NORMAL!** Node2Vec takes 5-10 minutes.
- ✅ Check CPU usage (should be high)
- ✅ Wait patiently
- ✅ Look for "Computing embedding features..." message

### Issue: Pipeline seems stuck at Stage 4
**This is NORMAL!** SEAL training takes 10-30 minutes.
- ✅ Check CPU usage (should be high)
- ✅ Wait patiently
- ✅ Look for "Training SEAL model..." messages
- ✅ You'll see epoch updates every 10 epochs

### Issue: Out of memory
**Solution:**
1. Edit `pipeline_integrated.py`
2. Change:
   ```python
   'N_NODES': 200,  # Reduced from 400
   'SEAL_MAX_SAMPLES': 600,  # Reduced from 1200
   ```

### Issue: No output files
**Check:**
1. Is pipeline still running? (check console)
2. Any error messages?
3. Write permissions:
   ```bash
   touch synthetic_criminal_network/test.txt
   ```

## 📊 What to Expect

### During Stage 2 (Node2Vec):
- **Console:** "Computing embedding features..."
- **Time:** 5-10 minutes
- **CPU:** High (80-100%)
- **Files:** None yet (still computing)

### During Stage 4 (SEAL):
- **Console:** 
  ```
  Epoch 001 | Train Loss: 0.6234 | Val AUC: 0.7123
  Epoch 010 | Train Loss: 0.5891 | Val AUC: 0.7234
  ...
  ```
- **Time:** 10-30 minutes
- **CPU:** High (80-100%)
- **Files:** `seal_checkpoints/` directory being created

### After Completion:
- **Files:** All CSV, PNG, MD files in `synthetic_criminal_network/`
- **Report:** `FINAL_REPORT.md` with all metrics
- **Visualizations:** Multiple PNG files

## 🎓 Quick Test

**To test quickly (5-10 minutes):**

1. Edit `pipeline_integrated.py`, line ~30:
   ```python
   'N_NODES': 50,  # Change from 400
   'SEAL_MAX_SAMPLES': 100,  # Change from 1200
   ```

2. Run:
   ```bash
   python run_pipeline.py
   ```

3. Should complete in ~5-10 minutes

## 📞 Need Help?

1. **Check status:**
   ```bash
   python check_status.py
   ```

2. **Check console output** for specific errors

3. **Read QUICK_START.md** for more details

4. **Check INTEGRATION_GUIDE.md** for integration issues

## ✅ Final Checklist

Before running:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] All stage files present (`ls stage*.py`)
- [ ] Output directory exists or will be created

While running:
- [ ] Console shows progress
- [ ] CPU usage is high (during Stage 2 & 4)
- [ ] Files appearing in output directory

After completion:
- [ ] `FINAL_REPORT.md` exists
- [ ] Visualization PNG files exist
- [ ] Status checker shows all complete
