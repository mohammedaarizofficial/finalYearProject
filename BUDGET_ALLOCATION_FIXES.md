# Budget Allocation + MO Collapse Metric Fixes

## ✅ TASK 1 — FIX ADAPTIVE REWIRING BUDGET ALLOCATION

### Problem Fixed
Previously, `bridge_restoration()` consumed the entire healing budget, preventing:
- Role substitution
- Triadic closure
- Community healing

### Solution Implemented

#### 1️⃣ Pre-allocated Healing Budget Per Mechanism
After computing `total_budget = int(0.15 * original_edge_count)`, budgets are pre-allocated:

| Mechanism | Budget Allocation | Hard Cap |
|-----------|------------------|----------|
| Bridge Restoration | 35% of total_budget | ✅ Enforced |
| Role Substitution | 15% of total_budget | ✅ Enforced |
| Triadic Closure | 25% of total_budget | ✅ Enforced |
| Community Healing | 25% of total_budget | ✅ Enforced |
| **Total** | **100% of total_budget** | **15% of original edges** |

#### 2️⃣ Independent Budget Enforcement
Each healing function:
- Tracks its own edge counter (`bridge_counter`, `role_counter`, etc.)
- Stops when its own budget is exhausted
- Never borrows from another mechanism
- Uses `_can_add_edge(mechanism_budget, mechanism_count)` for checks

#### 3️⃣ Execution Order (MANDATORY)
Healing always runs in this order:
1. Bridge restoration
2. Role substitution
3. Triadic closure
4. Community healing

No early exits unless the mechanism's own budget is exhausted.

#### 4️⃣ Logging (MANDATORY)
At the end of rewiring, logs:
```
[AdaptiveRewire] Healing budget: X
[AdaptiveRewire] Bridge restoration: A edges
[AdaptiveRewire] Role substitution: B edges
[AdaptiveRewire] Triadic closure: C edges
[AdaptiveRewire] Community healing: D edges
[AdaptiveRewire] Total edges added: A+B+C+D
```

## ✅ TASK 2 — FIX MO COLLAPSE METRIC (CRITICAL)

### Problem Fixed
`mo_collapse_after_recovery` was always 0.0, making behavioral analysis meaningless.

### Solution Implemented

#### Step 1️⃣ Capture Pre- and Post-Recovery Role Counts
For each strategy:
- `pre_counts = count_roles(G_disrupted)` - counts before recovery
- `post_counts = count_roles(G_recovered)` - counts after recovery
- Specifically tracks: Coordinators and Brokers

#### Step 2️⃣ Compute Percentage Drops
```
Coordinator drop = (pre - post) / max(pre, 1)
Broker drop      = (pre - post) / max(pre, 1)
```

#### Step 3️⃣ Measure Clustering Instability (Optional)
If HDBSCAN is used:
- Compute noise ratio before and after recovery
- `Noise increase = post_noise − pre_noise`
- If unavailable, set `noise_increase = 0`

#### Step 4️⃣ Final MO Collapse Formula (MANDATORY)
```
MO_Collapse = 0.5 × Coordinator_Drop
            + 0.3 × Broker_Drop
            + 0.2 × Noise_Increase
```
Result is clamped to [0, 1].

#### Step 5️⃣ Validation Check
Added check in `evaluate_strategy_with_recovery()`:
```python
if strategy_name == 'MO-Based' and mo_collapse_recovered == 0.0:
    print(f"⚠️  WARNING: MO-based strategy shows no behavioral collapse")
    print(f"     This may indicate over-healing. Check rewiring constraints.")
```

## 📊 Expected Final Behavior

After implementation:
- ✅ All healing mechanisms execute every run
- ✅ Bridge restoration does NOT exceed 35% of healing budget
- ✅ MO collapse is non-zero for MO-based strategy
- ✅ Random strategy recovers best
- ✅ LCC after recovery falls in ~0.5–0.8 range for most strategies

## 🔍 Code Changes Summary

### `stage5_adaptive_rewiring.py`
1. **Budget tracking**: Changed from global `self.edges_added` to per-mechanism counters
2. **Pre-allocation**: Added `self.bridge_budget`, `self.role_budget`, `self.triadic_budget`, `self.community_budget`
3. **Budget enforcement**: Updated `_can_add_edge()` and `_add_edge_with_budget()` to use mechanism-specific budgets
4. **All sub-methods**: Updated to track and enforce their own budgets independently
5. **MO collapse function**: Enhanced `compute_mo_collapse_continuous()` with detailed comments and proper formula

### `disruption_with_recovery.py`
1. **MO collapse validation**: Added warning if MO-Based strategy shows no collapse
2. **Continuous metric**: Uses `compute_mo_collapse_continuous()` for all MO collapse calculations

## 🚀 Testing

Run the pipeline:
```bash
python run_pipeline.py
```

Or force re-run Phase 4:
```bash
python pipeline_integrated.py --start-from 4 --force-rerun
```

Expected output:
- All mechanisms show edges added (not just bridge restoration)
- MO collapse > 0 for MO-Based strategy
- Clear budget allocation logs
- Validation warnings if behavior is unexpected

## 📝 Notes

- **Preferential attachment** was removed as a separate mechanism (can be included in triadic closure budget if needed)
- **Budget allocation** is fixed and cannot be exceeded by any mechanism
- **MO collapse** is now continuous and measures behavioral degradation, not just structural damage
- All function signatures preserved - no breaking changes
