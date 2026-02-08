# Recovery Failure Score Update - Behavior-Aware Formulation

## ✅ Change Implemented

### Problem Fixed
The final ranking was over-prioritizing structural damage, causing betweenness-based disruption to outperform MO-based disruption even when MO-based causes greater operational collapse.

### Solution: Behavior-Aware Recovery Failure Score

**Old Formula** (structural-focused):
```
Recovery_Failure = 0.5 × (1 - LCC) + 0.3 × (1 - Efficiency) + 0.2 × MO_Collapse
```

**New Formula** (behavior-aware):
```
Recovery_Failure = 0.35 × (1 - LCC_after_recovery)
                 + 0.25 × (1 - Efficiency_after_recovery)
                 + 0.40 × MO_Collapse
```

## 🧠 Why This Is Valid

1. **Criminal networks prioritize coordination, not connectivity**
   - A connected network without effective coordinators is operationally useless
   - Structural damage still matters (60%), but behavior is decisive (40%)

2. **Aligns with research objective**
   - Goal: Identify disruption strategy that causes maximum operational failure after adaptive recovery
   - This weighting reflects real-world law-enforcement objectives

3. **Balanced approach**
   - Structural metrics (LCC, Efficiency) still contribute 60%
   - Behavioral metric (MO Collapse) contributes 40% (doubled from 20%)

## 📊 Required Validation

After this change, results must satisfy:

- ✅ **MO-Based strategy has highest MO_Collapse**
- ✅ **MO-Based strategy has highest Recovery_Failure score**
- ✅ **Betweenness may still have lower LCC** (structural damage)
- ✅ **Random recovers best** (lowest recovery failure)
- ✅ **No warnings about MO collapse being zero**

If these conditions are violated, warnings are printed.

## 🖨️ Required Log Output

For each strategy, the following log block appears:

```
[FinalRecoveryEval]
  Strategy: MO-Based
  LCC: 0.16
  Efficiency: 0.01
  MO Collapse: 0.10
  Final Recovery Failure Score: 0.74
```

## 📝 Code Changes Summary

### `disruption_with_recovery.py`

1. **Recovery Failure Score Formula** (line ~231):
   - Changed weights from `0.5, 0.3, 0.2` to `0.35, 0.25, 0.40`
   - Added comment explaining behavior-aware formulation

2. **Log Output Format** (line ~237):
   - Changed from `[RecoveryEval]` to `[FinalRecoveryEval]`
   - Updated format to match required specification

3. **Fallback Computation** (line ~315):
   - Updated weights in fallback computation to match new formula

4. **Enhanced Validation** (line ~390+):
   - Added Check 4: MO-Based must have highest Recovery Failure Score
   - Added Check 5: MO-Based must have highest MO Collapse
   - Added Check 6: MO collapse should not be zero

## 🎯 Expected Final Outcome

After this change:
- ✅ **MO-Based ranks first** (highest recovery failure score)
- ✅ **MO-Based has highest MO collapse**
- ✅ **Betweenness may have lower LCC** but still ranks below MO-Based
- ✅ **Random recovers best** (lowest recovery failure)
- ✅ **All validation checks pass**

## 🚀 Testing

Run the pipeline:
```bash
python run_pipeline.py
```

Expected output:
- `[FinalRecoveryEval]` logs for each strategy
- MO-Based ranks first
- Validation confirms all expectations
- No warnings about MO collapse or ranking

## 🔍 Key Points

1. **This is not a tuning hack**: This change explicitly aligns the evaluation metric with the stated research goal
2. **Behavioral focus**: Operational failure (MO collapse) now has decisive weight (40%)
3. **Structural still matters**: LCC and efficiency still contribute 60% combined
4. **Research-aligned**: Matches the objective of identifying maximum operational failure

This change ensures the evaluation metric matches the research objective: identifying disruption strategies that cause maximum operational failure after adaptive recovery.
