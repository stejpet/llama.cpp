# GFX900 TG Optimization Notes

## Attempted Optimizations That BROKE llama

### ❌ VDR Increases for MMVQ (BROKEN)
**Commits:**
- `c767785b6`: VDR_Q4_K_Q8_1_MMVQ 2→4
- `097a33387`: VDR_Q6_K_Q8_1_MMVQ 1→2

**Why it broke:**
- These VDR increases caused llama to fail (broken output/incorrect results)
- Even though benchmarks showed TG +4.2% and PP +5.2% improvements
- Likely causes register pressure or memory access issues that break correctness

**DO NOT TRY AGAIN**

### ❌ MMVQ num_warps 2→4 (BROKEN)
**Commit:** `3b9e0e732` (reverted in `8c3bea722`)

**Why it broke:**
- TG regressed from ~40 to ~30 t/s (-25%)
- BEAM search theory was wrong for this kernel
- Already documented in mar13 branch that this causes regression

**DO NOT TRY AGAIN**

---

## Current Working State (~40 t/s TG)

### ✅ Applied Optimizations (Working)

1. **DPP-based warp reductions** (`gated_delta_net.cu`, `mmvq.cu`, `mmvf.cu`)
   - Uses AMD GCN DPP instructions instead of shuffle
   - Correct and stable

2. **Fast memcpy loads in vecdotq.cuh**
   - Compiler emits efficient flat_load instructions
   - TG neutral or slight improvement

3. **VDR_Q8_0_Q8_1_MMQ 8→4**
   - Reduced VDR for MMQ (prompt processing)
   - Less register pressure, better occupancy
   - PP improvement, TG neutral

4. **num_warps 8→16 for MMQ**
   - Better occupancy on gfx900 64 CU architecture
   - PP +46.5% improvement

5. **gated_delta_net num_warps 4→8**
   - Significant PP improvement
   - TG neutral

---

## Next Steps for TG Optimization

**Current TG bottleneck:**
- `mul_mat_vec_q` is 52.7% of TG time (from mar13 profiling)
- Already has DPP warp reductions applied
- Current VDR values seem optimal

**Potential avenues (untested):**
1. Different memory access patterns in mmvq
2. Tile size adjustments
3. Software prefetching (was -2.5% regression in mar13 tests)
4. Vectorized loads in different parts of mmvq kernel

**Status:** TG at ~40 t/s is baseline after correctness fixes.
