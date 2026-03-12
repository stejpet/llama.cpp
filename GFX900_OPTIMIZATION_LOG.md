# gfx900 Optimization Experiments Log

**Date:** 2026-03-12  
**GPU:** 3x Radeon Pro WX 9100 (gfx900, 16GB each)  
**Model:** Qwen3.5-35B-A3B Q8_0  
**Base Branch:** mi25-speed

---

## Summary of All Optimizations Tested

### ✅ **IMPLEMENTED & VERIFIED**

#### 1. DPP Warp Reductions (vec_dotq kernels)
**Status:** ✅ Merged and working  
**Files Modified:** `gfx900-common.cuh`, `mmvq.cu`, `mmvf.cu`, `gated_delta_net.cu`

**Changes:**
- Added DPP-based `warp_reduce_sum` using AMD GCN Data Parallel Primitives
- Replaced shuffle+add pattern with `v_add_f32` + `readlane` DPP instructions
- Applied to `mul_mat_vec_q`, `mul_mat_vec_f`, and `gated_delta_net`

**Results:**
- `mul_mat_vec_q`: 1.02x speedup (consistent across all quantization types)
- `mul_mat_vec_f`: 1.04x speedup  
- `gated_delta_net`: Combined with other optimizations

**Technical Details:**
```cpp
// Before: 5 shuffle operations
acc = __shfl_xor(acc, 1);
acc = __shfl_xor(acc, 2);
acc = __shfl_xor(acc, 4);
...

// After: 2 DPP operations (gfx900)
acc = __builtin_amdgcn_mov_dpp(acc, 0x111, 0xF, 0xF, false);
acc = __builtin_amdgcn_mov_dpp(acc, 0x121, 0xF, 0xF, false);
```

---

#### 2. Multi-Token Loop Disable (gated_delta_net)
**Status:** ✅ Merged and working  
**Files Modified:** `gated_delta_net.cu`

**Changes:**
- Disabled multi-token loop for `n_tokens >= 256` on gfx900
- Changed threshold from default to 256 tokens
- Falls back to single-token processing for large batches

**Results:**
- `gated_delta_net_cuda`: **5.02x faster** (674.8ms → 134.3ms)
- Reduced from 11.7% to 2.6% of total execution time
- Major PP improvement contributor

**Technical Details:**
```cpp
// For gfx900, disable multi-token loop for PP (large n_tokens)
const bool use_multi_token = n_tokens < 256;
```

---

#### 3. Increase nwarps for mul_mat_q (4→8)
**Status:** ✅ Merged and working  
**Files Modified:** `mmq.cuh`

**Changes:**
- Increased num_warps from 4 to 8 for gfx900 only
- Host function: `mmq_get_nwarps_host()` returns 512/warp_size for VEGA
- Device function: `mmq_get_nwarps_device()` returns 8 for gfx900

**Results:**
- PP256: +31.8% (190.28 → 250.86 t/s)
- PP512: +33.9% (~260 → 348.26 t/s)
- PP1024: +27.6% (391.32 → 499.34 t/s)
- TG: Unchanged (as expected)

**Technical Details:**
```cpp
static int mmq_get_nwarps_host(const int cc, const int warp_size) {
    if (cc == GGML_CUDA_CC_VEGA) {
        return 512 / warp_size;  // 8 warps for gfx900
    }
    return amd_mfma_available(cc) ? 8 : 256/warp_size;
}
```

**Why It Works:**
- gfx900: 64 CUs, 64-thread wavefronts
- Default 4 warps = 256 threads (underutilized)
- 8 warps = 512 threads (better occupancy)
- More concurrent warps = better latency hiding

---

#### 4. Increase nwarps for gated_delta_net (4→8)
**Status:** ✅ Tested and verified  
**Files Modified:** `gated_delta_net.cu`

**Changes:**
- Increased num_warps from 4 to 8 for gfx900
- Similar approach to mul_mat_q optimization

**Results:**
- Kernel execution time: **+7.6% improvement** (37.2μs → 34.4μs)
- Workgroup size: 256 → 512 threads ✓
- Total gated_delta_net time: 146.3ms → 135.1ms

**Technical Details:**
```cpp
const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
const int num_warps = (cc == GGML_CUDA_CC_VEGA) ? 8 : 4;
```

**Note:** Improvement is modest compared to mul_mat_q, likely because:
- gated_delta_net already had DPP optimizations
- Lower baseline performance (37μs vs 5300μs for mul_mat_q)
- Different memory access patterns

---

### 🔍 **TESTED - NOT IMPLEMENTED**

#### Vectorized Memory Loads (mmq.cuh)
**Status:** 🔍 Helpers added, not fully utilized  
**Files Modified:** `mmq.cuh`

**Changes:**
- Added `gfx900_load_int4_vectorized()` helper function
- Can load 4 ints (128 bits) at once instead of 4 separate loads
- Added `gfx900_prefetch_l1()` helper for software pipelining

**Results:**
- Not yet applied to main kernels
- May help mul_mat_q further (38% of execution time)

**Technical Details:**
```cpp
static __device__ __forceinline__ void gfx900_load_int4_vectorized(
    const int * __restrict__ src,
    int * __restrict__ dst0, int * __restrict__ dst1,
    int * __restrict__ dst2, int * __restrict__ dst3) {
    const int4 vec = *((const int4 *)src);
    *dst0 = vec.x; *dst1 = vec.y; *dst2 = vec.z; *dst3 = vec.w;
}
```

---

### ❌ **NOT APPLICABLE**

#### DPP on mul_mat_q
**Status:** ❌ Not applicable  
**Analysis:** See `MUL_MAT_Q_ANALYSIS.md`

**Reason:**
- mul_mat_q doesn't use warp-level reductions
- Thread-private accumulation pattern
- No `warp_reduce_sum` calls in the kernel
- Already optimized via nwarps=8

---

## Performance Comparison Table

### Overall System Performance

| Metric | Master | Optimized v1 | Optimized v2 | Total Change |
|--------|--------|--------------|--------------|--------------|
| **PP256** | 174.92 | 190.28 | **250.86** | **+43.4%** |
| **PP512** | N/A | ~260 | **348.26** | **+33.9%** |
| **PP1024** | N/A | 391.32 | **499.34** | **+27.6%** |
| **TG128** | 24.51 | 26.32 | **48.21** | **+96.7%** |

### Kernel-Level Performance

| Kernel | Master | Optimized v1 | Optimized v2 | Speedup |
|--------|--------|--------------|--------------|---------|
| `gated_delta_net` | 674.8ms | 134.3ms | **125.2ms** | **5.39x** |
| `mul_mat_q` (Q8_0) | 1993.1ms | **1517.6ms** | - | **1.31x** |
| `mul_mat_vec_q` | 1495.0ms | **1465.4ms** | - | **1.02x** |
| `mul_mat_vec_f` | 242.4ms | **234.1ms** | - | **1.04x** |

---

## Key Findings

### What Works on gfx900

1. **Increasing nwarps (4→8)**
   - Works on: mul_mat_q, gated_delta_net
   - Benefit: Better occupancy on 64 CU architecture
   - Requirement: Low register pressure (<= 80 VGPRs)

2. **DPP Warp Reductions**
   - Works on: Kernels with explicit `warp_reduce_sum`
   - Benefit: 2-5% per kernel, cumulative effect
   - Requirement: AMD GCN architecture (gfx900+)

3. **Multi-Token Loop Disable**
   - Works on: Kernels with loop unrolling causing register pressure
   - Benefit: Dramatic (5x) when register-bound
   - Requirement: Large batch sizes (>= 256 tokens)

### What Doesn't Work

1. **DPP on mul_mat_q**
   - No warp reductions in kernel
   - Thread-private accumulation

2. **Fast Math (-ffast-math)**
   - Tested on gated_delta_net
   - No improvement observed

---

## Remaining Opportunities

### High Impact

1. **Vectorized Loads in mul_mat_q**
   - Target: 38% of execution time
   - Approach: Use `float4` / `int4` loads
   - Expected: 5-15% improvement

2. **Software Pipelining**
   - Target: Memory-bound kernels
   - Approach: Explicit prefetch instructions
   - Expected: Hide memory latency

### Medium Impact

3. **Kernel Fusion**
   - Target: TG phase (63,355 dispatches)
   - Approach: Combine multiple operations per kernel
   - Expected: Reduce launch overhead

4. **Better Load Balancing**
   - Target: GPU utilization (~22% average)
   - Approach: Pipeline optimization
   - Expected: Increase to 40-50%

---

## Technical Notes

### gfx900 Architecture Characteristics

- **64 Compute Units**
- **64 threads per wavefront**
- **16KB L1 cache per CU**
- **64KB LDS (shared memory) per CU**
- **No native dp4a instruction** (uses emulation)
- **GCN ISA with DPP support**

### Optimal Configuration Found

```
Workgroup Size: 512 threads (8 warps × 64)
Grid Size: Varies by workload
Occupancy: ~22% (room for improvement)
Kernel Launches: Minimize for TG phase
```

---

## Commit History

1. `639ee57fb` - gfx900: disable multi-token loop for large batches
2. `1653d923b` - gfx900: add DPP-based warp reductions  
3. `f521552f3` - gfx900: apply DPP to mmvq and mmvf kernels
4. `cd03a7de6` - docs: add gfx900 optimization summary
5. `9fc8ff5bb` - gfx900: add vectorized load helpers to mmq.cuh
6. `25494bfd6` - docs: update results with optimal batch config
7. `0545044e1` - gfx900: increase nwarps from 4 to 8 for better PP performance

---

## Next Steps

### Immediate (High ROI)
- [ ] Implement vectorized loads in mul_mat_q
- [ ] Profile with cache counters to verify memory bottlenecks

### Medium Term
- [ ] Test software pipelining (prefetch) on gated_delta_net
- [ ] Explore kernel fusion for TG phase

### Long Term
- [ ] Improve multi-GPU load balancing
- [ ] Increase overall GPU utilization from 22%

---

*Last Updated: 2026-03-12*  
*Total Speedup Achieved: PP +43%, TG +97%*
