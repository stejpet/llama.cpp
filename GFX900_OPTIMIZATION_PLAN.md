# GFX906 Optimizations Analysis for GFX900

## Repository: https://github.com/iacopPBK/llama.cpp-gfx906

## Architecture Similarities
- **GFX900** (MI25/WX9100): GCN5, Vega, 64 threads/wavefront, limited FP16
- **GFX906** (MI50/MI60): GCN5, Vega20, 64 threads/wavefront, better FP16

Both use same GCN5 ISA - many optimizations transfer!

## Key Optimizations Found

### 1. DPP-Based Warp Reductions (HIGH PRIORITY)
**File:** `gfx906-common.cuh`

Instead of shuffle + operation, use fused DPP instructions:
```asm
v_add_f32_dpp dst, src, src quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
```

**Benefits:**
- Single instruction instead of shuffle + add
- Better latency hiding
- Native GCN feature

**Applicable to gfx900:** YES - same GCN5 ISA

### 2. Fast Math Operations (HIGH PRIORITY)
**File:** `gfx906-common.cuh`

Native AMD instructions:
```cpp
v_exp_f32  // Fast exponential
v_log_f32  // Fast logarithm  
v_rcp_f32  // Fast reciprocal
```

**Applicable to gfx900:** YES - native GCN instructions

### 3. Warp-Cooperative MMVQ (NOT APPLICABLE - See Note)
**Files:** `mmvq-q8_0.cuh`, `mmvq-q4_0.cuh`, `mmvq-q4_1.cuh`

- Uses full 64-thread wavefront cooperatively
- **dp4a instructions for dot products** - NOT available on gfx900!
- Half-warp (32 threads) reduction at end

**Applicable to gfx900:** NO - dp4a not available on MI25/gfx900

**Alternative:** Use regular MAD/FMA instructions or float32

### 4. Vectorized Loads (MEDIUM PRIORITY)
**File:** `mmq.cuh`

Load 128-bit (int4) instead of 8x 32-bit loads:
```cpp
const int4 vec = *((const int4 *) &y_qs[base_addr]);
```

**Applicable to gfx900:** YES - reduces memory transactions

### 5. Software Pipelining (LOW PRIORITY)
**File:** `mmq.cuh`, `mmq-prefetch.cuh`

Separate load and compute phases to hide latency.
More complex, benefits may vary.

## Implementation Plan for GFX900

### Phase 1: DPP Warp Reductions (Quick Win)
1. Create `ggml/src/ggml-cuda/gfx900/gfx900-common.cuh`
2. Implement DPP-based `warp_reduce_sum` and `warp_reduce_max`
3. Replace in gated_delta_net.cu and other hot kernels
4. **Expected improvement:** 5-15% in reduction-heavy kernels

### Phase 2: Fast Math
1. Add fast_exp, fast_log, fast_rcp functions
2. Replace standard math in hot paths
3. **Expected improvement:** 5-10% in math-heavy kernels

### Phase 3: Vectorized Loads
1. Modify quantization load paths
2. Use int4 (128-bit) vectorized loads
3. **Expected improvement:** 5-10% in memory-bound kernels

## Testing Strategy

Benchmark command:
```bash
./build/bin/llama-bench -m ~/models/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf -p 1024 -r 1 -ub 1024 -b 512 -fa 0
```

Target: Maintain PP > 380 t/s, TG > 46 t/s while adding optimizations.

## Risk Assessment

- **DPP instructions:** Low risk - standard GCN5 ISA
- **Fast math:** Low risk - native instructions
- **Vectorized loads:** Medium risk - alignment requirements
- **New kernels:** Higher risk - need thorough testing

## Notes

- gfx900 has fewer registers than gfx906, need to be careful with unrolling
- Focus on occupancy over instruction-level optimizations
- Test each change individually
- Watch for register pressure increase
