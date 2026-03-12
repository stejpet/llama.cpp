# Optimization Log for mi25-speed Branch
# Goal: Improve PP speed without hurting TG speed
# Reference: pp1024=352.00 t/s, tg128=41.59 t/s
# Current:   pp1024=288.39 t/s, tg128=46.48 t/s
# Target:    pp1024>=352.00 t/s, tg128>=41.59 t/s

# AGENTS.md backup location: preserved in /home/steffen/llama.cpp/AGENTS.md

## Initial State
- Branch: mi25-speed
- Latest commit: c86d89a30 "Account for register pressure of KDA due to per-row g"
- Commits show: KDA optimization, prefetching, data-access analysis
- Issue: PP is -18% slower than reference, TG is +12% faster

## Test Command
./build/bin/llama-bench -m ~/models/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf -p 1024 -r 1 -ub 1024 -b 512 -fa 0

## Build Command
HIPCXX="/opt/rocm-6.3.3/llvm/bin/clang" HIP_PATH="/opt/rocm-6.3.3" cmake -S . -B build -DGGML_HIP=ON -DGML_CURL=ON -DAMDGPU_TARGETS=gfx900 -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -- -j 22

## Experiment 1: Understanding the Issue

**Observation:** The latest commit (c86d89a30) reduced `n_tokens_per_loop` from 20 to 14 for KDA mode.
- This was done to reduce register pressure for TG (token generation)
- But it hurts PP (prompt processing) because we process fewer tokens per iteration
- Current: nt_thresh = KDA ? 14 : 20

**Hypothesis:** We need different thresholds for PP vs TG, or find a sweet spot that works for both.

**Approach:** 
1. First test: Try nt_thresh = 16 (middle ground between 14 and 20)
2. If that doesn't work, try using n_tokens to decide: larger batches = higher threshold

**Plan:**
- [ ] Test 1: nt_thresh = 16 for both S_v cases
- [ ] Benchmark and compare PP/TG
- [ ] If PP improves but TG hurts, try conditional threshold based on n_tokens

## Experiment 6: Disable multi-token loop for PP (n_tokens >= 512)

**Hypothesis:** The multi-token loop with unrolling causes high register pressure,
hurting occupancy on gfx900 for PP. For TG, the benefits outweigh the costs.

**Change:** Add runtime check: only use multi-token loop when n_tokens < 512
- TG (n_tokens < 512): use multi-token loop (nt_thresh=14 or 20)
- PP (n_tokens >= 512): use simple single-token loop (n_tokens_per_loop=1)

**Result:** 🎉 HUGE SUCCESS!
- PP: 384.50 t/s (+33% vs baseline 289.17, +10% vs master 349.32)
- TG: 46.35 t/s (maintained 46.37)

This exceeds the reference performance!

## Experiment 7: Optimize threshold value

**Testing different thresholds:**

| Threshold | PP t/s | TG t/s | Notes |
|-----------|--------|--------|-------|
| 512 (baseline) | 384.50 | 46.35 | Good starting point |
| 256 | 386.37 | 46.94 | **Best overall** |
| 128 | 385.81 | 46.91 | Slightly worse |

**Optimal threshold: 256**

## Final Results

**Reference build (master):**
- PP: 349.32 t/s
- TG: 42.48 t/s

**mi25-speed baseline (before my changes):**
- PP: 289.17 t/s (-17.2%)
- TG: 46.37 t/s (+9.2%)

**After optimization (threshold=256):**
- PP: 386.37 t/s (**+10.6% vs reference!**)
- TG: 46.94 t/s (**+10.5% vs reference!**)

**Summary:**
✅ PP improved from 289→386 t/s (+33%)
✅ TG maintained at 47 t/s (was 46.37)
✅ Both metrics now exceed reference performance

**The fix:** Disable multi-token loop unrolling for large batch sizes (n_tokens >= 256)
on gfx900 to reduce register pressure and improve GPU occupancy.

This optimization is specific to AMD gfx900 architecture which has limited
register file compared to newer GPUs. The loop unrolling that helps NVIDIA
actually hurts gfx900 for large batches.