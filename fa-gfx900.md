# fa-gfx900: FA tile kernel occupancy tuning for gfx900 (Vega 10)

Date: 2026-08-05
Scope: prompt-processing (PP) flash-attention tuning on 3x Radeon Pro WX 9100
(gfx900, 16 GB HBM2), Qwen3.6-35B-A3B Q5_K_XL, MTP spec decoding.

## Problem

PP throughput collapsed as context grew (476 -> 224 t/s at 2k -> 10k ctx).
Rocprofv2 kernel trace of an 8510-token PP showed `flash_attn_tile` (the FA
kernel used on gfx900, since it has no tensor cores) eating **67.6% of all GPU
kernel time**, avg 224 ms/dispatch, growing ~linearly with KV length
(113 ms at 2k ctx -> 386 ms at 8.5k ctx). At 45k ctx it extrapolates to ~2 s
per dispatch.

## Root cause: occupancy, not tile compute

The tile kernel was LDS-starved on gfx900 (64 KB LDS/CU):

- config `(256,256,32)`: LDS 27,136 B/block, VGPR 128, 256 threads
- -> only **2 blocks/SM** (8 waves/CU, ~20% occupancy), so HBM latency was
  never hidden (kernel ran at ~0.5% of peak)

The q8_0 KV cache dequant pass was NOT a factor (0.4% of kernel time), which
matches the earlier finding that `-ctv f16` was neutral in A/B tests.

## Change (ggml/src/ggml-cuda/fattn-tile.cuh)

1. AMD config table: cut `nbatch_K` to shrink per-block LDS
   - `(256,256,16)`: nbatch_K 128 -> 64
   - `(256,256,32)`: nbatch_K 128 -> 32
2. Dispatch: on gfx900, force the 16-col path for DKQ=DV=256 (the generic
   path otherwise always picks cols_per_block=32, which cannot reach higher
   occupancy because `Q_tmp` alone is 16 KB)

Result: `(256,256,16)` now uses LDS 13,824 B -> **4 blocks/SM**.

Note: nbatch_fa must stay a multiple of warp size (`static_assert
nbatch_fa % (np*warp_size) == 0`), so nbatch_fa=16 is not allowed for
ncols=32.

## Results (llama-server, 8510-tok PP, b2048 unless noted)

| config | blocks/SM | pp t/s | vs orig |
|--------|-----------|-------:|--------|
| orig config, DPP off | 2 | 337.0 | - |
| orig config, DPP on | 2 | 341.3 | +1% |
| Exp A: ncols=32, nbatch_K=32 | 3 | 436.9 | +30% |
| Exp B: ncols=16, nbatch_K=64 | 4 | 869.8 | +158% |

Production config (with MTP):

| -b | before | after |
|-----|-------:|------:|
| 2048 | 255.96 | 680.5 |
| 8192 | 289.15 | 813.1 |

Deep context (8.5k-tok PP at 8-17k ctx):

| -b | before | after |
|-----|-------:|------:|
| 2048 | 117.6 | 589.9 |
| 8192 | 128.1 | 699.4 |

Profile after fix: `flash_attn_tile` share 67.6% -> 9.7%, max TILE duration
386 ms -> 21 ms. MoE `mul_mat_q` is now the top cost (59.9%) - the next
bottleneck if PP is still too slow.

## Follow-up sweep (2026-08-05): other tile sizes?

Question: can we push past 4 blocks/SM? The LDS budget said yes on paper
(64 KB/CU, cpy_ne=4): shrink nbatch_K 64->32 gives 11,776 B/block (5 blocks),
or drop to 8-col tiles at 9,216 B/block (7 blocks). Both were tested with the
same harness (llama-server, 8510-tok PP, b2048, q8_0 KV):

| config (ncols, nthreads, nbatch_fa, nbatch_K, occ) | LDS | target | pp t/s | vs base |
|----------------------------------------------------|-----|--------|-------:|--------|
| base: (16, 256, 32, 64, 2) | 13,824 | 4 blk | 839-840 | - |
| (16, 256, 32, 32, 2) | 11,776 | 5 blk | 394.3 | -53% |
| (16, 256, 32, 32, 5) | 11,776 | 5 blk | 243.9 | -71% |
| (8, 128, 32, 64, 7) | 9,216 | 7 blk | 425.8 | -49% |

Both families regressed; the 4-block/16-col config is the optimum. Why:

- warps/CU = 2048*nthreads/LDS, capped at 40 waves by HW. Base config sits at
  32 warps (4 blk x 8 warps) - the max that needs neither fewer registers nor
  more KQ barriers.
- nbatch_K 64->32 doubles the KQ loop iterations -> ~2x __syncthreads cost.
  At occupancy=2 that alone costs ~2x (394 vs 840); the occupancy=5 hint
  (<=51 VGPR) then spills registers on top (244).
- 8-col at 7 blocks is only 28 warps (fewer than base's 32) AND halves KV
  reuse (each HBM KV read serves 8 Q columns instead of 16) on a kernel that
  is HBM-latency-bound. The occupancy=7 hint (<=73 VGPR) adds spill risk.

No route to >32 warps exists that avoids one of the two proven regressions,
so tuning is done. All experiment edits reverted; base config restored and
re-validated.

## Follow-up (2026-08-05): mul_mat_q occupancy sweep - INVALID, REVERTED

After the FA fix, `mul_mat_q` became the top cost (59.9% of kernel time).
Analysis: it runs at **2 blocks/CU (20% occupancy)** - VGPR 96 + LDS 27,648 B
(reg-limit 2.67, LDS-limit 2.37) - the exact low-occupancy state FA had pre-fix.
The Q5_K dot is per-byte `v_mul_i32_i24_sdwa`+`v_add3_u32` (no packed dot on
gfx900), so it is HBM-LDS-latency-bound with nothing to hide it.

An occupancy sweep (shrink I/J to fit more blocks/CU) measured large speedups
(2->4 blocks: 9216 -> 4446 ms mul_mat_q, PP 840 -> ~1150 t/s) BUT was later
found to produce **garbage output** (repeated `/` tokens) - the measured gains
were real occupancy gains on a numerically wrong kernel. **All MMQ sweep
changes reverted.**

Root cause: the MMQ kernel's accumulator indexing is only valid for
`I == warp_size` (64 on gfx900). `vec_dot` writes
`sum[j0/nwarps*I/warp_size + i0/warp_size]` (left-assoc = `(j0/nwarps)*I/warp_size`)
but `write_back` reads `sum[(j0/nwarps)*(I/warp_size) + i0/warp_size]`. These
agree only when `I/warp_size == 1`; at I=32, vec_dot writes `sum[j0/8]` while
write_back reads `sum[0]` for every output -> garbage logits. I=32 is
fundamentally unsupported on wave64 (I must be >= warp_size). This was not
caught by `test-backend-ops -o MUL_MAT` (passed 1188/1188) because the test
shapes did not exercise the I=32 path.

Consequence: with correct I=64, the LDS floor for Q5_K is 18,752 B (the X
tile), so 3 blocks/CU requires J<=8 (Q6_K cannot reach 3 at all). Config-only
occupancy tuning of `mul_mat_q` is a dead end on gfx900; going past 2 blocks
needs a real kernel change (I<warp_size support or a smaller X-tile layout),
not a config tweak. Baseline (J=64, I=64) restored and coherence re-verified.

## Validation
- `test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT`: 2922/2922 pass
  (incl. hsk=256, hsv=256, nb=512, kv=16384)
- Coherent server output (test prompt/continuation), incl. Q6_K model at
  8.5k-tok PP - re-verified after MMQ sweep revert
- Initial dispatch guard was too broad and aborted on DKQ=320 (no 16-col
  config exists for 320); fixed by gating to `DKQ==256 && DV==256 && VEGA`

## Caveats

- The `(256,256,32)` nbatch_K=32 entry is now dormant on gfx900 (dispatch
  forces the 16-col path) - keep or drop.
- Config-table changes apply to all AMD tile-kernel users (e.g. RDNA1
  gfx1010), only tested on gfx900 here.
- DPP warp-reduces (GCN path in common.cuh) were re-enabled; measured ~1%
  faster than the `__gfx906__`-only gate.
- Lesson: MMQ tile-size experiments must be validated for coherence (not just
  timing); `test-backend-ops -o MUL_MAT` did not catch the I=32 indexing bug.
