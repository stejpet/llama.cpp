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

## Validation

- `test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT`: 2920/2920 pass
  (incl. hsk=256, hsv=256, nb=512, kv=16384)
- Coherent server output (test prompt/continuation)
- Initial dispatch guard was too broad and aborted on DKQ=320 (no 16-col
  config exists for 320); fixed by gating to `DKQ==256 && DV==256 && VEGA`

## Caveats

- The `(256,256,32)` nbatch_K=32 entry is now dormant on gfx900 (dispatch
  forces the 16-col path) - keep or drop.
- Config-table changes apply to all AMD tile-kernel users (e.g. RDNA1
  gfx1010), only tested on gfx900 here.
- DPP warp-reduces (GCN path in common.cuh) were re-enabled; measured ~1%
  faster than the `__gfx906__`-only gate.
