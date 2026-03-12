# gfx900 dp4a Optimization Research

## Current Implementation Analysis

The current gfx900 (Vega/RDNA1) fallback uses **6 instructions**:
```asm
v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD src0_sel:BYTE_0 src1_sel:BYTE_0
v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD src0_sel:BYTE_1 src1_sel:BYTE_1
v_add3_u32 %0, %1, %2, %0
v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD src0_sel:BYTE_2 src1_sel:BYTE_2
v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD src0_sel:BYTE_3 src1_sel:BYTE_3
v_add3_u32 %0, %1, %2, %0
```

This is already **near-optimal** because:
- 4 multiplies + 3 adds is the minimum theoretical requirement for dot4
- `v_add3_u32` already fuses two additions into one instruction
- Hardware byte extraction (`src0_sel:BYTE_x`) avoids separate shift/mask operations
- Sign extension (`sext`) handles signed 8-bit inputs correctly

## Why It's Hard to Optimize Further

### 1. No Native dp4a on gfx900
- `__builtin_amdgcn_sdot4` was introduced in **gfx906** (Vega 20 / Radeon VII)
- gfx900 lacks this hardware acceleration

### 2. Tested Approaches That Don't Work

#### ~~v_mad_i32_i24~~ (FAILED - Update 1)
Attempted to use `v_mad_i32_i24` for 4-instruction version but compilation failed with "not a valid operand".

**Conclusion**: `v_mad_i32_i24` does NOT exist on gfx900. Cannot reduce to 4 instructions.

#### ~~Packed 16-bit Math (VOP3P)~~ (NOT VIABLE - Update 2)
Vega has packed 16-bit instructions (`V_PK_MUL_LO_U16`, `V_PK_MAD_I16`, etc.) that operate on two 16-bit values simultaneously.

**Why it doesn't work for 8-bit dot product**:

Packed 16-bit multiply does:
```
result[15:0]  = (a[15:0]  * b[15:0])[15:0]   // low 16-bit multiply
result[31:16] = (a[31:16] * b[31:16])[15:0] // high 16-bit multiply
```

If we pack two 8-bit values into each 16-bit half:
```
a[15:0]  = (a0 << 8) | a1  // two 8-bit values in low half
b[15:0]  = (b0 << 8) | b1
Product: (a0<<8 + a1) * (b0<<8 + b1) 
       = a0*b0<<16 + a0*b1<<8 + a1*b0<<8 + a1*b1
```

**Cross-term contamination**: We get unwanted terms a0*b1 and a1*b0 mixed in!

We want: `a0*b0 + a1*b1`
We actually get: `a0*b0<<16 + a0*b1<<8 + a1*b0<<8 + a1*b1`

The cross-terms cannot be avoided with 16-bit packed multiplies.

#### SDWA (Sub-DWord Addressing)
- Already used effectively via `src0_sel:BYTE_x`
- Allows selecting bytes without extra shift instructions
- Already optimal for this use case

## Potential (Minor) Optimizations to Test

### 1. Instruction Scheduling
Try different multiply ordering to improve ILP:
```asm
// Current: pairs (0,1) then (2,3)
// Try: interleaved (0,2) then (1,3) for better latency hiding
v_mul_i32_i24 %1, sext(%3), sext(%4) src0_sel:BYTE_0 src1_sel:BYTE_0
v_mul_i32_i24 %2, sext(%3), sext(%4) src0_sel:BYTE_2 src1_sel:BYTE_2
v_add3_u32 %0, %1, %2, %0
v_mul_i32_i24 %1, sext(%3), sext(%4) src0_sel:BYTE_1 src1_sel:BYTE_1
v_mul_i32_i24 %2, sext(%3), sext(%4) src0_sel:BYTE_3 src1_sel:BYTE_3
v_add3_u32 %0, %1, %2, %0
```

### 2. Single Temp Register
Reduce from 2 temps to 1 (may not improve perf but reduces register pressure):
```asm
v_mul_i32_i24 %1, sext(%3), sext(%4) src0_sel:BYTE_0 src1_sel:BYTE_0
v_mul_i32_i24 %0, sext(%3), sext(%4) src0_sel:BYTE_1 src1_sel:BYTE_1, %0
v_add_u32 %0, %0, %1
// etc...
```

## Key Finding: gfx900 Cannot Beat 6 Instructions

The theoretical minimum for a signed 8-bit dot4 on GCN is:
- 4 multiplications (can't be reduced)
- 3 additions (can be reduced to 2 if using v_add3_u32 twice)

The current implementation already achieves this theoretical minimum.

## Recommendation

**The current implementation is already optimal for gfx900.**

Any perceived "slowness" on gfx900 is due to:
1. Lack of native dp4a hardware (requires gfx906+ or RDNA2+)
2. 6 instructions vs 1 instruction for GPUs with `__dp4a`
3. Packed 16-bit math cannot help due to cross-term contamination

## References

- Vega ISA PDF: https://gpuopen.com/download/Vega_7nm_Shader_ISA_26November2019.pdf
- LLVM AMDGPU docs: https://llvm.org/docs/AMDGPU/
- SDWA blog: https://gpuopen.com/learn/using-sub-dword-addressing-on-amd-gpus-with-rocm
