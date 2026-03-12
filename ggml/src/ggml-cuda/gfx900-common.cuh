#pragma once

// GFX900 (MI25/WX9100) Optimized CUDA/HIP Kernels
// Based on gfx906 optimizations, adapted for gfx900 architecture
// Key difference: NO dp4a instruction on gfx900

#include "common.cuh"

#if defined(GGML_USE_HIP)

// ============================================================================
// DPP-based Warp Reductions for GFX900
// Using AMD GCN Data Parallel Primitives instead of shuffle + op
// This is more efficient on GCN architecture
// ============================================================================

// DPP XOR(1): quad_perm pattern [1,0,3,2] - swap adjacent pairs
template<typename T>
static __device__ __forceinline__ T gfx900_dpp_xor1(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 4\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

// DPP XOR(2): quad_perm pattern [2,3,0,1] - swap pairs
template<typename T>
static __device__ __forceinline__ T gfx900_dpp_xor2(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

// DPP XOR(4): row shift left 4 / row shift right 4 combination
template<typename T>
static __device__ __forceinline__ T gfx900_dpp_xor4(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int v_src = *reinterpret_cast<int*>(&value);
    int v_dst;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa"
        : "=v"(v_dst) : "v"(v_src) : "memory"
    );
    return *reinterpret_cast<T*>(&v_dst);
}

// DPP XOR(8): row rotate right 8
template<typename T>
static __device__ __forceinline__ T gfx900_dpp_xor8(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

// DPP XOR(16): LDS swizzle pattern
template<typename T>
static __device__ __forceinline__ T gfx900_dpp_xor16(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

// Generic DPP shuffle - dispatches to appropriate pattern
template<int width = WARP_SIZE, typename T>
static __device__ __forceinline__ T gfx900_shfl_xor_sync(T x, int offset) {
    // Only handle power-of-2 offsets up to 16, fallback to shuffle for others
    switch (~offset) {
        case ~1:  return gfx900_dpp_xor1(x);
        case ~2:  return gfx900_dpp_xor2(x);
        case ~4:  return gfx900_dpp_xor4(x);
        case ~8:  return gfx900_dpp_xor8(x);
        case ~16: return gfx900_dpp_xor16(x);
        default:  return __shfl_xor(x, offset, width);
    }
}

// ============================================================================
// Warp Reduction Operations using DPP
// ============================================================================

template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx900_warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += gfx900_shfl_xor_sync<width>(x, offset);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx900_warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, gfx900_shfl_xor_sync<width>(x, offset));
    }
    return x;
}

// ============================================================================
// Fast Math Operations using Native GCN Instructions
// ============================================================================

// Fast exponential using v_exp_f32
static __device__ __forceinline__ float gfx900_fast_exp_f32(float x) {
    constexpr float LOG2_E = 1.4426950408889634f;
    float result;
    asm volatile(
        "v_exp_f32 %0, %1"
        : "=v"(result)
        : "v"(x * LOG2_E)
    );
    return result;
}

// Fast exp2 using v_exp_f32
static __device__ __forceinline__ float gfx900_fast_exp2_f32(float x) {
    float result;
    asm volatile(
        "v_exp_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

// Fast log2 using v_log_f32
static __device__ __forceinline__ float gfx900_fast_log2_f32(float x) {
    float result;
    asm volatile(
        "v_log_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

// Fast reciprocal using v_rcp_f32
static __device__ __forceinline__ float gfx900_fast_rcp_f32(float x) {
    float result;
    asm volatile(
        "v_rcp_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

// Fast tanh approximation
static __device__ __forceinline__ float gfx900_fast_tanh_f32(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;

    const float exp2x = gfx900_fast_exp_f32(2.0f * x);
    return 1.0f - 2.0f / (exp2x + 1.0f);
}

#endif // GGML_USE_HIP
