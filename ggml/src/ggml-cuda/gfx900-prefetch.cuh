#pragma once

// GFX900 Software Prefetching for MMQ
// Hides memory latency by prefetching next iteration's data during compute
// Adapted from gfx906 implementation for Vega 10 / MI25 / WX9100

#if defined(GGML_USE_HIP) && defined(__gfx900__)

// Prefetch Y tile for next iteration
// Uses warp 0 threads to warm L2 cache while warps 1-7 do compute
// This overlaps memory fetch with ALU operations
static __device__ __forceinline__ int gfx900_prefetch_y_tile(
    const int * __restrict__ y,
    const int ncols_y,
    const int kb0,
    const int kb0_stop,
    const int qk,
    const int blocks_per_iter) {

    const int kb0_next = kb0 + blocks_per_iter;

    // Don't prefetch if this is the last iteration
    if (kb0_next >= kb0_stop) {
        return 0;
    }

    // Only warp 0 participates in prefetching
    // With 8 warps, warp 0 can prefetch while warps 1-7 compute
    const int lane_id = threadIdx.x;
    if (threadIdx.y != 0 || lane_id >= 16) {
        return 0;
    }

    // Calculate address for next iteration's Y data
    constexpr int block_q8_1_mmq_bytes = 144;
    constexpr int QK8_1_val = 32;
    const int stride_factor = qk * block_q8_1_mmq_bytes / (4 * QK8_1_val * sizeof(int));
    const int * by_next = y + ncols_y * (kb0_next * stride_factor);

    // Each thread prefetches a different cache line (64 bytes = 16 ints apart)
    const int prefetch_offset = lane_id * 16;
    const int * prefetch_addr = by_next + prefetch_offset;

    // Issue global load to L2 (non-blocking, uses spare VGPR)
    int prefetch_data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(prefetch_data)
        : "v"(prefetch_addr)
        : "memory"
    );
    
    return prefetch_data;
}

// Prefetch X tile for next iteration
// Uses warp 1 threads to prefetch X data
static __device__ __forceinline__ int gfx900_prefetch_x_tile(
    const char * __restrict__ x,
    const int offset_x,
    const int kb0,
    const int kb0_stop,
    const int blocks_per_iter,
    const int stride_row_x) {

    const int kb0_next = kb0 + blocks_per_iter;

    if (kb0_next >= kb0_stop) {
        return 0;
    }

    // Use warp 1 for X prefetch (different from Y prefetch)
    const int lane_id = threadIdx.x;
    if (threadIdx.y != 1 || lane_id >= 16) {
        return 0;
    }

    // Prefetch X data for next iteration
    const int row = lane_id;
    const char * x_row = x + (offset_x + kb0_next) + row * stride_row_x;
    const int * x_ptr = (const int *)x_row;

    int prefetch_data;
    asm volatile(
        "global_load_dword %0, %1, off\n"
        : "=v"(prefetch_data)
        : "v"(x_ptr)
        : "memory"
    );
    
    return prefetch_data;
}

// Consume prefetch result (prevents compiler optimization away)
static __device__ __forceinline__ void gfx900_prefetch_consume(int prefetch_data) {
    // Dummy operation to keep the prefetch data alive
    asm volatile(
        "v_mov_b32 %0, %0\n"
        : "+v"(prefetch_data)
    );
}

#endif // defined(GGML_USE_HIP) && defined(__gfx900__)
