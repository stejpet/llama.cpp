// Test kernel for gfx900 DPP instructions
// Compile and run to verify DPP works on MI25

#include "ggml/src/ggml-cuda/gfx900-common.cuh"
#include </opt/rocm-6.3.3/include/hip/hip_runtime.h>
#include <stdio.h>

__global__ void test_dpp_reduction(float* out) {
    int tid = threadIdx.x;
    float val = (float)(tid % 64);  // Values 0-63
    
    // Test DPP-based warp reduction
    float sum = gfx900_warp_reduce_sum<64>(val);
    
    if (tid == 0) {
        // Sum of 0..63 = 2016
        out[0] = sum;
    }
}

__global__ void test_dpp_shuffle(float* out) {
    int tid = threadIdx.x;
    float val = (float)tid;
    
    // Test each DPP pattern
    float xor1 = gfx900_dpp_xor1(val);
    float xor2 = gfx900_dpp_xor2(val);
    
    if (tid == 0) {
        out[0] = xor1;  // Should be 1 (tid 0 gets tid 1's value via xor1)
        out[1] = xor2;  // Should be 2 (tid 0 gets tid 2's value via xor2)
    }
}

int main() {
    float *d_out, h_out[2];
    hipMalloc(&d_out, 2 * sizeof(float));
    
    printf("Testing gfx900 DPP instructions on MI25...\n");
    
    // Test 1: Reduction
    hipLaunchKernelGGL(test_dpp_reduction, dim3(1), dim3(64), 0, 0, d_out);
    hipMemcpy(h_out, d_out, sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    
    printf("Warp reduction test: sum = %.0f (expected 2016)\n", h_out[0]);
    if (h_out[0] == 2016.0f) {
        printf("✓ DPP reduction WORKS!\n");
    } else {
        printf("✗ DPP reduction FAILED\n");
    }
    
    // Test 2: Shuffle
    hipLaunchKernelGGL(test_dpp_shuffle, dim3(1), dim3(64), 0, 0, d_out);
    hipMemcpy(h_out, d_out, 2 * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    
    printf("DPP shuffle test: xor1 = %.0f (expected 1), xor2 = %.0f (expected 2)\n", 
           h_out[0], h_out[1]);
    if (h_out[0] == 1.0f && h_out[1] == 2.0f) {
        printf("✓ DPP shuffle WORKS!\n");
    } else {
        printf("✗ DPP shuffle FAILED\n");
    }
    
    hipFree(d_out);
    return 0;
}
