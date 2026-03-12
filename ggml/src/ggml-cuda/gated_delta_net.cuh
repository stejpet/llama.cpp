#include "common.cuh"
#include "ggml.h"

// Include gfx900 optimizations for AMD GCN architecture
#if defined(GGML_USE_HIP) && defined(__gfx900__)
#include "gfx900-common.cuh"
#endif

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
