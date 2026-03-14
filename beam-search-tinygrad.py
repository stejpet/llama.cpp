#!/usr/bin/env python3
"""
TinyGrad BEAM Search Integration for llama.cpp Kernel Optimization

This script uses TinyGrad's BEAM search to find optimal kernel configurations
and extracts the actual parameters to translate them to llama.cpp settings.

BEAM search tries multiple optimization configurations and picks the fastest.
The BEAM=N number means "keep the top N configurations at each stage":
  - BEAM=0: Use default heuristics (no search)
  - BEAM=4: Try 4 configs, keep best (fast, ~30 sec per kernel)
  - BEAM=16: Try 16 configs, keep best (thorough, ~2 min per kernel)
  - BEAM=32: Try 32 configs, keep best (very thorough, ~5 min per kernel)

Higher BEAM = more search time but potentially better performance.
Results are cached, so re-runs use previously found optimal configs.

Usage:
    BEAM=4 python3 beam-search.py
    BEAM=4 python3 beam-search.py --compare  # Compare different BEAM sizes
    BEAM=4 python3 beam-search.py --extract  # Extract and display kernel configs
"""

import os
import sys
import time
import json
import math
import argparse
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import asdict

# Add tinygrad to path
sys.path.insert(0, '/home/steffen/tinygrad')

# Check if HIP is available and set it as default device before importing tinygrad
if 'HIP' not in os.environ and 'DEVICE' not in os.environ:
    if os.path.exists('/opt/rocm') or os.path.exists('/usr/lib/x86_64-linux-gnu/libamdhip64.so'):
        os.environ['HIP'] = '1'
        print("HIP detected, setting HIP=1 for AMD GPU acceleration")

import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import BEAM, DEBUG, diskcache_get, diskcache_put, CACHELEVEL, IGNORE_BEAM_CACHE
from tinygrad.device import Buffer
from tinygrad.engine.realize import get_runner, CompiledRunner
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.renderer import ProgramSpec

def cleanup_beam_pool():
    """Clean up BEAM search multiprocessing pool to prevent resource leaks."""
    try:
        import tinygrad.codegen.opt.search as search_module
        pool = getattr(search_module, 'beam_pool', None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except:
                pass
            finally:
                search_module.beam_pool = None
    except:
        pass

# Store captured kernels
captured_kernels: List[Dict[str, Any]] = []

class KernelCapture:
    """Captures kernel configurations during BEAM search"""
    
    def __init__(self):
        self.kernels = []
        self.original_get_runner = None
    
    def start(self):
        """Start capturing kernel configurations"""
        import tinygrad.engine.realize as realize_module
        self.original_get_runner = realize_module.get_runner
        
        def capturing_get_runner(device: str, ast):
            runner = self.original_get_runner(device, ast)
            if isinstance(runner, CompiledRunner):
                self._capture_kernel(runner.p)
            return runner
        
        realize_module.get_runner = capturing_get_runner
        return self
    
    def stop(self):
        """Stop capturing"""
        import tinygrad.engine.realize as realize_module
        if self.original_get_runner:
            realize_module.get_runner = self.original_get_runner
    
    def _capture_kernel(self, p: ProgramSpec):
        """Capture kernel configuration from ProgramSpec"""
        kernel_info = {
            'name': p.name,
            'device': p.device,
            'global_size': p.global_size,
            'local_size': p.local_size,
            'applied_opts': [str(opt) for opt in p.applied_opts] if p.applied_opts else [],
            'raw_opts': self._serialize_opts(p.applied_opts) if p.applied_opts else [],
            'vars': [{'name': v.expr, 'min': v.vmin, 'max': v.vmax} for v in p.vars],
            'estimates': {
                'ops': str(p.estimates.ops),
                'mem': str(p.estimates.mem),
                'lds': str(p.estimates.lds),
            }
        }
        self.kernels.append(kernel_info)
    
    def _serialize_opts(self, opts):
        """Serialize Opt objects to dict"""
        if not opts:
            return []
        return [
            {
                'op': opt.op.name,
                'axis': opt.axis,
                'arg': opt.arg if not isinstance(opt.arg, tuple) else list(opt.arg)
            }
            for opt in opts
        ]

def extract_from_beam_cache(ast_key: bytes, beam_size: int, device: str) -> Optional[List[Opt]]:
    """
    Extract BEAM search results from disk cache.
    
    Args:
        ast_key: The AST key used for caching
        beam_size: The BEAM size that was used
        device: The device string
    
    Returns:
        List of Opt objects that represent the best configuration, or None if not found
    """
    key = {
        "ast": ast_key,
        "amt": beam_size,
        "allow_test_size": True,
        "device": device,
        "suffix": ""  # Default suffix
    }
    
    cached_opts = diskcache_get("beam_search", key)
    return cached_opts

def parse_opts_to_params(opts: List[Opt]) -> Dict[str, Any]:
    """
    Parse a list of Opt objects into kernel parameters.
    
    Args:
        opts: List of Opt objects from BEAM search
    
    Returns:
        Dictionary of extracted parameters
    """
    params = {
        'local_dims': {},
        'upcast_dims': {},
        'unroll_dims': {},
        'group_sizes': {},
        'tensor_core': None,
        'tiling_strategy': [],
    }
    
    if not opts:
        return params
    
    for opt in opts:
        if opt.op == OptOps.LOCAL:
            params['local_dims'][opt.axis] = opt.arg
        elif opt.op == OptOps.UPCAST:
            params['upcast_dims'][opt.axis] = opt.arg
        elif opt.op == OptOps.UNROLL:
            params['unroll_dims'][opt.axis] = opt.arg
        elif opt.op == OptOps.GROUP or opt.op == OptOps.GROUPTOP:
            params['group_sizes'][opt.axis] = opt.arg
        elif opt.op == OptOps.TC:
            params['tensor_core'] = opt.arg
        elif opt.op == OptOps.THREAD:
            params['tiling_strategy'].append(('thread', opt.axis, opt.arg))
        elif opt.op == OptOps.PADTO:
            params['tiling_strategy'].append(('pad', opt.axis, opt.arg))
        elif opt.op == OptOps.NOLOCALS:
            params['use_locals'] = False
    
    return params

def translate_to_llama_cpp(params: Dict[str, Any], shape: Tuple[int, ...], kernel_type: str) -> Dict[str, Any]:
    """
    Translate TinyGrad kernel parameters to llama.cpp configuration.
    
    TinyGrad BEAM search produces Opt objects that control:
    - LOCAL: workgroup/thread block dimensions (local_size)
    - UPCAST: vectorization/tiling factors
    - UNROLL: loop unrolling factors
    - TC: tensor core usage
    
    These map to llama.cpp parameters:
    - num_warps: local_size / 64 (AMD) or / 32 (NVIDIA)
    - tile_x/y: derived from upcast factors and matrix dimensions
    - vdr: vector dot product rate from unroll factors
    """
    M, N, K = shape[:3]
    config = {
        'kernel_type': kernel_type,
        'shape': {'M': M, 'N': N, 'K': K},
        'llama_cpp_params': {},
        'derived_from': params
    }
    
    # Calculate workgroup size from local_dims
    local_size_total = 1
    local_dims = params.get('local_dims', {})
    for axis, size in local_dims.items():
        if size:
            local_size_total *= size
    
    # llama.cpp num_warps = threads_per_block / warp_size
    # AMD HIP uses 64 threads per warp (wavefront)
    # NVIDIA uses 32 threads per warp
    if local_size_total > 0:
        is_amd = 'AMD' in str(Device.DEFAULT).upper() or 'HIP' in str(Device.DEFAULT).upper()
        warp_size = 64 if is_amd else 32
        num_warps = local_size_total // warp_size
        num_warps = max(1, min(num_warps, 16))  # Clamp to [1, 16]
        config['llama_cpp_params']['num_warps'] = num_warps
        config['llama_cpp_params']['threads_per_block'] = local_size_total
        config['llama_cpp_params']['warp_size'] = warp_size
    
    # Determine tile sizes from upcast dimensions and matrix shape
    upcast_dims = params.get('upcast_dims', {})
    
    if kernel_type == 'matmul':
        # For matrix multiplication C[M,N] = A[M,K] @ B[K,N]:
        # - tile_y corresponds to M dimension (rows of output)
        # - tile_x corresponds to N dimension (cols of output)
        
        # Extract upcast factors for each axis
        # In TinyGrad's scheduler, axis mapping depends on the specific tensor layout
        upcast_values = [size for size in upcast_dims.values() if size and size > 1]
        
        if len(upcast_values) >= 2:
            # Sort to get largest factors
            upcast_values.sort(reverse=True)
            # Use largest factors as tile sizes
            tile_y = upcast_values[0]
            tile_x = upcast_values[1]
            # Ensure tile sizes are reasonable (power of 2, between 16 and 256)
            tile_y = 2 ** round(math.log2(min(max(tile_y, 16), 256)))
            tile_x = 2 ** round(math.log2(min(max(tile_x, 16), 256)))
            config['llama_cpp_params']['tile_y'] = int(tile_y)
            config['llama_cpp_params']['tile_x'] = int(tile_x)
        elif len(upcast_values) == 1:
            # Single large upcast - use for both dimensions
            tile = 2 ** round(math.log2(min(max(upcast_values[0], 16), 256)))
            config['llama_cpp_params']['tile_y'] = int(tile)
            config['llama_cpp_params']['tile_x'] = int(tile)
        else:
            # Default based on matrix size
            if N >= 4096 or M >= 4096:
                config['llama_cpp_params']['tile_y'] = 128
                config['llama_cpp_params']['tile_x'] = 128
            else:
                config['llama_cpp_params']['tile_y'] = 64
                config['llama_cpp_params']['tile_x'] = 64
        
        # VDR (vector dot product rate) - from unroll factor
        unroll_dims = params.get('unroll_dims', {})
        unroll_values = [v for v in unroll_dims.values() if v]
        if unroll_values:
            max_unroll = max(unroll_values)
            # llama.cpp typically uses 1, 2, 4, or 8 for VDR
            vdr = min(2 ** round(math.log2(max_unroll)), 8)
            config['llama_cpp_params']['vdr'] = max(1, int(vdr))
        else:
            config['llama_cpp_params']['vdr'] = 4  # Default
        
        # Also consider K dimension tiling for matmul
        group_sizes = params.get('group_sizes', {})
        if group_sizes:
            config['llama_cpp_params']['k_split'] = max(group_sizes.values())
        
    elif kernel_type == 'matvec':
        # For vector-matrix (M=1), focus on efficient column access
        config['llama_cpp_params']['num_warps'] = config['llama_cpp_params'].get('num_warps', 8)
        
        # VDR for vector ops - typically smaller
        unroll_dims = params.get('unroll_dims', {})
        unroll_values = [v for v in unroll_dims.values() if v]
        if unroll_values:
            vdr = min(max(unroll_values), 4)
            config['llama_cpp_params']['vdr'] = max(1, vdr)
        else:
            config['llama_cpp_params']['vdr'] = 2  # Smaller default for vec
    
    # Check for tensor core usage
    tensor_core = params.get('tensor_core')
    if tensor_core:
        config['llama_cpp_params']['use_tensor_cores'] = True
        config['llama_cpp_params']['tc_config'] = tensor_core
        # When using tensor cores, tile sizes should match TC dimensions
        # AMD MFMA typically uses 16x16 tiles
        if 'AMD' in str(Device.DEFAULT).upper() or 'HIP' in str(Device.DEFAULT).upper():
            config['llama_cpp_params']['tile_y'] = 16
            config['llama_cpp_params']['tile_x'] = 16
            config['llama_cpp_params']['tc_tile'] = 16
    
    return config

def benchmark_kernel(shape: Tuple[int, ...], beam_size: int = 4, iterations: int = 10) -> Optional[Dict]:
    """
    Run BEAM search on a matrix multiplication with given shape and extract configurations.
    
    Args:
        shape: (M, N, K, name) or (M, N, K) for matmul/matvec
        beam_size: Number of configurations to search
        iterations: Number of benchmark iterations
    
    Returns:
        Dict with optimal configuration and performance metrics
    """
    os.environ['BEAM'] = str(beam_size)
    os.environ['CACHELEVEL'] = '2'  # Enable caching
    # Don't ignore cache - we want to read BEAM results
    if 'IGNORE_BEAM_CACHE' in os.environ:
        del os.environ['IGNORE_BEAM_CACHE']
    
    if len(shape) == 4:
        M, N, K, name = shape
    else:
        M, N, K = shape
        name = f"matmul_{M}x{N}x{K}"
    
    kernel_type = 'matvec' if M == 1 else 'matmul'
    
    print(f"\n{'='*70}")
    print(f"BEAM search: {name} (M={M}, N={N}, K={K})")
    print(f"BEAM size: {beam_size}, type: {kernel_type}")
    print(f"{'='*70}")
    
    try:
        # Start kernel capture
        capture = KernelCapture().start()
        
        # Reset device cache to ensure fresh compilation
        Device[Device.DEFAULT].synchronize()
        
        # Create tensors
        if M == 1:
            a = Tensor.randn(1, K).realize()
            b = Tensor.randn(K, N).realize()
        else:
            a = Tensor.randn(M, K).realize()
            b = Tensor.randn(K, N).realize()
        
        # Warmup with BEAM search - this triggers compilation
        print("Running BEAM search compilation...")
        c = a @ b
        c.realize()
        Device[Device.DEFAULT].synchronize()
        
        # Stop capture
        capture.stop()
        
        # Benchmark
        times = []
        for i in range(iterations):
            if M == 1:
                a = Tensor.randn(1, K).realize()
                b = Tensor.randn(K, N).realize()
            else:
                a = Tensor.randn(M, K).realize()
                b = Tensor.randn(K, N).realize()
            
            start = time.perf_counter()
            c = a @ b
            c.realize()
            Device[Device.DEFAULT].synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times[1:])  # Skip first iteration
        std_time = np.std(times[1:])
        
        # Extract kernel info from captured kernels
        kernel_info = None
        llama_config = None
        
        if capture.kernels:
            # Find the matmul kernel (usually the last one or has matmul in name)
            matmul_kernels = [k for k in capture.kernels if 'matmul' in k['name'].lower() or 'gemm' in k['name'].lower()]
            if not matmul_kernels:
                matmul_kernels = capture.kernels  # Use all if no specific match
            
            if matmul_kernels:
                kernel_info = matmul_kernels[-1]  # Use the last one (most recent)
                
                # Parse the applied opts
                if kernel_info.get('raw_opts'):
                    opts = []
                    for opt_dict in kernel_info['raw_opts']:
                        try:
                            opt = Opt(
                                op=OptOps[opt_dict['op']],
                                axis=opt_dict['axis'],
                                arg=tuple(opt_dict['arg']) if isinstance(opt_dict['arg'], list) else opt_dict['arg']
                            )
                            opts.append(opt)
                        except:
                            pass
                    
                    params = parse_opts_to_params(opts)
                    llama_config = translate_to_llama_cpp(params, (M, N, K), kernel_type)
                
                print(f"\nCaptured kernel: {kernel_info['name']}")
                print(f"  Global size: {kernel_info['global_size']}")
                print(f"  Local size: {kernel_info['local_size']}")
                print(f"  Applied opts: {kernel_info['applied_opts']}")
        
        result = {
            'name': name,
            'shape': (M, N, K),
            'kernel_type': kernel_type,
            'beam_size': beam_size,
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'throughput_gflops': (2 * M * N * K / avg_time) / 1e9,
            'kernel_info': kernel_info,
            'llama_cpp_config': llama_config,
            'raw_kernels': capture.kernels,
        }
        
        print(f"\n  Average time: {avg_time*1000:.3f} ms (std: {std_time*1000:.3f} ms)")
        print(f"  Throughput: {result['throughput_gflops']:.2f} GFLOPS")
        
        if llama_config:
            print(f"\n  llama.cpp recommendations:")
            for param, value in llama_config['llama_cpp_params'].items():
                print(f"    {param}: {value}")
        
        # Cleanup BEAM pool to prevent resource leaks
        cleanup_beam_pool()
        
        return result
        
    except Exception as e:
        print(f"  Error during BEAM search: {e}")
        import traceback
        traceback.print_exc()
        cleanup_beam_pool()
        return None

def compare_beam_sizes(shapes: List[Tuple], beam_sizes: List[int] = [0, 4, 16, 32], iterations: int = 20) -> Dict:
    """Compare performance across different BEAM sizes."""
    print("\n" + "="*70)
    print("BEAM Size Comparison")
    print("="*70)
    
    results = {}
    
    for shape in shapes:
        shape_results = []
        print(f"\nShape: {shape[-1] if len(shape) == 4 else str(shape)}")
        print("-"*70)
        
        baseline = None
        for beam in beam_sizes:
            result = benchmark_kernel(shape, beam_size=beam, iterations=iterations)
            if result:
                shape_results.append(result)
                
                if beam == 0:
                    baseline = result['avg_time_ms']
                    print(f"  BEAM={beam}: {result['avg_time_ms']:.3f} ms (baseline)")
                else:
                    improvement = ((baseline - result['avg_time_ms']) / baseline * 100) if baseline else 0
                    print(f"  BEAM={beam}: {result['avg_time_ms']:.3f} ms ({improvement:+.1f}% vs BEAM=0)")
        
        results[shape[-1] if len(shape) == 4 else str(shape)] = shape_results
    
    return results

def generate_report(results: List[Dict], comparison: Optional[Dict] = None):
    """Generate a comprehensive report of BEAM search results."""
    print("\n" + "="*70)
    print("BEAM Search Results Summary")
    print("="*70)
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': Device.DEFAULT,
        'summary': {},
        'detailed_results': results,
        'comparison_results': comparison,
    }
    
    # Summary statistics
    if results:
        avg_throughput = np.mean([r['throughput_gflops'] for r in results])
        max_throughput = max([r['throughput_gflops'] for r in results])
        
        report['summary'] = {
            'kernels_tested': len(results),
            'average_throughput_gflops': avg_throughput,
            'max_throughput_gflops': max_throughput,
        }
        
        print(f"\nKernels tested: {len(results)}")
        print(f"Average throughput: {avg_throughput:.2f} GFLOPS")
        print(f"Max throughput: {max_throughput:.2f} GFLOPS")
        
        print("\nTop configurations by throughput:")
        sorted_results = sorted(results, key=lambda x: x['throughput_gflops'], reverse=True)
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {r['name']}: {r['throughput_gflops']:.2f} GFLOPS "
                  f"(BEAM={r['beam_size']}, {r['avg_time_ms']:.2f} ms)")
    
    # Save report
    report_path = 'beam_search_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")
    
    # Generate llama.cpp recommendations
    print("\n" + "="*70)
    print("llama.cpp Optimization Recommendations")
    print("="*70)
    
    for result in results:
        config = result.get('llama_cpp_config', {})
        if config and config.get('llama_cpp_params'):
            print(f"\n{result['name']} ({result['shape']}, {result['kernel_type']}):")
            print(f"  Throughput: {result['throughput_gflops']:.2f} GFLOPS")
            print(f"  llama.cpp parameters:")
            params = config['llama_cpp_params']
            for param, value in params.items():
                print(f"    {param}: {value}")
            if config.get('derived_from'):
                derived = config['derived_from']
                print(f"  Derived from TinyGrad opts:")
                if derived.get('local_dims'):
                    print(f"    local_dims: {derived['local_dims']}")
                if derived.get('upcast_dims'):
                    print(f"    upcast_dims: {derived['upcast_dims']}")
                if derived.get('unroll_dims'):
                    print(f"    unroll_dims: {derived['unroll_dims']}")
            
            # Add usage hints
            print(f"  How to apply in llama.cpp:")
            if result['kernel_type'] == 'matmul':
                print(f"    → Set num_warps={params.get('num_warps', 8)} in mul_mat_q kernel config")
                print(f"    → Set tile_y={params.get('tile_y', 64)}, tile_x={params.get('tile_x', 64)}")
                print(f"    → Set vdr={params.get('vdr', 4)} for quantization")
            else:
                print(f"    → Set num_warps={params.get('num_warps', 8)} in mul_mat_vec_q kernel config")
                print(f"    → Set vdr={params.get('vdr', 2)} for vector ops")

def main():
    parser = argparse.ArgumentParser(
        description='TinyGrad BEAM Search for llama.cpp kernel optimization'
    )
    parser.add_argument('--beam', type=int, default=16,
                        help='BEAM search width - number of top configs to keep at each stage '
                             '(0=no search, 4=fast, 16=thorough, 32=exhaustive) (default: 16)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different BEAM sizes')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: from tinygrad)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable DEBUG=2 for detailed output')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of benchmark iterations (default: 20)')
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ['DEBUG'] = '2'
        print("DEBUG=2 enabled - showing detailed kernel compilation output")
    
    if args.device:
        os.environ['DEVICE'] = args.device
    
    print("TinyGrad BEAM Search for llama.cpp Kernel Optimization")
    print("="*70)
    
    # Import after setting device env var
    from tinygrad import Device
    print(f"Device: {Device.DEFAULT}")
    print(f"Default BEAM size: {args.beam}")
    
    # llama.cpp hot kernel shapes (from rocprof analysis)
    HOT_SHAPES = [
        # Matrix-matrix multiplication (mul_mat_q) - prompt processing
        (512, 3584, 3584, "mul_mat_q_pp512"),
        (1024, 3584, 3584, "mul_mat_q_pp1024"),
        (2048, 3584, 3584, "mul_mat_q_pp2048"),
        
        # Vector-matrix multiplication (mul_mat_vec_q) - token generation
        (1, 3584, 3584, "mul_mat_vec_q_tokengen"),
        (1, 17920, 3584, "mul_mat_vec_q_large"),
        (1, 14336, 3584, "mul_mat_vec_q_mixed"),
        
        # Smaller token generation
        (1, 3584, 512, "mul_mat_vec_q_small"),
    ]
    
    print(f"Iterations per kernel: {args.iterations}")
    print(f"Total kernels to test: {len(HOT_SHAPES)}")
    
    # Run BEAM search
    if args.compare:
        comparison = compare_beam_sizes(HOT_SHAPES, beam_sizes=[0, 4, 16, 32], iterations=args.iterations)
        results = []
        for shape_results in comparison.values():
            if shape_results:
                best = max(shape_results, key=lambda x: x['throughput_gflops'])
                results.append(best)
    else:
        comparison = None
        results = []
        for shape in HOT_SHAPES:
            result = benchmark_kernel(shape, beam_size=args.beam, iterations=args.iterations)
            if result:
                results.append(result)
    
    # Generate report
    generate_report(results, comparison)
    
    # Final cleanup - force terminate all beam pools
    cleanup_beam_pool()
    
    # Suppress atexit errors by removing beam_pool before TinyGrad's handlers run
    try:
        import tinygrad.codegen.opt.search as search_module
        if hasattr(search_module, 'beam_pool'):
            search_module.beam_pool = None
    except:
        pass
    
    print("\n" + "="*70)
    print("BEAM Search Complete")
    print("="*70)
    print("\nNext steps:")
    print("1. Review beam_search_report.json for detailed results")
    print("2. Apply recommended configurations to llama.cpp kernel parameters")
    print("   (see ggml-cuda/ggml-hip files, modify kernel template parameters)")
    print("3. Rebuild llama.cpp and verify performance improvements")
    print("   (cd /path/to/llama.cpp && cmake --build build --target llama-cli)")
    print("4. Run rocprof to validate on actual model inference")
    print("\nNote: llama.cpp kernel parameters are defined in template code and require")
    print("      rebuilding after modification. The BEAM search results guide which")
    print("      values to test in the kernel template definitions.")

if __name__ == "__main__":
    main()
