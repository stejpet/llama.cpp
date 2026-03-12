#!/bin/bash
./build/bin/llama-server \
    --model ../models/Qwen_QwQ-32B-Q6_K.gguf \
    --threads 8 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    --seed 3407 \
    --prio 2 \
    --temp 0.7 \
    --repeat-penalty 1.1 \
    --dry-multiplier 0.5 \
    --min-p 0.1 \
    --top-k 40 \
    --top-p 0.95 \
    --host 10.0.0.224 \
    --mlock \
    --ubatch-size 128 \
    --samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc" \
