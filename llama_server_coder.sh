#!/bin/bash
./build/bin/llama-server \
    --model ../models/Qwen_Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf \
    --threads 8 \
    --ctx-size 65536 \
    --n-gpu-layers 123 \
    --jinja \
    --flash-attn on \
    --prio 2 \
    --temp 0.6 \
    --repeat-penalty 1.1 \
    --dry-multiplier 0.5 \
    --host 0.0.0.0 \
    --ubatch-size 1024 \
    --batch-size 1024 \
