#!/bin/bash
./build/bin/llama-server \
    --model ../models/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf \
    --threads 16 \
    --prio 3 \
    --flash-attn on \
    --predict 32768 \
    --jinja \
    --ctx-size 64568 \
    --n-gpu-layers 123 \
    --batch-size 1024 \
    --ubatch-size 1024 \
    --temp 0.7 \
    --repeat-penalty 1.05 \
    --top-k 20 \
    --top-p 0.8 \
    --host 10.0.0.197 \
