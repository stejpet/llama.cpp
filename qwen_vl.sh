#!/bin/bash
./build/bin/llama-server \
    --model ../models/Qwen3-VL-30B-A3B-Instruct-Q5_K_S.gguf \
    --mmproj ../models/mmproj-F16.gguf \
    --threads -1 \
    --ctx-size 65536 \
    --n-gpu-layers 123 \
    --jinja \
    --prio 2 \
    --host 0.0.0.0 \
    --top-p 0.8 \
    --top-k 20 \
    --temp 0.7 \
    --min-p 0.0 \
    --flash-attn on \
    --presence-penalty 1.5 \
