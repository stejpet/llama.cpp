#!/bin/bash
./build/bin/llama-server \
    --model ../models/Qwen_Qwen3-32B-Q5_K_L.gguf \
    --threads 8 \
    --jinja \
    --ubatch-size 128 \
    --batch-size 128 \
    --main-gpu 0 \
    --ctx-size 40960 \
    --predict 32768 \
    --n-gpu-layers 99 \
    --split-mode row \
    --temp 0.6 \
    --min-p 0 \
    --top-k 20 \
    --top-p 0.95 \
    --presence-penalty 1.5 \
    --host 10.0.0.224 \
    --no-mmap \
    --ubatch-size 128 \
