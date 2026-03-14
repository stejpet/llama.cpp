# autoresearch

This is an experiment to have the LLM do its own research on inference optimization.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is large. Read these key files for full context:
   - `README.md` — repository context and supported models.
   - `ggml/` — tensor operations and backend implementations. Do not modify core GGML.
   - `src/llama-model.cpp`, `src/llama-context.cpp` — model loading and inference context.
   - `common/` — shared utilities and argument parsing.
4. **Verify build exists**: Check that `/home/steffen/llama.cpp/build/bin/` contains binaries (especially `llama-bench`). If not, build with:
   ```bash
   cmake -B build
   cmake --build build --config Release -j 8
   ```
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs benchmarks on available hardware. The focus is on **inference performance optimization** — improving tokens/second, reducing latency, or optimizing memory usage.

**What you CAN do:**
- Modify ANY source code in the repository for experimentation purposes
- Adjust build flags and CMake configuration
- Modify `ggml/` tensor operations and backend implementations
- Modify `src/` model architecture, context handling, sampling code
- Modify benchmark tools or create new ones
- Experiment with different quantization formats
- Tune batch sizes, context lengths, and other inference parameters
- Test different backend configurations (CPU, GPU, hybrid)

**What you CANNOT do:**
- Submit AI-generated code to the main llama.cpp repository (see CONTRIBUTING.md)
- Add external dependencies without verifying they build cleanly
- Break the fundamental build system (must still compile successfully)
- Push your autoresearch branch to the main repository

**The goal is simple: get the best performance metrics.** This could mean:
- Higher throughput (tokens/second)
- Lower latency (time to first token, per-token latency)
- Better memory efficiency
- Optimal quantization with minimal quality loss

**Hardware constraints**: Respect the available hardware. Don't attempt GPU-only optimizations if no GPU is available. Use the build system to detect and use available backends.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always establish the baseline. Use the reference build:
```bash
/home/steffen/llamacpp-test/llama.cpp/build/bin/llama-bench -m ~/models/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf -p 1024 -r 1
```

## Output format

`llama-bench` prints a summary like this:

```
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -----------------: |
| qwen3.5 35B A3B UD-Q4_K_M    |  21.76 GiB |    35.65 B | HIP        |  99 |         pp512 |      515.73 ± 0.38 |
| qwen3.5 35B A3B UD-Q4_K_M    |  21.76 GiB |    35.65 B | HIP        |  99 |         tg128 |       34.82 ± 0.02 |

build: abc1234 (1234)
```

Key metrics:
- `pp512`: prompt processing speed (tokens/second) for 512 tokens
- `tg128`: token generation speed (tokens/second) for 128 tokens
- Model size and quantization format
- Backend used (HIP, CUDA, Metal, CPU)

Extract metrics:
```bash
llama-bench ... | grep -E "pp|tg|build:"
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	pp512_tg128	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. Combined metric: `pp512/tg128` scores (e.g., `515.73/34.82`)
3. peak memory in GB (from benchmark output or system monitoring)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	pp512_tg128	memory_gb	status	description
a1b2c3d	515.73/34.82	21.8	keep	baseline Q4_K_M quantization
b2c3d4e	520.15/35.10	21.8	keep	switch to Q5_K_M quantization
c3d4e5f	510.20/33.50	18.5	discard	Q3_K_M quantization (quality loss)
d4e5f6g	0/0	0.0	crash	OOM with Q8_0 quantization
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar13` or `autoresearch/mar13-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Formulate an experimental idea (e.g., different quantization, batch size, backend settings)
3. Execute the experiment: Run `llama-bench` with appropriate parameters
4. Read out the results: Extract pp512 and tg128 values
5. If the run crashed (no output, error message), read the error log and attempt a fix
6. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
7. If metrics improved (higher t/s), you "advance" the branch, keeping the approach
8. If metrics are equal or worse, you reset and try a different approach

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should complete within a few minutes. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, invalid parameters, etc.), use your judgment: If it's something dumb and easy to fix (e.g., wrong batch size), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the documentation for new angles, try combining previous near-misses, try more parameter combinations. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

