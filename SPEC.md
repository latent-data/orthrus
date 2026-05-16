# Instructions for Claude Code: Reproducible Orthrus Benchmark Repo

## Goal

Turn this fork into a self-contained, publishable repo that anyone with a DGX
Spark (or compatible GB10 / Blackwell sm_121 system) can clone and run to
reproduce benchmark numbers comparing the Orthrus-Qwen3-8B diffusion LM against
a stock Qwen/Qwen3-8B autoregressive baseline.

Existing artefact: `benchmark.py` at the repo root. Treat it as authoritative.
Do not rewrite its measurement logic. You may refactor for clarity if needed
but keep the same three measurements (Orthrus diffusion, Orthrus
use_diffusion_mode=False, stock Qwen3-8B AR).

## Hardware target

- NVIDIA DGX Spark (or equivalent): NVIDIA GB10, compute capability 12.1 (sm_121),
  aarch64 CPU, 128 GB unified memory, CUDA 13.x
- The repo MUST NOT assume x86_64 or discrete-GPU memory semantics
- Tested working configuration:
  - NGC container `nvcr.io/nvidia/pytorch:25.12-py3` (or newer if available)
  - torch 2.10.0a0+...nv25.12 (the container's custom build, do not replace)
  - flash-attn 2.7.x (preinstalled in container)
  - transformers >= 5.8.0
  - accelerate >= 1.13.0

## Repo structure to produce

```
.
├── README.md              # user-facing, see spec below
├── Dockerfile             # extends NGC pytorch container
├── benchmark.py           # already exists, do not rewrite measurement logic
├── run.sh                 # one-command runner
├── requirements.txt       # extras needed on top of NGC container
├── results/
│   ├── .gitkeep
│   └── EXAMPLE_results.json
├── .gitignore
└── LICENSE                # Apache-2.0
```

## File-by-file spec

### Dockerfile

- Base image: `nvcr.io/nvidia/pytorch:25.12-py3` (parameterise via ARG so the
  user can override to a newer tag)
- Working dir: `/workspace`
- Install requirements.txt with `pip install --no-deps -r requirements.txt`
  (the --no-deps is critical, see "Pitfalls" below)
- Copy benchmark.py into the image
- Default CMD runs `python benchmark.py`
- Do NOT install or upgrade torch, flash-attn, or triton
- Do NOT switch the base image to a generic Python image

### requirements.txt

Pin to working versions discovered in testing:

```
transformers==5.8.1
accelerate==1.13.0
```

Add nothing else. Numpy is already in the container. Do NOT add torch,
flash-attn, or triton.

### run.sh

A single bash script that:

1. Checks for `nvidia-smi` and the NVIDIA Container Toolkit, errors out clearly
   if missing
2. Builds the image (or pulls if you ship a prebuilt one, but local build is
   fine for v1)
3. Runs the container with:
   - `--gpus all`
   - `--ipc=host`
   - `--ulimit memlock=-1 --ulimit stack=67108864`
   - HF cache mount: `-v "$HOME/.cache/huggingface:/root/.cache/huggingface"`
     so the 19 GB Orthrus + 16 GB Qwen3 weights aren't redownloaded on every run
   - Results mount: `-v "$PWD/results:/workspace/results"`
   - `--rm`
4. The script should accept optional args that get forwarded to benchmark.py
   (e.g. `--prompt`, `--max-new-tokens`)

### benchmark.py changes (small, additive only)

Do NOT rewrite the measurement logic. Add:

1. argparse with flags:
   - `--prompt` (default = current hardcoded prompt)
   - `--max-new-tokens` (default 2048)
   - `--warmup-tokens` (default 32)
   - `--output` (path to JSON output, default `results/results.json`)
   - `--orthrus-revision` (HF commit SHA for chiennv/Orthrus-Qwen3-8B,
     default: pin to a specific SHA you fetch and bake in for reproducibility)
   - `--qwen-revision` (HF commit SHA for Qwen/Qwen3-8B, same treatment)
2. Pass `revision=` to both `from_pretrained` calls. This is critical for
   reproducibility, HF models can change under you.
3. Set deterministic decoding: `do_sample=False` for both models. Document this
   in the output.
4. At end of main(), emit a JSON file with:
   - hardware info (`torch.cuda.get_device_name`, capability, mem)
   - container/image info (read from env vars baked in by Dockerfile)
   - torch/transformers/accelerate/flash-attn versions
   - prompt, max_new_tokens, warmup_tokens
   - per-model: tokens generated, elapsed, throughput, output snippet
   - top-level speedup ratio
   - timestamp (UTC)

### results/EXAMPLE_results.json

A committed example showing the JSON schema, populated with the Orthrus
numbers from initial testing (see below). Mark the Qwen3-8B baseline as
`null` with a comment that it needs to be filled in after a real run.

### README.md

Sections, in this order:

1. **Title and one-line summary**: "Benchmark of Orthrus-Qwen3-8B diffusion LM
   on NVIDIA DGX Spark"
2. **TL;DR table**: tok/s for each of the three configurations and the diffusion
   vs stock Qwen3-8B AR speedup
3. **Hardware tested**: list the exact hardware, container, software versions
4. **What this measures and what it does NOT**:
   - We measure single-stream throughput on a fixed prompt with greedy decoding
   - Not measured: quality, latency under load, batch throughput, longer
     contexts, different prompts, sampling strategies
   - Why three numbers and not two: explain that Orthrus's
     use_diffusion_mode=False is NOT a clean AR path (no KV cache in a model
     trained with bidirectional attention), so stock Qwen3-8B AR is the honest
     baseline
5. **Quick start**:
   ```
   git clone <this repo>
   cd <this repo>
   ./run.sh
   ```
6. **Configuration**: flags documented
7. **Output format**: link to results/EXAMPLE_results.json, describe schema
8. **Reproducibility notes**:
   - Pinned model revisions (list the SHAs)
   - Pinned container tag
   - Pinned dependency versions
   - Greedy decoding (deterministic)
   - First run downloads ~35 GB of weights
9. **Limitations and caveats**:
   - sm_121 / aarch64 specific; will not run as-is on other Blackwell variants
     without container adjustments
   - flash-attn 2 only (FA3/FA4 do not yet support sm_121, as of repo authoring)
   - Single GPU only
   - 8B model only, no quantisation
10. **License**: Apache-2.0, link to LICENSE file
11. **Acknowledgements**: link to Orthrus paper and repo, Qwen3 model card,
    DGX Spark community guides (martimramos/dgx-spark-ml-guide,
    natolambert/dgx-spark-setup)

### .gitignore

Standard Python + add:
- `results/*.json`
- `!results/EXAMPLE_*.json`
- `!results/.gitkeep`
- `.venv/`
- `__pycache__/`

### LICENSE

Apache-2.0 boilerplate. Use the standard text.

## Pitfalls discovered during development (do not regress on these)

1. **Do not let pip upgrade torch.** The NGC container ships a custom alpha
   build of torch patched for sm_121. Any `pip install torch` or transitive
   upgrade will pull a generic wheel and break GB10 support. Use `--no-deps`
   when installing project requirements.
2. **`torch>=2.10.0` rejects `2.10.0a0`** per PEP 440. The repo must not have
   that constraint anywhere.
3. **Do not install flash-attn from pip.** The container has a working FA2
   build. flash-attn 4 (CuTe DSL) does not support sm_121. Building FA2 from
   source on the host without `TORCH_CUDA_ARCH_LIST="12.0"` will fail.
4. **Do not switch attention impl to sdpa for Orthrus.** Orthrus is a
   diffusion LM with bidirectional attention; sdpa produces degenerate output
   (repeating tokens). Use flash_attention_2.
5. **Do not skip `--ipc=host` and the memlock ulimits**. Some HF dataloaders
   and large allocations need them.
6. **HF cache must be mounted**. Without it, every container run redownloads
   ~35 GB of model weights.

## Numbers from initial testing (use in EXAMPLE_results.json and README TL;DR)

Same prompt ("Write a program to count the frequency of each word in a
paragraph."), max_new_tokens=2048, on a DGX Spark with the configuration
above:

- Orthrus diffusion mode: ~38.9 tok/s (generated 472 tokens before EOS)
- Orthrus use_diffusion_mode=False: ~3.1 tok/s
- Stock Qwen3-8B AR with KV cache: NOT YET MEASURED, expected ~12-18 tok/s

Populate EXAMPLE_results.json with the Orthrus numbers above and a `null` or
TODO marker for the Qwen3-8B baseline. Note in the README that the TL;DR table
will be finalised after the first benchmark run on real hardware.

## Style

- No emdashes in any prose (user preference)
- Concise README, no marketing prose, no benchmark-result hyperbole
- Code: PEP 8, no overengineering, no abstractions for their own sake
- Apache-2.0 headers in source files are optional, not required

## Acceptance criteria

A user with a DGX Spark and a clone of this repo should be able to:
1. Run `./run.sh` with zero prior setup other than docker + NVIDIA Container Toolkit
2. Wait for first-run model downloads
3. See a `results/results.json` file appear with full data
4. Confidently publish the numbers without footnotes about
   "we benchmarked our model's broken AR mode against itself"

When done, run a dry sanity check:
- `python -c "import ast; ast.parse(open('benchmark.py').read())"`
- `bash -n run.sh`

Do not run the benchmark itself, it requires GB10 hardware you don't have.


## Update: multi-prompt benchmarking

benchmark.py must now support running multiple prompts in a single invocation.

- Define a `PROMPTS` dict in benchmark.py with named entries:
  - `"short"`: the existing default prompt (word frequency program)
  - `"long"`: the BoundedPriorityQueue prompt (full text below)
- Add a `--prompts` CLI arg (action=append, default=["short", "long"]) that
  selects which named prompts to run. Reject unknown names with a clear error.
- For each selected prompt, run all three model configurations (Orthrus
  diffusion, Orthrus use_diffusion_mode=False, Qwen3-8B AR). Reuse the loaded
  model across prompts within a single config to avoid reloading.
- Summary table gets a new "prompt" column. Results JSON groups by prompt
  name then by config.
- Speedup ratio is reported per prompt AND as an overall (geometric mean)
  across prompts.
- Update README TL;DR table to show both prompts and both speedups.

PROMPTS["long"] text:
"Implement a Python class BoundedPriorityQueue backed by a binary heap, ..."
[full prompt text as above]
