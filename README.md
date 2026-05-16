# Orthrus-Qwen3-8B benchmark on NVIDIA DGX Spark

Reproducible throughput benchmark comparing the Orthrus-Qwen3-8B diffusion LM against a stock Qwen3-8B autoregressive baseline on GB10 (sm_121) hardware.

## TL;DR

Results on two prompts (greedy decoding, max 2048 new tokens):

| Prompt | Orthrus diffusion | Orthrus nodiff | Qwen3-8B AR | Speedup (diffusion vs AR) |
|---|---|---|---|---|
| short | ~38.9 tok/s | ~3.1 tok/s | TBD | TBD |
| long | TBD | TBD | TBD | TBD |
| **geometric mean** | | | | **TBD** |

> These numbers will be filled in after a complete run on real hardware. The Orthrus short-prompt numbers are from initial testing; all Qwen3-8B and long-prompt numbers are pending.

Diffusion's throughput advantage is expected to grow with output length because Orthrus generates tokens in parallel passes while AR decoding scales linearly. If the per-prompt speedups differ substantially (e.g. 3x short vs 6x long), that finding is reported explicitly -- the geometric mean is the right summary statistic but the per-prompt breakdown is the interesting result.

## Hardware tested

- **System:** NVIDIA DGX Spark
- **GPU:** NVIDIA GB10, compute capability 12.1 (sm_121), 128 GB unified memory
- **CPU:** aarch64
- **Container:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **torch:** 2.10.0a0+...nv25.12 (custom NGC build, not a PyPI release)
- **flash-attn:** 2.7.x (preinstalled in container)
- **transformers:** 5.8.1
- **accelerate:** 1.13.0

## What this measures and what it does not

**Measured:** single-stream throughput on two fixed prompts with greedy (deterministic) decoding, up to 2048 new tokens each.

**Not measured:** output quality, latency under concurrent load, batch throughput, different prompts, sampling strategies.

**Why three numbers and not two:**
Orthrus exposes a `use_diffusion_mode=False` flag, but this is not a clean autoregressive path. The model was trained with bidirectional attention, so it has no KV cache in that mode and its numbers are not a fair comparison to standard AR decoding. Stock Qwen3-8B with a proper KV cache is the honest baseline. The `use_diffusion_mode=False` result is included for transparency, not as the comparison worth quoting.

**Why two prompts:**
A single short prompt under-represents Orthrus's advantage. Diffusion models generate tokens in parallel passes; the benefit grows with output length. The long prompt (BoundedPriorityQueue implementation + test suite) produces significantly more tokens and gives a second data point on how speedup scales. The geometric mean across both prompts is the headline number; the per-prompt breakdown is worth inspecting.

**Expected runtime:** approximately 20-30 minutes total on a DGX Spark, dominated by the `use_diffusion_mode=False` run on the long prompt (O(n^2) in output length).

## Quick start

```
git clone <this repo>
cd <repo dir>
./run.sh
```

To skip the Docker build step and run directly in a pulled NGC container:

```
./run.sh --no-build
```

Results appear in `results/results.json`. First run downloads approximately 35 GB of model weights into `~/.cache/huggingface`.

## Configuration

Flags are forwarded from `run.sh` to `benchmark.py`:

| Flag | Default | Description |
|---|---|---|
| `--prompts NAME` | short, long | Named prompt(s) to run. Repeatable. |
| `--max-new-tokens` | 2048 | Maximum tokens to generate per run |
| `--warmup-tokens` | 32 | Tokens generated before timing starts |
| `--output` | results/results.json | Path for JSON output |
| `--orthrus-revision` | pinned SHA | HF commit for chiennv/Orthrus-Qwen3-8B |
| `--qwen-revision` | pinned SHA | HF commit for Qwen/Qwen3-8B |

Examples:

```bash
# Run only the short prompt
./run.sh --no-build --prompts short

# Custom output path
./run.sh --no-build --output results/my_run.json
```

## Output format

See `results/EXAMPLE_results.json` for the full schema. Top-level fields:

- `timestamp_utc` - ISO 8601 timestamp
- `hardware` - device name, compute capability, memory
- `container` - image tag and name (baked in by Dockerfile)
- `software` - torch, CUDA, transformers, accelerate, flash-attn versions
- `config` - prompts selected, token limits, decoding settings, pinned revisions
- `results` - keyed by prompt name, then by config: tokens generated, elapsed seconds, tok/s, 300-char output snippet
- `speedups` - per-prompt speedup (diffusion vs Qwen3-8B AR) and `geomean` across prompts

## Reproducibility notes

- **Pinned model revisions:**
  - `chiennv/Orthrus-Qwen3-8B`: `34429bd987c2750bed61d65583c6879964367059`
  - `Qwen/Qwen3-8B`: `b968826d9c46dd6066d109eabc6255188de91218`
- **Pinned container:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **Pinned dependencies:** transformers 5.8.1, accelerate 1.13.0, huggingface-hub 1.9.2 (see `requirements.txt`)
- **Deterministic decoding:** `do_sample=False` (greedy) for all configurations
- First run downloads approximately 19 GB (Orthrus) + 16 GB (Qwen3-8B) of weights

## Limitations and caveats

- Targets sm_121 (GB10) on aarch64. Will not run as-is on x86_64 or other Blackwell variants without container adjustments.
- Uses flash-attn 2 only. FA3/FA4 do not yet support sm_121 as of the time this repo was authored.
- Single GPU only.
- 8B model only, no quantisation.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

- Orthrus paper: [arxiv.org/abs/2605.12825](https://arxiv.org/abs/2605.12825) / [chiennv2000/orthrus](https://github.com/chiennv2000/orthrus)
- Qwen3-8B model card: [huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- DGX Spark community guides: [martimramos/dgx-spark-ml-guide](https://github.com/martimramos/dgx-spark-ml-guide), [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup)
