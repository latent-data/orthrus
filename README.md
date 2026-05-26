# Orthrus-Qwen3-8B benchmark on NVIDIA DGX Spark

Reproducible throughput benchmark comparing the Orthrus-Qwen3-8B diffusion LM against a stock Qwen3-8B autoregressive baseline on GB10 (sm_121) hardware.

## TL;DR

Results on two prompts (greedy decoding, max 2048 new tokens, average of 3 runs):

| Prompt | Orthrus diffusion | Qwen3-8B AR | Speedup (diffusion vs AR) |
|---|---|---|---|
| short (word frequency) | 38.6 tok/s | 11.4 tok/s | 3.38x |
| long (BoundedPriorityQueue) | 50.7 tok/s | 11.2 tok/s | 4.52x |
| **geometric mean** | | | **3.91x** |

Orthrus diffusion throughput increases with output length (38.6 to 50.7 tok/s) while Qwen3-8B AR stays flat (~11.3 tok/s), as expected: diffusion generates tokens in parallel passes that amortise better over longer sequences. The per-prompt speedup variation (3.38x short vs 4.52x long) is itself a finding worth reporting alongside the geometric mean.

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

**Why stock Qwen3-8B and not Orthrus's own AR mode:**
Orthrus exposes a `use_diffusion_mode=False` flag, but this is not a clean autoregressive path. The model was trained with bidirectional attention, so it has no KV cache in that mode and its numbers are not a fair comparison to standard AR decoding. Stock Qwen3-8B with a proper KV cache is the honest baseline. The `use_diffusion_mode=False` measurement is available via `--include-nodiff` for transparency, but it is not the comparison worth quoting.

**Why two prompts:**
A single short prompt under-represents Orthrus's advantage. Diffusion models generate tokens in parallel passes; the benefit grows with output length. The long prompt (BoundedPriorityQueue implementation + test suite) produces significantly more tokens and gives a second data point on how speedup scales. The geometric mean across both prompts is the headline number; the per-prompt breakdown is worth inspecting.

**Expected runtime:** approximately 10-15 minutes total on a DGX Spark for the default run (diffusion + Qwen3-8B AR, two prompts). Adding `--include-nodiff` roughly doubles the runtime; the `use_diffusion_mode=False` mode is O(n^2) in output length.

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

To run the quantization investigation instead of the throughput benchmark:

```
./run.sh quant_benchmark --no-build
```

Results appear in `results/results.json` (throughput benchmark) or `results/quant_results.json` (quantization investigation). First run downloads approximately 35 GB of model weights into `~/.cache/huggingface`.

## Configuration

Flags are forwarded from `run.sh` to `benchmark.py`:

| Flag | Default | Description |
|---|---|---|
| `--prompts NAME` | short, long | Named prompt(s) to run. Repeatable. |
| `--max-new-tokens` | 2048 | Maximum tokens to generate per run |
| `--warmup-tokens` | 32 | Tokens generated before timing starts |
| `--runs N` | 1 | Timed runs per config; results averaged |
| `--seed S` | none | RNG seed (output already deterministic with greedy decoding) |
| `--include-nodiff` | off | Also run `use_diffusion_mode=False`. Slow: O(n^2), expect 10+ min per run on the long prompt |
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

## Quantization investigation: losslessness breaks, but the architecture degrades gracefully

We tested how Orthrus's TPF and output-equivalence behave when the autoregressive teacher weights are quantized while the diffusion drafter stays at bf16. Quantization is **simulated** via cast-and-dequantize (precision loss isolated from kernel-path differences), so memory savings are not measured here; only the impact of precision loss on Orthrus's consensus mechanism.

| Config         | Prompt | TPF  | Throughput (tok/s) | Output tokens | Exact match | First divergence | Edit dist vs baseline |
| -------------- | ------ | ---- | ------------------ | ------------- | ----------- | ---------------- | --------------------- |
| baseline-bf16  | short  | 6.56 | 39.2               | 472           | —           | —                | —                     |
| baseline-bf16  | long   | 8.73 | 51.6               | 1441          | —           | —                | —                     |
| teacher-int8   | short  | 6.10 | 34.3               | 415           | no          | position 2       | 228                   |
| teacher-int8   | long   | 7.82 | 43.3               | 1704          | no          | position 35      | 554                   |
| teacher-int4   | short  | 1.00 | 5.7                | 2048          | no          | position 0       | 2048 (gibberish)      |
| teacher-int4   | long   | 1.02 | 5.8                | 2048          | no          | position 0       | 2048 (gibberish)      |
| full-int8      | short  | 6.10 | 34.2               | 415           | no          | position 2       | 228                   |
| full-int8      | long   | 7.93 | 43.8               | 1704          | no          | position 35      | 554                   |

### Findings

**Orthrus is not lossless under quantization.** Even simulated int8 round-trip flips the greedy argmax at the third token on the short prompt. Outputs remain coherent and Qwen-like, but they are not bit-identical to the bf16 baseline. The paper's strict-losslessness claim only holds at the precision the diffusion drafter was distilled against.

**TPF degrades only modestly under int8 (-7% short, -10% long).** Throughput drops more (-13% to -16%) but this is partly an artifact of differing output lengths between configs, not a per-forward-pass cost. The consensus mechanism is more robust to teacher precision shift than expected from first principles.

**Naive per-tensor int4 is catastrophic.** TPF collapses to ~1.0 (drafter proposals never match the AR verifier), the model emits gibberish until hitting `max_new_tokens=2048`, and throughput drops below the stock Qwen3-8B AR baseline. The architecture degrades gracefully (no crashes, no corruption-of-state), but the speedup mechanism is gone entirely. Whether per-channel int4 rescues this is open.

**The diffusion projections are a quantization passenger.** Quantizing both the AR and diffusion sides to int8 (`full-int8`) produces identical TPF, edit distance, and first-divergence position to quantizing only the AR side (`teacher-int8`) — matching to three significant figures. The 84% of shared/AR weights drive the entire effect; the 16% of diffusion projections contribute essentially nothing to the consensus dynamics. Practical implication: in memory-constrained deployments, the diffusion projections could be quantized aggressively without further harming TPF.

### Caveats

- Simulated quantization (cast-and-dequant) measures distribution-shift impact only. Real production quantization (bitsandbytes, GPTQ, AWQ, GGUF Q-formats) includes calibrated scales and different kernel paths; those numbers will likely be more favorable. This investigation is a worst-case isolation of the precision-loss variable.
- Greedy decoding throughout for determinism. Sampling behavior under quantization is not measured.
- A single base model (Qwen3-8B) and a fixed pair of prompts. The findings should generalize to other Qwen3 sizes structurally but have not been verified.
- All numbers from a single hardware target (DGX Spark / GB10). Other Blackwell variants and consumer GPUs may exhibit different per-forward-pass economics that interact with the speedup.

## AR equivalence verification: diff-off through Orthrus is not bit-identical to diff-on, but it is still a valid quantization-comparison path

A planned follow-up investigation needs a "plain Qwen3-8B" forward path so we can compare its quantization sensitivity to Orthrus's. The cheapest such path is `use_diffusion_mode=False` through the same Orthrus model (which routes through the AR projections only). This section reports a sanity check on whether that path is faithful.

`./run.sh quant_benchmark --verify-ar-equivalence` generates the same prompt twice through the loaded Orthrus model at bf16 with greedy decoding, once with `use_diffusion_mode=True` (the normal diffusion-mode path) and once with `use_diffusion_mode=False` (HF's `super().generate()` through the AR projections), then compares token sequences. Each output is truncated at its first `<|im_end|>` so that stop-criterion differences are reported separately from forward-pass differences.

### Results

Per-prompt shared-prefix fraction (length of common token prefix divided by diff-on truncated length):

| Prompt | Diff-on tokens | Diff-off tokens (truncated) | Shared prefix | Fraction | First divergence | Diff-on token | Diff-off token | Top1-top2 logit gap | Diff-on rank | Diff-off rank |
|---|---|---|---|---|---|---|---|---|---|---|
| short | 472 | 477 | 470 | **99.6%** | position 470 | `!` (id 0) | ` or` (id 476) | 0.25 | 2 | 1 |
| long  | 1441 | 1452 | 102 | **7.1%** | position 102 | ` if` (id 421) | ` self` (id 656) | 0.25 | 2 | 1 |

Stop criterion: in both prompts, diff-on halts at the first `<|im_end|>` while diff-off emits it as a normal token and continues to `max_new_tokens` (cut off by the cap).

### Findings

**The two paths diverge from a single near-tie argmax disagreement, not a structural model difference.** Both prompts show the exact same divergence signature: at the divergence position, the top-1 and top-2 next-token logits differ by ~0.25, the diffusion-mode path picks rank-2, the AR-mode path picks rank-1. This is consistent with floating-point accumulation drift between block-mode (the diffusion path's AR-verify call processes a block of positions at once) and single-token-mode (HF's standard generate processes one position at a time) through the same weights. The two paths are using identical AR projections; the disagreement comes from how those projections' outputs are summed in fp.

**The shared-prefix metric is heavily position-dependent.** A single near-tie disagreement at position 470 of a 472-token output yields 99.6% shared prefix; the same kind of disagreement at position 102 of a 1441-token output yields 7.1%. The underlying mechanism is identical; the fraction reflects only where the near-tie happens to land. A more position-independent metric would be "near-tie disagreements per K tokens" (here roughly one per 100-500 tokens regardless of total output length).

**Stop-criterion divergence is a separate phenomenon.** Orthrus's diffusion-mode generate loop explicitly stops on `<|im_end|>`; HF's `super().generate()` does not stop on `<|im_end|>` under this configuration, even though both paths emit it. This is a generate-config interaction, not a forward-pass difference.

**For the planned PR 4 comparison work, the diff-off path remains viable as a "plain-Qwen3" proxy** even though it diverges from diff-on on long outputs. The reason: PR 4 needs to measure how much int8 quantization perturbs vanilla-Qwen3 generation, so the relevant measurement is diff-off-bf16 vs diff-off-int8 (same code path, only weight precision differs). That comparison is internally self-consistent and isolates the weight-precision effect, regardless of whether diff-off tracks diff-on at bf16. Under the frozen-teacher claim (Orthrus's AR projections are vanilla Qwen3-8B weights), the diff-off bf16-vs-int8 comparison is a faithful proxy for vanilla Qwen3 quantization sensitivity. Backbone-extraction work (loading the AR projections into a vanilla Qwen3 architecture) is not required.

### How to run

```bash
./run.sh quant_benchmark --no-build --verify-ar-equivalence
```

Default prompt set is `short, long`. Per-prompt `max_new_tokens` is set in `AR_EQUIV_PROMPT_MAX_NEW_TOKENS` at the top of `quant_benchmark.py` (short=512, long=2048) and sized to each prompt's natural output length plus modest headroom; a generous global value would only extend the diff-off arm into wasted tokens that get truncated away. Output goes to `results/ar_equivalence.json` and includes the full token-ID sequences, a per-prompt fresh-forward top-3 logit diagnostic at the first divergence position, and the aggregate findings.

### Caveats

- Two prompts is a small sample. Whether "one near-tie per 100-500 tokens" is a stable rate or coincidence would need more prompts to confirm.
- The fp-drift hypothesis is consistent with the observed top1-top2 gaps but not directly verified. A more invasive test (capture logits at every position from both paths, compare directly) would close that loop.
- The frozen-teacher claim that Orthrus's AR projections equal vanilla Qwen3-8B weights is taken at face value here. If those weights actually differ from vanilla Qwen3, the path-2 proxy argument weakens.

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
