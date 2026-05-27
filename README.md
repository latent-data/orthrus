# Orthrus-Qwen3-8B benchmark on NVIDIA DGX Spark

Reproducible throughput and quantisation-sensitivity benchmarks for the Orthrus-Qwen3-8B diffusion language model on GB10 (sm_121) hardware. The full investigation, including methodology, per-prompt tables, and mechanistic discussion, lives in [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Headline results

**Throughput** (greedy decoding, max 2048 new tokens, average of 3 runs):

| Prompt | Orthrus diffusion | Qwen3-8B AR | Speedup |
|---|---|---|---|
| short (word frequency) | 38.6 tok/s | 11.4 tok/s | 3.38x |
| long (BoundedPriorityQueue) | 50.7 tok/s | 11.2 tok/s | 4.52x |
| **geometric mean** | | | **3.91x** |

AR = autoregressive (one token at a time through the model). Diffusion mode generates a block of tokens per forward pass and verifies them, so its advantage grows with output length.

**Quantisation sensitivity:** Orthrus is *not* uniquely fragile to post-training quantisation (PTQ — converting a trained checkpoint to a lower precision without retraining). Within-arm bf16-vs-int8 comparison shows Orthrus and a vanilla-Qwen3 AR path produce **bit-identical perturbation events on 6 of 7 prompts**. The diffusion consensus mechanism propagates the autoregressive head's output as-is rather than amplifying precision noise. This eliminative finding cleared the methodological doubt that motivated the productionisation work in the sibling [orthrus-serve](../orthrus-serve) repo, where real fp8 and NVFP4 schemes now run end-to-end on the same hardware.

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the full investigation: the AR-equivalence verification, the within-arm quant comparison, the TPF (tokens-per-forward-pass) variance breakdown, the explanation-prompt exception, and the handoff back to orthrus-serve.

## Quick start

```bash
git clone <this repo>
cd <repo dir>
./run.sh                     # throughput benchmark
./run.sh quant_benchmark     # quantisation investigation
./run.sh --no-build          # skip Docker build; use pulled NGC container
```

Results land in `results/results.json` (throughput) or `results/quant_results.json` (quant). First run downloads ~35 GB of model weights.

See [RESEARCH_LOG.md](RESEARCH_LOG.md#configuration) for the full flag set and configuration options.

## Hardware tested

- **System:** NVIDIA DGX Spark
- **GPU:** NVIDIA GB10, compute capability 12.1 (sm_121), 128 GB unified memory
- **Container:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **torch:** 2.10.0a0+...nv25.12 (custom NGC build)
- **transformers:** 5.8.1

## What this measures and what it does not

**Measured:** single-stream throughput on two fixed prompts with greedy decoding, and quantisation perturbation patterns under simulated cast-and-dequantise int8.

**Not measured:** output quality on downstream tasks (validated in [orthrus-serve](../orthrus-serve) via tool-eval-bench), latency under concurrent load, batch throughput, sampling strategies, real production quant kernels (also lives in orthrus-serve).

The baseline is stock Qwen3-8B in autoregressive mode with a KV cache, not Orthrus's own `use_diffusion_mode=False` path. The latter is available via `--include-nodiff` for transparency but is not a clean AR comparison (Orthrus was trained with bidirectional attention, so its AR-mode path has no KV cache).

## Related work

- **[orthrus-serve](../orthrus-serve)**: production OpenAI-compatible server with fp8 / NVFP4 quantisation, end-to-end tool-eval-bench validation, and the empirical answers to the open questions this bench raised.
- **Orthrus paper:** [arxiv.org/abs/2605.12825](https://arxiv.org/abs/2605.12825) / [chiennv2000/orthrus](https://github.com/chiennv2000/orthrus)
- **Qwen3-8B model card:** [huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **DGX Spark guides:** [martimramos/dgx-spark-ml-guide](https://github.com/martimramos/dgx-spark-ml-guide), [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup)

## Limitations

- sm_121 (GB10) on aarch64; will not run as-is on x86_64 or other Blackwell variants without container adjustments.
- flash-attn 2 only (FA3/FA4 do not yet support sm_121).
- Single GPU, 8B model, no quantisation in the throughput benchmark itself.

## License

Apache-2.0. See [LICENSE](LICENSE).
