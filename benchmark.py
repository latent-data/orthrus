"""
Benchmark Orthrus-Qwen3-8B against a fair baseline.

Compares three settings on each prompt:
  1. Orthrus, diffusion mode      (use_diffusion_mode=True)
  2. Orthrus, "AR" mode           (use_diffusion_mode=False)
     - not a true AR path, included for transparency
  3. Stock Qwen/Qwen3-8B          (proper AR with KV cache, FA2)
     - this is the real baseline worth quoting

Each model is loaded once and reused across all selected prompts to avoid
redundant weight loading. Models are freed between Orthrus and Qwen3.

Note: stock Qwen3-8B is ~16 GB and will download on first run.
"""
import argparse
import datetime
import gc
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ORTHRUS_ID = "chiennv/Orthrus-Qwen3-8B"
QWEN_ID = "Qwen/Qwen3-8B"

# Pinned revisions for reproducibility; override with --orthrus-revision / --qwen-revision
ORTHRUS_REVISION = "34429bd987c2750bed61d65583c6879964367059"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

PROMPTS = {
    "short": (
        "Write a program to count the frequency of each word in a paragraph."
    ),
    "long": (
        "Implement a Python class BoundedPriorityQueue backed by a binary heap. "
        "The class takes a capacity (int) and max_heap (bool, default False) at "
        "construction. Implement: push(item, priority) which adds the item and raises "
        "RuntimeError if at capacity; pop() which removes and returns the highest-priority "
        "item, raising IndexError if empty; peek() which returns the best item without "
        "removing it, raising IndexError if empty; __len__; __bool__; __iter__ yielding "
        "(item, priority) pairs in priority order without mutating the queue; and a "
        "classmethod from_items(items, capacity, max_heap=False) accepting an iterable "
        "of (item, priority) pairs. Use full type annotations throughout. Then write a "
        "complete unittest.TestCase covering: push/pop round-trip, capacity enforcement, "
        "min-heap and max-heap ordering, peek, iteration order, from_items bulk loading, "
        "empty-queue edge cases, and duplicate priorities."
    ),
}

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_WARMUP_TOKENS = 32
DEFAULT_OUTPUT = "results/results.json"


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Orthrus vs Qwen3-8B AR")
    p.add_argument(
        "--prompts", action="append", metavar="NAME", default=None,
        help=f"Named prompt to run (default: all). May be repeated. "
             f"Valid names: {', '.join(PROMPTS)}",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup-tokens", type=int, default=DEFAULT_WARMUP_TOKENS)
    p.add_argument("--runs", type=int, default=1, metavar="N",
                   help="Number of timed runs per config; results are averaged (default: 1)")
    p.add_argument("--include-nodiff", action="store_true", default=False,
                   help="Also benchmark Orthrus with use_diffusion_mode=False. "
                        "WARNING: this mode has no KV cache and bidirectional attention, "
                        "making it O(n^2) in output length. Expect 10+ minutes per run "
                        "on the long prompt. It is not a valid AR baseline and is included "
                        "only for transparency.")
    p.add_argument("--seed", type=int, default=None, metavar="S",
                   help="RNG seed set before each run (optional; output is already "
                        "deterministic with do_sample=False)")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--orthrus-revision", default=ORTHRUS_REVISION)
    p.add_argument("--qwen-revision", default=QWEN_REVISION)
    args = p.parse_args()
    if args.runs < 1:
        p.error("--runs must be at least 1")

    selected = args.prompts if args.prompts is not None else list(PROMPTS)
    unknown = [n for n in selected if n not in PROMPTS]
    if unknown:
        p.error(f"Unknown prompt name(s): {', '.join(unknown)}. "
                f"Valid names: {', '.join(PROMPTS)}")
    args.prompts = selected
    return args


def load(model_id, revision, trust_remote_code=False):
    print(f"\nLoading {model_id} (revision={revision}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer


def build_inputs(tokenizer, prompt, device):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).input_ids
    return input_ids.to(device)


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens, **extra):
    return model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        **extra,
    )


def _set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def benchmark(model, tokenizer, input_ids, label, warmup_tokens, max_new_tokens,
              num_runs=1, seed=None, **extra):
    print(f"\n=== {label} ===")

    _set_seed(seed)
    print(f"  warmup ({warmup_tokens} tokens) ...  "
          "[first call compiles CUDA kernels, may take several minutes]")
    _ = generate(model, input_ids, warmup_tokens, **extra)
    torch.cuda.synchronize()
    print("  warmup done")

    elapsed_list = []
    for i in range(num_runs):
        _set_seed(seed)
        run_label = f"run {i + 1}/{num_runs}" if num_runs > 1 else "measuring"
        print(f"  {run_label} (up to {max_new_tokens} tokens) ...")
        torch.cuda.synchronize()
        start = time.perf_counter()
        output_ids = generate(model, input_ids, max_new_tokens, **extra)
        torch.cuda.synchronize()
        elapsed_list.append(time.perf_counter() - start)

    new_tokens = output_ids.shape[-1] - input_ids.shape[-1]
    tps_list = [new_tokens / e for e in elapsed_list]
    elapsed_mean = sum(elapsed_list) / len(elapsed_list)
    tps_mean = new_tokens / elapsed_mean

    print(f"  tokens:     {new_tokens}")
    if num_runs > 1:
        run_summary = "  ".join(f"{e:.2f}s/{t:.1f}tok/s" for e, t in zip(elapsed_list, tps_list))
        print(f"  runs:       {run_summary}")
    print(f"  elapsed:    {elapsed_mean:.2f} s{'  (mean)' if num_runs > 1 else ''}")
    print(f"  throughput: {tps_mean:.1f} tok/s{'  (mean)' if num_runs > 1 else ''}")

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
    )
    snippet = response[:300].replace("\n", " ")
    print(f"  output[0:300]: {snippet!r}")

    result = {"tokens": new_tokens, "elapsed": round(elapsed_mean, 3), "tps": round(tps_mean, 2),
              "snippet": snippet}
    if num_runs > 1:
        result["elapsed_runs"] = [round(e, 3) for e in elapsed_list]
        result["tps_runs"] = [round(t, 2) for t in tps_list]
    return result


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _flash_attn_version():
    try:
        import flash_attn
        return flash_attn.__version__
    except ImportError:
        return None


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this benchmark assumes a GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)}")

    # results[prompt_name][config_key] = benchmark result dict
    results = {p: {} for p in args.prompts}

    common = dict(warmup_tokens=args.warmup_tokens, max_new_tokens=args.max_new_tokens,
                  num_runs=args.runs, seed=args.seed, do_sample=False)

    # ---- Orthrus: load once, run all prompts, then free
    model, tokenizer = load(ORTHRUS_ID, args.orthrus_revision, trust_remote_code=True)
    for prompt_name in args.prompts:
        input_ids = build_inputs(tokenizer, PROMPTS[prompt_name], model.device)
        results[prompt_name]["orthrus_diffusion"] = benchmark(
            model, tokenizer, input_ids,
            f"Orthrus diffusion mode [{prompt_name}]",
            **common, use_diffusion_mode=True,
        )
        if args.include_nodiff:
            results[prompt_name]["orthrus_nodiff"] = benchmark(
                model, tokenizer, input_ids,
                f"Orthrus use_diffusion_mode=False [{prompt_name}]",
                **common, use_diffusion_mode=False,
            )
    free_model(model)
    del tokenizer

    # ---- Qwen3-8B: load once, run all prompts, then free
    model, tokenizer = load(QWEN_ID, args.qwen_revision, trust_remote_code=False)
    for prompt_name in args.prompts:
        input_ids = build_inputs(tokenizer, PROMPTS[prompt_name], model.device)
        results[prompt_name]["qwen3_8b_ar"] = benchmark(
            model, tokenizer, input_ids,
            f"Stock Qwen3-8B, standard AR with KV cache [{prompt_name}]",
            **common,
        )
    free_model(model)
    del tokenizer

    # ---- Summary table
    print("\n=== Summary ===")
    label_map = {"orthrus_diffusion": "Orthrus diffusion", "qwen3_8b_ar": "Qwen3-8B AR (KV cache)"}
    if args.include_nodiff:
        label_map["orthrus_nodiff"] = "Orthrus use_diffusion_mode=False"
    prompt_w = max(len(p) for p in args.prompts) + 2
    label_w = max(len(v) for v in label_map.values()) + 2
    header = (f"  {'prompt':<{prompt_w}} {'config':<{label_w}}"
              f" {'tokens':>8}  {'elapsed':>10}  {'throughput':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prompt_name in args.prompts:
        for key, r in results[prompt_name].items():
            print(
                f"  {prompt_name:<{prompt_w}} {label_map[key]:<{label_w}}"
                f" {r['tokens']:>8}  {r['elapsed']:>8.2f} s  {r['tps']:>8.1f} tok/s"
            )

    # ---- Speedups (geometric mean across prompts)
    speedups = {}
    for prompt_name in args.prompts:
        baseline = results[prompt_name]["qwen3_8b_ar"]["tps"]
        diff_tps = results[prompt_name]["orthrus_diffusion"]["tps"]
        speedups[prompt_name] = diff_tps / baseline

    geomean = math.exp(sum(math.log(s) for s in speedups.values()) / len(speedups))

    print()
    for prompt_name, s in speedups.items():
        print(f"  Orthrus diffusion vs Qwen3-8B AR [{prompt_name}]: {s:.2f}x")
    if len(speedups) > 1:
        print(f"  Geometric mean speedup: {geomean:.2f}x")
    print("  (diffusion vs stock Qwen3-8B AR is the comparison worth quoting)")

    # ---- JSON output
    import accelerate
    import transformers

    cap = torch.cuda.get_device_capability(0)
    mem_bytes = torch.cuda.get_device_properties(0).total_memory

    output = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(cap),
            "total_memory_gb": round(mem_bytes / 1024**3, 1),
        },
        "container": {
            "image_tag": os.environ.get("BENCHMARK_IMAGE_TAG", "unknown"),
            "image_name": os.environ.get("BENCHMARK_IMAGE_NAME", "unknown"),
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "flash_attn": _flash_attn_version(),
        },
        "config": {
            "prompts": args.prompts,
            "include_nodiff": args.include_nodiff,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "runs": args.runs,
            "seed": args.seed,
            "do_sample": False,
            "orthrus_revision": args.orthrus_revision,
            "qwen_revision": args.qwen_revision,
        },
        "results": results,
        "speedups": {
            **{p: round(s, 2) for p, s in speedups.items()},
            "geomean": round(geomean, 2),
        },
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
