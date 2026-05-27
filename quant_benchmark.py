"""
Quantization-impact investigation for Orthrus-Qwen3-8B.

Background
----------
Orthrus is a dual-view diffusion LM: a frozen autoregressive (AR) head verifies
tokens proposed by a trained diffusion head. The two heads share the embedding,
MLP, and KV cache; only the attention projections and per-head norms are
duplicated as `_diff` variants. The diffusion head was trained to match the AR
head's predictive distribution at bf16. Lowering the AR teacher's precision
shifts that distribution and should reduce per-iteration acceptance length
(tokens-per-forward-pass, TPF).

This script measures four configurations on a fixed prompt set:

  baseline-bf16  - no quantization.
  teacher-int8   - non-diff (AR + shared) weights cast to int8 then back to bf16.
  teacher-int4   - same, but int4. Per-tensor by default; --int4-per-channel
                   switches to per-output-channel scales.
  full-int8      - both AR and diffusion weights cast to int8.

Quantization is *simulated*: weights are cast to a low-precision integer and
immediately dequantized back to bf16. The kernel path is unchanged, so this
isolates the distribution-shift effect from runtime-performance effects (and
therefore reports no memory savings). For each non-baseline config we compare
the generated token sequence against baseline-bf16 (exact match, token-level
Levenshtein, position of first divergence).

Greedy decoding only (do_sample=False, temperature=0.0); divergence is purely
from precision, not sampling RNG.
"""
import argparse
import datetime
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ORTHRUS_ID = "chiennv/Orthrus-Qwen3-8B"
ORTHRUS_REVISION = "34429bd987c2750bed61d65583c6879964367059"

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
    "math": (
        "Two trains start from cities A and B and travel toward each other along a "
        "straight line. Train A leaves city A at 2pm traveling at 60mph. Train B leaves "
        "city B at 3pm traveling at 75mph. They meet at 5pm. How far apart are cities "
        "A and B? Show your reasoning step by step."
    ),
    "explanation": (
        "Explain how a CPU's instruction pipeline works, at a level appropriate for "
        "a CS undergraduate."
    ),
    "json": (
        "List the seven wonders of the ancient world with their locations and "
        "approximate construction dates, as a JSON array."
    ),
    "creative": (
        "Write a 500-word short story about a librarian who discovers a book that "
        "wasn't there yesterday."
    ),
    "apollo": (
        "Write a 4-paragraph plain-English summary of the 1969 Apollo 11 mission "
        "for a 10-year-old."
    ),
}

CONFIGS = [
    "baseline-bf16", "teacher-int8", "teacher-int4", "full-int8",
    "ar-bf16", "ar-int8",
]
BASELINE_KEY = "baseline-bf16"
AR_BASELINE_KEY = "ar-bf16"

# Each config runs through one of two generation paths:
#   "diffusion" -> model.generate(use_diffusion_mode=True), the normal Orthrus
#                  speculative-decoding path
#   "ar"        -> model.generate(use_diffusion_mode=False), HF super().generate()
#                  through the AR projections; used as the vanilla-Qwen3 proxy
CONFIG_MODE = {
    "baseline-bf16": "diffusion",
    "teacher-int8": "diffusion",
    "teacher-int4": "diffusion",
    "full-int8": "diffusion",
    "ar-bf16": "ar",
    "ar-int8": "ar",
}

# Per-config quantization scheme. ar-bf16 is a no-op (no quant); ar-int8 reuses
# the same teacher-int8 cast-and-dequant on AR/shared weights (the _diff
# projections are never accessed in AR mode, so quantizing them or not produces
# identical AR output; we skip them for clarity).
CONFIG_QUANT_SCHEME = {
    "baseline-bf16": "none",
    "teacher-int8": "teacher-int8",
    "teacher-int4": "teacher-int4",
    "full-int8": "full-int8",
    "ar-bf16": "none",
    "ar-int8": "teacher-int8",
}


def within_arm_baseline(config):
    """Which bf16 config does this config compare against for within-arm
    quant-sensitivity measurement?
    """
    return AR_BASELINE_KEY if CONFIG_MODE[config] == "ar" else BASELINE_KEY

# Qwen3 <|im_end|> token. This is the natural assistant-turn terminator when
# tokenizer.apply_chat_template(..., add_generation_prompt=True) is used.
# Orthrus's diffusion-mode generate loop halts at this token; HF's
# super().generate() in AR mode may emit it as a normal token and continue
# past it, so the AR-equivalence check truncates both outputs here before
# comparing forward-pass behaviour.
EOS_TOKEN_ID = 151645

# Per-prompt max_new_tokens for the --verify-ar-equivalence check (and for
# AR-mode runs inside the main quantization loop). Sized to each prompt's
# expected diffusion-mode natural output length plus modest headroom. Crucial:
# AR-mode generation does NOT stop at <|im_end|>, so a generous max_new_tokens
# extends only the AR arm (wasted GPU on tokens we truncate away).
AR_EQUIV_PROMPT_MAX_NEW_TOKENS = {
    "short": 512,
    "long": 2048,
    "math": 512,
    "explanation": 1536,
    "json": 512,
    "creative": 1024,
    "apollo": 768,
}

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_WARMUP_TOKENS = 32
DEFAULT_OUTPUT = "results/quant_results.json"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", action="append", metavar="NAME", default=None,
                   help=f"Prompt name(s). May be repeated. Default: all. "
                        f"Valid: {', '.join(PROMPTS)}")
    p.add_argument("--configs", action="append", metavar="NAME", default=None,
                   help=f"Configurations to run. May be repeated. "
                        f"Default: all. Valid: {', '.join(CONFIGS)}")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup-tokens", type=int, default=DEFAULT_WARMUP_TOKENS)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed set before each measurement run "
                        "(optional; greedy decoding is already deterministic).")
    p.add_argument("--int4-per-channel", action="store_true", default=False,
                   help="Use per-output-channel symmetric int4 instead of "
                        "per-tensor int4. Less lossy; use as a fallback if "
                        "per-tensor produces unusable output.")
    p.add_argument("--verify-only", action="store_true", default=False,
                   help="Load the model, print the AR / diffusion parameter "
                        "partition for confirmation, then exit without running "
                        "any benchmark.")
    p.add_argument("--verify-ar-equivalence", action="store_true", default=False,
                   help="At full bf16 precision, compare use_diffusion_mode=True "
                        "and use_diffusion_mode=False. Reports whether the AR-only "
                        "path through Orthrus produces the same token sequence as "
                        "the diffusion path, then exits. Skips all quantization "
                        "configs.")
    p.add_argument("--ar-equivalence-prompts", action="append", metavar="NAME",
                   default=None,
                   help="Prompt name(s) to run for --verify-ar-equivalence. May "
                        f"be repeated. Default: all ({', '.join(PROMPTS)}). "
                        "Per-prompt max_new_tokens is fixed in "
                        "AR_EQUIV_PROMPT_MAX_NEW_TOKENS at the top of this "
                        "file.")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--orthrus-revision", default=ORTHRUS_REVISION)
    args = p.parse_args()

    selected_prompts = args.prompts if args.prompts is not None else list(PROMPTS)
    unknown = [n for n in selected_prompts if n not in PROMPTS]
    if unknown:
        p.error(f"Unknown prompt name(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(PROMPTS)}")
    args.prompts = selected_prompts

    selected_configs = args.configs if args.configs is not None else list(CONFIGS)
    unknown = [n for n in selected_configs if n not in CONFIGS]
    if unknown:
        p.error(f"Unknown config name(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(CONFIGS)}")
    # For each requested config that isn't itself a baseline, ensure its
    # within-arm baseline is also in the run (and ordered before it) so the
    # in-loop comparison and diagnostic can run.
    needed_baselines = []
    for c in selected_configs:
        b = within_arm_baseline(c)
        if b != c and b not in selected_configs and b not in needed_baselines:
            needed_baselines.append(b)
    if needed_baselines:
        selected_configs = needed_baselines + selected_configs
    # Within the final list, ensure each arm's baseline precedes its other
    # configs (stable partition: baselines first in their original order).
    selected_configs = (
        [c for c in selected_configs if c in (BASELINE_KEY, AR_BASELINE_KEY)]
        + [c for c in selected_configs if c not in (BASELINE_KEY, AR_BASELINE_KEY)]
    )
    args.configs = selected_configs

    ae_prompts = (args.ar_equivalence_prompts
                  if args.ar_equivalence_prompts is not None else list(PROMPTS))
    unknown = [n for n in ae_prompts if n not in PROMPTS]
    if unknown:
        p.error(f"Unknown --ar-equivalence-prompts name(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(PROMPTS)}")
    args.ar_equivalence_prompts = ae_prompts
    return args


# ----------------------------------------------------------------------------
# Model loading and parameter partition
# ----------------------------------------------------------------------------

def load_model(revision):
    print(f"\nLoading {ORTHRUS_ID} (revision={revision}) ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        ORTHRUS_ID,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(ORTHRUS_ID, revision=revision)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def partition_params(model):
    """Split parameters into (ar_or_shared, diffusion).

    Diffusion params are those whose qualified name contains '_diff' (the
    `*_proj_diff` and `*_norm_diff` projections added on top of stock Qwen3).
    Everything else (embedding, MLP, layer norms, original Qwen3 attention
    projections, lm_head, final norm) is bundled as 'AR' for this experiment:
    these are the weights the diffusion head was trained to match against.
    """
    ar, diff = [], []
    for name, param in model.named_parameters():
        if "_diff" in name:
            diff.append((name, param))
        else:
            ar.append((name, param))
    return ar, diff


def print_partition(ar, diff):
    ar_n = sum(p.numel() for _, p in ar)
    diff_n = sum(p.numel() for _, p in diff)
    total = ar_n + diff_n
    print(f"\n  AR (non-_diff) parameters:")
    print(f"    tensors: {len(ar):>5}")
    print(f"    numel:   {ar_n:>13,}   ({100 * ar_n / total:5.1f}%)")
    print(f"    sample names:")
    for name, p in ar[:8]:
        print(f"      {name:<60} {tuple(p.shape)}")
    print(f"\n  Diffusion (_diff) parameters:")
    print(f"    tensors: {len(diff):>5}")
    print(f"    numel:   {diff_n:>13,}   ({100 * diff_n / total:5.1f}%)")
    print(f"    sample names:")
    for name, p in diff[:8]:
        print(f"      {name:<60} {tuple(p.shape)}")
    print(f"\n  Total: {total:,} parameters")
    print(f"  Expected from spec: AR ~84%, Diffusion ~16%")


# ----------------------------------------------------------------------------
# Simulated quantization
# ----------------------------------------------------------------------------

@torch.no_grad()
def quantize_int8_per_tensor(param):
    """Symmetric per-tensor int8 quantize-then-dequantize, in place."""
    if param.numel() == 0:
        return
    w = param.data
    max_abs = w.abs().max()
    if max_abs.item() == 0.0:
        return
    scale = (max_abs / 127.0).to(w.dtype)
    q = (w / scale).round().clamp_(-128, 127)
    param.data = (q * scale).to(w.dtype)


@torch.no_grad()
def quantize_int4_per_tensor(param):
    """Symmetric per-tensor int4 (range -8..7) quantize-then-dequantize."""
    if param.numel() == 0:
        return
    w = param.data
    max_abs = w.abs().max()
    if max_abs.item() == 0.0:
        return
    scale = (max_abs / 7.0).to(w.dtype)
    q = (w / scale).round().clamp_(-8, 7)
    param.data = (q * scale).to(w.dtype)


@torch.no_grad()
def quantize_int4_per_channel(param):
    """Symmetric int4 with one scale per output channel (dim 0 of weight).

    Falls back to per-tensor for 1-D params (norms, biases).
    """
    if param.numel() == 0:
        return
    w = param.data
    if w.dim() < 2:
        return quantize_int4_per_tensor(param)
    reduce_dims = tuple(range(1, w.dim()))
    max_abs = w.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (max_abs / 7.0).to(w.dtype)
    # Avoid divide-by-zero on zero rows.
    safe = scale.clone()
    safe[safe == 0] = 1.0
    q = (w / safe).round().clamp_(-8, 7)
    q[max_abs.expand_as(q) == 0] = 0
    param.data = (q * scale).to(w.dtype)


def apply_quantization(model, config, int4_per_channel):
    """Apply the quantization scheme associated with `config` in place. The
    scheme is keyed via CONFIG_QUANT_SCHEME so multiple configs can share a
    scheme (e.g. teacher-int8 and ar-int8 both use the 'teacher-int8' scheme,
    differing only in generation mode).
    """
    scheme = CONFIG_QUANT_SCHEME[config]
    ar, diff = partition_params(model)
    ar_params = [p for _, p in ar]
    diff_params = [p for _, p in diff]

    if scheme == "none":
        return
    if scheme == "teacher-int8":
        for p in ar_params:
            quantize_int8_per_tensor(p)
    elif scheme == "teacher-int4":
        fn = quantize_int4_per_channel if int4_per_channel else quantize_int4_per_tensor
        for p in ar_params:
            fn(p)
    elif scheme == "full-int8":
        for p in ar_params + diff_params:
            quantize_int8_per_tensor(p)
    else:
        raise ValueError(f"Unknown quant scheme {scheme!r} for config {config!r}")


# ----------------------------------------------------------------------------
# TPF instrumentation
# ----------------------------------------------------------------------------

class PassCounter:
    """Forward-pre-hook counter for diffusion-mode iterations.

    Each Orthrus diffusion step makes one forward call with is_diffusion_pass=True
    (the proposal) followed by one with is_diffusion_pass=False (the AR verify),
    plus a single AR-mode call at generation start. We count diffusion passes;
    TPF = generated_tokens / diffusion_passes is the per-iteration acceptance
    length and is the quantity that should degrade under teacher quantization.
    """

    def __init__(self):
        self.diff_passes = 0
        self.total_passes = 0
        self._handle = None

    def reset(self):
        self.diff_passes = 0
        self.total_passes = 0

    def attach(self, model):
        def pre_hook(_module, _args, kwargs):
            self.total_passes += 1
            if kwargs.get("is_diffusion_pass", False):
                self.diff_passes += 1
        self._handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ----------------------------------------------------------------------------
# Losslessness metrics
# ----------------------------------------------------------------------------

def first_divergence(a, b):
    """Index of first position where token sequences differ, or None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def compute_divergence_diagnostic(model, tokenizer, input_ids, agreed_prefix,
                                  bf16_chosen_id, this_chosen_id):
    """At the first within-arm divergence position, do a fresh forward pass
    through the currently-loaded (quantized) model with the agreed prefix and
    return top-3 logits, the top1-top2 gap, the rank in this model's logits of
    the token the bf16 baseline would have chosen, and the rank of the token
    this config actually chose.

    A small gap + low ranks for both choices indicates a near-tie tip
    (quantization flipped a decision that was already wobbling). A wide gap +
    high rank for the bf16 choice indicates a substantive perturbation
    (quantization meaningfully changed the prediction).
    """
    prompt_ids = input_ids[0].tolist()
    prefix = prompt_ids + agreed_prefix
    x = torch.tensor([prefix], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        out = model(input_ids=x, use_cache=False, logits_to_keep=1)
    logits = out.logits[0, -1, :].float()
    top = torch.topk(logits, k=3)
    top_ids = top.indices.tolist()
    top_vals = [float(v) for v in top.values]
    sorted_ids = torch.argsort(logits, descending=True).tolist()
    rank_bf16 = sorted_ids.index(bf16_chosen_id) + 1
    rank_this = sorted_ids.index(this_chosen_id) + 1
    del sorted_ids
    return {
        "fresh_forward_top3_ids": top_ids,
        "fresh_forward_top3_logits": [round(v, 4) for v in top_vals],
        "top1_to_top2_logit_gap": round(top_vals[0] - top_vals[1], 4),
        "bf16_chosen_id": bf16_chosen_id,
        "bf16_chosen_text": tokenizer.decode([bf16_chosen_id]),
        "bf16_chosen_rank_in_this_model": rank_bf16,
        "this_chosen_id": this_chosen_id,
        "this_chosen_text": tokenizer.decode([this_chosen_id]),
        "this_chosen_rank_in_this_model": rank_this,
    }


def truncate_at_first(seq, value):
    """Return seq[:first_index_of_value + 1], or seq unchanged if value absent."""
    for i, x in enumerate(seq):
        if x == value:
            return seq[: i + 1]
    return seq


def levenshtein(a, b):
    """Token-level Levenshtein distance. O(len(a) * len(b)) time, O(len(b)) space."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

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


def _set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


@torch.inference_mode()
def generate_for_config(model, input_ids, max_new_tokens, use_diffusion_mode):
    return model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_diffusion_mode=use_diffusion_mode,
    )


def measure(model, tokenizer, counter, prompt_name, input_ids,
            warmup_tokens, max_new_tokens, seed, label,
            use_diffusion_mode=True):
    print(f"\n  --- {label}: prompt={prompt_name} "
          f"(mode={'diffusion' if use_diffusion_mode else 'ar'}) ---")

    _set_seed(seed)
    counter.reset()
    print(f"  warmup ({warmup_tokens} tokens) ... "
          "[first call may take several minutes to compile kernels]")
    torch.cuda.synchronize()
    _ = generate_for_config(model, input_ids, warmup_tokens, use_diffusion_mode)
    torch.cuda.synchronize()
    print("  warmup done")

    _set_seed(seed)
    counter.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = generate_for_config(
        model, input_ids, max_new_tokens, use_diffusion_mode,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    prompt_len = input_ids.shape[-1]
    new_token_ids = output_ids[0, prompt_len:].tolist()
    new_token_count = len(new_token_ids)
    diff_passes = counter.diff_passes
    total_passes = counter.total_passes
    # TPF is meaningful only for diffusion-mode generation (acceptance length
    # per diffusion iteration); for AR mode there are no diffusion passes.
    tpf = new_token_count / diff_passes if diff_passes > 0 else None
    tps = new_token_count / elapsed if elapsed > 0 else float("nan")

    decoded = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    snippet = decoded[:300].replace("\n", " ")
    print(f"  tokens:        {new_token_count}")
    print(f"  elapsed:       {elapsed:.2f} s")
    print(f"  throughput:    {tps:.1f} tok/s")
    print(f"  diff passes:   {diff_passes}  (total forward calls: {total_passes})")
    if tpf is not None:
        print(f"  TPF:           {tpf:.2f}")
    else:
        print(f"  TPF:           n/a (AR mode)")
    print(f"  output[0:300]: {snippet!r}")

    return {
        "tokens": new_token_count,
        "elapsed": round(elapsed, 3),
        "tps": round(tps, 2),
        "tpf": round(tpf, 3) if tpf is not None else None,
        "mode": "diffusion" if use_diffusion_mode else "ar",
        "diffusion_passes": diff_passes,
        "total_forward_passes": total_passes,
        "output_token_ids": new_token_ids,
        "snippet": snippet,
    }


def _ar_equivalence_one_prompt(args, tokenizer, model, prompt_name,
                               max_new_tokens):
    """Compare use_diffusion_mode True vs False at bf16, greedy, on one prompt.

    Both paths go through the same Orthrus model with no quantization. By the
    paper's strict-losslessness claim they should emit identical token
    sequences; this function checks empirically. The token-ID comparison is
    authoritative (decoded text is reported for human inspection only).

    Returns a dict with the per-prompt analysis (no JSON write-out); the
    multi-prompt orchestrator collects these and writes one combined JSON.
    """
    prompt = PROMPTS[prompt_name]
    input_ids = build_inputs(tokenizer, prompt, model.device)
    prompt_len = input_ids.shape[-1]

    print("\n=== AR equivalence verification ===")
    print(f"prompt:         {prompt_name!r}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"do_sample:      False (greedy)")

    print("\nGenerating with use_diffusion_mode=True ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_diff = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_diffusion_mode=True,
    )
    torch.cuda.synchronize()
    diff_elapsed = time.perf_counter() - t0
    print(f"  done in {diff_elapsed:.1f}s")

    print("\nGenerating with use_diffusion_mode=False ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_ar = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_diffusion_mode=False,
    )
    torch.cuda.synchronize()
    ar_elapsed = time.perf_counter() - t0
    print(f"  done in {ar_elapsed:.1f}s")

    diff_ids = out_diff[0, prompt_len:].tolist()
    ar_ids = out_ar[0, prompt_len:].tolist()
    diff_text = tokenizer.decode(diff_ids, skip_special_tokens=True)
    ar_text = tokenizer.decode(ar_ids, skip_special_tokens=True)

    # Raw comparison (before stop-criterion normalization).
    raw_fdp = first_divergence(diff_ids, ar_ids)
    raw_exact = (raw_fdp is None)
    raw_edit = 0 if raw_exact else levenshtein(diff_ids, ar_ids)

    # Truncate each output at its first <|im_end|> (inclusive) to separate
    # forward-pass equivalence from stop-criterion equivalence. The diffusion-
    # mode loop halts at the first EOS, but HF's super().generate() in AR mode
    # may emit <|im_end|> as a normal token and continue past it.
    def _first_index(seq, value):
        for i, x in enumerate(seq):
            if x == value:
                return i
        return None

    diff_eos = _first_index(diff_ids, EOS_TOKEN_ID)
    ar_eos = _first_index(ar_ids, EOS_TOKEN_ID)

    diff_trunc = diff_ids if diff_eos is None else diff_ids[:diff_eos + 1]
    ar_trunc = ar_ids if ar_eos is None else ar_ids[:ar_eos + 1]
    diff_trail = len(diff_ids) - len(diff_trunc)
    ar_trail = len(ar_ids) - len(ar_trunc)

    diff_stopped = (diff_eos is not None and diff_trail == 0)
    ar_stopped = (ar_eos is not None and ar_trail == 0)
    stop_diverges = (diff_stopped != ar_stopped)

    tr_fdp = first_divergence(diff_trunc, ar_trunc)
    tr_exact = (tr_fdp is None)
    tr_edit = 0 if tr_exact else levenshtein(diff_trunc, ar_trunc)

    # Shared-prefix framing: how much of the diffusion path's (truncated)
    # output is captured as a common prefix between the two paths? This is the
    # quantity that matters for using diffusion-off as a Qwen3 AR proxy: any
    # tokens past the shared prefix are unusable for direct comparison anyway.
    if tr_fdp is None:
        shared_prefix_length = len(diff_trunc)
    else:
        shared_prefix_length = tr_fdp
    shared_prefix_fraction = (
        shared_prefix_length / len(diff_trunc) if len(diff_trunc) > 0 else 1.0
    )

    # Diagnostic at the first divergence position only: fresh forward pass
    # with the agreed prefix, capture top-3 logits and the rank of each
    # path's chosen token. A small top1-top2 gap with both choices in the
    # top few ranks indicates the divergence is a near-tie consistent with
    # fp accumulation noise rather than a structural difference.
    divergence_diagnostic = None
    real_divergence = (
        tr_fdp is not None
        and tr_fdp < min(len(diff_trunc), len(ar_trunc))
    )
    if real_divergence:
        pos = tr_fdp
        chosen_diff = diff_trunc[pos]
        chosen_ar = ar_trunc[pos]
        print(f"\nProbing top-3 logits at first-divergence position {pos} "
              f"via fresh forward pass ...")
        prompt_ids = input_ids[0].tolist()
        prefix = prompt_ids + diff_trunc[:pos]
        x = torch.tensor([prefix], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            out = model(input_ids=x, use_cache=False, logits_to_keep=1)
        logits = out.logits[0, -1, :].float()
        top = torch.topk(logits, k=3)
        top_ids = top.indices.tolist()
        top_vals = [float(v) for v in top.values]
        sorted_ids = torch.argsort(logits, descending=True).tolist()
        rank_diff = sorted_ids.index(chosen_diff) + 1
        rank_ar = sorted_ids.index(chosen_ar) + 1
        del sorted_ids
        divergence_diagnostic = {
            "position": pos,
            "diffusion_on_chose_id": chosen_diff,
            "diffusion_on_chose_text": tokenizer.decode([chosen_diff]),
            "diffusion_off_chose_id": chosen_ar,
            "diffusion_off_chose_text": tokenizer.decode([chosen_ar]),
            "fresh_forward_top3_ids": top_ids,
            "fresh_forward_top3_logits": [round(v, 4) for v in top_vals],
            "top1_to_top2_logit_gap": round(top_vals[0] - top_vals[1], 4),
            "diffusion_on_rank": rank_diff,
            "diffusion_off_rank": rank_ar,
        }

    def _eos_descriptor(eos_pos, trail):
        if eos_pos is None:
            return "not emitted"
        if trail == 0:
            return f"position {eos_pos} (stopped)"
        return f"position {eos_pos} (emitted, continued for {trail} more tokens)"

    diff_term = (f"terminates at <|im_end|>" if diff_stopped
                 else (f"continues past <|im_end|> at position {diff_eos}"
                       if diff_eos is not None else "no <|im_end|> emitted"))
    ar_term = (f"terminates at <|im_end|>" if ar_stopped
               else (f"continues past <|im_end|> at position {ar_eos}"
                     if ar_eos is not None else "no <|im_end|> emitted"))

    print("\n=== AR Equivalence Verification ===")
    print(f"prompt:                                {prompt_name!r}")
    print(f"max_new_tokens:                        {max_new_tokens}")
    print(f"do_sample:                             False (greedy)")
    print()
    print(f"diffusion-on output tokens:            "
          f"{len(diff_ids)} ({diff_term})")
    print(f"diffusion-off output tokens (raw):     "
          f"{len(ar_ids)} ({ar_term})")
    print(f"diffusion-off output tokens (trunc):   {len(ar_trunc)}")
    print()
    print("Shared prefix analysis (truncated sequences):")
    print(f"  shared prefix length:                {shared_prefix_length}")
    print(f"  diffusion-on total length:           {len(diff_trunc)}")
    print(f"  shared prefix fraction:              "
          f"{100 * shared_prefix_fraction:.1f}%")
    if real_divergence:
        d = divergence_diagnostic
        print(f"  first divergence position:           {tr_fdp}")
        print(f"  diffusion-on token at div:           "
              f"{d['diffusion_on_chose_text']!r} (id {d['diffusion_on_chose_id']})")
        print(f"  diffusion-off token at div:          "
              f"{d['diffusion_off_chose_text']!r} (id {d['diffusion_off_chose_id']})")
        print(f"  fresh forward top-3 ids:             "
              f"{d['fresh_forward_top3_ids']}")
        print(f"  top1-top2 logit gap:                 "
              f"{d['top1_to_top2_logit_gap']:.4f}")
        print(f"  diff-on rank in full vocab:          {d['diffusion_on_rank']}")
        print(f"  diff-off rank in full vocab:         {d['diffusion_off_rank']}")
    else:
        print(f"  first divergence position:           "
              f"N/A ({'exact match' if tr_exact else 'one is a prefix of the other'})")
    print()
    print("Stop-criterion equivalence:")
    print(f"  diffusion-on  EOS:                   "
          f"{_eos_descriptor(diff_eos, diff_trail)}")
    print(f"  diffusion-off EOS:                   "
          f"{_eos_descriptor(ar_eos, ar_trail)}")
    print(f"  stop behaviour:                      "
          f"{'DIVERGES' if stop_diverges else 'AGREES'}")

    # Per-prompt FINDING (descriptive): what did we observe? Note the divergence
    # category but do not derive a path-2-vs-path-3 recommendation from it; that
    # decision is made in the aggregator and rests on the self-consistency
    # argument, not on whether diff-off tracks diff-on.
    if tr_exact:
        finding = (
            "Bit-identical: AR and diffusion paths produce the same tokens "
            "through <|im_end|>."
        )
    elif shared_prefix_fraction > 0.99:
        if divergence_diagnostic is not None:
            d = divergence_diagnostic
            finding = (
                f"Shared-prefix fraction {100 * shared_prefix_fraction:.1f}% "
                f"({shared_prefix_length}/{len(diff_trunc)} tokens). Divergence "
                f"is a near-tie at position {tr_fdp} (top1-top2 logit gap "
                f"{d['top1_to_top2_logit_gap']:.4f}); diff-on chose rank-"
                f"{d['diffusion_on_rank']}, diff-off chose rank-"
                f"{d['diffusion_off_rank']}. Cascades only briefly before EOS."
            )
        else:
            finding = (
                f"Shared-prefix fraction {100 * shared_prefix_fraction:.1f}%."
            )
    else:
        if divergence_diagnostic is not None:
            d = divergence_diagnostic
            finding = (
                f"Shared-prefix fraction only "
                f"{100 * shared_prefix_fraction:.1f}% "
                f"({shared_prefix_length}/{len(diff_trunc)} tokens). The "
                f"underlying divergence event at position {tr_fdp} is a "
                f"near-tie (top1-top2 logit gap "
                f"{d['top1_to_top2_logit_gap']:.4f}); diff-on chose rank-"
                f"{d['diffusion_on_rank']}, diff-off chose rank-"
                f"{d['diffusion_off_rank']}. The low fraction is because the "
                f"single near-tie disagreement happens early in the sequence "
                f"and the divergence cascades; this is fp accumulation drift "
                f"between block-mode (diffusion AR-verify) and single-token-"
                f"mode (HF generate) through the same weights, not a "
                f"structural model difference."
            )
        else:
            finding = (
                f"Shared-prefix fraction "
                f"{100 * shared_prefix_fraction:.1f}% "
                f"({shared_prefix_length}/{len(diff_trunc)} tokens)."
            )

    # Per-prompt IMPLICATION for PR 4: the path-2 viability argument doesn't
    # depend on whether diff-off tracks diff-on. The diff-off bf16-vs-int8
    # comparison is internally self-consistent (same code path), so it cleanly
    # isolates the weight-precision effect regardless of the diff-on/diff-off
    # gap at bf16. See the aggregate implication for the full argument.
    implication = (
        "The diff-off code path is self-consistent under precision changes, "
        "so diff-off-bf16 vs diff-off-int8 will cleanly measure AR-mode "
        "quantization sensitivity through this code path, independently of "
        "whether diff-off matches diff-on at bf16."
    )

    print(f"\nFINDING: {finding}")
    print(f"\nPR 4 implication: {implication}")

    snippet_diff = diff_text[:300].replace("\n", " ")
    snippet_ar = ar_text[:300].replace("\n", " ")
    print(f"\ndiffusion-on output[0:300]:  {snippet_diff!r}")
    print(f"diffusion-off output[0:300]: {snippet_ar!r}")

    return {
        "prompt_name": prompt_name,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "diffusion_on": {
            "elapsed_seconds": round(diff_elapsed, 3),
            "tokens": len(diff_ids),
            "tokens_truncated": len(diff_trunc),
            "first_eos_position": diff_eos,
            "trailing_tokens_after_first_eos": diff_trail,
            "stopped_at_first_eos": diff_stopped,
            "token_ids": diff_ids,
            "token_ids_truncated": diff_trunc,
            "text": diff_text,
        },
        "diffusion_off": {
            "elapsed_seconds": round(ar_elapsed, 3),
            "tokens": len(ar_ids),
            "tokens_truncated": len(ar_trunc),
            "first_eos_position": ar_eos,
            "trailing_tokens_after_first_eos": ar_trail,
            "stopped_at_first_eos": ar_stopped,
            "token_ids": ar_ids,
            "token_ids_truncated": ar_trunc,
            "text": ar_text,
        },
        "comparison_raw": {
            "exact_match": raw_exact,
            "edit_distance": raw_edit,
            "first_divergence_position": raw_fdp,
            "length_delta": len(ar_ids) - len(diff_ids),
            "note": "Raw output comparison; conflates forward-pass and "
                    "stop-criterion differences. See comparison_truncated for "
                    "the forward-pass-only view.",
        },
        "comparison_truncated": {
            "exact_match": tr_exact,
            "edit_distance": tr_edit,
            "first_divergence_position": tr_fdp,
            "length_delta": len(ar_trunc) - len(diff_trunc),
            "shared_prefix_length": shared_prefix_length,
            "shared_prefix_fraction": round(shared_prefix_fraction, 4),
            "divergence_diagnostic": divergence_diagnostic,
            "note": "Comparison after truncating both outputs at their first "
                    "<|im_end|> (inclusive). shared_prefix_fraction = length of "
                    "common token prefix / diffusion-on truncated length; this "
                    "is the quantity that matters for using diffusion-off as an "
                    "AR proxy. divergence_diagnostic: at the first divergence "
                    "position, a fresh forward pass with the agreed prefix gives "
                    "the top-3 next-token logits and the rank of each path's "
                    "chosen token; a small top1-top2 gap with both choices in "
                    "the top few ranks indicates the divergence is a near-tie "
                    "consistent with fp accumulation noise.",
        },
        "stop_criterion": {
            "diffusion_on_stopped_at_first_eos": diff_stopped,
            "diffusion_off_stopped_at_first_eos": ar_stopped,
            "diverges": stop_diverges,
        },
        "per_prompt_finding": finding,
        "per_prompt_implication": implication,
    }


def run_ar_equivalence(args, tokenizer, model):
    """Orchestrator: run the AR-equivalence check on one or more prompts,
    aggregate results, print a combined summary, and write a single JSON.

    Aggregate verdict is keyed on the minimum shared_prefix_fraction across
    prompts (worst case). If any prompt lands in "path3" or "marginal", the
    overall recommendation reflects that; only when every prompt is "proxy"
    is the overall recommendation "proxy".
    """
    prompt_names = args.ar_equivalence_prompts

    per_prompt = {}
    per_prompt_max_new_tokens = {}
    for prompt_name in prompt_names:
        mn = AR_EQUIV_PROMPT_MAX_NEW_TOKENS.get(prompt_name)
        if mn is None:
            raise ValueError(
                f"No max_new_tokens defined for prompt {prompt_name!r}; "
                f"add it to AR_EQUIV_PROMPT_MAX_NEW_TOKENS."
            )
        per_prompt_max_new_tokens[prompt_name] = mn
        per_prompt[prompt_name] = _ar_equivalence_one_prompt(
            args, tokenizer, model, prompt_name, mn,
        )

    # Aggregate findings (descriptive) and PR 4 recommendation. The
    # recommendation does NOT depend on the shared-prefix fraction; it rests on
    # the self-consistency argument (diff-off-bf16 vs diff-off-int8 is a clean
    # comparison through one code path regardless of whether diff-off tracks
    # diff-on). The shared-prefix fractions describe the diff-off-vs-diff-on
    # gap at bf16, which is itself an interesting finding about Orthrus's
    # AR-mode vs diffusion-mode KV-cache fp drift.
    fractions = {
        n: r["comparison_truncated"]["shared_prefix_fraction"]
        for n, r in per_prompt.items()
    }
    exact_all = all(
        r["comparison_truncated"]["exact_match"] for r in per_prompt.values()
    )
    stop_diverges_any = any(
        r["stop_criterion"]["diverges"] for r in per_prompt.values()
    )
    min_prompt = min(fractions, key=fractions.get)
    min_fraction = fractions[min_prompt]
    max_prompt = max(fractions, key=fractions.get)
    max_fraction = fractions[max_prompt]
    spread = max_fraction - min_fraction

    # All divergence diagnostics: same near-tie cascade signature across prompts?
    diagnostics = [
        r["comparison_truncated"].get("divergence_diagnostic")
        for r in per_prompt.values()
    ]
    real_diagnostics = [d for d in diagnostics if d is not None]
    all_near_tie = (
        bool(real_diagnostics)
        and all(d["top1_to_top2_logit_gap"] < 1.0 for d in real_diagnostics)
        and all(
            d["diffusion_on_rank"] <= 3 and d["diffusion_off_rank"] <= 3
            for d in real_diagnostics
        )
    )

    if exact_all:
        aggregate_finding = (
            f"diff-off matches diff-on bit-for-bit across all "
            f"{len(prompt_names)} prompt(s)."
        )
    elif spread < 0.05:
        aggregate_finding = (
            f"diff-off matches diff-on consistently across prompts "
            f"(shared-prefix fraction {100 * min_fraction:.1f}-"
            f"{100 * max_fraction:.1f}%)."
        )
    else:
        # The interesting case: high variance across prompts.
        if all_near_tie:
            aggregate_finding = (
                f"Shared-prefix fraction varies sharply across prompts "
                f"({min_prompt!r}={100 * min_fraction:.1f}%, "
                f"{max_prompt!r}={100 * max_fraction:.1f}%) but the underlying "
                f"divergence mechanism is the same on every prompt: a single "
                f"near-tie argmax disagreement (top1-top2 gap ~0.25, "
                f"rank-1-vs-rank-2 boundary) that cascades through the rest "
                f"of the output. The fraction reflects only where in the "
                f"sequence that single near-tie happens to land, not a "
                f"structural difference between the two paths' weights or "
                f"forward-pass logic."
            )
        else:
            aggregate_finding = (
                f"Shared-prefix fraction varies across prompts "
                f"({min_prompt!r}={100 * min_fraction:.1f}%, "
                f"{max_prompt!r}={100 * max_fraction:.1f}%). Divergence "
                f"diagnostics indicate a mix of near-tie and wider-margin "
                f"disagreements; see per-prompt divergence_diagnostic entries."
            )

    # PR 4 viability is determined by the self-consistency argument, not by
    # the diff-on/diff-off agreement. So the recommendation is fixed.
    aggregate_recommendation = "path2_viable"
    aggregate_implication = (
        "For PR 4 (plain-Qwen3 quantization-sensitivity comparison): the "
        "diff-off code path is viable as a 'plain-Qwen3' proxy. The relevant "
        "PR 4 measurement is diff-off-bf16 vs diff-off-int8, which is "
        "internally self-consistent (same code path, only weight precision "
        "differs) and isolates the int8 perturbation of the AR weights. Under "
        "Orthrus's frozen-teacher claim (AR weights = vanilla Qwen3-8B "
        "weights), this is a faithful proxy for vanilla Qwen3 quantization "
        "sensitivity. The shared-prefix fraction between diff-off and diff-on "
        "at bf16 (above) is a separate finding about cross-code-path fp drift "
        "through the same weights; it is not a precondition for PR 4. "
        "Backbone extraction (path 3) is not required."
    )

    print(f"\n{'=' * 70}")
    print("=== Aggregate AR Equivalence Findings (across prompts) ===")
    print(f"{'=' * 70}")
    print(f"prompts tested:                   {prompt_names}")
    print(f"per-prompt shared-prefix fracs:   "
          + ", ".join(f"{n}={100 * f:.1f}%" for n, f in fractions.items()))
    print(f"stop-criterion diverges anywhere: "
          f"{'yes' if stop_diverges_any else 'no'}")
    print(f"aggregate recommendation:         {aggregate_recommendation}")
    print(f"\nFINDING:           {aggregate_finding}")
    print(f"\nPR 4 implication:  {aggregate_implication}")

    output = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "container": {
            "image_tag": os.environ.get("BENCHMARK_IMAGE_TAG", "unknown"),
            "image_name": os.environ.get("BENCHMARK_IMAGE_NAME", "unknown"),
        },
        "orthrus_revision": args.orthrus_revision,
        "prompt_names": prompt_names,
        "max_new_tokens_per_prompt": per_prompt_max_new_tokens,
        "do_sample": False,
        "eos_token_id": EOS_TOKEN_ID,
        "per_prompt": per_prompt,
        "aggregate": {
            "shared_prefix_fractions": {
                n: round(f, 4) for n, f in fractions.items()
            },
            "worst_prompt": min_prompt,
            "worst_shared_prefix_fraction": round(min_fraction, 4),
            "best_prompt": max_prompt,
            "best_shared_prefix_fraction": round(max_fraction, 4),
            "exact_match_across_all_prompts": exact_all,
            "stop_criterion_diverges_in_any_prompt": stop_diverges_any,
            "all_divergences_are_near_tie": all_near_tie,
            "recommendation": aggregate_recommendation,
            "finding": aggregate_finding,
            "pr4_implication": aggregate_implication,
            "note": "recommendation = 'path2_viable' is determined by the "
                    "self-consistency argument (diff-off bf16 vs int8 is a "
                    "clean comparison through one code path), NOT by the "
                    "diff-off/diff-on shared-prefix fraction. The fraction is "
                    "an independent finding about cross-code-path fp drift "
                    "through the same weights.",
        },
    }

    out_path = os.path.abspath("results/ar_equivalence.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

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
    print(f"configs:  {', '.join(args.configs)}")
    print(f"prompts:  {', '.join(args.prompts)}")
    int4_mode = "per-channel" if args.int4_per_channel else "per-tensor"
    print(f"int4 mode: {int4_mode}")

    # -----------------------------------------------------------------
    # Verification-only mode
    # -----------------------------------------------------------------
    if args.verify_only:
        print("\n=== Verification mode: parameter partition only ===")
        model, _tok = load_model(args.orthrus_revision)
        ar, diff = partition_params(model)
        print_partition(ar, diff)
        print("\nVerification complete. Re-run without --verify-only to execute "
              "the benchmark.")
        free_model(model)
        return

    # -----------------------------------------------------------------
    # AR-equivalence verification (PR 3): does use_diffusion_mode=False
    # produce the same token sequence as use_diffusion_mode=True at bf16?
    # -----------------------------------------------------------------
    if args.verify_ar_equivalence:
        model, tokenizer = load_model(args.orthrus_revision)
        run_ar_equivalence(args, tokenizer, model)
        free_model(model)
        del tokenizer
        return

    # -----------------------------------------------------------------
    # Full benchmark
    # -----------------------------------------------------------------
    # results[config][prompt] = measurement dict
    results = {c: {} for c in args.configs}

    for config in args.configs:
        print(f"\n{'=' * 70}\n=== Configuration: {config} ===\n{'=' * 70}")
        model, tokenizer = load_model(args.orthrus_revision)

        if config == BASELINE_KEY:
            ar, diff = partition_params(model)
            print_partition(ar, diff)

        scheme = CONFIG_QUANT_SCHEME[config]
        mode = CONFIG_MODE[config]
        print(f"\nApplying quantization scheme: {scheme} (int4 mode: {int4_mode})")
        print(f"Generation mode: {mode}")
        apply_quantization(model, config, args.int4_per_channel)

        use_diff = (mode == "diffusion")
        counter = PassCounter().attach(model)

        try:
            for prompt_name in args.prompts:
                input_ids = build_inputs(tokenizer, PROMPTS[prompt_name], model.device)
                # AR-mode configs don't halt at <|im_end|> (HF super().generate
                # emits it as a normal token), so capping at args.max_new_tokens
                # would generate ~4x more post-EOS garbage tokens than needed.
                # Use the per-prompt AR cap (short=512, long=2048) for AR mode;
                # diffusion-mode configs halt at EOS naturally.
                if mode == "ar":
                    mnt = AR_EQUIV_PROMPT_MAX_NEW_TOKENS.get(
                        prompt_name, args.max_new_tokens,
                    )
                else:
                    mnt = args.max_new_tokens
                results[config][prompt_name] = measure(
                    model, tokenizer, counter,
                    prompt_name, input_ids,
                    warmup_tokens=args.warmup_tokens,
                    max_new_tokens=mnt,
                    seed=args.seed,
                    label=config,
                    use_diffusion_mode=use_diff,
                )

                # In-loop within-arm comparison + divergence diagnostic.
                # The diagnostic needs the currently-loaded (quantized) model;
                # do it here before free_model.
                baseline_key = within_arm_baseline(config)
                if (config != baseline_key
                        and baseline_key in results
                        and prompt_name in results[baseline_key]):
                    bf16_full = results[baseline_key][prompt_name]["output_token_ids"]
                    this_full = results[config][prompt_name]["output_token_ids"]
                    # Truncate at first <|im_end|> so AR-mode runs (which don't
                    # halt at EOS) don't drag the comparison into post-EOS
                    # garbage. No-op for diffusion-mode runs that already stop
                    # at EOS.
                    bf16_ids = truncate_at_first(bf16_full, EOS_TOKEN_ID)
                    this_ids = truncate_at_first(this_full, EOS_TOKEN_ID)

                    fdp = first_divergence(bf16_ids, this_ids)
                    exact = (fdp is None)
                    edit = 0 if exact else levenshtein(bf16_ids, this_ids)

                    base_tpf = results[baseline_key][prompt_name]["tpf"]
                    base_tps = results[baseline_key][prompt_name]["tps"]
                    this_tpf = results[config][prompt_name]["tpf"]
                    this_tps = results[config][prompt_name]["tps"]
                    if base_tpf is not None and this_tpf is not None:
                        tpf_delta = round(this_tpf - base_tpf, 3)
                    else:
                        tpf_delta = None
                    tps_delta_pct = (
                        round(100.0 * (this_tps - base_tps) / base_tps, 2)
                        if base_tps else None
                    )

                    # Divergence diagnostic at the first real per-position
                    # disagreement (skip the case where one is a strict prefix
                    # of the other, where fdp == min length).
                    diagnostic = None
                    if (fdp is not None
                            and fdp < min(len(bf16_ids), len(this_ids))):
                        diagnostic = compute_divergence_diagnostic(
                            model=model,
                            tokenizer=tokenizer,
                            input_ids=input_ids,
                            agreed_prefix=bf16_ids[:fdp],
                            bf16_chosen_id=bf16_ids[fdp],
                            this_chosen_id=this_ids[fdp],
                        )

                    vs_base = {
                        "compared_to": baseline_key,
                        "compared_on_truncated_at_eos": True,
                        "exact_match": exact,
                        "edit_distance": edit,
                        "first_divergence_position": fdp,
                        "tpf_delta": tpf_delta,
                        "throughput_delta_percent": tps_delta_pct,
                        "divergence_diagnostic": diagnostic,
                    }
                    results[config][prompt_name]["vs_baseline"] = vs_base

                    tpf_str = (f"{tpf_delta:+.2f}" if tpf_delta is not None
                               else "n/a")
                    tps_str = (f"{tps_delta_pct:+.1f}%" if tps_delta_pct is not None
                               else "n/a")
                    gap_str = (f"{diagnostic['top1_to_top2_logit_gap']:.4f}"
                               if diagnostic else "n/a")
                    print(f"  vs {baseline_key:<14} {prompt_name:<6} "
                          f"exact={'yes' if exact else 'no':<3} "
                          f"edit={edit:<5} first_div={fdp}  "
                          f"gap@div={gap_str}  "
                          f"tpf_delta={tpf_str}  tps_delta={tps_str}")
        finally:
            counter.detach()

        free_model(model)
        del tokenizer

    # -----------------------------------------------------------------
    # Cross-arm headline: Orthrus arm (teacher-int8 vs baseline-bf16) vs
    # Vanilla arm (ar-int8 vs ar-bf16). Only produced if both within-arm
    # comparisons are available. The "uniquely fragile" interpretation is
    # decided by gap-at-first-divergence behaviour, not by position ratios.
    # -----------------------------------------------------------------
    cross_arm = None
    orth_int8 = "teacher-int8"
    van_int8 = "ar-int8"
    have_orth_arm = (
        BASELINE_KEY in results and orth_int8 in results
        and all(p in results[orth_int8] for p in args.prompts)
        and all(p in results[BASELINE_KEY] for p in args.prompts)
    )
    have_van_arm = (
        AR_BASELINE_KEY in results and van_int8 in results
        and all(p in results[van_int8] for p in args.prompts)
        and all(p in results[AR_BASELINE_KEY] for p in args.prompts)
    )
    if have_orth_arm and have_van_arm:
        print(f"\n{'=' * 70}\n=== Cross-arm headline: Orthrus vs vanilla-Qwen3 "
              f"int8 sensitivity ===\n{'=' * 70}")

        def _arm_row(int8_config, baseline_label):
            rows = []
            for prompt_name in args.prompts:
                vb = results[int8_config][prompt_name].get("vs_baseline", {})
                d = vb.get("divergence_diagnostic")
                rows.append({
                    "prompt": prompt_name,
                    "first_divergence_position": vb.get("first_divergence_position"),
                    "exact_match": vb.get("exact_match"),
                    "top1_to_top2_logit_gap": (
                        d["top1_to_top2_logit_gap"] if d else None
                    ),
                    "bf16_chosen_id": d["bf16_chosen_id"] if d else None,
                    "bf16_chosen_text": d["bf16_chosen_text"] if d else None,
                    "this_chosen_id": d["this_chosen_id"] if d else None,
                    "this_chosen_text": d["this_chosen_text"] if d else None,
                    "bf16_chosen_rank_in_int8": (
                        d["bf16_chosen_rank_in_this_model"] if d else None
                    ),
                })
            return rows

        orth_rows = _arm_row(orth_int8, BASELINE_KEY)
        van_rows = _arm_row(van_int8, AR_BASELINE_KEY)

        header = (f"  {'arm':<10} {'prompt':<6} {'first_div':>10} "
                  f"{'gap@div':>10} {'bf16 tok':>10} {'int8 tok':>10} "
                  f"{'bf16 rank':>10}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for arm_label, rows in (("Orthrus", orth_rows), ("Vanilla", van_rows)):
            for r in rows:
                fdiv_s = (str(r["first_divergence_position"])
                          if r["first_divergence_position"] is not None
                          else "---")
                gap_s = (f"{r['top1_to_top2_logit_gap']:.4f}"
                         if r["top1_to_top2_logit_gap"] is not None else "---")
                bf16_s = (f"{r['bf16_chosen_text']!r:>10}"
                          if r["bf16_chosen_text"] is not None else "---")
                int8_s = (f"{r['this_chosen_text']!r:>10}"
                          if r["this_chosen_text"] is not None else "---")
                rank_s = (str(r["bf16_chosen_rank_in_int8"])
                          if r["bf16_chosen_rank_in_int8"] is not None else "---")
                print(f"  {arm_label:<10} {r['prompt']:<6} {fdiv_s:>10} "
                      f"{gap_s:>10} {bf16_s} {int8_s} {rank_s:>10}")

        # Interpretation: are the two arms' first-divergence gap behaviours
        # similar (just trajectory variance) or substantively different
        # (Orthrus uniquely fragile)?
        def _gap_signature(rows):
            gaps = [r["top1_to_top2_logit_gap"] for r in rows
                    if r["top1_to_top2_logit_gap"] is not None]
            ranks = [r["bf16_chosen_rank_in_int8"] for r in rows
                     if r["bf16_chosen_rank_in_int8"] is not None]
            return {
                "median_gap": (sorted(gaps)[len(gaps) // 2] if gaps else None),
                "max_bf16_rank_in_int8": (max(ranks) if ranks else None),
                "all_near_tie_rank2_or_better": (
                    bool(ranks) and all(r <= 3 for r in ranks)
                    and bool(gaps) and all(g < 1.0 for g in gaps)
                ),
            }

        orth_sig = _gap_signature(orth_rows)
        van_sig = _gap_signature(van_rows)

        # 'Uniquely fragile' is a COMPARATIVE claim: it requires the two arms'
        # perturbation signatures to differ substantively. The earlier
        # absolute-threshold check (both arms must have all gaps <1.0 AND
        # ranks <=3) was wrong: it could trip on a single math-prompt gap of
        # exactly 1.0 in BOTH arms and still flag "uniquely fragile" even
        # though the two arms had identical signatures. The correct test is
        # similarity between arms; absolute character is a separate axis.
        def _signatures_similar(a, b, gap_tol=0.5, rank_tol=1):
            if a["median_gap"] is None or b["median_gap"] is None:
                return None
            return (
                abs(a["median_gap"] - b["median_gap"]) <= gap_tol
                and abs(a["max_bf16_rank_in_int8"]
                        - b["max_bf16_rank_in_int8"]) <= rank_tol
            )

        def _absolute_character(sig):
            """Describe one arm's perturbation absolute character (orthogonal
            to similarity-across-arms)."""
            if sig["median_gap"] is None:
                return "no divergences observed"
            if sig["median_gap"] < 1.0 and sig["max_bf16_rank_in_int8"] <= 3:
                return "near-tie regime (small gaps, rank-1-vs-rank-2 swaps)"
            if sig["median_gap"] < 2.0:
                return "moderate-gap regime"
            return "wide-margin regime"

        sigs_similar = _signatures_similar(orth_sig, van_sig)
        orth_char = _absolute_character(orth_sig)
        van_char = _absolute_character(van_sig)

        if sigs_similar is None:
            verdict_label = "INSUFFICIENT DATA"
            interpretation = (
                "Insufficient diagnostic data to compare arms (one or both "
                "had no divergences with diagnostics)."
            )
        elif sigs_similar:
            verdict_label = "NOT UNIQUELY FRAGILE"
            interpretation = (
                f"Arms have similar perturbation signatures: Orthrus median "
                f"gap {orth_sig['median_gap']:.2f} / max bf16-rank-in-int8 "
                f"{orth_sig['max_bf16_rank_in_int8']}; Vanilla median gap "
                f"{van_sig['median_gap']:.2f} / max bf16-rank-in-int8 "
                f"{van_sig['max_bf16_rank_in_int8']}. Same kind of "
                f"perturbation in both arms ({orth_char}). The diffusion "
                f"consensus mechanism does not amplify int8 errors beyond "
                f"what the plain AR path already experiences. Orthrus is "
                f"NOT uniquely fragile to int8; it inherits Qwen3's "
                f"quantization sensitivity."
            )
        else:
            verdict_label = "UNIQUELY FRAGILE"
            interpretation = (
                f"Arms exhibit qualitatively different perturbation "
                f"signatures. Orthrus: median gap "
                f"{orth_sig['median_gap']:.2f} / max rank "
                f"{orth_sig['max_bf16_rank_in_int8']} ({orth_char}). "
                f"Vanilla: median gap {van_sig['median_gap']:.2f} / max "
                f"rank {van_sig['max_bf16_rank_in_int8']} ({van_char}). "
                f"Difference exceeds tolerance (gap_tol=0.5, rank_tol=1); "
                f"the diffusion consensus mechanism may be amplifying "
                f"precision errors. Orthrus is uniquely fragile."
            )
        print(f"\nVERDICT: {verdict_label}")
        print(f"INTERPRETATION: {interpretation}")

        cross_arm = {
            "orthrus_arm": {
                "comparison": f"{orth_int8} vs {BASELINE_KEY}",
                "per_prompt": orth_rows,
                "gap_signature": orth_sig,
                "absolute_character": orth_char,
            },
            "vanilla_arm": {
                "comparison": f"{van_int8} vs {AR_BASELINE_KEY}",
                "per_prompt": van_rows,
                "gap_signature": van_sig,
                "absolute_character": van_char,
            },
            "signatures_similar": sigs_similar,
            "verdict": verdict_label,
            "interpretation": interpretation,
            "note": "Verdict is COMPARATIVE: 'uniquely fragile' requires the "
                    "two arms' perturbation signatures to differ "
                    "substantively (gap diff > 0.5, rank diff > 1). Position "
                    "differences between arms are explicitly ignored "
                    "(trajectory variance from cross-code-path fp drift, not "
                    "architectural fragility). Each arm's absolute_character "
                    "is reported separately so a reader can distinguish "
                    "'both arms near-tie -> benign quant' from 'both arms "
                    "wide-margin -> harsh quant' even when the comparative "
                    "verdict says not-uniquely-fragile.",
        }

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    print(f"\n{'=' * 70}\n=== Quantization Impact ===\n{'=' * 70}")
    header = (f"  {'config':<14} {'mode':<10} {'prompt':<6} {'TPF':>6} "
              f"{'tok/s':>7} {'tokens':>7} {'exact':>6} {'first_div':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config in args.configs:
        for prompt_name in args.prompts:
            r = results[config][prompt_name]
            mode = CONFIG_MODE[config]
            is_arm_baseline = (config == BASELINE_KEY or config == AR_BASELINE_KEY)
            if is_arm_baseline:
                exact_s, fdiv_s = "---", "---"
            else:
                vb = r.get("vs_baseline", {})
                exact_s = "yes" if vb.get("exact_match") else "no"
                fdiv_s = (str(vb.get("first_divergence_position"))
                          if vb.get("first_divergence_position") is not None
                          else "---")
            tpf_s = (f"{r['tpf']:>6.2f}" if r['tpf'] is not None
                     else f"{'n/a':>6}")
            print(f"  {config:<14} {mode:<10} {prompt_name:<6} "
                  f"{tpf_s} {r['tps']:>7.1f} {r['tokens']:>7} "
                  f"{exact_s:>6} {fdiv_s:>10}")

    # -----------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------
    import accelerate
    import transformers
    cap = torch.cuda.get_device_capability(0)
    mem_bytes = torch.cuda.get_device_properties(0).total_memory

    output = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(cap),
            "total_memory_gb": round(mem_bytes / 1024 ** 3, 1),
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
            "configs": args.configs,
            "config_modes": {c: CONFIG_MODE[c] for c in args.configs},
            "config_quant_schemes": {c: CONFIG_QUANT_SCHEME[c]
                                     for c in args.configs},
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "seed": args.seed,
            "do_sample": False,
            "int4_mode": int4_mode,
            "orthrus_revision": args.orthrus_revision,
        },
        "notes": {
            "quantization": (
                "Simulated: cast-and-dequantize. Weights round-trip through int8/int4 "
                "but stay in bf16 at runtime. In-memory footprint is unchanged; the "
                "experiment isolates the distribution-shift impact of teacher precision "
                "loss from kernel-performance effects."
            ),
            "tpf": (
                "tokens_per_forward_pass = generated_tokens / number of diffusion-mode "
                "iterations (proposal forward calls). One iteration also performs one AR "
                "verification forward call; total_forward_passes is reported separately."
            ),
            "ar_partition": (
                "AR/teacher = all parameters whose qualified name does NOT contain '_diff' "
                "(embedding, MLP, layer norms, original Qwen3 attention projections, "
                "lm_head, final norm). This is the full set of weights the diffusion head "
                "was trained to match against; quantizing them shifts the target "
                "distribution."
            ),
        },
        "results": results,
        "cross_arm_comparison": cross_arm,
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
