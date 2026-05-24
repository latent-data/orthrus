import json
import glob
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


@dataclass
class OrthrusConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    block_size: int
    mask_token_id: int
    tie_word_embeddings: bool = False


class RingKVCache:
    def __init__(self, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self._keys: Optional[mx.array] = None
        self._values: Optional[mx.array] = None
        self.offset = 0
        self._head = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, L, D = keys.shape
        if self._keys is None:
            self._keys = mx.zeros((B, H, self.max_seq_len, D), dtype=keys.dtype)
            self._values = mx.zeros((B, H, self.max_seq_len, D), dtype=values.dtype)

        self._keys = mx.slice_update(self._keys, keys, mx.array([self._head]), [2])
        self._values = mx.slice_update(self._values, values, mx.array([self._head]), [2])
        self._head = (self._head + L) % self.max_seq_len
        self.offset += L
        return self._keys[:, :, :self.offset, :], self._values[:, :, :self.offset, :]

    @property
    def keys(self) -> Optional[mx.array]:
        return self._keys[:, :, :self.offset, :] if self._keys is not None else None

    @property
    def values(self) -> Optional[mx.array]:
        return self._values[:, :, :self.offset, :] if self._values is not None else None

    def trim(self, n: int) -> None:
        self.offset = max(0, self.offset - n)
        self._head = self.offset % self.max_seq_len


class OrthrusAttention(nn.Module):
    def __init__(self, config: OrthrusConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.q_proj_diff = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj_diff = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj_diff = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj_diff = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm_diff = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_diff = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, mask=None, rope=None, cache=None, is_diffusion_pass=False):
        B, L, _ = x.shape

        if not is_diffusion_pass:
            queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
            keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

            if cache is not None:
                queries = rope(queries, offset=cache.offset)
                keys = rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = rope(queries)
                keys = rope(keys)

            out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

        else:
            queries = self.q_norm_diff(self.q_proj_diff(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
            keys = self.k_norm_diff(self.k_proj_diff(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            values = self.v_proj_diff(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

            offset = cache.offset if cache is not None else 0
            queries = rope(queries, offset=offset)
            keys = rope(keys, offset=offset)

            if cache is not None:
                keys = mx.concatenate([cache.keys, keys], axis=2)
                values = mx.concatenate([cache.values, values], axis=2)

            out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=None)
            return self.o_proj_diff(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class QwenMLP(nn.Module):
    def __init__(self, config: OrthrusConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class OrthrusDecoderLayer(nn.Module):
    def __init__(self, config: OrthrusConfig):
        super().__init__()
        self.self_attn = OrthrusAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, mask=None, rope=None, cache=None, is_diffusion_pass=False):
        h = x + self.self_attn(self.input_layernorm(x), mask, rope, cache, is_diffusion_pass)
        return h + self.mlp(self.post_attention_layernorm(h))


class OrthrusModel(nn.Module):
    def __init__(self, config: OrthrusConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [OrthrusDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs, mask=None, rope=None, cache=None, is_diffusion_pass=False):
        h = self.embed_tokens(inputs)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask=mask, rope=rope, cache=cache[i] if cache is not None else None,
                      is_diffusion_pass=is_diffusion_pass)
        return self.norm(h)


class OrthrusForCausalLM(nn.Module):
    def __init__(self, config: OrthrusConfig):
        super().__init__()
        self.config = config
        self.model = OrthrusModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, inputs, cache=None, is_diffusion_pass=False):
        L = inputs.shape[1]
        S = cache[0].offset if cache is not None else 0

        if not is_diffusion_pass and L > 1:
            dtype = self.model.embed_tokens.weight.dtype
            mask = mx.arange(S + L)[None, :] > (mx.arange(L)[:, None] + S)
            mask = mx.where(mask, mx.finfo(dtype).min, 0.0).astype(dtype)
        else:
            mask = None

        h = self.model(inputs, mask=mask, rope=self.rope, cache=cache,
                       is_diffusion_pass=is_diffusion_pass)
        return self.lm_head(h)


def sample(logits: mx.array, temperature: float = 0.0) -> mx.array:
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits * (1.0 / temperature))


def mlx_generate(
    model: OrthrusForCausalLM,
    prompt_tokens: List[int],
    eos_token_id: int,
    max_tokens: int = 256,
    temperature: float = 0.0,
):
    block_size = model.config.block_size
    mask_id = model.config.mask_token_id

    prompt = mx.array([prompt_tokens])
    tokens = prompt.tolist()[0]
    cache = [RingKVCache() for _ in range(len(model.model.layers))]

    logits = model(prompt, cache=cache, is_diffusion_pass=False)
    token = sample(logits[:, -1, :], temperature).item()
    tokens.append(token)
    n = 1
    yield token

    while n < max_tokens:
        bs = min(block_size, max_tokens - n + 1)
        if bs <= 1:
            break

        diff_block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
        diff_logits = model(diff_block, cache=cache, is_diffusion_pass=True)
        diff_tokens = sample(diff_logits[:, :-1, :], temperature)
        mx.eval(diff_tokens)

        verify_input = mx.concatenate([mx.array([[tokens[-1]]]), diff_tokens], axis=1)
        ar_logits = model(verify_input, cache=cache, is_diffusion_pass=False)
        ar_tokens = sample(ar_logits, temperature)
        mx.eval(ar_tokens)

        d_list, t_list = diff_tokens[0].tolist(), ar_tokens[0].tolist()

        accepted = 0
        for d, t in zip(d_list, t_list[:-1]):
            if d == t:
                accepted += 1
            else:
                break

        new_tokens = d_list[:accepted] + [t_list[accepted]]
        trim_amount = verify_input.shape[1] - (accepted + 1)

        if trim_amount > 0:
            for c in cache:
                c.trim(trim_amount)

        for t in new_tokens:
            tokens.append(t)
            yield t
            if t == eos_token_id:
                return

        n += len(new_tokens)


def load_model_and_tokenizer(repo_id: str):
    print(f"Loading '{repo_id}' from Hugging Face...")
    repo_path = Path(snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors"]))

    with open(repo_path / "config.json", "r", encoding="utf-8") as f:
        hf_config = json.load(f)

    if "rope_parameters" in hf_config and isinstance(hf_config["rope_parameters"], dict):
        if "rope_theta" in hf_config["rope_parameters"] and "rope_theta" not in hf_config:
            hf_config["rope_theta"] = hf_config["rope_parameters"]["rope_theta"]

    required_keys = [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size",
        "rms_norm_eps", "rope_theta", "block_size", "mask_token_id",
    ]
    missing_keys = [k for k in required_keys if k not in hf_config]
    if missing_keys:
        raise ValueError(f"CRITICAL: Missing required keys in config.json: {missing_keys}")

    head_dim = hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"])

    config = OrthrusConfig(
        hidden_size=hf_config["hidden_size"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_key_value_heads=hf_config["num_key_value_heads"],
        head_dim=head_dim,
        intermediate_size=hf_config["intermediate_size"],
        vocab_size=hf_config["vocab_size"],
        rms_norm_eps=hf_config["rms_norm_eps"],
        rope_theta=hf_config["rope_theta"],
        block_size=hf_config["block_size"],
        mask_token_id=hf_config["mask_token_id"],
        tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
    )

    model = OrthrusForCausalLM(config)

    weight_files = glob.glob(str(repo_path / "*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    if config.tie_word_embeddings and "lm_head.weight" not in weights:
        if "model.embed_tokens.weight" in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    model.load_weights(list(weights.items()), strict=True).eval()
    mx.eval(model.parameters())

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer