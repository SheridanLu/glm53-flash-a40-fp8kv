# Design notes

## 1. Why the existing FP8 KV approach does not apply

`TRITON_MLA_SPARSE` (the sm_80/sm_86 sparse-MLA backend) originally declared only `auto`/`float16`/`bfloat16`, and its base class (`XPUMLASparseImpl`) raised `NotImplementedError("FP8 kv is not supported ...")` for any quantised cache. Triton cannot convert to/from `fp8e4nv` below sm_89, which is the usual reason given.

That restriction applies to *native* FP8 arithmetic. If FP8 is used purely as **storage** — queries stay bf16, pages are decoded on load — no native FP8 instruction is needed. Community prototypes ([nickus/vllm-fp8kv](https://github.com/nickus/vllm-fp8kv), vLLM RFC #48374) do exactly this for the DeepSeek-style `fp8_ds_mla` page:

```
656 B = 512 × fp8e4m3 (NoPE) + 4 × fp32 scales + 64 × bf16 (RoPE, raw)
```

GLM-5.3-Flash has `qk_rope_head_dim = 0`; its cache row is 512 values. The RoPE tail does not exist, so the 656 B layout (and the C++ `concat_and_cache_mla` writer, which asserts `pe_dim == 64` for `fp8_ds_mla`) cannot be reused as-is.

## 2. Page layout (NoPE)

```
offset   size   content
0        512 B  512 × fp8e4m3        (NoPE latent)
512      16 B   4 × fp32             (per-128-element tile scale = amax / 448)
528      –      (end; no RoPE bytes)
```

Capacity gain per MLA layer: 1024 B → 528 B (1.94×). The measured pool grew 1.85×, a little less, because other per-token cache content in the model does not shrink. We did not break that down, so we do not quote a per-layer figure for it.

## 3. Decode (Triton)

`_decode_fp8e4m3(b)` decodes an e4m3fn byte to fp16 with integer operations only:

```
sign  = (u & 0x80) << 8
mag   = (u & 0x7F) << 7
value = bitcast_f16(sign | mag) * 256
NaN   = 0x7F / 0xFF
```

It is bit-exact against `torch.float8_e4m3fn` for all 256 byte values. The kernel loads the 512 payload bytes as `uint8`, decodes, multiplies each 128-wide tile by its scale, and feeds the result to the existing `tl.dot`. Because MLA's V is the NoPE half of the same row, the decoded K tile is transposed and reused as V instead of being loaded again. The scale load is a broadcast of 4 values per row (gathering per element would stage 32 KB per stage and exceed sm_86's ~100 KB shared memory).

## 4. Write path

```
quantize_nope_pages(kv_c)   # torch: per-128 amax -> scale -> .to(float8_e4m3fn); [T,512] bf16 -> [T,528] u8
write_fp8_nope_kv(...)      # Triton: scatter page t to cache[slot // bs, slot % bs]; skip if slot < 0
```

- `torch`'s fp8 cast works on sm_86; Triton's does not, which is why quantisation is not inside the Triton kernel.
- `slot < 0` marks CUDA-graph padding. The scatter skips it without a host sync, so the whole write is graph-capturable.
- The write is addressed with the cache tensor's real strides, not an assumed contiguous layout (the hybrid allocator can leave gaps between blocks). The unit test deliberately uses a padded block stride and asserts that gap bytes are untouched.

## 5. Wiring (4 files, ≈213 lines)

| File | Change |
|---|---|
| `triton_mla_sparse_kernel.py` | FP8 branch in `_sparse_mla_compute_tile`; `IS_FP8` constexpr threaded through both kernels; quantise + scatter helpers |
| `triton_mla_sparse.py` | Backend declares the fp8 dtypes; `Impl.supports_fp8_ds_mla_nope`; autotune warm-up also compiles the fp8 variant; `do_kv_cache_update` override |
| `xpu_mla_sparse.py` | The quantised-KV rejection becomes opt-in via `supports_fp8_ds_mla_nope` |
| `mla_attention.py` | `fp8`/`fp8_e4m3`/`fp8_ds_mla` normalise to `fp8_ds_mla` for this backend; `state_content_bytes` is 528 for head_size 512 (656 otherwise) |

## 6. The patched 5-file base (`patches/patched_tree.diff`)

Independent of FP8 KV, the deployment also applies two upstream changes aimed at stale / uncommitted hybrid-state reuse (the failure class discussed in wtdcode/vllm-backport#107 and #119):

- vLLM #56525 (commit-fence guard for state-block prefix-cache hits) — applied as is.
- vLLM #56524 (zero Mamba state blocks on allocation) — **hand-ported**: the container's `KVBlockZeroer` is organised per KV-cache group, so the upstream hunk does not apply; the loop body is wrapped to iterate over the tuple of state tensors a Mamba layer binds.

We make **no claim** that these fix #107 or #119. See [validation.md](validation.md).

## 7. Failure we hit (worth knowing)

The first FP8 launch failed with `No valid attention backend found ... kv_cache_dtype=fp8_ds_mla`. The launch script took the changed files from a git commit that predated the edits, so the container mounted the old `triton_mla_sparse.py`. The script now copies from the working tree and refuses to start unless five markers are present in the files it is about to mount. The same script rolls back to the BF16 launch if the engine is not healthy in seven minutes.
