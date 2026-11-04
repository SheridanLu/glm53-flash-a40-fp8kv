# glm53-flash-a40-fp8kv

**FP8 KV cache for NoPE sparse-MLA on Ampere (sm_86) — serving GLM-5.3-Flash on 8× A40.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)
![Arch](https://img.shields.io/badge/CUDA-sm__86-76b900.svg)

[English](README.md) · [简体中文](README.zh-CN.md)

GLM-5.3-Flash is a 320B-parameter MoE with sparse MLA attention and **no RoPE tail** (`qk_rope_head_dim = 0`). On Ampere there is no sparse-MLA backend in mainline vLLM and no FP8 KV cache for the community Triton backend, so the KV pool is BF16-only. This repository adds a software-dequant **FP8 KV cache for the NoPE layout**, together with the tests, monitor and raw logs used to evaluate it on a shared production box.

> **Status: experimental.** The numbers below are real and reproducible from the logs in [`results/`](results), but several things are **not** verified — see [Limitations](#limitations) before relying on it.

## Highlights

- **1.85× larger KV pool** on the same memory: 955,759 → 1,771,693 tokens; concurrent 512K-token requests go from 1.82× to 3.38×.
- **528 B per token per layer** instead of 1024 B — a NoPE-specific page layout (the existing 656 B DeepSeek-style `fp8_ds_mla` page assumes a RoPE tail this model does not have).
- **Runs on sm_86**: decoding is bit-math inside the Triton kernel, so it never needs `fp8e4nv`, which does not exist below sm_89.
- **Needle retrieval 12/12** from 8K to 400K tokens on the FP8 cache; single-stream speed unchanged within measurement noise (107–110 tok/s).
- Ships with a **read-only canary monitor** and a **kernel/write-path test suite** (with negative controls that must fail).

![KV cache capacity](docs/img/kv-capacity.svg)

## Why this matters

Real multi-user traffic on this deployment is long-context: of 1,232 requests, 527 carried 200K–500K tokens and the mean was 167,820 tokens. With a BF16 KV cache the pool reached 99.7 % utilisation and the prefix-cache hit rate collapsed to 1–7 % for about an hour, which stalled every user. After the switch to FP8 KV that did not recur in the roughly four hours observed since.

![Prompt length distribution](docs/img/prompt-lengths.svg)

## How it works

![KV row layout](docs/img/kv-row-layout.svg)

| Part | What it does |
|---|---|
| Page layout | 512 × `fp8e4m3` + 4 × `fp32` per-128 scales = **528 B** per token per layer |
| Decode | Triton kernel loads the page as `uint8`, decodes e4m3 with integer bit-math (bit-exact against `torch.float8_e4m3fn` for all 256 byte values), applies the tile scale. MLA's V is the same row, so the decoded K tile is reused as V |
| Write | `torch` quantises per 128-wide tile (works on sm_86); a small Triton kernel scatters the 528 B pages into the paged cache and skips `slot < 0`. No host sync, so it is CUDA-graph capturable |
| Wiring | `TRITON_MLA_SPARSE` declares `fp8`/`fp8_e4m3`/`fp8_ds_mla`; quantised dtypes normalise to `fp8_ds_mla`; NoPE page size is 528 instead of 656; the base class's hard rejection of quantised KV becomes opt-in |

Details, including the failure modes we hit, are in [`docs/design.md`](docs/design.md).

## Results

Measured on 8× A40 (46 GB), TP8, MTP-3, prefix caching, 512K context, `--gpu-memory-utilization 0.88`, 2026-11-04.

| | BF16 KV | FP8 KV |
|---|---|---|
| KV pool | 955,759 tokens | **1,771,693 tokens** |
| 512K-request concurrency | 1.82× | **3.38×** |
| Single stream | 85–124 tok/s (mean ≈107) | 107–110 tok/s |
| 8 streams, aggregate | 259 tok/s | 289 tok/s ¹ |
| Needle retrieval, 8K–400K × 3 depths | — | **12 / 12** |
| Arithmetic / format / code (chat endpoint) | — | 12/12 · 3/3 · 3/3 |
| Attention-output error (synthetic) | — | 1.3–3.2 % rel., cos 0.9995–0.9999 |

¹ The two runs had different background load; differences within ±12 % should not be read as a speed-up.

Full method, raw numbers and the file each one comes from: [`docs/validation.md`](docs/validation.md).

## Quick start

> Requires the `lazymio/vllm-backport:v0.13.1-sm86` image and `wtdcode/GLM-5.3-Flash-AWQ-W4A16`. Paths in the scripts (`/data/glm_test/...`) are from our host; adjust them.

```bash
# 1. Extract the container's vllm *.py files as the baseline, then apply
git apply patches/patched_tree.diff      # #56525 + a hand-port of #56524 (see docs/design.md)
git apply patches/fp8kv_wiring.diff      # this repo's FP8 KV change (4 files)

# 2. Bind-mount the modified files read-only over the container's package and add
#       --kv-cache-dtype fp8_ds_mla
bash scripts/run_glm_fp8kv.sh            # verifies 5 markers, auto-rolls back if the engine does not come up

# 3. Test (needs a GPU; the scripts cap their own memory)
python3 tests/test_fp8kv_kernel.py
python3 tests/test_fp8kv_write.py
```

## Limitations

1. **No same-prompt BF16 vs FP8 comparison on real text** (perplexity or logit difference). The accuracy figures are absolute pass rates on an easy suite that BF16 would probably also ace; they cannot size the FP8 penalty.
2. **Kernel error was measured on synthetic Gaussian data.** Real latents, error accumulation over 11 sparse layers and INT4-weights × FP8-KV compounding are untested.
3. **The canary cannot see long-session degeneration** (e.g. wtdcode/vllm-backport#119). It found 0 wrong or cross-contaminated answers in 79 runs, but it only sends short probes.
4. **The abort-and-retry stress test does not discriminate**: 0/36 failures both without and with the patch set, and the baseline ran with MTP off.
5. The quantise step is torch ops plus a Triton scatter, not a fused kernel; its end-to-end cost was not isolated.
6. Speed numbers come from runs with different background load (see footnote above).
7. Image requests were not tested. The MTP draft model does not consume vision embeddings (engine warning).

## Repository layout

```
patches/    fp8kv_wiring.diff (this work) · patched_tree.diff (deployed 5-file set vs v0.13.1)
scripts/    launch + auto-rollback, patch dry-run, porting helpers
tests/      kernel and write-path tests, accuracy suites, needle-in-haystack, stress, metrics summary
monitor/    glm_monitor.py — read-only health + two-way concurrent canary (cron, every 5 min)
results/    raw logs behind every number above
docs/       design.md · validation.md · figures
```

## Related work and credits

- [nickus/vllm-fp8kv](https://github.com/nickus/vllm-fp8kv) (Apache-2.0) — the software-dequant FP8 approach and the bit-exact decoder this work adapts to the NoPE layout.
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — the Ampere image and GLM-5.3-Flash support this runs on.
- vLLM [#56524](https://github.com/vllm-project/vllm/pull/56524) / [#56525](https://github.com/vllm-project/vllm/pull/56525) (datacrystals) and the patch list posted by ehrhart in wtdcode/vllm-backport#119 — used for the patched set. `patches/patched_tree.diff` contains the **net effect** of #56525 and a hand-port of #56524 against v0.13.1 (vLLM is Apache-2.0; the authors are credited here). The other PRs from that list were only dry-run, not applied, and are not included.
- vLLM RFCs [#48374](https://github.com/vllm-project/vllm/issues/48374) and [#59016](https://github.com/vllm-project/vllm/issues/59016) — the upstream discussion of FP8 KV on Ampere sparse MLA.

## Maintainer

Created and maintained by **[SheridanLu](https://github.com/SheridanLu)**, who set the objectives, owned the deployment and release decisions on shared production hardware, and required that every claim be backed by a raw log.

*AI assistance:* the code, tests and documentation were written with Claude Code (Anthropic) under the maintainer's direction. All experiments ran on the maintainer's hardware. Please review the code accordingly.

## License

Apache-2.0 (derivative of vLLM). See [LICENSE](LICENSE) and [NOTICE](NOTICE). Provided without warranty.
