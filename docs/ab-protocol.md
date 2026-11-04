# A/B protocol: BF16 KV vs FP8 KV

Status: **written before any BF16/FP8 comparison data was collected; the comparison has since been run (PASS).** The thresholds below are the ones coded in `tests/ab_eval.py` (`THRESHOLDS`). Changing them after seeing results invalidates the verdict.

## Question

Does the NoPE FP8 KV cache change the model's outputs on real text by more than a small, pre-agreed amount?

## Design

- **One image, one flag.** Both arms run `glm53-fp8kv:v0.13.1-sm86`; the reference uses `--kv-cache-dtype auto` (BF16) and the candidate uses `--kv-cache-dtype fp8_ds_mla`. Everything else — weights, TP8, MTP-3, prefix caching, cudagraph sizes, `--max-model-len` — is identical, so the only difference is the cache dtype.
- **Evaluation configuration.** `--gpu-memory-utilization 0.80 --max-num-batched-tokens 2048` for both arms. The `ppl` step asks the engine for logits at every prompt position (chunk × ~155K vocabulary, fp32, on rank 0); with the production margin (≈2.9 GiB free per card, 8192-token chunks) that can exhaust memory and crash the engine, so it is run only on this configuration. `ab_eval.py ppl` refuses to run unless `AB_EVAL_SAFE=1` is set.
- **Same prompts, same order.** Prompts are generated once (`ab_eval.py prep`), hashed in `manifest.json`, and both arms read the same files. Token ids are hashed per document; a mismatch aborts the comparison.
- **Noise floor.** The reference arm is run twice; reference-vs-reference differences measure what the engine does to itself between runs (batch composition, kernel non-determinism). A candidate is judged against the reference *and* against that floor. The candidate arm is run once (decided before any data was collected, to shorten the window; a candidate-vs-candidate floor is not used by any pass criterion).
- **No other traffic.** The public front door is stopped during evaluation so the eval is the only load. (Under live traffic, same-config greedy continuations already diverge by a median of token 7 in a 4-prompt check, so uncontrolled load would swamp the signal.)

## Data

| Set | Content | Size |
|---|---|---|
| Perplexity | real **code** (Python source from an installed `vllm`) and real **technical English** (package READMEs from `*.dist-info/METADATA`) | per kind: 8 × 4K, 4 × 16K, 2 × 64K, 1 × 128K tokens (≈710K tokens in total) |
| Greedy | 20 code + 20 text prefixes (~1.2K tokens), 128 new tokens, temperature 0 | 40 |
| MMLU | random 500 questions of the 14,042-question test split (seed 7), 0-shot, scored by the logprob of A/B/C/D | 500 |

The corpora are a convenience sample, not agent traffic. MMLU is a sanity check on knowledge/reasoning, not a sensitive measure of cache error (see power).

## Metrics

1. **Perplexity ratio** (candidate / reference) on the actual next token, token-weighted, per kind and overall, with a 95 % document-bootstrap CI when ≥ 4 documents.
2. **Top-1 agreement:** fraction of positions where the two arms' argmax token is the same.
3. **Mean |Δ logprob|** of the actual token (reported, no threshold).
4. **MMLU accuracy** difference, with McNemar's exact p-value on the discordant pairs.
5. **Greedy continuations:** fraction identical and median position of first divergence (reported, no threshold, always shown beside its noise floor).

## Pre-registered pass criteria

| # | Check | Threshold |
|---|---|---|
| 1 | Perplexity ratio, pooled over all documents and for each kind | ≤ 1.01 |
| 2 | Upper end of the 95 % CI of that ratio | ≤ 1.02 |
| 3 | Top-1 agreement (all, code, text) | ≥ 0.97 |
| 4 | …and not more than 0.02 below the reference-vs-reference agreement | gap ≤ 0.02 |
| 5 | MMLU accuracy: reference − candidate | ≤ 0.03 |
| 6 | MMLU invalid answers (no A–D in the top-20 tokens) | ≤ 0.02 |

Verdict: **PASS** if every check passes, **FAIL** if any fails, **INCOMPLETE** if perplexity or MMLU data are missing. Per-length-bucket rows are reported but are not pass/fail (they have 1–8 documents each).

## What the verdict does and does not mean

- **Power.** With 500 questions the standard error of an accuracy is ≈ 2.2 points, so MMLU can only expose a drop of a few points; it is a sanity check. The perplexity and agreement metrics are far more sensitive and carry the verdict.
- A PASS means: on this corpus, at these lengths, FP8 KV changed next-token predictions by less than the thresholds relative to BF16 *and* no more than the engine already changes between identical runs. It does not cover long-session degeneration (wtdcode/vllm-backport#119), other models, other hardware, or image inputs.
- A FAIL will be published with the numbers, unchanged.

## Negative controls

`ab_eval.py selftest` builds synthetic runs and asserts: a lightly perturbed candidate **PASSes**; a candidate with shifted logprobs and flipped argmax tokens **FAILs**; a candidate with an 8-point MMLU drop **FAILs**. It must report `SELFTEST OK` before any real comparison is trusted.

## Procedure

`scripts/run_ab_window.sh` (needs `CONFIRM=yes`; stops the public front door, runs three restarts: BF16 eval config → FP8 eval config → FP8 production config, then restores the front door). Its trap restores service with the previous known-good launcher if any step fails. The driver ran end to end once on 2026-11-04 (16:11–16:45 UTC) and the result was PASS; see `docs/validation.md` ("A/B evaluation") and `results/ab-2026-11-05/`. Its parts were tested separately beforehand.
