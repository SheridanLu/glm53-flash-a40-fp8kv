# Validation

Every number in the README comes from a file in [`../results`](../results) or from a script in [`../tests`](../tests) run against the live service. Hardware: 8× A40, TP8. Date: 2026-11-04.

## Kernel and write-path tests

| Test | What it asserts | Result |
|---|---|---|
| `tests/test_fp8kv_kernel.py` | FP8 kernel vs "dequantise then BF16 kernel" (decode correctness); FP8 vs original BF16 (quantisation cost); zeroed-scale negative control | cos ≈ 1.0000000, rel. err 4e-6 – 4e-4 (decode); cos 0.9995–0.9999, rel. err 1.3–3.2 % (quantisation); control cos 0.0000 (detected) |
| `tests/test_fp8kv_write.py` | Write → read-back byte-equal; only target rows touched (padded block stride, `slot = -1` skipped); full path vs reference; negative control | all PASS, 19 MiB peak GPU memory |

Both run in a separate process with a hard per-process memory cap, on a card that is simultaneously serving traffic.

## Accuracy on the FP8-KV instance

Answers are machine-checkable; there is no reliance on a judge model.

| Suite | Cases | Result |
|---|---|---|
| Needle retrieval, 8K / 32K / 128K × depth 10/50/90 % | 9 | 9/9 |
| Needle retrieval, ≈400K tokens (400,016–400,017) × depth 10/50/90 % | 3 | 3/3 (first token ≈136 s on a cold prefill) |
| Arithmetic (`a*b+c`), chat endpoint | 12 | 12/12 |
| Format ("reply with exactly this word") | 3 | 3/3 |
| Code, executed and checked, chat endpoint | 3 | 3/3 |

Raw: `results/acc_fp8.log`, `results/needle400k_v2.log`. **Note on endpoints:** the completions endpoint applies no chat template, so the model keeps writing after its answer and a naive extractor fails (see the first arithmetic/code lines in `acc_fp8.log`). Those were harness errors, not model errors; `tests/code_chat.py` uses the chat endpoint.

## Speed

`glm_bench`-style runs, zero background load. FP8 vs BF16 measurements were taken on different occasions, so only gross differences should be read from them.

| Concurrency | BF16 aggregate | FP8 aggregate |
|---|---|---|
| 1 | 85–124 tok/s | 107–110 tok/s |
| 2 | 127 | 129 |
| 4 | 219 | 214 |
| 8 | 259 | 289 |

Real traffic on the FP8 instance (≈1,262 requests): mean per-request decode 63.6–75.4 tok/s; mean time-to-first-token 3.5 s (90 % ≤ 2.5 s; ≈2 % > 20 s). Summary script: `tests/metrics_summary.py`.

## Canary monitor

`monitor/glm_monitor.py` runs from cron every five minutes: container state, `/health`, log errors, prefix-cache hit rate, MTP acceptance length, nginx 5xx, plus **two concurrent probes sharing a ≈4K-token prefix, each carrying a unique code to repeat**. A wrong code or the other probe's code is a failure. `--selftest` forces a wrong expectation and must report failure.

79 runs (06:55–13:25 UTC): 71 pass, 5 fail, 3 skipped. **All 5 failures were timeouts** during a window when the BF16 KV pool was exhausted; there were no wrong or mixed answers and no log errors. Raw: `results/status.log`, `results/alerts.log`.

## Packaging checks (no GPU, 2026-11-04)

Run on the host against the stock image; nothing touched the serving container.

| Check | Result |
|---|---|
| Pristine files from `docker create` of the base image + `apply_patches.sh` → compare with the 9 files currently bind-mounted in production | **9/9 identical (sha256)** |
| `apply_patches.sh` dry-run, apply, second run ("already patched"), `--revert` | all as expected; revert restores the original hashes (0 differences) |
| Docker build (legacy builder, local base) | succeeds; patches apply, 10 markers present, all patched files `py_compile` |
| 9 patched files inside the built image vs production | **9/9 identical** |
| Negative control: corrupt a *removed* line in the diff | dry-run exits 1, pristine copy untouched |
| Negative control: delete one marker from a patched tree | `verify_markers.sh` exits 1 and names the missing marker |
| Negative control: apply only the first diff (half-patched) | `verify_markers.sh` exits 1 |
| Negative control: image whose kernel file lost a marker | entrypoint exits **78** and prints the failing marker; the intact image passes the guard |

Two of my first negative controls were wrong and are worth recording. (1) I corrupted an *added* line by appending a character to an identifier, which still contained the marker substring and applied cleanly, so it proved nothing; the control above corrupts a removed line instead. (2) I built the "broken image" with `docker commit` from a container started with `--entrypoint sh`, which saved that entrypoint into the image so my guard was never exercised (`sh: Illegal option`); the image is now committed with the entrypoint restored.

The image was later started on 8× A40 in both cache modes; see "A/B evaluation" below.

## A/B harness checks (2026-11-04)

| Check | Result |
|---|---|
| `ab_eval.py selftest` | a lightly perturbed candidate PASSes; a candidate with shifted logprobs and flipped argmax tokens FAILs; a candidate with an 8-point MMLU drop FAILs |
| `prompt_logprobs` parser on a real 12-token response from the engine | 11 positions scored, every position has a rank-1 entry, all logprobs ≤ 0 |
| MMLU prompt format, 60 questions on the live FP8 instance | accuracy 0.867, predictions spread over A–D, 0 invalid |
| Prompt set | 30 documents, 686,437 tokens (code and technical English, 4K–128K), 40 greedy prompts, 500 MMLU questions across 56 subjects; token counts checked with the server's tokenizer |

Findings while preparing the harness:

1. **The engine does not return token ids unless asked.** `prompt_token_ids` and `token_ids` are present but `null` by default; requests need `return_token_ids: true`. Without it the perplexity step would fail and greedy comparison would silently return nothing. Fixed.
2. **`prompt_logprobs` on long prompts can exhaust memory.** It materialises chunk × vocabulary (~155K) logits on rank 0; with the production margin (≈2.9 GiB free per card, 8192-token chunks) that is unsafe, so the perplexity step runs only on a dedicated evaluation configuration (`--gpu-memory-utilization 0.80 --max-num-batched-tokens 2048`) and `ab_eval.py ppl` refuses to start without `AB_EVAL_SAFE=1`. The long-prompt call was **not** run against the production instance.
3. **Greedy decoding is not reproducible under live traffic.** Two runs of the same four prompts on the same FP8 instance gave one identical continuation out of four, with the median first divergence at token 7. Greedy agreement between BF16 and FP8 would therefore be dominated by engine noise, so it is reported only beside its same-configuration noise floor and does not carry the verdict.

The comparison itself is below.

## A/B evaluation (2026-11-04, 16:11–16:45 UTC)

Run by `scripts/run_ab_window.sh` on the production host with the public front door and the monitor cron paused; three restarts of the same image (BF16 evaluation configuration → FP8 evaluation configuration → FP8 production configuration, which was left serving). Full report: `results/ab-2026-11-05/report.md`; protocol and thresholds, fixed before data collection: `docs/ab-protocol.md`.

| | |
|---|---|
| Verdict | **PASS, 14 of 14 pre-registered checks** |
| Perplexity ratio (all / code / text) | 1.0002 / 0.9999 / 1.0005; CI upper end 1.0013 / 1.0014 / 1.0021 |
| Top-1 agreement (BF16 vs FP8; BF16 vs BF16) | 0.9777; 0.9790 |
| Mean abs change in log-probability | 0.0514; 0.0473 |
| MMLU (500) | 0.870 → 0.860; 8 lost, 3 gained; McNemar p = 0.23; 0 invalid |
| Greedy, identical continuations | 0.10 (BF16 vs FP8); 0.10 (BF16 vs BF16) |
| KV pool, evaluation configuration | 728,108 → 1,349,597 tokens (1.853×) |
| KV pool, production configuration (final) | 1,771,693 tokens, 3.38× at 512K |

Deviations from the protocol as first written, all decided before any result was seen: the candidate (FP8) arm was run once instead of twice, to shorten the window; the cron monitor was paused for the window so its probes would not join the evaluation batches.

Things worth knowing:

- The window was the first time the image ran on a GPU. Both arms started in 290–300 s; periodic checks of the engine logs during each arm (`Traceback`, out-of-memory, illegal memory access) found nothing. The containers' logs were removed when the next arm started, so the full logs are not retained.
- The BF16 run-to-run noise is large (2.1 % of positions change argmax, 0.047 nats mean change). FP8 sits just beyond it on both measures and far inside the pass limits.
- Speed, same pass: FP8 / BF16 was 0.98 single-stream, 0.96–0.98 at 4 streams, 0.98–1.00 at 8 streams, 0.94–1.00 at 2 streams (one measurement per cell). The second pass had 22–48 % higher aggregate throughput than the first in both configurations (single-stream unchanged), so cold and warm numbers must not be mixed.
- The driver reopened the front door the moment the final instance was healthy, before it was warm. I stopped the door again, ran `scripts/warmup.py` twice (first pass: 20K prefill 8.5 s, 65K prefill 30.7 s, mixed 9.4 s; second pass: 0.6 s, 0.9 s, 1.9 s; the second pass also benefits from prefix-cache hits on the same prompts, so JIT cost is not isolated), then reopened. `run_ab_window.sh` now warms the instance before reopening the door.
- Not controlled: the second BF16 greedy run reused the prefix cache of the first run. `prompt_logprobs` requests recompute the whole prompt, so perplexity and agreement were unaffected.

## Patch dry-run (13 upstream PRs vs v0.13.1 container sources)

`scripts/dryrun.sh` extracts the container's `*.py` files, then runs `git apply --check` for each PR.

| Applies cleanly | Does not apply |
|---|---|
| #56525, #54076, #47123, #56756, #57477, #54368 | #56524, #37152, #42359, #43569, #55898, #51540, #57128 |

"Applies" means the hunks match; it is not evidence that the change is semantically correct on this fork.

## Abort-and-retry stress test (`tests/abort_storm.py`)

Four workers × nine cycles of a ≈40K-token prefix; the first request is aborted ≈1.2 s in, then retried immediately and verified, then checked once more against the cache. Results: 0/36 failures on the unpatched BF16 service (MTP off), 0/36 on the patched MTP-3 service. **The test could not reproduce the failure it targets, so it cannot be used as evidence that the patches help.**

## What is not verified

1. The BF16 comparison covers one model, one GPU type and one corpus family; the token-level noise floor is high, so small effects are not resolvable (see A/B evaluation).
2. Kernel error is from synthetic data; real-latent behaviour and 11-layer accumulation are untested.
3. Long-session degeneration (#119) is not detectable by this monitor.
4. End-to-end cost of the non-fused quantise step is not isolated.
5. Image inputs were not exercised on the FP8 path.

## Raw files

| File | Content |
|---|---|
| `results/acc_fp8.log` | accuracy suite (includes the endpoint-format failures described above) |
| `results/needle400k_v2.log` | 400K needle runs |
| `results/baseline_unpatched.txt`, `results/patched_mtp3_storm.txt` | abort-and-retry stress, before / after |
| `results/patched_mtp3_pc_test.txt` | prefix-cache cold/warm and concurrent shared-prefix canary |
| `results/ref_outputs.json` | BF16 reference outputs recorded before the FP8 restart |
| `results/status.log`, `results/alerts.log` | monitor output |
