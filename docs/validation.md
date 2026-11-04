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

## Patch dry-run (13 upstream PRs vs v0.13.1 container sources)

`scripts/dryrun.sh` extracts the container's `*.py` files, then runs `git apply --check` for each PR.

| Applies cleanly | Does not apply |
|---|---|
| #56525, #54076, #47123, #56756, #57477, #54368 | #56524, #37152, #42359, #43569, #55898, #51540, #57128 |

"Applies" means the hunks match; it is not evidence that the change is semantically correct on this fork.

## Abort-and-retry stress test (`tests/abort_storm.py`)

Four workers × nine cycles of a ≈40K-token prefix; the first request is aborted ≈1.2 s in, then retried immediately and verified, then checked once more against the cache. Results: 0/36 failures on the unpatched BF16 service (MTP off), 0/36 on the patched MTP-3 service. **The test could not reproduce the failure it targets, so it cannot be used as evidence that the patches help.**

## What is not verified

1. No same-prompt BF16 vs FP8 comparison on real text.
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
