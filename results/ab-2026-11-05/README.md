# A/B evaluation, BF16 KV vs FP8 KV

2026-11-04, 16:11–16:45 UTC (2026-11-05 local), 8× A40. Protocol and pass criteria: [`../../docs/ab-protocol.md`](../../docs/ab-protocol.md).

| File | Content |
|---|---|
| `report.md`, `report.json` | the comparison (all numbers in the README come from `report.json`) |
| `window.log` | the driver's log, including the full per-document perplexity run lines |
| `speed_ref.txt`, `speed_cand.txt` | idle-host speed benchmark per arm (two passes each) |
| `kv_ref.txt`, `kv_cand.txt` | engine KV-pool lines per arm (evaluation configuration) |
| `prompts_manifest.json` | sha256 of the prompt sets both arms read |
| `raw/` | per-token log-probabilities and argmax ids (`*_ppl.jsonl.gz`), greedy continuations, MMLU answers. `ref1`/`ref2` = BF16 runs 1 and 2, `cand1` = FP8 |

Re-analyse without a GPU: `python3 tests/ab_eval.py compare --ref <dir with ppl/greedy/mmlu.jsonl> ...` after `gunzip`-ing the raw files into three directories.
