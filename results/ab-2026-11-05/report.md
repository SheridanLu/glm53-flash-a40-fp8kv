# A/B report (reference = BF16 KV, candidate = FP8 KV)

## Perplexity and top-1 agreement

| group | docs | tokens | ppl ref | ppl cand | ratio | 95% CI | top-1 agree | mean abs dlogp |
|---|---|---|---|---|---|---|---|---|
| all | 30 | 686407 | 1.9074 | 1.9078 | 1.0002 | 0.9993-1.0013 | 0.9777 | 0.0514 |
| code | 15 | 325448 | 1.8421 | 1.8418 | 0.9999 | 0.9984-1.0014 | 0.9778 | 0.0539 |
| text | 15 | 360959 | 1.9683 | 1.9694 | 1.0005 | 0.9997-1.0021 | 0.9777 | 0.0492 |
| code@4K | 8 | 30917 | 1.8996 | 1.9026 | 1.0016 | 1.0005-1.0028 | 0.9788 | 0.0522 |
| code@16K | 4 | 61938 | 1.8444 | 1.8467 | 1.0012 | 1.0007-1.0020 | 0.9786 | 0.0492 |
| code@64K | 2 | 115842 | 1.9725 | 1.9695 | 0.9985 | - | 0.9732 | 0.0685 |
| code@128K | 1 | 116751 | 1.7061 | 1.7062 | 1.0001 | - | 0.9816 | 0.0423 |
| text@4K | 8 | 31625 | 2.1090 | 2.1108 | 1.0008 | 0.9997-1.0020 | 0.9760 | 0.0563 |
| text@16K | 4 | 66338 | 1.9349 | 1.9356 | 1.0003 | 0.9997-1.0008 | 0.9773 | 0.0537 |
| text@64K | 2 | 130895 | 2.0487 | 2.0518 | 1.0015 | - | 0.9767 | 0.0491 |
| text@128K | 1 | 132101 | 1.8768 | 1.8760 | 0.9996 | - | 0.9793 | 0.0452 |

reference vs reference (run-to-run noise): ratio 0.9997, top-1 agreement 0.9790, mean abs dlogp 0.0473

## MMLU (0-shot, logprob of A-D)

n=500  ref 0.870  cand 0.860  drop +0.010  lost 8 / gained 3  McNemar p=0.227  invalid 0.000

## Greedy continuations (128 tokens, informational)

n=40  identical 0.10  median first divergence at token 32
reference vs reference: identical 0.10  median first divergence 47

## Pre-registered checks

| check | value | result |
|---|---|---|
| ppl ratio all <= 1.010 | 1.0002 | PASS |
| ppl ratio all CI upper <= 1.020 | 1.0013 | PASS |
| top-1 agreement all >= 0.97 | 0.9777 | PASS |
| top-1 agreement all within 0.02 of ref-vs-ref noise (0.9790) | 0.9777 | PASS |
| ppl ratio code <= 1.010 | 0.9999 | PASS |
| ppl ratio code CI upper <= 1.020 | 1.0014 | PASS |
| top-1 agreement code >= 0.97 | 0.9778 | PASS |
| top-1 agreement code within 0.02 of ref-vs-ref noise (0.9793) | 0.9778 | PASS |
| ppl ratio text <= 1.010 | 1.0005 | PASS |
| ppl ratio text CI upper <= 1.020 | 1.0021 | PASS |
| top-1 agreement text >= 0.97 | 0.9777 | PASS |
| top-1 agreement text within 0.02 of ref-vs-ref noise (0.9788) | 0.9777 | PASS |
| MMLU drop <= 0.03 | +0.010 | PASS |
| MMLU invalid <= 0.02 | 0.000 | PASS |

**Verdict: PASS**
