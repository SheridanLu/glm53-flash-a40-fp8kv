#!/usr/bin/env python3
"""Convert the MMLU test split (cais/mmlu, config "all") from parquet to JSONL for ab_eval.py.

Needs pyarrow (the eval harness itself is stdlib-only):  pip install pyarrow
Download (the host cannot reach huggingface.co; hf-mirror.com works):
  curl -L -o mmlu_all_test.parquet https://hf-mirror.com/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet
"""
import json
import sys

import pyarrow.parquet as pq

src, dst = sys.argv[1], sys.argv[2]
t = pq.read_table(src).to_pylist()
with open(dst, "w") as f:
    for r in t:
        f.write(json.dumps({"subject": r["subject"], "question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"])}, ensure_ascii=False) + "\n")
print("wrote %d questions to %s" % (len(t), dst))
