#!/usr/bin/env python3
"""录 BF16 参考输出: 3 个固定提示(算术 / 代码 / 8K 针检索), 温度 0, 固定 max_tokens。
输出 ref_bf16.json, 之后 FP8 实例用同一脚本同参数对比。"""
import json, sys, http.client, time
HOST, PORT = "127.0.0.1", 18010
def ask(prompt, max_tokens):
    c = http.client.HTTPConnection(HOST, PORT, timeout=600)
    body = json.dumps({"model": "glm-5.3-flash", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}})
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse(); d = r.read().decode("utf-8", "replace"); c.close()
    if r.status != 200: return f"HTTP{r.status}:{d[:200]}"
    return json.loads(d)["choices"][0]["text"]

needle_code = "REF-7K3Q"
lines = [f"Log {i:04d}: " + " ".join("abcdefgh"[i % 8] * 5 for _ in range(5)) + "." for i in range(560)]
lines[200] = f"Record 0421: the access code is {needle_code} and it must be quoted verbatim."
ctx = "\n".join(lines)
prompts = [
    ("math", "23 * 17 = ? Answer with the number only.\nAnswer:", 32),
    ("code", "Write a Python function add(a, b) returning their sum. Code only:\n```python\n", 96),
    ("needle8k", ctx + "\n\nQuestion: What is the access code in record 0421? Quote it exactly.\nAnswer: The access code is", 32),
]
out = {}
t0 = time.time()
for name, p, mt in prompts:
    out[name] = ask(p, mt)
    print(f"{name}: {out[name][:80]!r}", flush=True)
json.dump(out, open("/data/glm_test/fp8kv/ref_outputs.json", "w"), ensure_ascii=False, indent=1)
print("needle_expected:", needle_code, "| needle_ok:", needle_code in out["needle8k"], f"| 用时 {time.time()-t0:.0f}s")
