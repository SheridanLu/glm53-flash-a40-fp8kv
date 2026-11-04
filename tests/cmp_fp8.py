import json, http.client, sys, time
HOST, PORT = "127.0.0.1", 18010
def ask(prompt, max_tokens):
    c = http.client.HTTPConnection(HOST, PORT, timeout=600)
    body = json.dumps({"model": "glm-5.3-flash", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"reasoning_effort": "low"}})
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse(); d = r.read().decode("utf-8","replace"); c.close()
    if r.status != 200: return f"HTTP{r.status}:{d[:200]}"
    return json.loads(d)["choices"][0]["text"]
needle_code = "REF-7K3Q"
lines = [f"Log {i:04d}: " + " ".join("abcdefgh"[i % 8] * 5 for _ in range(5)) + "." for i in range(560)]
lines[200] = f"Record 0421: the access code is {needle_code} and it must be quoted verbatim."
ctx = "\n".join(lines)
prompts = [("math","23 * 17 = ? Answer with the number only.\nAnswer:",32),
           ("code","Write a Python function add(a, b) returning their sum. Code only:\n```python\n",96),
           ("needle8k", ctx + "\n\nQuestion: What is the access code in record 0421? Quote it exactly.\nAnswer: The access code is",32)]
ref = json.load(open("/data/glm_test/fp8kv/ref_outputs.json"))
ok = True
for name, p, mt in prompts:
    t0 = time.time(); out = ask(p, mt); dt = time.time()-t0
    same = (out == ref[name]); f8ok = needle_code in out if name=="needle8k" else True
    ok &= f8ok
    print(f"{name:9s} 与BF16逐字相同={same} 针正确={f8ok} {dt:.1f}s")
    if not same:
        print("   FP8:", repr(out[:70])); print("   BF16:", repr(ref[name][:70]))
print("针检索判定:", "PASS" if ok else "FAIL")
