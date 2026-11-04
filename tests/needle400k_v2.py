#!/usr/bin/env python3
"""40 万 token 针检索(FP8 KV):先用 tokenizer 精确控制长度,再测三个深度。
用法: python3 needle400k_v2.py [目标token=400000]
"""
import json, http.client, os, random, sys, time
from transformers import AutoTokenizer
HOST, PORT = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "18010"))
TOK_PATH = os.environ.get("TOK_PATH", "/model")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
LIMIT = 460_000                     # 留出输出与模板余量,远低于 524288
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

def make_lines(n):
    return [f"Log {i:06d}: " + " ".join("".join(random.Random(i * 7 + j).choice("abcdefghijklmnop") for _ in range(5)) for j in range(5)) + "." for i in range(n)]

def build(depth):
    lo, hi = 1, 40_000
    while lo < hi:                   # 二分出使 token 数接近 TARGET 的行数
        mid = (lo + hi + 1) // 2
        n_tok = len(tok.encode("\n".join(make_lines(mid))))
        if n_tok <= TARGET: lo = mid
        else: hi = mid - 1
    lines = make_lines(lo)
    rnd = random.Random(f"needle-{depth}-{lo}")
    code = "VX-" + "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(8))
    pos = max(1, int(lo * depth))
    lines[pos] = f"Record 8888: the audit token is {code} and it must be reproduced exactly."
    prompt = "\n".join(lines) + "\n\nQuestion: What is the audit token in record 8888? Quote it exactly.\nAnswer: The audit token is"
    n_meas = len(tok.encode(prompt))
    assert n_meas <= LIMIT, f"prompt {n_meas} 超过 {LIMIT}"
    return prompt, code, n_meas

def ask(prompt, max_tokens=24):
    c = http.client.HTTPConnection(HOST, PORT, timeout=2400)
    c.request("POST", "/v1/completions",
              json.dumps({"model": "glm-5.3-flash", "prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0, "stream": True,
                          "chat_template_kwargs": {"reasoning_effort": "low"}}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return None, None, r.read().decode("utf-8", "replace")[:300], 0
    t0 = time.time(); first = None; text = ""
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        try: d = json.loads(line[5:])
        except Exception: continue
        ch = d.get("choices")
        if ch:
            piece = ch[0].get("text") or ""
            if piece:
                if first is None: first = time.time() - t0
                text += piece
    c.close()
    return text, first, None, time.time() - t0

res = []
for depth in (0.1, 0.5, 0.9):
    prompt, code, n_meas = build(depth)
    print(f"[depth{int(depth*100)}%] 提示实测 {n_meas} tokens, 开始请求...", flush=True)
    text, first, err, total = ask(prompt)
    ok = bool(text) and code in text
    res.append({"depth": depth, "prompt_tokens": n_meas, "ok": ok, "ttft": first, "total": total})
    print(f"  -> {'OK' if ok else 'MISS'} 首字={first if first else '-'}s 总耗时={total:.0f}s 输出={(text or err or '')[:50]!r}", flush=True)

n_ok = sum(1 for r in res if r["ok"])
print(f"\n汇总: {n_ok}/3 答对 @ 约 {TARGET//1000}K tokens")
json.dump(res, open("/data/glm_test/fp8kv/needle400k_v2.json", "w"), ensure_ascii=False, indent=1)
sys.exit(0 if n_ok == 3 else 1)
