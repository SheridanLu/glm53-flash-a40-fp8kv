#!/usr/bin/env python3
"""FP8 KV 绝对精度测试: 答案可自动判定, 不依赖 BF16 参考。
 1) 针检索: 不同上下文长度 x 不同深度
 2) 算术: 两位数乘加(可校验)
 3) 指令遵循: 严格格式(只输出一个词)
 4) 代码: 生成的函数用 eval 验证正确性
"""
import json, http.client, random, re, sys, time
HOST, PORT = "127.0.0.1", 18010
def ask(prompt, max_tokens=40, temp=0):
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    body = json.dumps({"model": "glm-5.3-flash", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": temp, "chat_template_kwargs": {"reasoning_effort": "low"}})
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse(); d = r.read().decode("utf-8", "replace"); c.close()
    if r.status != 200: return f"HTTP{r.status}"
    return json.loads(d)["choices"][0]["text"]

def filler(n_tokens, rnd):
    out = []
    for i in range(n_tokens // 14):
        out.append(f"Log {i:05d}: " + " ".join("".join(rnd.choice("abcdefghijklmnop") for _ in range(5)) for _ in range(5)) + ".")
    return "\n".join(out)

res = []
# 1) 针检索
for ktok in (8, 32, 128):
    for depth in (0.1, 0.5, 0.9):
        rnd = random.Random(f"{ktok}-{depth}")
        code = "KV-" + "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(6))
        body = filler(ktok * 1000, rnd).split("\n")
        pos = max(1, int(len(body) * depth))
        body[pos] = f"Record 9999: the vault code is {code} and it must be quoted exactly."
        ctx = "\n".join(body)
        t0 = time.time()
        out = ask(ctx + "\n\nQuestion: What is the vault code in record 9999? Quote it exactly.\nAnswer: The vault code is", 24)
        ok = code in out
        res.append(("needle", f"{ktok}k/depth{int(depth*100)}", ok, time.time() - t0))
        print(f"needle {ktok}k depth{int(depth*100)}%: {'OK' if ok else 'MISS'} {time.time()-t0:.0f}s {out[:40]!r}", flush=True)

# 2) 算术
ok_n = 0
for i in range(12):
    rnd = random.Random(i)
    a, b, c = rnd.randint(11, 99), rnd.randint(11, 99), rnd.randint(100, 999)
    out = ask(f"Compute {a} * {b} + {c}. Answer with the number only.\nAnswer:", 16)
    m = re.search(r"-?\d+", out.replace(",", ""))
    good = bool(m) and int(m.group()) == a * b + c
    ok_n += good
res.append(("arith", "12题", ok_n == 12, None))
print(f"算术: {ok_n}/12 正确")

# 3) 指令遵循: 只能输出这个词
ok_f = 0
for word in ("BANANA", "SEVENTEEN", "PURPLE"):
    out = ask(f"Reply with exactly this word and nothing else: {word}", 24)
    ok_f += out.strip().startswith(word)
res.append(("format", "3题", ok_f == 3, None))
print(f"指令遵循: {ok_f}/3")

# 4) 代码正确性
cases = [("add", "def add(a, b):", lambda f: f(2, 3) == 5),
         ("square", "def square(x):", lambda f: f(7) == 49),
         ("rev", "def rev(s):", lambda f: f("abc") == "cba")]
ok_c = 0
for name, sig, check in cases:
    out = ask(f"Write a Python function {name}. Output only the code, no explanation.\n{sig}\n", 160)
    src = out.split("```")[0] if "```" not in out else out.split("```")[1]
    src = src.replace("python", "", 1)
    try:
        ns = {}; exec(src, ns)
        fn = ns[name]
        ok_c += bool(check(fn))
    except Exception as e:
        print(f"  {name} 执行失败: {type(e).__name__}")
res.append(("code", "3题", ok_c == 3, None))
print(f"代码: {ok_c}/3 可执行且正确")

total = sum(1 for r in res if r[2]); n = len(res)
print(f"\n汇总: {total}/{n} 项全对 | 明细: " + " ".join(f"{r[1]}{'✓' if r[2] else '✗'}" for r in res))
json.dump([{"kind": r[0], "case": r[1], "pass": r[2]} for r in res], open("/data/glm_test/fp8kv/accuracy_fp8.json", "w"), ensure_ascii=False, indent=1)
