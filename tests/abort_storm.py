#!/usr/bin/env python3
"""取消后立刻重试 压力测试(针对 #56524/#56525 的触发场景)。
每个周期:构造一份唯一的 ~N 万 token 前缀(内含针:记录777的密码),
 1) 发流式请求,约 ABORT_AFTER 秒后直接断开连接(模拟客户端取消预填);
 2) 立刻重发同一请求,temperature=0,问针的密码;
 3) 输出里必须出现正确密码,否则记为污染/失败。
最后再用同前缀发一次(命中缓存)复核,防止"被污染的状态已被提交"。
注意: 本测试走 /v1/completions(纯文本),直连容器 127.0.0.1:18010,不经 nginx。
用法: python3 abort_storm.py --workers 4 --cycles 9 --ktok 40
"""
import argparse, http.client, json, random, string, threading, time, sys

HOST, PORT = "127.0.0.1", 18010

def make_prompt(seed, ktok):
    rnd = random.Random(seed)
    code = "QX-" + "".join(rnd.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    lines = []
    n = int(ktok * 1000 / 14)          # 每行约 14 token(粗估)
    needle_at = n // 3
    for i in range(n):
        w = " ".join("".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(3, 7))) for _ in range(5))
        if i == needle_at:
            lines.append(f"Record 777: the secret code of record 777 is {code}.")
        lines.append(f"Record {i:05d}: {w}.")
    body = "\n".join(lines)
    prompt = body + "\n\nQuestion: What is the secret code of record 777? Answer with the code only.\nAnswer: The secret code of record 777 is"
    return prompt, code

def post(prompt, stream, max_tokens=14, abort_after=None):
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    payload = json.dumps({"model": "glm-5.3-flash", "prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0, "stream": stream})
    c.request("POST", "/v1/completions", payload, {"Content-Type": "application/json"})
    if abort_after is not None:
        time.sleep(abort_after)
        c.sock.close(); c.close()      # 直接断开 = 客户端取消
        return None
    r = c.getresponse(); data = r.read().decode("utf-8", "replace"); c.close()
    if r.status != 200:
        return f"HTTP{r.status}:{data[:200]}"
    return json.loads(data)["choices"][0]["text"]

results = []
lock = threading.Lock()

def worker(wid, cycles, ktok, abort_after):
    for k in range(cycles):
        seed = f"{time.time_ns()}-{wid}-{k}"
        prompt, code = make_prompt(seed, ktok)
        t0 = time.time()
        try:
            post(prompt, True, abort_after=abort_after)               # 取消
            out1 = post(prompt, False)                                # 立刻重试
            out2 = post(prompt, False)                                # 命中缓存复核
        except Exception as e:
            out1 = out2 = f"EXC:{type(e).__name__}:{e}"
        ok1 = code in (out1 or ""); ok2 = code in (out2 or "")
        with lock:
            results.append((wid, k, ok1, ok2, (out1 or "")[:60].replace("\n", "\\n"), (out2 or "")[:60].replace("\n", "\\n"), time.time() - t0))
            print(f"w{wid} c{k} retry={'OK' if ok1 else 'BAD'} recheck={'OK' if ok2 else 'BAD'} {time.time()-t0:5.1f}s | {(out1 or '')[:50]!r}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=9)
    ap.add_argument("--ktok", type=int, default=40)
    ap.add_argument("--abort-after", type=float, default=1.2)
    a = ap.parse_args()
    ths = [threading.Thread(target=worker, args=(i, a.cycles, a.ktok, a.abort_after)) for i in range(a.workers)]
    t = time.time()
    [x.start() for x in ths]; [x.join() for x in ths]
    n = len(results); b1 = sum(1 for r in results if not r[2]); b2 = sum(1 for r in results if not r[3])
    print(f"\n汇总: 周期={n} 重试污染/失败={b1} 复核污染/失败={b2} 用时={time.time()-t:.0f}s")
    sys.exit(0 if (b1 == 0 and b2 == 0) else 1)
