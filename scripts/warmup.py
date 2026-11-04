#!/usr/bin/env python3
"""Warm up a freshly started instance before it takes real traffic (stdlib only).

Triggers, once each, the paths that otherwise JIT-compile or autotune on the first real requests:
short and long prefill chunks (indexer / kpool Triton kernels), single and concurrent decode at several batch
sizes (cudagraph sizes + MTP draft/verify), and a repeated long prompt (prefix-cache hit and the hybrid-state
checkpoint path). Prints one line per step with its latency; a step much slower than its neighbours is a
compile that has now been paid for.

  python3 scripts/warmup.py [--host 127.0.0.1] [--port 18010] [--model glm-5.3-flash]
"""
import argparse
import http.client
import json
import random
import threading
import time


def lines(n, seed):
    r = random.Random(seed)
    return "\n".join("Line %06d: %s." % (i, " ".join("".join(r.choice("abcdefghijklmnop") for _ in range(5)) for _ in range(6))) for i in range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18010)
    ap.add_argument("--model", default="glm-5.3-flash")
    a = ap.parse_args()

    def call(prompt, max_tokens):
        c = http.client.HTTPConnection(a.host, a.port, timeout=1800)
        c.request("POST", "/v1/completions", json.dumps({"model": a.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        body = r.read()
        c.close()
        if r.status != 200:
            raise RuntimeError("HTTP %d: %r" % (r.status, body[:200]))

    def step(name, fn):
        t = time.time()
        fn()
        print("%-34s %6.2fs" % (name, time.time() - t), flush=True)

    long_a = lines(1500, 1)    # roughly 20K tokens
    long_b = lines(5000, 2)    # roughly 65K tokens
    step("short decode x3", lambda: [call("Say hello.", 32) for _ in range(3)])
    step("prefill ~2K", lambda: call(lines(150, 3), 16))
    step("prefill ~20K", lambda: call(long_a, 16))
    step("prefill ~65K", lambda: call(long_b, 16))
    step("repeat ~20K (prefix-cache hit)", lambda: call(long_a, 16))
    step("repeat ~65K (prefix-cache hit)", lambda: call(long_b, 16))
    for n in (2, 4, 8):
        def conc(n=n):
            ts = [threading.Thread(target=call, args=("Count from 1 to 40, one number per line.", 160)) for _ in range(n)]
            [t.start() for t in ts]
            [t.join() for t in ts]
        step("decode x%d concurrent" % n, conc)
    step("mixed: 20K prefill + 3 decodes", lambda: [t.join() for t in [t for t in (
        [threading.Thread(target=call, args=(lines(1500, 9), 16))] +
        [threading.Thread(target=call, args=("Write a haiku.", 96)) for _ in range(3)]) if (t.start() or True)]])
    print("warm-up finished")


if __name__ == "__main__":
    main()
