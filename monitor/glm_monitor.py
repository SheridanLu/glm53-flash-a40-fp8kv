#!/usr/bin/env python3
"""GLM 服务监测(只读,不重启、不回退、不记录用户内容)。
cron 每 5 分钟跑一次:容器/健康/日志报错/缓存命中率/MTP 接受长度/nginx 状态码/金丝雀探针。
输出: /data/glm_test/monitor/status.log (每次一行)  alerts.log (仅异常)
用法: glm_monitor.py            正常运行
      glm_monitor.py --selftest 自检:故意用错误的期望值,必须报出金丝雀失败(结果只打印,不写 alerts.log)
卸载: crontab -l | grep -v glm_monitor | crontab -
"""
import json, os, random, re, string, subprocess, sys, threading, time, http.client, fcntl

os.environ["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
DIR = "/data/glm_test/monitor"
STATE = f"{DIR}/state.json"; STATUS = f"{DIR}/status.log"; ALERTS = f"{DIR}/alerts.log"
HOST, PORT = "127.0.0.1", 18010
CONTAINER, FRONT = "glm53flash", "glm-front"
SELFTEST = "--selftest" in sys.argv
os.makedirs(DIR, exist_ok=True)
now = time.time()
iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
alerts = []                      # (级别, 文本)
def alert(level, text): alerts.append((level, text))

def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except Exception as e:
        return f"ERR {type(e).__name__}"

# 单实例锁,避免上一轮没跑完又起一轮
lk = open(f"{DIR}/.lock", "w")
try:
    fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(0)

state = {}
if os.path.exists(STATE):
    try: state = json.load(open(STATE))
    except Exception: state = {}
last = state.get("ts", now - 360)
since = iso(max(last, now - 1800))

summary = {}
# 1. 容器
ins = sh(f"docker inspect {CONTAINER} --format '{{{{.State.Status}}}}|{{{{.State.StartedAt}}}}|{{{{.RestartCount}}}}'").strip()
parts = ins.split("|")
if len(parts) != 3 or parts[0] != "running":
    alert("CRIT", f"容器 {CONTAINER} 不在运行: {ins[:120]}")
    cstat = "down"
else:
    cstat = "up"
    if state.get("started_at") and state["started_at"] != parts[1]:
        alert("WARN", f"容器已重启/重建: {state['started_at']} -> {parts[1]}")
    state["started_at"] = parts[1]
summary["container"] = cstat

hit_rate = acc_len = None; running = None; canary = "skip"; canary_ms = None
if cstat == "up":
    # 2. 健康
    code = sh(f"curl -s -o /dev/null -m 10 -w '%{{http_code}}' http://{HOST}:{PORT}/health").strip()
    summary["health"] = code
    if code != "200": alert("CRIT", f"/health 返回 {code}")
    # 3. 指标(按标签求和,与上一轮做差)
    m = sh(f"curl -s -m 15 http://{HOST}:{PORT}/metrics")
    def total(name):
        s = 0.0; found = False
        for ln in m.splitlines():
            if ln.startswith(name + "{") or ln.startswith(name + " "):
                try: s += float(ln.rsplit(" ", 1)[1]); found = True
                except Exception: pass
        return s if found else None
    cur = {k: total("vllm:" + k) for k in ["prefix_cache_hits_total", "prefix_cache_queries_total",
           "spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total", "num_requests_running", "num_requests_waiting"]}
    running = cur["num_requests_running"]
    prev = state.get("counters", {})
    def delta(k):
        if cur.get(k) is None or prev.get(k) is None or cur[k] < prev[k]: return None   # 重启后计数清零则跳过
        return cur[k] - prev[k]
    dq, dh = delta("prefix_cache_queries_total"), delta("prefix_cache_hits_total")
    if dq and dq >= 20000:
        hit_rate = dh / dq
        if hit_rate < 0.5: alert("INFO", f"前缀缓存命中率偏低: {hit_rate:.1%} (本轮 {int(dq)} token 查询)")
    dd, da = delta("spec_decode_num_drafts_total"), delta("spec_decode_num_accepted_tokens_total")
    if dd and dd >= 200:
        acc_len = 1 + da / dd
        if acc_len < 1.8: alert("WARN", f"MTP 平均接受长度偏低: {acc_len:.2f} (基线约2.3, #119事故时曾降到1.5~1.8)")
    state["counters"] = {k: v for k, v in cur.items() if v is not None}
    if cur["num_requests_waiting"] and cur["num_requests_waiting"] >= 8:
        alert("INFO", f"排队请求较多: waiting={int(cur['num_requests_waiting'])}")
    # 4. 容器日志报错(只看自上一轮以来)
    lg = sh(f"docker logs --since {since} {CONTAINER} 2>&1 | grep -aE 'Traceback|illegal memory access|AssertionError|CUDA error|NCCL.*(error|Error)|EngineDead|out of memory|OutOfMemory' | head -5", 90)
    lines = [l for l in lg.splitlines() if l.strip()]
    summary["log_errors"] = len(lines)
    if lines: alert("CRIT", f"日志出现报错 {len(lines)}+ 条, 首条: {lines[0][:200]}")
    # 5. nginx 状态码
    ng = sh(f"docker logs --since {since} {FRONT} 2>&1", 60)
    codes = re.findall(r'" (\d{3}) \d+ "', ng)
    c5 = sum(1 for c in codes if c.startswith("5")); c499 = sum(1 for c in codes if c == "499"); c401 = sum(1 for c in codes if c == "401")
    summary.update(req=len(codes), s5xx=c5, s499=c499)
    if "No such container" in ng: alert("WARN", f"前置代理 {FRONT} 不存在")
    if c5 >= 3: alert("CRIT", f"nginx 5xx {c5} 次 (本轮共 {len(codes)} 个请求)")
    # 6. 金丝雀探针
    if running is not None and running >= 12:
        canary = "skip(忙)"
    elif code == "200":
        rnd = random.Random(20261104)
        pre = "\n".join(f"Log {i:04d}: " + " ".join("".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(3, 7))) for _ in range(5)) + "." for i in range(450))
        codes_ = ["ZK-" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)) for _ in range(2)]
        outs = [None, None]; t0 = time.time()
        def probe(i):
            try:
                c = http.client.HTTPConnection(HOST, PORT, timeout=150)
                p = pre + f"\n\nNote: the session code for this check is {codes_[i]}.\nQuestion: What is the session code? Answer with the code only.\nAnswer: The session code is"
                c.request("POST", "/v1/completions", json.dumps({"model": "glm-5.3-flash", "prompt": p, "max_tokens": 20, "temperature": 0}), {"Content-Type": "application/json"})
                r = c.getresponse(); d = r.read().decode("utf-8", "replace"); c.close()
                outs[i] = json.loads(d)["choices"][0]["text"] if r.status == 200 else f"HTTP{r.status}"
            except Exception as e:
                outs[i] = f"EXC:{type(e).__name__}"
        ths = [threading.Thread(target=probe, args=(i,)) for i in range(2)]
        [t.start() for t in ths]; [t.join() for t in ths]
        canary_ms = int((time.time() - t0) * 1000)
        expect = ["ZK-WRONGCODE", "ZK-WRONGCODE"] if SELFTEST else codes_
        bad = []
        for i in range(2):
            o = outs[i] or ""
            if expect[i] not in o: bad.append(f"探针{i}期望{expect[i]}实得{o[:60]!r}")
            elif codes_[1 - i] in o and not SELFTEST: bad.append(f"探针{i}串入了另一路的暗号")
        canary = "FAIL" if bad else "ok"
        if bad: alert("CRIT", "金丝雀失败: " + "; ".join(bad))
        if canary_ms > 60000: alert("WARN", f"金丝雀探针耗时 {canary_ms/1000:.0f}s (正常应在几秒内)")

if not SELFTEST:                                # 自检不改状态,免得污染下一轮的差值
    state["ts"] = now
    json.dump(state, open(STATE, "w"))
line = (f"{iso(now)} alerts={len(alerts)} " + " ".join(f"{k}={v}" for k, v in summary.items()) +
        f" running={None if running is None else int(running)} hit={'-' if hit_rate is None else f'{hit_rate:.0%}'}"
        f" acc={'-' if acc_len is None else f'{acc_len:.2f}'} canary={canary}" + (f"({canary_ms}ms)" if canary_ms else ""))
if SELFTEST:
    print("[SELFTEST]", line)
    for lv, tx in alerts: print("[SELFTEST-ALERT]", lv, tx)
    sys.exit(0 if any("金丝雀失败" in t for _, t in alerts) else 3)
open(STATUS, "a").write(line + "\n")
if alerts:
    with open(ALERTS, "a") as f:
        for lv, tx in alerts: f.write(f"{iso(now)} [{lv}] {tx}\n")
for fn in (STATUS, ALERTS):                     # 简单轮转: 超过 2MB 只留最后 5000 行
    try:
        if os.path.getsize(fn) > 2_000_000:
            ls = open(fn).read().splitlines()[-5000:]; open(fn, "w").write("\n".join(ls) + "\n")
    except Exception: pass
