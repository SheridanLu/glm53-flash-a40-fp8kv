import re, collections
m = open("/tmp/m.txt").read().splitlines()

def tot(name):
    s = 0.0; f = False
    for l in m:
        if l.startswith(name + "{") or l.startswith(name + " "):
            s += float(l.rsplit(" ", 1)[1]); f = True
    return s if f else None

def buckets(name):
    d = collections.OrderedDict()
    for l in m:
        if l.startswith(name + "_bucket{"):
            le = re.search(r'le="([^"]*)"', l).group(1)
            d[le] = d.get(le, 0) + float(l.rsplit(" ", 1)[1])
    return d

def mean(name):
    s, c = tot(name + "_sum"), tot(name + "_count")
    return (s / c if c else None, c)

print("=== 平均值(自 09:36 重启起,含我少量测试请求)")
for n, lab in [("vllm:time_to_first_token_seconds", "首字延迟 TTFT"),
               ("vllm:inter_token_latency_seconds", "相邻 token 间隔 ITL"),
               ("vllm:request_time_per_output_token_seconds", "每请求每输出 token 耗时"),
               ("vllm:request_queue_time_seconds", "排队等待"),
               ("vllm:request_prefill_time_seconds", "预填耗时"),
               ("vllm:request_decode_time_seconds", "解码耗时"),
               ("vllm:request_inference_time_seconds", "推理总耗时(不含排队)")]:
    v, c = mean(n)
    if v is not None:
        print("%s: 平均 %.3f s (样本 %d)" % (lab, v, int(c)))

gen = tot("vllm:request_generation_tokens_sum"); cnt = tot("vllm:request_generation_tokens_count")
dec = tot("vllm:request_decode_time_seconds_sum")
if gen and dec:
    print("\n按请求加总: 共生成 %d tokens / 解码耗时 %.0f s = 平均每请求解码速度 %.1f tok/s" % (gen, dec, gen / dec))
if gen and cnt:
    print("平均每请求生成 %.0f tokens" % (gen / cnt))
v, c = mean("vllm:request_time_per_output_token_seconds")
if v:
    print("按 '每输出 token 耗时' 倒数: %.1f tok/s" % (1 / v))
v, c = mean("vllm:inter_token_latency_seconds")
if v:
    print("按 ITL 倒数: %.1f tok/s" % (1 / v))

for n in ("vllm:inter_token_latency_seconds", "vllm:request_time_per_output_token_seconds"):
    b = buckets(n)
    if b:
        total = list(b.values())[-1]
        print("\n%s 分布(累计占比)" % n)
        for le, c in b.items():
            if le in ("0.01", "0.025", "0.05", "0.1", "0.2", "0.5", "1.0", "2.5", "5.0", "+Inf"):
                rate = (1.0 / float(le)) if le != "+Inf" else 0
                print("  <=%ss: %.1f%%  (相当于 >=%d tok/s)" % (le, 100 * c / total, rate))

b = buckets("vllm:time_to_first_token_seconds")
if b:
    total = list(b.values())[-1]
    print("\nTTFT 分布(累计占比)")
    for le, c in b.items():
        if le in ("0.1", "0.5", "1.0", "2.5", "5.0", "10.0", "20.0", "40.0", "80.0", "160.0", "+Inf"):
            print("  <=%ss: %.1f%%" % (le, 100 * c / total))
