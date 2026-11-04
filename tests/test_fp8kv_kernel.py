#!/usr/bin/env python3
"""FP8-KV 内核级测试(NoPE: 512 潜向量 -> 528B 页)。在服务容器里的独立进程跑,显存硬上限。
比较三者:  A=bf16 内核(原始 bf16 潜向量)   B=fp8 内核(编码后的 528B 页)   C=bf16 内核(用 fp8 反量化后的 bf16 潜向量)
  B vs C : 应几乎逐位一致 -> 证明内核里的解码/缩放逻辑正确(与量化误差无关)
  B vs A : 量化误差 -> FP8 KV 的精度代价
"""
import importlib.util, sys, time, os
import torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", "0.02")), 0)   # ~0.9 GiB 硬上限, 只会让本进程 OOM
spec = importlib.util.spec_from_file_location("k8", "/tmp/fp8kv/triton_mla_sparse_kernel_fp8.py")
k8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k8)
attn = k8.triton_mla_sparse_attention
dev = "cuda:0"
# Triton 自动调优默认分配 256MiB 的缓存清理缓冲区; 共享卡上余量小, 换成 4MiB(只影响调优计时的偏置, 不影响数值正确性)
from triton import runtime as _tr
_tr.driver.active.get_empty_cache_for_benchmark = lambda: torch.empty(1 << 20, dtype=torch.int, device="cuda")
torch.manual_seed(0)

def encode(x):                       # x: [N,512] bf16 -> pages [N,528] uint8, deq [N,512] bf16
    xf = x.float().view(-1, 4, 128)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 448.0).float()                              # [N,4,1]
    q = (xf / scale).to(torch.float8_e4m3fn)
    deq = (q.float() * scale).view(-1, 512).to(torch.bfloat16)
    pages = torch.cat([q.view(torch.uint8).view(-1, 512), scale.view(-1, 4).contiguous().view(torch.uint8).view(-1, 16)], dim=1)
    assert pages.shape[1] == 528
    return pages.contiguous(), deq

def metrics(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return cos, (a - b).abs().max().item(), ((a - b).norm() / b.norm()).item()

def make(N, H, T, bs, outliers):
    x = torch.randn(N, 512, device=dev)
    if outliers:                                      # 少数通道放大, 考验分块缩放
        x[:, torch.randint(0, 512, (6,))] *= 20
    x = x.to(torch.bfloat16)
    q = torch.randn(bs, H, 512, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, N, (bs, 1, T), device=dev, dtype=torch.int32)
    idx[:, :, -37:] = -1                              # 带无效填充位
    return x, q, idx

ok_all = True
sm = 1.0 / (512 ** 0.5)
print(f"GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")
for (N, H, T, bs, outl) in [(8192, 16, 512, 1, False), (16384, 16, 2048, 1, False), (16384, 16, 2048, 8, True), (16384, 16, 2048, 1, True)]:
    x, q, idx = make(N, H, T, bs, outl)
    pages, deq = encode(x)
    A = attn(q, x.view(N, 1, 512), idx, sm)
    C = attn(q, deq.view(N, 1, 512), idx, sm)
    B = attn(q, pages, idx, sm)
    torch.cuda.synchronize()
    bc = metrics(B, C); ba = metrics(B, A)
    pass_dec = bc[0] > 0.99999 and bc[2] < 5e-3
    ok_all &= pass_dec and not torch.isnan(B).any().item()
    print(f"N={N} H={H} topk={T} bs={bs} outliers={outl} | B-vs-C(解码正确性) cos={bc[0]:.7f} 相对误差={bc[2]:.2e} maxabs={bc[1]:.2e} {'PASS' if pass_dec else 'FAIL'} | B-vs-A(量化代价) cos={ba[0]:.6f} 相对误差={ba[2]:.2e} | NaN={torch.isnan(B).any().item()}")

# 能失败的对照: 故意把缩放因子清零, B 必须明显偏离 C, 否则测试没有判别力
x, q, idx = make(16384, 16, 2048, 1, False); pages, deq = encode(x)
bad = pages.clone(); bad[:, 512:528] = 0
Bb = attn(q, bad, idx, sm); Cc = attn(q, deq.view(-1, 1, 512), idx, sm); torch.cuda.synchronize()
m = metrics(Bb, Cc)
print(f"对照(缩放清零, 必须判 FAIL): cos={m[0]:.4f} 相对误差={m[2]:.2e} -> {'判别力OK(检出)' if (m[0] < 0.99 or m[2] > 0.1 or torch.isnan(Bb).any().item()) else '无判别力!'}")

# 粗略速度(服务在同卡上跑, 仅看相对值, 噪声大)
def bench(fn, n=30):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e6
for bs in (1, 8):
    x, q, idx = make(16384, 16, 2048, bs, False); pages, _ = encode(x)
    tb = bench(lambda: attn(q, x.view(-1, 1, 512), idx, sm)); tf = bench(lambda: attn(q, pages, idx, sm))
    print(f"[速度,同卡有服务负载,仅参考] bs={bs} topk=2048: bf16 {tb:.0f}us  fp8 {tf:.0f}us  fp8/bf16={tf/tb:.2f}x")
print("峰值显存(本进程): %.0f MiB" % (torch.cuda.max_memory_allocated(0) / 2**20))
print("总判定:", "PASS" if ok_all else "FAIL")
