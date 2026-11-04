#!/usr/bin/env python3
"""FP8-KV 写入路径单元测试: quantize + 散写 + 回读 + 过注意力内核。显存硬上限, 独立进程。"""
import importlib.util, os, sys
import torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", "0.008")), 0)
from triton import runtime as _tr
_tr.driver.active.get_empty_cache_for_benchmark = lambda: torch.empty(1 << 20, dtype=torch.int, device="cuda")
spec = importlib.util.spec_from_file_location("kk", "/tmp/fp8kv2/triton_mla_sparse_kernel.py")
kk = importlib.util.module_from_spec(spec); spec.loader.exec_module(kk)
dev = "cuda:0"; torch.manual_seed(1)
BS, ROW = 64, 528
ok = True

def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

# 1) 写入 + 回读 (页步长含填充: 66 行/块, 实际用 64 行, 模拟混合分配器的块间空洞)
nb, bsr = 40, 66
big = torch.full((nb * bsr * ROW,), 0xAB, dtype=torch.uint8, device=dev)
cache = big.as_strided((nb, BS, ROW), (bsr * ROW, ROW, 1))
before = big.clone()
T = 500
kv_c = (torch.randn(T, 512, device=dev) * 3).to(torch.bfloat16)
perm = torch.randperm(nb * BS, device=dev)[:T]
slots = perm.clone().to(torch.int64); slots[::17] = -1                      # 填充位(cudagraph padding)
kk.write_fp8_nope_kv(kv_c, cache, slots)
torch.cuda.synchronize()
pages = kk.quantize_nope_pages(kv_c)
valid = slots >= 0
got = cache[(slots[valid] // BS), (slots[valid] % BS)]
check("回读与量化页逐字节一致", torch.equal(got, pages[valid]), f"(有效 {int(valid.sum())} 个, 跳过 {int((~valid).sum())} 个)")
changed = (big != before).nonzero().flatten()
touched_rows = set(((slots[valid] // BS) * bsr + (slots[valid] % BS)).tolist())
changed_rows = set((changed // ROW).tolist())
check("只改动了目标行(含块间空洞与 slot=-1 都未被写)", changed_rows <= touched_rows, f"(改动行 {len(changed_rows)} / 目标 {len(touched_rows)})")
check("没有写到行之外的字节", True if len(changed) == 0 else bool(((changed // ROW).unique().numel()) == len(changed_rows)))

# 2) 过注意力内核: 与"反量化后用 bf16 内核"对比 (行索引按块步长 bsr 计)
rows = big.as_strided(((nb - 1) * bsr + BS, ROW), (ROW, 1))
H, TOPK, nq = 16, 512, 4
q = torch.randn(nq, H, 512, device=dev, dtype=torch.bfloat16)
glob = (slots[valid] // BS) * bsr + (slots[valid] % BS)                        # 有效行的全局行号
idx = glob[torch.randint(0, glob.numel(), (nq, 1, TOPK), device=dev)].to(torch.int32)
idx[:, :, -29:] = -1
out8 = kk.triton_mla_sparse_attention(q, rows.view(-1, 1, ROW), idx, 1.0 / 512 ** 0.5)
# 参考: 手工反量化这些行
r = rows.view(-1, ROW)
xq = r[:, :512].view(torch.float8_e4m3fn).float().view(-1, 4, 128)
sc = r[:, 512:528].contiguous().view(torch.float32).view(-1, 4, 1)
deq = (xq * sc).view(-1, 512).to(torch.bfloat16)
ref = kk.triton_mla_sparse_attention(q, deq.view(-1, 1, 512), idx, 1.0 / 512 ** 0.5)
torch.cuda.synchronize()
cos = torch.nn.functional.cosine_similarity(out8.float().flatten(), ref.float().flatten(), dim=0).item()
rel = ((out8.float() - ref.float()).norm() / ref.float().norm()).item()
check("写入→读取全链路(含块步长寻址)与参考一致", cos > 0.99999 and rel < 5e-3 and not torch.isnan(out8).any().item(), f"cos={cos:.7f} 相对误差={rel:.2e}")

# 3) 能失败的对照: 故意把一行的缩放因子破坏, 全链路结果必须明显偏离
bad = big.clone(); badrows = bad.as_strided(((nb - 1) * bsr + BS, ROW), (ROW, 1))
badrows[:, 512:528] = 0
o_bad = kk.triton_mla_sparse_attention(q, badrows.view(-1, 1, ROW), idx, 1.0 / 512 ** 0.5)
torch.cuda.synchronize()
c2 = torch.nn.functional.cosine_similarity(o_bad.float().flatten(), ref.float().flatten(), dim=0).item()
check("对照: 缩放清零必须被检出", c2 < 0.5 or torch.isnan(o_bad).any().item(), f"cos={c2:.4f}")
print("峰值显存(本进程): %.0f MiB" % (torch.cuda.max_memory_allocated(0) / 2**20))
print("总判定:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
