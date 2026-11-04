#!/usr/bin/env python3
"""FP8-KV(NoPE, 528B/token)接线补丁: 在副本树 /data/glm_test/tree 的 fp8kv 分支上修改 4 个文件。不触碰运行中的容器。
 1. triton_mla_sparse_kernel.py : 换成已移植的 fp8 解码内核 + 追加 量化写入(torch 编码 + Triton 散写, 对 slot<0 跳过)
 2. xpu_mla_sparse.py           : 量化 KV 的拒绝改为可被子类声明支持
 3. triton_mla_sparse.py        : 后端声明 fp8 dtype; Impl 声明支持、warmup 预热 fp8 变体、重写 do_kv_cache_update
 4. mla_attention.py            : TRITON_MLA_SPARSE 的量化 dtype 规范化为 fp8_ds_mla; NoPE 页字节 528
"""
import os, subprocess, py_compile, shutil
T = "/data/glm_test/tree"
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, cwd=T)
sh("git checkout -q patched 2>/dev/null")
# 先把已部署的"补丁版"(#56524移植+#56525)固化成提交, 之后的接线才有干净的 diff; 重复运行时用 -f 丢弃上次接线
if sh("git rev-parse --abbrev-ref HEAD").stdout.strip() == "patched" and sh("git status --short").stdout.strip():
    sh("git add -A"); sh("git -c user.email=a@b -c user.name=t commit -qm 'patched: 56524-port + 56525 (deployed set)'")
sh("git checkout -q -f patched"); sh("git branch -D fp8kv"); r = sh("git checkout -q -b fp8kv"); assert r.returncode == 0, r.stderr

def edit(path, pairs):
    p = f"{T}/{path}"; s = open(p).read()
    for old, new, cnt in pairs:
        assert s.count(old) == cnt, f"{path}: 锚点出现 {s.count(old)} 次(期望 {cnt}): {old[:80]!r}"
        s = s.replace(old, new)
    open(p, "w").write(s); py_compile.compile(p, doraise=True)

# ---- 1. 内核文件: 用已移植版本, 追加写入函数 ----
K = "vllm/v1/attention/ops/triton_mla_sparse_kernel.py"
shutil.copy("/data/glm_test/fp8kv/triton_mla_sparse_kernel_fp8.py", f"{T}/{K}")
with open(f"{T}/{K}", "a") as f:
    f.write('''

# --------------------------------------------------------------------------
# fp8_ds_mla (NoPE) 缓存写入: 先用 torch 做分块量化(sm86 的 torch fp8 转换可用, Triton 的不行),
# 再用 Triton 把 528B 页散写进分页缓存; slot<0 (cudagraph 填充) 跳过, 全程无同步, 可被 cudagraph 捕获。
# --------------------------------------------------------------------------
_FP8_NOPE_ROW = 528


@triton.jit
def _fp8_nope_store_kernel(
    pages_ptr,
    cache_ptr,
    slot_ptr,
    block_size,
    stride_block,
    stride_row,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot >= 0:
        blk = slot // block_size
        off = slot % block_size
        base = blk * stride_block + off * stride_row
        offs = tl.arange(0, BLOCK)
        m = offs < ROW
        v = tl.load(pages_ptr + t.to(tl.int64) * ROW + offs, mask=m)
        tl.store(cache_ptr + base + offs, v, mask=m)


def quantize_nope_pages(kv_c: torch.Tensor) -> torch.Tensor:
    """[T,512] bf16 -> [T,528] uint8: 512 x fp8e4m3 + 4 x fp32 分块缩放(amax/448, 每128元素)。"""
    n = kv_c.shape[0]
    xf = kv_c.float().view(n, 4, 128)
    scale = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    q = (xf / scale).to(torch.float8_e4m3fn)
    return torch.cat(
        [
            q.view(torch.uint8).view(n, 512),
            scale.view(n, 4).contiguous().view(torch.uint8).view(n, 16),
        ],
        dim=1,
    ).contiguous()


def write_fp8_nope_kv(
    kv_c: torch.Tensor, cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    """cache: [num_blocks, block_size, 528] uint8(按物理步长寻址); slot_mapping: [T] int."""
    n = kv_c.shape[0]
    if n == 0:
        return
    assert kv_c.shape[1] == 512 and cache.dtype == torch.uint8
    assert cache.shape[2] == _FP8_NOPE_ROW and cache.stride(2) == 1
    pages = quantize_nope_pages(kv_c)
    _fp8_nope_store_kernel[(n,)](
        pages,
        cache,
        slot_mapping,
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        ROW=_FP8_NOPE_ROW,
        BLOCK=1024,
    )
''')
py_compile.compile(f"{T}/{K}", doraise=True)

# ---- 2. xpu_mla_sparse.py ----
edit("vllm/v1/attention/backends/mla/xpu_mla_sparse.py", [(
    "        if is_quantized_kv_cache(self.kv_cache_dtype):\n            raise NotImplementedError(\"FP8 kv is not supported with XPU MLA Sparse yet\")",
    "        if is_quantized_kv_cache(self.kv_cache_dtype) and not getattr(\n            self, \"supports_fp8_ds_mla_nope\", False\n        ):\n            raise NotImplementedError(\"FP8 kv is not supported with XPU MLA Sparse yet\")",
    1)])

# ---- 3. triton_mla_sparse.py ----
edit("vllm/v1/attention/backends/mla/triton_mla_sparse.py", [
    ("from vllm.v1.attention.ops.triton_mla_sparse_kernel import (\n    KV_SPLITS_CANDIDATES,\n    triton_mla_sparse_attention,\n)",
     "from vllm.v1.attention.ops.triton_mla_sparse_kernel import (\n    KV_SPLITS_CANDIDATES,\n    triton_mla_sparse_attention,\n    write_fp8_nope_kv,\n)", 1),
    ("class TritonMLASparseImpl(XPUMLASparseImpl):\n",
     "class TritonMLASparseImpl(XPUMLASparseImpl):\n    # fp8_ds_mla(NoPE 528B 页): 解码在 Triton 内核里软件完成, 写入走 write_fp8_nope_kv\n    supports_fp8_ds_mla_nope = True\n", 1),
    ("        for splits in KV_SPLITS_CANDIDATES:\n            triton_mla_sparse_attention(\n                q,\n                kv,\n                indices,\n                sm_scale=self.softmax_scale,\n                num_kv_splits=splits,\n                sm_count=self._sm_count,\n            )\n",
     "        for splits in KV_SPLITS_CANDIDATES:\n            triton_mla_sparse_attention(\n                q,\n                kv,\n                indices,\n                sm_scale=self.softmax_scale,\n                num_kv_splits=splits,\n                sm_count=self._sm_count,\n            )\n        if self.kv_cache_dtype == \"fp8_ds_mla\":\n            # 预热 fp8 变体, 避免第一次请求/cudagraph 捕获时才 JIT\n            kv8 = torch.zeros(\n                64, 1, 528 + 2 * (dim_qk - 512), dtype=torch.uint8, device=device\n            )\n            for splits in KV_SPLITS_CANDIDATES:\n                triton_mla_sparse_attention(\n                    q,\n                    kv8,\n                    indices,\n                    sm_scale=self.softmax_scale,\n                    num_kv_splits=splits,\n                    sm_count=self._sm_count,\n                )\n", 1),
    ("    def _forward_bf16_kv(\n",
     "    def do_kv_cache_update(\n        self,\n        kv_c_normed: torch.Tensor,\n        k_pe: torch.Tensor,\n        kv_cache: torch.Tensor,\n        slot_mapping: torch.Tensor,\n        kv_cache_dtype: str,\n        k_scale: torch.Tensor,\n    ) -> None:\n        if kv_cache_dtype != \"fp8_ds_mla\":\n            return super().do_kv_cache_update(\n                kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n            )\n        if kv_cache.numel() == 0:\n            return\n        assert self.head_size == 512 and k_pe.numel() == 0, (\n            \"fp8_ds_mla on TRITON_MLA_SPARSE currently supports NoPE (head_size=512) only\"\n        )\n        write_fp8_nope_kv(kv_c_normed, kv_cache, slot_mapping.flatten())\n\n    def _forward_bf16_kv(\n", 1),
    ("class TritonMLASparseBackend(XPUMLASparseBackend):\n    \"\"\"Same bf16 sparse-MLA contract as the XPU backend, CUDA Triton kernels.\"\"\"\n",
     "class TritonMLASparseBackend(XPUMLASparseBackend):\n    \"\"\"Same bf16 sparse-MLA contract as the XPU backend, CUDA Triton kernels.\"\"\"\n\n    supported_kv_cache_dtypes: ClassVar[list] = [\n        \"auto\",\n        \"float16\",\n        \"bfloat16\",\n        \"fp8\",\n        \"fp8_e4m3\",\n        \"fp8_ds_mla\",\n    ]\n", 1),
])

# ---- 4. mla_attention.py ----
edit("vllm/model_executor/layers/attention/mla_attention.py", [
    ("    backend_name = attn_backend.get_name()\n    if backend_name == \"FLASHMLA_SPARSE\" and is_quantized_kv_cache(kv_cache_dtype):",
     "    backend_name = attn_backend.get_name()\n    if backend_name == \"TRITON_MLA_SPARSE\" and kv_cache_dtype in (\n        \"fp8\",\n        \"fp8_e4m3\",\n        \"fp8_ds_mla\",\n    ):\n        return \"fp8_ds_mla\"\n    if backend_name == \"FLASHMLA_SPARSE\" and is_quantized_kv_cache(kv_cache_dtype):", 1),
    ("            state_content_bytes={\"fp8_ds_mla\": 656, \"nvfp4_ds_mla\": 352}.get(\n                self.kv_cache_dtype\n            ),",
     "            state_content_bytes={\n                \"fp8_ds_mla\": 528 if self.head_size == 512 else 656,\n                \"nvfp4_ds_mla\": 352,\n            }.get(self.kv_cache_dtype),", 1),
])
print("接线完成:", sh("git status --short").stdout.strip().replace("\n", " | "))
print("相对 patched 分支的差异:", sh("git diff patched --stat | tail -6").stdout)
