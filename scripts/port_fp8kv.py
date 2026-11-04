#!/usr/bin/env python3
"""把 nickus 的 FP8-KV 软件反量化思路手工移植到本树内核(副本,不改树内文件、不改运行中的服务)。
适配点: GLM-5.3-Flash 是 NoPE(BLOCK_DPE=0) -> 页 = 512 fp8 + 4 fp32 缩放 = 528 B; 带 RoPE 时 = 528 + 2*BLOCK_DPE。
"""
src = "/data/glm_test/tree/vllm/v1/attention/ops/triton_mla_sparse_kernel.py"
dst = "/data/glm_test/fp8kv/triton_mla_sparse_kernel_fp8.py"
import os
os.makedirs("/data/glm_test/fp8kv", exist_ok=True)
s = open(src).read()

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, f"锚点出现 {s.count(old)} 次(期望 {count}): {old[:70]!r}"
    s = s.replace(old, new)

# 1. 常量与解码函数
rep("# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.\n", '''# fp8 页: [0:512) fp8e4m3 NoPE 潜向量 | [512:528) 4 x fp32 分块缩放(amax/448,每128元素一块) | [528:) RoPE bf16 原样(仅 BLOCK_DPE>0)
_FP8_SCALE_OFF = tl.constexpr(512)
_FP8_ROPE_OFF = tl.constexpr(528)
_FP8_TILE = tl.constexpr(128)
_FP8_N_TILES = tl.constexpr(4)


@triton.jit
def _decode_fp8e4m3(b):
    """fp8e4m3fn 字节 -> fp16, 寄存器内位运算, 不用 Triton 的 fp8e4nv(sm80/86 没有)。
    (移植自 nickus/vllm-fp8kv; 对 256 个字节值与 torch.float8_e4m3fn 逐位一致)"""
    u = b.to(tl.uint16)
    sgn = (u & 0x80) << 8
    mag = (u & 0x7F) << 7
    val = (sgn | mag).to(tl.float16, bitcast=True) * 256.0
    nan = (sgn | 0x7F80).to(tl.float16, bitcast=True)
    return tl.where((u & 0x7F) == 0x7F, nan, val.to(tl.float16))


# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants.
''')

# 2. compute_tile 签名
rep('''    BLOCK_DPE: tl.constexpr,
):
    """Shared stage-1 body''', '''    BLOCK_DPE: tl.constexpr,
    IS_FP8: tl.constexpr = False,
):
    """Shared stage-1 body''')

# 3. K / KPE 加载
old_k = '''        offs_k = (
            indices[None, :].to(tl.int64) * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_d[:, None]
        )
        k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_kpe = (
                indices[None, :].to(tl.int64) * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=mask_kv[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))
'''
new_k = '''        if IS_FP8:
            ROW_BYTES: tl.constexpr = 528 + 2 * BLOCK_DPE
            row = indices.to(tl.int64) * ROW_BYTES
            kb = tl.load(
                k_buffer + row[None, :] + offs_d[:, None],
                mask=mask_kv[None, :],
                other=0,
            )
            kd = _decode_fp8e4m3(kb).to(tl.float32)
            sc = tl.load(
                (k_buffer + row[None, :] + _FP8_SCALE_OFF).to(
                    tl.pointer_type(tl.float32)
                )
                + tl.arange(0, _FP8_N_TILES)[:, None],
                mask=mask_kv[None, :],
                other=0.0,
            )
            kd3 = tl.reshape(kd, (_FP8_N_TILES, _FP8_TILE, BLOCK_N))
            k = tl.reshape(kd3 * sc[:, None, :], (BLOCK_DMODEL, BLOCK_N)).to(q.dtype)
            qk = tl.dot(q, k)
            if BLOCK_DPE > 0:
                kpe = tl.load(
                    (k_buffer + row[None, :] + _FP8_ROPE_OFF).to(
                        tl.pointer_type(tl.bfloat16)
                    )
                    + tl.arange(0, BLOCK_DPE)[:, None],
                    mask=mask_kv[None, :],
                    other=0.0,
                ).to(q.dtype)
                qk += tl.dot(qpe, kpe)
        else:
            offs_k = (
                indices[None, :].to(tl.int64) * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_d[:, None]
            )
            k = tl.load(k_buffer + offs_k, mask=mask_kv[None, :], other=0.0)
            qk = tl.dot(q, k.to(q.dtype))

            if BLOCK_DPE > 0:
                offs_kpe = (
                    indices[None, :].to(tl.int64) * stride_kv_token
                    + cur_kv_head_id * stride_kv_head
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    k_buffer + offs_kpe,
                    mask=mask_kv[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(q.dtype))
'''
rep(old_k, new_k)

# 4. V 加载(MLA: V 就是同一行的 NoPE 部分, 复用已解码的 k)
old_v = '''        offs_v = (
            indices[:, None].to(tl.int64) * stride_kv_token
            + cur_kv_head_id * stride_kv_head
            + offs_dv[None, :]
        )
        v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)
'''
new_v = '''        if IS_FP8:
            v = tl.trans(k)
        else:
            offs_v = (
                indices[:, None].to(tl.int64) * stride_kv_token
                + cur_kv_head_id * stride_kv_head
                + offs_dv[None, :]
            )
            v = tl.load(k_buffer + offs_v, mask=mask_kv[:, None], other=0.0)
'''
rep(old_v, new_v)

# 5. 两个 kernel 的签名与调用
rep('''    BLOCK_DPE: tl.constexpr,
):
    """Single-pass fast path''', '''    BLOCK_DPE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """Single-pass fast path''')
rep('''    BLOCK_DPE: tl.constexpr,
    LOGE2: tl.constexpr,
):''', '''    BLOCK_DPE: tl.constexpr,
    IS_FP8: tl.constexpr,
    LOGE2: tl.constexpr,
):''')
rep('''        BLOCK_DMODEL,
        BLOCK_DPE,
    )
''', '''        BLOCK_DMODEL,
        BLOCK_DPE,
        IS_FP8,
    )
''', count=2)

# 6. 调度函数
rep('''    assert kv.shape[1] == 1 and kv.shape[2] == dim_qk
''', '''    is_fp8 = kv.dtype in (torch.uint8, torch.float8_e4m3fn)
    if is_fp8:
        kv = kv.view(torch.uint8).reshape(-1, 1, 528 + 2 * block_dpe)
    else:
        assert kv.shape[1] == 1 and kv.shape[2] == dim_qk
''')
rep('''            BLOCK_DPE=block_dpe,
        )
        return out''', '''            BLOCK_DPE=block_dpe,
            IS_FP8=is_fp8,
        )
        return out''')
rep('''        BLOCK_DPE=block_dpe,
        LOGE2=LOGE2,''', '''        BLOCK_DPE=block_dpe,
        IS_FP8=is_fp8,
        LOGE2=LOGE2,''')

open(dst, "w").write(s)
import py_compile; py_compile.compile(dst, doraise=True)
print("移植完成并通过语法检查 ->", dst, "(", len(s.splitlines()), "行 )")
