#!/usr/bin/env python3
# 手工移植 #56524 的 utils.py 部分到本树(KVBlockZeroer 按组分元数据版本)。
import re, sys
f = "/data/glm_test/tree/vllm/v1/worker/utils.py"
s = open(f).read()
assert "isinstance(spec, (AttentionSpec, MambaSpec))" not in s, "已经移植过"

# 限定在 KVBlockZeroer.__init__ 内
a = s.index("class KVBlockZeroer:")
b = s.index("    def build_meta(", a)
blk = s[a:b]

# 1) 文档字符串
old = "        Only AttentionSpec layers are processed; Mamba layers are skipped.\n"
new = "        Only AttentionSpec and MambaSpec layers are processed.\n"
assert blk.count(old) == 1; blk = blk.replace(old, new)

# 2) spec 过滤
old = "            if not isinstance(spec, AttentionSpec):\n                continue\n"
new = "            if not isinstance(spec, (AttentionSpec, MambaSpec)):\n                continue\n"
assert blk.count(old) == 1; blk = blk.replace(old, new)

# 3) 把 kv 循环体包进 for kv in kvs
start_marker = ("                kv = static_forward_context[layer_name].kv_cache\n"
                "                if not isinstance(kv, torch.Tensor):\n"
                "                    continue\n")
assert blk.count(start_marker) == 1
i = blk.index(start_marker)
j_marker = "                        seg_page_sizes.append(kernel_page_bytes // 4)\n"
assert blk.count(j_marker) == 1
j = blk.index(j_marker) + len(j_marker)
body = blk[i + len(start_marker.split("if not isinstance")[0]):j]   # 从 "if not isinstance(kv..." 开始
body = start_marker.split("\n", 1)[1] + blk[i + len(start_marker):j]  # if not isinstance... + 其后
head = (
    "                layer_cache = static_forward_context[layer_name].kv_cache\n"
    "                # Mamba layers bind a tuple of state tensors (conv, recurrent, ...),\n"
    "                # each block-strided from the same backing pages; every tensor\n"
    "                # gets its own segment. (port of vllm#56524)\n"
    "                kvs = (\n"
    "                    list(layer_cache)\n"
    "                    if isinstance(layer_cache, (list, tuple))\n"
    "                    else [layer_cache]\n"
    "                )\n"
    "                for kv in kvs:\n"
)
ind = "".join(("    " + l if l.strip() else l) for l in body.splitlines(True))
blk = blk[:i] + head + ind + blk[j:]
s = s[:a] + blk + s[b:]
open(f, "w").write(s)
print("移植完成")
