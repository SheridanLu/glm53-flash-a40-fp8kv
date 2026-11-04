# GLM-5.3-Flash on 8×A40 (sm_86): NoPE FP8 KV cache port, patch notes, monitor

**作者 / Author: [SheridanLu](https://github.com/SheridanLu)** — 项目负责人与运维。选定目标与取舍,在共享的生产硬件上承担全部风险,并订下三条工作原则:**先找社区的成熟方案、找不到再自己写;没有证据不说"修好了";每一项结论都要能被原始日志复核**。本仓库里的每一次重启、压测和上线都由作者本人授权并在作者的 8×A40 服务器上完成。

*AI assistance disclosure:* code, scripts and this document were written with Claude Code (Anthropic) under the author's direction; all experiments were run on the author's hardware and the raw logs are in `results/`. Treat the code as AI-assisted and review it accordingly.

**Status: experimental. Read "What is NOT verified" before using anything here.**

English summary: GLM-5.3-Flash (AWQ W4A16) served with the `lazymio/vllm-backport:v0.13.1-sm86` image on 8×A40 (TP8, MTP-3, prefix caching, 512K context). This repo contains (1) a port of a software-dequant FP8 KV cache for the *NoPE* (qk_rope_head_dim=0) sparse-MLA path on Ampere — 528 B/token/layer instead of 1024 B, (2) the result of dry-running 13 community vLLM PRs against v0.13.1, (3) a read-only monitor with a two-way concurrent "canary" probe, and (4) the test scripts and raw logs. On our box the KV pool went from 955,759 to 1,771,693 tokens (1.85×); needle retrieval was 12/12 correct up to 400K tokens. It has **not** been A/B-compared against BF16 on the same prompts, and the canary cannot detect the long-session degeneration reported in wtdcode/vllm-backport#119.

---

## 这是什么

在 8×A40(sm_86,无原生 FP8,无 NVLink)上跑 GLM-5.3-Flash AWQ W4A16 的一次完整记录:

| | |
|---|---|
| 镜像 | `lazymio/vllm-backport:v0.13.1-sm86`(wtdcode/vllm-backport) |
| 模型 | `wtdcode/GLM-5.3-Flash-AWQ-W4A16`,TP8,`--max-model-len 524288`,`--gpu-memory-utilization 0.88` |
| 配置 | MTP-3、前缀缓存、cudagraph `FULL_AND_PIECEWISE`(捕获尺寸到 64)、`--max-num-batched-tokens 8192` |
| 内容 | ① NoPE 版 FP8 KV 移植 ② 上游补丁在 v0.13.1 上的干跑记录 ③ 监视器 ④ 测试脚本与原始日志 |

## 结果(2026-11-04,同一台机器)

**KV 容量**

| | BF16 KV | FP8 KV(本仓库) |
|---|---|---|
| KV 总量 | 955,759 tokens | **1,771,693 tokens(1.85×)** |
| 512K 请求并发上限 | 1.82× | **3.38×** |
| 每卡可用 KV 显存 | 11.53 GiB | 11.53 GiB |

BF16 时,真实多人使用下 KV 使用率 p90=89%、最大 99.7%,约 44% 的采样有请求在排队;08:30–09:20 一小时内监视器的探针连续超时(KV 被撑满后前缀缓存命中率掉到 1%~7%)。换成 FP8 KV 后,同类拥堵没有再出现。

**速度**(零负载;FP8 与 BF16 两次测量的背景负载不同,差异 ≤±12% 不能当作确定结论)

| 并发 | BF16 | FP8 KV |
|---|---|---|
| 单流 | 85–124 tok/s(多次测量,均值约 107) | 107–110 tok/s |
| 4 路合计 | 219 | 214 |
| 8 路合计 | 259 | 289 |

真实用户流量(FP8 版,约 1262 个请求,平均输入 167,820 tokens):每请求平均解码 63.6–75.4 tok/s,首字延迟平均 3.5 s(90% ≤2.5 s),但约 2% 的请求首字 >20 s —— 见下文"已知瓶颈"。

**精度**(FP8 KV 实例上,答案可自动判定)

| 项目 | 结果 |
|---|---|
| 针检索 8K / 32K / 128K × 深度 10/50/90% | 9/9 |
| 针检索 400K(实测 400,016–400,017 tokens)× 深度 10/50/90% | 3/3 |
| 算术 12 题(聊天接口) | 12/12 |
| 指令遵循 3 题 | 3/3 |
| 代码 3 题(实际 exec 验证,聊天接口) | 3/3 |
| 内核级误差(合成高斯数据) | 相对误差 1.3%–3.2%,余弦 0.9995–0.9999 |

注意:补全接口(`/v1/completions`)不套对话模板,模型会在答案后继续编新题,所以算术/代码题必须用聊天接口测(见 `tests/code_chat.py`;`results/acc_fp8.log` 里那几项失败是该接口的格式问题,不是精度问题)。

## 做了什么

### 1. NoPE 版 FP8 KV(`patches/fp8kv_wiring.diff`)

GLM-5.3-Flash 的 `qk_rope_head_dim = 0`,KV 行只有 512 维。社区现有的软件反量化方案([nickus/vllm-fp8kv](https://github.com/nickus/vllm-fp8kv))假设 DeepSeek 式 576 维、656 字节的 `fp8_ds_mla` 页(含 64 维 RoPE),不能直接用。本仓库改成 **528 字节页**:

```
[0:512)    512 × fp8e4m3        NoPE 潜向量
[512:528)  4 × fp32             每 128 元素一块的缩放(amax/448)
```

- 解码:Triton 内核里按位解码(`_decode_fp8e4m3`,对 256 个字节值与 `torch.float8_e4m3fn` 逐位一致),不依赖 sm_89+ 才有的 `fp8e4nv`。MLA 的 V 就是同一行的 NoPE 部分,直接复用已解码的 K。
- 写入:torch 做分块量化(sm_86 上 torch 的 fp8 转换可用),Triton 把 528 B 页散写进分页缓存;`slot < 0`(cudagraph 填充)跳过,全程无同步,可被 cudagraph 捕获。
- 接线(4 个文件,约 213 行):`TRITON_MLA_SPARSE` 声明 `fp8`/`fp8_e4m3`/`fp8_ds_mla`;量化 dtype 规范化为 `fp8_ds_mla`;NoPE 页字节 528(而非 656);基类里写死的量化拒绝改为可由子类声明支持;预热 fp8 变体。
- 启用:`--kv-cache-dtype fp8_ds_mla`,并把 4 个文件只读挂载覆盖容器里的同名文件(见 `scripts/run_glm_fp8kv.sh`,启动失败会自动回退)。

踩过的坑:第一次启动失败是因为我从 git 分支读文件、挂载进去的是旧提交的内容,后端没声明 fp8,报 `No valid attention backend found`。现在脚本从工作区复制并校验 5 个关键标记,缺一个就中止。

### 2. 补丁栈干跑(`patches/patched_tree.diff`、`scripts/dryrun.sh`)

[wtdcode/vllm-backport#119](https://github.com/wtdcode/vllm-backport/issues/119) 里有人(ehrhart)报告打了 12 个上游 PR 后"深上下文编造"消失(无消融)。我把相关上游 PR 对 v0.13.1 容器里的 `.py` 文件做 `git apply --check`:

| 结果 | PR |
|---|---|
| 能干净打上 | #56525、#54076、#47123、#56756、#57477、#54368 |
| 打不上 | #56524、#37152、#42359、#43569、#55898、#51540、#57128 |

线上只用了 #56525(完整)和 #56524(`utils.py` 的 `KVBlockZeroer` 结构不同,手工移植,`scripts/port56524.py`)。本仓库**不重新发布**这些上游 PR 的 diff 全文,请去原 PR 取。`patches/patched_tree.diff` 是线上实际生效的 5 个文件相对 v0.13.1 的改动。

### 3. 监视器(`monitor/glm_monitor.py`)

只读、不重启、不记录用户内容。每 5 分钟一次:容器/`/health`、日志报错、前缀缓存命中率、MTP 平均接受长度、nginx 5xx,以及**两路并发的金丝雀探针**(共享约 4K token 前缀,各带唯一暗号要求原样复述;答错或串入另一路暗号即判失败)。`--selftest` 会故意用错误期望值,必须报出失败。

6.5 小时 79 次运行:71 次通过,5 次失败**全部是超时**(KV 被撑满时段),**0 次答错、0 次串内容**,日志报错 0。原始记录在 `results/status.log`、`results/alerts.log`。

## 未验证的(请务必读)

1. **没有同题 BF16 对照。** 以上精度是 FP8 KV 实例上的绝对正确率,这批题对 BF16 大概也是满分,量不出 FP8 比 BF16 差多少。更敏感的指标(长文本困惑度、logit 差异)没测。
2. **监视器抓不到 #119 那类问题。** 探针只发 4K token 短请求,查得到串内容和明显损坏,查不到数小时真实会话里逐渐出现的编造路径、空参数工具调用、失忆。#119 的报告者自己的合成复现也全是阴性。
3. **取消重试压力测试(`tests/abort_storm.py`)区分不了补丁有无效果**:未打补丁 0/36,打补丁后也是 0/36,在本环境里复现不出污染。且基线是 MTP 关、补丁版是 MTP 开,不是同配置对照。
4. FP8 KV 的内核级误差是合成高斯数据上测的;真实潜向量分布、11 层误差叠加、INT4 权重叠加 FP8 KV,没有人在 GLM-5.3-Flash 上验证过。
5. 速度对比的两次测量背景负载不同(见上表说明)。
6. 写入路径用 torch 做量化 + Triton 散写,不是单个融合内核;每层多几次小内核启动,端到端开销没单独测。
7. MTP 草稿层与视觉塔:MTP 草稿模型不接收图像嵌入(启动日志有警告);带图请求没测。

## 已知瓶颈(与本补丁无关)

`--max-num-batched-tokens 8192` 时,一个全新的 20–50 万 token 长上下文进来会让所有人的解码掉到约 1 tok/s,持续几十秒到两分钟(一个 8192 块约 1.8 s,同一步里别人的解码只能跟着走)。真实流量里平均每个请求输入 16.8 万 token,约 8% 的 10 秒窗口生成吞吐 <10 tok/s,其中多数同时有请求排队。把该参数降到 2048–4096 应能缓解,但**我们还没测**。

## 复现

```bash
# 前提: 容器内 vllm 的 python 包路径为 /usr/local/lib/python3.12/dist-packages/vllm
# 1. 取出容器里的 .py 作为基线, 在其上应用 patches/patched_tree.diff 与 patches/fp8kv_wiring.diff
# 2. 按 scripts/run_glm_fp8kv.sh 把改动的文件只读挂载覆盖到容器同路径
# 3. 启动参数加 --kv-cache-dtype fp8_ds_mla
# 4. 跑 tests/test_fp8kv_kernel.py、tests/test_fp8kv_write.py(需 GPU, 脚本有显存上限保护)
```

脚本里的路径(`/data/glm_test/...`)是我们机器上的,请自行替换。

## 出处与致谢

- [nickus/vllm-fp8kv](https://github.com/nickus/vllm-fp8kv)(Apache-2.0):软件反量化内核思路与 `_decode_fp8e4m3`;本仓库内核是针对 NoPE 的改写。
- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport):镜像与 GLM-5.3-Flash 在 Ampere 上的运行基础。
- vLLM 上游 PR #56524、#56525(datacrystals)、ehrhart 在 wtdcode/vllm-backport#119 给出的补丁清单。
- [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8):其 `fp8_sm80.py`(逐位一致的 fp8 编码器、查表解码)被调研但**未使用**,文中提到它是因为它是 SM80 上最成熟的同类零件。

## 许可

Apache-2.0(vLLM 衍生),见 `LICENSE` 与 `NOTICE`。**不提供任何担保。**
