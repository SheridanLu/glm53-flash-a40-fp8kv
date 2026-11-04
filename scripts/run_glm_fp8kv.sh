#!/bin/bash
# FP8 KV 实验启动(#kv-cache-dtype fp8_ds_mla): 挂载 5 个已部署补丁文件 + 4 个 fp8kv 分支文件。
# 启动后最多等 7 分钟; 不通就打印日志并自动回退到 run_glm_patched.sh(BF16)。
GPU_UTIL=${GPU_UTIL:-0.88}
SPEC=${SPEC:-1}
IMG=lazymio/vllm-backport:v0.13.1-sm86
L=/usr/lib/x86_64-linux-gnu; V=580.178.04
T=/data/glm_test/tree
D=/usr/local/lib/python3.12/dist-packages
PATCHED="vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/sched/scheduler.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/worker/utils.py"
FP8KV="vllm/v1/attention/ops/triton_mla_sparse_kernel.py vllm/v1/attention/backends/mla/triton_mla_sparse.py vllm/v1/attention/backends/mla/xpu_mla_sparse.py vllm/model_executor/layers/attention/mla_attention.py"
FILES="$PATCHED $FP8KV"
mkdir -p /data/glm_test/cache
echo "=== 从工作区直接复制(不走 git, 上次因此挂到旧版本)"
rm -rf /data/glm_test/fp8kv_mnt && mkdir -p /data/glm_test/fp8kv_mnt
for f in $FILES; do
  mkdir -p /data/glm_test/fp8kv_mnt/$(dirname $f)
  cp -f $T/$f /data/glm_test/fp8kv_mnt/$f || { echo "复制失败 $f"; exit 1; }
  python3 -m py_compile /data/glm_test/fp8kv_mnt/$f || { echo "语法错误 $f"; exit 1; }
done
echo "=== 关键标记校验(缺一即中止, 避免再次静默挂到旧版)"
grep -q "supports_fp8_ds_mla_nope" /data/glm_test/fp8kv_mnt/vllm/v1/attention/backends/mla/triton_mla_sparse.py || { echo "缺 supports_fp8_ds_mla_nope"; exit 1; }
grep -q "write_fp8_nope_kv" /data/glm_test/fp8kv_mnt/vllm/v1/attention/ops/triton_mla_sparse_kernel.py || { echo "缺 write_fp8_nope_kv"; exit 1; }
grep -q "TRITON_MLA_SPARSE\" and kv_cache_dtype in" /data/glm_test/fp8kv_mnt/vllm/model_executor/layers/attention/mla_attention.py || { echo "缺 dtype 规范化"; exit 1; }
grep -q "528 if self.head_size == 512 else 656" /data/glm_test/fp8kv_mnt/vllm/model_executor/layers/attention/mla_attention.py || { echo "缺 NoPE 528 页字节"; exit 1; }
grep -q "port of vllm#56524" /data/glm_test/fp8kv_mnt/vllm/v1/worker/utils.py || { echo "缺 56524 移植"; exit 1; }
echo "=== 5+4 个文件校验通过: $(cd /data/glm_test/fp8kv_mnt && ls -1 $(echo $FILES) | wc -l) 个"
sha256sum /data/glm_test/fp8kv_mnt/vllm/v1/attention/ops/triton_mla_sparse_kernel.py /data/glm_test/fp8kv_mnt/vllm/v1/attention/backends/mla/triton_mla_sparse.py | cut -c1-16
SPECARG=""; [ "$SPEC" = "1" ] && SPECARG='--speculative-config {"method":"mtp","num_speculative_tokens":3}'
docker rm -f glm53flash >/dev/null 2>&1
DEV=""; for i in 0 1 2 3 4 5 6 7; do DEV="$DEV --device /dev/nvidia$i"; done
MNT=""; for f in $FILES; do MNT="$MNT -v /data/glm_test/fp8kv_mnt/$f:$D/$f:ro"; done
docker run -d --name glm53flash --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add SYS_NICE --restart=no \
  $DEV --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e NVIDIA_VISIBLE_DEVICES=none -e LD_LIBRARY_PATH=$L -e NCCL_ALGO=Ring \
  -v $L/libcuda.so.$V:$L/libcuda.so.1:ro -v $L/libnvidia-ml.so.$V:$L/libnvidia-ml.so.1:ro \
  -v $L/libnvidia-ptxjitcompiler.so.$V:$L/libnvidia-ptxjitcompiler.so.1:ro -v $L/libnvidia-nvvm.so.$V:$L/libnvidia-nvvm.so.4:ro \
  -v /data/models/GLM-5.3-Flash-AWQ-W4A16:/model:ro -v /data/glm_test/cache:/root/.cache \
  $MNT -p 127.0.0.1:18010:8000 \
  $IMG /model --host 0.0.0.0 --port 8000 --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --max-model-len 524288 --gpu-memory-utilization $GPU_UTIL \
  --max-num-seqs 16 --max-num-batched-tokens 8192 --disable-custom-all-reduce --numa-bind \
  --kv-cache-dtype fp8_ds_mla \
  --enable-prefix-caching --enable-prompt-tokens-details $SPECARG \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,24,32,48,64],\"max_cudagraph_capture_size\":64}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
echo "fp8kv 容器已启动 $(date -u +%FT%TZ), 等待就绪(最多7分钟)"
for i in $(seq 1 42); do
  st=$(docker inspect glm53flash --format '{{.State.Status}}' 2>/dev/null)
  if [ "$st" != "running" ]; then echo "容器退出: $st"; break; fi
  curl -sf -m 5 localhost:18010/health >/dev/null && { echo "READY @ $(date -u +%FT%TZ)"; docker logs glm53flash 2>&1 | grep -E "GPU KV cache size|Available KV cache|Using .* attention backend|kv_cache_dtype" | tail -5 | cut -c1-200; exit 0; }
  sleep 10
done
echo "!!! FP8 KV 未能在限定时间内就绪, 采集错误并自动回退到 BF16(已打补丁版)"
docker logs glm53flash > /data/glm_test/fp8kv/fp8kv_boot_fail.log 2>&1
docker logs glm53flash 2>&1 | grep -E "Traceback|Error|error|assert|AssertionError|RuntimeError|ValueError|KeyError|NotImplemented" | tail -15 | cut -c1-220
docker rm -f glm53flash >/dev/null 2>&1
bash /data/glm_test/run_glm_patched.sh
for i in $(seq 1 42); do curl -sf -m 5 localhost:18010/health >/dev/null && { echo "已回退到 BF16 并就绪 @ $(date -u +%FT%TZ)"; exit 0; }; sleep 10; done
echo "!!! 回退后仍未就绪, 需人工介入"
exit 1
