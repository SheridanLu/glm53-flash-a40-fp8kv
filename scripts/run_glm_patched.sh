#!/bin/bash
# 打补丁版 GLM 启动脚本(#56524 移植 + #56525),默认开 MTP-3(用户 2026-11-04 授权)。
# 回退: bash /data/glm_test/run_glm_pc.sh   (原版,未打补丁,MTP 关)
GPU_UTIL=${GPU_UTIL:-0.88}
SPEC=${SPEC:-1}
IMG=lazymio/vllm-backport:v0.13.1-sm86
L=/usr/lib/x86_64-linux-gnu; V=580.178.04
T=/data/glm_test/tree
D=/usr/local/lib/python3.12/dist-packages
FILES="vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/sched/scheduler.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/worker/utils.py"
mkdir -p /data/glm_test/cache
for f in $FILES; do [ -f $T/$f ] || { echo "缺文件 $T/$f"; exit 1; }; done
SPECARG=""
if [ "$SPEC" = "1" ]; then SPECARG='--speculative-config {"method":"mtp","num_speculative_tokens":3}'; fi
docker rm -f glm53flash >/dev/null 2>&1
DEV=""; for i in 0 1 2 3 4 5 6 7; do DEV="$DEV --device /dev/nvidia$i"; done
MNT=""; for f in $FILES; do MNT="$MNT -v $T/$f:$D/$f:ro"; done
docker run -d --name glm53flash --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add SYS_NICE --restart=no \
  $DEV --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e NVIDIA_VISIBLE_DEVICES=none -e LD_LIBRARY_PATH=$L -e NCCL_ALGO=Ring \
  -v $L/libcuda.so.$V:$L/libcuda.so.1:ro -v $L/libnvidia-ml.so.$V:$L/libnvidia-ml.so.1:ro \
  -v $L/libnvidia-ptxjitcompiler.so.$V:$L/libnvidia-ptxjitcompiler.so.1:ro -v $L/libnvidia-nvvm.so.$V:$L/libnvidia-nvvm.so.4:ro \
  -v /data/models/GLM-5.3-Flash-AWQ-W4A16:/model:ro -v /data/glm_test/cache:/root/.cache \
  $MNT \
  -p 127.0.0.1:18010:8000 \
  $IMG /model --host 0.0.0.0 --port 8000 --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --max-model-len 524288 --gpu-memory-utilization $GPU_UTIL \
  --max-num-seqs 16 --max-num-batched-tokens 8192 --disable-custom-all-reduce --numa-bind \
  --enable-prefix-caching --enable-prompt-tokens-details \
  $SPECARG \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,4,8,16,24,32,48,64],\"max_cudagraph_capture_size\":64}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
echo "started SPEC=$SPEC util=$GPU_UTIL at $(date -u +%FT%TZ)"
