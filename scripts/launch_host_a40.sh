#!/bin/bash
# launch_host_a40.sh - start the glm53-fp8kv image on OUR 8x A40 host.
#
# The host has no nvidia-container-toolkit, so GPU device nodes and the driver libraries are passed by hand.
# On a machine with the toolkit, replace the DEV/-v lib lines with `--gpus all`.
#
# Env: IMG (glm53-fp8kv:v0.13.1-sm86)  KV_DTYPE (fp8_ds_mla|auto)  GPU_UTIL (0.88)  MAX_BT (8192)
#      SPEC (1 = MTP-3, 0 = off)  NAME (glm53flash)  HOSTPORT (18010)  WAIT (seconds to wait for /health, 480)
set -u
IMG=${IMG:-glm53-fp8kv:v0.13.1-sm86}
KV_DTYPE=${KV_DTYPE:-fp8_ds_mla}
GPU_UTIL=${GPU_UTIL:-0.88}
MAX_BT=${MAX_BT:-8192}
SPEC=${SPEC:-1}
NAME=${NAME:-glm53flash}
HOSTPORT=${HOSTPORT:-18010}
WAIT=${WAIT:-480}
L=/usr/lib/x86_64-linux-gnu; V=580.178.04
mkdir -p /data/glm_test/cache
SPECARG=""; [ "$SPEC" = "1" ] && SPECARG='--speculative-config {"method":"mtp","num_speculative_tokens":3}'
docker rm -f "$NAME" >/dev/null 2>&1
DEV=""; for i in 0 1 2 3 4 5 6 7; do DEV="$DEV --device /dev/nvidia$i"; done
docker run -d --name "$NAME" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add SYS_NICE --restart=no \
  $DEV --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e NVIDIA_VISIBLE_DEVICES=none -e LD_LIBRARY_PATH=$L -e NCCL_ALGO=Ring \
  -v $L/libcuda.so.$V:$L/libcuda.so.1:ro -v $L/libnvidia-ml.so.$V:$L/libnvidia-ml.so.1:ro \
  -v $L/libnvidia-ptxjitcompiler.so.$V:$L/libnvidia-ptxjitcompiler.so.1:ro -v $L/libnvidia-nvvm.so.$V:$L/libnvidia-nvvm.so.4:ro \
  -v /data/models/GLM-5.3-Flash-AWQ-W4A16:/model:ro -v /data/glm_test/cache:/root/.cache \
  -p 127.0.0.1:$HOSTPORT:8000 \
  "$IMG" /model --host 0.0.0.0 --port 8000 --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --max-model-len 524288 --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 16 --max-num-batched-tokens "$MAX_BT" --disable-custom-all-reduce --numa-bind \
  --kv-cache-dtype "$KV_DTYPE" \
  --enable-prefix-caching --enable-prompt-tokens-details $SPECARG \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64],"max_cudagraph_capture_size":64}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 >/dev/null || { echo "docker run failed"; exit 1; }
echo "started $NAME image=$IMG kv=$KV_DTYPE util=$GPU_UTIL max_bt=$MAX_BT spec=$SPEC at $(date -u +%FT%TZ)"
t=0
while [ $t -lt "$WAIT" ]; do
  st=$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null)
  [ "$st" = "running" ] || { echo "container exited ($st)"; docker logs "$NAME" 2>&1 | tail -25; exit 1; }
  curl -sf -m 5 "localhost:$HOSTPORT/health" >/dev/null && { echo "READY after ${t}s"; docker logs "$NAME" 2>&1 | grep -E "GPU KV cache size|Maximum concurrency" | tail -2; exit 0; }
  sleep 10; t=$((t+10))
done
echo "not ready within ${WAIT}s"; docker logs "$NAME" 2>&1 | tail -25; exit 1
