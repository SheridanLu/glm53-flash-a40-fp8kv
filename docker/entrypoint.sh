#!/bin/sh
# Entrypoint for the glm53-fp8kv image: refuse to start on a half-patched tree, then run `vllm serve`.
set -eu
if ! /opt/fp8kv/verify_markers.sh >/tmp/fp8kv_markers.log 2>&1; then
  cat /tmp/fp8kv_markers.log >&2
  echo "[fp8kv] marker check FAILED - refusing to start (exit 78)" >&2
  exit 78
fi
case " $* " in
  *" --kv-cache-dtype "*) ;;
  *) echo "[fp8kv] note: no --kv-cache-dtype given, the KV cache will be BF16. Pass --kv-cache-dtype fp8_ds_mla for the FP8 path." >&2 ;;
esac
exec vllm serve "$@"
