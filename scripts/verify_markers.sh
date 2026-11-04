#!/bin/sh
# verify_markers.sh [SITE_PACKAGES_DIR]
# Checks that every file touched by this repo's patches carries the expected change.
# Exit 0 = all markers present; non-zero = at least one is missing. Used by the Docker
# entrypoint (refuse to start on a half-patched tree) and by apply_patches.sh.
set -u
ROOT="${1:-}"
if [ -z "$ROOT" ]; then
  ROOT=$(python3 -c 'import importlib.util as u,os;print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))' 2>/dev/null)
fi
[ -d "$ROOT/vllm" ] || { echo "verify_markers: no vllm package under '$ROOT'" >&2; exit 2; }
bad=0
check() {
  f="$ROOT/$1"; pat="$2"
  if grep -qF -- "$pat" "$f" 2>/dev/null; then echo "ok   $1 :: $pat"; else echo "FAIL $1 :: $pat"; bad=1; fi
}
# FP8 KV (NoPE, 528 B pages)
check vllm/v1/attention/backends/mla/triton_mla_sparse.py        'supports_fp8_ds_mla_nope = True'
check vllm/v1/attention/backends/mla/xpu_mla_sparse.py           'supports_fp8_ds_mla_nope'
check vllm/v1/attention/ops/triton_mla_sparse_kernel.py          'def write_fp8_nope_kv'
check vllm/v1/attention/ops/triton_mla_sparse_kernel.py          'def _decode_fp8e4m3'
check vllm/model_executor/layers/attention/mla_attention.py      'backend_name == "TRITON_MLA_SPARSE" and kv_cache_dtype in'
check vllm/model_executor/layers/attention/mla_attention.py      '"fp8_ds_mla": 528 if self.head_size == 512 else 656'
# Hybrid-state set: vllm#56525 (commit fence) + hand-port of vllm#56524 (zero state blocks)
check vllm/v1/worker/utils.py                                    'port of vllm#56524'
check vllm/v1/core/single_type_kv_cache_manager.py               '_hash_publish_step'
check vllm/v1/core/kv_cache_manager.py                           'commit_step'
check vllm/v1/core/sched/scheduler.py                            'commit_step'
exit $bad
