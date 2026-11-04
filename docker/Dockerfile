# glm53-fp8kv: lazymio/vllm-backport v0.13.1 + NoPE FP8 KV cache + the hybrid-state patch set.
# Build from the repository root:   docker build -f docker/Dockerfile -t glm53-fp8kv:v0.13.1-sm86 .
# (If the daemon cannot reach a registry, use the legacy builder so the local base image is used:
#  DOCKER_BUILDKIT=0 docker build ... )
ARG BASE=lazymio/vllm-backport:v0.13.1-sm86
FROM ${BASE}

ARG SITE=/usr/local/lib/python3.12/dist-packages
LABEL org.opencontainers.image.title="glm53-fp8kv" \
      org.opencontainers.image.description="vllm-backport v0.13.1 with NoPE FP8 KV cache (528 B/token/layer) for GLM-5.3-Flash on sm_86" \
      org.opencontainers.image.source="https://github.com/SheridanLu/glm53-flash-a40-fp8kv" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY patches/patched_tree.diff patches/fp8kv_wiring.diff /opt/fp8kv/patches/
COPY scripts/verify_markers.sh /opt/fp8kv/verify_markers.sh
COPY docker/entrypoint.sh /opt/fp8kv/entrypoint.sh

# Fails the build if either diff does not apply cleanly to the base image or a marker is missing.
RUN set -eu; \
    chmod +x /opt/fp8kv/*.sh; \
    patch -p1 --dry-run -d "${SITE}" < /opt/fp8kv/patches/patched_tree.diff; \
    patch -p1 --no-backup-if-mismatch -d "${SITE}" < /opt/fp8kv/patches/patched_tree.diff; \
    patch -p1 --no-backup-if-mismatch -d "${SITE}" < /opt/fp8kv/patches/fp8kv_wiring.diff; \
    /opt/fp8kv/verify_markers.sh "${SITE}"; \
    sed -n 's#^+++ b/##p' /opt/fp8kv/patches/*.diff | sort -u | while read -r f; do python3 -m py_compile "${SITE}/$f"; done

ENTRYPOINT ["/opt/fp8kv/entrypoint.sh"]
