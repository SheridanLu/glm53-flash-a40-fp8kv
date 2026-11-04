#!/bin/bash
# apply_patches.sh <site-packages-dir> [--dry-run | --revert]
#
# Applies patches/patched_tree.diff then patches/fp8kv_wiring.diff to an installed
# vllm-backport v0.13.1 tree, verifies the result, and keeps a backup so it can be undone.
#   (default)   backup -> dry-run on a scratch copy -> apply -> verify markers
#   --dry-run   only the scratch-copy check; touches nothing
#   --revert    restore the backed-up files
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(dirname "$here")
ROOT=${1:?usage: apply_patches.sh <site-packages-dir> [--dry-run|--revert]}
MODE=${2:-apply}
P1="$repo/patches/patched_tree.diff"; P2="$repo/patches/fp8kv_wiring.diff"
BK="$ROOT/.fp8kv_backup"
[ -d "$ROOT/vllm" ] || { echo "no vllm package under $ROOT" >&2; exit 2; }
command -v patch >/dev/null || { echo "'patch' is required" >&2; exit 2; }

files() { cat "$P1" "$P2" | sed -n 's#^+++ b/##p' | sort -u; }

if [ "$MODE" = "--revert" ]; then
  [ -d "$BK" ] || { echo "no backup at $BK" >&2; exit 1; }
  ( cd "$BK" && find . -type f ! -name MANIFEST.sha256 | sed 's#^\./##' ) | while read -r f; do
    cp -p "$BK/$f" "$ROOT/$f"; echo "restored $f"
  done
  echo "reverted. (backup kept at $BK)"; exit 0
fi

if "$here/verify_markers.sh" "$ROOT" >/dev/null 2>&1; then
  echo "already patched: all markers present"; exit 0
fi

# scratch copy of exactly the files the patches touch
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
files | while read -r f; do mkdir -p "$tmp/$(dirname "$f")"; cp -p "$ROOT/$f" "$tmp/$f"; done
mkdir -p "$tmp/vllm"
patch -p1 --no-backup-if-mismatch -d "$tmp" < "$P1" >/dev/null
patch -p1 --no-backup-if-mismatch -d "$tmp" < "$P2" >/dev/null
"$here/verify_markers.sh" "$tmp" >/dev/null || { echo "scratch copy failed marker check" >&2; exit 1; }
echo "dry-run OK ($(files | wc -l | tr -d ' ') files)"
[ "$MODE" = "--dry-run" ] && exit 0

mkdir -p "$BK"
files | while read -r f; do mkdir -p "$BK/$(dirname "$f")"; [ -e "$BK/$f" ] || cp -p "$ROOT/$f" "$BK/$f"; done
( cd "$ROOT" && files | xargs sha256sum ) > "$BK/MANIFEST.sha256"
patch -p1 --no-backup-if-mismatch -d "$ROOT" < "$P1" >/dev/null
patch -p1 --no-backup-if-mismatch -d "$ROOT" < "$P2" >/dev/null
"$here/verify_markers.sh" "$ROOT"
echo "applied. Undo with: $0 $ROOT --revert"
