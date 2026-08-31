#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"$THIS_DIR/build"}
SKIP_BUILD=${SKIP_BUILD:-0}

if [[ "$SKIP_BUILD" != "1" ]]; then
  "$THIS_DIR/build_all.sh"
fi

manifest="$BUILD_DIR/manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  echo "no manifest at $manifest; run build_all.sh first" >&2
  exit 1
fi

echo
echo "Compile-crash matrix (nvcc -arch=sm_75 -std=c++17 -O<opt> -ptx test.cu):"
echo
printf "%-12s %-6s %-6s %s\n" "compiler" "opt" "rc" "status"
tail -n +2 "$manifest" | while IFS=$'\t' read -r label opt rc status ptx; do
  printf "%-12s %-6s %-6s %s\n" "$label" "$opt" "$rc" "$status"
done

echo
echo "Expected differential: cuda-11.5 OK everywhere; cuda-12.5+ CRASH (SIGSEGV in cicc)."
