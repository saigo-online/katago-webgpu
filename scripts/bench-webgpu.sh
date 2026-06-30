#!/usr/bin/env bash
# Benchmark the native WebGPU (Dawn) backend on this box — the ground-truth perf +
# correctness check that does NOT need a browser. Reports nnEvals/s and avgBatchSize
# at a few thread counts, and (with KATAGO_WEBGPU_TIMING=1) the per-forward-pass phase
# breakdown + dispatch histogram.
#
#   scripts/bench-webgpu.sh [net.bin.gz]            # default: bundled b6c96
#   KATAGO_WEBGPU_TIMING=1 scripts/bench-webgpu.sh  # + phase timing + dispatch hist
#   KATAGO_WEBGPU_NO_WINOGRAD=1 / NO_FUSION=1 / KGE_MULTIPASS=1 / FP16=1  # A/B toggles
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KATA="$ROOT/cpp/build-webgpu/katago"
LIB="$ROOT/third_party/dawn-install/lib"
CFG="$ROOT/cpp/configs/gtp_example.cfg"
NET="${1:-$ROOT/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz}"
[ -x "$KATA" ] || { echo "build first: cmake --build cpp/build-webgpu -j"; exit 1; }

echo "net: $(basename "$NET")"
for t in 4 8 16 32; do
  line=$(LD_LIBRARY_PATH="$LIB" timeout 120 "$KATA" benchmark -model "$NET" -config "$CFG" \
           -override-config nnMaxBatchSize=64 -v 600 -t "$t" 2>&1 \
         | grep -oE "nnEvals/s = [0-9.]+ nnBatches/s = [0-9.]+ avgBatchSize = [0-9.]+" | tail -1)
  printf "  t=%-3s %s\n" "$t" "$line"
done
