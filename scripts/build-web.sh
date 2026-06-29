#!/usr/bin/env bash
# ============================================================================
# Emscripten (WASM) build of the WebGPU kernel smoke test — the first step of
# the browser port. Compiles our real WGSL kernel library + a tiny harness to
# WebAssembly, linking Dawn's emdawnwebgpu port so <webgpu/webgpu_cpp.h> works
# in the browser. See engines/katago-web/WEBGPU_STATUS.md.
#
# Output: web/dist/wasm_smoke.{html,js,wasm}. Serve over http and open in a
# WebGPU-capable browser; it prints PASS/FAIL.
#
# Requires an emsdk (set EMSDK_DIR; defaults to ~/emsdk).
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # engines/katago-web
EMSDK_DIR="${EMSDK_DIR:-$HOME/emsdk}"
OUT="$HERE/web/dist"

if [ ! -f "$EMSDK_DIR/emsdk_env.sh" ]; then
  echo "ERROR: emsdk not found at $EMSDK_DIR (set EMSDK_DIR)"; exit 1
fi
# shellcheck disable=SC1091
source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1
echo "==> emcc: $(emcc --version | head -1)"

mkdir -p "$OUT"
DEMO="$HERE/web/demo"
mkdir -p "$DEMO"

COMMON=( --use-port=emdawnwebgpu -std=c++20 -O2 -sASYNCIFY -sALLOW_MEMORY_GROWTH=1 )

echo "==> [1/2] kernel smoke test -> web/dist/wasm_smoke.html"
emcc "$HERE/web/wasm_smoke.cpp" "$HERE/cpp/neuralnet/webgpukernels.cpp" \
  "${COMMON[@]}" -sEXIT_RUNTIME=1 -o "$OUT/wasm_smoke.html"

echo "==> [2/2] goban demo engine -> web/demo/engine.js"
emcc "$HERE/web/engine.cpp" "$HERE/cpp/neuralnet/webgpukernels.cpp" \
  "${COMMON[@]}" \
  -sMODULARIZE=1 -sEXPORT_NAME=createEngine \
  -sEXPORTED_FUNCTIONS=_engineInit,_engineDeviceName,_engineReady,_engineInfluence,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,getValue,setValue,HEAP32,HEAPF32,UTF8ToString \
  -o "$DEMO/engine.js"

echo "==> built:"
ls -la "$OUT" "$DEMO"
echo
echo "==> run the goban demo:   engines/katago-web/scripts/serve-demo.sh"
echo "    (or:  cd $DEMO && python3 -m http.server 8080  -> http://localhost:8080/ )"
