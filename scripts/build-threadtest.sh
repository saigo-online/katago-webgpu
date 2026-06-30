#!/usr/bin/env bash
# Path-A runtime gate: build the threaded WebGPU PoC (WebGPU device + compute on a
# dedicated pthread). Serve with scripts/serve-demo-tls.sh (sends COOP/COEP) and open
# web/demo/threadtest.html in a WebGPU browser — it must report PASS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${EMSDK_DIR:-$HOME/emsdk}/emsdk_env.sh" >/dev/null 2>&1
emcc "$ROOT/web/threadtest.cpp" -std=c++20 -O2 --use-port=emdawnwebgpu \
  -pthread -sPROXY_TO_PTHREAD -sPTHREAD_POOL_SIZE=4 -sASYNCIFY -sINITIAL_MEMORY=256MB \
  -sMODULARIZE=1 -sEXPORT_NAME=createTT -sEXPORTED_RUNTIME_METHODS=ccall \
  -o "$ROOT/web/demo/threadtest.js"
echo "built web/demo/threadtest.js"
