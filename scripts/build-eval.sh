#!/usr/bin/env bash
# Build the kataeval WASM target: real KataGo net evaluation in the browser.
#
# kataeval is the one clean dependency (cpp/kataeval/): a C ABI (kataeval.h) over
# the minimal KataGo subset listed in cpp/kataeval/sources.txt. This script
# compiles that exact set + the WebGPU backend to WASM via emcc/emdawnwebgpu.
#
# Mixed C++ standard: webgpubackend.cpp needs C++20 (webgpu_cpp.h), but KataGo's
# fileutils.cpp breaks under C++20 — so compile per-file, then link.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMSDK_DIR="${EMSDK_DIR:-$HOME/emsdk}"
source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1
cd "$ROOT/cpp"
OBJ="$ROOT/cpp/build-wasm/obj"; mkdir -p "$OBJ"

CFLAGS=( -O2 -fexceptions -DNO_GIT_REVISION -DHALF_ENABLE_CPP11_CFENV=0
         -I external -isystem external/filesystem-1.5.8/include
         --use-port=emdawnwebgpu -sUSE_ZLIB=1 )

# MT=1 -> threaded build with KataGo's REAL Search (kgeSearchKata): adds -pthread,
# the search source closure, and the kgeSearchKata export; emits kataeval-mt.js
# (meant to be hosted in a Web Worker). The default single-thread build is unchanged
# (kataeval.js, no search sources, runs on the main thread).
MANIFESTS=( kataeval/sources.txt )
OUTNAME=kataeval
EXPORTS=_kgeLoad,_kgeEval,_kgeEvalSeq,_kgeEvalBatch,_kgeSearch,_kgeError,_kgeBoardSize,_kgeModelVersion,_kgeBackendIsGpu,_kgeSetForceCpu,_kgeSetFp16,_malloc,_free
LINKEXTRA=( -sINITIAL_MEMORY=64MB )
if [ "${MT:-0}" = 1 ]; then
  CFLAGS+=( -pthread -DKGE_THREADS )
  MANIFESTS+=( kataeval/sources-search.txt )
  OUTNAME=kataeval-mt
  EXPORTS="$EXPORTS,_kgeSearchKata,_kgeEvalSeqKata,_kgeSearchBegin,_kgePollAll,_kgePonderBegin,_kgeStopSearch,_kgeSetStrength"
  # 512MB initial so big nets (b18/b20, loaded ~2x in CPU) fit WITHOUT triggering
  # threaded-WASM memory growth (a fragile path that traps with "unaligned accesses").
  LINKEXTRA=( -pthread -sPTHREAD_POOL_SIZE=33 -sINITIAL_MEMORY=512MB )  # 1 NN server + up to 32 search threads
fi

# Read the canonical dependency manifest(s) (one clean dep, one source of truth).
mapfile -t SRCS < <(cat "${MANIFESTS[@]}" | grep -vE '^\s*(#|$)')
echo "==> $OUTNAME: ${#SRCS[@]} sources from ${MANIFESTS[*]}"

OBJS=()
for f in "${SRCS[@]}"; do
  extra=()
  std=c++17
  if [ "$f" = "kataeval/backend_gpu.cpp" ]; then
    std=c++20                                        # includes webgpubackend.cpp (webgpu_cpp.h)
  elif [ "$f" = "kataeval/backend_cpu.cpp" ]; then
    extra=( -DUSE_EIGEN_BACKEND -isystem /usr/include/eigen3 )   # includes eigenbackend.cpp
  fi
  o="$OBJ/$(echo "$f" | tr '/.' '__').o"
  emcc -c "$f" -std=$std "${CFLAGS[@]}" "${extra[@]}" -o "$o"
  OBJS+=("$o")
done

# ASSERTIONS=1 turns on emscripten's runtime checks + real abort() messages (stack
# traces instead of a bare "Aborted()"). Opt-in for debugging a crash; off in shipped
# builds (it's slower + larger). Usage: ASSERTIONS=1 MT=1 bash scripts/build-eval.sh
ASSERT=( )
if [ "${ASSERTIONS:-0}" = 1 ]; then ASSERT=( -sASSERTIONS=2 ); echo "==> ASSERTIONS on"; fi

echo "==> linking -> web/demo/$OUTNAME.js"
emcc "${OBJS[@]}" --use-port=emdawnwebgpu -sUSE_ZLIB=1 \
  -fexceptions \
  -sASYNCIFY -sALLOW_MEMORY_GROWTH=1 -sFORCE_FILESYSTEM=1 \
  -sSTACK_SIZE=16MB "${LINKEXTRA[@]}" "${ASSERT[@]}" \
  -sMODULARIZE=1 -sEXPORT_NAME=createKata \
  -sEXPORTED_FUNCTIONS="$EXPORTS" \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,FS,HEAPF32,HEAP32,UTF8ToString \
  -O2 -o "$ROOT/web/demo/$OUTNAME.js"

echo "==> built:"; ls -la "$ROOT/web/demo/$OUTNAME.js" "$ROOT/web/demo/$OUTNAME.wasm"

# Bundle the demo nets (web/demo/model-*.bin.gz are gitignored, so regenerate
# here for an out-of-the-box demo). The analyze.html net selector picks among
# them via ?net=. All are g170, modelVersion 8 (plain residual + gpool) — the
# family this backend supports — in increasing strength.
bundle() { [ -f "$1" ] && { cp "$1" "$ROOT/web/demo/$2"; echo "==> bundled $(basename "$1") -> web/demo/$2"; }; }
bundle "$ROOT/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz"     model-b6c96.bin.gz
bundle "$ROOT/cpp/tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz" model-b10c128.bin.gz
# Stronger nets, not committed: supply via $B20_NET / $B18_NET, else skipped — the
# selector still offers them and shows a friendly note if not bundled. b20c256 is
# g170 (plain residual); b18c384nbt is modelVersion 14 (nested-bottleneck + mish8).
bundle "${B20_NET:-/srv/nfs/fleet/go/models_ref/bin_gz/kata1-b20c256x2-s5303129600-d1228401921.bin.gz}" model-b20c256.bin.gz
bundle "${B18_NET:-/srv/nfs/fleet/go/models_ref/bin_gz/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz}" model-b18c384nbt.bin.gz
# modelVersion 17 transformer (RMSNorm + attention + SwiGLU) — a small TEST net
# (random-ish weights, weak play) bundled to prove the browser runs the v17 stack.
bundle "$ROOT/cpp/tests/models/b4c256h4nbttflrs-fson-silu-rsnh.bin.gz" model-b4c256nbttf.bin.gz
