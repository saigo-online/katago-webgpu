#!/usr/bin/env bash
# ============================================================================
# Build KataGo with the native WebGPU (Dawn) backend.   SCAFFOLD.
# See engines/katago-web/WEBGPU_STATUS.md for what currently works.
#
# This fetches and builds Dawn (Google's native WebGPU implementation), then
# configures + builds KataGo's cpp/ with -DUSE_BACKEND=WEBGPU pointed at it.
#
# "Native" = runs against the desktop GPU through Dawn's webgpu-native. The same
# WGSL kernels are intended to later compile to the browser via Emscripten.
#
# Usage:
#   engines/katago-web/scripts/build-webgpu.sh            # fetch Dawn + build
#   DAWN_ONLY=1 engines/katago-web/scripts/build-webgpu.sh # just build Dawn
#   JOBS=8 engines/katago-web/scripts/build-webgpu.sh      # parallelism
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # engines/katago-web
CPP_DIR="$HERE/cpp"
THIRD_PARTY="$HERE/third_party"
DAWN_SRC="$THIRD_PARTY/dawn"
DAWN_BUILD="$THIRD_PARTY/dawn/out/Release"
DAWN_INSTALL="$THIRD_PARTY/dawn-install"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
DAWN_REF="${DAWN_REF:-main}"   # Dawn branch/tag; a chromium/NNNN branch is more stable if you know one

echo "==> KataGo WebGPU build helper"
echo "    repo:        $HERE"
echo "    dawn src:    $DAWN_SRC"
echo "    dawn install:$DAWN_INSTALL"
echo "    jobs:        $JOBS"

mkdir -p "$THIRD_PARTY"

# --- 1. Fetch Dawn ----------------------------------------------------------
if [ ! -d "$DAWN_SRC/.git" ]; then
  echo "==> Cloning Dawn ($DAWN_REF)"
  git clone --depth 1 --branch "$DAWN_REF" https://dawn.googlesource.com/dawn "$DAWN_SRC" \
    || git clone --depth 1 https://dawn.googlesource.com/dawn "$DAWN_SRC"
fi

# Dawn vendors its dependencies via a helper (no depot_tools/gclient needed).
if [ -f "$DAWN_SRC/tools/fetch_dawn_dependencies.py" ]; then
  echo "==> Fetching Dawn dependencies"
  python3 "$DAWN_SRC/tools/fetch_dawn_dependencies.py" --directory "$DAWN_SRC"
fi

# --- 2. Build Dawn's monolithic webgpu_dawn lib + install headers -----------
echo "==> Configuring Dawn"
cmake -S "$DAWN_SRC" -B "$DAWN_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAWN_FETCH_DEPENDENCIES=ON \
  -DDAWN_ENABLE_INSTALL=ON \
  -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED \
  -DDAWN_BUILD_SAMPLES=OFF \
  -DTINT_BUILD_TESTS=OFF \
  -DTINT_BUILD_CMD_TOOLS=OFF \
  -DDAWN_BUILD_TESTS=OFF \
  -DDAWN_USE_GLFW=OFF \
  -DDAWN_USE_WAYLAND=OFF \
  -DDAWN_USE_X11=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="$DAWN_INSTALL"

echo "==> Building + installing Dawn (this is the slow part)"
cmake --build "$DAWN_BUILD" --target install -j "$JOBS"

if [ "${DAWN_ONLY:-0}" = "1" ]; then
  echo "==> DAWN_ONLY set; Dawn installed at $DAWN_INSTALL. Stopping."
  exit 0
fi

# --- 3. Configure + build KataGo with the WebGPU backend --------------------
KATAGO_BUILD="$CPP_DIR/build-webgpu"
echo "==> Configuring KataGo (USE_BACKEND=WEBGPU)"
cmake -S "$CPP_DIR" -B "$KATAGO_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_BACKEND=WEBGPU \
  -DNO_GIT_REVISION=1 \
  -DCMAKE_PREFIX_PATH="$DAWN_INSTALL" \
  -DKATAGO_DAWN_DIR="$DAWN_INSTALL"

echo "==> Building KataGo"
cmake --build "$KATAGO_BUILD" -j "$JOBS"

echo "==> Done. Binary at: $KATAGO_BUILD/katago"
echo "    NOTE: the forward pass is still a scaffold — see WEBGPU_STATUS.md."
