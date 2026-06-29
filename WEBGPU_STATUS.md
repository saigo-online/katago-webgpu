# KataGo on WebGPU — status & roadmap

A `USE_BACKEND=WEBGPU` neural-net backend for KataGo in **WGSL** via
[Dawn](https://dawn.googlesource.com/dawn), plus the tooling to compile the
evaluation to **WebAssembly** and run a real net **in the browser**. The same
hand-written kernels run natively (desktop GPU) and in the browser
(`emdawnwebgpu`). KataGo's C++ search / rules / GTP are untouched — only the NN
backend is replaced.

This file is the honest engineering status. The polished overview is in
[`README.md`](README.md); upstream KataGo's README is
[`README-KATAGO-UPSTREAM.md`](README-KATAGO-UPSTREAM.md).

## Current status

**Both targets work and are validated.**

- **Native** (`-DUSE_BACKEND=WEBGPU`) evaluates whole nets on the desktop GPU via
  Dawn, **byte-identical to KataGo's Eigen CPU reference**.
- **Browser** — the same kernels, compiled to WASM, evaluate a real net offline.
  [`web/demo/analyze.html`](web/demo/analyze.html) is a playable goban that shows
  live policy / win-rate / ownership and plays its top move.
- **One WASM, automatic fallback** — `kataeval.wasm` ships both the WebGPU and the
  Eigen-CPU backend and falls back to CPU transparently when there's no adapter.
- Published as a fork: **`saigo-online/katago-webgpu`** (branched from
  `lightvector/KataGo`), referenced by saigo.online as a submodule.

Supported nets: plain-residual + global-pooling, **modelVersion 8** (the g170
b6/b10/b15/b20/b40 family). Architectures not yet ported are **rejected with a
clear error**, never silently mis-evaluated — see *Limits* below.

## Validation

```text
# per-op kernels vs the CPU reference (native WEBGPU build)
$ katago runnnlayertests
Tested 7 configurations          # conv ×3, batchnorm ×2, residual, global-pooling
Done                             # no mismatches

# full-net eval, WebGPU vs Eigen, same net:
#   b6c96   : byte-identical (Win/Loss/Score/Policy/Ownership), symmetries 0/3/5
#   b10c128 : matches within float rounding (Win 73.51c vs 73.52c — CPU↔GPU accumulation)
```

The dual-backend WASM is also cross-checked: the **WebGPU and Eigen-CPU paths
agree** on the same position (same win-rate, same policy), which confirms the
in-browser GPU forward pass matches the reference end to end.

Two real bugs that this validation caught and fixed:
- BN fold must use `scale[]`/`bias[]` unconditionally (`hasScale`/`hasBias` do
  **not** gate them — `computeMerged()` semantics); gating silently dropped a bias.
- Head output buffers must be `CopySrc`-capable, or `wgReadFloats` silently reads
  zeros → a degenerate uniform policy.

## Architecture

| Layer | What | Where |
|-------|------|-------|
| **WebGPU backend** | KataGo `NeuralNet::` backend in WGSL/Dawn | `cpp/neuralnet/webgpubackend.cpp`, `webgpukernels.cpp` |
| **`kataeval`** | minimal eval as one clean C++ dep | `cpp/kataeval/` (`kataeval.h`, `sources.txt`, `CMakeLists.txt`) |
| **Browser** | WASM build + goban demo | `web/`, `scripts/build-eval.sh`, `scripts/serve-demo-tls.sh` |

`kataeval` wraps the minimal KataGo subset (32 sources: neuralnet eval + game +
core — no search/GTP/`command/`) behind a tiny C ABI (`kgeLoad` / `kgeEval` →
policy/value/ownership). The dual-backend single binary compiles each backend
under its own namespace (rename macros over the upstream `.cpp`) and
`cpp/kataeval/dispatch.cpp` implements `NeuralNet::`, trying WebGPU then falling
back to Eigen. JS does the final softmax (so it can mask illegal moves).

## Performance

`katago benchmark -v 256 -t 8`, b6c96, 19×19 (search nnEvals/s):

| change | nnEvals/s |
|--------|-----------|
| baseline (per-dispatch submit, per-output readback) | ~616 |
| + persistent weight upload + buffer pool | ~640 |
| + single command submit + coalesced readback | ~765 |
| + 1×1 fast conv | **~810** (**+31%**) |

The big win was **latency**, not FLOPs: one `Submit` per eval (a single
`WgRecorder` command encoder) and **one** coalesced `MapAsync` readback instead
of ~50 submits and 5 blocking round-trips. Weights upload once
(`weightCache`/`bnCache`); intermediates/uniforms are reused from a per-handle
`BufferPool`; `conv1x1NCHW` skips the spatial loop for 1×1 layers (the heads).

Still open: **Winograd 3×3** (the direct conv is the remaining compute hot spot),
larger fixed batch.

## fp16

The kernels use a `STO` storage-scalar alias the backend prepends as
`alias STO = f32;` or `enable f16; alias STO = f16;` — one source, both
precisions; math always accumulates in **f32** (fp16 *storage*, fp32 *compute*:
halves bandwidth/VRAM, keeps accuracy). The device requests `shader-f16` when
available, else falls back to fp32 (the same, validated kernels). Host
float→half **clamps finite values to ±65504** (not Inf) so KataGo's large finite
sentinels — e.g. the `1e9` off-board attention-mask bias — don't become
`0*Inf = NaN` in softmax. Buffers/readbacks are fp16-aware and 4-byte padded.

> Not yet exercised end-to-end: the dev GPU's Dawn/Vulkan driver doesn't expose
> `shader-f16`, and the in-browser eval currently requests fp32. fp16 is built and
> the fallback is correct; validating the fp16 GPU path needs a `shader-f16`
> adapter (widely available in browsers). **Selective-fp32 heads** (keep the
> policy/value heads + RMSNorm reductions in fp32 even in fp16 mode — a lesson
> from KataGo's TensorRT FP16 path, which measured ~0.08% winrate error and
> ~2.4× throughput) is the next fp16 refinement, gated on that validation.

## Build & run

**Native (desktop GPU):**

```bash
scripts/build-webgpu.sh                          # fetches + builds Dawn, then KataGo
./cpp/build-webgpu/katago runnnlayertests        # per-op validation vs CPU reference
```

**Browser (WASM eval + goban demo):**

```bash
scripts/build-eval.sh        # -> web/demo/kataeval.{js,wasm} (needs emsdk, emcc >= 6)
scripts/serve-demo-tls.sh    # serve over HTTPS, then open analyze.html in Chrome/Edge
```

WebGPU is a *secure-context* API — `navigator.gpu` only exists over `https://`
or `localhost`, hence the TLS server (accept the self-signed cert once).

### Build gotchas worth knowing (encoded in the scripts)

- Dawn: `-DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED`, windowing off
  (`-DDAWN_USE_{GLFW,WAYLAND,X11}=OFF`) for a headless compute build.
- The WebGPU TU needs **C++20** (`webgpu_cpp.h` uses requires-expressions);
  KataGo's `fileutils.cpp` breaks under C++20 — so compile per-file (native CMake
  scopes it via `set_source_files_properties`; the WASM build compiles per-file).
- Use the **standard** WebGPU API (`CreateInstance`/`RequestAdapter`/`RequestDevice`),
  not `dawn::native::` — same path works natively and in-browser.
- Browser specifics: `-sASYNCIFY` + `emscripten_sleep(0)` to pump the event loop
  (`wgPump`), `CallbackMode::AllowSpontaneous` for the callbacks, `-fexceptions`
  at **link** (the CPU fallback relies on catching the no-adapter exception), and
  a bigger `-sSTACK_SIZE` (Eigen's 19×19 tensors overflow the default).
- half-float lib: `-DHALF_ENABLE_CPP11_CFENV=0` (Emscripten lacks the `FE_*` flags).

## Limits

- **Nets**: modelVersion 8, plain-residual + global-pooling. Nested-bottleneck /
  transformer (b18nbt/b28), RMSNorm trunks, and optimism policy
  (`numPolicyChannels != 1`, modelVersion ≥ 12) are rejected with a clear error.
- **Demo**: single-position analysis — no move history is fed yet, so ko/superko
  and "the actual game" aren't modeled; score is approximate; 19×19, Tromp-Taylor.
- **Perf**: direct convs (no Winograd yet); the bundled b6c96 is small and weak —
  great for a fast demo, modest strength (its flat opening policy is genuine, and
  reproduced identically on both backends).

## Roadmap

1. **Nested-bottleneck blocks** — recurse the block list; drop the guard. Unlocks
   b18nbt nets.
2. **Transformer blocks** (attention + FFN, RoPE, RMSNorm) + optimism policy —
   for b28-style and modelVersion ≥ 12 nets. The largest chunk.
3. **Winograd 3×3** — the main remaining conv speedup.
4. **Selective-fp32 heads** + validate the fp16 GPU path on a `shader-f16` adapter.
5. **Demo**: feed move history (ko/superko, real games), a stronger net (b10/b18),
   and ownership/score polish.

## Key source references (`cpp/`)

- `neuralnet/nninterface.h` — the backend contract.
- `neuralnet/webgpubackend.cpp`, `webgpukernels.cpp` — the WebGPU backend + WGSL.
- `neuralnet/eigenbackend.cpp` — the CPU reference (and the WASM fallback backend).
- `kataeval/` — the clean eval dependency (header + source manifest + dispatcher).
- `neuralnet/desc.h` — `ModelDesc`/`TrunkDesc`/layer descs, `*_BLOCK_KIND` consts.
