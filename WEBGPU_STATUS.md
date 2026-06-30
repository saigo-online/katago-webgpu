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

Supported nets: the **full KataGo architecture through modelVersion 17** —
ordinary / global-pooling / nested-bottleneck / **transformer** blocks; BatchNorm
and RMSNorm (per-position + spatial); attention with learnable/fixed RoPE,
grouped-query attention, SwiGLU FFN; relu / mish / silu / scaled-mish; optimism +
q-value policy. That spans g170 b6–b40, kata1 **b18c384nbt**, and v17 transformer
nets. The only pieces still **rejected with a clear error** are the SGF-metadata
encoder and grouped RMSNorm — see *Limits*.

## Validation

```text
# per-op kernels vs the CPU reference (native WEBGPU build)
$ katago runnnlayertests
Tested 7 configurations          # conv ×3, batchnorm ×2, residual, global-pooling
Done                             # no mismatches

# full-net eval, WebGPU vs Eigen, same net:
#   b6c96      : byte-identical (Win/Loss/Score/Policy/Ownership), symmetries 0/3/5
#   b10c128    : matches within float rounding (Win 73.51c vs 73.52c — CPU↔GPU accumulation)
#   b20c256x2  : agree exactly (Win 99.93c / Lead 7.86; 23.5M params)
#   b18c384nbt : agree exactly (Win 96.04c / Lead 15.02; modelVersion 14, nbt + optimism + mish8)
#   b4c256nbttf: agree exactly (Win 88.80c / Lead -2.40; modelVersion 17, transformer + RMSNorm + SwiGLU)
#   b7c96 GQA  : agree exactly (modelVersion 17, grouped-query attention + BN + transformer)
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
(`weightCache`/`bnCache`/`winogradCache`); intermediates/uniforms are reused from
a per-handle `BufferPool`; `conv1x1NCHW` skips the spatial loop for 1×1 layers.

**Winograd F(2,3) for 3×3 convs** (`winogradInput`/`winogradMatmul`/
`winogradOutput`, filters transformed once on host into `winogradCache`):
4 multiplies/output vs 9. A/B from one binary (`KATAGO_WEBGPU_NO_WINOGRAD=1`
forces direct conv), b6c96 19×19 8 threads on a GB10:

| 3×3 path | nnEvals/s | visits/s |
|----------|-----------|----------|
| direct `conv2dNCHW` | 296.9 | ~472 |
| **Winograd F(2,3)** | **511.4** (**+72%**) | ~937 |

Validated: `runnnlayertests` routes its 3×3 case through Winograd (matches the
CPU reference), and a full b6c96 tiny-board eval is **byte-identical** to direct
conv.

**Further SOTA passes** (each with an `A/B` env flag, each validated byte-identical
to Eigen on conv / nbt / transformer / GQA nets + per-op):

| # | optimization | what | flag |
|---|--------------|------|------|
| 1 | **Flash Attention** | fused single-kernel attention with online softmax — no materialized `[heads×seq×seq]` score matrix (3 dispatches → 1) | `KATAGO_WEBGPU_NO_FLASHATTN` |
| 2 | **Tiled GEMM** | shared-memory tiling for 1×1 conv / transformer projections | `KATAGO_WEBGPU_NO_TILEDGEMM` |
| 3 | **Kernel fusion** | fold pre-activation BN+act+mask into the 1×1 GEMM's input load (no intermediate) | `KATAGO_WEBGPU_NO_FUSION` |
| 6 | **Register tiling** | 2×2 output micro-tile per thread on top of shared-memory tiling | `KATAGO_WEBGPU_NO_REGTILE` |

Pending (need GPU features this dev adapter — NVIDIA GB10, software Vulkan — does
**not** expose, so they can't be validated here; to do on a `shader-f16`/subgroup
browser/GPU):
- **#4 fp16 compute + selective-fp32 heads** — the fp16 *storage* path exists; true
  fp16 compute + keeping policy/value/RMSNorm reductions in fp32 needs a `shader-f16`
  adapter to validate (this one reports "using fp32").
- **#5 subgroup reductions** — `subgroupAdd`/`subgroupShuffle` for pooling / softmax /
  GEMM dot-products; needs the WebGPU subgroups feature (absent here, and a missing
  feature would fail shader compilation, so it must be feature-gated on the device).

Still open (validatable here): tiled attention for big nets; larger fixed batch;
sharing the Winograd input transform across output channels.

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

- **Nets**: the full architecture through modelVersion 17 is supported. Still
  rejected with a clear error: the **SGF-metadata encoder** (`metaEncoderVersion ≠ 0`
  — train without it) and **grouped** RMSNorm (`cgroupSize ≠ 0`).
- **fp16**: the WebGPU load no longer applies the scale8 mish rescale — it's an
  fp16-only trick and the desc is shared with the Eigen CPU fallback (which asserts
  on scaled mish). When the fp16 GPU path is validated, apply scale8 to a per-handle
  copy instead.
- **Demo**: single-position analysis — no move history is fed yet, so ko/superko
  and "the actual game" aren't modeled; score is approximate; 19×19, Tromp-Taylor.
- **Perf**: 3×3 convs use **Winograd F(2,3)** (~+72% nnEvals/s); other filter
  sizes are direct. Attention is the simple O(seq²) form (one thread per (head,
  query, key)) — fine at 19×19, not yet tiled. The net selector spans b6c96 →
  b10c128 → b20c256 → b18c384nbt → a v17 transformer test net.

## Roadmap

1. ✅ **Winograd 3×3** — done + validated (F(2,3), ~+72% nnEvals/s vs direct,
   full-net byte-identical). Remaining: tiled attention for big nets.
2. **SGF-metadata encoder** + grouped RMSNorm — the last architecture gaps.
3. **Selective-fp32 heads** + validate the fp16 GPU path on a `shader-f16` adapter
   (then re-introduce the scale8 rescale per-handle for fp16 stability).
4. **Demo**: feed move history (ko/superko, real games) and ownership/score polish.
   *(Done: conv + nested-bottleneck + the full transformer stack through v17 —
   RMSNorm, RoPE, grouped-query attention, SwiGLU, optimism/q-value policy.)*

## Key source references (`cpp/`)

- `neuralnet/nninterface.h` — the backend contract.
- `neuralnet/webgpubackend.cpp`, `webgpukernels.cpp` — the WebGPU backend + WGSL.
- `neuralnet/eigenbackend.cpp` — the CPU reference (and the WASM fallback backend).
- `kataeval/` — the clean eval dependency (header + source manifest + dispatcher).
- `neuralnet/desc.h` — `ModelDesc`/`TrunkDesc`/layer descs, `*_BLOCK_KIND` consts.
