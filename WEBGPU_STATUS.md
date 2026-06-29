# KataGo native WebGPU (Dawn) backend — status & roadmap

**Goal:** a `USE_BACKEND=WEBGPU` for KataGo that runs the neural net on the GPU
through **Dawn** (native WebGPU) on the desktop, with the *same* WGSL kernels
later reusable in-browser via Emscripten. This keeps KataGo's real C++ search,
rules, GTP and analysis engine intact and replaces only the NN backend.

This is the path we chose over delegating inference to onnxruntime-web — see the
chat that kicked this off. It is a **multi-week port**, not a weekend job. This
file tracks exactly where it stands and what's left.

## Current status: FULL NET EVALUATION WORKS, validated vs the Eigen CPU reference

The backend **compiles, links, runs on a real GPU via Dawn, and evaluates whole
KataGo nets correctly**. `getOutput()` runs the full trunk + policy + value graph
and writes real `NNOutput`s — the old throw is gone. Plain-residual +
global-pooling nets (the g170 b6/b10/b15/b20/b40 family, modelVersion 8) work.

Two layers of validation, both green:

```
$ katago runnnlayertests                       # USE_BACKEND=WEBGPU build
Tested 7 configurations                         # conv ×3, batchnorm ×2, residual, global-pooling
Done                                            # no mismatches → all kernels correct to fp32 tol

# Full-net eval vs Eigen CPU reference, same net, tiny-board test:
$ diff <(katago-eigen  runnnontinyboardtest NET true  true  0 false) \
       <(katago-webgpu runnnontinyboardtest NET false false 0 false)
# b6c96  : byte-identical (Win/Loss/Score/Policy/Ownership) across symmetries 0/3/5
# b10c128: matches within float rounding (Win 73.51c vs 73.52c — CPU↔GPU accumulation)
```

Milestones **M1–M6 are done and validated**:
- M1 dispatch harness · M2 conv2dNCHW · M3 scaleBiasMaskAct
- M4 residual block · M5 global-pooling residual block (caught a real `computeMerged`
  BN-fold bug: `hasScale`/`hasBias` must NOT gate `scale[]`/`bias[]`)
- M6 trunk (initialConv + initialMatMul + blocks + trunk-tip) + policy head
  (incl. pass) + value head (value/score/ownership; value-head pooling uses a
  distinct mean/scaled/mean-of-squares kernel) + `getOutput` (mask from spatial
  feature 0, symmetry-correct output writing, modelVersion-aware score fields).
  Bug found: head output buffers must be created `CopySrc`-capable or `wgReadFloats`
  silently returns zeros (→ degenerate uniform policy).

Architectures not yet ported are **rejected with a clear error** (never silently
wrong): nested-bottleneck, transformer (b18nbt/b28), RMSNorm trunk, optimism
policy (numPolicyChannels != 1, i.e. modelVersion ≥ 12).

It ran on a real **NVIDIA GPU** via Dawn's Vulkan path (the per-op tests had
fallen back to software Vulkan / lavapipe).

### Performance (started)

Measured with `katago benchmark -v 256 -t 8` on b6c96 (NVIDIA, search nnEvals/s):

| change | nnEvals/s | visits/s |
|--------|-----------|----------|
| baseline (per-dispatch submit, per-output readback) | ~616 | ~978 |
| + persistent weight cache (upload once, not per eval) | ~623 | — |
| + single command encoder + coalesced readback | **~765** | **~1396** |

- **Persistent weights** — conv/matmul/BN weights upload once into the handle
  (`weightCache`/`bnCache`, keyed by source pointer) instead of ~4 MB re-uploaded
  every eval. Small win here (fast GPU, small net) but matters for big nets and
  for the browser, where host→GPU upload is slow.
- **One Submit per eval** — the whole forward pass records into a single
  `WgRecorder` command encoder (one compute pass per dispatch for ordering), vs.
  ~50 `Submit`s before. This + reading all 5 head outputs back through **one**
  `MapAsync` (instead of 5 blocking round-trips) was the real win: ~+24% eval
  throughput, ~+43% visits/s (lower latency also improves search efficiency).

**fp16 storage path (implemented).** The kernels use a `STO` storage-scalar
alias the backend prepends as `alias STO = f32;` or `enable f16; alias STO = f16;`.
Every storage load is `f32(...)`, every store `STO(...)`, so one source compiles
both ways; math always accumulates in **f32** (fp16 storage + fp32 compute —
halves bandwidth/VRAM, keeps accuracy). The device requests the `shader-f16`
feature when `useFP16` is on and the adapter supports it, else it **falls back to
fp32** (the fp32 path is the same kernels and stays fully validated). Host
float→half conversion **clamps finite values to ±65504** (not Inf) — see fork
audit below. Buffers/readbacks are fp16-aware and 4-byte-padded.

> Caveat: this machine's Dawn/Vulkan adapter (NVIDIA GB10, new Blackwell driver)
> does **not** advertise `shader-f16`, and the build has no software fallback
> adapter, so the fp16 GPU path **could not be exercised here** — it falls back to
> fp32. fp16 is primarily for the browser, where `shader-f16` is widely exposed.

### Audit of the `/home/taro/code/katago` fork (gweber/KataGo)

The fork diverges from upstream by **2 commits**, both **TensorRT-11-specific**
(port to strongly-typed networks + FP16 via ONNX graph rewrite) — touching only
`trtbackend.cpp` / `onnxmodelbuilder.cpp` / `sandbox.cpp`. None of that backend
code is portable to WebGPU. But its FP16 commit carries two **design lessons**
that do apply:

1. **Clamp float→half to ±65504, not Inf** — KataGo's `1e9` off-board attention
   mask bias becomes `0*Inf = NaN` in softmax if promoted to Inf. **Applied** to
   `wgMakeStorage`'s fp16 upload (matters once transformer nets are supported;
   defensive now).
2. **Keep numerically-sensitive layers in FP32** even in FP16 mode — the fork
   keeps the policy/value heads, trunk tip, and RMSNorm reductions FP32. **Not yet
   applied** (our fp16 makes everything fp16-storage); a future refinement once a
   `shader-f16` adapter is available to validate against. The fork measured
   ~0.08% winrate error and ~2.4× throughput (256-ch transformer, 19×19) with its
   selective-fp16, which is the bar to match.

Update: **1×1 fast conv** and **buffer pooling** are now done + Eigen-validated
(`conv1x1NCHW`; a `thread_local` per-handle `BufferPool` reused across evals,
weights/BN opt out). Cumulative benchmark ~616 → ~810 nnEvals/s (+31%) on b6c96.
Still open: **Winograd 3×3** (the direct conv is the remaining compute hot spot),
**selective-fp32 heads** (lesson 2 above — needs a shader-f16 adapter to
validate), larger fixed batch. Correctness re-checked against Eigen after each.

### Browser build (Emscripten / WASM) — started

The backend was written against the **standard** WebGPU API specifically so it
ports to the browser. The toolchain is confirmed and the first WASM artifact
builds:

- **Toolchain**: emsdk (emcc 6.0.1) + Dawn's `emdawnwebgpu` Emscripten port,
  which Emscripten vendors as a remote port — `emcc --use-port=emdawnwebgpu`
  auto-fetches it. No manual Dawn-for-WASM build needed.
- **`web/wasm_smoke.cpp`** + **`scripts/build-web.sh`** compile our *actual*
  kernel library (`cpp/neuralnet/webgpukernels.cpp`, which has zero KataGo deps)
  plus a tiny harness to `web/dist/wasm_smoke.{html,js,wasm}`. It runs
  `scaleBiasMaskAct` on a known input via the browser's WebGPU and checks the
  result. **Build succeeds** (≈84 KB wasm), proving the WGSL kernels + standard
  WebGPU C++ API are WASM-portable through emdawnwebgpu.
- **Async readback** is browser-ready: all blocking waits go through `wgPump`,
  which is `ProcessEvents()` natively and `emscripten_sleep(0)` (Asyncify/JSPI)
  in the browser; the harness uses `CallbackMode::AllowSpontaneous` (browser
  event loop drives callbacks). Build flags: `--use-port=emdawnwebgpu -sASYNCIFY
  -std=c++20`.

- **Goban demo** — `web/demo/index.html` is a playable board (WGo.js, vendored)
  wired to an engine WASM module (`web/engine.cpp` → `web/demo/engine.{js,wasm}`)
  that exposes `engineInit` / `engineInfluence` to JS. Placing stones runs our
  real `conv2dNCHW` kernel on the GPU (N diffusion passes) and overlays a live
  stone-influence map — driving the exact upload → dispatch → readback pipeline
  the net eval will use. Build with `scripts/build-web.sh`, serve with
  `scripts/serve-demo.sh`, open in Chrome/Edge.

> Could not run the *GPU* path in this environment: the headless server exposes no
> WebGPU to the browser (`navigator.gpu` is undefined — the GPU's Vulkan driver
> is incompatible for Chromium and SwiftShader-WebGPU didn't materialize either).
> What *was* verified headless via `--dump-dom`: the demo page loads, the WGo
> board renders (3 canvas layers), and the engine WASM loads + runs `engineInit`,
> reporting `(no WebGPU adapter)` and degrading gracefully (board still works).
> On a desktop WebGPU browser the influence overlay and `wasm_smoke`'s
> `... PASS` line will appear. This is also where **fp16** and **selective-fp32
> heads** finally become testable (browsers expose shader-f16).

**Path to a slim eval-only WASM module** (the saigo.online target): compile the
WebGPU backend + `desc.cpp` (model load, needs zlib via `-sUSE_ZLIB`) + the
minimal `NNOutput`/`ModelDesc` types, behind a small C API
(`evaluate(spatial, global) -> {policy, value}`) where **JS supplies the input
feature tensors** (so we skip `game/` board+rules). Add `__EMSCRIPTEN__` guards
for `AllowSpontaneous` at the RequestAdapter/Device/MapAsync call sites. This is
far smaller than a full KataGo→WASM port (no search/GTP/command), and is the
next concrete step.

### Build recipe that works (proven on this machine)

`scripts/build-webgpu.sh` encodes all of this; the points that mattered:
- Dawn: `-DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED` (not `ON`), and disable
  windowing for a headless compute build (`-DDAWN_USE_GLFW=OFF`,
  `-DDAWN_USE_WAYLAND=OFF`, `-DDAWN_USE_X11=OFF`). Produces `libwebgpu_dawn.so`
  + a `find_package(Dawn)` package.
- KataGo: the backend TU needs **C++20** (Dawn's `webgpu_cpp.h` uses
  requires-expressions). Scoped to just `webgpubackend.cpp` via
  `set_source_files_properties(... -std=gnu++20)` — bumping the whole target
  breaks KataGo's C++17 code (`fileutils.cpp` / `path::u8string()`).
- Use the **standard** WebGPU API (`wgpu::CreateInstance` / `RequestAdapter` /
  `RequestDevice` / `ProcessEvents`), not `dawn::native::` — the monolithic lib
  only exports the standard API, and this path also works in-browser.
- `wgpu::ShaderSourceWGSL` (not the older `ShaderModuleWGSLDescriptor`).
- Callback lambdas must **not** be `noexcept` (Dawn's `CppFTraits` only
  specializes the non-noexcept `operator()`).
- Run with `LD_LIBRARY_PATH` including `third_party/dawn-install/lib`.

## (historical) initial scaffold

What is in place and real:

- **Source vendored** — full KataGo at `cpp/` (this directory is the upstream
  repo with its `.git` removed so it lives in the monorepo).
- **Build wiring** — `cpp/CMakeLists.txt` has a `WEBGPU` branch:
  - adds `neuralnet/webgpubackend.cpp` + `neuralnet/webgpukernels.cpp`
  - finds Dawn via `find_package(Dawn)` (→ `dawn::webgpu_dawn`) or
    `-DKATAGO_DAWN_DIR=<dir>`, and defines `USE_WEBGPU_BACKEND`.
- **Backend plumbing** — `neuralnet/webgpubackend.cpp` implements the entire
  `NeuralNet::` interface so the target links and runs:
  - acquires a real native WebGPU device via Dawn, prefers a discrete GPU,
    enumerates adapters (`printDevices`), compiles the WGSL module;
  - loads the model with KataGo's normal `ModelDesc` parser;
  - marshals MCTS inputs (spatial/global/meta, **with symmetry**) exactly like
    the OpenCL backend.
- **Starter kernels** — `neuralnet/webgpukernels.cpp` (WGSL): `conv2dNCHW`,
  `scaleBiasMaskAct` (fused BN+act+mask), `addInPlace` (residual),
  `matMulBiasAct`, `globalPoolMeanMax`. fp32, direct (no Winograd/fp16).
- **Dispatch harness (Milestone 1)** — `webgpubackend.cpp` has the buffer
  helpers, a per-entry-point pipeline cache, `wgDispatch`, and a blocking
  `wgReadFloats` readback. **Written, not yet run** (needs the Dawn build).
- **`testEvaluateConv` + `testEvaluateBatchNorm` (Milestones 2–3)** — wired to
  `conv2dNCHW` / `scaleBiasMaskAct` (NCHW fp32 path). **Written, not yet
  validated** — the moment Dawn is built, `katago runnnlayertests` checks them
  against KataGo's own CPU reference.
- **Build helper** — `scripts/build-webgpu.sh` fetches + builds Dawn, then
  configures + builds KataGo against it.

> ⚠️ The new C++/WGSL above is **unvalidated** until KataGo is built against
> Dawn and `runnnlayertests` is run. Three Dawn API surfaces may need a one-line
> tweak to match the pulled Dawn version (flagged `DAWN API NOTE` in the code):
> the WGSL shader-source descriptor, the `MapAsync` callback signature, and
> `OnSubmittedWorkDone`.

What is deliberately **not** done (so we never feed wrong evals to the search):

- `getOutput()` marshals inputs then **throws** — no forward pass yet.
- `createComputeHandle()` **throws** for nets with nested-bottleneck or
  transformer blocks (i.e. most modern b18/b28 nets); only plain + global-
  pooling residual trunks are in scope for the first milestone.
- `testEvaluate*` hooks return `false` (unimplemented).
- No fp16, no Winograd/1x1 fast convs, no async readback batching, no tuner.

> The honesty rule for this backend: **a throwing backend beats a wrong one.**
> Every path either produces a validated-correct result or throws a clear error
> pointing here.

## How to build right now

```bash
engines/katago-web/scripts/build-webgpu.sh        # fetches+builds Dawn, then KataGo
# or, if you already have Dawn installed:
cmake -S engines/katago-web/cpp -B build-webgpu \
  -DUSE_BACKEND=WEBGPU -DNO_GIT_REVISION=1 \
  -DKATAGO_DAWN_DIR=/path/to/dawn-install
cmake --build build-webgpu -j
```

Building Dawn is the slow, heavy step (large download + long compile). The
KataGo side is fast once Dawn exists.

## Roadmap (each milestone is independently validatable)

The `testEvaluate*` hooks in `nninterface.h` exist precisely so each op can be
checked numerically against the OpenCL/CUDA/Eigen backends in isolation. Do them
in order; don't move on until the previous one matches to tolerance.

1. **Device + dispatch harness.** ✅ **VALIDATED on GPU** — `wgDispatch` +
   `wgReadFloats` + pipeline cache in `webgpubackend.cpp`, blocking
   `ProcessEvents` readback. Proven by M2/M3 running through it.
2. **`testEvaluateConv`** → ✅ **VALIDATED** — `conv2dNCHW`; 3 NCHW-fp32 configs
   match the CPU reference (weight layout outC×inC×H×W + SAME padding correct).
3. **`testEvaluateBatchNorm`** → ✅ **VALIDATED** — `scaleBiasMaskAct` with
   host-folded `mergedScale`/`mergedBias` + mask; 2 NCHW-fp32 configs match.
4. **`testEvaluateResidualBlock`** → conv → scaleBiasMaskAct → conv →
   `addInPlace`.
5. **`testEvaluateGlobalPoolingResidualBlock`** → `globalPoolMeanMax` +
   `matMulBiasAct` broadcast-add. Confirm the size-scaled-mean formula matches
   `openclkernels.cpp` exactly.
6. **Trunk** — `initialConv`, iterate `trunk.blocks` (ordinary + gpool), trunk
   tip BN+act. Mirror `struct Trunk` in `openclbackend.cpp`.
7. **Policy head** — `p1Conv`/`g1Conv`/`g1BN`/gpool→bias/`p1BN`/`p2Conv` and the
   pass logits (`gpoolToPassMul`/`Bias`). Mirror `struct PolicyHead`.
8. **Value head** — `v1Conv`/`v1BN`/gpool/`v2`/`v3`/`sv3`/ownership conv. Mirror
   `struct ValueHead`. Then **`getOutput`** writes real `NNOutput`s; remove the
   throw. First full net eval — validate vs OpenCL on a known position.
9. **Nested-bottleneck blocks** (`NESTED_BOTTLENECK_BLOCK_KIND`) — recurse the
   block list. Then drop them from `assertSupportedArchitectureOrThrow`.
10. **Transformer blocks** (attention + FFN, RoPE, RMSNorm) for b28-style nets.
    Largest chunk; needs a matmul/softmax/attention kernel set.
11. **Performance** — fp16 storage/compute (`shader-f16` feature, fp32 fallback),
    Winograd 3×3, fused 1×1, persistent buffers, batch the readback.
12. **Browser target** — compile with Emscripten `-sUSE_WEBGPU`; replace the
    blocking readback with JSPI/Asyncify or a Web Worker. Kernels are unchanged.

## Key source references (in `cpp/neuralnet/`)

- `nninterface.h` — the contract every backend implements.
- `openclbackend.cpp` — closest reference (hand-written GPU kernels + host graph:
  `struct Trunk`/`PolicyHead`/`ValueHead`/`Model`/`Buffers`).
- `openclkernels.cpp` — the OpenCL kernels to port to WGSL.
- `desc.h` — `ModelDesc`/`TrunkDesc`/layer descs and the `*_BLOCK_KIND` consts.
- `activations.h` — `ACTIVATION_*` codes (kept in sync in the WGSL `activate()`).
