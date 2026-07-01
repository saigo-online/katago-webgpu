# KataGo on WebGPU — native **and** in the browser

A WebGPU neural-net backend for [KataGo](https://github.com/lightvector/KataGo),
plus the tooling to compile its evaluation to **WebAssembly** and run a real
KataGo net **entirely in the browser** — offline, on your GPU, with a clean
fallback to CPU when WebGPU isn't available.

The same hand-written **WGSL** kernels run two places:

- **Native** — `-DUSE_BACKEND=WEBGPU`, on the desktop GPU via [Dawn](https://dawn.googlesource.com/dawn).
- **Browser** — the identical kernels compiled to WASM via Dawn's `emdawnwebgpu`
  port, so a webpage can evaluate KataGo with no server round-trip.

> Status: the WebGPU backend evaluates **the full KataGo architecture through
> modelVersion 17** — convolutional (g170 b6/b10/b20), nested-bottleneck
> (b18c384nbt), **and transformer** nets (attention + RoPE + SwiGLU + RMSNorm,
> incl. grouped-query attention and optimism/q-value policy) — all **validated
> byte-identical to KataGo's Eigen CPU reference**. A real, working backend, not a
> mock. The threaded WASM build (`MT=1`) additionally runs KataGo's **real Search**
> (`NNEvaluator` + `AsyncBot` — tree reuse, ponder, batched eval) in a Web Worker, so
> the browser gets analysis *and* play, not just single-position eval. See
> [`WEBGPU_STATUS.md`](WEBGPU_STATUS.md) for the engineering log.

---

## See it

A playable goban that analyzes and plays using a real KataGo net, evaluated in
*your* browser — pick the strength from the **net selector** (b6c96 → b10c128 →
b20c256):

```bash
scripts/build-eval.sh        # compile the eval to WASM (emcc + emdawnwebgpu)
scripts/serve-demo-tls.sh    # serve over HTTPS (WebGPU needs a secure context)
# open https://<host>:8443/analyze.html in desktop Chrome/Edge
```

- **`analyze.html`** — place stones and watch the **policy** (move probabilities),
  **win-rate**, **score**, and **ownership** update live; **"Engine plays"** makes
  KataGo take its top move. The **net selector** swaps strength (defaults to
  b10c128; `?net=b18c384nbt` for the strongest, a modelVersion-14 nested-bottleneck
  net). Add **`?cpu`** to the URL to force the CPU backend and compare.
  `build-eval.sh` bundles b6c96 + b10c128 from the repo's test models; the larger
  b20c256 / b18c384nbt are opt-in via `$B20_NET` / `$B18_NET`.
- **`index.html`** — a lighter demo that runs the raw `conv2dNCHW` WGSL kernel as
  a live stone-influence map (proves the GPU pipeline without the full net).

> WebGPU is a *secure-context* API: `navigator.gpu` only exists over `https://`
> or `http://localhost`. Plain `http://<lan-ip>` will show no WebGPU — use the
> HTTPS server above (accept the self-signed cert once).

## One WASM, automatic fallback

`kataeval.wasm` ships **both** backends — WebGPU and Eigen (CPU) — in a single
binary. At load it tries WebGPU and **transparently falls back to the Eigen CPU
backend** if there's no adapter. Same correct net everywhere; GPU speed when you
have it, universal compatibility when you don't. Both paths are validated, and
they agree.

## Highlights

- **Full forward pass on WebGPU** — initial conv + global-input matmul; ordinary,
  global-pooling, nested-bottleneck **and transformer** blocks (recursive); policy
  head (pass + optimism/q-value, modelVersion ≥ 12) and value head (value / score /
  ownership). relu / mish / silu / scaled-mish activations. Direct WGSL, fp32 accum.
- **Transformer stack** (modelVersion ≥ 15) — RMSNorm (per-position + spatial),
  multi-head attention with **learnable/fixed RoPE**, **grouped-query attention**,
  masked softmax, and **SwiGLU** feed-forward. All byte-identical to Eigen.
- **Validated** — byte-identical to the Eigen CPU backend on the per-op tests
  (`runnnlayertests`) and full-net evaluation; the GPU and CPU WASM paths agree.
- **Tuned** — profiled on a GB10 as *latency-bound* (per-op GPU launch/barrier
  overhead dominates, ~7ms/batch fixed): **Winograd F(2,3)** for 3×3 convs (**+80%**
  vs direct), BN+act fused into the Winograd input transform, **one compute pass** +
  one Submit + one coalesced readback per eval, 1×1 fast conv, persistent weights +
  per-handle buffer pool, and **batch/thread scaling** (a bigger batch amortizes the
  fixed latency — native t=4→32: **418→1814 nnEvals/s**; the demo auto-scales threads
  to your cores). Profiling + numbers + ranked next wins in `WEBGPU_STATUS.md`;
  reproduce with `scripts/bench-webgpu.sh`.
- **fp16 storage path** — `shader-f16` when available, but **fp32 by default**: fp16
  overflows the trunk to garbage on g170 nets (needs a scale8-style rescale, not done
  yet). Opt in per net via `kgeSetFp16` / the demo's `fp16` toggle for fp16-stable
  (modern mish_scale8/silu) nets to get the ~2× win.
- **`kataeval`** — a clean, single dependency (one header, one source manifest)
  wrapping the minimal KataGo subset needed to load a model and evaluate a
  position. No search, no GTP, no `command/`.

## Architecture

Three layers, each independently useful:

| Layer | What | Where |
|-------|------|-------|
| **WebGPU backend** | KataGo `NeuralNet::` backend in WGSL/Dawn | [`cpp/neuralnet/webgpubackend.cpp`](cpp/neuralnet/webgpubackend.cpp), [`webgpukernels.cpp`](cpp/neuralnet/webgpukernels.cpp) |
| **`kataeval`** | minimal eval as one clean C++ dep | [`cpp/kataeval/`](cpp/kataeval/) — `kataeval.h`, `sources.txt`, `CMakeLists.txt` |
| **Browser** | WASM build + goban demo (WGo.js) | [`web/`](web/), [`scripts/build-eval.sh`](scripts/build-eval.sh) |

The dual-backend single binary is a small dispatcher: each backend is compiled
under its own namespace (rename macros over the upstream `.cpp`), and
[`cpp/kataeval/dispatch.cpp`](cpp/kataeval/dispatch.cpp) implements the public
`NeuralNet::` interface, trying WebGPU then falling back to Eigen.

## Build

**Native WebGPU backend** (full KataGo binary, GPU via Dawn):

```bash
scripts/build-webgpu.sh                 # fetches + builds Dawn, then KataGo
./cpp/build-webgpu/katago runnnlayertests   # per-op validation vs CPU reference
```

**Browser eval (WASM):**

```bash
scripts/build-eval.sh                   # -> web/demo/kataeval.{js,wasm}
```

Requires an [emsdk](https://emscripten.org) (emcc ≥ 6) for the WASM builds; CMake
≥ 3.18 and a C++17 compiler for native. `kataeval` itself is consumable as a
normal CMake target — `add_subdirectory(cpp/kataeval)` then link `kataeval`
(plus Dawn and zlib).

## Repo layout

```
cpp/                     vendored KataGo + the WebGPU backend
  neuralnet/webgpu*      the WebGPU backend + WGSL kernels  (new)
  kataeval/              the clean eval dependency           (new)
web/                     WASM harnesses + the goban demo     (new)
scripts/                 build/serve helpers                 (new)
WEBGPU_STATUS.md         detailed status + engineering log
README-KATAGO-UPSTREAM.md   the original KataGo README
```

## Honest limits

- **Nets**: the full architecture through modelVersion 17 — conv / global-pooling /
  nested-bottleneck / transformer blocks, BatchNorm + RMSNorm, optimism + q-value
  policy. Still **rejected with a clear error** (not silently mis-evaluated): the
  **SGF-metadata encoder** (`metaEncoderVersion ≠ 0` — train without it) and
  **grouped** RMSNorm (`cgroupSize ≠ 0`).
- **Demo**: a full browser analysis/play/review app — instant policy+value+ownership,
  KataGo's **real threaded search** (tree reuse, ponder, live candidate moves + a PV
  "train of thought" board, streamed metrics), **self-play → SGF** + training-data
  export, **SGF review** (step through with per-move analysis), and **9/13/19 +
  handicap + komi**. Moves replay through KataGo's rules (captures, ko/superko); score
  is approximate; Tromp-Taylor.
- **Perf**: Winograd 3×3 + fused BN/act + single compute pass; profiled *latency-bound*
  so `nnEvals/s` scales with batch/threads (auto-scaled to your cores). fp16 is opt-in
  (fp32 default). b6c96 is fast but weak; bigger nets are stronger but larger downloads
  and slower per eval — pick per your hardware. See `WEBGPU_STATUS.md`.

## Credits & license

This is KataGo with a WebGPU backend and browser tooling added. **KataGo** is by
David J. Wu ([@lightvector](https://github.com/lightvector)) and contributors,
under the MIT license — see [`LICENSE`](LICENSE) and the upstream README at
[`README-KATAGO-UPSTREAM.md`](README-KATAGO-UPSTREAM.md). The WebGPU backend,
`kataeval`, and the browser demo are added here under the same license. Built on
[Dawn](https://dawn.googlesource.com/dawn) / `emdawnwebgpu` and
[WGo.js](https://github.com/waltheri/wgo.js).
