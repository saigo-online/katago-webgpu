# KataGo on WebGPU — native **and** in the browser

A WebGPU neural-net backend for [KataGo](https://github.com/lightvector/KataGo),
plus the tooling to compile its evaluation to **WebAssembly** and run a real
KataGo net **entirely in the browser** — offline, on your GPU, with a clean
fallback to CPU when WebGPU isn't available.

The same hand-written **WGSL** kernels run two places:

- **Native** — `-DUSE_BACKEND=WEBGPU`, on the desktop GPU via [Dawn](https://dawn.googlesource.com/dawn).
- **Browser** — the identical kernels compiled to WASM via Dawn's `emdawnwebgpu`
  port, so a webpage can evaluate KataGo with no server round-trip.

> Status: the WebGPU backend evaluates plain-residual + global-pooling **and
> nested-bottleneck** nets — g170 **b6c96 / b10c128 / b20c256** (modelVersion 8)
> through kata1 **b18c384nbt** (modelVersion 14: nested-bottleneck + optimism
> policy + scaled-mish) — all **validated byte-identical to KataGo's Eigen CPU
> reference**. It is a real, working backend — not a mock — with honest limits
> noted below. See [`WEBGPU_STATUS.md`](WEBGPU_STATUS.md) for the engineering log.

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

- **Full forward pass on WebGPU** — initial conv + global-input matmul, ordinary,
  global-pooling **and nested-bottleneck** residual blocks (recursive), policy head
  (incl. pass + optimism, modelVersion ≥ 12) and value head (value / score /
  ownership). relu / mish / silu / scaled-mish activations. Direct WGSL, fp32 accum.
- **Validated** — byte-identical to the Eigen CPU backend on the per-op tests
  (`runnnlayertests`) and full-net evaluation; the GPU and CPU WASM paths agree.
- **Tuned** — 1×1 fast conv, persistent weight upload, per-handle buffer pooling,
  one command submit per eval, and a single coalesced readback (~+31% eval
  throughput over the first working version).
- **fp16 storage path** — uses `shader-f16` when the adapter exposes it, else
  falls back to fp32 (one kernel source, a storage-type alias).
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

- **Nets**: convolutional trunks through modelVersion 14 — ordinary, global-pooling
  and **nested-bottleneck** blocks; optimism policy (the standard policy, channel 0).
  Still **rejected with a clear error** (not silently mis-evaluated): **transformer**
  blocks (attention/FFN, b28-style) and **RMSNorm** trunks — next on the roadmap.
- **Demo**: single-position analysis (no move history fed yet, so ko/superko and
  "the actual game" aren't modeled); score is approximate; 19×19, Tromp-Taylor.
- **Perf**: direct convs (no Winograd yet). b6c96 is fast but weak; b20c256
  (23.5M params, ~9 dan) is strong but an 87 MB download and slower per eval —
  pick per your hardware.

## Credits & license

This is KataGo with a WebGPU backend and browser tooling added. **KataGo** is by
David J. Wu ([@lightvector](https://github.com/lightvector)) and contributors,
under the MIT license — see [`LICENSE`](LICENSE) and the upstream README at
[`README-KATAGO-UPSTREAM.md`](README-KATAGO-UPSTREAM.md). The WebGPU backend,
`kataeval`, and the browser demo are added here under the same license. Built on
[Dawn](https://dawn.googlesource.com/dawn) / `emdawnwebgpu` and
[WGo.js](https://github.com/waltheri/wgo.js).
