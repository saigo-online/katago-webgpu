// ============================================================================
// Native WebGPU (Dawn) backend for KataGo.  SCAFFOLD.
//
// See engines/katago-web/WEBGPU_STATUS.md for the full status and roadmap.
//
// What works here:
//   - Implements the whole NeuralNet:: interface so the target links.
//   - Acquires a real native WebGPU device via Dawn, enumerates adapters,
//     compiles the WGSL kernel module (webgpukernels.cpp).
//   - Loads the model via KataGo's normal ModelDesc parser and uploads layer
//     weights to GPU buffers.
//   - Marshals MCTS inputs (spatial/global/meta, with symmetry) exactly like
//     the OpenCL backend.
//
// What is NOT done yet (the multi-week port — tracked in WEBGPU_STATUS.md):
//   - The forward-pass kernel graph: residual / global-pooling / nested-
//     bottleneck / transformer blocks and the policy & value heads.
//   - fp16 path, Winograd/1x1 fast convs, async readback batching, tuning.
//
// To avoid silently feeding wrong evaluations into the search, getOutput()
// THROWS until the forward pass is implemented and validated against the
// reference backends via the testEvaluate* hooks. This is intentional: a
// throwing-but-honest backend beats a running-but-wrong one.
// ============================================================================

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/webgpukernels.h"
#include "../core/global.h"

// Dawn's monolithic webgpu_dawn library exposes the *standard* WebGPU API via
// <webgpu/webgpu_cpp.h>. We use that (CreateInstance / RequestAdapter /
// RequestDevice / ProcessEvents) rather than Dawn's internal dawn::native:: API,
// because the exact same code path is what Emscripten exposes in-browser later.
#if defined(__has_include)
  #if __has_include(<webgpu/webgpu_cpp.h>)
    #include <webgpu/webgpu_cpp.h>
    #define KATAGO_HAVE_WEBGPU 1
  #elif __has_include(<dawn/webgpu_cpp.h>)
    #include <dawn/webgpu_cpp.h>
    #define KATAGO_HAVE_WEBGPU 1
  #endif
#endif

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <cstring>

#include "../external/half-2.2.0/include/half.hpp"
using half_t = half_float::half;

using namespace std;

// ----------------------------------------------------------------------------
// Process-global Dawn instance
// ----------------------------------------------------------------------------

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#ifdef KATAGO_HAVE_WEBGPU
static wgpu::Instance gInstance = nullptr;

// Pump the WebGPU event loop until `done`. Natively we drive Dawn's event loop
// directly. In the browser there is a single event loop we must NOT busy-block:
// emscripten_sleep(0) unwinds and yields to it (so WebGPU callbacks can fire)
// then resumes — requires the build to enable Asyncify or JSPI. Keeping all
// blocking waits behind this one helper is what makes the backend browser-ready.
// `done` is volatile: it's flipped by a callback that ProcessEvents() invokes
// synchronously, and without volatile the compiler — which can't see that
// callback from inside this function — would hoist the load and spin forever.
static inline void wgPump(volatile bool& done) {
#ifdef __EMSCRIPTEN__
  while(!done) emscripten_sleep(0);
#else
  while(!done) gInstance.ProcessEvents();
#endif
}

// Callback mode must match how wgPump drives the event loop:
//  - native: AllowProcessEvents (callbacks fire inside ProcessEvents()).
//  - browser: AllowSpontaneous (callbacks fire from the JS event loop, which
//    emscripten_sleep(0) yields to). Using AllowProcessEvents in the browser
//    hangs forever — the callback never runs.
#ifdef __EMSCRIPTEN__
static constexpr wgpu::CallbackMode KGE_CB_MODE = wgpu::CallbackMode::AllowSpontaneous;
#else
static constexpr wgpu::CallbackMode KGE_CB_MODE = wgpu::CallbackMode::AllowProcessEvents;
#endif
#endif

void NeuralNet::globalInitialize() {
#ifdef KATAGO_HAVE_WEBGPU
  gInstance = wgpu::CreateInstance(nullptr);
  if(gInstance == nullptr)
    throw StringError("WebGPU backend: wgpu::CreateInstance() failed.");
#else
  throw StringError(
    "WebGPU backend was compiled without WebGPU headers. "
    "Build Dawn via engines/katago-web/scripts/build-webgpu.sh and re-run cmake "
    "with -DKATAGO_DAWN_DIR or -DCMAKE_PREFIX_PATH. See WEBGPU_STATUS.md.");
#endif
}

void NeuralNet::globalCleanup() {
#ifdef KATAGO_HAVE_WEBGPU
  gInstance = nullptr;
#endif
}

// ----------------------------------------------------------------------------
// Model loading (identical to other backends — reuses ModelDesc)
// ----------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    // NB: do NOT applyScale8ToReduceActivations() here. That rescale only matters
    // for fp16 numerical range; we compute in fp32, and our WGSL activate() handles
    // plain mish/silu directly. Crucially, this LoadedModel/ModelDesc is SHARED with
    // the Eigen CPU fallback in the single dual-backend binary, and Eigen asserts on
    // scaled activations (eigenbackend.cpp: "Eigen does not use scaled mish"). When
    // the fp16 GPU path is properly validated, apply the rescale to a per-handle copy.
  }
  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}
void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}
const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

// ----------------------------------------------------------------------------
// Compute context: holds the device, queue, and compiled kernel module
// ----------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  bool useFP16 = false;  // resolved: fp16 requested AND adapter supports shader-f16

#ifdef KATAGO_HAVE_WEBGPU
  wgpu::Device device;
  wgpu::Queue queue;
  wgpu::ShaderModule kernels;
  // Compute pipelines are expensive to create; cache one per WGSL entry point.
  std::unordered_map<std::string, wgpu::ComputePipeline> pipelineCache;

  wgpu::ComputePipeline getPipeline(const char* entryPoint) {
    auto it = pipelineCache.find(entryPoint);
    if(it != pipelineCache.end())
      return it->second;
    wgpu::ComputePipelineDescriptor desc{};
    desc.compute.module = kernels;
    desc.compute.entryPoint = entryPoint;
    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&desc);
    if(pipeline == nullptr)
      throw StringError(string("WebGPU backend: failed to create compute pipeline for ") + entryPoint);
    pipelineCache[entryPoint] = pipeline;
    return pipeline;
  }
#endif

  ComputeContext(int nnXLen_, int nnYLen_, enabled_t useFP16Mode_)
    : nnXLen(nnXLen_), nnYLen(nnYLen_), useFP16Mode(useFP16Mode_) {}
};

#ifdef KATAGO_HAVE_WEBGPU
// Standard-WebGPU synchronous adapter/device acquisition: kick off the async
// request, then pump the instance event loop until it resolves. This is the
// same API Emscripten exposes in-browser.
//
// DAWN/WEBGPU API NOTE: the callback-bearing entry points here are the surfaces
// most likely to need a one-line tweak against the pulled headers — the
// RequestAdapter/RequestDevice/MapAsync callback signatures (CallbackMode +
// wgpu::StringView) and the WGSL shader-source descriptor
// (ShaderModuleWGSLDescriptor vs newer ShaderSourceWGSL).
static wgpu::Adapter requestAdapterSync(Logger* logger) {
  wgpu::RequestAdapterOptions options{};
  options.powerPreference = wgpu::PowerPreference::HighPerformance;
  wgpu::Adapter result = nullptr;
  volatile bool done = false;
  gInstance.RequestAdapter(
    &options, KGE_CB_MODE,
    [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
      if(status == wgpu::RequestAdapterStatus::Success)
        result = adapter;
      else if(logger != NULL)
        logger->write(string("WebGPU backend: RequestAdapter failed: ") + string(message.data, message.length));
      done = true;
    });
  wgPump(done);
  if(!result)
    throw StringError("WebGPU backend: no suitable WebGPU adapter (RequestAdapter failed).");
  return result;
}

// Request a device, enabling the shader-f16 feature when wanted and supported.
static wgpu::Device requestDeviceSync(const wgpu::Adapter& adapter, Logger* logger, bool wantFP16, bool& gotFP16) {
  gotFP16 = wantFP16 && adapter.HasFeature(wgpu::FeatureName::ShaderF16);
  wgpu::FeatureName features[1] = { wgpu::FeatureName::ShaderF16 };
  wgpu::DeviceDescriptor devDesc{};
  if(gotFP16) { devDesc.requiredFeatureCount = 1; devDesc.requiredFeatures = features; }
  wgpu::Device result = nullptr;
  volatile bool done = false;
  adapter.RequestDevice(
    &devDesc, KGE_CB_MODE,
    [&](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
      if(status == wgpu::RequestDeviceStatus::Success)
        result = device;
      else if(logger != NULL)
        logger->write(string("WebGPU backend: RequestDevice failed: ") + string(message.data, message.length));
      done = true;
    });
  wgPump(done);
  if(!result)
    throw StringError("WebGPU backend: RequestDevice failed.");
  return result;
}

static wgpu::Device acquireDevice(Logger* logger, bool wantFP16, bool& gotFP16) {
  wgpu::Adapter adapter = requestAdapterSync(logger);
  if(logger != NULL) {
    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    logger->write(string("WebGPU backend: using adapter ") +
                  string(info.device.data, info.device.length));
  }
  return requestDeviceSync(adapter, logger, wantFP16, gotFP16);
}

// Acquire device + queue and compile the WGSL module into a context. Selects the
// fp16 (storage) or fp32 kernel variant by prepending the STO type alias.
// Shared by createComputeContext() and the testEvaluate* hooks.
static void initContextGpu(ComputeContext* context, Logger* logger) {
  if(!gInstance)
    NeuralNet::globalInitialize();
  bool wantFP16 = (context->useFP16Mode != enabled_t::False);
  bool gotFP16 = false;
  context->device = acquireDevice(logger, wantFP16, gotFP16);
  context->useFP16 = gotFP16;
  context->queue = context->device.GetQueue();

  std::string src = (gotFP16 ? "enable f16;\nalias STO = f16;\n" : "alias STO = f32;\n");
  src += KataGoWebGPU::WGSL_KERNELS;
  wgpu::ShaderSourceWGSL wgslDesc{};
  wgslDesc.code = src.c_str();
  wgpu::ShaderModuleDescriptor smDesc{};
  smDesc.nextInChain = &wgslDesc;
  context->kernels = context->device.CreateShaderModule(&smDesc);
  if(!context->kernels)
    throw StringError("WebGPU backend: failed to compile WGSL kernel module.");
  if(logger != NULL) {
    if(gotFP16)
      logger->write("WebGPU backend: fp16 storage ENABLED (shader-f16)");
    else if(wantFP16)
      logger->write("WebGPU backend: fp16 requested but this adapter lacks shader-f16; using fp32");
    else
      logger->write("WebGPU backend: using fp32");
  }
}
#endif

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs; (void)homeDataDirOverride; (void)loadedModel; (void)cfg;

  ComputeContext* context = new ComputeContext(nnXLen, nnYLen, useFP16Mode);

#ifdef KATAGO_HAVE_WEBGPU
  initContextGpu(context, logger);  // resolves + logs fp16 storage on/off
#else
  (void)logger;
  delete context;
  throw StringError("WebGPU backend compiled without Dawn native headers. See WEBGPU_STATUS.md.");
#endif

  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

// ----------------------------------------------------------------------------
// Dispatch harness (Milestone 1): buffers, pipeline dispatch, blocking readback.
// Everything the kernel-port milestones build on. Dawn-native only.
// ----------------------------------------------------------------------------

#ifdef KATAGO_HAVE_WEBGPU
namespace {

// Host mirrors of the WGSL uniform structs. All fields are u32 and tightly
// packed; each struct's size is a multiple of 16 bytes as WGSL uniforms require.
struct ConvParamsHost {
  uint32_t n, inC, outC, h, w, fy, fx, dilY, dilX, pad0, pad1, pad2;
};
struct Conv1x1ParamsHost { uint32_t n, inC, outC, hw; };
struct TGParamsHost { uint32_t n, inC, outC, hw, wInMajor, pad0, pad1, pad2; };
struct TGBParamsHost { uint32_t n, inC, outC, hw, act, pad0, pad1, pad2; };
struct SBMAParamsHost { uint32_t n, c, hw, act; };
struct AddParamsHost { uint32_t total, pad0, pad1, pad2; };
struct WgParamsHost { uint32_t n, inC, outC, h, w, nTilesX, nTilesY, xi; };

// Winograd F(2x2,3x3) filter transform U = G g G^T per (oc,ic).
// w: [outC][inC][3][3] (ConvLayerDesc order) -> U: [16][outC][inC] (xi-major,
// matching the winogradMatmul kernel's U[xi][oc][ic] layout).
static std::vector<float> winogradFilterTransform(const std::vector<float>& w, int outC, int inC) {
  static const float G[4][3] = {{1.f,0.f,0.f},{0.5f,0.5f,0.5f},{0.5f,-0.5f,0.5f},{0.f,0.f,1.f}};
  std::vector<float> U((size_t)16 * outC * inC);
  for(int oc = 0; oc < outC; oc++) {
    for(int ic = 0; ic < inC; ic++) {
      const float* g = w.data() + (size_t)(oc * inC + ic) * 9;
      float Gg[4][3];
      for(int i = 0; i < 4; i++)
        for(int j = 0; j < 3; j++) {
          float s = 0.f;
          for(int k = 0; k < 3; k++) s += G[i][k] * g[k * 3 + j];
          Gg[i][j] = s;
        }
      for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++) {
          float s = 0.f;
          for(int k = 0; k < 3; k++) s += Gg[i][k] * G[j][k];
          U[(size_t)(i * 4 + j) * outC * inC + oc * inC + ic] = s;
        }
    }
  }
  return U;
}
struct GPParamsHost { uint32_t n, c, hw, pad; };
struct MMParamsHost { uint32_t m, k, o, act, hasBias, pad0, pad1, pad2; };
struct AddBiasParamsHost { uint32_t n, c, hw, pad; };
// Transformer / RMSNorm (modelVersion >= 15). Layouts mirror the WGSL uniform
// structs (all 16-byte multiples).
struct RMSParamsHost { uint32_t n, c, hw, act; float eps; uint32_t hasBeta, pad0, pad1; };
struct ProjParamsHost { uint32_t n, inC, outC, hw; };
struct RopeParamsHost { uint32_t n, numHeads, headDim, numPairs, hw, numKVHeads, learnable, pad0; };
struct AttnSParamsHost { uint32_t n, numHeads, numKVHeads, headDim, hw, kvGroupSize; float scale; uint32_t pad0; };
struct AttnSMParamsHost { uint32_t n, numHeads, hw, pad0; };
struct AttnOParamsHost { uint32_t n, numHeads, numKVHeads, vHeadDim, hw, kvGroupSize, pad0, pad1; };
struct FlashParamsHost { uint32_t n, numHeads, numKVHeads, qHeadDim, vHeadDim, hw, kvGroupSize; float scale; };
struct SwiParamsHost { uint32_t total, pad0, pad1, pad2; };
struct RMSRParamsHost { uint32_t n, c, hw, pad0; float eps; uint32_t pad1, pad2, pad3; };
struct RMSAParamsHost { uint32_t n, c, hw, act, hasBeta, pad0, pad1, pad2; };

// Fold batchnorm into per-channel scale/bias, mirroring
// BatchNormLayerDesc::computeMerged() EXACTLY:
//   mergedScale[c] = scale[c] / sqrt(variance[c] + epsilon)
//   mergedBias[c]  = bias[c] - mergedScale[c] * mean[c]
// Note: computeMerged() uses scale[]/bias[] unconditionally — the hasScale /
// hasBias flags do NOT gate them (the loader always fills the arrays, e.g. with
// 1/0). Gating on those flags silently drops a real bias/scale and was a bug.
void foldBatchNorm(const BatchNormLayerDesc* desc, std::vector<float>& mergedScale, std::vector<float>& mergedBias) {
  const int C = desc->numChannels;
  mergedScale.resize(C);
  mergedBias.resize(C);
  for(int c = 0; c < C; c++) {
    float ms = desc->scale[c] / std::sqrt(desc->variance[c] + desc->epsilon);
    mergedScale[c] = ms;
    mergedBias[c] = desc->bias[c] - ms * desc->mean[c];
  }
}

// Storage scalar size and 4-byte-padded byte count (WebGPU requires buffer /
// copy / write sizes be a multiple of 4; an odd f16 count would otherwise fail).
static inline size_t wgScalarBytes(ComputeContext* ctx) { return ctx->useFP16 ? 2 : 4; }
static inline uint64_t wgPaddedBytes(ComputeContext* ctx, size_t count) {
  uint64_t b = (uint64_t)count * wgScalarBytes(ctx);
  return (b + 3ull) & ~3ull;
}

// Per-handle pool of reusable GPU buffers, keyed by (byte size, usage). reset()
// at the start of each eval marks all free; get() reuses a free match or makes a
// new one. Avoids allocating ~80 fresh buffers per eval. A single eval's buffers
// all stay live until its one Submit completes (getOutput blocks on readback),
// so reuse only ever happens across evals — safe, and single-threaded per handle.
struct BufferPool {
  ComputeContext* ctx = nullptr;
  struct Slot { uint64_t bytes; uint32_t usage; wgpu::Buffer buf; bool inUse; };
  std::vector<Slot> slots;
  void reset() { for(auto& s : slots) s.inUse = false; }
  wgpu::Buffer get(uint64_t bytes, wgpu::BufferUsage usage) {
    uint32_t u = (uint32_t)usage;
    for(auto& s : slots) if(!s.inUse && s.bytes == bytes && s.usage == u) { s.inUse = true; return s.buf; }
    wgpu::BufferDescriptor d{}; d.size = bytes; d.usage = usage;
    wgpu::Buffer b = ctx->device.CreateBuffer(&d);
    slots.push_back({bytes, u, b, true});
    return b;
  }
};
// Current thread's active pool (set by getOutput; null in the one-shot test hooks
// and for cached weights, which must persist rather than be recycled).
static thread_local BufferPool* tlPool = nullptr;

// Create a storage buffer of `count` scalars in the context's storage type
// (f32 or f16), optionally uploading float `data` (converted to half for fp16).
// When `pooled` and a thread pool is active, the buffer is drawn from / returned
// to that pool (weights pass pooled=false so cached buffers are never recycled).
wgpu::Buffer wgMakeStorage(ComputeContext* ctx, const float* data, size_t count, bool copySrc, bool pooled = true) {
  uint64_t bytes = wgPaddedBytes(ctx, count);
  wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  if(copySrc) usage |= wgpu::BufferUsage::CopySrc;
  wgpu::Buffer buf;
  if(pooled && tlPool != nullptr) {
    buf = tlPool->get(bytes, usage);
  } else {
    wgpu::BufferDescriptor d{}; d.size = bytes; d.usage = usage;
    buf = ctx->device.CreateBuffer(&d);
  }
  if(data != nullptr) {
    if(ctx->useFP16) {
      std::vector<half_t> half(bytes / sizeof(half_t), half_t(0));  // padded, zero-filled
      for(size_t i = 0; i < count; i++) {
        // Clamp finite out-of-range values to the max finite half (±65504) rather
        // than letting them overflow to Inf. KataGo uses large finite sentinels
        // (e.g. the 1e9 off-board attention mask bias) that must stay finite in
        // fp16 — an Inf there yields 0*Inf = NaN in softmax. (Mirrors the upstream
        // TensorRT fp16 path's floatToHalf / onnxconverter-common.)
        float v = data[i];
        if(std::isfinite(v)) v = std::max(-65504.0f, std::min(65504.0f, v));
        half[i] = half_float::half_cast<half_t>(v);
      }
      ctx->queue.WriteBuffer(buf, 0, half.data(), (size_t)bytes);
    } else {
      ctx->queue.WriteBuffer(buf, 0, data, count * sizeof(float));
    }
  }
  return buf;
}

template<typename T>
wgpu::Buffer wgMakeUniform(ComputeContext* ctx, const T& value) {
  wgpu::BufferUsage usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer buf;
  if(tlPool != nullptr) {
    buf = tlPool->get(sizeof(T), usage);
  } else {
    wgpu::BufferDescriptor d{}; d.size = sizeof(T); d.usage = usage;
    buf = ctx->device.CreateBuffer(&d);
  }
  ctx->queue.WriteBuffer(buf, 0, &value, sizeof(T));
  return buf;
}

// Bind buffers in order at bindings 0..k-1, then dispatch wgX*wgY*wgZ workgroups.
void wgDispatch(ComputeContext* ctx, const char* entryPoint,
                const std::vector<wgpu::Buffer>& binds,
                uint32_t wgX, uint32_t wgY = 1, uint32_t wgZ = 1) {
  wgpu::ComputePipeline pipeline = ctx->getPipeline(entryPoint);

  std::vector<wgpu::BindGroupEntry> entries(binds.size());
  for(size_t i = 0; i < binds.size(); i++) {
    entries[i].binding = (uint32_t)i;
    entries[i].buffer = binds[i];
    entries[i].offset = 0;
    entries[i].size = binds[i].GetSize();
  }
  wgpu::BindGroupDescriptor bgDesc{};
  bgDesc.layout = pipeline.GetBindGroupLayout(0);
  bgDesc.entryCount = entries.size();
  bgDesc.entries = entries.data();
  wgpu::BindGroup bindGroup = ctx->device.CreateBindGroup(&bgDesc);

  wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
  wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.DispatchWorkgroups(wgX, wgY, wgZ);
  pass.End();
  wgpu::CommandBuffer commands = encoder.Finish();
  ctx->queue.Submit(1, &commands);
}

// Copy `count` storage scalars (f32 or f16) from mapped bytes into `out` as float.
static void wgUnpack(ComputeContext* ctx, const void* mapped, size_t count, std::vector<float>& out) {
  out.resize(count);
  if(ctx->useFP16) {
    const half_t* h = (const half_t*)mapped;
    for(size_t i = 0; i < count; i++) out[i] = (float)h[i];
  } else {
    memcpy(out.data(), mapped, count * sizeof(float));
  }
}

// Blocking readback of `count` scalars from a CopySrc storage buffer (→ float).
void wgReadFloats(ComputeContext* ctx, const wgpu::Buffer& src, size_t count, std::vector<float>& out) {
  const uint64_t bytes = wgPaddedBytes(ctx, count);
  wgpu::BufferDescriptor d{};
  d.size = bytes;
  d.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer readback = ctx->device.CreateBuffer(&d);

  wgpu::CommandEncoder encoder = ctx->device.CreateCommandEncoder();
  encoder.CopyBufferToBuffer(src, 0, readback, 0, bytes);
  wgpu::CommandBuffer commands = encoder.Finish();
  ctx->queue.Submit(1, &commands);

  volatile bool done = false;
  bool ok = false;
  readback.MapAsync(
    wgpu::MapMode::Read, 0, (size_t)bytes, KGE_CB_MODE,
    [&done, &ok](wgpu::MapAsyncStatus status, wgpu::StringView) {
      ok = (status == wgpu::MapAsyncStatus::Success);
      done = true;
    });
  wgPump(done);
  if(!ok)
    throw StringError("WebGPU backend: buffer MapAsync for readback failed.");

  wgUnpack(ctx, readback.GetConstMappedRange(0, (size_t)bytes), count, out);
  readback.Unmap();
}

// Records a whole forward pass into ONE command encoder so it costs ONE Submit
// instead of one per dispatch. Each dispatch gets its own compute pass, which
// gives the WebGPU-guaranteed ordering + memory barrier between data-dependent
// ops. Bind groups are kept alive until submit (they hold refs to their buffers,
// including the per-dispatch uniform buffers, so nothing is freed early).
struct WgRecorder {
  ComputeContext* ctx;
  wgpu::CommandEncoder encoder;
  std::vector<wgpu::BindGroup> keepAlive;
  explicit WgRecorder(ComputeContext* c) : ctx(c), encoder(c->device.CreateCommandEncoder()) {}

  void dispatch(const char* entryPoint, const std::vector<wgpu::Buffer>& binds,
                uint32_t wgX, uint32_t wgY = 1, uint32_t wgZ = 1) {
    wgpu::ComputePipeline pipeline = ctx->getPipeline(entryPoint);
    std::vector<wgpu::BindGroupEntry> entries(binds.size());
    for(size_t i = 0; i < binds.size(); i++) {
      entries[i].binding = (uint32_t)i; entries[i].buffer = binds[i];
      entries[i].offset = 0; entries[i].size = binds[i].GetSize();
    }
    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = pipeline.GetBindGroupLayout(0);
    bgDesc.entryCount = entries.size(); bgDesc.entries = entries.data();
    wgpu::BindGroup bindGroup = ctx->device.CreateBindGroup(&bgDesc);
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline); pass.SetBindGroup(0, bindGroup);
    pass.DispatchWorkgroups(wgX, wgY, wgZ); pass.End();
    keepAlive.push_back(bindGroup);
  }

  // Append copies of several CopySrc outputs into one staging buffer, submit the
  // whole frame once, map once, and split into the destination vectors. Collapses
  // the per-output readbacks into a single GPU round-trip.
  void submitAndReadback(const std::vector<std::pair<wgpu::Buffer,size_t>>& srcs,
                         std::vector<std::vector<float>*>& outs) {
    // Each region's byte size is padded to a multiple of 4 (also keeps the next
    // copy offset 4-aligned, which CopyBufferToBuffer requires).
    std::vector<uint64_t> rbytes(srcs.size());
    uint64_t totalBytes = 0;
    for(size_t i = 0; i < srcs.size(); i++) { rbytes[i] = wgPaddedBytes(ctx, srcs[i].second); totalBytes += rbytes[i]; }
    wgpu::BufferDescriptor d{};
    d.size = totalBytes;
    d.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer staging = ctx->device.CreateBuffer(&d);
    uint64_t off = 0;
    for(size_t i = 0; i < srcs.size(); i++) { encoder.CopyBufferToBuffer(srcs[i].first, 0, staging, off, rbytes[i]); off += rbytes[i]; }
    wgpu::CommandBuffer commands = encoder.Finish();
    ctx->queue.Submit(1, &commands);

    volatile bool done = false; bool ok = false;
    staging.MapAsync(wgpu::MapMode::Read, 0, (size_t)totalBytes, KGE_CB_MODE,
      [&done,&ok](wgpu::MapAsyncStatus st, wgpu::StringView){ ok = (st==wgpu::MapAsyncStatus::Success); done = true; });
    wgPump(done);
    if(!ok) throw StringError("WebGPU backend: coalesced readback MapAsync failed.");
    const char* base = (const char*)staging.GetConstMappedRange(0, (size_t)totalBytes);
    off = 0;
    for(size_t i = 0; i < srcs.size(); i++) { wgUnpack(ctx, base + off, srcs[i].second, *outs[i]); off += rbytes[i]; }
    staging.Unmap();
  }
};

}  // namespace
#endif  // KATAGO_HAVE_WEBGPU

// ----------------------------------------------------------------------------
// Compute handle: per-thread; uploads weights and owns scratch buffers
// ----------------------------------------------------------------------------

// Reject architectures whose kernels are not yet ported, so we never pretend to
// support a net we cannot evaluate. As blocks are implemented + validated they
// move out of this list (see WEBGPU_STATUS.md milestones).
static void assertSupportedArchitectureOrThrow(const ModelDesc& desc) {
  // Supported block kinds (recursing into nested-bottleneck, where transformer
  // blocks live): ordinary / global-pooling / nbt / transformer-attention / -ffn.
  std::function<void(const std::vector<std::pair<int,unique_ptr_void>>&)> check =
    [&](const std::vector<std::pair<int,unique_ptr_void>>& bs){
    for(const auto& kb : bs) {
      int kind = kb.first;
      if(kind == NESTED_BOTTLENECK_BLOCK_KIND)
        check(((const NestedBottleneckResidualBlockDesc*)kb.second.get())->blocks);
      else if(kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        if(((const TransformerAttentionDesc*)kb.second.get())->vHeadDim > 64)
          throw StringError("WebGPU backend: attention vHeadDim > 64 not supported "
            "(flash-attention accumulator). See WEBGPU_STATUS.md.");
      }
      else if(kind != ORDINARY_BLOCK_KIND && kind != GLOBAL_POOLING_BLOCK_KIND &&
              kind != TRANSFORMER_FFN_BLOCK_KIND)
        throw StringError("WebGPU backend: unsupported block kind " +
          Global::intToString(kind) + ". See WEBGPU_STATUS.md.");
    }
  };
  check(desc.trunk.blocks);

  if(desc.metaEncoderVersion != 0)
    throw StringError("WebGPU backend: SGF metadata encoder (metaEncoderVersion != 0) "
      "not ported yet. Train without metadata, or use another backend. See WEBGPU_STATUS.md.");
  // Trunk norm: standard BatchNorm, or RMSNorm (per-position or spatial). Grouped
  // RMSNorm (cgroupSize != 0) isn't ported.
  if(desc.trunk.trunkNormKind == TRUNK_NORM_KIND_RMSNORM && desc.trunk.trunkTipRMSNorm.cgroupSize != 0)
    throw StringError("WebGPU backend: grouped trunk RMSNorm (cgroupSize != 0) not ported. "
      "See WEBGPU_STATUS.md.");
  if(desc.trunk.trunkNormKind != TRUNK_NORM_KIND_STANDARD &&
     desc.trunk.trunkNormKind != TRUNK_NORM_KIND_RMSNORM)
    throw StringError("WebGPU backend: unsupported trunkNormKind " +
      Global::intToString(desc.trunk.trunkNormKind) + ". See WEBGPU_STATUS.md.");
  // numPolicyChannels: 1 (pre-v12), 2 (optimism, v>=12), 4 (+ q-value policy, v>=16).
  // Readback uses channel 0 (the standard policy), so any width evaluates correctly.
  const int npc = desc.numPolicyChannels;
  if(!(npc == 1 || npc == 2 || (npc == 4 && desc.modelVersion >= 16)))
    throw StringError("WebGPU backend: numPolicyChannels=" + Global::intToString(npc) +
      " not supported (1, 2, or 4 with modelVersion>=16). See WEBGPU_STATUS.md.");
}

struct ComputeHandle {
  ComputeContext* context;
  const ModelDesc* modelDesc;
  int nnXLen;
  int nnYLen;
  int maxBatchSize;
  bool inputsUseNHWC;

#ifdef KATAGO_HAVE_WEBGPU
  // Persistent weight buffers, uploaded once and reused across every eval.
  // Keyed by the source vector's data pointer (conv/matmul/matbias raw weights)
  // and by the BatchNormLayerDesc address (folded scale+bias). The ModelDesc is
  // immutable for the handle's lifetime, so these pointers are stable keys.
  std::unordered_map<const void*, wgpu::Buffer> weightCache;
  std::unordered_map<const void*, std::pair<wgpu::Buffer, wgpu::Buffer>> bnCache;
  std::unordered_map<const void*, wgpu::Buffer> winogradCache;  // Winograd-transformed 3x3 filters
  BufferPool pool;  // reused intermediate/uniform buffers across evals
#endif

  ComputeHandle(
    ComputeContext* context_, const LoadedModel* loadedModel, int maxBatchSize_, bool inputsUseNHWC_
  )
    : context(context_),
      modelDesc(&loadedModel->modelDesc),
      nnXLen(context_->nnXLen),
      nnYLen(context_->nnYLen),
      maxBatchSize(maxBatchSize_),
      inputsUseNHWC(inputsUseNHWC_)
  {
    assertSupportedArchitectureOrThrow(*modelDesc);
    pool.ctx = context;  // weights/intermediates are uploaded lazily on first getOutput
  }
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)requireExactNNLen; (void)gpuIdxForThisThread;
  if(logger != NULL)
    logger->write(
      "WebGPU backend thread " + Global::intToString(serverThreadIdx) +
      " Model version " + Global::intToString(loadedModel->modelDesc.modelVersion) +
      " name: " + loadedModel->modelDesc.name +
      " (" + loadedModel->modelDesc.getShortInfoString() + ")");
  return new ComputeHandle(context, loadedModel, maxBatchSize, inputsUseNHWC);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->context->useFP16;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle; (void)isWarmup;
  return false;
}

void NeuralNet::printDevices() {
#ifdef KATAGO_HAVE_WEBGPU
  if(!gInstance)
    NeuralNet::globalInitialize();
  // The standard API requests one (preferred) adapter rather than enumerating;
  // report it. Power preference picks the discrete GPU where present.
  wgpu::Adapter adapter = requestAdapterSync(NULL);
  wgpu::AdapterInfo info{};
  adapter.GetInfo(&info);
  cout << "WebGPU default adapter: " << string(info.device.data, info.device.length) << endl;
#else
  cout << "WebGPU backend compiled without WebGPU headers." << endl;
#endif
}

// ----------------------------------------------------------------------------
// Input buffers: host staging (same shape as the OpenCL backend)
// ----------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  std::vector<float> userInputBuffer;
  std::vector<float> userInputGlobalBuffer;
  std::vector<float> userInputMetaBuffer;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSize_, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    int version = m.modelVersion;
    maxBatchSize = maxBatchSize_;
    singleInputElts = (size_t)NNModelVersion::getNumSpatialFeatures(version) * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)NNModelVersion::getNumGlobalFeatures(version);
    singleInputMetaElts = (size_t)m.numInputMetaChannels;

    userInputBuffer.resize(singleInputElts * maxBatchSize);
    userInputGlobalBuffer.resize(singleInputGlobalElts * maxBatchSize);
    userInputMetaBuffer.resize(singleInputMetaElts * maxBatchSize);
  }
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  delete buffers;
}

// ----------------------------------------------------------------------------
// getOutput: marshal inputs, then (for now) throw rather than return wrong evals
// ----------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->modelDesc->modelVersion;
  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = (int)inputBuffers->singleInputMetaElts;

  // Input marshaling is real and mirrors the OpenCL backend, so the dispatch
  // milestone can build straight on top of it.
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer.data() + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->userInputMetaBuffer.data() + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    if(numMetaFeatures > 0)
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures,
      gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

#ifdef KATAGO_HAVE_WEBGPU
  ComputeContext* ctx = gpuHandle->context;
  const ModelDesc& md = *gpuHandle->modelDesc;
  const TrunkDesc& trunk = md.trunk;
  const size_t hw = (size_t)nnXLen * nnYLen;

  // Activate this handle's buffer pool for the duration of the eval: intermediate
  // and uniform allocations are drawn from it and reused next eval. RAII-cleared
  // so an exception (or the one-shot test hooks) never leaves a dangling pool.
  struct PoolGuard {
    PoolGuard(BufferPool* p) { p->reset(); tlPool = p; }
    ~PoolGuard() { tlPool = nullptr; }
  } poolGuard(&gpuHandle->pool);
  const int trunkC = trunk.trunkNumChannels;

  // ---- Upload inputs; derive mask from spatial feature 0 (NCHW) ----
  wgpu::Buffer inBuf = wgMakeStorage(ctx, inputBuffers->userInputBuffer.data(),
                                     (size_t)batchSize * numSpatialFeatures * hw, false);
  wgpu::Buffer globalBuf = wgMakeStorage(ctx, inputBuffers->userInputGlobalBuffer.data(),
                                         (size_t)batchSize * numGlobalFeatures, false);
  std::vector<float> maskHost((size_t)batchSize * hw);
  for(int n = 0; n < batchSize; n++)
    for(size_t xy = 0; xy < hw; xy++)
      maskHost[n*hw + xy] = inputBuffers->userInputBuffer[((size_t)n*numSpatialFeatures + 0)*hw + xy];
  wgpu::Buffer maskBuf = wgMakeStorage(ctx, maskHost.data(), maskHost.size(), false);

  // ---- Op helpers (upload weights from desc, dispatch, return new buffer) ----
  // Persistent weight upload: cache GPU buffers on the handle, keyed by source
  // pointer, so every layer's weights upload once and are reused on later evals.
  // One recorder for the whole forward pass: all dispatches go into a single
  // command encoder and cost one Submit; the head outputs read back in one map.
  WgRecorder rec(ctx);

  auto wbuf = [&](const std::vector<float>& w) {
    auto& cache = gpuHandle->weightCache;
    auto it = cache.find(w.data());
    if(it != cache.end()) return it->second;
    wgpu::Buffer b = wgMakeStorage(ctx, w.data(), w.size(), false, /*pooled=*/false);
    cache.emplace(w.data(), b);
    return b;
  };
  // Shared-memory tiled GEMM for 1x1 conv / projections (into a caller buffer).
  // KATAGO_WEBGPU_NO_TILEDGEMM=1 forces the naive per-output kernels (A/B).
  const bool useTiledGemm = (std::getenv("KATAGO_WEBGPU_NO_TILEDGEMM") == nullptr);
  // Register tiling (2x2 micro-tile/thread) on top of shared-memory tiling.
  const char* gemmKernel = std::getenv("KATAGO_WEBGPU_NO_REGTILE") ? "tiledGemm" : "tiledGemmRT";
  auto tgemmInto = [&](wgpu::Buffer in, wgpu::Buffer out, int inC, int outC, wgpu::Buffer W, bool wInMajor) {
    TGParamsHost p{(uint32_t)batchSize,(uint32_t)inC,(uint32_t)outC,(uint32_t)hw, wInMajor?1u:0u, 0u,0u,0u};
    rec.dispatch(gemmKernel, {wgMakeUniform(ctx,p), in, W, out},
                 (uint32_t)((outC+15)/16), (uint32_t)((hw+15)/16), (uint32_t)batchSize);
  };
  auto bnbuf = [&](const BatchNormLayerDesc& bn) {
    auto& cache = gpuHandle->bnCache;
    auto it = cache.find(&bn);
    if(it != cache.end()) return it->second;
    std::vector<float> ms, mb; foldBatchNorm(&bn, ms, mb);
    std::pair<wgpu::Buffer, wgpu::Buffer> p{wgMakeStorage(ctx, ms.data(), ms.size(), false, /*pooled=*/false),
                                            wgMakeStorage(ctx, mb.data(), mb.size(), false, /*pooled=*/false)};
    cache.emplace(&bn, p);
    return p;
  };

  // Outputs are created CopySrc-capable so any of them can be read back; for a
  // tiny-board eval the extra usage flag is free. (Only the 5 head outputs
  // strictly need it; intermediate-buffer pooling is a later perf step.)
  // Winograd-transformed 3x3 filters, transformed once on host and cached on the handle.
  auto wbufWino = [&](const ConvLayerDesc& cd, int outC, int inC) {
    auto& cache = gpuHandle->winogradCache;
    const void* key = cd.weights.data();
    auto it = cache.find(key);
    if(it != cache.end()) return it->second;
    std::vector<float> U = winogradFilterTransform(cd.weights, outC, inC);
    wgpu::Buffer b = wgMakeStorage(ctx, U.data(), U.size(), false);
    cache.emplace(key, b);
    return b;
  };
  auto conv = [&](wgpu::Buffer in, int inC, int outC, const ConvLayerDesc& cd) {
    size_t outElts = (size_t)batchSize * outC * hw;
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, outElts, true);
    if(cd.convXSize == 1 && cd.convYSize == 1) {
      // 1x1 = per-pixel channel GEMM. Tiled (shared-memory) or naive.
      if(useTiledGemm) tgemmInto(in, out, inC, outC, wbuf(cd.weights), /*wInMajor=*/false);
      else {
        Conv1x1ParamsHost p{(uint32_t)batchSize, (uint32_t)inC, (uint32_t)outC, (uint32_t)hw};
        rec.dispatch("conv1x1NCHW", {wgMakeUniform(ctx,p), in, wbuf(cd.weights), out}, (uint32_t)((outElts+63)/64));
      }
    } else if(!std::getenv("KATAGO_WEBGPU_NO_WINOGRAD")
              && cd.convXSize == 3 && cd.convYSize == 3 && cd.dilationX == 1 && cd.dilationY == 1) {
      // Winograd F(2,3): 4 mults/output vs 9 (validated by testEvaluateConv).
      // KATAGO_WEBGPU_NO_WINOGRAD=1 forces the direct conv (for A/B benchmarking).
      const int nTilesX = (nnXLen+1)/2, nTilesY = (nnYLen+1)/2, nTiles = nTilesX*nTilesY;
      WgParamsHost p{(uint32_t)batchSize,(uint32_t)inC,(uint32_t)outC,(uint32_t)nnYLen,(uint32_t)nnXLen,
                     (uint32_t)nTilesX,(uint32_t)nTilesY,0u};
      wgpu::Buffer pbuf = wgMakeUniform(ctx, p);
      wgpu::Buffer V = wgMakeStorage(ctx, nullptr, (size_t)16*batchSize*inC*nTiles, false);
      wgpu::Buffer M = wgMakeStorage(ctx, nullptr, (size_t)16*batchSize*outC*nTiles, false);
      rec.dispatch("winogradInput",  {pbuf, in, V}, (uint32_t)(((size_t)batchSize*inC*nTiles+63)/64));
      rec.dispatch("winogradMatmul", {pbuf, wbufWino(cd,outC,inC), V, M}, (uint32_t)(((size_t)16*batchSize*outC*nTiles+63)/64));
      rec.dispatch("winogradOutput", {pbuf, M, out}, (uint32_t)(((size_t)batchSize*outC*nTiles+63)/64));
    } else {
      ConvParamsHost p{}; p.n=batchSize; p.inC=(uint32_t)inC; p.outC=(uint32_t)outC; p.h=(uint32_t)nnYLen; p.w=(uint32_t)nnXLen;
      p.fy=(uint32_t)cd.convYSize; p.fx=(uint32_t)cd.convXSize; p.dilY=(uint32_t)cd.dilationY; p.dilX=(uint32_t)cd.dilationX;
      rec.dispatch("conv2dNCHW", {wgMakeUniform(ctx,p), in, wbuf(cd.weights), out}, (uint32_t)((outElts+63)/64));
    }
    return out;
  };
  auto bnAct = [&](wgpu::Buffer in, int C, const BatchNormLayerDesc& bn, int actCode) {
    auto sb = bnbuf(bn);
    size_t elts = (size_t)batchSize * C * hw;
    SBMAParamsHost p{(uint32_t)batchSize,(uint32_t)C,(uint32_t)hw,(uint32_t)actCode};
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, elts, true);
    rec.dispatch("scaleBiasMaskAct",
      {wgMakeUniform(ctx,p), in, sb.first, sb.second, maskBuf, out}, (uint32_t)((elts+63)/64));
    return out;
  };
  // Fused bnAct -> 1x1 conv (pre-activation block pattern): one dispatch, no
  // intermediate. KATAGO_WEBGPU_NO_FUSION=1 falls back to separate bnAct + conv.
  const bool useFusion = useTiledGemm && (std::getenv("KATAGO_WEBGPU_NO_FUSION") == nullptr);
  auto convBnAct1x1 = [&](wgpu::Buffer in, int inC, int outC, const ConvLayerDesc& cd, const BatchNormLayerDesc& bn, int actCode) {
    auto sb = bnbuf(bn);
    TGBParamsHost p{(uint32_t)batchSize,(uint32_t)inC,(uint32_t)outC,(uint32_t)hw,(uint32_t)actCode,0u,0u,0u};
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, (size_t)batchSize*outC*hw, true);
    rec.dispatch("tiledGemmInBnAct", {wgMakeUniform(ctx,p), in, wbuf(cd.weights), sb.first, sb.second, maskBuf, out},
                 (uint32_t)((outC+15)/16), (uint32_t)((hw+15)/16), (uint32_t)batchSize);
    return out;
  };
  auto matmul = [&](wgpu::Buffer a, int M, int K, int O, const std::vector<float>& W, const std::vector<float>* bias, int actCode) {
    MMParamsHost p{(uint32_t)M,(uint32_t)K,(uint32_t)O,(uint32_t)actCode, bias?1u:0u, 0u,0u,0u};
    wgpu::Buffer biasB = bias ? wbuf(*bias) : wgMakeStorage(ctx,nullptr,1,false);
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, (size_t)M*O, true);
    rec.dispatch("matMulBiasAct", {wgMakeUniform(ctx,p), a, wbuf(W), biasB, out},
      (uint32_t)((M+7)/8), (uint32_t)((O+7)/8));
    return out;
  };
  auto gpool = [&](const char* entry, wgpu::Buffer in, int C) {
    GPParamsHost p{(uint32_t)batchSize,(uint32_t)C,(uint32_t)hw,0u};
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, (size_t)batchSize*3*C, false);
    rec.dispatch(entry, {wgMakeUniform(ctx,p), in, maskBuf, out}, (uint32_t)(batchSize*C));
    return out;
  };
  auto addChanBias = [&](wgpu::Buffer data, wgpu::Buffer bias, int C) {
    AddBiasParamsHost p{(uint32_t)batchSize,(uint32_t)C,(uint32_t)hw,0u};
    rec.dispatch("addChannelBias", {wgMakeUniform(ctx,p), bias, data}, (uint32_t)(((size_t)batchSize*C*hw+63)/64));
  };
  auto addInPlace = [&](wgpu::Buffer dst, wgpu::Buffer src, size_t elts) {
    AddParamsHost p{(uint32_t)elts,0u,0u,0u};
    rec.dispatch("addInPlace", {wgMakeUniform(ctx,p), src, dst}, (uint32_t)((elts+63)/64));
  };
  // ---- Transformer / RMSNorm primitives (modelVersion >= 15) ----
  // RMSNorm over channels per position. beta=null -> transformer preLN (weight only).
  auto rmsNorm = [&](wgpu::Buffer in, int Ch, const std::vector<float>& gamma, const std::vector<float>* beta, int actCode, float eps) {
    RMSParamsHost p{(uint32_t)batchSize,(uint32_t)Ch,(uint32_t)hw,(uint32_t)actCode, eps, beta?1u:0u, 0u,0u};
    wgpu::Buffer betaB = beta ? wbuf(*beta) : wgMakeStorage(ctx,nullptr,1,false);
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, (size_t)batchSize*Ch*hw, true);
    rec.dispatch("rmsNorm", {wgMakeUniform(ctx,p), in, wbuf(gamma), betaB, maskBuf, out},
                 (uint32_t)(((size_t)batchSize*hw+63)/64));
    return out;
  };
  // Per-position projection with MatMulLayerDesc [inC][outC] weights (bias-free).
  auto proj = [&](wgpu::Buffer in, int inC, int outC, const std::vector<float>& W) {
    size_t outElts = (size_t)batchSize*outC*hw;
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, outElts, true);
    if(useTiledGemm) tgemmInto(in, out, inC, outC, wbuf(W), /*wInMajor=*/true);
    else {
      ProjParamsHost p{(uint32_t)batchSize,(uint32_t)inC,(uint32_t)outC,(uint32_t)hw};
      rec.dispatch("proj1x1", {wgMakeUniform(ctx,p), in, wbuf(W), out}, (uint32_t)((outElts+63)/64));
    }
    return out;
  };
  auto rope = [&](wgpu::Buffer data, int numHeads, int headDim, int numPairs, int numKVHeads, bool learnable, wgpu::Buffer cosB, wgpu::Buffer sinB) {
    RopeParamsHost p{(uint32_t)batchSize,(uint32_t)numHeads,(uint32_t)headDim,(uint32_t)numPairs,(uint32_t)hw,(uint32_t)numKVHeads,learnable?1u:0u,0u};
    size_t total = (size_t)batchSize*numHeads*numPairs*hw;
    rec.dispatch("ropeApply", {wgMakeUniform(ctx,p), data, cosB, sinB}, (uint32_t)((total+63)/64));
  };
  auto attnScores = [&](wgpu::Buffer q, wgpu::Buffer k, int nH, int nKV, int headDim, int kvGroup, float scale) {
    AttnSParamsHost p{(uint32_t)batchSize,(uint32_t)nH,(uint32_t)nKV,(uint32_t)headDim,(uint32_t)hw,(uint32_t)kvGroup, scale, 0u};
    size_t total = (size_t)batchSize*nH*hw*hw;
    wgpu::Buffer scores = wgMakeStorage(ctx, nullptr, total, true);
    rec.dispatch("attnScores", {wgMakeUniform(ctx,p), q, k, scores}, (uint32_t)((total+63)/64));
    return scores;
  };
  auto attnSoftmax = [&](wgpu::Buffer scores, int nH) {
    AttnSMParamsHost p{(uint32_t)batchSize,(uint32_t)nH,(uint32_t)hw,0u};
    size_t total = (size_t)batchSize*nH*hw;
    rec.dispatch("attnSoftmax", {wgMakeUniform(ctx,p), scores, maskBuf}, (uint32_t)((total+63)/64));
  };
  auto attnOutput = [&](wgpu::Buffer scores, wgpu::Buffer v, int nH, int nKV, int vHeadDim, int kvGroup) {
    AttnOParamsHost p{(uint32_t)batchSize,(uint32_t)nH,(uint32_t)nKV,(uint32_t)vHeadDim,(uint32_t)hw,(uint32_t)kvGroup,0u,0u};
    size_t total = (size_t)batchSize*nH*vHeadDim*hw;
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, total, true);
    rec.dispatch("attnOutput", {wgMakeUniform(ctx,p), scores, v, out}, (uint32_t)((total+63)/64));
    return out;
  };
  // Flash Attention: fused single-kernel attention (online softmax), no [seq x seq]
  // score matrix. KATAGO_WEBGPU_NO_FLASHATTN=1 forces the 3-kernel path (A/B).
  const bool useFlashAttn = (std::getenv("KATAGO_WEBGPU_NO_FLASHATTN") == nullptr);
  auto flashAttn = [&](wgpu::Buffer q, wgpu::Buffer k, wgpu::Buffer v, int nH, int nKV, int qHD, int vHD, int kvGroup, float scale) {
    FlashParamsHost p{(uint32_t)batchSize,(uint32_t)nH,(uint32_t)nKV,(uint32_t)qHD,(uint32_t)vHD,(uint32_t)hw,(uint32_t)kvGroup, scale};
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, (size_t)batchSize*nH*vHD*hw, true);
    rec.dispatch("flashAttention", {wgMakeUniform(ctx,p), q, k, v, maskBuf, out}, (uint32_t)(((size_t)batchSize*nH*hw+63)/64));
    return out;
  };
  auto swiglu = [&](wgpu::Buffer a, wgpu::Buffer gate, size_t total) {
    SwiParamsHost p{(uint32_t)total,0u,0u,0u};
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, total, true);
    rec.dispatch("swigluGate", {wgMakeUniform(ctx,p), a, gate, out}, (uint32_t)((total+63)/64));
    return out;
  };
  // Spatial RMSNorm (trunk tip): reduce one rms per batch element, then apply.
  auto rmsNormSpatial = [&](wgpu::Buffer in, int Ch, const std::vector<float>& gamma, const std::vector<float>& beta, int actCode, float eps) {
    RMSRParamsHost pr{(uint32_t)batchSize,(uint32_t)Ch,(uint32_t)hw,0u, eps, 0u,0u,0u};
    wgpu::Buffer rmsv = wgMakeStorage(ctx, nullptr, batchSize, false);  // f32 [n]
    rec.dispatch("rmsReduceSpatial", {wgMakeUniform(ctx,pr), in, maskBuf, rmsv}, (uint32_t)((batchSize+63)/64));
    RMSAParamsHost pa{(uint32_t)batchSize,(uint32_t)Ch,(uint32_t)hw,(uint32_t)actCode, 1u,0u,0u,0u};
    wgpu::Buffer out = wgMakeStorage(ctx, nullptr, (size_t)batchSize*Ch*hw, true);
    rec.dispatch("rmsApplySpatial", {wgMakeUniform(ctx,pa), in, wbuf(gamma), wbuf(beta), maskBuf, rmsv, out},
                 (uint32_t)(((size_t)batchSize*Ch*hw+63)/64));
    return out;
  };

  // ---- Trunk ----
  wgpu::Buffer x = conv(inBuf, numSpatialFeatures, trunkC, trunk.initialConv);
  wgpu::Buffer gbias = matmul(globalBuf, batchSize, numGlobalFeatures, trunk.initialMatMul.outChannels,
                              trunk.initialMatMul.weights, nullptr, ACTIVATION_IDENTITY);
  addChanBias(x, gbias, trunkC);

  // Process a block list operating at channel width C, mutating buffer `bx` in
  // place. Recursive so nested-bottleneck sub-blocks reuse the same code at the
  // bottleneck width. (Transformer blocks are rejected at load.)
  std::function<void(const std::vector<std::pair<int,unique_ptr_void>>&, wgpu::Buffer, int)> applyBlocks =
    [&](const std::vector<std::pair<int,unique_ptr_void>>& blocks, wgpu::Buffer bx, int C) {
    for(const auto& kindAndBlock : blocks) {
      if(kindAndBlock.first == ORDINARY_BLOCK_KIND) {
        const ResidualBlockDesc* b = (const ResidualBlockDesc*)kindAndBlock.second.get();
        int midC = b->regularConv.outChannels;
        wgpu::Buffer s1 = bnAct(bx, C, b->preBN, b->preActivation.activation);
        wgpu::Buffer mid = conv(s1, C, midC, b->regularConv);
        wgpu::Buffer s2 = bnAct(mid, midC, b->midBN, b->midActivation.activation);
        wgpu::Buffer fin = conv(s2, midC, C, b->finalConv);
        addInPlace(bx, fin, (size_t)batchSize*C*hw);
      }
      else if(kindAndBlock.first == GLOBAL_POOLING_BLOCK_KIND) {
        const GlobalPoolingResidualBlockDesc* b = (const GlobalPoolingResidualBlockDesc*)kindAndBlock.second.get();
        int regularC = b->regularConv.outChannels, gpoolC = b->gpoolConv.outChannels;
        wgpu::Buffer ts = bnAct(bx, C, b->preBN, b->preActivation.activation);
        wgpu::Buffer reg = conv(ts, C, regularC, b->regularConv);
        wgpu::Buffer gp = conv(ts, C, gpoolC, b->gpoolConv);
        wgpu::Buffer gp2 = bnAct(gp, gpoolC, b->gpoolBN, b->gpoolActivation.activation);
        wgpu::Buffer cat = gpool("globalPoolMeanMax", gp2, gpoolC);
        wgpu::Buffer bias = matmul(cat, batchSize, 3*gpoolC, regularC, b->gpoolToBiasMul.weights, nullptr, ACTIVATION_IDENTITY);
        addChanBias(reg, bias, regularC);
        wgpu::Buffer s2 = bnAct(reg, regularC, b->midBN, b->midActivation.activation);
        wgpu::Buffer fin = conv(s2, regularC, C, b->finalConv);
        addInPlace(bx, fin, (size_t)batchSize*C*hw);
      }
      else if(kindAndBlock.first == NESTED_BOTTLENECK_BLOCK_KIND) {
        const NestedBottleneckResidualBlockDesc* b = (const NestedBottleneckResidualBlockDesc*)kindAndBlock.second.get();
        int bottleC = b->preConv.outChannels;
        bool pre1x1  = b->preConv.convXSize == 1 && b->preConv.convYSize == 1;
        bool post1x1 = b->postConv.convXSize == 1 && b->postConv.convYSize == 1;
        // m = preConv(act(preBN(x)))         -- project down (fused if 1x1)
        wgpu::Buffer mid;
        if(useFusion && pre1x1) mid = convBnAct1x1(bx, C, bottleC, b->preConv, b->preBN, b->preActivation.activation);
        else { wgpu::Buffer pre = bnAct(bx, C, b->preBN, b->preActivation.activation); mid = conv(pre, C, bottleC, b->preConv); }
        // sub-blocks at bottleneck width, in place on mid
        applyBlocks(b->blocks, mid, bottleC);
        // x += postConv(act(postBN(m)))      -- project up, residual to x
        wgpu::Buffer fin;
        if(useFusion && post1x1) fin = convBnAct1x1(mid, bottleC, C, b->postConv, b->postBN, b->postActivation.activation);
        else { wgpu::Buffer post = bnAct(mid, bottleC, b->postBN, b->postActivation.activation); fin = conv(post, bottleC, C, b->postConv); }
        addInPlace(bx, fin, (size_t)batchSize*C*hw);
      }
      else if(kindAndBlock.first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        const TransformerAttentionDesc* a = (const TransformerAttentionDesc*)kindAndBlock.second.get();
        int nH = a->numHeads, nKV = a->numKVHeads, qHD = a->qHeadDim, vHD = a->vHeadDim;
        int qTot = nH*qHD, kTot = nKV*qHD, vTot = nKV*vHD;
        // preLN (transformer RMSNorm: weight only, no beta/act), then q/k/v projections.
        wgpu::Buffer normed = rmsNorm(bx, C, a->preLN.weight, nullptr, ACTIVATION_IDENTITY, a->preLN.epsilon);
        wgpu::Buffer q = proj(normed, C, qTot, a->qProj.weights);
        wgpu::Buffer k = proj(normed, C, kTot, a->kProj.weights);
        wgpu::Buffer v = proj(normed, C, vTot, a->vProj.weights);
        if(a->useRope) {
          std::vector<float> cosT, sinT;
          a->computeRopeCosSin(nnXLen, nnYLen, hw, cosT, sinT);
          wgpu::Buffer cosB = wgMakeStorage(ctx, cosT.data(), cosT.size(), false, /*pooled=*/false);
          wgpu::Buffer sinB = wgMakeStorage(ctx, sinT.data(), sinT.size(), false, /*pooled=*/false);
          int numPairs = qHD/2;
          rope(q, nH, qHD, numPairs, nKV, a->learnableRope, cosB, sinB);
          rope(k, nKV, qHD, numPairs, nKV, a->learnableRope, cosB, sinB);
        }
        int kvGroup = nH / nKV;
        float scale = 1.0f/std::sqrt((float)qHD);
        wgpu::Buffer attnOut;
        if(useFlashAttn) {
          attnOut = flashAttn(q, k, v, nH, nKV, qHD, vHD, kvGroup, scale);  // fused, no score matrix
        } else {
          wgpu::Buffer scores = attnScores(q, k, nH, nKV, qHD, kvGroup, scale);
          attnSoftmax(scores, nH);
          attnOut = attnOutput(scores, v, nH, nKV, vHD, kvGroup);
        }  // [n, nH*vHD, hw]
        // outProj input is numHeads*vHeadDim (concatenated heads) — NOT vTot
        // (numKVHeads*vHeadDim); they differ under grouped-query attention.
        wgpu::Buffer projOut = proj(attnOut, nH*vHD, C, a->outProj.weights);  // -> trunk width
        addInPlace(bx, projOut, (size_t)batchSize*C*hw);
      }
      else {  // TRANSFORMER_FFN_BLOCK_KIND (SwiGLU)
        const TransformerFFNDesc* f = (const TransformerFFNDesc*)kindAndBlock.second.get();
        int ffnC = f->ffnChannels;
        wgpu::Buffer normed = rmsNorm(bx, C, f->preLN.weight, nullptr, ACTIVATION_IDENTITY, f->preLN.epsilon);
        wgpu::Buffer a1 = proj(normed, C, ffnC, f->linear1.weights);
        wgpu::Buffer gate = proj(normed, C, ffnC, f->linearGate.weights);
        wgpu::Buffer hidden = swiglu(a1, gate, (size_t)batchSize*ffnC*hw);  // silu(a1)*gate
        wgpu::Buffer out = proj(hidden, ffnC, C, f->linear2.weights);
        addInPlace(bx, out, (size_t)batchSize*C*hw);
      }
    }
  };
  applyBlocks(trunk.blocks, x, trunkC);
  // Trunk tip: RMSNorm (v15+ RMSNorm nets) or BatchNorm (legacy). RMSNorm may be
  // spatial (one rms over the whole board) or per-position.
  wgpu::Buffer trunkOut;
  if(trunk.trunkNormKind == TRUNK_NORM_KIND_RMSNORM) {
    const RMSNormLayerDesc& tn = trunk.trunkTipRMSNorm;
    int act = trunk.trunkTipActivation.activation;
    trunkOut = tn.spatial ? rmsNormSpatial(x, trunkC, tn.gamma, tn.beta, act, tn.epsilon)
                          : rmsNorm(x, trunkC, tn.gamma, &tn.beta, act, tn.epsilon);
  } else
    trunkOut = bnAct(x, trunkC, trunk.trunkTipBN, trunk.trunkTipActivation.activation);

  // ---- Policy head ----
  const PolicyHeadDesc& ph = md.policyHead;
  int p1C = ph.p1Conv.outChannels, g1C = ph.g1Conv.outChannels;
  int numPolicyChannels = md.numPolicyChannels;  // 1, or 2 with optimism (use ch.0)
  wgpu::Buffer p1 = conv(trunkOut, trunkC, p1C, ph.p1Conv);
  wgpu::Buffer g1 = conv(trunkOut, trunkC, g1C, ph.g1Conv);
  wgpu::Buffer g1b = bnAct(g1, g1C, ph.g1BN, ph.g1Activation.activation);
  wgpu::Buffer g1cat = gpool("globalPoolMeanMax", g1b, g1C);
  wgpu::Buffer g1bias = matmul(g1cat, batchSize, 3*g1C, p1C, ph.gpoolToBiasMul.weights, nullptr, ACTIVATION_IDENTITY);
  addChanBias(p1, g1bias, p1C);
  wgpu::Buffer p1b = bnAct(p1, p1C, ph.p1BN, ph.p1Activation.activation);
  wgpu::Buffer policyBuf = conv(p1b, p1C, numPolicyChannels, ph.p2Conv);
  // Pass logit. v<15: policyPass = gpoolToPassMul(g1cat). v>=15: an extra hidden
  // layer — passAct(gpoolToPassMul(g1cat) + gpoolToPassBias) then gpoolToPassMul2.
  wgpu::Buffer policyPassBuf;
  if(modelVersion >= 15) {
    int passHidden = ph.gpoolToPassMul.outChannels;
    wgpu::Buffer h1 = matmul(g1cat, batchSize, 3*g1C, passHidden,
                             ph.gpoolToPassMul.weights, &ph.gpoolToPassBias.weights, ph.passActivation.activation);
    policyPassBuf = matmul(h1, batchSize, passHidden, numPolicyChannels,
                           ph.gpoolToPassMul2.weights, nullptr, ACTIVATION_IDENTITY);
  } else {
    policyPassBuf = matmul(g1cat, batchSize, 3*g1C, numPolicyChannels, ph.gpoolToPassMul.weights, nullptr, ACTIVATION_IDENTITY);
  }

  // ---- Value head ----
  const ValueHeadDesc& vh = md.valueHead;
  int v1C = vh.v1Conv.outChannels, v2C = vh.v2Mul.outChannels;
  int numValueChannels = md.numValueChannels;          // 3
  int numSV = md.numScoreValueChannels;
  int numOwn = md.numOwnershipChannels;                // 1
  wgpu::Buffer v1 = conv(trunkOut, trunkC, v1C, vh.v1Conv);
  wgpu::Buffer v1b = bnAct(v1, v1C, vh.v1BN, vh.v1Activation.activation);
  wgpu::Buffer v1mean = gpool("globalPoolValueHead", v1b, v1C);
  wgpu::Buffer v2 = matmul(v1mean, batchSize, 3*v1C, v2C, vh.v2Mul.weights, &vh.v2Bias.weights, vh.v2Activation.activation);
  wgpu::Buffer valueBuf = matmul(v2, batchSize, v2C, numValueChannels, vh.v3Mul.weights, &vh.v3Bias.weights, ACTIVATION_IDENTITY);
  wgpu::Buffer svBuf = matmul(v2, batchSize, v2C, numSV, vh.sv3Mul.weights, &vh.sv3Bias.weights, ACTIVATION_IDENTITY);
  wgpu::Buffer ownBuf = conv(v1b, v1C, numOwn, vh.vOwnershipConv);

  // ---- Submit the whole frame once and read all head outputs back in one map ----
  std::vector<float> policyHost, policyPassHost, valueHost, svHost, ownHost;
  std::vector<std::pair<wgpu::Buffer,size_t>> srcs = {
    {policyBuf, (size_t)batchSize*numPolicyChannels*hw},
    {policyPassBuf, (size_t)batchSize*numPolicyChannels},
    {valueBuf, (size_t)batchSize*numValueChannels},
    {svBuf, (size_t)batchSize*numSV},
    {ownBuf, (size_t)batchSize*numOwn*hw},
  };
  std::vector<std::vector<float>*> outs = {&policyHost, &policyPassHost, &valueHost, &svHost, &ownHost};
  rec.submitAndReadback(srcs, outs);

  // ---- Write NNOutputs (logits; NNEvaluator applies softmax/tanh) ----
  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    const int symmetry = inputBufs[row]->symmetry;
    const float* policySrc = policyHost.data() + (size_t)row*numPolicyChannels*hw;
    const float* passSrc = policyPassHost.data() + (size_t)row*numPolicyChannels;
    SymmetryHelpers::copyOutputsWithSymmetry(policySrc, output->policyProbs, 1, nnYLen, nnXLen, symmetry);
    output->policyProbs[nnXLen*nnYLen] = passSrc[0];
    output->policyOptimismUsed = 0.0f;

    const float* v = valueHost.data() + (size_t)row*numValueChannels;
    output->whiteWinProb = v[0];
    output->whiteLossProb = v[1];
    output->whiteNoResultProb = v[2];

    const float* sv = svHost.data() + (size_t)row*numSV;
    if(numSV >= 6) {
      output->whiteScoreMean = sv[0]; output->whiteScoreMeanSq = sv[1]; output->whiteLead = sv[2];
      output->varTimeLeft = sv[3]; output->shorttermWinlossError = sv[4]; output->shorttermScoreError = sv[5];
    } else if(numSV >= 4) {
      output->whiteScoreMean = sv[0]; output->whiteScoreMeanSq = sv[1]; output->whiteLead = sv[2];
      output->varTimeLeft = sv[3]; output->shorttermWinlossError = 0; output->shorttermScoreError = 0;
    } else if(numSV >= 2) {
      output->whiteScoreMean = sv[0]; output->whiteScoreMeanSq = sv[1]; output->whiteLead = sv[0];
      output->varTimeLeft = 0; output->shorttermWinlossError = 0; output->shorttermScoreError = 0;
    } else {
      output->whiteScoreMean = sv[0]; output->whiteScoreMeanSq = sv[0]*sv[0]; output->whiteLead = sv[0];
      output->varTimeLeft = 0; output->shorttermWinlossError = 0; output->shorttermScoreError = 0;
    }

    if(output->whiteOwnerMap != NULL) {
      const float* ownSrc = ownHost.data() + (size_t)row*numOwn*hw;
      SymmetryHelpers::copyOutputsWithSymmetry(ownSrc, output->whiteOwnerMap, 1, nnYLen, nnXLen, symmetry);
    }
  }
#else
  (void)gpuHandle; (void)outputs;
  throw StringError("WebGPU backend compiled without WebGPU headers. See WEBGPU_STATUS.md.");
#endif
}

// ----------------------------------------------------------------------------
// Testing hooks — return false until the corresponding kernels are validated.
// These are the intended *first* things to implement: each maps onto one WGSL
// entry point and gives a direct numeric check against the reference backends.
// ----------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC,
  const std::vector<float>& inputBuffer, std::vector<float>& outputBuffer) {
#ifdef KATAGO_HAVE_WEBGPU
  // Scaffold supports the NCHW fp32 path only (mirrors OpenCL's test hook, which
  // skips NHWC/fp16). The harness uploads input+weights, dispatches conv2dNCHW,
  // and reads back; the test compares against its own CPU reference.
  if(useFP16 || useNHWC)
    return false;

  const size_t hw = (size_t)nnXLen * nnYLen;
  const size_t inElts = (size_t)batchSize * desc->inChannels * hw;
  const size_t outElts = (size_t)batchSize * desc->outChannels * hw;
  if(inputBuffer.size() != inElts)
    throw StringError("WebGPU testEvaluateConv: unexpected input buffer size");

  std::unique_ptr<ComputeContext> ctx(new ComputeContext(nnXLen, nnYLen, enabled_t::False));
  initContextGpu(ctx.get(), NULL);

  // 3x3 stride-1 dil-1 -> Winograd F(2,3). Validates the 3-stage winograd kernels
  // against the CPU reference (the layer test's 3x3 conv case routes through here).
  if(desc->convXSize == 3 && desc->convYSize == 3 && desc->dilationX == 1 && desc->dilationY == 1) {
    const int inC = desc->inChannels, outC = desc->outChannels;
    const int nTilesX = (nnXLen + 1) / 2, nTilesY = (nnYLen + 1) / 2;
    const int nTiles = nTilesX * nTilesY;
    std::vector<float> U = winogradFilterTransform(desc->weights, outC, inC);
    WgParamsHost wp{(uint32_t)batchSize,(uint32_t)inC,(uint32_t)outC,(uint32_t)nnYLen,(uint32_t)nnXLen,
                    (uint32_t)nTilesX,(uint32_t)nTilesY,0u};
    wgpu::Buffer wpBuf = wgMakeUniform(ctx.get(), wp);
    wgpu::Buffer inBuf = wgMakeStorage(ctx.get(), inputBuffer.data(), inElts, false);
    wgpu::Buffer Ubuf  = wgMakeStorage(ctx.get(), U.data(), U.size(), false);
    wgpu::Buffer Vbuf  = wgMakeStorage(ctx.get(), nullptr, (size_t)16 * batchSize * inC * nTiles, false);
    wgpu::Buffer Mbuf  = wgMakeStorage(ctx.get(), nullptr, (size_t)16 * batchSize * outC * nTiles, false);
    wgpu::Buffer outBuf = wgMakeStorage(ctx.get(), nullptr, outElts, true);
    wgDispatch(ctx.get(), "winogradInput",  {wpBuf, inBuf, Vbuf},
               (uint32_t)(((size_t)batchSize * inC * nTiles + 63) / 64));
    wgDispatch(ctx.get(), "winogradMatmul", {wpBuf, Ubuf, Vbuf, Mbuf},
               (uint32_t)(((size_t)16 * batchSize * outC * nTiles + 63) / 64));
    wgDispatch(ctx.get(), "winogradOutput", {wpBuf, Mbuf, outBuf},
               (uint32_t)(((size_t)batchSize * outC * nTiles + 63) / 64));
    wgReadFloats(ctx.get(), outBuf, outElts, outputBuffer);
    return true;
  }

  ConvParamsHost p{};
  p.n = (uint32_t)batchSize;
  p.inC = (uint32_t)desc->inChannels;
  p.outC = (uint32_t)desc->outChannels;
  p.h = (uint32_t)nnYLen;
  p.w = (uint32_t)nnXLen;
  p.fy = (uint32_t)desc->convYSize;
  p.fx = (uint32_t)desc->convXSize;
  p.dilY = (uint32_t)desc->dilationY;
  p.dilX = (uint32_t)desc->dilationX;

  wgpu::Buffer uBuf = wgMakeUniform(ctx.get(), p);
  wgpu::Buffer inBuf = wgMakeStorage(ctx.get(), inputBuffer.data(), inElts, false);
  wgpu::Buffer wBuf = wgMakeStorage(ctx.get(), desc->weights.data(), desc->weights.size(), false);
  wgpu::Buffer outBuf = wgMakeStorage(ctx.get(), nullptr, outElts, true);

  wgDispatch(ctx.get(), "conv2dNCHW", {uBuf, inBuf, wBuf, outBuf}, (uint32_t)((outElts + 63) / 64));
  wgReadFloats(ctx.get(), outBuf, outElts, outputBuffer);
  return true;
#else
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen; (void)useFP16; (void)useNHWC;
  (void)inputBuffer; (void)outputBuffer;
  return false;
#endif
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC,
  const std::vector<float>& inputBuffer, const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
#ifdef KATAGO_HAVE_WEBGPU
  if(useFP16 || useNHWC)
    return false;

  const int C = desc->numChannels;
  const size_t hw = (size_t)nnXLen * nnYLen;
  const size_t elts = (size_t)batchSize * C * hw;
  if(inputBuffer.size() != elts)
    throw StringError("WebGPU testEvaluateBatchNorm: unexpected input buffer size");

  // Fold batchnorm into per-channel scale/bias (computeMerged semantics).
  // Activation is identity here (the test wraps batchnorm with ACTIVATION_IDENTITY).
  std::vector<float> mergedScale, mergedBias;
  foldBatchNorm(desc, mergedScale, mergedBias);

  std::unique_ptr<ComputeContext> ctx(new ComputeContext(nnXLen, nnYLen, enabled_t::False));
  initContextGpu(ctx.get(), NULL);

  SBMAParamsHost p{};
  p.n = (uint32_t)batchSize;
  p.c = (uint32_t)C;
  p.hw = (uint32_t)hw;
  p.act = (uint32_t)ACTIVATION_IDENTITY;

  wgpu::Buffer uBuf = wgMakeUniform(ctx.get(), p);
  wgpu::Buffer inBuf = wgMakeStorage(ctx.get(), inputBuffer.data(), elts, false);
  wgpu::Buffer scaleBuf = wgMakeStorage(ctx.get(), mergedScale.data(), C, false);
  wgpu::Buffer biasBuf = wgMakeStorage(ctx.get(), mergedBias.data(), C, false);
  wgpu::Buffer maskBuf = wgMakeStorage(ctx.get(), maskBuffer.data(), maskBuffer.size(), false);
  wgpu::Buffer outBuf = wgMakeStorage(ctx.get(), nullptr, elts, true);

  wgDispatch(ctx.get(), "scaleBiasMaskAct",
             {uBuf, inBuf, scaleBuf, biasBuf, maskBuf, outBuf}, (uint32_t)((elts + 63) / 64));
  wgReadFloats(ctx.get(), outBuf, elts, outputBuffer);
  return true;
#else
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen; (void)useFP16; (void)useNHWC;
  (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
#endif
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC,
  const std::vector<float>& inputBuffer, const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
#ifdef KATAGO_HAVE_WEBGPU
  if(useFP16 || useNHWC)
    return false;

  // Pre-activation ResNet block, mirroring NormActConv + NormActConv in the
  // reference backends:
  //   s1   = mask * preAct(preBN(trunk))         [scaleBiasMaskAct]
  //   mid  = regularConv(s1)                      [conv2dNCHW]
  //   s2   = mask * midAct(midBN(mid))            [scaleBiasMaskAct]
  //   out  = trunk + finalConv(s2)                [conv2dNCHW + addInPlace]
  // The final residual sum is NOT re-masked (matches the reference).
  const int trunkC = desc->preBN.numChannels;          // = regularConv.inChannels = finalConv.outChannels
  const int midC = desc->regularConv.outChannels;       // = midBN.numChannels = finalConv.inChannels
  const size_t hw = (size_t)nnXLen * nnYLen;
  const size_t trunkElts = (size_t)batchSize * trunkC * hw;
  const size_t midElts = (size_t)batchSize * midC * hw;
  if(inputBuffer.size() != trunkElts)
    throw StringError("WebGPU testEvaluateResidualBlock: unexpected input buffer size");

  std::vector<float> preScale, preBias, midScale, midBias;
  foldBatchNorm(&desc->preBN, preScale, preBias);
  foldBatchNorm(&desc->midBN, midScale, midBias);

  std::unique_ptr<ComputeContext> ctx(new ComputeContext(nnXLen, nnYLen, enabled_t::False));
  initContextGpu(ctx.get(), NULL);

  // trunkBuf holds the input and, after the final add, the output.
  wgpu::Buffer trunkBuf = wgMakeStorage(ctx.get(), inputBuffer.data(), trunkElts, true);
  wgpu::Buffer maskBuf = wgMakeStorage(ctx.get(), maskBuffer.data(), maskBuffer.size(), false);
  wgpu::Buffer s1Buf = wgMakeStorage(ctx.get(), nullptr, trunkElts, false);
  wgpu::Buffer midBuf = wgMakeStorage(ctx.get(), nullptr, midElts, false);
  wgpu::Buffer s2Buf = wgMakeStorage(ctx.get(), nullptr, midElts, false);
  wgpu::Buffer finalBuf = wgMakeStorage(ctx.get(), nullptr, trunkElts, false);

  wgpu::Buffer preScaleBuf = wgMakeStorage(ctx.get(), preScale.data(), trunkC, false);
  wgpu::Buffer preBiasBuf = wgMakeStorage(ctx.get(), preBias.data(), trunkC, false);
  wgpu::Buffer midScaleBuf = wgMakeStorage(ctx.get(), midScale.data(), midC, false);
  wgpu::Buffer midBiasBuf = wgMakeStorage(ctx.get(), midBias.data(), midC, false);
  wgpu::Buffer regWBuf = wgMakeStorage(ctx.get(), desc->regularConv.weights.data(), desc->regularConv.weights.size(), false);
  wgpu::Buffer finWBuf = wgMakeStorage(ctx.get(), desc->finalConv.weights.data(), desc->finalConv.weights.size(), false);

  // Step 1: pre BN + activation + mask  -> s1
  SBMAParamsHost pre{(uint32_t)batchSize, (uint32_t)trunkC, (uint32_t)hw, (uint32_t)desc->preActivation.activation};
  wgpu::Buffer preU = wgMakeUniform(ctx.get(), pre);
  wgDispatch(ctx.get(), "scaleBiasMaskAct",
             {preU, trunkBuf, preScaleBuf, preBiasBuf, maskBuf, s1Buf}, (uint32_t)((trunkElts + 63) / 64));

  // Step 2: regularConv (3x3) -> mid
  ConvParamsHost regC{};
  regC.n = (uint32_t)batchSize; regC.inC = (uint32_t)trunkC; regC.outC = (uint32_t)midC;
  regC.h = (uint32_t)nnYLen; regC.w = (uint32_t)nnXLen;
  regC.fy = (uint32_t)desc->regularConv.convYSize; regC.fx = (uint32_t)desc->regularConv.convXSize;
  regC.dilY = (uint32_t)desc->regularConv.dilationY; regC.dilX = (uint32_t)desc->regularConv.dilationX;
  wgpu::Buffer regU = wgMakeUniform(ctx.get(), regC);
  wgDispatch(ctx.get(), "conv2dNCHW", {regU, s1Buf, regWBuf, midBuf}, (uint32_t)((midElts + 63) / 64));

  // Step 3: mid BN + activation + mask -> s2
  SBMAParamsHost mid{(uint32_t)batchSize, (uint32_t)midC, (uint32_t)hw, (uint32_t)desc->midActivation.activation};
  wgpu::Buffer midU = wgMakeUniform(ctx.get(), mid);
  wgDispatch(ctx.get(), "scaleBiasMaskAct",
             {midU, midBuf, midScaleBuf, midBiasBuf, maskBuf, s2Buf}, (uint32_t)((midElts + 63) / 64));

  // Step 4: finalConv (1x1) -> finalBuf
  ConvParamsHost finC{};
  finC.n = (uint32_t)batchSize; finC.inC = (uint32_t)midC; finC.outC = (uint32_t)trunkC;
  finC.h = (uint32_t)nnYLen; finC.w = (uint32_t)nnXLen;
  finC.fy = (uint32_t)desc->finalConv.convYSize; finC.fx = (uint32_t)desc->finalConv.convXSize;
  finC.dilY = (uint32_t)desc->finalConv.dilationY; finC.dilX = (uint32_t)desc->finalConv.dilationX;
  wgpu::Buffer finU = wgMakeUniform(ctx.get(), finC);
  wgDispatch(ctx.get(), "conv2dNCHW", {finU, s2Buf, finWBuf, finalBuf}, (uint32_t)((trunkElts + 63) / 64));

  // Step 5: residual add  trunk += finalBuf
  AddParamsHost add{(uint32_t)trunkElts, 0, 0, 0};
  wgpu::Buffer addU = wgMakeUniform(ctx.get(), add);
  wgDispatch(ctx.get(), "addInPlace", {addU, finalBuf, trunkBuf}, (uint32_t)((trunkElts + 63) / 64));

  wgReadFloats(ctx.get(), trunkBuf, trunkElts, outputBuffer);
  return true;
#else
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen; (void)useFP16; (void)useNHWC;
  (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
#endif
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC,
  const std::vector<float>& inputBuffer, const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
#ifdef KATAGO_HAVE_WEBGPU
  if(useFP16 || useNHWC)
    return false;

  // Global-pooling residual block, mirroring the reference:
  //   ts   = mask * preAct(preBN(trunk))               [scaleBiasMaskAct]
  //   reg  = regularConv(ts)                            [conv2dNCHW]
  //   gp   = gpoolConv(ts)                              [conv2dNCHW]
  //   gp2  = mask * gpoolAct(gpoolBN(gp))               [scaleBiasMaskAct]
  //   cat  = [mean, scaledMean, max] per channel        [globalPoolMeanMax]
  //   bias = gpoolToBiasMul(cat)                         [matMulBiasAct, no bias/act]
  //   reg += bias  (broadcast over space)               [addChannelBias]
  //   out  = trunk + finalConv(mask*midAct(midBN(reg)))  [scaleBiasMaskAct+conv+addInPlace]
  const int trunkC = desc->preBN.numChannels;            // = finalConv.outChannels
  const int regularC = desc->regularConv.outChannels;     // = midBN.numChannels = gpoolToBiasMul.outChannels
  const int gpoolC = desc->gpoolConv.outChannels;         // gpoolBN.numChannels
  const size_t hw = (size_t)nnXLen * nnYLen;
  const size_t trunkElts = (size_t)batchSize * trunkC * hw;
  const size_t regElts = (size_t)batchSize * regularC * hw;
  const size_t gpElts = (size_t)batchSize * gpoolC * hw;
  const size_t concatElts = (size_t)batchSize * 3 * gpoolC;
  const size_t biasElts = (size_t)batchSize * regularC;
  if(inputBuffer.size() != trunkElts)
    throw StringError("WebGPU testEvaluateGlobalPoolingResidualBlock: unexpected input buffer size");

  std::vector<float> preScale, preBias, gpScale, gpBias, midScale, midBias;
  foldBatchNorm(&desc->preBN, preScale, preBias);
  foldBatchNorm(&desc->gpoolBN, gpScale, gpBias);
  foldBatchNorm(&desc->midBN, midScale, midBias);

  std::unique_ptr<ComputeContext> ctx(new ComputeContext(nnXLen, nnYLen, enabled_t::False));
  initContextGpu(ctx.get(), NULL);

  wgpu::Buffer trunkBuf = wgMakeStorage(ctx.get(), inputBuffer.data(), trunkElts, true);
  wgpu::Buffer maskBuf = wgMakeStorage(ctx.get(), maskBuffer.data(), maskBuffer.size(), false);
  wgpu::Buffer tsBuf = wgMakeStorage(ctx.get(), nullptr, trunkElts, false);
  wgpu::Buffer regBuf = wgMakeStorage(ctx.get(), nullptr, regElts, false);
  wgpu::Buffer gpBuf = wgMakeStorage(ctx.get(), nullptr, gpElts, false);
  wgpu::Buffer gp2Buf = wgMakeStorage(ctx.get(), nullptr, gpElts, false);
  wgpu::Buffer concatBuf = wgMakeStorage(ctx.get(), nullptr, concatElts, false);
  wgpu::Buffer biasBuf = wgMakeStorage(ctx.get(), nullptr, biasElts, false);
  wgpu::Buffer s2Buf = wgMakeStorage(ctx.get(), nullptr, regElts, false);
  wgpu::Buffer finalBuf = wgMakeStorage(ctx.get(), nullptr, trunkElts, false);
  wgpu::Buffer dummyBias = wgMakeStorage(ctx.get(), nullptr, 1, false);  // unused (hasBias=0)

  wgpu::Buffer preScaleBuf = wgMakeStorage(ctx.get(), preScale.data(), trunkC, false);
  wgpu::Buffer preBiasBuf = wgMakeStorage(ctx.get(), preBias.data(), trunkC, false);
  wgpu::Buffer gpScaleBuf = wgMakeStorage(ctx.get(), gpScale.data(), gpoolC, false);
  wgpu::Buffer gpBiasBuf = wgMakeStorage(ctx.get(), gpBias.data(), gpoolC, false);
  wgpu::Buffer midScaleBuf = wgMakeStorage(ctx.get(), midScale.data(), regularC, false);
  wgpu::Buffer midBiasBuf = wgMakeStorage(ctx.get(), midBias.data(), regularC, false);
  wgpu::Buffer regWBuf = wgMakeStorage(ctx.get(), desc->regularConv.weights.data(), desc->regularConv.weights.size(), false);
  wgpu::Buffer gpWBuf = wgMakeStorage(ctx.get(), desc->gpoolConv.weights.data(), desc->gpoolConv.weights.size(), false);
  wgpu::Buffer mmWBuf = wgMakeStorage(ctx.get(), desc->gpoolToBiasMul.weights.data(), desc->gpoolToBiasMul.weights.size(), false);
  wgpu::Buffer finWBuf = wgMakeStorage(ctx.get(), desc->finalConv.weights.data(), desc->finalConv.weights.size(), false);

  auto convParams = [&](const ConvLayerDesc& cd, int inC, int outC) {
    ConvParamsHost p{};
    p.n = (uint32_t)batchSize; p.inC = (uint32_t)inC; p.outC = (uint32_t)outC;
    p.h = (uint32_t)nnYLen; p.w = (uint32_t)nnXLen;
    p.fy = (uint32_t)cd.convYSize; p.fx = (uint32_t)cd.convXSize;
    p.dilY = (uint32_t)cd.dilationY; p.dilX = (uint32_t)cd.dilationX;
    return p;
  };

  // 1: pre BN+act+mask -> ts
  SBMAParamsHost pre{(uint32_t)batchSize, (uint32_t)trunkC, (uint32_t)hw, (uint32_t)desc->preActivation.activation};
  wgDispatch(ctx.get(), "scaleBiasMaskAct",
             {wgMakeUniform(ctx.get(), pre), trunkBuf, preScaleBuf, preBiasBuf, maskBuf, tsBuf}, (uint32_t)((trunkElts + 63) / 64));

  // 2: regularConv -> reg
  wgDispatch(ctx.get(), "conv2dNCHW",
             {wgMakeUniform(ctx.get(), convParams(desc->regularConv, trunkC, regularC)), tsBuf, regWBuf, regBuf}, (uint32_t)((regElts + 63) / 64));

  // 3: gpoolConv -> gp
  wgDispatch(ctx.get(), "conv2dNCHW",
             {wgMakeUniform(ctx.get(), convParams(desc->gpoolConv, trunkC, gpoolC)), tsBuf, gpWBuf, gpBuf}, (uint32_t)((gpElts + 63) / 64));

  // 4: gpool BN+act+mask -> gp2
  SBMAParamsHost gpa{(uint32_t)batchSize, (uint32_t)gpoolC, (uint32_t)hw, (uint32_t)desc->gpoolActivation.activation};
  wgDispatch(ctx.get(), "scaleBiasMaskAct",
             {wgMakeUniform(ctx.get(), gpa), gpBuf, gpScaleBuf, gpBiasBuf, maskBuf, gp2Buf}, (uint32_t)((gpElts + 63) / 64));

  // 5: global pooling -> concat [batch, 3*gpoolC] (one workgroup per (n,c))
  GPParamsHost gpp{(uint32_t)batchSize, (uint32_t)gpoolC, (uint32_t)hw, 0};
  wgDispatch(ctx.get(), "globalPoolMeanMax",
             {wgMakeUniform(ctx.get(), gpp), gp2Buf, maskBuf, concatBuf}, (uint32_t)(batchSize * gpoolC));


  // 6: gpoolToBias matmul -> bias [batch, regularC]
  MMParamsHost mm{(uint32_t)batchSize, (uint32_t)(3 * gpoolC), (uint32_t)regularC, (uint32_t)ACTIVATION_IDENTITY, 0u, 0u, 0u, 0u};
  wgDispatch(ctx.get(), "matMulBiasAct",
             {wgMakeUniform(ctx.get(), mm), concatBuf, mmWBuf, dummyBias, biasBuf},
             (uint32_t)((batchSize + 7) / 8), (uint32_t)((regularC + 7) / 8));


  // 7: reg += bias (broadcast)
  AddBiasParamsHost ab{(uint32_t)batchSize, (uint32_t)regularC, (uint32_t)hw, 0};
  wgDispatch(ctx.get(), "addChannelBias",
             {wgMakeUniform(ctx.get(), ab), biasBuf, regBuf}, (uint32_t)((regElts + 63) / 64));

  // 8: mid BN+act+mask -> s2
  SBMAParamsHost mid{(uint32_t)batchSize, (uint32_t)regularC, (uint32_t)hw, (uint32_t)desc->midActivation.activation};
  wgDispatch(ctx.get(), "scaleBiasMaskAct",
             {wgMakeUniform(ctx.get(), mid), regBuf, midScaleBuf, midBiasBuf, maskBuf, s2Buf}, (uint32_t)((regElts + 63) / 64));

  // 9: finalConv -> finalBuf
  wgDispatch(ctx.get(), "conv2dNCHW",
             {wgMakeUniform(ctx.get(), convParams(desc->finalConv, regularC, trunkC)), s2Buf, finWBuf, finalBuf}, (uint32_t)((trunkElts + 63) / 64));

  // 10: residual add  trunk += finalBuf
  AddParamsHost add{(uint32_t)trunkElts, 0, 0, 0};
  wgDispatch(ctx.get(), "addInPlace",
             {wgMakeUniform(ctx.get(), add), finalBuf, trunkBuf}, (uint32_t)((trunkElts + 63) / 64));

  wgReadFloats(ctx.get(), trunkBuf, trunkElts, outputBuffer);

  return true;
#else
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen; (void)useFP16; (void)useNHWC;
  (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
#endif
}
