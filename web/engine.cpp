// Minimal in-browser WebGPU engine for the goban demo.
//
// Exposes a tiny JS-callable API that runs our REAL conv kernel
// (cpp/neuralnet/webgpukernels.cpp, zero KataGo deps) on the GPU via Dawn's
// emdawnwebgpu port. It computes a stone "influence map": the board (B=+1,
// W=-1, empty=0) diffused by N passes of a normalized 3x3 smoothing conv on the
// GPU. This drives the exact upload -> dispatch -> readback pipeline the full
// net evaluation will use, so the demo proves the engine end-to-end in-browser.
//
// Build: scripts/build-web.sh -> web/demo/engine.{js,wasm}

#include <webgpu/webgpu_cpp.h>
#include <emscripten.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../cpp/neuralnet/webgpukernels.h"

static wgpu::Instance gInstance;
static wgpu::Device gDevice;
static wgpu::Queue gQueue;
static wgpu::ShaderModule gModule;
static wgpu::ComputePipeline gConvPipe;
static std::string gDeviceName = "(uninitialized)";
static bool gReady = false;

static inline void pump(volatile bool& done) { while(!done) emscripten_sleep(0); }

struct ConvParams { uint32_t n, inC, outC, h, w, fy, fx, dilY, dilX, p0, p1, p2; };

static wgpu::Buffer makeBuf(uint64_t bytes, wgpu::BufferUsage usage, const void* data) {
  wgpu::BufferDescriptor d{}; d.size = bytes; d.usage = usage;
  wgpu::Buffer b = gDevice.CreateBuffer(&d);
  if(data) gQueue.WriteBuffer(b, 0, data, bytes);
  return b;
}

extern "C" {

// Acquire the browser's WebGPU device and compile the kernel module.
// Async (blocks via Asyncify); call from JS with ccall(..., {async:true}).
EMSCRIPTEN_KEEPALIVE int engineInit() {
  if(gReady) return 1;
  gInstance = wgpu::CreateInstance(nullptr);

  wgpu::Adapter adapter;
  { volatile bool done = false;
    gInstance.RequestAdapter(nullptr, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) {
        if(s == wgpu::RequestAdapterStatus::Success) adapter = a; done = true; });
    pump(done);
  }
  if(adapter == nullptr) { gDeviceName = "(no WebGPU adapter)"; return 0; }
  wgpu::AdapterInfo info{}; adapter.GetInfo(&info);
  gDeviceName = std::string(info.device.data, info.device.length);

  { volatile bool done = false;
    adapter.RequestDevice(nullptr, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) {
        if(s == wgpu::RequestDeviceStatus::Success) gDevice = d; done = true; });
    pump(done);
  }
  if(gDevice == nullptr) { gDeviceName = "(device request failed)"; return 0; }
  gQueue = gDevice.GetQueue();

  std::string src = std::string("alias STO = f32;\n") + KataGoWebGPU::WGSL_KERNELS;
  wgpu::ShaderSourceWGSL wgsl{}; wgsl.code = src.c_str();
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &wgsl;
  gModule = gDevice.CreateShaderModule(&smd);
  if(gModule == nullptr) { gDeviceName = "(shader compile failed)"; return 0; }

  wgpu::ComputePipelineDescriptor pd{}; pd.compute.module = gModule; pd.compute.entryPoint = "conv2dNCHW";
  gConvPipe = gDevice.CreateComputePipeline(&pd);

  gReady = true;
  return 1;
}

EMSCRIPTEN_KEEPALIVE const char* engineDeviceName() { return gDeviceName.c_str(); }
EMSCRIPTEN_KEEPALIVE int engineReady() { return gReady ? 1 : 0; }

// board: n*n int32 (-1 white, 0 empty, +1 black). out: n*n float32 influence.
// Runs `passes` GPU smoothing-conv iterations (ping-pong buffers). Async.
EMSCRIPTEN_KEEPALIVE void engineInfluence(const int32_t* board, int n, float* out, int passes) {
  if(!gReady) return;
  const int hw = n * n;
  std::vector<float> init(hw);
  for(int i = 0; i < hw; i++) init[i] = (float)board[i];

  using U = wgpu::BufferUsage;
  wgpu::Buffer a = makeBuf((uint64_t)hw*4, U::Storage|U::CopyDst|U::CopySrc, init.data());
  wgpu::Buffer b = makeBuf((uint64_t)hw*4, U::Storage|U::CopyDst|U::CopySrc, nullptr);

  float k[9] = {1,2,1, 2,4,2, 1,2,1}; for(float& x : k) x /= 16.0f;   // normalized Gaussian-ish
  wgpu::Buffer wBuf = makeBuf(9*4, U::Storage|U::CopyDst, k);
  ConvParams p{1u, 1u, 1u, (uint32_t)n, (uint32_t)n, 3u, 3u, 1u, 1u, 0u, 0u, 0u};
  wgpu::Buffer uBuf = makeBuf(sizeof(p), U::Uniform|U::CopyDst, &p);

  wgpu::Buffer cur = a, nxt = b;
  for(int pass = 0; pass < passes; pass++) {
    wgpu::BindGroupEntry e[4];
    wgpu::Buffer binds[4] = {uBuf, cur, wBuf, nxt};
    for(int i = 0; i < 4; i++) { e[i] = {}; e[i].binding = i; e[i].buffer = binds[i]; e[i].size = binds[i].GetSize(); }
    wgpu::BindGroupDescriptor bgd{}; bgd.layout = gConvPipe.GetBindGroupLayout(0); bgd.entryCount = 4; bgd.entries = e;
    wgpu::BindGroup bg = gDevice.CreateBindGroup(&bgd);
    wgpu::CommandEncoder enc = gDevice.CreateCommandEncoder();
    wgpu::ComputePassEncoder cp = enc.BeginComputePass();
    cp.SetPipeline(gConvPipe); cp.SetBindGroup(0, bg); cp.DispatchWorkgroups((hw + 63) / 64); cp.End();
    wgpu::CommandBuffer cb = enc.Finish(); gQueue.Submit(1, &cb);
    wgpu::Buffer tmp = cur; cur = nxt; nxt = tmp;
  }

  wgpu::Buffer rb = makeBuf((uint64_t)hw*4, U::MapRead|U::CopyDst, nullptr);
  wgpu::CommandEncoder enc = gDevice.CreateCommandEncoder();
  enc.CopyBufferToBuffer(cur, 0, rb, 0, (uint64_t)hw*4);
  wgpu::CommandBuffer cb = enc.Finish(); gQueue.Submit(1, &cb);
  { volatile bool done = false; bool ok = false;
    rb.MapAsync(wgpu::MapMode::Read, 0, (uint64_t)hw*4, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::MapAsyncStatus s, wgpu::StringView) { ok = (s == wgpu::MapAsyncStatus::Success); done = true; });
    pump(done);
    if(!ok) return;
  }
  memcpy(out, rb.GetConstMappedRange(0, (uint64_t)hw*4), (size_t)hw*4);
  rb.Unmap();
}

}  // extern "C"
