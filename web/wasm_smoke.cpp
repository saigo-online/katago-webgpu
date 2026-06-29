// WASM smoke test for the KataGo WebGPU backend's kernels in the browser.
//
// Proves the de-risking thesis: our *actual* WGSL kernel library
// (cpp/neuralnet/webgpukernels.cpp — zero KataGo deps) compiles to WebAssembly
// and runs on the browser's WebGPU via Dawn's emdawnwebgpu port, using the same
// standard WebGPU C++ API the native backend uses. It runs scaleBiasMaskAct on a
// tiny known input and checks the result against a CPU reference.
//
// Build: engines/katago-web/scripts/build-web.sh  (emcc --use-port=emdawnwebgpu)
// Run:   serve web/dist/ over http and open wasm_smoke.html in a WebGPU browser;
//        the PASS/FAIL line prints to the page (and the JS console).

#include <webgpu/webgpu_cpp.h>
#include <emscripten.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../cpp/neuralnet/webgpukernels.h"

static wgpu::Instance gInstance;

// Browser event loop is single-threaded: emscripten_sleep(0) (Asyncify/JSPI)
// yields so the spontaneous WebGPU callbacks can fire, then resumes. `done` is
// volatile so the callback's write isn't optimized away (same lesson as the
// native wgPump). This mirrors the backend's wgPump for the web target.
static inline void pump(volatile bool& done) { while(!done) emscripten_sleep(0); }

struct SBMAParams { uint32_t n, c, hw, act; };

int main() {
  gInstance = wgpu::CreateInstance(nullptr);

  // --- adapter + device (async, browser-driven) ---
  wgpu::Adapter adapter;
  { volatile bool done = false;
    gInstance.RequestAdapter(nullptr, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) {
        if(s == wgpu::RequestAdapterStatus::Success) adapter = a; done = true; });
    pump(done);
  }
  if(adapter == nullptr) { printf("WASM smoke: NO WEBGPU ADAPTER (does this browser/build expose WebGPU?)\n"); return 1; }

  wgpu::Device device;
  { volatile bool done = false;
    adapter.RequestDevice(nullptr, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) {
        if(s == wgpu::RequestDeviceStatus::Success) device = d; done = true; });
    pump(done);
  }
  if(device == nullptr) { printf("WASM smoke: RequestDevice FAILED\n"); return 1; }
  wgpu::Queue queue = device.GetQueue();

  // --- compile the real kernel module (fp32 storage variant) ---
  std::string src = std::string("alias STO = f32;\n") + KataGoWebGPU::WGSL_KERNELS;
  wgpu::ShaderSourceWGSL wgsl{}; wgsl.code = src.c_str();
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &wgsl;
  wgpu::ShaderModule mod = device.CreateShaderModule(&smd);
  if(mod == nullptr) { printf("WASM smoke: kernel module FAILED to compile\n"); return 1; }

  // --- tiny problem: scaleBiasMaskAct, n=1 c=1 hw=4, relu(in*scale+bias)*mask ---
  const uint32_t N = 1, C = 1, HW = 4;
  std::vector<float> in   = {1, 2, 3, 4};
  std::vector<float> scl  = {2};
  std::vector<float> bia  = {1};
  std::vector<float> mask = {1, 1, 0, 1};       // 3rd cell masked off
  std::vector<float> expected(HW);
  for(uint32_t i = 0; i < HW; i++) { float v = in[i]*scl[0]+bia[0]; v = v > 0 ? v : 0; expected[i] = v * mask[i]; }
  // => relu([3,5,7,9]) * [1,1,0,1] = [3,5,0,9]

  auto storage = [&](const std::vector<float>& d, bool copySrc) {
    wgpu::BufferDescriptor bd{}; bd.size = d.size()*sizeof(float);
    bd.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | (copySrc ? wgpu::BufferUsage::CopySrc : wgpu::BufferUsage::None);
    wgpu::Buffer b = device.CreateBuffer(&bd);
    if(!d.empty()) queue.WriteBuffer(b, 0, d.data(), bd.size);
    return b;
  };
  wgpu::Buffer inB = storage(in,false), sclB = storage(scl,false), biaB = storage(bia,false), maskB = storage(mask,false);
  wgpu::Buffer outB = storage(std::vector<float>(HW,0.0f), true);

  SBMAParams p{N, C, HW, 1u /*relu*/};
  wgpu::BufferDescriptor ud{}; ud.size = sizeof(p); ud.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer uB = device.CreateBuffer(&ud); queue.WriteBuffer(uB, 0, &p, sizeof(p));

  wgpu::ComputePipelineDescriptor pd{}; pd.compute.module = mod; pd.compute.entryPoint = "scaleBiasMaskAct";
  wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pd);

  wgpu::Buffer binds[6] = {uB, inB, sclB, biaB, maskB, outB};
  wgpu::BindGroupEntry entries[6];
  for(int i = 0; i < 6; i++) { entries[i] = {}; entries[i].binding = i; entries[i].buffer = binds[i]; entries[i].size = binds[i].GetSize(); }
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pipeline.GetBindGroupLayout(0); bgd.entryCount = 6; bgd.entries = entries;
  wgpu::BindGroup bg = device.CreateBindGroup(&bgd);

  wgpu::CommandEncoder enc = device.CreateCommandEncoder();
  wgpu::ComputePassEncoder pass = enc.BeginComputePass();
  pass.SetPipeline(pipeline); pass.SetBindGroup(0, bg); pass.DispatchWorkgroups(1); pass.End();

  wgpu::BufferDescriptor rd{}; rd.size = HW*sizeof(float); rd.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer readback = device.CreateBuffer(&rd);
  enc.CopyBufferToBuffer(outB, 0, readback, 0, rd.size);
  wgpu::CommandBuffer cmds = enc.Finish();
  queue.Submit(1, &cmds);

  { volatile bool done = false; bool ok = false;
    readback.MapAsync(wgpu::MapMode::Read, 0, rd.size, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::MapAsyncStatus s, wgpu::StringView) { ok = (s == wgpu::MapAsyncStatus::Success); done = true; });
    pump(done);
    if(!ok) { printf("WASM smoke: readback map FAILED\n"); return 1; }
  }
  const float* got = (const float*)readback.GetConstMappedRange(0, rd.size);

  bool pass_ok = true;
  for(uint32_t i = 0; i < HW; i++) if(std::abs(got[i] - expected[i]) > 1e-4f) pass_ok = false;
  printf("WASM smoke: got [%.1f %.1f %.1f %.1f] expected [%.1f %.1f %.1f %.1f] => %s\n",
    got[0],got[1],got[2],got[3], expected[0],expected[1],expected[2],expected[3], pass_ok ? "PASS" : "FAIL");
  readback.Unmap();
  return pass_ok ? 0 : 1;
}
