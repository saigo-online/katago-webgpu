// Runtime gate: WebGPU device + compute + readback on a dedicated pthread (the
// NN-server-thread model). main() is proxied to a pthread; it spawns a server
// pthread that owns the WebGPU device. Reports back to the page via window.__tt.
#include <webgpu/webgpu_cpp.h>
#include <emscripten.h>
#include <pthread.h>
#include <utility>
static void report(int status, double val) { EM_ASM({ if (window.__tt) window.__tt($0, $1); }, status, val); }
static void pump(volatile bool& done) { while(!done) emscripten_sleep(0); }
static void runWebGPU() {
  wgpu::Instance instance = wgpu::CreateInstance(nullptr);
  wgpu::Adapter adapter;
  { volatile bool d=false; instance.RequestAdapter(nullptr, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView){ if(s==wgpu::RequestAdapterStatus::Success) adapter=std::move(a); d=true; }); pump(d); }
  if(!adapter){ report(2,-1); return; }
  wgpu::Device device;
  { volatile bool d=false; adapter.RequestDevice(nullptr, wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::RequestDeviceStatus s, wgpu::Device dv, wgpu::StringView){ if(s==wgpu::RequestDeviceStatus::Success) device=std::move(dv); d=true; }); pump(d); }
  if(!device){ report(3,-1); return; }
  wgpu::Queue queue = device.GetQueue();
  const char* wgsl = "@group(0)@binding(0) var<storage,read_write> o:array<f32>;\n@compute @workgroup_size(1) fn main(){o[0]=42.0;}\n";
  wgpu::ShaderSourceWGSL src{}; src.code = wgsl;
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain=&src;
  wgpu::ShaderModule mod = device.CreateShaderModule(&smd);
  wgpu::BufferDescriptor bd{}; bd.size=16; bd.usage=wgpu::BufferUsage::Storage|wgpu::BufferUsage::CopySrc;
  wgpu::Buffer buf=device.CreateBuffer(&bd);
  wgpu::ComputePipelineDescriptor pd{}; pd.compute.module=mod; pd.compute.entryPoint="main";
  wgpu::ComputePipeline pipe=device.CreateComputePipeline(&pd);
  wgpu::BindGroupEntry e{}; e.binding=0; e.buffer=buf; e.size=16;
  wgpu::BindGroupDescriptor bgd{}; bgd.layout=pipe.GetBindGroupLayout(0); bgd.entryCount=1; bgd.entries=&e;
  wgpu::BindGroup bg=device.CreateBindGroup(&bgd);
  wgpu::CommandEncoder enc=device.CreateCommandEncoder();
  wgpu::ComputePassEncoder p=enc.BeginComputePass(); p.SetPipeline(pipe); p.SetBindGroup(0,bg); p.DispatchWorkgroups(1); p.End();
  wgpu::BufferDescriptor rbd{}; rbd.size=16; rbd.usage=wgpu::BufferUsage::MapRead|wgpu::BufferUsage::CopyDst;
  wgpu::Buffer rb=device.CreateBuffer(&rbd);
  enc.CopyBufferToBuffer(buf,0,rb,0,16);
  wgpu::CommandBuffer cmd=enc.Finish(); queue.Submit(1,&cmd);
  double val=-1; { volatile bool d=false; bool ok=false;
    rb.MapAsync(wgpu::MapMode::Read,0,16,wgpu::CallbackMode::AllowSpontaneous,
      [&](wgpu::MapAsyncStatus s, wgpu::StringView){ ok=(s==wgpu::MapAsyncStatus::Success); d=true; }); pump(d);
    if(ok){ const float* q=(const float*)rb.GetConstMappedRange(0,16); val=q[0]; rb.Unmap(); } }
  report(val==42.0 ? 0 : 1, val);
}
void* serverThread(void*) { runWebGPU(); return nullptr; }
int main() { pthread_t t; pthread_create(&t,nullptr,serverThread,nullptr); pthread_join(t,nullptr); return 0; }
