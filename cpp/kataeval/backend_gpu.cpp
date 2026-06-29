// WebGPU backend, renamed into the nnGpu namespace + Gpu-suffixed handle types so
// it can coexist with the Eigen backend in one binary. Exposes gpuimpl:: shims.
#define NeuralNet nnGpu
#define ComputeContext ComputeContextGpu
#define ComputeHandle ComputeHandleGpu
#define LoadedModel LoadedModelGpu
#define InputBuffers InputBuffersGpu
#include "../neuralnet/webgpubackend.cpp"
#undef NeuralNet
#undef ComputeContext
#undef ComputeHandle
#undef LoadedModel
#undef InputBuffers

#include "backends.h"

namespace gpuimpl {
  void globalInitialize() { nnGpu::globalInitialize(); }
  void* loadModelFile(const std::string& f, const std::string& s) { return nnGpu::loadModelFile(f, s); }
  void freeLoadedModel(void* m) { nnGpu::freeLoadedModel((LoadedModelGpu*)m); }
  const ModelDesc& getModelDesc(const void* m) { return nnGpu::getModelDesc((const LoadedModelGpu*)m); }
  void* createComputeContext(const std::vector<int>& g, Logger* l, int x, int y, const std::string& h,
                             enabled_t fp, const void* m, ConfigParser& c) {
    return nnGpu::createComputeContext(g, l, x, y, h, fp, (const LoadedModelGpu*)m, c);
  }
  void freeComputeContext(void* c) { nnGpu::freeComputeContext((ComputeContextGpu*)c); }
  void* createComputeHandle(void* ctx, const void* m, Logger* l, int b, bool r, bool n, int gi, int st) {
    return nnGpu::createComputeHandle((ComputeContextGpu*)ctx, (const LoadedModelGpu*)m, l, b, r, n, gi, st);
  }
  void freeComputeHandle(void* h) { nnGpu::freeComputeHandle((ComputeHandleGpu*)h); }
  bool isUsingFP16(const void* h) { return nnGpu::isUsingFP16((const ComputeHandleGpu*)h); }
  void* createInputBuffers(const void* m, int b, int x, int y) { return nnGpu::createInputBuffers((const LoadedModelGpu*)m, b, x, y); }
  void freeInputBuffers(void* b) { nnGpu::freeInputBuffers((InputBuffersGpu*)b); }
  void getOutput(void* h, void* i, int n, NNResultBuf** bufs, std::vector<NNOutput*>& outs) {
    nnGpu::getOutput((ComputeHandleGpu*)h, (InputBuffersGpu*)i, n, bufs, outs);
  }
}
