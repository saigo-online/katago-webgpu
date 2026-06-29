// Eigen (CPU) backend, renamed into the nnCpu namespace + Cpu-suffixed handle
// types so it can coexist with the WebGPU backend in one binary. Exposes
// cpuimpl:: shims. Built with -DUSE_EIGEN_BACKEND (eigenbackend.cpp is guarded).
#define NeuralNet nnCpu
#define ComputeContext ComputeContextCpu
#define ComputeHandle ComputeHandleCpu
#define LoadedModel LoadedModelCpu
#define InputBuffers InputBuffersCpu
#include "../neuralnet/eigenbackend.cpp"
#undef NeuralNet
#undef ComputeContext
#undef ComputeHandle
#undef LoadedModel
#undef InputBuffers

#include "backends.h"

namespace cpuimpl {
  void globalInitialize() { nnCpu::globalInitialize(); }
  void* loadModelFile(const std::string& f, const std::string& s) { return nnCpu::loadModelFile(f, s); }
  void freeLoadedModel(void* m) { nnCpu::freeLoadedModel((LoadedModelCpu*)m); }
  const ModelDesc& getModelDesc(const void* m) { return nnCpu::getModelDesc((const LoadedModelCpu*)m); }
  void* createComputeContext(const std::vector<int>& g, Logger* l, int x, int y, const std::string& h,
                             enabled_t fp, const void* m, ConfigParser& c) {
    return nnCpu::createComputeContext(g, l, x, y, h, fp, (const LoadedModelCpu*)m, c);
  }
  void freeComputeContext(void* c) { nnCpu::freeComputeContext((ComputeContextCpu*)c); }
  void* createComputeHandle(void* ctx, const void* m, Logger* l, int b, bool r, bool n, int gi, int st) {
    return nnCpu::createComputeHandle((ComputeContextCpu*)ctx, (const LoadedModelCpu*)m, l, b, r, n, gi, st);
  }
  void freeComputeHandle(void* h) { nnCpu::freeComputeHandle((ComputeHandleCpu*)h); }
  bool isUsingFP16(const void* h) { return nnCpu::isUsingFP16((const ComputeHandleCpu*)h); }
  void* createInputBuffers(const void* m, int b, int x, int y) { return nnCpu::createInputBuffers((const LoadedModelCpu*)m, b, x, y); }
  void freeInputBuffers(void* b) { nnCpu::freeInputBuffers((InputBuffersCpu*)b); }
  void getOutput(void* h, void* i, int n, NNResultBuf** bufs, std::vector<NNOutput*>& outs) {
    nnCpu::getOutput((ComputeHandleCpu*)h, (InputBuffersCpu*)i, n, bufs, outs);
  }
}
