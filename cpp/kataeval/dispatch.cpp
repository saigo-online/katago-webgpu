// Runtime backend dispatcher: implements the real NeuralNet:: interface by
// routing to the WebGPU (gpuimpl) or Eigen-CPU (cpuimpl) backend. createCompute-
// Context tries WebGPU first and falls back to CPU if no GPU/adapter is
// available; the choice is global for the rest of the session. This is what lets
// one WASM binary "just fall back" when WebGPU is missing.
//
// Handle types (ComputeContext/Handle/LoadedModel/InputBuffers) are opaque here;
// each backend's concrete type is passed as void*. Both backends' LoadedModel is
// layout-identical ({ModelDesc}), so the model (loaded before the backend is
// chosen) is usable by whichever backend wins.

#include "../neuralnet/nninterface.h"
#include "backends.h"

using namespace std;

namespace { bool gGpu = true; bool gForceCpu = false; }

extern "C" int kgeBackendIsGpu() { return gGpu ? 1 : 0; }
// Force the CPU (Eigen) backend even if WebGPU is available — for testing the
// fallback path and comparing speed. Call before kgeLoad.
extern "C" void kgeSetForceCpu(int v) { gForceCpu = (v != 0); }

void NeuralNet::globalInitialize() { gpuimpl::globalInitialize(); cpuimpl::globalInitialize(); }
void NeuralNet::globalCleanup() {}
void NeuralNet::printDevices() {}

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& sha) {
  return (LoadedModel*)gpuimpl::loadModelFile(file, sha);  // {ModelDesc}, backend-agnostic
}
void NeuralNet::freeLoadedModel(LoadedModel* m) { gpuimpl::freeLoadedModel((void*)m); }
const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* m) { return gpuimpl::getModelDesc((const void*)m); }

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs, Logger* logger, int nnXLen, int nnYLen,
  const string& homeDir, enabled_t fp16, const LoadedModel* model, ConfigParser& cfg) {
  if(!gForceCpu) {
    try {
      void* c = gpuimpl::createComputeContext(gpuIdxs, logger, nnXLen, nnYLen, homeDir, fp16, (const void*)model, cfg);
      gGpu = true;
      return (ComputeContext*)c;
    } catch(const std::exception&) { /* no WebGPU adapter — fall through to CPU */ }
  }
  gGpu = false;
  return (ComputeContext*)cpuimpl::createComputeContext(gpuIdxs, logger, nnXLen, nnYLen, homeDir, fp16, (const void*)model, cfg);
}
void NeuralNet::freeComputeContext(ComputeContext* c) { gGpu ? gpuimpl::freeComputeContext((void*)c) : cpuimpl::freeComputeContext((void*)c); }

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* ctx, const LoadedModel* m, Logger* l, int maxBatch,
  bool requireExactNNLen, bool inputsUseNHWC, int gpuIdx, int serverIdx) {
  return (ComputeHandle*)(gGpu
    ? gpuimpl::createComputeHandle((void*)ctx, (const void*)m, l, maxBatch, requireExactNNLen, inputsUseNHWC, gpuIdx, serverIdx)
    : cpuimpl::createComputeHandle((void*)ctx, (const void*)m, l, maxBatch, requireExactNNLen, inputsUseNHWC, gpuIdx, serverIdx));
}
void NeuralNet::freeComputeHandle(ComputeHandle* h) { gGpu ? gpuimpl::freeComputeHandle((void*)h) : cpuimpl::freeComputeHandle((void*)h); }
bool NeuralNet::isUsingFP16(const ComputeHandle* h) { return gGpu ? gpuimpl::isUsingFP16((const void*)h) : cpuimpl::isUsingFP16((const void*)h); }
bool NeuralNet::setIsWarmup(const ComputeHandle*, bool) { return false; }

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* m, int maxBatch, int nnXLen, int nnYLen) {
  return (InputBuffers*)(gGpu
    ? gpuimpl::createInputBuffers((const void*)m, maxBatch, nnXLen, nnYLen)
    : cpuimpl::createInputBuffers((const void*)m, maxBatch, nnXLen, nnYLen));
}
void NeuralNet::freeInputBuffers(InputBuffers* b) { gGpu ? gpuimpl::freeInputBuffers((void*)b) : cpuimpl::freeInputBuffers((void*)b); }

void NeuralNet::getOutput(ComputeHandle* h, InputBuffers* i, int numFilled, NNResultBuf** bufs, vector<NNOutput*>& outs) {
  if(gGpu) gpuimpl::getOutput((void*)h, (void*)i, numFilled, bufs, outs);
  else     cpuimpl::getOutput((void*)h, (void*)i, numFilled, bufs, outs);
}
