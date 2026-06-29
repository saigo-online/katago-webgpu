// Internal: two backend implementations behind void* shims so they can coexist
// in one binary (dispatch.cpp routes NeuralNet:: to gpuimpl or cpuimpl). The
// opaque handle types (ComputeContext/Handle/LoadedModel/InputBuffers) are
// passed as void*; all other types are the shared KataGo types.
#ifndef KATAEVAL_BACKENDS_H_
#define KATAEVAL_BACKENDS_H_

#include "../neuralnet/nninterface.h"   // Logger, enabled_t, ConfigParser, ModelDesc, NNOutput, ...
#include "../neuralnet/nneval.h"        // NNResultBuf
#include <string>
#include <vector>

#define KATAEVAL_BACKEND_DECL(NS)                                                                   \
  namespace NS {                                                                                    \
    void globalInitialize();                                                                        \
    void* loadModelFile(const std::string& file, const std::string& sha);                          \
    void freeLoadedModel(void* m);                                                                  \
    const ModelDesc& getModelDesc(const void* m);                                                   \
    void* createComputeContext(const std::vector<int>& gpuIdxs, Logger* logger, int nnXLen,         \
                               int nnYLen, const std::string& homeDir, enabled_t fp16,              \
                               const void* loadedModel, ConfigParser& cfg);                         \
    void freeComputeContext(void* c);                                                               \
    void* createComputeHandle(void* ctx, const void* loadedModel, Logger* logger, int maxBatch,     \
                              bool requireExactNNLen, bool inputsUseNHWC, int gpuIdx, int serverIdx);\
    void freeComputeHandle(void* h);                                                                \
    bool isUsingFP16(const void* h);                                                                \
    void* createInputBuffers(const void* loadedModel, int maxBatch, int nnXLen, int nnYLen);        \
    void freeInputBuffers(void* b);                                                                 \
    void getOutput(void* h, void* inputBuffers, int numFilled,                                      \
                   NNResultBuf** inputBufs, std::vector<NNOutput*>& outputs);                       \
  }

KATAEVAL_BACKEND_DECL(gpuimpl)   // WebGPU
KATAEVAL_BACKEND_DECL(cpuimpl)   // Eigen (CPU)

#endif  // KATAEVAL_BACKENDS_H_
