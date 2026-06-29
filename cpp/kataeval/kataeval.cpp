// kataeval implementation — see kataeval.h.
//
// Loads a .bin.gz model, builds the V7 input features via the validated
// NNInputs::fillRowV7 path, runs the WebGPU forward pass (webgpubackend.cpp,
// validated byte-identical to Eigen natively), and returns raw logits.

#include "kataeval.h"

#include <cstring>
#include <string>
#include <vector>

#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define KATAEVAL_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define KATAEVAL_EXPORT
#endif

// Set by the dispatcher (dispatch.cpp) to whichever backend was selected.
extern "C" int kgeBackendIsGpu();

namespace {
LoadedModel* gModel = nullptr;
ComputeContext* gCtx = nullptr;
ComputeHandle* gHandle = nullptr;
InputBuffers* gInputs = nullptr;
int gXLen = 19, gYLen = 19, gModelVersion = 0;
bool gUseNHWC = false;  // WebGPU wants NCHW (false); the Eigen CPU backend wants NHWC (true)
std::string gErr = "";
}  // namespace

extern "C" {

KATAEVAL_EXPORT const char* kgeError(void) { return gErr.c_str(); }
KATAEVAL_EXPORT int kgeBoardSize(void) { return gXLen; }
KATAEVAL_EXPORT int kgeModelVersion(void) { return gModelVersion; }

KATAEVAL_EXPORT int kgeLoad(const char* modelPath, int boardSize) {
  try {
    Board::initHash();
    ScoreValue::initTables();
    NeuralNet::globalInitialize();
    gModel = NeuralNet::loadModelFile(modelPath, "");
    const ModelDesc& desc = NeuralNet::getModelDesc(gModel);
    gModelVersion = desc.modelVersion;
    gXLen = gYLen = boardSize;

    ConfigParser cfg;
    std::vector<int> gpuIdxs = { -1 };
    gCtx = NeuralNet::createComputeContext(
      gpuIdxs, /*logger*/nullptr, gXLen, gYLen, /*homeDataDir*/"",
      enabled_t::False /*fp32*/, gModel, cfg);
    // createComputeContext picks the backend (WebGPU, else Eigen CPU); match the
    // input layout to it (Eigen only supports NHWC).
    gUseNHWC = !kgeBackendIsGpu();
    gHandle = NeuralNet::createComputeHandle(
      gCtx, gModel, /*logger*/nullptr, /*maxBatch*/1,
      /*requireExactNNLen*/true, /*inputsUseNHWC*/gUseNHWC, /*gpuIdx*/-1, /*serverThread*/0);
    gInputs = NeuralNet::createInputBuffers(gModel, 1, gXLen, gYLen);
    return 1;
  } catch(const std::exception& e) {
    gErr = e.what();
    return 0;
  }
}

KATAEVAL_EXPORT int kgeEval(const int* stones, int pla, double komi,
                            float* policyOut, float* valueOut, float* ownerOut) {
  if(gHandle == nullptr) { gErr = "not loaded"; return 0; }
  try {
    Board board(gXLen, gYLen);
    for(int y = 0; y < gYLen; y++)
      for(int x = 0; x < gXLen; x++) {
        int c = stones[y * gXLen + x];
        if(c == 1 || c == 2) board.setStone(Location::getLoc(x, y, gXLen), (Color)c);
      }

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = (float)komi;
    Player nextPla = (Player)pla;
    BoardHistory hist(board, nextPla, rules, 0);

    NNResultBuf buf;
    const int spatialLen = NNModelVersion::getNumSpatialFeatures(gModelVersion) * gXLen * gYLen;
    const int globalLen = NNModelVersion::getNumGlobalFeatures(gModelVersion);
    buf.rowSpatialBuf.resize(spatialLen);
    buf.rowGlobalBuf.resize(globalLen);
    buf.hasRowMeta = false;
    buf.symmetry = 0;
    MiscNNInputParams nnInputParams;

    // modelVersion 8..17 -> inputs version 7
    NNInputs::fillRowV7(board, hist, nextPla, nnInputParams, gXLen, gYLen,
                        gUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());

    NNOutput output;
    output.nnXLen = gXLen;
    output.nnYLen = gYLen;
    output.whiteOwnerMap = ownerOut;

    NNResultBuf* bufs[1] = { &buf };
    std::vector<NNOutput*> outputs = { &output };
    NeuralNet::getOutput(gHandle, gInputs, 1, bufs, outputs);

    const int hw = gXLen * gYLen;
    std::memcpy(policyOut, output.policyProbs, (hw + 1) * sizeof(float));  // raw logits
    valueOut[0] = output.whiteWinProb;
    valueOut[1] = output.whiteLossProb;
    valueOut[2] = output.whiteNoResultProb;
    valueOut[3] = output.whiteScoreMean;
    valueOut[4] = output.whiteLead;
    return 1;
  } catch(const std::exception& e) {
    gErr = e.what();
    return 0;
  }
}

}  // extern "C"
