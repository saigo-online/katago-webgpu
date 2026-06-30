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

// Run the net on a prepared board + history and write outputs (raw logits).
static int evalBoardHist(const Board& board, const BoardHistory& hist, Player nextPla,
                         float* policyOut, float* valueOut, float* ownerOut) {
  NNResultBuf buf;
  const int spatialLen = NNModelVersion::getNumSpatialFeatures(gModelVersion) * gXLen * gYLen;
  const int globalLen = NNModelVersion::getNumGlobalFeatures(gModelVersion);
  buf.rowSpatialBuf.resize(spatialLen);
  buf.rowGlobalBuf.resize(globalLen);
  buf.hasRowMeta = false;
  buf.symmetry = 0;
  MiscNNInputParams nnInputParams;
  // modelVersion 8..17 -> inputs version 7 (encodes the last few moves + ko from hist)
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
  // valueOut[0..2] are raw win/loss/noResult logits too — the caller softmaxes
  // them (see web/demo/analyze.html); valueOut[3..4] are scoreMean / lead.
  valueOut[0] = output.whiteWinProb;
  valueOut[1] = output.whiteLossProb;
  valueOut[2] = output.whiteNoResultProb;
  valueOut[3] = output.whiteScoreMean;
  valueOut[4] = output.whiteLead;
  return 1;
}

// Stones-only eval (no move history) — single-position analysis.
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
    return evalBoardHist(board, hist, nextPla, policyOut, valueOut, ownerOut);
  } catch(const std::exception& e) {
    gErr = e.what();
    return 0;
  }
}

// Move-sequence eval: replays moves through BoardHistory (captures + ko/superko +
// the recent-move input features), writes the resulting board to boardOut (0 empty,
// 1 black, 2 white, row-major), then evaluates for player toPla. moveLocs[i] is
// y*boardSize+x, or < 0 for a pass; moveCols[i] is 1 (black) / 2 (white).
KATAEVAL_EXPORT int kgeEvalSeq(const int* moveLocs, const int* moveCols, int numMoves,
                               int toPla, double komi, int* boardOut,
                               float* policyOut, float* valueOut, float* ownerOut) {
  if(gHandle == nullptr) { gErr = "not loaded"; return 0; }
  try {
    Board board(gXLen, gYLen);
    Rules rules = Rules::getTrompTaylorish();
    rules.komi = (float)komi;
    Player firstPla = (numMoves > 0) ? (Player)moveCols[0] : P_BLACK;
    BoardHistory hist(board, firstPla, rules, 0);
    for(int i = 0; i < numMoves; i++) {
      Player pla = (Player)moveCols[i];
      Loc loc = (moveLocs[i] < 0) ? Board::PASS_LOC
                                  : Location::getLoc(moveLocs[i] % gXLen, moveLocs[i] / gXLen, gXLen);
      if(hist.isLegal(board, loc, pla))
        hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    }
    if(boardOut != nullptr)
      for(int y = 0; y < gYLen; y++)
        for(int x = 0; x < gXLen; x++)
          boardOut[y * gXLen + x] = (int)board.colors[Location::getLoc(x, y, gXLen)];
    return evalBoardHist(board, hist, (Player)toPla, policyOut, valueOut, ownerOut);
  } catch(const std::exception& e) {
    gErr = e.what();
    return 0;
  }
}

}  // extern "C"
