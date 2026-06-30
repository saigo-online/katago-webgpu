// kataeval implementation — see kataeval.h.
//
// Loads a .bin.gz model, builds the V7 input features via the validated
// NNInputs::fillRowV7 path, runs the WebGPU forward pass (webgpubackend.cpp,
// validated byte-identical to Eigen natively), and returns raw logits.

#include "kataeval.h"

#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

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
const int KGE_MAX_BATCH = 16;  // batched eval: many MCTS leaves per forward pass

// ---- Batched MCTS (PUCT) ---------------------------------------------------
// AlphaZero PUCT with KataGo's formula + virtual-loss batched leaf collection.
// Refs: Silver et al. 2017 (AlphaZero); Wu 2019, "Accelerating Self-Play Learning
// in Go" (arXiv:1902.10565) for the c_PUCT=1.1 / exploration form; Danihelka et al.
// 2022, "Policy improvement by planning with Gumbel" (the SOTA low-visit upgrade,
// not yet wired — selection is isolated in selectChildPUCT for that swap).
struct SNode {
  std::vector<int> childMoves;     // candidate child moves (y*N+x, or -1 pass)
  std::vector<float> childPriors;  // normalized policy priors over the children
  std::vector<SNode*> children;    // lazily created on first descent
  double valueSumWhite = 0.0;      // Σ backed-up utilities (white perspective, [-1,1])
  int visits = 0;
  int virtualLoss = 0;             // transient, for batched divergence
  bool expanded = false;
};
void freeTree(SNode* n) { if(!n) return; for(SNode* c : n->children) freeTree(c); delete n; }

// White-perspective utility from win/loss/noResult logits: pWin - pLoss in [-1,1].
double whiteUtil(float wL, float lL, float nL) {
  double m = std::max(wL, std::max(lL, nL));
  double a = std::exp((double)wL - m), b = std::exp((double)lL - m), c = std::exp((double)nL - m);
  return (a - b) / (a + b + c);
}

// Expand a leaf: its legal top-K policy moves become children (priors softmaxed over K).
void expandNode(SNode* node, const float* policyLogits, const Board& board,
                const BoardHistory& hist, Player pla, int topK) {
  const int hw = gXLen * gYLen;
  std::vector<std::pair<float,int>> cand;  // (logit, move)
  cand.reserve(hw + 1);
  for(int y = 0; y < gYLen; y++)
    for(int x = 0; x < gXLen; x++) {
      Loc loc = Location::getLoc(x, y, gXLen);
      if(hist.isLegal(board, loc, pla)) cand.push_back({policyLogits[y*gXLen + x], y*gXLen + x});
    }
  cand.push_back({policyLogits[hw], -1});  // pass is always legal
  int K = std::min((int)cand.size(), topK);
  std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                    [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; });
  double mx = -1e30; for(int i = 0; i < K; i++) mx = std::max(mx, (double)cand[i].first);
  double s = 0; for(int i = 0; i < K; i++) s += std::exp((double)cand[i].first - mx);
  node->childMoves.resize(K); node->childPriors.resize(K); node->children.assign(K, nullptr);
  for(int i = 0; i < K; i++) {
    node->childMoves[i] = cand[i].second;
    node->childPriors[i] = (float)(std::exp((double)cand[i].first - mx) / s);
  }
  node->expanded = true;
}

// PUCT child selection. pla = player to move at `node` (sign-flips white utility).
int selectChildPUCT(const SNode* node, Player pla, double cPUCT) {
  double parentN = 0.0;
  for(SNode* c : node->children) if(c) parentN += c->visits + c->virtualLoss;
  double explore = cPUCT * std::sqrt(parentN + 1e-8);
  double sign = (pla == P_WHITE) ? 1.0 : -1.0;
  double best = -1e30; int bestIdx = -1;
  for(size_t i = 0; i < node->children.size(); i++) {
    SNode* c = node->children[i];
    double childN = c ? (c->visits + c->virtualLoss) : 0.0;
    double q = (c && c->visits > 0) ? sign * (c->valueSumWhite / c->visits) : 0.0;  // FPU 0
    double u = explore * node->childPriors[i] / (1.0 + childN);
    double score = q + u;
    if(score > best) { best = score; bestIdx = (int)i; }
  }
  return bestIdx;
}
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
      gCtx, gModel, /*logger*/nullptr, /*maxBatch*/KGE_MAX_BATCH,
      /*requireExactNNLen*/true, /*inputsUseNHWC*/gUseNHWC, /*gpuIdx*/-1, /*serverThread*/0);
    gInputs = NeuralNet::createInputBuffers(gModel, KGE_MAX_BATCH, gXLen, gYLen);
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

// Batched eval: B positions in ONE forward pass (the GPU evaluates them in
// parallel, ~the cost of one). policyOut is B*(hw+1) raw logits; valueOut is B*5
// (win/loss/noResult logits, scoreMean, lead). No ownership (search doesn't need it).
static int evalBatch(const std::vector<const Board*>& boards,
                     const std::vector<const BoardHistory*>& hists,
                     const std::vector<Player>& plas,
                     float* policyOut, float* valueOut) {
  const int B = (int)boards.size();
  const int spatialLen = NNModelVersion::getNumSpatialFeatures(gModelVersion) * gXLen * gYLen;
  const int globalLen = NNModelVersion::getNumGlobalFeatures(gModelVersion);
  std::vector<NNResultBuf> bufs(B);
  std::vector<NNResultBuf*> bufPtrs(B);
  std::vector<NNOutput> outs(B);
  std::vector<NNOutput*> outPtrs(B);
  MiscNNInputParams nnInputParams;
  for(int i = 0; i < B; i++) {
    bufs[i].rowSpatialBuf.resize(spatialLen);
    bufs[i].rowGlobalBuf.resize(globalLen);
    bufs[i].hasRowMeta = false;
    bufs[i].symmetry = 0;
    NNInputs::fillRowV7(*boards[i], *hists[i], plas[i], nnInputParams, gXLen, gYLen,
                        gUseNHWC, bufs[i].rowSpatialBuf.data(), bufs[i].rowGlobalBuf.data());
    bufPtrs[i] = &bufs[i];
    outs[i].nnXLen = gXLen; outs[i].nnYLen = gYLen; outs[i].whiteOwnerMap = nullptr;
    outPtrs[i] = &outs[i];
  }
  NeuralNet::getOutput(gHandle, gInputs, B, bufPtrs.data(), outPtrs);
  const int hw = gXLen * gYLen;
  for(int i = 0; i < B; i++) {
    std::memcpy(policyOut + (size_t)i * (hw + 1), outs[i].policyProbs, (hw + 1) * sizeof(float));
    valueOut[i*5+0] = outs[i].whiteWinProb;
    valueOut[i*5+1] = outs[i].whiteLossProb;
    valueOut[i*5+2] = outs[i].whiteNoResultProb;
    valueOut[i*5+3] = outs[i].whiteScoreMean;
    valueOut[i*5+4] = outs[i].whiteLead;
  }
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

// Batched stones-only eval — for validating batched vs single. numPos positions,
// each a hw stone-grid (stonesBatch[i*hw..]); plas[i] = side to move.
//   policyOut: numPos*(hw+1) logits; valueOut: numPos*5.
KATAEVAL_EXPORT int kgeEvalBatch(const int* stonesBatch, const int* plas, int numPos,
                                 double komi, float* policyOut, float* valueOut) {
  if(gHandle == nullptr) { gErr = "not loaded"; return 0; }
  if(numPos < 1 || numPos > KGE_MAX_BATCH) { gErr = "batch size out of range"; return 0; }
  try {
    const int hw = gXLen * gYLen;
    Rules rules = Rules::getTrompTaylorish();
    rules.komi = (float)komi;
    std::vector<Board> boards;
    boards.reserve(numPos);                 // reserve so element addresses stay stable
    for(int i = 0; i < numPos; i++) {
      Board board(gXLen, gYLen);
      const int* s = stonesBatch + (size_t)i * hw;
      for(int y = 0; y < gYLen; y++)
        for(int x = 0; x < gXLen; x++) {
          int c = s[y * gXLen + x];
          if(c == 1 || c == 2) board.setStone(Location::getLoc(x, y, gXLen), (Color)c);
        }
      boards.push_back(board);
    }
    std::vector<BoardHistory> hists;
    hists.reserve(numPos);
    std::vector<const Board*> bp;
    std::vector<const BoardHistory*> hp;
    std::vector<Player> plav;
    for(int i = 0; i < numPos; i++) hists.emplace_back(boards[i], (Player)plas[i], rules, 0);
    for(int i = 0; i < numPos; i++) { bp.push_back(&boards[i]); hp.push_back(&hists[i]); plav.push_back((Player)plas[i]); }
    return evalBatch(bp, hp, plav, policyOut, valueOut);
  } catch(const std::exception& e) {
    gErr = e.what();
    return 0;
  }
}

// Batched MCTS search from the position after replaying the move sequence.
// Runs until maxVisits or maxTimeMs (whichever first), collecting up to
// KGE_MAX_BATCH leaves per NN forward pass (virtual loss diverges them).
// Outputs: bestMoveOut[1] (y*N+x, -1 pass), winrateOut[1] (toPla win prob 0..1),
// pvOut[<=pvCap] principal variation moves + pvLenOut[1], visitsOut[1].
KATAEVAL_EXPORT int kgeSearch(const int* moveLocs, const int* moveCols, int numMoves,
                              int toPla, double komi, int maxVisits, double maxTimeMs,
                              int* bestMoveOut, float* winrateOut,
                              int* pvOut, int pvCap, int* pvLenOut, int* visitsOut) {
  if(gHandle == nullptr) { gErr = "not loaded"; return 0; }
  try {
    const int hw = gXLen * gYLen;
    const int topK = 32;
    const double cPUCT = 1.1;            // KataGo / AlphaZero exploration constant
    Rules rules = Rules::getTrompTaylorish();
    rules.komi = (float)komi;
    Board rootBoard(gXLen, gYLen);
    Player firstPla = (numMoves > 0) ? (Player)moveCols[0] : P_BLACK;
    BoardHistory rootHist(rootBoard, firstPla, rules, 0);
    for(int i = 0; i < numMoves; i++) {
      Player pla = (Player)moveCols[i];
      Loc loc = (moveLocs[i] < 0) ? Board::PASS_LOC : Location::getLoc(moveLocs[i] % gXLen, moveLocs[i] / gXLen, gXLen);
      if(rootHist.isLegal(rootBoard, loc, pla)) rootHist.makeBoardMoveAssumeLegal(rootBoard, loc, pla, NULL);
    }
    Player rootPla = (Player)toPla;

    SNode* root = new SNode();
    std::vector<float> polBuf((size_t)KGE_MAX_BATCH * (hw + 1)), valBuf((size_t)KGE_MAX_BATCH * 5);
    int totalVisits = 0;
    auto t0 = std::chrono::steady_clock::now();

    // Expand the root once up front (so the batch loop sees priors).
    {
      std::vector<const Board*> bp{&rootBoard};
      std::vector<const BoardHistory*> hp{&rootHist};
      std::vector<Player> pl{rootPla};
      evalBatch(bp, hp, pl, polBuf.data(), valBuf.data());
      expandNode(root, polBuf.data(), rootBoard, rootHist, rootPla, topK);
      root->valueSumWhite += whiteUtil(valBuf[0], valBuf[1], valBuf[2]);
      root->visits++; totalVisits++;
    }

    bool timeUp = false;
    while(totalVisits < maxVisits && !timeUp) {
      int target = std::min(KGE_MAX_BATCH, maxVisits - totalVisits);
      std::vector<SNode*> leaves;
      std::vector<std::vector<SNode*>> paths;
      std::vector<Board> boards; std::vector<BoardHistory> hists; std::vector<Player> plas;
      boards.reserve(target); hists.reserve(target);
      for(int b = 0; b < target; b++) {
        Board board = rootBoard; BoardHistory hist = rootHist; Player pla = rootPla;
        std::vector<SNode*> path; SNode* node = root; node->virtualLoss++; path.push_back(node);
        while(node->expanded) {
          int ci = selectChildPUCT(node, pla, cPUCT);
          if(ci < 0) break;
          int mv = node->childMoves[ci];
          Loc loc = (mv < 0) ? Board::PASS_LOC : Location::getLoc(mv % gXLen, mv / gXLen, gXLen);
          hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);  // legal by construction
          pla = getOpp(pla);
          if(!node->children[ci]) node->children[ci] = new SNode();
          node = node->children[ci]; node->virtualLoss++; path.push_back(node);
        }
        leaves.push_back(node); paths.push_back(std::move(path));
        boards.push_back(std::move(board)); hists.push_back(std::move(hist)); plas.push_back(pla);
      }
      int B = (int)leaves.size();
      std::vector<const Board*> bp; std::vector<const BoardHistory*> hp;
      for(int i = 0; i < B; i++) { bp.push_back(&boards[i]); hp.push_back(&hists[i]); }
      evalBatch(bp, hp, plas, polBuf.data(), valBuf.data());
      for(int i = 0; i < B; i++) {
        SNode* leaf = leaves[i];
        double wu = whiteUtil(valBuf[i*5+0], valBuf[i*5+1], valBuf[i*5+2]);
        if(!leaf->expanded) expandNode(leaf, polBuf.data() + (size_t)i * (hw + 1), boards[i], hists[i], plas[i], topK);
        for(SNode* n : paths[i]) { n->valueSumWhite += wu; n->visits++; n->virtualLoss--; }
        totalVisits++;
      }
      double el = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if(el >= maxTimeMs) timeUp = true;
    }

    // Best move = most-visited root child.
    int bestIdx = -1, bestVisits = -1;
    for(size_t i = 0; i < root->children.size(); i++) {
      int v = root->children[i] ? root->children[i]->visits : 0;
      if(v > bestVisits) { bestVisits = v; bestIdx = (int)i; }
    }
    if(bestMoveOut) *bestMoveOut = (bestIdx >= 0) ? root->childMoves[bestIdx] : -1;
    if(visitsOut) *visitsOut = totalVisits;
    double rootWhite = root->visits > 0 ? root->valueSumWhite / root->visits : 0.0;
    double plaUtil = (rootPla == P_WHITE) ? rootWhite : -rootWhite;
    if(winrateOut) *winrateOut = (float)((plaUtil + 1.0) * 0.5);
    int pvLen = 0;
    for(SNode* node = root; node && node->expanded && pvLen < pvCap; ) {
      int bi = -1, bv = -1;
      for(size_t i = 0; i < node->children.size(); i++) {
        int v = node->children[i] ? node->children[i]->visits : 0;
        if(v > bv) { bv = v; bi = (int)i; }
      }
      if(bi < 0 || bv <= 0) break;
      if(pvOut) pvOut[pvLen] = node->childMoves[bi];
      pvLen++; node = node->children[bi];
    }
    if(pvLenOut) *pvLenOut = pvLen;
    freeTree(root);
    return 1;
  } catch(const std::exception& e) {
    gErr = e.what();
    return 0;
  }
}

}  // extern "C"
