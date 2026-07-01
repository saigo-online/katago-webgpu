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
#include <random>

#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#ifdef KGE_THREADS
#include <utility>
#include <atomic>
#include <mutex>
#include "../core/logger.h"
#include "../search/search.h"
#include "../search/asyncbot.h"
#include "../search/searchparams.h"
#include "../search/timecontrols.h"
#include "../search/reportedsearchvalues.h"
#include "../search/analysisdata.h"
#endif

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
const int KGE_MAX_BATCH = 16;  // batched eval: per-batch GPU latency is ~fixed, so a
                               // bigger batch (more in-flight search threads) raises
                               // nnEvals/s — but the batch is thread-limited anyway, so
                               // 16 matches 32 for <=16-core clients while halving the
                               // intermediate buffers (big nets + batch 32 forced fragile
                               // threaded-WASM memory growth -> "unaligned" trap on b18).

// ---- Real KataGo engine (Path A): NNEvaluator + Search ---------------------
std::string gModelPath;            // stashed by kgeLoad so the NNEvaluator can load it
bool gWantFp16 = false;            // opt-in fp16 (kgeSetFp16) — default off; fp16 overflows
                                   // the trunk on g170 nets, but is ~2x on fp16-stable nets
int gGumbelN = 0;                  // kgeSearch root selection: 0 = PUCT; >0 = Gumbel-top-N + sequential halving
#ifdef KGE_THREADS
NNEvaluator* gNNEval = nullptr;    // 1 server thread owns the WebGPU device (deferred init)
AsyncBot* gBot = nullptr;          // KataGo's real engine: tree reuse (makeMove) + ponder
std::vector<std::pair<Loc, Player>> gBotLine;  // the bot's current line — for tree reuse
bool gPondering = false;           // is the bot thinking in the background right now?
#endif

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

// Opt into fp16 for the threaded engine (call BEFORE the first eval/search). Default
// off; only safe on fp16-stable nets (g170 overflow to garbage). See WEBGPU_STATUS.
KATAEVAL_EXPORT void kgeSetFp16(int v) { gWantFp16 = (v != 0); }
// kgeSearch root selection: n<=0 -> PUCT (default). n>0 -> Gumbel-top-n + sequential
// halving (Danihelka 2022) — stronger at low visits. Typical n = 16.
KATAEVAL_EXPORT void kgeSetGumbel(int n) { gGumbelN = n > 0 ? n : 0; }

KATAEVAL_EXPORT int kgeLoad(const char* modelPath, int boardSize) {
  try {
    Board::initHash();
    ScoreValue::initTables();
    NeuralNet::globalInitialize();
    gModelPath = modelPath;  // NNEvaluator (kgeSearchKata) reloads from this path
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
#ifndef KGE_THREADS
    // Single-thread build: the direct handle is the backend (kgeEval / kgeEvalSeq).
    gHandle = NeuralNet::createComputeHandle(
      gCtx, gModel, /*logger*/nullptr, /*maxBatch*/KGE_MAX_BATCH,
      /*requireExactNNLen*/true, /*inputsUseNHWC*/gUseNHWC, /*gpuIdx*/-1, /*serverThread*/0);
    gInputs = NeuralNet::createInputBuffers(gModel, KGE_MAX_BATCH, gXLen, gYLen);
#endif
    // Threaded build: skip the direct handle — the lazily-created NNEvaluator is the
    // SOLE backend for both analysis (kgeEvalSeqKata) and search (kgeSearchKata), so
    // the net loads to the GPU exactly once and both share its NN cache. gCtx stays
    // (cheap: device init is deferred) purely so kgeBackendIsGpu reports correctly.
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

// Build a Board + BoardHistory from a flat move list — the ONE replay used by every
// eval/search entry, so all paths agree. Handles handicap setup: a leading run of >= 2
// same-color stones is placed as SETUP, not as moves, because after the first stone it's
// the other player's turn and isLegal() rejects an out-of-turn move (boardhistory.cpp) —
// replaying them as moves would drop all but the first. Mirrors KataGo's placeFreeHandicap
// (gtp.cpp): place the stones, then start the history with the opponent to move. moveLocs[i]
// is y*xLen+x (or < 0 for a pass); moveCols[i] is 1 (black) / 2 (white).
static void buildBoardHist(const int* moveLocs, const int* moveCols, int numMoves, double komi,
                           Board& board, BoardHistory& hist) {
  Rules rules = Rules::getTrompTaylorish();
  rules.komi = (float)komi;
  board = Board(gXLen, gYLen);
  auto toLoc = [](int mv) {
    return (mv < 0) ? Board::PASS_LOC : Location::getLoc(mv % gXLen, mv / gXLen, gXLen);
  };
  // Real Go handicap is ALWAYS a run of BLACK setup stones; only treat a leading run of
  // >= 2 BLACK stones as handicap. (Restricting to black avoids misreading a malformed
  // leading white run as "white handicap" and applying the wrong komi.)
  int setup = 0;
  if(numMoves >= 2 && moveCols[0] == P_BLACK)
    while(setup < numMoves && moveLocs[setup] >= 0 && moveCols[setup] == P_BLACK) setup++;
  if(setup >= 2) {
    for(int i = 0; i < setup; i++) board.setStone(toLoc(moveLocs[i]), C_BLACK);
    Player nextPla = (numMoves > setup) ? (Player)moveCols[setup] : P_WHITE;
    hist = BoardHistory(board, nextPla, rules, 0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);  // correct komi/score for handicap
  } else {
    setup = 0;
    Player firstPla = (numMoves > 0) ? (Player)moveCols[0] : P_BLACK;
    hist = BoardHistory(board, firstPla, rules, 0);
  }
  for(int i = setup; i < numMoves; i++) {
    Loc loc = toLoc(moveLocs[i]);
    Player pla = (Player)moveCols[i];
    if(hist.isLegal(board, loc, pla)) hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
  }
}

// Move-sequence eval: replays moves (captures + ko/superko + the recent-move input
// features), writes the resulting board to boardOut (0 empty, 1 black, 2 white, row-major),
// then evaluates for player toPla.
KATAEVAL_EXPORT int kgeEvalSeq(const int* moveLocs, const int* moveCols, int numMoves,
                               int toPla, double komi, int* boardOut,
                               float* policyOut, float* valueOut, float* ownerOut) {
  if(gHandle == nullptr) { gErr = "not loaded"; return 0; }
  try {
    Board board; BoardHistory hist;
    buildBoardHist(moveLocs, moveCols, numMoves, komi, board, hist);
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
// One simulation for the Gumbel path: descend from the root, forcing the FIRST move to be
// root child `forceRootChild`, then PUCT below; expand + eval the leaf and back up. Batch 1
// (Gumbel runs in the low-visit regime, so a per-sim eval is acceptable).
static void simulateForced(SNode* root, int forceRootChild, const Board& rootBoard,
                           const BoardHistory& rootHist, Player rootPla, int topK, double cPUCT,
                           float* polBuf, float* valBuf) {
  Board board = rootBoard; BoardHistory hist = rootHist; Player pla = rootPla;
  std::vector<SNode*> path; SNode* node = root; path.push_back(node);
  bool first = true;
  while(node->expanded) {
    int ci = (first && forceRootChild >= 0) ? forceRootChild : selectChildPUCT(node, pla, cPUCT);
    first = false;
    if(ci < 0) break;
    int mv = node->childMoves[ci];
    Loc loc = (mv < 0) ? Board::PASS_LOC : Location::getLoc(mv % gXLen, mv / gXLen, gXLen);
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    pla = getOpp(pla);
    if(!node->children[ci]) node->children[ci] = new SNode();
    node = node->children[ci]; path.push_back(node);
  }
  std::vector<const Board*> bp{&board}; std::vector<const BoardHistory*> hp{&hist}; std::vector<Player> pl{pla};
  evalBatch(bp, hp, pl, polBuf, valBuf);
  double wu = whiteUtil(valBuf[0], valBuf[1], valBuf[2]);
  if(!node->expanded) expandNode(node, polBuf, board, hist, pla, topK);
  for(SNode* n : path) { n->valueSumWhite += wu; n->visits++; }
}

// Gumbel-AlphaZero root action selection (Danihelka et al. 2022, "Policy improvement by
// planning with Gumbel"). Sample n candidate root actions via Gumbel-top-k on the policy
// LOGITS, then run SEQUENTIAL HALVING over them under the visit budget, ranking each round by
// g(a) + logit(a) + sigma(q̂(a)). At low visits this is markedly stronger + better-calibrated
// than plain PUCT (which over-commits to the policy argmax). Returns the chosen root child
// index. rootLogits are the raw policy logits for the root's children (same order).
static int runGumbelRoot(SNode* root, const std::vector<float>& rootLogits, int nConsider,
                         int maxVisits, const Board& rootBoard, const BoardHistory& rootHist,
                         Player rootPla, int topK, double cPUCT, uint32_t seed,
                         float* polBuf, float* valBuf) {
  const int C = (int)root->childMoves.size();
  if(C <= 0) return -1;
  const double sign = (rootPla == P_WHITE) ? 1.0 : -1.0;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(1e-12, 1.0);
  std::vector<double> g(C);
  for(int i = 0; i < C; i++) g[i] = -std::log(-std::log(U(rng)));   // Gumbel(0,1) noise
  // Candidate set = top-m root actions by (logit + Gumbel).
  std::vector<int> cand(C);
  for(int i = 0; i < C; i++) cand[i] = i;
  int m = std::min(std::max(1, nConsider), C);
  std::partial_sort(cand.begin(), cand.begin() + m, cand.end(),
    [&](int a, int b){ return rootLogits[a] + g[a] > rootLogits[b] + g[b]; });
  cand.resize(m);
  if(m == 1) {  // still spend the budget deepening the single candidate
    for(int used = 0; used < maxVisits; used++)
      simulateForced(root, cand[0], rootBoard, rootHist, rootPla, topK, cPUCT, polBuf, valBuf);
    return cand[0];
  }
  const double cVisit = 50.0, cScale = 1.0;   // Danihelka's sigma monotone transform constants
  int numRounds = std::max(1, (int)std::ceil(std::log2((double)m)));
  auto qhat = [&](int ci) {                    // completed Q from the ROOT player's perspective, [-1,1]
    SNode* c = root->children[ci];
    return (c && c->visits > 0) ? sign * (c->valueSumWhite / c->visits) : 0.0;
  };
  int used = 0;
  while((int)cand.size() > 1 && used < maxVisits) {
    int mr = (int)cand.size();
    int per = std::max(1, (int)std::floor((double)maxVisits / (double)(numRounds * mr)));
    for(int a : cand)
      for(int s = 0; s < per && used < maxVisits; s++) {
        simulateForced(root, a, rootBoard, rootHist, rootPla, topK, cPUCT, polBuf, valBuf); used++;
      }
    double maxN = 0; for(int a : cand) { SNode* c = root->children[a]; if(c) maxN = std::max(maxN, (double)c->visits); }
    double sigmaScale = (cVisit + maxN) * cScale;
    std::sort(cand.begin(), cand.end(), [&](int a, int b){
      return g[a] + rootLogits[a] + sigmaScale * qhat(a) > g[b] + rootLogits[b] + sigmaScale * qhat(b);
    });
    cand.resize((mr + 1) / 2);                 // keep the top half (ceil)
  }
  int winner = cand[0];
  while(used < maxVisits) {                    // leftover budget -> deepen the survivor
    simulateForced(root, winner, rootBoard, rootHist, rootPla, topK, cPUCT, polBuf, valBuf); used++;
  }
  return winner;
}

KATAEVAL_EXPORT int kgeSearch(const int* moveLocs, const int* moveCols, int numMoves,
                              int toPla, double komi, int maxVisits, double maxTimeMs,
                              int* bestMoveOut, float* winrateOut,
                              int* pvOut, int pvCap, int* pvLenOut, int* visitsOut) {
  if(gHandle == nullptr) { gErr = "not loaded"; return 0; }
  try {
    const int hw = gXLen * gYLen;
    const int topK = 32;
    const double cPUCT = 1.1;            // KataGo / AlphaZero exploration constant
    Board rootBoard; BoardHistory rootHist;
    buildBoardHist(moveLocs, moveCols, numMoves, komi, rootBoard, rootHist);
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

    // Gumbel-AlphaZero root selection (kgeSetGumbel) — sample n root actions + sequential
    // halving. Much stronger than PUCT in this exact low-visit regime. The root's child
    // policy LOGITS (captured now, before polBuf is overwritten) drive the Gumbel scores.
    int gumbelWinner = -1;
    if(gGumbelN > 0 && !root->childMoves.empty()) {
      std::vector<float> rootLogits(root->childMoves.size());
      for(size_t i = 0; i < root->childMoves.size(); i++)
        rootLogits[i] = (root->childMoves[i] >= 0) ? polBuf[root->childMoves[i]] : polBuf[hw];
      gumbelWinner = runGumbelRoot(root, rootLogits, gGumbelN, std::max(0, maxVisits - 1),
                                   rootBoard, rootHist, rootPla, topK, cPUCT,
                                   0x9E3779B9u ^ (uint32_t)numMoves ^ ((uint32_t)maxVisits << 8),
                                   polBuf.data(), valBuf.data());
      totalVisits = root->visits;
    }

    bool timeUp = false;
    while(gumbelWinner < 0 && totalVisits < maxVisits && !timeUp) {
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

    // Best move: the Gumbel winner if Gumbel ran, else the most-visited root child (PUCT).
    int bestIdx = gumbelWinner, bestVisits = -1;
    if(bestIdx < 0)
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

#ifdef KGE_THREADS
// ---- Real KataGo engine (Path A) -------------------------------------------
// ONE NNEvaluator (1 server thread owns the WebGPU device — init deferred to
// createComputeHandle so the thread-local device lives on that thread) backs BOTH
// analysis (kgeEvalSeqKata) and search (kgeSearchKata): one net load, one shared NN
// cache. AsyncBot adds tree reuse (makeMove) + background ponder. Search worker
// threads only queue batched NN requests — they never touch WebGPU objects.
static Logger& kgeLogger() {
  static Logger logger(nullptr, false, false, false, false);  // silent
  return logger;
}

static bool ensureKataEngine() {
  if(gBot != nullptr) return true;
  ConfigParser cfg;  // consulted-only by the backend during construction; not stored
  gNNEval = new NNEvaluator(
    "kge", gModelPath, "", &kgeLogger(),
    KGE_MAX_BATCH, gXLen, gYLen,
    /*requireExactNNLen*/true, /*inputsUseNHWC*/false,  // WebGPU = NCHW
    /*nnCacheSizePowerOfTwo*/16, /*nnMutexPoolSizePowerofTwo*/14,  // 64K entries: plenty
    // for browser searches (a few k visits), vs the desktop 1M that could reach GBs of
    // cached NNOutputs (ownership+policy) on big nets — a prime cause of the fragile
    // threaded-WASM memory growth that crashed b18.
    /*debugSkipNeuralNet*/false, /*homeDataDirOverride*/"",
    // fp32 by default — fp16 overflows the trunk on g170 nets (garbage); opt in via
    // kgeSetFp16 for fp16-stable (modern mish_scale8/silu) nets to get the ~2x win.
    gWantFp16 ? enabled_t::Auto : enabled_t::False,
    /*numThreads (NN server)*/1, /*gpuIdxByServerThread*/std::vector<int>{-1},
    /*randSeed*/"kge-nneval", /*doRandomize*/false, /*defaultSymmetry*/0,
    /*disableWarmup*/true, cfg);
  gNNEval->spawnServerThreads();
  gBot = new AsyncBot(SearchParams::basicDecentParams(), gNNEval, /*humanEval*/NULL, &kgeLogger(), "kge-search");
  gBot->setAlwaysIncludeOwnerMap(true);  // so getAverageTreeOwnership (the territory heatmap) is available
  // Free the CPU LoadedModel now: the NNEvaluator loaded its OWN copy from gModelPath,
  // and the threaded worker never touches gModel again (metadata is cached in
  // gModelVersion/gXLen; the "loaded" sentinel is gModelPath). Saves the net's full
  // weight size in CPU RAM (~94MB for b18) — the double-load was blowing the browser's
  // memory budget on big nets and forcing the fragile threaded-WASM growth.
  if(gModel != nullptr) { NeuralNet::freeLoadedModel(gModel); gModel = nullptr; }
  return true;
}

// Cross-thread state. gSearchBestLoc is written by the async move-callback on the bot
// thread and read in kgePoll on the main thread; gPollPla is written on the main thread
// (kgeSearchBegin) and read in writeSnapshot on the callback thread; gPollReused likewise
// crosses threads. All atomic to avoid torn reads / visibility hazards.
static std::atomic<bool> gSearchDone{true};
static std::atomic<Loc> gSearchBestLoc{Board::NULL_LOC};
static std::atomic<Player> gPollPla{P_BLACK};
static std::atomic<int> gPollReused{0};
// Strength limiting (kgeSetStrength). Written on the main thread (from JS) and read on the
// main thread (prepareSearch) — same-thread, but atomic for cheap peace-of-mind. gStrMaxVisits
// > 0 caps the search depth (fewer visits = weaker); gStrPolicyTemp / gStrChosenTemp flatten
// the root policy and sample a non-best move (more human/varied = weaker). See kgeSetStrength.
static std::atomic<int> gStrMaxVisits{0};
static std::atomic<float> gStrChosenTemp{0.0f};
static std::atomic<float> gStrPolicyTemp{1.0f};
// Adjust playing strength, applied to the NEXT search. Three orthogonal knobs:
//   maxVisits<=0 leaves the per-request budget alone; >0 caps it (fewer = weaker).
//   chosenMoveTemp>0 sets chosenMoveTemperature: the FINAL move is sampled instead of taken
//     as the max. At many visits this samples ~ visits^(1/T); at maxVisits=1 there is no
//     visit distribution, so KataGo falls back to sampling the RAW POLICY ^ (1/T) — i.e. a
//     policy-sampling weak mode, NOT visit-sampling (be honest about which regime you're in).
//   policyTemp>1 sets rootPolicyTemperature: flattens the root policy so weak play spreads
//     over more moves (and the low-visit policy-sampling above draws from a wider set),
//     giving more natural variety than temperature alone. 1.0 = unchanged.
KATAEVAL_EXPORT void kgeSetStrength(int maxVisits, float chosenMoveTemp, float policyTemp) {
  gStrMaxVisits.store(maxVisits);
  gStrChosenTemp.store(chosenMoveTemp > 0.0f ? chosenMoveTemp : 0.0f);
  gStrPolicyTemp.store(policyTemp > 0.0f ? policyTemp : 1.0f);
}
static void stopPonder() {
  if(gBot != nullptr) { gBot->stopAndWait(); gPondering = false; gSearchDone = true; }
}

static int locToIdx(Loc loc) {
  if(loc == Board::PASS_LOC || loc == Board::NULL_LOC) return -1;
  return Location::getY(loc, gXLen) * gXLen + Location::getX(loc, gXLen);
}
static Loc decodeLoc(int mv) {
  return (mv < 0) ? Board::PASS_LOC : Location::getLoc(mv % gXLen, mv / gXLen, gXLen);
}

// ---- Analysis snapshot: the ENGINE pushes its current thinking here from inside the
// search's periodic analyze callback (a safe point where reading the tree is legal —
// this is how KataGo's kata-analyze streams). kgePoll/kgeCandidates just read this
// buffer under a mutex, never touching the live tree — so no reader-vs-search race.
static const int SNAP_PV = 32, SNAP_CAND = 8, SNAP_MAXPTS = 361;
struct Snap {
  int best = -1; float wr = 0.5f, score = 0.0f; int visits = 0;
  float rawWr = 0.5f, rawScore = 0.0f;             // instant NN read (pre-search), side-to-move
  float scoreStdev = 0.0f;                          // root score uncertainty, points
  float surprise = 0.0f, searchEntropy = 0.0f, policyEntropy = 0.0f;  // position descriptors (nats)
  int pv[SNAP_PV]; int pvVisits[SNAP_PV]; int pvLen = 0;
  struct Cand { int loc, visits; float wr, score, prior, lcb, radius, stdev; } cand[SNAP_CAND]; int nCand = 0;
  float own[SNAP_MAXPTS]; int ownLen = 0;           // tree-averaged ownership, WHITE-perspective (+1 = white)
};
static Snap gSnap;
static std::mutex gSnapMutex;
// Bumped by kgeSearchBegin at the start of each search so writeSnapshot recomputes the
// once-per-search fields (raw NN read). gSnapWantOwn gates the expensive ownership map.
static std::atomic<uint32_t> gSnapSearchId{0};
static std::atomic<bool> gSnapWantOwn{true};

// Called by AsyncBot's analyze machinery (genMoveAsyncAnalyze / analyzeAsync) at a safe
// point during the search — so getAnalysisData / appendPV are safe here. Runs on the single
// callbackLoopThread, so the function-local statics below (per-search / throttle caches) are
// only ever touched by one thread and need no locking.
static void writeSnapshot(const Search* s) {
  Snap snap;
  // EXCEPTION FIREWALL: this runs on KataGo's callbackLoopThread (asyncbot.cpp), which
  // invokes us with NO try/catch WHILE the search worker threads mutate the tree. Any
  // exception that escapes here unwinds out of the std::thread -> std::terminate() ->
  // Aborted(). So we catch everything and simply skip publishing this frame — the reader
  // keeps the last good snapshot. (This is the fix for the self-play "Aborted()" crash.)
  try {
  const bool stmWhite = (gPollPla == P_WHITE);
  ReportedSearchValues vals;
  if(s->getRootValues(vals)) {
    snap.wr    = (float)(stmWhite ? vals.winValue : vals.lossValue);
    snap.score = (float)(stmWhite ? vals.lead : -vals.lead);
    snap.scoreStdev = (float)vals.expectedScoreStdev;
    snap.visits = (int)vals.visits;
  }
  // Instant NN read (pre-search) — the delta vs the searched values shows what reading is
  // buying. This is a pure function of the fixed root NN eval, so it's constant for the whole
  // search: compute it ONCE per search (keyed on gSnapSearchId) and reuse. Perspective is
  // fixed per search too, so caching the side-to-move value is fine.
  static uint32_t rawComputedFor = 0xFFFFFFFFu;
  static float sRawWr = 0.5f, sRawScore = 0.0f;
  uint32_t sid = gSnapSearchId.load();
  if(sid != rawComputedFor) {
    ReportedSearchValues raw;
    if(s->getRootRawNNValues(raw)) {  // only mark done on success (root may not be evaluated on the first frame)
      sRawWr    = (float)(stmWhite ? raw.winValue : raw.lossValue);
      sRawScore = (float)(stmWhite ? raw.lead : -raw.lead);
      rawComputedFor = sid;
    }
  }
  snap.rawWr = sRawWr; snap.rawScore = sRawScore;
  // Position descriptors: surprise = how far search moved from the raw policy (tactical),
  // entropy = how spread the move choice is (sharp vs calm).
  double sp = 0.0, se = 0.0, pe = 0.0;
  if(s->getPolicySurpriseAndEntropy(sp, se, pe)) {
    snap.surprise = (float)sp; snap.searchEntropy = (float)se; snap.policyEntropy = (float)pe;
  }
  std::vector<AnalysisData> buf;
  s->getAnalysisData(buf, SNAP_CAND, false, 1, false);
  for(size_t i = 0; i < buf.size() && snap.nCand < SNAP_CAND; i++) {
    const AnalysisData& a = buf[i];
    double whiteWr = (a.winLossValue + 1.0) * 0.5;
    snap.cand[snap.nCand] = { locToIdx(a.move), (int)a.numVisits,
      (float)(stmWhite ? whiteWr : 1.0 - whiteWr),
      (float)(stmWhite ? a.lead : -a.lead), (float)a.policyPrior,
      (float)(stmWhite ? a.lcb : -a.lcb), (float)a.radius, (float)a.scoreStdev };
    snap.nCand++;
  }
  snap.best = buf.empty() ? -1 : locToIdx(buf[0].move);
  std::vector<Loc> pvBuf; std::vector<int64_t> vb, eb; std::vector<Loc> sl; std::vector<double> sv;
  if(s->getRootNode() != NULL) s->appendPV(pvBuf, vb, eb, sl, sv, s->getRootNode(), SNAP_PV);
  for(size_t i = 0; i < pvBuf.size() && snap.pvLen < SNAP_PV; i++) {
    snap.pv[snap.pvLen] = locToIdx(pvBuf[i]);
    snap.pvVisits[snap.pvLen] = (i < vb.size()) ? (int)vb[i] : 0;
    snap.pvLen++;
  }
  // Tree-averaged territory heatmap (white-perspective). This is the ONE expensive read —
  // getAverageTreeOwnership traverses the tree (O(visits^0.75)) + allocates each call. So:
  // (a) only when the UI wants it (gSnapWantOwn), and (b) throttled — recompute only when
  // visits grew >= ~25% since the last map (it barely changes between 80ms frames) or on a
  // new search; otherwise carry the cached map forward. Guarded: throws if the root isn't
  // evaluated yet or the owner map is off.
  static uint32_t ownComputedFor = 0xFFFFFFFFu;
  static int ownAtVisits = 0, ownLen = 0;
  static float ownCache[SNAP_MAXPTS];
  if(gSnapWantOwn.load()) {
    bool newSearch = (sid != ownComputedFor);
    if(newSearch) { ownLen = 0; ownAtVisits = 0; }
    if(newSearch || snap.visits >= ownAtVisits + ownAtVisits/4 + 8) {
      try {
        std::vector<double> own = s->getAverageTreeOwnership(P_WHITE, s->getRootNode(), 0);
        ownLen = 0;
        for(size_t i = 0; i < own.size() && ownLen < SNAP_MAXPTS; i++) ownCache[ownLen++] = (float)own[i];
        ownComputedFor = sid; ownAtVisits = snap.visits;
      } catch(...) { /* keep the last cached map */ }
    }
    snap.ownLen = ownLen;
    for(int i = 0; i < ownLen; i++) snap.own[i] = ownCache[i];
  }
  } catch(...) { return; }  // firewall (see top): never let an exception escape this thread
  { std::lock_guard<std::mutex> lk(gSnapMutex); gSnap = snap; }
}
static void analyzeCb(const Search* s) noexcept { writeSnapshot(s); }

// Instant analysis via the shared NNEvaluator. Returns PROBABILITIES (distinct from
// the direct kgeEvalSeq, which gives logits): policyOut in [0,1] with illegal = -1
// (NNPos layout, pass at index hw); valueOut = {pWhiteWin, pWhiteLoss, pNoResult,
// whiteScoreMean (points), whiteLead}; ownerOut in [-1,1] (white positive).
KATAEVAL_EXPORT int kgeEvalSeqKata(const int* moveLocs, const int* moveCols, int numMoves,
                                   int toPla, double komi,
                                   int* boardOut, float* policyOut, float* valueOut, float* ownerOut) {
  if(gModelPath.empty()) { gErr = "not loaded"; return 0; }
  try {
    if(!ensureKataEngine()) { gErr = "engine init failed"; return 0; }
    stopPonder();
    Board board; BoardHistory hist;
    buildBoardHist(moveLocs, moveCols, numMoves, komi, board, hist);
    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    gNNEval->evaluate(board, hist, (Player)toPla, nnInputParams, buf,
                      /*skipCache*/false, /*includeOwnerMap*/ownerOut != nullptr);
    std::shared_ptr<NNOutput> out = buf.result;
    const int hw = gXLen * gYLen;
    if(boardOut)
      for(int y = 0; y < gYLen; y++) for(int x = 0; x < gXLen; x++)
        boardOut[y*gXLen+x] = (int)board.colors[Location::getLoc(x, y, gXLen)];
    if(policyOut) {
      for(int i = 0; i < hw; i++) policyOut[i] = out->policyProbs[i];
      policyOut[hw] = out->policyProbs[NNPos::locToPos(Board::PASS_LOC, gXLen, gXLen, gYLen)];
    }
    if(valueOut) {
      valueOut[0] = out->whiteWinProb; valueOut[1] = out->whiteLossProb; valueOut[2] = out->whiteNoResultProb;
      valueOut[3] = out->whiteScoreMean; valueOut[4] = out->whiteLead;
    }
    if(ownerOut && out->whiteOwnerMap != NULL)
      for(int i = 0; i < hw; i++) ownerOut[i] = out->whiteOwnerMap[i];
    return 1;
  } catch(const std::exception& e) { gErr = e.what(); return 0; }
}

// Shared prologue for BOTH search entry points: engine init, tree reuse (advance via
// makeMove when the request forward-extends the current line, else rebuild from scratch),
// and SearchParams incl. strength limiting. ponderMode adds the unbounded pondering caps.
// Sets gBotLine + gPollReused. Returns false (with gErr) on failure.
static bool prepareSearch(const int* moveLocs, const int* moveCols, int numMoves,
                          Player rootPla, double komi, int maxVisits, double maxTimeMs,
                          int numSearchThreads, bool ponderMode) {
  if(!ensureKataEngine()) { gErr = "engine init failed"; return false; }
  stopPonder();
  std::vector<std::pair<Loc,Player>> newLine;
  for(int i = 0; i < numMoves; i++) newLine.push_back({ decodeLoc(moveLocs[i]), (Player)moveCols[i] });
  size_t common = 0;
  while(common < gBotLine.size() && common < newLine.size() && gBotLine[common] == newLine[common]) common++;
  bool reused = false;
  if(gBot->getSearch()->getRootNode() != NULL && common == gBotLine.size()) {
    reused = true;
    for(size_t i = common; i < newLine.size() && reused; i++)
      if(!gBot->makeMove(newLine[i].first, newLine[i].second)) reused = false;
  }
  if(!reused) {
    Board board; BoardHistory hist;
    buildBoardHist(moveLocs, moveCols, numMoves, komi, board, hist);
    gBot->setPosition(rootPla, board, hist);
  }
  gBotLine = newLine;
  gPollReused.store(reused ? 1 : 0);

  SearchParams params = SearchParams::basicDecentParams();
  params.maxVisits = maxVisits; params.maxPlayouts = maxVisits;
  params.maxTime = maxTimeMs / 1000.0;
  params.numThreads = numSearchThreads > 0 ? numSearchThreads : 1;
  if(ponderMode) { params.maxVisitsPondering = 1000000000; params.maxTimePondering = 1e20; }  // ponder till stopped
  // Strength limiting (kgeSetStrength) — applies to BOTH entry points: visit cap, plus
  // move-selection + root-policy temperature (see kgeSetStrength for the honest semantics).
  int strVisits = gStrMaxVisits.load();
  if(strVisits > 0 && (int64_t)strVisits < params.maxVisits) {
    params.maxVisits = strVisits; params.maxPlayouts = strVisits;
  }
  float chosenT = gStrChosenTemp.load();
  if(chosenT > 0.0f) { params.chosenMoveTemperature = chosenT; params.chosenMoveTemperatureEarly = chosenT; }
  float polT = gStrPolicyTemp.load();
  if(polT > 0.0f && polT != 1.0f) { params.rootPolicyTemperature = polT; params.rootPolicyTemperatureEarly = polT; }
  gBot->setParamsNoClearing(params);  // keep the reused tree
  return true;
}

KATAEVAL_EXPORT int kgeSearchKata(const int* moveLocs, const int* moveCols, int numMoves,
                                  int toPla, double komi, int maxVisits, double maxTimeMs,
                                  int numSearchThreads,
                                  int* bestMoveOut, float* winrateOut, float* scoreOut,
                                  int* pvOut, int pvCap, int* pvLenOut, int* visitsOut, int* reusedOut) {
  if(gModelPath.empty()) { gErr = "not loaded"; return 0; }
  try {
    Player rootPla = (Player)toPla;
    if(!prepareSearch(moveLocs, moveCols, numMoves, rootPla, komi, maxVisits, maxTimeMs,
                      numSearchThreads, /*ponderMode*/false)) return 0;
    bool reused = gPollReused.load();

    TimeControls tc;
    Loc best = gBot->genMoveSynchronous(rootPla, tc);
    if(bestMoveOut) *bestMoveOut = locToIdx(best);
    if(reusedOut) *reusedOut = reused ? 1 : 0;

    const Search* s = gBot->getSearch();
    ReportedSearchValues vals;
    if(s->getRootValues(vals)) {
      if(winrateOut) *winrateOut = (float)((rootPla == P_WHITE) ? vals.winValue : vals.lossValue);
      if(scoreOut)   *scoreOut   = (float)((rootPla == P_WHITE) ? vals.lead : -vals.lead);
      if(visitsOut)  *visitsOut  = (int)vals.visits;
    }
    int pvLen = 0;
    if(pvOut && pvCap > 0) {
      std::vector<Loc> pvBuf; std::vector<int64_t> vb, eb; std::vector<Loc> sl; std::vector<double> sv;
      s->appendPV(pvBuf, vb, eb, sl, sv, s->getRootNode(), pvCap);
      for(size_t i = 0; i < pvBuf.size() && pvLen < pvCap; i++) pvOut[pvLen++] = locToIdx(pvBuf[i]);
    }
    if(pvLenOut) *pvLenOut = pvLen;

    // Keep thinking in the background: a re-search reuses the deeper tree, and if the
    // user plays a move we makeMove straight into an already-explored subtree.
    gBot->ponder();
    gPondering = true;
    return 1;
  } catch(const std::exception& e) { gErr = e.what(); return 0; }
}

// ---- Async search with live polling --------------------------------------------
// To WATCH the engine think, start the search asynchronously (genMoveAsync runs it on
// the bot's threads, non-blocking) and let the worker poll kgePoll to stream live,
// monotonically-climbing stats. When it finishes, kgePonderBegin starts a REAL
// background ponder() that keeps deepening the SAME tree — so visits genuinely climb
// during ponder too. One continuous search per phase (no per-slice restarts).
KATAEVAL_EXPORT int kgeSearchBegin(const int* moveLocs, const int* moveCols, int numMoves,
                                   int toPla, double komi, int maxVisits, double maxTimeMs,
                                   int numSearchThreads) {
  if(gModelPath.empty()) { gErr = "not loaded"; return 0; }
  try {
    Player rootPla = (Player)toPla;
    gPollPla.store(rootPla);
    if(!prepareSearch(moveLocs, moveCols, numMoves, rootPla, komi, maxVisits, maxTimeMs,
                      numSearchThreads, /*ponderMode*/true)) return 0;

    gSearchDone = false;
    gSearchBestLoc.store(Board::NULL_LOC);
    gSnapWantOwn.store(true);   // interactive search: keep the ownership heatmap fresh
    gSnapSearchId.fetch_add(1); // new search -> writeSnapshot recomputes the once-per-search fields
    { std::lock_guard<std::mutex> lk(gSnapMutex); gSnap = Snap(); }  // clear any stale snapshot
    // genMoveAsyncAnalyze: search to the target AND fire analyzeCb periodically from a
    // safe point (writes the snapshot buffer that kgePoll/kgeCandidates read).
    gBot->genMoveAsyncAnalyze(rootPla, 0, TimeControls(), 1.0,
      [](Loc loc, int, Search*) { gSearchBestLoc.store(loc); gSearchDone.store(true); },
      /*callbackPeriod*/0.08, /*firstCallbackAfter*/0.05, analyzeCb);
    return 1;
  } catch(const std::exception& e) { gErr = e.what(); return 0; }
}

// ONE combined reader for the whole snapshot — the JS worker polls this every ~90ms, so we
// take the lock and copy gSnap ONCE per poll (the old kgePoll/kgeCandidates/kgeInsights/
// kgeOwnership each took the lock and full-copied the 361-float struct separately: 4x the
// work per cycle). Fills caller-owned buffers; no tree access, so no race with the search.
//   scalarsOut[7] = {best, done, reused, visits, pvLen, nCand, ownLen}
//   valuesOut[8]  = {wr, score, scoreStdev, rawWr, rawScore, surprise, searchEntropy, policyEntropy}
//   pv/pvVisits: up to pvCap;  cand*: up to candCap (8 parallel arrays);  own: up to ownCap.
KATAEVAL_EXPORT int kgePollAll(int* scalarsOut, float* valuesOut,
                               int* pvOut, int* pvVisitsOut, int pvCap,
                               int* candLocOut, int* candVisOut, float* candWrOut, float* candScOut,
                               float* candPrOut, float* candLcbOut, float* candRadOut, float* candStdOut, int candCap,
                               float* ownOut, int ownCap) {
  if(gBot == nullptr) { gErr = "no engine"; return 0; }
  Snap snap;
  { std::lock_guard<std::mutex> lk(gSnapMutex); snap = gSnap; }  // one lock, one copy per poll
  bool done = gSearchDone.load();
  int pvLen = 0;
  for(; pvLen < snap.pvLen && pvLen < pvCap; pvLen++) { pvOut[pvLen] = snap.pv[pvLen]; pvVisitsOut[pvLen] = snap.pvVisits[pvLen]; }
  int nc = 0;
  for(; nc < snap.nCand && nc < candCap; nc++) {
    candLocOut[nc] = snap.cand[nc].loc;   candVisOut[nc] = snap.cand[nc].visits;
    candWrOut[nc]  = snap.cand[nc].wr;    candScOut[nc]  = snap.cand[nc].score;   candPrOut[nc]  = snap.cand[nc].prior;
    candLcbOut[nc] = snap.cand[nc].lcb;   candRadOut[nc] = snap.cand[nc].radius;  candStdOut[nc] = snap.cand[nc].stdev;
  }
  int nOwn = 0;
  for(; nOwn < snap.ownLen && nOwn < ownCap; nOwn++) ownOut[nOwn] = snap.own[nOwn];
  scalarsOut[0] = done ? locToIdx(gSearchBestLoc.load()) : snap.best;
  scalarsOut[1] = done ? 1 : 0;   scalarsOut[2] = gPollReused.load();
  scalarsOut[3] = snap.visits;    scalarsOut[4] = pvLen;   scalarsOut[5] = nc;   scalarsOut[6] = nOwn;
  valuesOut[0] = snap.wr;         valuesOut[1] = snap.score;       valuesOut[2] = snap.scoreStdev;
  valuesOut[3] = snap.rawWr;      valuesOut[4] = snap.rawScore;
  valuesOut[5] = snap.surprise;   valuesOut[6] = snap.searchEntropy; valuesOut[7] = snap.policyEntropy;
  return 1;
}

KATAEVAL_EXPORT int kgePonderBegin() {  // background ponder that ALSO streams the snapshot
  if(gBot == nullptr) { gErr = "no engine"; return 0; }
  try { gSnapSearchId.fetch_add(1);  // ponder is a fresh analysis pass -> recompute once-per-search fields
        gBot->analyzeAsync(gPollPla.load(), 1.0, /*callbackPeriod*/0.15, /*firstAfter*/0.05, analyzeCb);
        gPondering = true; return 1; }
  catch(const std::exception& e) { gErr = e.what(); return 0; }
}

KATAEVAL_EXPORT int kgeStopSearch() {  // stop search/ponder before a new request
  if(gBot == nullptr) return 1;
  try { gBot->stopAndWait(); gPondering = false; gSearchDone = true; return 1; }
  catch(const std::exception& e) { gErr = e.what(); return 0; }
}
#endif  // KGE_THREADS

}  // extern "C"
