// kataeval — the one clean dependency for "KataGo neural-net evaluation".
//
// A small C ABI over the minimal KataGo subset needed to load a model and
// evaluate a position on the WebGPU backend: no search, no GTP, no command/.
// Both the WASM browser build and any native embedder depend on just THIS
// (kataeval.h + the kataeval library); the internal KataGo source set is
// encapsulated in kataeval/sources.txt and built into one library target.
//
// Outputs are raw logits — the caller applies softmax (so it can mask illegal
// moves for the policy and pick the value/score transform it wants).

#ifndef KATAEVAL_H_
#define KATAEVAL_H_

#ifdef __cplusplus
extern "C" {
#endif

// Load a model from a filesystem path (a MEMFS path in the browser) and set up
// the WebGPU compute handle for a square board of `boardSize`. Returns 1 on
// success, 0 on failure (see kgeError). Acquires the GPU device asynchronously
// (blocks via Asyncify in the browser).
int kgeLoad(const char* modelPath, int boardSize);

// Evaluate one position.
//   stones:    boardSize*boardSize ints, row-major (y*size + x): 0 empty, 1 black, 2 white.
//   pla:       side to move — 1 black, 2 white.
//   komi:      white komi.
//   policyOut: boardSize*boardSize + 1 raw policy logits (last entry = pass).
//   valueOut:  5 floats — win/loss/noResult logits, then whiteScoreMean, whiteLead.
//   ownerOut:  boardSize*boardSize ownership (white perspective) or NULL to skip.
// Returns 1 on success, 0 on failure (see kgeError). Async (GPU readback).
int kgeEval(const int* stones, int pla, double komi,
            float* policyOut, float* valueOut, float* ownerOut);

// Evaluate after replaying a move sequence — models captures, ko/superko, and the
// net's recent-move input features (unlike kgeEval, which sees only the final stones).
//   moveLocs:  numMoves ints, each y*size + x, or < 0 for a pass.
//   moveCols:  numMoves ints, the mover's color (1 black / 2 white).
//   toPla:     side to move for the eval (1 black / 2 white).
//   boardOut:  boardSize*boardSize ints written with the post-replay board
//              (0 empty, 1 black, 2 white), or NULL to skip. Lets the UI show captures.
//   policyOut/valueOut/ownerOut: as kgeEval. Returns 1 on success, 0 on failure.
int kgeEvalSeq(const int* moveLocs, const int* moveCols, int numMoves,
               int toPla, double komi, int* boardOut,
               float* policyOut, float* valueOut, float* ownerOut);

// Batched stones-only eval — numPos positions (each a boardSize*boardSize stone-grid
// in stonesBatch, plas[i] the side to move) evaluated in ONE forward pass. Outputs:
// policyOut numPos*(size*size+1) logits, valueOut numPos*5. numPos in [1,16].
int kgeEvalBatch(const int* stonesBatch, const int* plas, int numPos,
                 double komi, float* policyOut, float* valueOut);

// Batched MCTS search from the position after replaying the move sequence (as
// kgeEvalSeq). Runs to maxVisits or maxTimeMs, whichever first.
//   bestMoveOut[1]: chosen move (y*size+x, or -1 pass).
//   winrateOut[1]:  searched win probability for toPla (0..1).
//   pvOut[<=pvCap]: principal variation moves; pvLenOut[1] its length.
//   visitsOut[1]:   total visits performed.
// Returns 1 on success, 0 on failure (see kgeError).
int kgeSearch(const int* moveLocs, const int* moveCols, int numMoves,
              int toPla, double komi, int maxVisits, double maxTimeMs,
              int* bestMoveOut, float* winrateOut,
              int* pvOut, int pvCap, int* pvLenOut, int* visitsOut);

// kgeEvalSeqKata: instant analysis through the SAME NNEvaluator that backs search (one
// net load, shared cache). Returns probabilities: policyOut[hw+1] in [0,1] (illegal=-1,
// pass at hw); valueOut[5] = {pWhiteWin,pWhiteLoss,pNoResult,whiteScoreMean,whiteLead};
// ownerOut[hw] in [-1,1] (pass NULL to skip). Threaded build only.
int kgeEvalSeqKata(const int* moveLocs, const int* moveCols, int numMoves,
                   int toPla, double komi,
                   int* boardOut, float* policyOut, float* valueOut, float* ownerOut);

// kgeSearchKata: KataGo's REAL engine (AsyncBot — tree reuse + ponder) over the WebGPU
// backend, via an NNEvaluator (1 server thread owns the device) + numSearchThreads worker
// threads queueing batched NN requests. winrateOut/scoreOut are side-to-move; reusedOut=1
// if the tree was reused. Requires the threaded build (-pthread). See Path A.
int kgeSearchKata(const int* moveLocs, const int* moveCols, int numMoves,
                  int toPla, double komi, int maxVisits, double maxTimeMs,
                  int numSearchThreads,
                  int* bestMoveOut, float* winrateOut, float* scoreOut,
                  int* pvOut, int pvCap, int* pvLenOut, int* visitsOut, int* reusedOut);

// Async search for LIVE stats (threaded build only). kgeSearchBegin starts a single
// continuous search on the bot's threads (non-blocking, with tree reuse); poll kgePoll
// to stream monotonically-climbing visits/PV/win-rate/score (doneOut=1 when finished).
// Then kgePonderBegin starts a real background ponder (keep polling); kgeStopSearch
// halts before a new request.
int kgeSearchBegin(const int* moveLocs, const int* moveCols, int numMoves,
                   int toPla, double komi, int maxVisits, double maxTimeMs, int numSearchThreads);
int kgePoll(int* bestOut, float* winrateOut, float* scoreOut,
            int* pvOut, int pvCap, int* pvLenOut, int* visitsOut, int* doneOut, int* reusedOut);
int kgePonderBegin(void);
int kgeStopSearch(void);
// Top-N root candidate moves (best-first), each with visits / side-to-move win-rate /
// score lead / policy prior. Live-pollable during search. Returns the count (or -1).
int kgeCandidates(int maxN, int* locsOut, int* visitsOut,
                  float* winrateOut, float* scoreOut, float* priorOut);

const char* kgeError(void);       // last error message ("" if none)
int kgeBoardSize(void);           // configured board size
int kgeModelVersion(void);        // loaded model's version

#ifdef __cplusplus
}
#endif

#endif  // KATAEVAL_H_
