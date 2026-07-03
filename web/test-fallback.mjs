// Tests the AUTOMATIC WebGPU -> Eigen-CPU fallback: unlike test-eval.mjs this
// does NOT call kgeSetForceCpu. Under Node there is no WebGPU adapter, so the
// backend's adapter probe in createComputeContext must throw and the kataeval
// dispatcher must fall back to the CPU backend transparently — kgeLoad succeeds,
// kgeBackendIsGpu() reports 0, and a real eval comes back sane. This is the
// "one WASM binary just falls back" promise, exercised end to end.
//
//   node web/test-fallback.mjs [path/to/model.bin.gz]
//
// Requires a prior `scripts/build-eval.sh` (kataeval.js/.wasm are build outputs).
import { readFileSync, existsSync } from 'fs';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const here = new URL('.', import.meta.url).pathname;
const kataevalJs = here + 'demo/kataeval.js';
const model = process.argv[2] || here + 'demo/model-b10c128.bin.gz';

if (!existsSync(kataevalJs)) { console.error('missing', kataevalJs, '- run scripts/build-eval.sh first'); process.exit(2); }
if (!existsSync(model)) { console.error('missing model', model, '- run scripts/build-eval.sh first'); process.exit(2); }

const fail = (m) => { console.error('FAIL:', m); process.exit(1); };
const createKata = require(kataevalJs);
const M = await createKata();

// No kgeSetForceCpu here — the fallback must engage on its own.
M.FS.writeFile('/m.bin.gz', new Uint8Array(readFileSync(model)));
if (!(await M.ccall('kgeLoad', 'number', ['string', 'number'], ['/m.bin.gz', 19], { async: true })))
  fail('kgeLoad (auto-fallback should have engaged): ' + M.ccall('kgeError', 'string', [], []));
if (M.ccall('kgeBackendIsGpu', 'number', [], [])) fail('expected automatic CPU fallback under Node');

const HW = 361;
const sPtr = M._malloc(HW * 4), pPtr = M._malloc((HW + 1) * 4), vPtr = M._malloc(5 * 4);
M.HEAP32.set(new Int32Array(HW), sPtr >> 2);                       // empty board, black to play
if (!(await M.ccall('kgeEval', 'number', ['number','number','number','number','number','number'],
                    [sPtr, 1, 7.5, pPtr, vPtr, 0], { async: true })))
  fail('kgeEval: ' + M.ccall('kgeError', 'string', [], []));

// value outputs are RAW logits (win, loss, noResult) — softmax to probabilities.
const v = M.HEAPF32.subarray(vPtr >> 2, (vPtr >> 2) + 5);
if (![v[0], v[1], v[2]].every(Number.isFinite)) fail('non-finite value logits');
const mx = Math.max(v[0], v[1], v[2]);
const e = [v[0], v[1], v[2]].map(x => Math.exp(x - mx));
const winProb = e[0] / (e[0] + e[1] + e[2]);
if (!(winProb > 0.2 && winProb < 0.8)) fail(`empty-board winrate implausible: ${winProb}`);

console.log(`PASS  auto-fallback to CPU  winrate=${(winProb * 100).toFixed(1)}%`);
