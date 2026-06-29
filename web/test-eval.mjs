// Smoke test for the WASM eval — runs the exact browser code path under Node,
// with the CPU (Eigen) backend forced so it needs no GPU. Validates that
// kataeval loads a net and produces a sane evaluation.
//
//   node web/test-eval.mjs [path/to/model.bin.gz]
//
// Defaults to web/demo/model-b10c128.bin.gz (bundled by scripts/build-eval.sh).
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

M.ccall('kgeSetForceCpu', null, ['number'], [1]);                 // no GPU under Node
M.FS.writeFile('/m.bin.gz', new Uint8Array(readFileSync(model)));
if (!(await M.ccall('kgeLoad', 'number', ['string', 'number'], ['/m.bin.gz', 19], { async: true })))
  fail('kgeLoad: ' + M.ccall('kgeError', 'string', [], []));

if (M.ccall('kgeBackendIsGpu', 'number', [], [])) fail('expected CPU backend under Node');
const ver = M.ccall('kgeModelVersion', 'number', [], []);
if (!(ver >= 3 && ver <= 18)) fail('implausible modelVersion ' + ver);   // v8 (g170) .. v14 (b18nbt)

const HW = 361;
const sPtr = M._malloc(HW * 4), pPtr = M._malloc((HW + 1) * 4), vPtr = M._malloc(5 * 4);
M.HEAP32.set(new Int32Array(HW), sPtr >> 2);                       // empty board, black to play
if (!(await M.ccall('kgeEval', 'number', ['number','number','number','number','number','number'],
                    [sPtr, 1, 7.5, pPtr, vPtr, 0], { async: true })))
  fail('kgeEval: ' + M.ccall('kgeError', 'string', [], []));

const v = M.HEAPF32.subarray(vPtr >> 2, (vPtr >> 2) + 5);
// value outputs are RAW logits (win, loss, noResult) — softmax to probabilities,
// exactly as the demo does. (policyOut is likewise raw logits.)
if (![v[0], v[1], v[2]].every(Number.isFinite)) fail('non-finite value logits');
const mx = Math.max(v[0], v[1], v[2]);
const e = [v[0], v[1], v[2]].map(x => Math.exp(x - mx));
const z = e[0] + e[1] + e[2];
const winProb = e[0] / z;
if (!(winProb > 0.2 && winProb < 0.8)) fail(`empty-board winrate implausible: ${winProb}`);

const p = M.HEAPF32.subarray(pPtr >> 2, (pPtr >> 2) + HW + 1);
let bi = -1, bm = -1e30;
for (let i = 0; i < HW; i++) if (Number.isFinite(p[i]) && p[i] > bm) { bm = p[i]; bi = i; }
if (bi < 0) fail('no finite policy logits');

const coord = `${'ABCDEFGHJKLMNOPQRST'[bi % 19]}${19 - (bi / 19 | 0)}`;
console.log(`PASS  v${ver} CPU  winrate=${(winProb * 100).toFixed(1)}%  topMove=${coord}`);
