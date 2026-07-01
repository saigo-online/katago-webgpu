// Policy optimism blend (kgeSetPolicyOptimism) end-to-end on a modelVersion>=12 net (b18,
// npc=2 — it has the optimism policy channel). Asserts that turning optimism on actually
// CHANGES the policy (p -> p + (pOpt-p)*optimism) and that the result stays sane. This runs
// the CPU (Eigen) path, whose blend the WebGPU backend mirrors line-for-line
// (webgpubackend.cpp), so it exercises the full nninputparams -> blend -> nneval wiring.
//
//   node web/test-optimism.mjs [b18.bin.gz]
import { readFileSync, existsSync } from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const here = new URL('.', import.meta.url).pathname;
const model = process.argv[2] || here + 'demo/model-b18c384nbt.bin.gz';
if (!existsSync(model)) { console.error('need a v>=12 net (npc>=2):', model); process.exit(2); }
const fail = (m) => { console.error('FAIL:', m); process.exit(1); };

const N = 19, HW = N * N;
const M = await require(here + 'demo/kataeval.js')();
M.ccall('kgeSetForceCpu', null, ['number'], [1]);
M.FS.writeFile('/m.bin.gz', new Uint8Array(readFileSync(model)));
if (!(await M.ccall('kgeLoad', 'number', ['string', 'number'], ['/m.bin.gz', N], { async: true })))
  fail('kgeLoad: ' + M.ccall('kgeError', 'string', [], []));
const ver = M.ccall('kgeModelVersion', 'number', [], []);
if (ver < 12) fail(`net is modelVersion ${ver} (<12) — no optimism channel to blend`);

const sPtr = M._malloc(HW*4), pPtr = M._malloc((HW+1)*4), vPtr = M._malloc(5*4);
// A non-empty position so channel 0 and the optimism channel actually diverge.
const board = new Int32Array(HW); board[16*N+3] = 1; board[3*N+15] = 2; board[15*N+15] = 1;
M.HEAP32.set(board, sPtr>>2);
async function policy(optimism) {
  M.ccall('kgeSetPolicyOptimism', null, ['number'], [optimism]);
  if (!(await M.ccall('kgeEval', 'number', ['number','number','number','number','number','number'],
                      [sPtr, 2, 7.5, pPtr, vPtr, 0], { async: true })))
    fail('kgeEval: ' + M.ccall('kgeError', 'string', [], []));
  return M.HEAPF32.slice(pPtr>>2, (pPtr>>2) + HW + 1);
}

const p0 = await policy(0.0);   // plain policy (channel 0)
const p1 = await policy(1.0);   // fully optimistic policy (channel 1)

if (!p0.every(Number.isFinite) || !p1.every(Number.isFinite)) fail('non-finite policy logits');
// The blend must actually change the policy (else the optimism channel is being ignored).
let maxDiff = 0; for (let i = 0; i <= HW; i++) maxDiff = Math.max(maxDiff, Math.abs(p0[i] - p1[i]));
if (maxDiff < 1e-3) fail('optimism=1 produced the same policy as optimism=0 — blend not applied');
// Both argmaxes must be legal (on an empty point).
const argmax = (p) => { let bi = -1, bm = -1e30; for (let i = 0; i < HW; i++) if (p[i] > bm && board[i] === 0) { bm = p[i]; bi = i; } return bi; };
const a0 = argmax(p0), a1 = argmax(p1);
if (a0 < 0 || a1 < 0) fail('no legal argmax');
const co = (l) => `${'ABCDEFGHJKLMNOPQRST'[l%N]}${N-(l/N|0)}`;
console.log(`PASS  optimism blend (v${ver}, npc>=2): opt0 top=${co(a0)}  opt1 top=${co(a1)}  maxΔlogit=${maxDiff.toFixed(3)}`);
process.exit(0);
