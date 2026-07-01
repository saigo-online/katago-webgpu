// Regression test for handicap setup: a leading run of >=2 same-color stones must all
// land on the board. They can't be replayed as MOVES (out-of-turn moves are illegal, so
// only the first would stick) — replayLine places them as setup stones instead. Here we
// replay a 4-stone handicap (all black) on 9x9 via kgeEvalSeq and assert all 4 appear.
//
//   node web/test-handicap.mjs [path/to/model.bin.gz]
import { readFileSync, existsSync } from 'fs';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const here = new URL('.', import.meta.url).pathname;
const kataevalJs = here + 'demo/kataeval.js';
const model = process.argv[2] || here + 'demo/model-b10c128.bin.gz';
if (!existsSync(kataevalJs) || !existsSync(model)) { console.error('run scripts/build-eval.sh first'); process.exit(2); }
const fail = (m) => { console.error('FAIL:', m); process.exit(1); };

const N = 9;
const createKata = require(kataevalJs);
const M = await createKata();
M.ccall('kgeSetForceCpu', null, ['number'], [1]);
M.FS.writeFile('/m.bin.gz', new Uint8Array(readFileSync(model)));
if (!(await M.ccall('kgeLoad', 'number', ['string', 'number'], ['/m.bin.gz', N], { async: true })))
  fail('kgeLoad: ' + M.ccall('kgeError', 'string', [], []));

// 4-stone handicap on the 9x9 star points (matches the demo's handicapStones()): all BLACK.
const m = 2, lo = m, hi = N - 1 - m;                         // 9x9 -> star at 2 and 6
const hc = [[lo,lo],[hi,lo],[lo,hi],[hi,hi]].map(([x,y]) => y*N+x);
const nMoves = hc.length;
const MAXMV = 1024, HW = N*N;
const mlPtr = M._malloc(MAXMV*4), mcPtr = M._malloc(MAXMV*4);
const bPtr = M._malloc(HW*4), pPtr = M._malloc((HW+1)*4), vPtr = M._malloc(5*4);
const ml = new Int32Array(MAXMV), mc = new Int32Array(MAXMV);
for (let i = 0; i < nMoves; i++) { ml[i] = hc[i]; mc[i] = 1; }   // col 1 = black
M.HEAP32.set(ml, mlPtr>>2); M.HEAP32.set(mc, mcPtr>>2);

// White to move after black handicap setup.
if (!(await M.ccall('kgeEvalSeq', 'number',
    ['number','number','number','number','number','number','number','number','number'],
    [mlPtr, mcPtr, nMoves, 2 /*white*/, 0.5, bPtr, pPtr, vPtr, 0], { async: true })))
  fail('kgeEvalSeq: ' + M.ccall('kgeError', 'string', [], []));

const board = M.HEAP32.subarray(bPtr>>2, (bPtr>>2) + HW);
let blacks = 0; for (let i = 0; i < HW; i++) if (board[i] === 1) blacks++;
for (const loc of hc) if (board[loc] !== 1) fail(`handicap stone at ${loc} not on board (got ${board[loc]})`);
if (blacks !== nMoves) fail(`expected ${nMoves} black stones, board has ${blacks}`);

console.log(`PASS  handicap: all ${nMoves} setup stones present on the ${N}x${N} board`);
process.exit(0);
