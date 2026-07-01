// Gumbel-AlphaZero root selection (kgeSetGumbel) vs plain PUCT, on the custom batched MCTS
// (kgeSearch), CPU backend. Asserts Gumbel returns a legal, sane move; then measures the
// effect the way the demo's Measure tool does — how fast each converges to the deep-search
// (800-visit PUCT) move from a few low-visit searches. Gumbel should agree with the deep
// answer at least as often as PUCT in the low-visit regime.
//
//   node web/test-gumbel.mjs [model.bin.gz]
import { readFileSync, existsSync } from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const here = new URL('.', import.meta.url).pathname;
const model = process.argv[2] || here + 'demo/model-b10c128.bin.gz';
if (!existsSync(model)) { console.error('missing model', model); process.exit(2); }
const fail = (m) => { console.error('FAIL:', m); process.exit(1); };

const N = 9;
const M = await require(here + 'demo/kataeval.js')();
M.ccall('kgeSetForceCpu', null, ['number'], [1]);
M.FS.writeFile('/m.bin.gz', new Uint8Array(readFileSync(model)));
if (!(await M.ccall('kgeLoad', 'number', ['string', 'number'], ['/m.bin.gz', N], { async: true })))
  fail('kgeLoad: ' + M.ccall('kgeError', 'string', [], []));

const MAXMV = 256, PVCAP = 16, HW = N * N;
const mlPtr = M._malloc(MAXMV*4), mcPtr = M._malloc(MAXMV*4);
const bm = M._malloc(4), wr = M._malloc(4), pv = M._malloc(PVCAP*4), pvl = M._malloc(4), vis = M._malloc(4);
const co = (l) => l < 0 ? 'pass' : `${'ABCDEFGHJKLMNOPQRST'[l % N]}${N - (l/N|0)}`;

async function search(moves, visits, gumbelN) {
  const ml = new Int32Array(MAXMV), mc = new Int32Array(MAXMV);
  for (let i = 0; i < moves.length; i++) { ml[i] = moves[i].loc; mc[i] = moves[i].col; }
  M.HEAP32.set(ml, mlPtr>>2); M.HEAP32.set(mc, mcPtr>>2);
  M.ccall('kgeSetGumbel', null, ['number'], [gumbelN]);
  const ok = await M.ccall('kgeSearch', 'number',
    ['number','number','number','number','number','number','number','number','number','number','number','number','number'],
    [mlPtr, mcPtr, moves.length, 1, 7.5, visits, 60000, bm, wr, pv, PVCAP, pvl, vis], { async: true });
  if (!ok) fail('kgeSearch: ' + M.ccall('kgeError', 'string', [], []));
  return { best: M.HEAP32[bm>>2], wr: M.HEAPF32[wr>>2], visits: M.HEAP32[vis>>2] };
}

// 1) Gumbel returns a legal, on-board move on the empty board.
const g = await search([], 32, 16);
if (!(g.best >= 0 && g.best < HW)) fail('Gumbel best move off board: ' + g.best);
if (!(g.wr > 0.2 && g.wr < 0.8)) fail('Gumbel empty-board winrate implausible: ' + g.wr);
console.log(`  Gumbel(32v): ${co(g.best)}  wr=${(g.wr*100).toFixed(1)}%`);

// 2) Convergence to the deep PUCT answer across a few opening positions.
const deepMoves = new Map();   // position key -> deep best (800v PUCT)
const positions = [[], [{loc:40,col:1}], [{loc:40,col:1},{loc:20,col:2}]];  // empty, tengen, +approach
let gumbelHits = 0, puctHits = 0, total = 0;
for (const pos of positions) {
  const key = pos.map(m => m.loc).join(',');
  const deep = await search(pos, 800, 0);
  deepMoves.set(key, deep.best);
  const lowP = await search(pos, 24, 0);      // low-visit PUCT
  const lowG = await search(pos, 24, 16);     // low-visit Gumbel
  if (lowP.best === deep.best) puctHits++;
  if (lowG.best === deep.best) gumbelHits++;
  total++;
  console.log(`  pos[${key||'empty'}] deep=${co(deep.best)}  puct24=${co(lowP.best)}${lowP.best===deep.best?'✓':' '}  gumbel24=${co(lowG.best)}${lowG.best===deep.best?'✓':' '}`);
}

// Gumbel must not be WORSE than PUCT at agreeing with the deep answer (it's designed to be
// >= in the low-visit regime). This is a smoke-level guarantee, not a strength proof.
if (gumbelHits < puctHits) fail(`Gumbel agreed with deep search on ${gumbelHits}/${total}, PUCT on ${puctHits}/${total} — Gumbel should be >=`);
console.log(`PASS  Gumbel: legal + sane; low-visit deep-agreement Gumbel ${gumbelHits}/${total} >= PUCT ${puctHits}/${total}`);
process.exit(0);
