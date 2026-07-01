// Regression test for SGF setup-stone parsing (handicap). A real handicap SGF encodes the
// handicap stones as AB[..][..] setup properties (not ;B[] moves); if the parser drops them
// the review position desyncs from the engine. This mirrors analyze.html's parseSGF setup
// extraction and asserts a 4-stone AB handicap + first white move are recovered in order.
//
//   node web/test-sgf-parse.mjs
const fail = (m) => { console.error('FAIL:', m); process.exit(1); };

// --- extraction identical to analyze.html parseSGF (keep in sync) ---
function parseSGF(text) {
  const g = { size: 19, setup: [], mv: [] };
  const sz = text.match(/\bSZ\[(\d+)\]/); if (sz) g.size = +sz[1];
  for (const m of text.matchAll(/\b(AB|AW)((?:\s*\[[a-z]{2}\])+)/g)) {
    const col = m[1] === 'AB' ? 1 : 2;
    for (const c of m[2].matchAll(/\[([a-z]{2})\]/g)) g.setup.push({ col, co: c[1] });
  }
  g.setup.sort((a, b) => a.col - b.col);
  for (const m of text.matchAll(/;\s*([BW])\[([a-z]{0,2})\]/g)) g.mv.push({ col: m[1] === 'B' ? 1 : 2, co: m[2] });
  return g;
}

// 4-stone handicap on 19x19 (star points), then White C4, Black passes.
const sgf = '(;GM[1]FF[4]SZ[19]HA[4]KM[0.5]AB[dp][pd][dd][pp];W[dj];B[])';
const g = parseSGF(sgf);
if (g.size !== 19) fail('size ' + g.size);
if (g.setup.length !== 4) fail('expected 4 AB setup stones, got ' + g.setup.length);
if (!g.setup.every(s => s.col === 1)) fail('handicap setup stones must all be black');
const co = g.setup.map(s => s.co).sort().join(',');
if (co !== 'dd,dp,pd,pp') fail('setup coords wrong: ' + co);
if (g.mv.length !== 2) fail('expected 2 moves (W + pass), got ' + g.mv.length);
if (g.mv[0].col !== 2 || g.mv[0].co !== 'dj') fail('first move should be W dj, got ' + JSON.stringify(g.mv[0]));
if (g.mv[1].co !== '') fail('second move should be a pass (empty co)');

// AW is parsed too (mixed setup), blacks sorted first.
const g2 = parseSGF('(;SZ[9]AB[cc][gg]AW[ee];B[dd])');
if (g2.setup.length !== 3) fail('mixed AB/AW: expected 3 setup, got ' + g2.setup.length);
if (g2.setup[0].col !== 1 || g2.setup[2].col !== 2) fail('setup should be sorted blacks-first');

console.log('PASS  SGF parse: AB/AW setup stones recovered (4-stone handicap + mixed)');
