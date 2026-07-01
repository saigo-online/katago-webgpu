// Hermetic smoke test for the THREADED search build (kataeval-mt): it loads the module
// and asserts the full async-search + insight + strength ABI is exported and callable.
//
//   node web/test-search-mt.mjs
//
// NOTE: the *live* threaded search (kgeSearchBegin + the callbackLoopThread that streams
// writeSnapshot) can only be exercised in a browser worker — under Node, emscripten's
// ASYNCIFY unwind for the pthread/Worker spawn in kgeSearchBegin does not rewind, so a
// real search throws "unwind" (a Node+ASYNCIFY+pthreads limitation, not a product bug;
// the browser reaches and runs the search normally). So here we validate what Node CAN:
// the module builds, loads, and exposes the exact ABI the worker calls. The search
// stability fix (exception firewall in writeSnapshot) is verified in-browser.
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const here = new URL('.', import.meta.url).pathname;
const kataevalJs = here + 'demo/kataeval-mt.js';

const fail = (m) => { console.error('FAIL:', m); process.exit(1); };
const createKata = require(kataevalJs);
const M = await createKata();

// Every symbol the worker (kata-worker.js) ccalls must be present in the MT binary.
const need = [
  '_kgeLoad', '_kgeSetForceCpu', '_kgeSetFp16', '_kgeBackendIsGpu', '_kgeModelVersion',
  '_kgeSearchBegin', '_kgePollAll', '_kgePonderBegin', '_kgeStopSearch',
  '_kgeSetStrength', '_kgeSetPolicyOptimism', '_kgeEvalSeqKata',
];
const missing = need.filter(s => typeof M[s] !== 'function');
if (missing.length) fail('MT build missing exports: ' + missing.join(', '));

// kgePollAll must be safe with no engine loaded (guards on gBot == nullptr → returns 0, no
// crash). Give it real buffers so a stray write can't corrupt — exercises the null path.
const i7 = M._malloc(7 * 4), f8 = M._malloc(8 * 4), pv = M._malloc(32 * 4), pvv = M._malloc(32 * 4);
const cl = M._malloc(8 * 4), cv = M._malloc(8 * 4), cw = M._malloc(8 * 4), cs = M._malloc(8 * 4);
const cp = M._malloc(8 * 4), cb = M._malloc(8 * 4), cr = M._malloc(8 * 4), cd = M._malloc(8 * 4), ow = M._malloc(4 * 4);
const rc = M.ccall('kgePollAll', 'number',
  ['number','number','number','number','number','number','number','number','number','number','number','number','number','number','number','number'],
  [i7, f8, pv, pvv, 32, cl, cv, cw, cs, cp, cb, cr, cd, 8, ow, 4]);
if (rc !== 0) fail('kgePollAll with no engine should return 0, got ' + rc);
M.ccall('kgeSetStrength', null, ['number', 'number', 'number'], [80, 0.35, 1.4]);  // pure setter, always safe

console.log('PASS  kataeval-mt: ' + need.length + ' exports present; kgePollAll null-path + strength ABI callable');
process.exit(0);
