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
  '_kgeSearchBegin', '_kgePoll', '_kgePonderBegin', '_kgeStopSearch',
  '_kgeCandidates', '_kgeInsights', '_kgeOwnership', '_kgeSetStrength',
  '_kgeEvalSeqKata',
];
const missing = need.filter(s => typeof M[s] !== 'function');
if (missing.length) fail('MT build missing exports: ' + missing.join(', '));

// The insight/ownership readers must be safe to call with no engine loaded (they guard on
// gBot == nullptr and must not crash) — this exercises the null path of the new ABI.
const insPtr = M._malloc(6 * 4), ownPtr = M._malloc(4 * 4);
M.ccall('kgeInsights', 'number', ['number'], [insPtr]);          // returns 0, no crash
M.ccall('kgeOwnership', 'number', ['number', 'number'], [ownPtr, 4]);
M.ccall('kgeSetStrength', null, ['number', 'number'], [80, 0.35]);  // pure setter, always safe

console.log('PASS  kataeval-mt: ' + need.length + ' exports present; insight/ownership/strength ABI callable');
process.exit(0);
