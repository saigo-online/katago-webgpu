// Web Worker hosting the THREADED kataeval (kataeval-mt.js) — KataGo's REAL Search.
//
// Why a worker: kgeSearchKata runs KataGo's Search, which spawns its own pthreads
// (1 NN-server thread that owns the WebGPU device + N search threads) and BLOCKS the
// caller until the search finishes. Blocking is forbidden on the page's main thread
// (Atomics.wait throws there) but fine on a worker — so the engine lives here and the
// UI thread stays responsive. Requires cross-origin isolation (COOP/COEP) for threads.
importScripts('kataeval-mt.js');

// Surface errors that would otherwise be swallowed (pthread pool spawn, rejections).
const diag = (m) => { try { postMessage({ diag: m }); } catch (_) {} };
self.onerror = (e) => diag('worker.onerror: ' + ((e && e.message) || e));
self.addEventListener('unhandledrejection', (e) => diag('unhandledrejection: ' + ((e.reason && (e.reason.stack || e.reason.message)) || e.reason)));

const MAXMV = 1024, PVCAP = 32;
let M = null, N = 19, ready = false;

// Net caching (Cache API), shared with the page — a net downloads once across both
// contexts and survives refreshes. The page may have already cached it on load.
async function cachedNet(url) {
  let cache = null;
  try { cache = await caches.open('saigo-nets-v1'); } catch (_) { cache = null; }
  if (cache) { const hit = await cache.match(url); if (hit) return await hit.arrayBuffer(); }
  const res = await fetch(url);
  if (!res.ok) throw new Error('fetch ' + url + ' -> ' + res.status);
  if (cache) { try { await cache.put(url, res.clone()); } catch (_) {} }
  return await res.arrayBuffer();
}
let mlPtr, mcPtr, bmPtr, wrPtr, pvPtr, pvlPtr, visPtr;  // search buffers
let bPtr, pPtr, vPtr, oPtr;                             // analysis (kgeEvalSeq) buffers

async function init(netFile, boardSize, wasmBinary, jsText, forceCpu) {
  // The module is loaded via importScripts INTO this worker, so emscripten's
  // _scriptName would point at kata-worker.js — it would then spawn pthread workers
  // as new Worker('kata-worker.js') (re-running THIS file) instead of the pthread
  // bootstrap. mainScriptUrlOrBlob tells it the real module URL so pthread + wasm
  // loading resolve correctly.
  // Spawn pthread workers from a blob: URL (built from the page-fetched JS text), not
  // the https URL — a nested new Worker(https-url) fails over a self-signed/overridden
  // cert. The blob is same-origin and in-memory, so no cert is involved.
  const moduleBlob = new Blob([jsText], { type: 'text/javascript' });
  diag('createKata: spawning thread pool…');
  M = await createKata({
    // The page fetched the wasm for us (a worker's fetch() also fails over the cert);
    // use those bytes directly so the worker never fetches the wasm.
    wasmBinary,
    mainScriptUrlOrBlob: moduleBlob,
    locateFile: (p) => new URL(p, self.location.href).href,
  });
  diag('createKata: runtime + thread pool ready');
  N = boardSize;
  if (forceCpu) M.ccall('kgeSetForceCpu', null, ['number'], [1]);  // ?cpu — Eigen fallback
  M.FS.writeFile('/model.bin.gz', new Uint8Array(await cachedNet(netFile)));
  const ok = await M.ccall('kgeLoad', 'number', ['string', 'number'], ['/model.bin.gz', N], { async: true });
  if (!ok) throw new Error('kgeLoad: ' + M.ccall('kgeError', 'string', [], []));
  mlPtr = M._malloc(MAXMV * 4); mcPtr = M._malloc(MAXMV * 4);
  bmPtr = M._malloc(4); wrPtr = M._malloc(4); visPtr = M._malloc(4); pvlPtr = M._malloc(4);
  pvPtr = M._malloc(PVCAP * 4);
  const HW = N * N;
  bPtr = M._malloc(HW * 4); pPtr = M._malloc((HW + 1) * 4); vPtr = M._malloc(5 * 4); oPtr = M._malloc(HW * 4);
  ready = true;
  const backend = M.ccall('kgeBackendIsGpu', 'number', [], []) ? 'WebGPU' : 'CPU (Eigen)';
  return { backend, version: M.ccall('kgeModelVersion', 'number', [], []) };
}

async function search(req) {
  if (!ready) throw new Error('engine not initialized');
  const moves = req.moves || [];
  const n = Math.min(moves.length, MAXMV);
  const ml = new Int32Array(MAXMV), mc = new Int32Array(MAXMV);
  for (let i = 0; i < n; i++) { ml[i] = moves[i].loc; mc[i] = moves[i].col; }
  M.HEAP32.set(ml, mlPtr >> 2); M.HEAP32.set(mc, mcPtr >> 2);
  const t0 = performance.now();
  const ok = await M.ccall('kgeSearchKata', 'number',
    ['number','number','number','number','number','number','number','number','number','number','number','number','number','number'],
    [mlPtr, mcPtr, n, req.toPlay, req.komi ?? 7.5, req.visits | 0, req.ms | 0, req.threads | 0,
     bmPtr, wrPtr, pvPtr, PVCAP, pvlPtr, visPtr], { async: true });
  if (!ok) throw new Error('kgeSearchKata: ' + M.ccall('kgeError', 'string', [], []));
  const plen = M.HEAP32[pvlPtr >> 2];
  return {
    best: M.HEAP32[bmPtr >> 2],
    wr: M.HEAPF32[wrPtr >> 2],
    nv: M.HEAP32[visPtr >> 2],
    pv: [...Array(plen)].map((_, i) => M.HEAP32[(pvPtr >> 2) + i]),
    ms: Math.round(performance.now() - t0),
  };
}

// Instant per-move analysis (raw NN policy/value/ownership) — same NNEvaluator-backed
// net as search, so it's one net load and the analysis warms the cache search reuses.
async function evalPos(req) {
  if (!ready) throw new Error('engine not initialized');
  const moves = req.moves || [];
  const n = Math.min(moves.length, MAXMV);
  const ml = new Int32Array(MAXMV), mc = new Int32Array(MAXMV);
  for (let i = 0; i < n; i++) { ml[i] = moves[i].loc; mc[i] = moves[i].col; }
  M.HEAP32.set(ml, mlPtr >> 2); M.HEAP32.set(mc, mcPtr >> 2);
  const HW = N * N;
  const ok = await M.ccall('kgeEvalSeq', 'number',
    ['number','number','number','number','number','number','number','number','number'],
    [mlPtr, mcPtr, n, req.toPlay, req.komi ?? 7.5, bPtr, pPtr, vPtr, req.owner ? oPtr : 0], { async: true });
  if (!ok) throw new Error('kgeEvalSeq: ' + M.ccall('kgeError', 'string', [], []));
  // Copy HEAP slices into standalone buffers we can transfer back to the page.
  return {
    board: M.HEAP32.slice(bPtr >> 2, (bPtr >> 2) + HW),
    policy: M.HEAPF32.slice(pPtr >> 2, (pPtr >> 2) + HW + 1),
    value: M.HEAPF32.slice(vPtr >> 2, (vPtr >> 2) + 5),
    owner: req.owner ? M.HEAPF32.slice(oPtr >> 2, (oPtr >> 2) + HW) : null,
  };
}

onmessage = async (e) => {
  const { id, type } = e.data;
  try {
    if (type === 'init') postMessage({ id, ok: true, ...(await init(e.data.netFile, e.data.boardSize, e.data.wasmBinary, e.data.jsText, e.data.forceCpu)) });
    else if (type === 'eval') {
      const r = await evalPos(e.data);
      const xfer = [r.board.buffer, r.policy.buffer, r.value.buffer]; if (r.owner) xfer.push(r.owner.buffer);
      postMessage({ id, ok: true, ...r }, xfer);
    }
    else if (type === 'search') postMessage({ id, ok: true, ...(await search(e.data)) });
    else throw new Error('unknown message type: ' + type);
  } catch (err) {
    postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
