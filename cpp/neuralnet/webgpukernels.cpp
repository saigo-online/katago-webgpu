#include "../neuralnet/webgpukernels.h"

// Native WebGPU (Dawn) backend kernel library. See WEBGPU_STATUS.md.
//
// Direct (no Winograd/tensor-core) kernels, easy to validate first.
//
// fp16: storage scalar is the alias `STO`, which the backend prepends as either
//   `alias STO = f32;`              (fp32 path), or
//   `enable f16;  alias STO = f16;` (fp16 storage path).
// Every storage LOAD is wrapped `f32(...)` and every STORE wrapped `STO(...)`,
// so the *same* source compiles both ways; math always accumulates in f32 (fp16
// storage + fp32 compute — halves bandwidth/VRAM, keeps accumulation accuracy).
//
// Conventions:
//   - Spatial tensors are NCHW packed as data[((n*C + c)*H + y)*W + x].
//   - "mask" is NHW packed as mask[(n*H + y)*W + x], 1.0 on-board else 0.0.
//   - Activation codes match ACTIVATION_* in activations.h:
//       0 = identity, 1 = relu, 2 = mish, 3 = silu, 12 = mish_scale8.

namespace KataGoWebGPU {

const char* const WGSL_KERNELS = R"WGSL(
// ===========================================================================
// Shared helpers
// ===========================================================================

fn activate(x: f32, kind: u32) -> f32 {
  // ACTIVATION_*: 0=identity, 1=relu, 2=mish, 3=silu, 12=mish_scale8
  if (kind == 1u) {
    return max(x, 0.0);
  } else if (kind == 2u) {
    // mish(x) = x * tanh(softplus(x)); numerically-stable softplus.
    let sp = select(log(1.0 + exp(x)), x + log(1.0 + exp(-x)), x > 20.0);
    return x * tanh(sp);
  } else if (kind == 3u) {
    // silu(x) = x * sigmoid(x); stable two-sided sigmoid (avoids exp overflow).
    let s = select(1.0 / (1.0 + exp(-x)), exp(x) / (1.0 + exp(x)), x < 0.0);
    return x * s;
  } else if (kind == 12u) {
    // mish_scale8(x) = x * tanh(softplus(8x)) = mish(8x)/8 (fp16-stability rescale).
    let z = 8.0 * x;
    let sp = select(log(1.0 + exp(z)), z + log(1.0 + exp(-z)), z > 20.0);
    return x * tanh(sp);
  }
  return x;
}

// ===========================================================================
// scaleBiasMaskAct: out[n,c,xy] = act(in[n,c,xy] * scale[c] + bias[c]) * mask[n,xy]
// Mirrors the fused BN(merged)+activation+mask used throughout the trunk.
// ===========================================================================

struct SBMAParams {
  n : u32,   // batch size
  c : u32,   // channels
  hw : u32,  // H*W
  act : u32, // activation kind
};

@group(0) @binding(0) var<uniform> sbma : SBMAParams;
@group(0) @binding(1) var<storage, read>        sbmaIn    : array<STO>;
@group(0) @binding(2) var<storage, read>        sbmaScale : array<STO>;
@group(0) @binding(3) var<storage, read>        sbmaBias  : array<STO>;
@group(0) @binding(4) var<storage, read>        sbmaMask  : array<STO>;
@group(0) @binding(5) var<storage, read_write>  sbmaOut   : array<STO>;

@compute @workgroup_size(64)
fn scaleBiasMaskAct(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let total = sbma.n * sbma.c * sbma.hw;
  if (idx >= total) { return; }
  let xy = idx % sbma.hw;
  let c  = (idx / sbma.hw) % sbma.c;
  let n  = idx / (sbma.hw * sbma.c);
  let m  = f32(sbmaMask[n * sbma.hw + xy]);
  let v  = activate(f32(sbmaIn[idx]) * f32(sbmaScale[c]) + f32(sbmaBias[c]), sbma.act);
  sbmaOut[idx] = STO(v * m);
}

// ===========================================================================
// addInPlace: dst[i] += src[i]   (residual connection)
// ===========================================================================

struct AddParams { total : u32, pad0 : u32, pad1 : u32, pad2 : u32, };

@group(0) @binding(0) var<uniform> addP : AddParams;
@group(0) @binding(1) var<storage, read>       addSrc : array<STO>;
@group(0) @binding(2) var<storage, read_write> addDst : array<STO>;

@compute @workgroup_size(64)
fn addInPlace(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= addP.total) { return; }
  addDst[i] = STO(f32(addDst[i]) + f32(addSrc[i]));
}

// ===========================================================================
// matMulBiasAct: C[m,o] = act( sum_k A[m,k] * B[k,o] + bias[o] )
// A is row-major [M,K]; B is row-major [K,O] (matches MatMulLayerDesc inC x outC).
// bias may be absent (hasBias == 0). Used for FC layers and gpool->bias mults.
// ===========================================================================

struct MMParams {
  m : u32,
  k : u32,
  o : u32,
  act : u32,
  hasBias : u32,
  pad0 : u32, pad1 : u32, pad2 : u32,
};

@group(0) @binding(0) var<uniform> mm : MMParams;
@group(0) @binding(1) var<storage, read>       mmA    : array<STO>;
@group(0) @binding(2) var<storage, read>       mmB    : array<STO>;
@group(0) @binding(3) var<storage, read>       mmBias : array<STO>;
@group(0) @binding(4) var<storage, read_write> mmC    : array<STO>;

@compute @workgroup_size(8, 8)
fn matMulBiasAct(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x; // [0,M)
  let col = gid.y; // [0,O)
  if (row >= mm.m || col >= mm.o) { return; }
  var acc : f32 = 0.0;
  for (var kk : u32 = 0u; kk < mm.k; kk = kk + 1u) {
    acc = acc + f32(mmA[row * mm.k + kk]) * f32(mmB[kk * mm.o + col]);
  }
  if (mm.hasBias != 0u) { acc = acc + f32(mmBias[col]); }
  mmC[row * mm.o + col] = STO(activate(acc, mm.act));
}

// ===========================================================================
// globalPoolMeanMax: per (n,c) reduce the masked board to 3 stats KataGo uses:
//   out[n, c*3 + 0] = mean over valid area
//   out[n, c*3 + 1] = mean * (sqrt(validArea) - 14) * 0.1   (size-scaled mean)
//   out[n, c*3 + 2] = max over valid area
// One workgroup per (n,c); reduces hw elements with shared memory.
// ===========================================================================

struct GPParams { n : u32, c : u32, hw : u32, pad : u32, };

@group(0) @binding(0) var<uniform> gp : GPParams;
@group(0) @binding(1) var<storage, read>       gpIn   : array<STO>;
@group(0) @binding(2) var<storage, read>       gpMask : array<STO>;
@group(0) @binding(3) var<storage, read_write> gpOut  : array<STO>;

const GP_WG : u32 = 64u;
var<workgroup> gpSum : array<f32, 64>;
var<workgroup> gpMax : array<f32, 64>;
var<workgroup> gpCnt : array<f32, 64>;

@compute @workgroup_size(64)
fn globalPoolMeanMax(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
) {
  let nc = wid.x;            // which (n,c) this workgroup handles
  let c  = nc % gp.c;
  let n  = nc / gp.c;
  let base = (n * gp.c + c) * gp.hw;
  let maskBase = n * gp.hw;

  var localSum : f32 = 0.0;
  var localMax : f32 = -3.4e38;
  var localCnt : f32 = 0.0;
  var i : u32 = lid.x;
  loop {
    if (i >= gp.hw) { break; }
    let m = f32(gpMask[maskBase + i]);
    let v = f32(gpIn[base + i]);
    localSum = localSum + v * m;
    localCnt = localCnt + m;
    if (m > 0.5) { localMax = max(localMax, v); }
    i = i + GP_WG;
  }
  gpSum[lid.x] = localSum;
  gpMax[lid.x] = localMax;
  gpCnt[lid.x] = localCnt;
  workgroupBarrier();

  // tree reduction
  var stride : u32 = GP_WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      gpSum[lid.x] = gpSum[lid.x] + gpSum[lid.x + stride];
      gpCnt[lid.x] = gpCnt[lid.x] + gpCnt[lid.x + stride];
      gpMax[lid.x] = max(gpMax[lid.x], gpMax[lid.x + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    let area = max(gpCnt[0], 1.0);
    let mean = gpSum[0] / area;
    let scaled = mean * (sqrt(area) - 14.0) * 0.1;
    // Stat-major, channel-minor layout to match poolRowsGPool / the gpool->bias
    // matmul's K dimension: [mean(0..C-1), scaledMean(C..2C-1), max(2C..3C-1)] per n.
    let outBase = n * 3u * gp.c;
    gpOut[outBase + c]            = STO(mean);
    gpOut[outBase + gp.c + c]     = STO(scaled);
    gpOut[outBase + 2u * gp.c + c] = STO(gpMax[0]);
  }
}

// ===========================================================================
// globalPoolValueHead: value-head pooling. Per (n,c), 3 stats:
//   out[n, c]      = mean
//   out[n, c+C]    = mean * (sqrt(area) - 14) * 0.1
//   out[n, c+2C]   = mean * ((sqrt(area)-14)^2 * 0.01 - 0.1)
// Mirrors poolRowsValueHead (no max; quadratic 3rd stat). Stat-major layout.
// One workgroup per (n,c).
// ===========================================================================

struct VHParams { n : u32, c : u32, hw : u32, pad : u32, };

@group(0) @binding(0) var<uniform> vh : VHParams;
@group(0) @binding(1) var<storage, read>       vhIn   : array<STO>;
@group(0) @binding(2) var<storage, read>       vhMask : array<STO>;
@group(0) @binding(3) var<storage, read_write> vhOut  : array<STO>;

var<workgroup> vhSum : array<f32, 64>;
var<workgroup> vhCnt : array<f32, 64>;

@compute @workgroup_size(64)
fn globalPoolValueHead(
  @builtin(workgroup_id) wid : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
) {
  let nc = wid.x;
  let c  = nc % vh.c;
  let n  = nc / vh.c;
  let base = (n * vh.c + c) * vh.hw;
  let maskBase = n * vh.hw;

  var localSum : f32 = 0.0;
  var localCnt : f32 = 0.0;
  var i : u32 = lid.x;
  loop {
    if (i >= vh.hw) { break; }
    let m = f32(vhMask[maskBase + i]);
    localSum = localSum + f32(vhIn[base + i]) * m;
    localCnt = localCnt + m;
    i = i + 64u;
  }
  vhSum[lid.x] = localSum;
  vhCnt[lid.x] = localCnt;
  workgroupBarrier();

  var stride : u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      vhSum[lid.x] = vhSum[lid.x] + vhSum[lid.x + stride];
      vhCnt[lid.x] = vhCnt[lid.x] + vhCnt[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid.x == 0u) {
    let area = max(vhCnt[0], 1.0);
    let mean = vhSum[0] / area;
    let d = sqrt(area) - 14.0;
    let outBase = n * 3u * vh.c;
    vhOut[outBase + c]            = STO(mean);
    vhOut[outBase + vh.c + c]     = STO(mean * d * 0.1);
    vhOut[outBase + 2u * vh.c + c] = STO(mean * (d * d * 0.01 - 0.1));
  }
}

// ===========================================================================
// addChannelBias: data[n,c,xy] += bias[n,c]   (broadcast over space)
// Mirrors addNCBiasInplace; used after the gpool->bias matmul.
// ===========================================================================

struct AddBiasParams { n : u32, c : u32, hw : u32, pad : u32, };

@group(0) @binding(0) var<uniform> abP : AddBiasParams;
@group(0) @binding(1) var<storage, read>       abBias : array<STO>;
@group(0) @binding(2) var<storage, read_write> abData : array<STO>;

@compute @workgroup_size(64)
fn addChannelBias(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let total = abP.n * abP.c * abP.hw;
  if (idx >= total) { return; }
  let c = (idx / abP.hw) % abP.c;
  let n = idx / (abP.hw * abP.c);
  abData[idx] = STO(f32(abData[idx]) + f32(abBias[n * abP.c + c]));
}

// ===========================================================================
// conv2dNCHW: direct convolution, arbitrary square-ish filter + dilation.
// weights layout matches ConvLayerDesc: outC x inC x H x W (W has least stride).
// No bias (KataGo folds bias into the following BN). SAME padding (zero).
// Slow but correct; Winograd/1x1 fast paths are a later milestone.
// ===========================================================================

struct ConvParams {
  n : u32, inC : u32, outC : u32,
  h : u32, w : u32,
  fy : u32, fx : u32,
  dilY : u32, dilX : u32,
  pad0 : u32, pad1 : u32, pad2 : u32,
};

@group(0) @binding(0) var<uniform> cv : ConvParams;
@group(0) @binding(1) var<storage, read>       cvIn  : array<STO>;
@group(0) @binding(2) var<storage, read>       cvW   : array<STO>;
@group(0) @binding(3) var<storage, read_write> cvOut : array<STO>;

@compute @workgroup_size(64)
fn conv2dNCHW(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let hw = cv.h * cv.w;
  let total = cv.n * cv.outC * hw;
  if (idx >= total) { return; }

  let x  = idx % cv.w;
  let y  = (idx / cv.w) % cv.h;
  let oc = (idx / hw) % cv.outC;
  let n  = idx / (hw * cv.outC);

  // top-left of the (dilated) filter window for SAME padding
  let halfY = (cv.fy / 2u) * cv.dilY;
  let halfX = (cv.fx / 2u) * cv.dilX;

  var acc : f32 = 0.0;
  for (var ic : u32 = 0u; ic < cv.inC; ic = ic + 1u) {
    let inChanBase = (n * cv.inC + ic) * hw;
    let wChanBase = (oc * cv.inC + ic) * cv.fy * cv.fx;
    for (var ky : u32 = 0u; ky < cv.fy; ky = ky + 1u) {
      let iy = i32(y) + i32(ky * cv.dilY) - i32(halfY);
      if (iy < 0 || iy >= i32(cv.h)) { continue; }
      for (var kx : u32 = 0u; kx < cv.fx; kx = kx + 1u) {
        let ix = i32(x) + i32(kx * cv.dilX) - i32(halfX);
        if (ix < 0 || ix >= i32(cv.w)) { continue; }
        let inVal = f32(cvIn[inChanBase + u32(iy) * cv.w + u32(ix)]);
        let wVal  = f32(cvW[wChanBase + ky * cv.fx + kx]);
        acc = acc + inVal * wVal;
      }
    }
  }
  cvOut[idx] = STO(acc);
}

// ===========================================================================
// conv1x1NCHW: specialized 1x1 convolution (a per-pixel channel GEMM).
//   out[n,oc,xy] = sum_ic in[n,ic,xy] * w[oc*inC + ic]
// No spatial window / padding / bounds checks — just the channel reduction.
// weights layout matches ConvLayerDesc for a 1x1 filter: outC x inC.
// ===========================================================================

struct Conv1x1Params { n : u32, inC : u32, outC : u32, hw : u32, };

@group(0) @binding(0) var<uniform> c1 : Conv1x1Params;
@group(0) @binding(1) var<storage, read>       c1In  : array<STO>;
@group(0) @binding(2) var<storage, read>       c1W   : array<STO>;
@group(0) @binding(3) var<storage, read_write> c1Out : array<STO>;

@compute @workgroup_size(64)
fn conv1x1NCHW(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let total = c1.n * c1.outC * c1.hw;
  if (idx >= total) { return; }
  let xy = idx % c1.hw;
  let oc = (idx / c1.hw) % c1.outC;
  let n  = idx / (c1.hw * c1.outC);

  let inBase = n * c1.inC * c1.hw + xy;     // in[n, 0, xy]
  let wBase  = oc * c1.inC;                  // w[oc, 0]
  var acc : f32 = 0.0;
  for (var ic : u32 = 0u; ic < c1.inC; ic = ic + 1u) {
    acc = acc + f32(c1In[inBase + ic * c1.hw]) * f32(c1W[wBase + ic]);
  }
  c1Out[idx] = STO(acc);
}

// ===========================================================================
// tiledGemm: shared-memory tiled version of the per-position channel GEMM
//   out[n,o,xy] = sum_k W * in[n,k,xy]   (1x1 conv / transformer projection)
// Computes one TILE×TILE block of out[outC, hw] per workgroup (batch n = wid.z),
// staging A (weights) and B (input) tiles into workgroup memory. wInMajor picks
// the weight layout: 1 = MatMulLayerDesc W[k*outC+o]; 0 = ConvLayerDesc W[o*inC+k].
// ===========================================================================

struct TGParams { n : u32, inC : u32, outC : u32, hw : u32, wInMajor : u32, pad0 : u32, pad1 : u32, pad2 : u32, };

@group(0) @binding(0) var<uniform> tg : TGParams;
@group(0) @binding(1) var<storage, read>       tgIn  : array<STO>;
@group(0) @binding(2) var<storage, read>       tgW   : array<STO>;
@group(0) @binding(3) var<storage, read_write> tgOut : array<STO>;

const TG_TILE : u32 = 16u;
var<workgroup> tgAs : array<f32, 256>;   // [TG_TILE][TG_TILE]  (o, k)
var<workgroup> tgBs : array<f32, 256>;   // [TG_TILE][TG_TILE]  (k, xy)

@compute @workgroup_size(16, 16)
fn tiledGemm(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let n  = wid.z;
  let o  = wid.x * TG_TILE + lid.x;      // output channel (row)
  let xy = wid.y * TG_TILE + lid.y;      // spatial position (col)
  var acc : f32 = 0.0;
  let nTiles = (tg.inC + TG_TILE - 1u) / TG_TILE;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * TG_TILE + lid.y;        // A's reduction index varies with lid.y
    var av : f32 = 0.0;
    if (o < tg.outC && kA < tg.inC) {
      if (tg.wInMajor != 0u) { av = f32(tgW[kA * tg.outC + o]); } else { av = f32(tgW[o * tg.inC + kA]); }
    }
    tgAs[lid.x * TG_TILE + lid.y] = av;
    let kB = t * TG_TILE + lid.x;        // B's reduction index varies with lid.x
    var bv : f32 = 0.0;
    if (kB < tg.inC && xy < tg.hw) { bv = f32(tgIn[(n * tg.inC + kB) * tg.hw + xy]); }
    tgBs[lid.x * TG_TILE + lid.y] = bv;
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < TG_TILE; kk = kk + 1u) {
      acc = acc + tgAs[lid.x * TG_TILE + kk] * tgBs[kk * TG_TILE + lid.y];
    }
    workgroupBarrier();
  }
  if (o < tg.outC && xy < tg.hw) { tgOut[(n * tg.outC + o) * tg.hw + xy] = STO(acc); }
}

// ===========================================================================
// tiledGemmRT: register-tiled GEMM — 8x8 workgroup, each thread computes a 2x2
// output micro-tile (16x16 tile/workgroup, cooperative 4-per-thread tile loads).
// More arithmetic per shared-memory load than tiledGemm; same math (byte-exact).
// ===========================================================================

@group(0) @binding(0) var<uniform> rt : TGParams;
@group(0) @binding(1) var<storage, read>       rtIn  : array<STO>;
@group(0) @binding(2) var<storage, read>       rtW   : array<STO>;
@group(0) @binding(3) var<storage, read_write> rtOut : array<STO>;

var<workgroup> rtAs : array<f32, 256>;   // [16 o][16 k]
var<workgroup> rtBs : array<f32, 256>;   // [16 k][16 xy]

@compute @workgroup_size(8, 8)
fn tiledGemmRT(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let n = wid.z;
  let oTile = wid.x * 16u;
  let xyTile = wid.y * 16u;
  let tid = lid.x * 8u + lid.y;            // 0..63
  var acc : array<f32, 4>;                 // [di*2 + dj]
  acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0; acc[3] = 0.0;
  let nTiles = (rt.inC + 15u) / 16u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    // cooperative load: 64 threads x 4 = 256 elements per tile
    for (var r : u32 = 0u; r < 4u; r = r + 1u) {
      let e = tid + r * 64u;               // 0..255
      let er = e / 16u; let ec = e % 16u;
      // A[oTile+er, t*16+ec]
      let ao = oTile + er; let ak = t * 16u + ec;
      var av : f32 = 0.0;
      if (ao < rt.outC && ak < rt.inC) { if (rt.wInMajor != 0u) { av = f32(rtW[ak * rt.outC + ao]); } else { av = f32(rtW[ao * rt.inC + ak]); } }
      rtAs[er * 16u + ec] = av;
      // B[t*16+er, xyTile+ec]
      let bk = t * 16u + er; let bxy = xyTile + ec;
      var bv : f32 = 0.0;
      if (bk < rt.inC && bxy < rt.hw) { bv = f32(rtIn[(n * rt.inC + bk) * rt.hw + bxy]); }
      rtBs[er * 16u + ec] = bv;
    }
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < 16u; kk = kk + 1u) {
      let a0 = rtAs[(lid.x * 2u) * 16u + kk];
      let a1 = rtAs[(lid.x * 2u + 1u) * 16u + kk];
      let b0 = rtBs[kk * 16u + lid.y * 2u];
      let b1 = rtBs[kk * 16u + lid.y * 2u + 1u];
      acc[0] = acc[0] + a0 * b0; acc[1] = acc[1] + a0 * b1;
      acc[2] = acc[2] + a1 * b0; acc[3] = acc[3] + a1 * b1;
    }
    workgroupBarrier();
  }
  for (var di : u32 = 0u; di < 2u; di = di + 1u) {
    for (var dj : u32 = 0u; dj < 2u; dj = dj + 1u) {
      let o = oTile + lid.x * 2u + di; let xy = xyTile + lid.y * 2u + dj;
      if (o < rt.outC && xy < rt.hw) { rtOut[(n * rt.outC + o) * rt.hw + xy] = STO(acc[di * 2u + dj]); }
    }
  }
}

// ===========================================================================
// tiledGemmInBnAct: tiledGemm fused with the PRE-activation it consumes. KataGo
// blocks are norm->act->conv, so we fold scaleBiasMaskAct into the B-tile load:
//   B[k,xy] = act(in[n,k,xy]*scale[k] + bias[k]) * mask[n,xy]
// equals conv1x1(bnAct(in)) in one dispatch, no intermediate tensor. (out-major
// conv weights only; used for the nbt pre/post 1x1 projections + heads.)
// ===========================================================================

struct TGBParams { n : u32, inC : u32, outC : u32, hw : u32, act : u32, pad0 : u32, pad1 : u32, pad2 : u32, };

@group(0) @binding(0) var<uniform> tgb : TGBParams;
@group(0) @binding(1) var<storage, read>       tgbIn    : array<STO>;
@group(0) @binding(2) var<storage, read>       tgbW     : array<STO>;
@group(0) @binding(3) var<storage, read>       tgbScale : array<STO>;
@group(0) @binding(4) var<storage, read>       tgbBias  : array<STO>;
@group(0) @binding(5) var<storage, read>       tgbMask  : array<STO>;
@group(0) @binding(6) var<storage, read_write> tgbOut   : array<STO>;

var<workgroup> tgbAs : array<f32, 256>;
var<workgroup> tgbBs : array<f32, 256>;

@compute @workgroup_size(16, 16)
fn tiledGemmInBnAct(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let n  = wid.z;
  let o  = wid.x * 16u + lid.x;
  let xy = wid.y * 16u + lid.y;
  var acc : f32 = 0.0;
  let nTiles = (tgb.inC + 15u) / 16u;
  for (var t : u32 = 0u; t < nTiles; t = t + 1u) {
    let kA = t * 16u + lid.y;
    var av : f32 = 0.0;
    if (o < tgb.outC && kA < tgb.inC) { av = f32(tgbW[o * tgb.inC + kA]); }
    tgbAs[lid.x * 16u + lid.y] = av;
    let kB = t * 16u + lid.x;
    var bv : f32 = 0.0;
    if (kB < tgb.inC && xy < tgb.hw) {
      let m = f32(tgbMask[n * tgb.hw + xy]);
      bv = activate(f32(tgbIn[(n * tgb.inC + kB) * tgb.hw + xy]) * f32(tgbScale[kB]) + f32(tgbBias[kB]), tgb.act) * m;
    }
    tgbBs[lid.x * 16u + lid.y] = bv;
    workgroupBarrier();
    for (var kk : u32 = 0u; kk < 16u; kk = kk + 1u) { acc = acc + tgbAs[lid.x * 16u + kk] * tgbBs[kk * 16u + lid.y]; }
    workgroupBarrier();
  }
  if (o < tgb.outC && xy < tgb.hw) { tgbOut[(n * tgb.outC + o) * tgb.hw + xy] = STO(acc); }
}

// ===========================================================================
// Winograd F(2x2, 3x3) convolution — 3 stages (SAME padding, stride 1, dil 1).
// Reduces the 3x3 conv's per-output multiplies from 9 to 4 by transforming
// 4x4 input tiles and 3x3 filters into a 4x4 elementwise (Hadamard) domain.
//
//   B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]   (input transform V = B^T d B)
//   G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]        (filter transform U = G g G^T, on host)
//   A^T = [[1,1,1,0],[0,1,-1,-1]]                          (output transform Y = A^T M A)
//
// Tiling: nTilesY=ceil(H/2), nTilesX=ceil(W/2), tile=ty*nTilesX+tx.
// Buffers (STO):
//   V : [16][n][inC ][nTiles]   M : [16][n][outC][nTiles]   U(host) : [16][outC][inC]
// ===========================================================================

struct WgParams {
  n : u32, inC : u32, outC : u32,
  h : u32, w : u32, nTilesX : u32, nTilesY : u32, xi : u32,
};

// --- Stage 1: input transform.  thread per (n, ic, tile) ---
@group(0) @binding(0) var<uniform> wgi : WgParams;
@group(0) @binding(1) var<storage, read>       wgiIn : array<STO>;
@group(0) @binding(2) var<storage, read_write> wgiV  : array<STO>;

@compute @workgroup_size(64)
fn winogradInput(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nTiles = wgi.nTilesX * wgi.nTilesY;
  let idx = gid.x;
  if (idx >= wgi.n * wgi.inC * nTiles) { return; }
  let tile = idx % nTiles;
  let ic   = (idx / nTiles) % wgi.inC;
  let nn   = idx / (nTiles * wgi.inC);
  let ty   = tile / wgi.nTilesX;
  let tx   = tile % wgi.nTilesX;

  // Load 4x4 input tile d (top-left at 2*tile-1 for SAME pad=1), 0 out of bounds.
  let chanBase = (nn * wgi.inC + ic) * wgi.h * wgi.w;
  var d : array<f32, 16>;
  for (var a = 0u; a < 4u; a = a + 1u) {
    let iy = i32(2u * ty + a) - 1;
    for (var b = 0u; b < 4u; b = b + 1u) {
      let ix = i32(2u * tx + b) - 1;
      var v = 0.0;
      if (iy >= 0 && iy < i32(wgi.h) && ix >= 0 && ix < i32(wgi.w)) {
        v = f32(wgiIn[chanBase + u32(iy) * wgi.w + u32(ix)]);
      }
      d[a * 4u + b] = v;
    }
  }
  // t = B^T d   (rows: d0-d2, d1+d2, -d1+d2, d1-d3)
  var t : array<f32, 16>;
  for (var j = 0u; j < 4u; j = j + 1u) {
    let d0 = d[0u*4u+j]; let d1 = d[1u*4u+j]; let d2 = d[2u*4u+j]; let d3 = d[3u*4u+j];
    t[0u*4u+j] = d0 - d2;
    t[1u*4u+j] = d1 + d2;
    t[2u*4u+j] = -d1 + d2;
    t[3u*4u+j] = d1 - d3;
  }
  // V = t B   (cols: t0-t2, t1+t2, -t1+t2, t1-t3)
  let nTilesAll = wgi.n * wgi.inC * nTiles;
  let vOff = (nn * wgi.inC + ic) * nTiles + tile;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let t0 = t[i*4u+0u]; let t1 = t[i*4u+1u]; let t2 = t[i*4u+2u]; let t3 = t[i*4u+3u];
    wgiV[( (i*4u+0u) ) * nTilesAll + vOff] = STO(t0 - t2);
    wgiV[( (i*4u+1u) ) * nTilesAll + vOff] = STO(t1 + t2);
    wgiV[( (i*4u+2u) ) * nTilesAll + vOff] = STO(-t1 + t2);
    wgiV[( (i*4u+3u) ) * nTilesAll + vOff] = STO(t1 - t3);
  }
}

// --- Stage 1 fused: winogradInput with pre-activation BN+act+mask folded into the
// load (act(in*scale[ic]+bias[ic])*mask) so a pre-conv scaleBiasMaskAct dispatch is
// eliminated. xi field carries the activation kind. Math identical to the unfused
// path (validated byte-exact in fp32). ---
@group(0) @binding(0) var<uniform> wgib : WgParams;
@group(0) @binding(1) var<storage, read>       wgibIn    : array<STO>;
@group(0) @binding(2) var<storage, read>       wgibScale : array<STO>;
@group(0) @binding(3) var<storage, read>       wgibBias  : array<STO>;
@group(0) @binding(4) var<storage, read>       wgibMask  : array<STO>;
@group(0) @binding(5) var<storage, read_write> wgibV     : array<STO>;

@compute @workgroup_size(64)
fn winogradInputBnAct(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nTiles = wgib.nTilesX * wgib.nTilesY;
  let idx = gid.x;
  if (idx >= wgib.n * wgib.inC * nTiles) { return; }
  let tile = idx % nTiles;
  let ic   = (idx / nTiles) % wgib.inC;
  let nn   = idx / (nTiles * wgib.inC);
  let ty   = tile / wgib.nTilesX;
  let tx   = tile % wgib.nTilesX;
  let hw = wgib.h * wgib.w;
  let chanBase = (nn * wgib.inC + ic) * hw;
  let sc = f32(wgibScale[ic]); let bi = f32(wgibBias[ic]);
  var d : array<f32, 16>;
  for (var a = 0u; a < 4u; a = a + 1u) {
    let iy = i32(2u * ty + a) - 1;
    for (var b = 0u; b < 4u; b = b + 1u) {
      let ix = i32(2u * tx + b) - 1;
      var v = 0.0;
      if (iy >= 0 && iy < i32(wgib.h) && ix >= 0 && ix < i32(wgib.w)) {
        let raw = f32(wgibIn[chanBase + u32(iy) * wgib.w + u32(ix)]);
        let m   = f32(wgibMask[nn * hw + u32(iy) * wgib.w + u32(ix)]);
        v = activate(raw * sc + bi, wgib.xi) * m;
      }
      d[a * 4u + b] = v;
    }
  }
  var t : array<f32, 16>;
  for (var j = 0u; j < 4u; j = j + 1u) {
    let d0 = d[0u*4u+j]; let d1 = d[1u*4u+j]; let d2 = d[2u*4u+j]; let d3 = d[3u*4u+j];
    t[0u*4u+j] = d0 - d2;
    t[1u*4u+j] = d1 + d2;
    t[2u*4u+j] = -d1 + d2;
    t[3u*4u+j] = d1 - d3;
  }
  let nTilesAll = wgib.n * wgib.inC * nTiles;
  let vOff = (nn * wgib.inC + ic) * nTiles + tile;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let t0 = t[i*4u+0u]; let t1 = t[i*4u+1u]; let t2 = t[i*4u+2u]; let t3 = t[i*4u+3u];
    wgibV[( (i*4u+0u) ) * nTilesAll + vOff] = STO(t0 - t2);
    wgibV[( (i*4u+1u) ) * nTilesAll + vOff] = STO(t1 + t2);
    wgibV[( (i*4u+2u) ) * nTilesAll + vOff] = STO(-t1 + t2);
    wgibV[( (i*4u+3u) ) * nTilesAll + vOff] = STO(t1 - t3);
  }
}

// --- Stage 2: per-component matmul  M[xi] = U[xi](outC x inC) . V[xi](inC x nTiles)
// thread per (xi, n, oc, tile) ---
@group(0) @binding(0) var<uniform> wgm : WgParams;
@group(0) @binding(1) var<storage, read>       wgmU : array<STO>;
@group(0) @binding(2) var<storage, read>       wgmV : array<STO>;
@group(0) @binding(3) var<storage, read_write> wgmM : array<STO>;

@compute @workgroup_size(64)
fn winogradMatmul(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nTiles = wgm.nTilesX * wgm.nTilesY;
  let perComp = wgm.n * wgm.outC * nTiles;
  let idx = gid.x;
  if (idx >= 16u * perComp) { return; }
  let xi   = idx / perComp;
  let rem  = idx % perComp;
  let tile = rem % nTiles;
  let oc   = (rem / nTiles) % wgm.outC;
  let nn   = rem / (nTiles * wgm.outC);

  let vAll = wgm.n * wgm.inC * nTiles;
  let uBase = xi * (wgm.outC * wgm.inC) + oc * wgm.inC;     // U[xi][oc][*]
  let vBase = xi * vAll + nn * wgm.inC * nTiles + tile;     // V[xi][nn][*][tile]
  var acc = 0.0;
  for (var ic = 0u; ic < wgm.inC; ic = ic + 1u) {
    acc = acc + f32(wgmU[uBase + ic]) * f32(wgmV[vBase + ic * nTiles]);
  }
  wgmM[idx] = STO(acc);
}

// --- Stage 3: output transform.  thread per (n, oc, tile) ---
@group(0) @binding(0) var<uniform> wgo : WgParams;
@group(0) @binding(1) var<storage, read>       wgoM   : array<STO>;
@group(0) @binding(2) var<storage, read_write> wgoOut : array<STO>;

@compute @workgroup_size(64)
fn winogradOutput(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nTiles = wgo.nTilesX * wgo.nTilesY;
  let idx = gid.x;
  if (idx >= wgo.n * wgo.outC * nTiles) { return; }
  let tile = idx % nTiles;
  let oc   = (idx / nTiles) % wgo.outC;
  let nn   = idx / (nTiles * wgo.outC);
  let ty   = tile / wgo.nTilesX;
  let tx   = tile % wgo.nTilesX;

  let perComp = wgo.n * wgo.outC * nTiles;
  let mOff = nn * wgo.outC * nTiles + oc * nTiles + tile;   // within a component
  var m : array<f32, 16>;
  for (var k = 0u; k < 16u; k = k + 1u) { m[k] = f32(wgoM[k * perComp + mOff]); }

  // s = A^T m   (2x4): row0 = m0+m1+m2 ; row1 = m1-m2-m3
  var s : array<f32, 8>;
  for (var j = 0u; j < 4u; j = j + 1u) {
    s[0u*4u+j] = m[0u*4u+j] + m[1u*4u+j] + m[2u*4u+j];
    s[1u*4u+j] = m[1u*4u+j] - m[2u*4u+j] - m[3u*4u+j];
  }
  // Y = s A   (2x2): col0 = s0+s1+s2 ; col1 = s1-s2-s3
  let outChanBase = (nn * wgo.outC + oc) * wgo.h * wgo.w;
  for (var i = 0u; i < 2u; i = i + 1u) {
    let s0 = s[i*4u+0u]; let s1 = s[i*4u+1u]; let s2 = s[i*4u+2u]; let s3 = s[i*4u+3u];
    let oy = 2u * ty + i;
    if (oy < wgo.h) {
      let ox0 = 2u * tx;
      if (ox0 < wgo.w)      { wgoOut[outChanBase + oy * wgo.w + ox0]      = STO(s0 + s1 + s2); }
      if (ox0 + 1u < wgo.w) { wgoOut[outChanBase + oy * wgo.w + ox0 + 1u] = STO(s1 - s2 - s3); }
    }
  }
}

// ===========================================================================
// rmsNorm: per-position RMSNorm across channels (transformer preLN + trunk tip).
//   ms = sum_c in[n,c,xy]^2 / C ;  r = 1/sqrt(ms + eps)
//   out[n,c,xy] = act(in[n,c,xy] * r * gamma[c] (+ beta[c])) * mask[n,xy]
// One thread per (n,xy). hasBeta picks trunk-tip (gamma+beta+act) vs transformer
// preLN (weight only). Masked positions output 0.
// ===========================================================================

struct RMSParams { n : u32, c : u32, hw : u32, act : u32, eps : f32, hasBeta : u32, pad0 : u32, pad1 : u32, };

@group(0) @binding(0) var<uniform> rms : RMSParams;
@group(0) @binding(1) var<storage, read>       rmsIn    : array<STO>;
@group(0) @binding(2) var<storage, read>       rmsGamma : array<STO>;
@group(0) @binding(3) var<storage, read>       rmsBeta  : array<STO>;
@group(0) @binding(4) var<storage, read>       rmsMask  : array<STO>;
@group(0) @binding(5) var<storage, read_write> rmsOut   : array<STO>;

@compute @workgroup_size(64)
fn rmsNorm(@builtin(global_invocation_id) gid : vec3<u32>) {
  let pos = gid.x;                 // [0, n*hw)
  if (pos >= rms.n * rms.hw) { return; }
  let n  = pos / rms.hw;
  let xy = pos % rms.hw;
  let base = n * rms.c * rms.hw + xy;
  if (f32(rmsMask[n * rms.hw + xy]) == 0.0) {
    for (var c : u32 = 0u; c < rms.c; c = c + 1u) { rmsOut[base + c * rms.hw] = STO(0.0); }
    return;
  }
  var sumSq : f32 = 0.0;
  for (var c : u32 = 0u; c < rms.c; c = c + 1u) { let v = f32(rmsIn[base + c * rms.hw]); sumSq = sumSq + v * v; }
  let r = inverseSqrt(sumSq / f32(rms.c) + rms.eps);
  for (var c : u32 = 0u; c < rms.c; c = c + 1u) {
    var v = f32(rmsIn[base + c * rms.hw]) * r * f32(rmsGamma[c]);
    if (rms.hasBeta != 0u) { v = v + f32(rmsBeta[c]); }
    rmsOut[base + c * rms.hw] = STO(activate(v, rms.act));
  }
}

// ===========================================================================
// proj1x1: per-position channel projection with MatMulLayerDesc weights, which
// are [inC][outC] (in-major) -- the TRANSPOSE of conv1x1's [outC][inC]. No bias
// (transformer q/k/v/out and FFN linears are bias-free matmuls).
//   out[n,o,xy] = sum_ic in[n,ic,xy] * W[ic*outC + o]
// ===========================================================================

struct ProjParams { n : u32, inC : u32, outC : u32, hw : u32, };

@group(0) @binding(0) var<uniform> pj : ProjParams;
@group(0) @binding(1) var<storage, read>       pjIn  : array<STO>;
@group(0) @binding(2) var<storage, read>       pjW   : array<STO>;
@group(0) @binding(3) var<storage, read_write> pjOut : array<STO>;

@compute @workgroup_size(64)
fn proj1x1(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= pj.n * pj.outC * pj.hw) { return; }
  let xy = idx % pj.hw;
  let o  = (idx / pj.hw) % pj.outC;
  let n  = idx / (pj.hw * pj.outC);
  let inBase = n * pj.inC * pj.hw + xy;
  var acc : f32 = 0.0;
  for (var ic : u32 = 0u; ic < pj.inC; ic = ic + 1u) {
    acc = acc + f32(pjIn[inBase + ic * pj.hw]) * f32(pjW[ic * pj.outC + o]);
  }
  pjOut[idx] = STO(acc);
}

// ===========================================================================
// ropeApply: in-place rotary embedding on q/k. For each (n, head, pair p, xy)
// rotate channels (head*headDim+2p, +1) by cos/sin[tableIdx].
//   learnable: tableIdx = (kvh*numPairs+p)*hw + xy,  kvh = head*numKVHeads/numHeads
//   fixed:     tableIdx = p*hw + xy
// ===========================================================================

struct RopeParams { n : u32, numHeads : u32, headDim : u32, numPairs : u32, hw : u32, numKVHeads : u32, learnable : u32, pad0 : u32, };

@group(0) @binding(0) var<uniform> rp : RopeParams;
@group(0) @binding(1) var<storage, read_write> rpData : array<STO>;
@group(0) @binding(2) var<storage, read>       rpCos  : array<STO>;
@group(0) @binding(3) var<storage, read>       rpSin  : array<STO>;

@compute @workgroup_size(64)
fn ropeApply(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= rp.n * rp.numHeads * rp.numPairs * rp.hw) { return; }
  let xy = idx % rp.hw;
  let p  = (idx / rp.hw) % rp.numPairs;
  let h  = (idx / (rp.hw * rp.numPairs)) % rp.numHeads;
  let n  = idx / (rp.hw * rp.numPairs * rp.numHeads);
  let totalDim = rp.numHeads * rp.headDim;
  let c0 = h * rp.headDim + 2u * p;
  let idx0 = (n * totalDim + c0) * rp.hw + xy;
  let idx1 = idx0 + rp.hw;
  var tableIdx : u32;
  if (rp.learnable != 0u) { let kvh = h * rp.numKVHeads / rp.numHeads; tableIdx = (kvh * rp.numPairs + p) * rp.hw + xy; }
  else { tableIdx = p * rp.hw + xy; }
  let cv = f32(rpCos[tableIdx]);
  let sv = f32(rpSin[tableIdx]);
  let x0 = f32(rpData[idx0]);
  let x1 = f32(rpData[idx1]);
  rpData[idx0] = STO(x0 * cv - x1 * sv);
  rpData[idx1] = STO(x0 * sv + x1 * cv);
}

// ===========================================================================
// attnScores: scores[n,h,qi,ki] = (1/sqrt(headDim)) * sum_d q[n,h,d,qi]*k[n,kvh,d,ki]
//   kvh = h / kvGroupSize ; q is NCHW [n, numHeads*headDim, hw], k [n, numKVHeads*headDim, hw]
// ===========================================================================

struct AttnSParams { n : u32, numHeads : u32, numKVHeads : u32, headDim : u32, hw : u32, kvGroupSize : u32, scale : f32, pad0 : u32, };

@group(0) @binding(0) var<uniform> asP : AttnSParams;
@group(0) @binding(1) var<storage, read>       asQ : array<STO>;
@group(0) @binding(2) var<storage, read>       asK : array<STO>;
@group(0) @binding(3) var<storage, read_write> asScores : array<STO>;

@compute @workgroup_size(64)
fn attnScores(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= asP.n * asP.numHeads * asP.hw * asP.hw) { return; }
  let ki = idx % asP.hw;
  let qi = (idx / asP.hw) % asP.hw;
  let h  = (idx / (asP.hw * asP.hw)) % asP.numHeads;
  let n  = idx / (asP.hw * asP.hw * asP.numHeads);
  let kvh = h / asP.kvGroupSize;
  let qBase = (n * asP.numHeads * asP.headDim + h * asP.headDim) * asP.hw + qi;
  let kBase = (n * asP.numKVHeads * asP.headDim + kvh * asP.headDim) * asP.hw + ki;
  var acc : f32 = 0.0;
  for (var d : u32 = 0u; d < asP.headDim; d = d + 1u) {
    acc = acc + f32(asQ[qBase + d * asP.hw]) * f32(asK[kBase + d * asP.hw]);
  }
  asScores[idx] = STO(acc * asP.scale);
}

// ===========================================================================
// attnSoftmax: in-place row-softmax of scores[n,h,qi,:] over ki, masked.
// One thread per (n,h,qi). Masked qi -> zero row; masked ki excluded.
// ===========================================================================

struct AttnSMParams { n : u32, numHeads : u32, hw : u32, pad0 : u32, };

@group(0) @binding(0) var<uniform> smP : AttnSMParams;
@group(0) @binding(1) var<storage, read_write> smScores : array<STO>;
@group(0) @binding(2) var<storage, read>       smMask : array<STO>;

@compute @workgroup_size(64)
fn attnSoftmax(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= smP.n * smP.numHeads * smP.hw) { return; }
  let qi = idx % smP.hw;
  let n  = idx / (smP.hw * smP.numHeads);
  let rowBase = idx * smP.hw;
  if (f32(smMask[n * smP.hw + qi]) == 0.0) {
    for (var ki : u32 = 0u; ki < smP.hw; ki = ki + 1u) { smScores[rowBase + ki] = STO(0.0); }
    return;
  }
  var mx : f32 = -3.0e38;
  for (var ki : u32 = 0u; ki < smP.hw; ki = ki + 1u) {
    if (f32(smMask[n * smP.hw + ki]) != 0.0) { mx = max(mx, f32(smScores[rowBase + ki])); }
  }
  var sum : f32 = 0.0;
  for (var ki : u32 = 0u; ki < smP.hw; ki = ki + 1u) {
    var e : f32 = 0.0;
    if (f32(smMask[n * smP.hw + ki]) != 0.0) { e = exp(f32(smScores[rowBase + ki]) - mx); }
    smScores[rowBase + ki] = STO(e);
    sum = sum + e;
  }
  for (var ki : u32 = 0u; ki < smP.hw; ki = ki + 1u) { smScores[rowBase + ki] = STO(f32(smScores[rowBase + ki]) / sum); }
}

// ===========================================================================
// attnOutput: out[n,h,e,qi] = sum_ki scores[n,h,qi,ki] * v[n,kvh,e,ki]
//   out NCHW [n, numHeads*vHeadDim, hw]; v NCHW [n, numKVHeads*vHeadDim, hw]
// ===========================================================================

struct AttnOParams { n : u32, numHeads : u32, numKVHeads : u32, vHeadDim : u32, hw : u32, kvGroupSize : u32, pad0 : u32, pad1 : u32, };

@group(0) @binding(0) var<uniform> aoP : AttnOParams;
@group(0) @binding(1) var<storage, read>       aoScores : array<STO>;
@group(0) @binding(2) var<storage, read>       aoV : array<STO>;
@group(0) @binding(3) var<storage, read_write> aoOut : array<STO>;

@compute @workgroup_size(64)
fn attnOutput(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= aoP.n * aoP.numHeads * aoP.vHeadDim * aoP.hw) { return; }
  let qi = idx % aoP.hw;
  let e  = (idx / aoP.hw) % aoP.vHeadDim;
  let h  = (idx / (aoP.hw * aoP.vHeadDim)) % aoP.numHeads;
  let n  = idx / (aoP.hw * aoP.vHeadDim * aoP.numHeads);
  let kvh = h / aoP.kvGroupSize;
  let rowBase = ((n * aoP.numHeads + h) * aoP.hw + qi) * aoP.hw;
  let vBase = (n * aoP.numKVHeads * aoP.vHeadDim + kvh * aoP.vHeadDim + e) * aoP.hw;
  var acc : f32 = 0.0;
  for (var ki : u32 = 0u; ki < aoP.hw; ki = ki + 1u) {
    acc = acc + f32(aoScores[rowBase + ki]) * f32(aoV[vBase + ki]);
  }
  aoOut[(n * aoP.numHeads * aoP.vHeadDim + h * aoP.vHeadDim + e) * aoP.hw + qi] = STO(acc);
}

// ===========================================================================
// flashAttention: fused scaled-dot-product attention with ONLINE softmax — never
// materializes the [seq x seq] score matrix (vs attnScores/Softmax/Output). One
// thread per (n, head, qi); streams over keys keeping running max m, sum l, and a
// vHeadDim accumulator. Mathematically equal to the 3-kernel path (the running
// sum reorders additions -> float-rounding only). vHeadDim must be <= 64.
// ===========================================================================

struct FlashParams { n : u32, numHeads : u32, numKVHeads : u32, qHeadDim : u32, vHeadDim : u32, hw : u32, kvGroupSize : u32, scale : f32, };

@group(0) @binding(0) var<uniform> fa : FlashParams;
@group(0) @binding(1) var<storage, read>       faQ    : array<STO>;
@group(0) @binding(2) var<storage, read>       faK    : array<STO>;
@group(0) @binding(3) var<storage, read>       faV    : array<STO>;
@group(0) @binding(4) var<storage, read>       faMask : array<STO>;
@group(0) @binding(5) var<storage, read_write> faOut  : array<STO>;

@compute @workgroup_size(64)
fn flashAttention(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= fa.n * fa.numHeads * fa.hw) { return; }
  let qi = idx % fa.hw;
  let h  = (idx / fa.hw) % fa.numHeads;
  let n  = idx / (fa.hw * fa.numHeads);
  let outBase = (n * fa.numHeads * fa.vHeadDim + h * fa.vHeadDim) * fa.hw + qi;
  if (f32(faMask[n * fa.hw + qi]) == 0.0) {
    for (var e : u32 = 0u; e < fa.vHeadDim; e = e + 1u) { faOut[outBase + e * fa.hw] = STO(0.0); }
    return;
  }
  let kvh = h / fa.kvGroupSize;
  let qBase = (n * fa.numHeads * fa.qHeadDim + h * fa.qHeadDim) * fa.hw + qi;
  var acc : array<f32, 64>;
  for (var e : u32 = 0u; e < fa.vHeadDim; e = e + 1u) { acc[e] = 0.0; }
  var m : f32 = -3.0e38;
  var l : f32 = 0.0;
  for (var ki : u32 = 0u; ki < fa.hw; ki = ki + 1u) {
    if (f32(faMask[n * fa.hw + ki]) == 0.0) { continue; }
    let kBase = (n * fa.numKVHeads * fa.qHeadDim + kvh * fa.qHeadDim) * fa.hw + ki;
    var s : f32 = 0.0;
    for (var d : u32 = 0u; d < fa.qHeadDim; d = d + 1u) { s = s + f32(faQ[qBase + d * fa.hw]) * f32(faK[kBase + d * fa.hw]); }
    s = s * fa.scale;
    let newMax = max(m, s);
    let corr = exp(m - newMax);
    let p = exp(s - newMax);
    l = l * corr + p;
    let vBase = (n * fa.numKVHeads * fa.vHeadDim + kvh * fa.vHeadDim) * fa.hw + ki;
    for (var e : u32 = 0u; e < fa.vHeadDim; e = e + 1u) { acc[e] = acc[e] * corr + p * f32(faV[vBase + e * fa.hw]); }
    m = newMax;
  }
  let inv = 1.0 / l;
  for (var e : u32 = 0u; e < fa.vHeadDim; e = e + 1u) { faOut[outBase + e * fa.hw] = STO(acc[e] * inv); }
}

// ===========================================================================
// swigluGate: out[i] = silu(a[i]) * gate[i]   (SwiGLU FFN hidden activation)
// ===========================================================================

struct SwiParams { total : u32, pad0 : u32, pad1 : u32, pad2 : u32, };

@group(0) @binding(0) var<uniform> swP : SwiParams;
@group(0) @binding(1) var<storage, read>       swA : array<STO>;
@group(0) @binding(2) var<storage, read>       swGate : array<STO>;
@group(0) @binding(3) var<storage, read_write> swOut : array<STO>;

@compute @workgroup_size(64)
fn swigluGate(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= swP.total) { return; }
  let a = f32(swA[i]);
  let s = select(1.0 / (1.0 + exp(-a)), exp(a) / (1.0 + exp(a)), a < 0.0);
  swOut[i] = STO(a * s * f32(swGate[i]));
}

// ===========================================================================
// Spatial RMSNorm (trunk tip): one rms over ALL channels AND valid positions per
// batch element. Two passes: reduce -> per-batch rms, then apply.
//   rms[n] = 1/sqrt( (sum over valid c,xy of x^2) / (validPositions*C) + eps )
// ===========================================================================

struct RMSRParams { n : u32, c : u32, hw : u32, pad0 : u32, eps : f32, pad1 : u32, pad2 : u32, pad3 : u32, };

@group(0) @binding(0) var<uniform> rmsr : RMSRParams;
@group(0) @binding(1) var<storage, read>       rmsrIn   : array<STO>;
@group(0) @binding(2) var<storage, read>       rmsrMask : array<STO>;
@group(0) @binding(3) var<storage, read_write> rmsrOut  : array<f32>;   // [n]

@compute @workgroup_size(64)
fn rmsReduceSpatial(@builtin(global_invocation_id) gid : vec3<u32>) {
  let nn = gid.x;
  if (nn >= rmsr.n) { return; }
  var sumSq : f32 = 0.0;
  var valid : u32 = 0u;
  for (var xy : u32 = 0u; xy < rmsr.hw; xy = xy + 1u) {
    if (f32(rmsrMask[nn * rmsr.hw + xy]) != 0.0) {
      valid = valid + 1u;
      for (var c : u32 = 0u; c < rmsr.c; c = c + 1u) {
        let v = f32(rmsrIn[(nn * rmsr.c + c) * rmsr.hw + xy]);
        sumSq = sumSq + v * v;
      }
    }
  }
  let total = max(1.0, f32(valid) * f32(rmsr.c));
  rmsrOut[nn] = inverseSqrt(sumSq / total + rmsr.eps);
}

struct RMSAParams { n : u32, c : u32, hw : u32, act : u32, hasBeta : u32, pad0 : u32, pad1 : u32, pad2 : u32, };

@group(0) @binding(0) var<uniform> rmsa : RMSAParams;
@group(0) @binding(1) var<storage, read>       rmsaIn    : array<STO>;
@group(0) @binding(2) var<storage, read>       rmsaGamma : array<STO>;
@group(0) @binding(3) var<storage, read>       rmsaBeta  : array<STO>;
@group(0) @binding(4) var<storage, read>       rmsaMask  : array<STO>;
@group(0) @binding(5) var<storage, read>       rmsaRms   : array<f32>;   // [n]
@group(0) @binding(6) var<storage, read_write> rmsaOut   : array<STO>;

@compute @workgroup_size(64)
fn rmsApplySpatial(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= rmsa.n * rmsa.c * rmsa.hw) { return; }
  let xy = idx % rmsa.hw;
  let c  = (idx / rmsa.hw) % rmsa.c;
  let nn = idx / (rmsa.hw * rmsa.c);
  if (f32(rmsaMask[nn * rmsa.hw + xy]) == 0.0) { rmsaOut[idx] = STO(0.0); return; }
  var v = f32(rmsaIn[idx]) * rmsaRms[nn] * f32(rmsaGamma[c]);
  if (rmsa.hasBeta != 0u) { v = v + f32(rmsaBeta[c]); }
  rmsaOut[idx] = STO(activate(v, rmsa.act));
}
)WGSL";

}  // namespace KataGoWebGPU
