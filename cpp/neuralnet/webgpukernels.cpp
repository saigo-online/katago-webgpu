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
)WGSL";

}  // namespace KataGoWebGPU
