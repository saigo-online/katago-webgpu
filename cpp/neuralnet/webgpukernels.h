#ifndef NEURALNET_WEBGPUKERNELS_H_
#define NEURALNET_WEBGPUKERNELS_H_

// WGSL compute-shader source for the native WebGPU (Dawn) backend.
//
// See engines/katago-web/WEBGPU_STATUS.md. The full kernel library for the
// KataGo architecture through modelVersion 17, validated against the CPU
// reference (runnnlayertests per-op; full-net byte-identical in fp32).
//
// Storage precision is the `STO` alias the backend prepends (`f32`, or
// `enable f16; ... f16` for fp16 storage); math always accumulates in f32.
//
// All spatial tensors use NCHW layout (channel-major, then row-major H,W),
// matching KataGo's internal convention. "mask" is NHW (no channel dim): 1.0
// for on-board locations, 0.0 for padding outside the (boardY,boardX) region.

namespace KataGoWebGPU {
  // A single WGSL module string containing every compute entry point:
  //   scaleBiasMaskAct  - fused batchnorm(merged scale/bias) + activation + mask
  //   addInPlace / addChannelBias / swigluGate      - elementwise ops
  //   matMulBiasAct     - C[M,N] = act(A[M,K] @ B[K,N] + bias[N])
  //   globalPoolMeanMax / globalPoolValueHead       - masked poolings
  //   conv2dNCHW / conv1x1NCHW / tiledGemm(RT) / tiledGemmInBnAct - convolutions
  //   winogradInput(BnAct) / winogradMatmul / winogradOutput      - F(2,3) 3x3 conv
  //   rmsNorm / rmsReduceSpatial / rmsApplySpatial / proj1x1      - RMSNorm + projections
  //   ropeApply / attnScores / attnSoftmax / attnOutput / flashAttention - attention
  extern const char* const WGSL_KERNELS;

  // Activation enum values shared with neuralnet/activations.h, passed into
  // kernels as a uniform so a single pipeline handles identity/relu/mish.
  // Keep in sync with ACTIVATION_* in activations.h.
}  // namespace KataGoWebGPU

#endif  // NEURALNET_WEBGPUKERNELS_H_
