#ifndef NEURALNET_WEBGPUKERNELS_H_
#define NEURALNET_WEBGPUKERNELS_H_

// WGSL compute-shader source for the native WebGPU (Dawn) backend.
//
// SCAFFOLD — see engines/katago-web/WEBGPU_STATUS.md. This is the starter
// kernel library that mirrors the core ops in the OpenCL backend
// (neuralnet/openclkernels.cpp). It is fp32-only for now (WebGPU's shader-f16
// is an optional feature; an fp16 path is a later milestone) and is NOT yet
// numerically validated against the reference backends.
//
// All spatial tensors use NCHW layout (channel-major, then row-major H,W),
// matching KataGo's internal convention. "mask" is NHW (no channel dim): 1.0
// for on-board locations, 0.0 for padding outside the (boardY,boardX) region.

namespace KataGoWebGPU {
  // A single WGSL module string containing every compute entry point.
  // Entry points (see the source for bind-group layouts):
  //   scaleBiasMaskAct  - fused batchnorm(merged scale/bias) + activation + mask
  //   addInPlace        - residual add (dst += src)
  //   matMulBiasAct     - C[M,N] = act(A[M,K] @ B[K,N] + bias[N])
  //   globalPoolMeanMax - per (n,c): mean, layer-scaled mean, max over the masked board
  //   conv2dNCHW        - direct convolution, arbitrary filter size + dilation
  extern const char* const WGSL_KERNELS;

  // Activation enum values shared with neuralnet/activations.h, passed into
  // kernels as a uniform so a single pipeline handles identity/relu/mish.
  // Keep in sync with ACTIVATION_* in activations.h.
}  // namespace KataGoWebGPU

#endif  // NEURALNET_WEBGPUKERNELS_H_
