#ifdef ZFP_WITH_METAL_NATIVE

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "metal_runtime.h"

/* GPU profiling: set ZFP_METAL_PROFILE=1 to print per-dispatch GPU timing.
   Uses MTLCommandBuffer GPUStartTime/GPUEndTime for precise GPU-side
   measurement, excluding all CPU-side dispatch/buffer overhead. */
static int zfp_metal_profile_enabled = -1; /* -1 = not checked yet */
static inline int zfp_metal_profile(void) {
  if (zfp_metal_profile_enabled < 0) {
    const char* env = getenv("ZFP_METAL_PROFILE");
    zfp_metal_profile_enabled = env ? atoi(env) : 0;
  }
  return zfp_metal_profile_enabled;
}
static int zfp_metal_pso_info_printed = 0;
static inline void zfp_metal_report_gpu_time(id<MTLCommandBuffer> cb,
                                              const char* label,
                                              unsigned int total_blocks,
                                              size_t data_bytes) {
  if (!zfp_metal_profile())
    return;
  double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
  double gb_s = (data_bytes > 0 && gpu_s > 0.0)
    ? ((double)data_bytes / (1024.0*1024.0*1024.0)) / gpu_s : 0.0;
  fprintf(stderr, "[Metal GPU] %-40s  blocks=%6u  gpu=%.6f s  %.2f GB/s\n",
          label, total_blocks, gpu_s, gb_s);
}
static inline void zfp_metal_report_pso_info(id<MTLComputePipelineState> pso,
                                              const char* label) {
  if (zfp_metal_profile() < 2 || zfp_metal_pso_info_printed)
    return;
  fprintf(stderr, "[Metal PSO] %-40s  maxThreads=%lu  simdWidth=%lu  staticMem=%lu\n",
          label,
          (unsigned long)pso.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso.threadExecutionWidth,
          (unsigned long)pso.staticThreadgroupMemoryLength);
}

typedef struct ZfpMetalLayoutParams {
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
  long sx;
  long sy;
  long sz;
  unsigned long elem_size;
  unsigned long total;
  long offset;
} ZfpMetalLayoutParams;

typedef struct ZfpMetalCodec1dParams {
  unsigned int dim;
  int sx;
  unsigned int maxbits;
  unsigned int padded_dim;
  unsigned int total_blocks;
} ZfpMetalCodec1dParams;

typedef struct ZfpMetalCodec2dParams {
  unsigned int nx;
  unsigned int ny;
  long sx;
  long sy;
  unsigned int maxbits;
  unsigned int bx;
  unsigned int by;
  unsigned int total_blocks;
} ZfpMetalCodec2dParams;

typedef struct ZfpMetalCodec3dParams {
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
  long sx;
  long sy;
  long sz;
  unsigned int maxbits;
  unsigned int bx;
  unsigned int by;
  unsigned int bz;
  unsigned int total_blocks;
} ZfpMetalCodec3dParams;

typedef struct ZfpMetalContext {
  int initialized;
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pack_ps;
  id<MTLComputePipelineState> unpack_ps;
  id<MTLComputePipelineState> encode1_ps;
  id<MTLComputePipelineState> decode1_ps;
  id<MTLComputePipelineState> encode2_ps;
  id<MTLComputePipelineState> decode2_ps;
  id<MTLComputePipelineState> encode3_ps;
  id<MTLComputePipelineState> decode3_ps;
  id<MTLComputePipelineState> encode1d_float_ps;
  id<MTLComputePipelineState> decode1d_float_ps;
  id<MTLComputePipelineState> encode2d_float_ps;
  id<MTLComputePipelineState> decode2d_float_ps;
  id<MTLComputePipelineState> encode3d_float_ps;
  id<MTLComputePipelineState> decode3d_float_ps;
  id<MTLComputePipelineState> encode1d_float_r8_ps;
  id<MTLComputePipelineState> decode1d_float_r8_ps;
  id<MTLComputePipelineState> encode1d_float_r16_ps;
  id<MTLComputePipelineState> decode1d_float_r16_ps;
  id<MTLComputePipelineState> encode1d_float_r32_ps;
  id<MTLComputePipelineState> decode1d_float_r32_ps;
  id<MTLComputePipelineState> encode2d_float_r8_ps;
  id<MTLComputePipelineState> decode2d_float_r8_ps;
  id<MTLComputePipelineState> encode2d_float_r16_ps;
  id<MTLComputePipelineState> decode2d_float_r16_ps;
  id<MTLComputePipelineState> encode2d_float_r32_ps;
  id<MTLComputePipelineState> decode2d_float_r32_ps;
  id<MTLComputePipelineState> encode3d_float_r8_ps;
  id<MTLComputePipelineState> decode3d_float_r8_ps;
  id<MTLComputePipelineState> encode3d_float_r16_ps;
  id<MTLComputePipelineState> decode3d_float_r16_ps;
  id<MTLComputePipelineState> encode3d_float_r32_ps;
  id<MTLComputePipelineState> decode3d_float_r32_ps;
  id<MTLComputePipelineState> decode3d_float_tg_ps;
  id<MTLComputePipelineState> encode1d_int32_ps;
  id<MTLComputePipelineState> decode1d_int32_ps;
  id<MTLComputePipelineState> encode2d_int32_ps;
  id<MTLComputePipelineState> decode2d_int32_ps;
  id<MTLComputePipelineState> encode3d_int32_ps;
  id<MTLComputePipelineState> decode3d_int32_ps;
  id<MTLComputePipelineState> encode1d_int64_ps;
  id<MTLComputePipelineState> decode1d_int64_ps;
  id<MTLComputePipelineState> encode2d_int64_ps;
  id<MTLComputePipelineState> decode2d_int64_ps;
  id<MTLComputePipelineState> encode3d_int64_ps;
  id<MTLComputePipelineState> decode3d_int64_ps;
  /* Diagnostic PSOs for profiling */
  id<MTLComputePipelineState> diag_bitplane_only_3d_ps;
  id<MTLComputePipelineState> diag_transform_only_3d_ps;
  id<MTLComputePipelineState> diag_memcopy_3d_ps;
  id<MTLComputePipelineState> diag_bitplane_devub_3d_ps;
  id<MTLComputePipelineState> diag_bitread_only_3d_ps;
  id<MTLComputePipelineState> diag_bitplane_tgub_3d_ps;
  id<MTLComputePipelineState> diag_bitplane_split32_3d_ps;
  id<MTLComputePipelineState> decode3d_float_tgub_ps;
  id<MTLBuffer> src_buf;
  id<MTLBuffer> dst_buf;
  id<MTLBuffer> params_buf;
  size_t src_cap;
  size_t dst_cap;
  size_t params_cap;
} ZfpMetalContext;

static ZfpMetalContext zfp_metal_ctx = {
  0, nil, nil,
  nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
  nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
  nil, nil, nil, nil, nil,
  nil, nil, nil, nil, nil, nil,
  nil, nil, nil, 0, 0, 0
};
static int zfp_metal_error_logged = 0;

static void
zfp_metal_log_error_once(const char* where, NSError* err)
{
  if (zfp_metal_error_logged)
    return;
  if (err)
    fprintf(stderr, "zfp: Metal runtime %s failed: %s\n", where, [[err localizedDescription] UTF8String]);
  else
    fprintf(stderr, "zfp: Metal runtime %s failed\n", where);
  zfp_metal_error_logged = 1;
}

static int
zfp_metal_ensure_buffer(id<MTLDevice> device, id<MTLBuffer>* buffer, size_t* cap, size_t need)
{
  if (*buffer && *cap >= need)
    return 1;

  *buffer = nil;
  *cap = 0;
  *buffer = [device newBufferWithLength:need options:MTLResourceStorageModeShared];
  if (!*buffer)
    return 0;

  *cap = need;
  return 1;
}

static int
zfp_metal_init_context()
{
  if (zfp_metal_ctx.initialized)
    return zfp_metal_ctx.pack_ps && zfp_metal_ctx.unpack_ps &&
           zfp_metal_ctx.encode1_ps && zfp_metal_ctx.decode1_ps &&
           zfp_metal_ctx.encode2_ps && zfp_metal_ctx.decode2_ps &&
           zfp_metal_ctx.encode3_ps && zfp_metal_ctx.decode3_ps &&
           zfp_metal_ctx.encode1d_float_ps && zfp_metal_ctx.decode1d_float_ps &&
           zfp_metal_ctx.encode2d_float_ps && zfp_metal_ctx.decode2d_float_ps &&
           zfp_metal_ctx.encode3d_float_ps && zfp_metal_ctx.decode3d_float_ps;

  zfp_metal_ctx.initialized = 1;
  zfp_metal_ctx.device = MTLCreateSystemDefaultDevice();
  if (!zfp_metal_ctx.device)
    return 0;

  zfp_metal_ctx.queue = [zfp_metal_ctx.device newCommandQueue];
  if (!zfp_metal_ctx.queue)
    return 0;

  NSError* err = nil;
  NSString* metallib_path = [NSString stringWithUTF8String:ZFP_METAL_LIB_PATH];
  NSURL* metallib_url = [NSURL fileURLWithPath:metallib_path];
  id<MTLLibrary> lib = [zfp_metal_ctx.device newLibraryWithURL:metallib_url error:&err];
  if (!lib) {
    zfp_metal_log_error_once("newLibraryWithURL", err);
    return 0;
  }

  id<MTLFunction> fn_pack = [lib newFunctionWithName:@"zfp_pack_strided"];
  id<MTLFunction> fn_unpack = [lib newFunctionWithName:@"zfp_unpack_strided"];
  id<MTLFunction> fn_encode1 = [lib newFunctionWithName:@"zfp_encode_contig_1d"];
  id<MTLFunction> fn_decode1 = [lib newFunctionWithName:@"zfp_decode_contig_1d"];
  id<MTLFunction> fn_encode2 = [lib newFunctionWithName:@"zfp_encode_contig_2d"];
  id<MTLFunction> fn_decode2 = [lib newFunctionWithName:@"zfp_decode_contig_2d"];
  id<MTLFunction> fn_encode3 = [lib newFunctionWithName:@"zfp_encode_contig_3d"];
  id<MTLFunction> fn_decode3 = [lib newFunctionWithName:@"zfp_decode_contig_3d"];
  id<MTLFunction> fn_encode1d_float = [lib newFunctionWithName:@"zfp_encode1d_float"];
  id<MTLFunction> fn_decode1d_float = [lib newFunctionWithName:@"zfp_decode1d_float"];
  id<MTLFunction> fn_encode2d_float = [lib newFunctionWithName:@"zfp_encode2d_float"];
  id<MTLFunction> fn_decode2d_float = [lib newFunctionWithName:@"zfp_decode2d_float"];
  id<MTLFunction> fn_encode3d_float = [lib newFunctionWithName:@"zfp_encode3d_float"];
  id<MTLFunction> fn_decode3d_float = [lib newFunctionWithName:@"zfp_decode3d_float"];
  if (!fn_pack || !fn_unpack ||
      !fn_encode1 || !fn_decode1 || !fn_encode2 || !fn_decode2 || !fn_encode3 || !fn_decode3 ||
      !fn_encode1d_float || !fn_decode1d_float || !fn_encode2d_float || !fn_decode2d_float ||
      !fn_encode3d_float || !fn_decode3d_float) {
    zfp_metal_log_error_once("newFunctionWithName", nil);
    return 0;
  }

  zfp_metal_ctx.pack_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_pack error:&err];
  zfp_metal_ctx.unpack_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_unpack error:&err];
  zfp_metal_ctx.encode1_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_encode1 error:&err];
  zfp_metal_ctx.decode1_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_decode1 error:&err];
  zfp_metal_ctx.encode2_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_encode2 error:&err];
  zfp_metal_ctx.decode2_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_decode2 error:&err];
  zfp_metal_ctx.encode3_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_encode3 error:&err];
  zfp_metal_ctx.decode3_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_decode3 error:&err];
  zfp_metal_ctx.encode1d_float_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_encode1d_float error:&err];
  zfp_metal_ctx.decode1d_float_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_decode1d_float error:&err];
  zfp_metal_ctx.encode2d_float_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_encode2d_float error:&err];
  zfp_metal_ctx.decode2d_float_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_decode2d_float error:&err];
  zfp_metal_ctx.encode3d_float_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_encode3d_float error:&err];
  zfp_metal_ctx.decode3d_float_ps = [zfp_metal_ctx.device newComputePipelineStateWithFunction:fn_decode3d_float error:&err];

  /* Rate-specialized PSOs (r8, r16, r32) -- optional, used when maxbits
     matches a compile-time constant so the Metal compiler can optimize
     the bit-plane loop and eliminate the atomic_mode branch. */
  {
    struct { const char* name; id<MTLComputePipelineState>* ps; } rate_kernels[] = {
      { "zfp_encode1d_float_r8",  &zfp_metal_ctx.encode1d_float_r8_ps },
      { "zfp_decode1d_float_r8",  &zfp_metal_ctx.decode1d_float_r8_ps },
      { "zfp_encode1d_float_r16", &zfp_metal_ctx.encode1d_float_r16_ps },
      { "zfp_decode1d_float_r16", &zfp_metal_ctx.decode1d_float_r16_ps },
      { "zfp_encode1d_float_r32", &zfp_metal_ctx.encode1d_float_r32_ps },
      { "zfp_decode1d_float_r32", &zfp_metal_ctx.decode1d_float_r32_ps },
      { "zfp_encode2d_float_r8",  &zfp_metal_ctx.encode2d_float_r8_ps },
      { "zfp_decode2d_float_r8",  &zfp_metal_ctx.decode2d_float_r8_ps },
      { "zfp_encode2d_float_r16", &zfp_metal_ctx.encode2d_float_r16_ps },
      { "zfp_decode2d_float_r16", &zfp_metal_ctx.decode2d_float_r16_ps },
      { "zfp_encode2d_float_r32", &zfp_metal_ctx.encode2d_float_r32_ps },
      { "zfp_decode2d_float_r32", &zfp_metal_ctx.decode2d_float_r32_ps },
      { "zfp_encode3d_float_r8",  &zfp_metal_ctx.encode3d_float_r8_ps },
      { "zfp_decode3d_float_r8",  &zfp_metal_ctx.decode3d_float_r8_ps },
      { "zfp_encode3d_float_r16", &zfp_metal_ctx.encode3d_float_r16_ps },
      { "zfp_decode3d_float_r16", &zfp_metal_ctx.decode3d_float_r16_ps },
      { "zfp_encode3d_float_r32", &zfp_metal_ctx.encode3d_float_r32_ps },
      { "zfp_decode3d_float_r32", &zfp_metal_ctx.decode3d_float_r32_ps },
    };
    for (size_t i = 0; i < sizeof(rate_kernels) / sizeof(rate_kernels[0]); ++i) {
      id<MTLFunction> fn = [lib newFunctionWithName:
        [NSString stringWithUTF8String:rate_kernels[i].name]];
      if (fn)
        *(rate_kernels[i].ps) = [zfp_metal_ctx.device
          newComputePipelineStateWithFunction:fn error:&err];
    }
  }

  /* Threadgroup-prefetch 3D decode PSO (Phase B optimization) */
  {
    id<MTLFunction> fn = [lib newFunctionWithName:@"zfp_decode3d_float_tg"];
    if (fn)
      zfp_metal_ctx.decode3d_float_tg_ps = [zfp_metal_ctx.device
        newComputePipelineStateWithFunction:fn error:&err];
  }

  /* Int32 codec PSOs */
  {
    struct { const char* name; id<MTLComputePipelineState>* ps; } int32_kernels[] = {
      { "zfp_encode1d_int32", &zfp_metal_ctx.encode1d_int32_ps },
      { "zfp_decode1d_int32", &zfp_metal_ctx.decode1d_int32_ps },
      { "zfp_encode2d_int32", &zfp_metal_ctx.encode2d_int32_ps },
      { "zfp_decode2d_int32", &zfp_metal_ctx.decode2d_int32_ps },
      { "zfp_encode3d_int32", &zfp_metal_ctx.encode3d_int32_ps },
      { "zfp_decode3d_int32", &zfp_metal_ctx.decode3d_int32_ps },
    };
    for (size_t i = 0; i < sizeof(int32_kernels) / sizeof(int32_kernels[0]); ++i) {
      id<MTLFunction> fn = [lib newFunctionWithName:
        [NSString stringWithUTF8String:int32_kernels[i].name]];
      if (fn)
        *(int32_kernels[i].ps) = [zfp_metal_ctx.device
          newComputePipelineStateWithFunction:fn error:&err];
    }
  }

  /* Int64 codec PSOs */
  {
    struct { const char* name; id<MTLComputePipelineState>* ps; } int64_kernels[] = {
      { "zfp_encode1d_int64", &zfp_metal_ctx.encode1d_int64_ps },
      { "zfp_decode1d_int64", &zfp_metal_ctx.decode1d_int64_ps },
      { "zfp_encode2d_int64", &zfp_metal_ctx.encode2d_int64_ps },
      { "zfp_decode2d_int64", &zfp_metal_ctx.decode2d_int64_ps },
      { "zfp_encode3d_int64", &zfp_metal_ctx.encode3d_int64_ps },
      { "zfp_decode3d_int64", &zfp_metal_ctx.decode3d_int64_ps },
    };
    for (size_t i = 0; i < sizeof(int64_kernels) / sizeof(int64_kernels[0]); ++i) {
      id<MTLFunction> fn = [lib newFunctionWithName:
        [NSString stringWithUTF8String:int64_kernels[i].name]];
      if (fn)
        *(int64_kernels[i].ps) = [zfp_metal_ctx.device
          newComputePipelineStateWithFunction:fn error:&err];
    }
  }

  /* Diagnostic profiling PSOs (optional, loaded only when ZFP_METAL_PROFILE) */
  if (zfp_metal_profile()) {
    struct { const char* name; id<MTLComputePipelineState>* ps; } diag_kernels[] = {
      { "zfp_diag_bitplane_only_3d",  &zfp_metal_ctx.diag_bitplane_only_3d_ps },
      { "zfp_diag_transform_only_3d", &zfp_metal_ctx.diag_transform_only_3d_ps },
      { "zfp_diag_memcopy_3d",        &zfp_metal_ctx.diag_memcopy_3d_ps },
      { "zfp_diag_bitplane_devub_3d", &zfp_metal_ctx.diag_bitplane_devub_3d_ps },
      { "zfp_diag_bitread_only_3d",   &zfp_metal_ctx.diag_bitread_only_3d_ps },
      { "zfp_diag_bitplane_tgub_3d",  &zfp_metal_ctx.diag_bitplane_tgub_3d_ps },
      { "zfp_diag_bitplane_split32_3d", &zfp_metal_ctx.diag_bitplane_split32_3d_ps },
      { "zfp_decode3d_float_tgub",    &zfp_metal_ctx.decode3d_float_tgub_ps },
    };
    for (size_t i = 0; i < sizeof(diag_kernels) / sizeof(diag_kernels[0]); ++i) {
      id<MTLFunction> fn = [lib newFunctionWithName:
        [NSString stringWithUTF8String:diag_kernels[i].name]];
      if (fn)
        *(diag_kernels[i].ps) = [zfp_metal_ctx.device
          newComputePipelineStateWithFunction:fn error:&err];
    }
  }

  /* params_buf no longer needed — codec launchers use setBytes for inline
     params (all param structs are <4KB, well within Metal's inline limit). */

  return zfp_metal_ctx.pack_ps && zfp_metal_ctx.unpack_ps &&
         zfp_metal_ctx.encode1_ps && zfp_metal_ctx.decode1_ps &&
         zfp_metal_ctx.encode2_ps && zfp_metal_ctx.decode2_ps &&
         zfp_metal_ctx.encode3_ps && zfp_metal_ctx.decode3_ps &&
         zfp_metal_ctx.encode1d_float_ps && zfp_metal_ctx.decode1d_float_ps &&
         zfp_metal_ctx.encode2d_float_ps && zfp_metal_ctx.decode2d_float_ps &&
         zfp_metal_ctx.encode3d_float_ps && zfp_metal_ctx.decode3d_float_ps;
}

static size_t
zfp_calc_stream_bytes_2d(unsigned int nx, unsigned int ny, unsigned int maxbits)
{
  size_t px = (nx + 3u) & ~3u;
  size_t py = (ny + 3u) & ~3u;
  size_t blocks = (px * py) / 16u;
  size_t total_bits = blocks * (size_t)maxbits;
  size_t words = total_bits / 64u;
  if (total_bits % 64u)
    words++;
  return words * sizeof(unsigned long long);
}

static size_t
zfp_calc_stream_bytes_3d(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int maxbits)
{
  size_t px = (nx + 3u) & ~3u;
  size_t py = (ny + 3u) & ~3u;
  size_t pz = (nz + 3u) & ~3u;
  size_t blocks = (px * py * pz) / 64u;
  size_t total_bits = blocks * (size_t)maxbits;
  size_t words = total_bits / 64u;
  if (total_bits % 64u)
    words++;
  return words * sizeof(unsigned long long);
}

static size_t
zfp_calc_stream_bytes_1d(unsigned int dim, unsigned int maxbits)
{
  const size_t vals_per_block = 4;
  size_t total_blocks = dim / vals_per_block;
  if (dim % vals_per_block)
    total_blocks++;
  size_t total_bits = total_blocks * (size_t)maxbits;
  size_t words = total_bits / 64u;
  if (total_bits % 64u)
    words++;
  return words * sizeof(unsigned long long);
}

static int
zfp_metal_launch_codec1d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec1dParams* params,
                         size_t fill_dst_bytes,
                         size_t src_offset_bytes,
                         size_t dst_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!pso)
    return 0;

  id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];

  if (fill_dst_bytes > 0) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst_buf range:NSMakeRange(0, fill_dst_bytes) value:0];
    [blit endEncoding];
  }

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:src_offset_bytes atIndex:0];
  [enc setBuffer:dst_buf offset:dst_offset_bytes atIndex:1];
  [enc setBytes:params length:sizeof(ZfpMetalCodec1dParams) atIndex:2];

  NSUInteger width = pso.threadExecutionWidth ? pso.threadExecutionWidth : 64;
  NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
  NSUInteger group_size = width * 2u;
  if (tg && group_size > tg)
    group_size = tg;
  if (group_size < width)
    group_size = width;
  MTLSize grid = MTLSizeMake((NSUInteger)params->total_blocks, 1, 1);
  MTLSize group = MTLSizeMake(group_size, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:group];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  zfp_metal_report_gpu_time(cb, fill_dst_bytes ? "encode1d" : "decode1d",
                            params->total_blocks,
                            (size_t)params->total_blocks * 4u * sizeof(float));
  return [cb status] == MTLCommandBufferStatusCompleted;
}

/* Rate-specialized PSO selection.
   maxbits = rate * block_size, where block_size = 4^dims.
   Kernel names r8/r16/r32 refer to rate (bits per value).
   1D block = 4 values:  rate 8->maxbits 32, rate 16->64, rate 32->128
   2D block = 16 values: rate 8->128, rate 16->256, rate 32->512
   3D block = 64 values: rate 8->512, rate 16->1024, rate 32->2048 */

static id<MTLComputePipelineState>
zfp_metal_select_1d_encode_pso(unsigned int maxbits)
{
  if (maxbits == 32u && zfp_metal_ctx.encode1d_float_r8_ps)
    return zfp_metal_ctx.encode1d_float_r8_ps;
  if (maxbits == 64u && zfp_metal_ctx.encode1d_float_r16_ps)
    return zfp_metal_ctx.encode1d_float_r16_ps;
  if (maxbits == 128u && zfp_metal_ctx.encode1d_float_r32_ps)
    return zfp_metal_ctx.encode1d_float_r32_ps;
  return zfp_metal_ctx.encode1d_float_ps;
}

static id<MTLComputePipelineState>
zfp_metal_select_1d_decode_pso(unsigned int maxbits)
{
  if (maxbits == 32u && zfp_metal_ctx.decode1d_float_r8_ps)
    return zfp_metal_ctx.decode1d_float_r8_ps;
  if (maxbits == 64u && zfp_metal_ctx.decode1d_float_r16_ps)
    return zfp_metal_ctx.decode1d_float_r16_ps;
  if (maxbits == 128u && zfp_metal_ctx.decode1d_float_r32_ps)
    return zfp_metal_ctx.decode1d_float_r32_ps;
  return zfp_metal_ctx.decode1d_float_ps;
}

static id<MTLComputePipelineState>
zfp_metal_select_2d_encode_pso(unsigned int maxbits)
{
  if (maxbits == 128u && zfp_metal_ctx.encode2d_float_r8_ps)
    return zfp_metal_ctx.encode2d_float_r8_ps;
  if (maxbits == 256u && zfp_metal_ctx.encode2d_float_r16_ps)
    return zfp_metal_ctx.encode2d_float_r16_ps;
  if (maxbits == 512u && zfp_metal_ctx.encode2d_float_r32_ps)
    return zfp_metal_ctx.encode2d_float_r32_ps;
  return zfp_metal_ctx.encode2d_float_ps;
}

static id<MTLComputePipelineState>
zfp_metal_select_2d_decode_pso(unsigned int maxbits)
{
  if (maxbits == 128u && zfp_metal_ctx.decode2d_float_r8_ps)
    return zfp_metal_ctx.decode2d_float_r8_ps;
  if (maxbits == 256u && zfp_metal_ctx.decode2d_float_r16_ps)
    return zfp_metal_ctx.decode2d_float_r16_ps;
  if (maxbits == 512u && zfp_metal_ctx.decode2d_float_r32_ps)
    return zfp_metal_ctx.decode2d_float_r32_ps;
  return zfp_metal_ctx.decode2d_float_ps;
}

static id<MTLComputePipelineState>
zfp_metal_select_3d_encode_pso(unsigned int maxbits)
{
  /* Rate-specialized 3D PSOs hurt performance (register pressure causes
     occupancy drop), so always use the generic kernel for 3D. */
  (void)maxbits;
  return zfp_metal_ctx.encode3d_float_ps;
}

static id<MTLComputePipelineState>
zfp_metal_select_3d_decode_pso(unsigned int maxbits)
{
  /* Phase B experiment: threadgroup-prefetch kernel was tested but showed
     -24% regression on Apple Silicon unified memory. The cooperative load
     into threadgroup SRAM adds overhead on UMA where L2 cache already
     provides the fast path. Keep the TG kernel compiled but disabled. */
  (void)maxbits;
  return zfp_metal_ctx.decode3d_float_ps;
}

/* ========================================================================== */
/* Int32 runtime encode/decode functions                                      */
/* ========================================================================== */

extern "C" size_t
zfp_metal_encode1d_int32_runtime(const int* src,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes,
                                 size_t data_span_bytes,
                                 size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.encode1d_int32_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_1d(dim, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int padded = dim;
  if (padded % 4u)
    padded += 4u - (padded % 4u);
  unsigned int blocks = padded / 4u;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                     length:data_span_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                     length:stream_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec1dParams params;
  params.dim = dim;
  params.sx = sx;
  params.maxbits = maxbits;
  params.padded_dim = padded;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.encode1d_int32_ps, src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode1d_int32_runtime(const void* stream_words,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 int* dst,
                                 size_t data_span_bytes,
                                 size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.decode1d_int32_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_1d(dim, maxbits);

  unsigned int padded = dim;
  if (padded % 4u)
    padded += 4u - (padded % 4u);
  unsigned int blocks = padded / 4u;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                     length:stream_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                     length:data_span_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec1dParams params;
  params.dim = dim;
  params.sx = sx;
  params.maxbits = maxbits;
  params.padded_dim = padded;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.decode1d_int32_ps, src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

/* Forward declarations for 2D/3D launch helpers (defined later in file). */
static int
zfp_metal_launch_codec2d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec2dParams* params,
                         size_t fill_dst_bytes,
                         size_t src_offset_bytes,
                         size_t dst_offset_bytes);

static int
zfp_metal_launch_codec3d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec3dParams* params,
                         size_t fill_dst_bytes,
                         size_t src_offset_bytes,
                         size_t dst_offset_bytes);

extern "C" size_t
zfp_metal_encode2d_int32_runtime(const int* src,
                                 unsigned int nx,
                                 unsigned int ny,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes,
                                 size_t data_span_bytes,
                                 size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.encode2d_int32_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_2d(nx, ny, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int blocks = bx * by;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                     length:data_span_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                     length:stream_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec2dParams params;
  params.nx = nx;
  params.ny = ny;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.encode2d_int32_ps, src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode2d_int32_runtime(const void* stream_words,
                                 unsigned int nx,
                                 unsigned int ny,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 unsigned int maxbits,
                                 int* dst,
                                 size_t data_span_bytes,
                                 size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.decode2d_int32_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_2d(nx, ny, maxbits);

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int blocks = bx * by;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                     length:stream_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                     length:data_span_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec2dParams params;
  params.nx = nx;
  params.ny = ny;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.decode2d_int32_ps, src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode3d_int32_runtime(const int* src,
                                 unsigned int nx,
                                 unsigned int ny,
                                 unsigned int nz,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 ptrdiff_t sz,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes,
                                 size_t data_span_bytes,
                                 size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.encode3d_int32_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_3d(nx, ny, nz, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int pz = (nz + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int bz = pz / 4u;
  unsigned int blocks = bx * by * bz;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                     length:data_span_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                     length:stream_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec3dParams params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.sz = (long)sz;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.bz = bz;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.encode3d_int32_ps, src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode3d_int32_runtime(const void* stream_words,
                                 unsigned int nx,
                                 unsigned int ny,
                                 unsigned int nz,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 ptrdiff_t sz,
                                 unsigned int maxbits,
                                 int* dst,
                                 size_t data_span_bytes,
                                 size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.decode3d_int32_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_3d(nx, ny, nz, maxbits);
  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int pz = (nz + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int bz = pz / 4u;
  unsigned int blocks = bx * by * bz;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                     length:stream_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                     length:data_span_bytes
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec3dParams params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.sz = (long)sz;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.bz = bz;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.decode3d_int32_ps, src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

/* ========================================================================== */
/* Int64 runtime encode/decode functions                                      */
/* ========================================================================== */

extern "C" size_t
zfp_metal_encode1d_int64_runtime(const long* src,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.encode1d_int64_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_1d(dim, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int padded = dim;
  if (padded % 4u)
    padded += 4u - (padded % 4u);
  unsigned int blocks = padded / 4u;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                      length:data_span_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                      length:stream_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec1dParams params;
  params.dim = dim;
  params.sx = sx;
  params.maxbits = maxbits;
  params.padded_dim = padded;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.encode1d_int64_ps, src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode1d_int64_runtime(const void* stream_words,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  long* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.decode1d_int64_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_1d(dim, maxbits);

  unsigned int padded = dim;
  if (padded % 4u)
    padded += 4u - (padded % 4u);
  unsigned int blocks = padded / 4u;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                      length:stream_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                      length:data_span_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec1dParams params;
  params.dim = dim;
  params.sx = sx;
  params.maxbits = maxbits;
  params.padded_dim = padded;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.decode1d_int64_ps, src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode2d_int64_runtime(const long* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.encode2d_int64_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_2d(nx, ny, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int blocks = bx * by;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                      length:data_span_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                      length:stream_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec2dParams params;
  params.nx = nx;
  params.ny = ny;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.encode2d_int64_ps, src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode2d_int64_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  long* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.decode2d_int64_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_2d(nx, ny, maxbits);

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int blocks = bx * by;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                      length:stream_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                      length:data_span_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec2dParams params;
  params.nx = nx;
  params.ny = ny;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.decode2d_int64_ps, src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode3d_int64_runtime(const long* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.encode3d_int64_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_3d(nx, ny, nz, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int pz = (nz + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int bz = pz / 4u;
  unsigned int blocks = bx * by * bz;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                      length:data_span_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                      length:stream_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec3dParams params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.sz = (long)sz;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.bz = bz;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.encode3d_int64_ps, src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode3d_int64_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  long* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!zfp_metal_ctx.decode3d_int64_ps)
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_3d(nx, ny, nz, maxbits);
  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int pz = (nz + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int bz = pz / 4u;
  unsigned int blocks = bx * by * bz;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                      length:stream_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                      length:data_span_bytes
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec3dParams params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.sz = (long)sz;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.bz = bz;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.decode3d_int64_ps, src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode1d_float_runtime(const float* src,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_1d(dim, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int padded = dim;
  if (padded % 4u)
    padded += 4u - (padded % 4u);
  unsigned int blocks = padded / 4u;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                    length:data_span_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                    length:stream_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec1dParams params;
  params.dim = dim;
  params.sx = sx;
  params.maxbits = maxbits;
  params.padded_dim = padded;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec1d(zfp_metal_select_1d_encode_pso(maxbits), src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode1d_float_runtime(const void* stream_words,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  float* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_1d(dim, maxbits);

  unsigned int padded = dim;
  if (padded % 4u)
    padded += 4u - (padded % 4u);
  unsigned int blocks = padded / 4u;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                    length:stream_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                    length:data_span_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec1dParams params;
  params.dim = dim;
  params.sx = sx;
  params.maxbits = maxbits;
  params.padded_dim = padded;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec1d(zfp_metal_select_1d_decode_pso(maxbits), src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

static int
zfp_metal_launch_codec2d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec2dParams* params,
                         size_t fill_dst_bytes,
                         size_t src_offset_bytes,
                         size_t dst_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!pso)
    return 0;

  id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];

  if (fill_dst_bytes > 0) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst_buf range:NSMakeRange(0, fill_dst_bytes) value:0];
    [blit endEncoding];
  }

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:src_offset_bytes atIndex:0];
  [enc setBuffer:dst_buf offset:dst_offset_bytes atIndex:1];
  [enc setBytes:params length:sizeof(ZfpMetalCodec2dParams) atIndex:2];

  NSUInteger width = pso.threadExecutionWidth ? pso.threadExecutionWidth : 64;
  NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
  NSUInteger group_size = width * 2u;
  if (tg && group_size > tg)
    group_size = tg;
  if (group_size < width)
    group_size = width;
  MTLSize grid = MTLSizeMake((NSUInteger)params->total_blocks, 1, 1);
  MTLSize group = MTLSizeMake(group_size, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:group];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  zfp_metal_report_gpu_time(cb, fill_dst_bytes ? "encode2d" : "decode2d",
                            params->total_blocks,
                            (size_t)params->total_blocks * 16u * sizeof(float));
  return [cb status] == MTLCommandBufferStatusCompleted;
}

static int
zfp_metal_launch_codec3d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec3dParams* params,
                         size_t fill_dst_bytes,
                         size_t src_offset_bytes,
                         size_t dst_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!pso)
    return 0;

  id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];

  if (fill_dst_bytes > 0) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst_buf range:NSMakeRange(0, fill_dst_bytes) value:0];
    [blit endEncoding];
  }

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  zfp_metal_report_pso_info(pso, "codec3d");
  [enc setBuffer:src_buf offset:src_offset_bytes atIndex:0];
  [enc setBuffer:dst_buf offset:dst_offset_bytes atIndex:1];
  [enc setBytes:params length:sizeof(ZfpMetalCodec3dParams) atIndex:2];

  /* Threadgroup size: 2 SIMD groups (width * 2). Tested 1 SIMD group for 3D
     to reduce register pressure but no measurable improvement on Apple GPU. */
  NSUInteger width = pso.threadExecutionWidth ? pso.threadExecutionWidth : 64;
  NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
  NSUInteger group_size = width * 2u;
  if (tg && group_size > tg)
    group_size = tg;
  if (group_size < width)
    group_size = width;
  MTLSize grid = MTLSizeMake((NSUInteger)params->total_blocks, 1, 1);
  MTLSize group = MTLSizeMake(group_size, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:group];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  zfp_metal_report_gpu_time(cb, fill_dst_bytes ? "encode3d" : "decode3d",
                            params->total_blocks,
                            (size_t)params->total_blocks * 64u * sizeof(float));
  return [cb status] == MTLCommandBufferStatusCompleted;
}

/* Diagnostic: run bitplane-only, transform-only, and memcopy kernels on the
   same data to isolate cost breakdown. Called once per process when
   ZFP_METAL_PROFILE >= 3. */
static int zfp_metal_diag_3d_done = 0;
static void
zfp_metal_run_diagnostic_3d(id<MTLBuffer> stream_buf,
                             const ZfpMetalCodec3dParams* params)
{
  if (zfp_metal_diag_3d_done || zfp_metal_profile() < 3)
    return;
  zfp_metal_diag_3d_done = 1;

  unsigned int total = params->total_blocks;
  size_t data_bytes = (size_t)total * 64u * sizeof(float);

  /* Temp buffer for uint intermediate (64 uints per block) */
  id<MTLBuffer> tmp_buf = [zfp_metal_ctx.device newBufferWithLength:data_bytes
                                                            options:MTLResourceStorageModeShared];
  if (!tmp_buf) {
    fprintf(stderr, "[Metal Diag] Failed to allocate temp buffer (%zu bytes)\n", data_bytes);
    return;
  }

  /* Dedicated output buffer so diagnostics never corrupt caller's data */
  id<MTLBuffer> diag_out_buf = [zfp_metal_ctx.device newBufferWithLength:data_bytes
                                                                 options:MTLResourceStorageModeShared];
  if (!diag_out_buf) {
    fprintf(stderr, "[Metal Diag] Failed to allocate diag output buffer (%zu bytes)\n", data_bytes);
    return;
  }

  struct { const char* name; id<MTLComputePipelineState> pso;
           id<MTLBuffer> src; id<MTLBuffer> dst; NSUInteger tg_per_thread; } tests[] = {
    { "diag:memcopy_3d",        zfp_metal_ctx.diag_memcopy_3d_ps,        tmp_buf, diag_out_buf, 0 },
    { "diag:transform_only_3d", zfp_metal_ctx.diag_transform_only_3d_ps, tmp_buf, diag_out_buf, 0 },
    { "diag:bitplane_only_3d",  zfp_metal_ctx.diag_bitplane_only_3d_ps,  stream_buf, tmp_buf, 0 },
    { "diag:bitplane_devub_3d", zfp_metal_ctx.diag_bitplane_devub_3d_ps, stream_buf, tmp_buf, 0 },
    { "diag:bitread_only_3d",   zfp_metal_ctx.diag_bitread_only_3d_ps,   stream_buf, tmp_buf, 0 },
    { "diag:bitplane_tgub_3d",  zfp_metal_ctx.diag_bitplane_tgub_3d_ps,  stream_buf, tmp_buf, 256 },
    { "diag:bitplane_split32",  zfp_metal_ctx.diag_bitplane_split32_3d_ps, stream_buf, tmp_buf, 0 },
    { "diag:full_decode_tgub",  zfp_metal_ctx.decode3d_float_tgub_ps,    stream_buf, diag_out_buf, 256 },
    { "diag:full_decode_3d",    zfp_metal_ctx.decode3d_float_ps,         stream_buf, diag_out_buf, 0 },
  };

  fprintf(stderr, "[Metal Diag] Running diagnostic kernels (3D, %u blocks, %.1f MB)\n",
          total, (double)data_bytes / (1024.0*1024.0));

  /* params passed inline via setBytes */

  int num_tests = sizeof(tests) / sizeof(tests[0]);
  for (int t = 0; t < num_tests; ++t) {
    if (!tests[t].pso) {
      fprintf(stderr, "[Metal Diag] %-30s  SKIPPED (no PSO)\n", tests[t].name);
      continue;
    }
    /* Warm up */
    for (int w = 0; w < 2; ++w) {
      id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:tests[t].pso];
      [enc setBuffer:tests[t].src offset:0 atIndex:0];
      [enc setBuffer:tests[t].dst offset:0 atIndex:1];
      [enc setBytes:params length:sizeof(ZfpMetalCodec3dParams) atIndex:2];
      NSUInteger width = tests[t].pso.threadExecutionWidth ?: 64;
      NSUInteger gs = width * 2u;
      NSUInteger maxt = tests[t].pso.maxTotalThreadsPerThreadgroup;
      if (maxt && gs > maxt) gs = maxt;
      if (tests[t].tg_per_thread > 0)
        [enc setThreadgroupMemoryLength:gs * tests[t].tg_per_thread atIndex:0];
      [enc dispatchThreads:MTLSizeMake(total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(gs, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
    /* Measure (5 runs, report min) */
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
      id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:tests[t].pso];
      [enc setBuffer:tests[t].src offset:0 atIndex:0];
      [enc setBuffer:tests[t].dst offset:0 atIndex:1];
      [enc setBytes:params length:sizeof(ZfpMetalCodec3dParams) atIndex:2];
      NSUInteger width = tests[t].pso.threadExecutionWidth ?: 64;
      NSUInteger gs = width * 2u;
      NSUInteger maxt = tests[t].pso.maxTotalThreadsPerThreadgroup;
      if (maxt && gs > maxt) gs = maxt;
      if (tests[t].tg_per_thread > 0)
        [enc setThreadgroupMemoryLength:gs * tests[t].tg_per_thread atIndex:0];
      [enc dispatchThreads:MTLSizeMake(total, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(gs, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      double gpu_s = cb.GPUEndTime - cb.GPUStartTime;
      if (gpu_s < best) best = gpu_s;
    }
    double gb_s = (data_bytes > 0 && best > 0.0)
      ? ((double)data_bytes / (1024.0*1024.0*1024.0)) / best : 0.0;
    fprintf(stderr, "[Metal Diag] %-30s  gpu=%.6f s  %.2f GB/s  maxThr=%lu\n",
            tests[t].name, best, gb_s,
            (unsigned long)tests[t].pso.maxTotalThreadsPerThreadgroup);
  }
  fprintf(stderr, "[Metal Diag] Done.\n");
}

extern "C" size_t
zfp_metal_encode2d_float_runtime(const float* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_2d(nx, ny, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int blocks = bx * by;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                    length:data_span_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                    length:stream_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec2dParams params;
  params.nx = nx;
  params.ny = ny;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec2d(zfp_metal_select_2d_encode_pso(maxbits), src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode2d_float_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  float* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_2d(nx, ny, maxbits);

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int blocks = bx * by;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                    length:stream_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                    length:data_span_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec2dParams params;
  params.nx = nx;
  params.ny = ny;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec2d(zfp_metal_select_2d_decode_pso(maxbits), src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode3d_float_runtime(const float* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_3d(nx, ny, nz, maxbits);
  if (stream_capacity_bytes < stream_bytes)
    return 0;

  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int pz = (nz + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int bz = pz / 4u;
  unsigned int blocks = bx * by * bz;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                    length:data_span_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:stream_words
                                                                    length:stream_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec3dParams params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.sz = (long)sz;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.bz = bz;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec3d(zfp_metal_select_3d_encode_pso(maxbits), src_buf, dst_buf, &params, stream_bytes, data_offset_bytes, 0))
    return 0;

  /* Run diagnostic breakdown after first encode (uses compressed output) */
  zfp_metal_run_diagnostic_3d(dst_buf, &params);

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode3d_float_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  float* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  if (!zfp_metal_init_context())
    return 0;

  size_t stream_bytes = zfp_calc_stream_bytes_3d(nx, ny, nz, maxbits);
  unsigned int px = (nx + 3u) & ~3u;
  unsigned int py = (ny + 3u) & ~3u;
  unsigned int pz = (nz + 3u) & ~3u;
  unsigned int bx = px / 4u;
  unsigned int by = py / 4u;
  unsigned int bz = pz / 4u;
  unsigned int blocks = bx * by * bz;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)stream_words
                                                                    length:stream_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!src_buf)
    return 0;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                    length:data_span_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  if (!dst_buf)
    return 0;

  ZfpMetalCodec3dParams params;
  params.nx = nx;
  params.ny = ny;
  params.nz = nz;
  params.sx = (long)sx;
  params.sy = (long)sy;
  params.sz = (long)sz;
  params.maxbits = maxbits;
  params.bx = bx;
  params.by = by;
  params.bz = bz;
  params.total_blocks = blocks;

  if (!zfp_metal_launch_codec3d(zfp_metal_select_3d_decode_pso(maxbits), src_buf, dst_buf, &params, 0, 0, data_offset_bytes))
    return 0;

  /* Run diagnostic breakdown if ZFP_METAL_PROFILE >= 3 (first call only) */
  zfp_metal_run_diagnostic_3d(src_buf, &params);

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode1d_double_runtime(const double* src,
                                   unsigned int dim,
                                   int sx,
                                   unsigned int maxbits,
                                   void* stream_words,
                                   size_t stream_capacity_bytes,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  /* Metal does not support double-precision; fall through to CPU. */
  (void)src; (void)dim; (void)sx; (void)maxbits;
  (void)stream_words; (void)stream_capacity_bytes;
  (void)data_span_bytes; (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_double_runtime(const void* stream_words,
                                   unsigned int dim,
                                   int sx,
                                   unsigned int maxbits,
                                   double* dst,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)stream_words; (void)dim; (void)sx; (void)maxbits; (void)dst;
  (void)data_span_bytes; (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode2d_double_runtime(const double* src,
                                   unsigned int nx,
                                   unsigned int ny,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   unsigned int maxbits,
                                   void* stream_words,
                                   size_t stream_capacity_bytes,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)src; (void)nx; (void)ny; (void)sx; (void)sy; (void)maxbits;
  (void)stream_words; (void)stream_capacity_bytes;
  (void)data_span_bytes; (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode2d_double_runtime(const void* stream_words,
                                   unsigned int nx,
                                   unsigned int ny,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   unsigned int maxbits,
                                   double* dst,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)stream_words; (void)nx; (void)ny; (void)sx; (void)sy;
  (void)maxbits; (void)dst;
  (void)data_span_bytes; (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode3d_double_runtime(const double* src,
                                   unsigned int nx,
                                   unsigned int ny,
                                   unsigned int nz,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   ptrdiff_t sz,
                                   unsigned int maxbits,
                                   void* stream_words,
                                   size_t stream_capacity_bytes,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)src; (void)nx; (void)ny; (void)nz; (void)sx; (void)sy; (void)sz;
  (void)maxbits; (void)stream_words; (void)stream_capacity_bytes;
  (void)data_span_bytes; (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode3d_double_runtime(const void* stream_words,
                                   unsigned int nx,
                                   unsigned int ny,
                                   unsigned int nz,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   ptrdiff_t sz,
                                   unsigned int maxbits,
                                   double* dst,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)stream_words; (void)nx; (void)ny; (void)nz;
  (void)sx; (void)sy; (void)sz; (void)maxbits; (void)dst;
  (void)data_span_bytes; (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_host_encode2d_float(void* zfp,
                              const float* src,
                              unsigned int nx,
                              unsigned int ny,
                              ptrdiff_t sx,
                              ptrdiff_t sy,
                              unsigned int maxbits,
                              void* stream_words,
                              size_t stream_capacity_bytes)
{
  (void)zfp;
  (void)src;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  return 0;
}

extern "C" int
zfp_metal_host_decode2d_float(void* zfp,
                              const void* stream_words,
                              size_t stream_bytes,
                              float* dst,
                              unsigned int nx,
                              unsigned int ny,
                              ptrdiff_t sx,
                              ptrdiff_t sy,
                              unsigned int maxbits)
{
  (void)zfp;
  (void)stream_words;
  (void)stream_bytes;
  (void)dst;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  return 0;
}

static int
zfp_metal_launch_layout(id<MTLComputePipelineState> pso,
                        const void* src,
                        size_t src_bytes,
                        void* dst,
                        size_t dst_bytes,
                        const ZfpMetalLayoutParams* params)
{
  id<MTLCommandBuffer> cb;
  id<MTLComputeCommandEncoder> enc;
  NSUInteger width;
  MTLSize grid;
  MTLSize group;

  if (!zfp_metal_init_context())
    return 0;
  if (!pso)
    return 0;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                    length:src_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                    length:dst_bytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];

  if (!src_buf || !dst_buf) {
    if (!zfp_metal_ensure_buffer(zfp_metal_ctx.device, &zfp_metal_ctx.src_buf, &zfp_metal_ctx.src_cap, src_bytes)) {
      zfp_metal_log_error_once("src buffer allocation", nil);
      return 0;
    }
    if (!zfp_metal_ensure_buffer(zfp_metal_ctx.device, &zfp_metal_ctx.dst_buf, &zfp_metal_ctx.dst_cap, dst_bytes)) {
      zfp_metal_log_error_once("dst buffer allocation", nil);
      return 0;
    }
    memcpy([zfp_metal_ctx.src_buf contents], src, src_bytes);
    src_buf = zfp_metal_ctx.src_buf;
    dst_buf = zfp_metal_ctx.dst_buf;
  }

  cb = [zfp_metal_ctx.queue commandBuffer];
  enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:0 atIndex:0];
  [enc setBuffer:dst_buf offset:0 atIndex:1];
  [enc setBytes:params length:sizeof(ZfpMetalLayoutParams) atIndex:2];

  width = pso.threadExecutionWidth;
  if (!width)
    width = 64;

  grid = MTLSizeMake((NSUInteger)params->total, 1, 1);
  group = MTLSizeMake(width, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:group];
  [enc endEncoding];

  [cb commit];
  [cb waitUntilCompleted];
  if ([cb status] != MTLCommandBufferStatusCompleted) {
    zfp_metal_log_error_once("command buffer execution", nil);
    return 0;
  }

  if (dst_buf == zfp_metal_ctx.dst_buf)
    memcpy(dst, [zfp_metal_ctx.dst_buf contents], dst_bytes);
  return 1;
}

extern "C" int
zfp_metal_pack_runtime(const void* src,
                       size_t src_bytes,
                       void* packed,
                       const unsigned int dims[3],
                       const ptrdiff_t stride[3],
                       long long int offset,
                       size_t item_size)
{
  ZfpMetalLayoutParams params;
  params.nx = dims[0] ? dims[0] : 1;
  params.ny = dims[1] ? dims[1] : 1;
  params.nz = dims[2] ? dims[2] : 1;
  params.sx = (long)stride[0];
  params.sy = (long)stride[1];
  params.sz = (long)stride[2];
  params.elem_size = (unsigned long)item_size;
  params.total = (unsigned long)(params.nx * params.ny * params.nz);
  params.offset = (long)offset;
  return zfp_metal_launch_layout(zfp_metal_ctx.pack_ps, src, src_bytes, packed,
                                 (size_t)params.total * item_size, &params);
}

extern "C" int
zfp_metal_unpack_runtime(const void* packed,
                         size_t packed_bytes,
                         void* dst,
                         size_t dst_bytes,
                         const unsigned int dims[3],
                         const ptrdiff_t stride[3],
                         long long int offset,
                         size_t item_size)
{
  ZfpMetalLayoutParams params;
  params.nx = dims[0] ? dims[0] : 1;
  params.ny = dims[1] ? dims[1] : 1;
  params.nz = dims[2] ? dims[2] : 1;
  params.sx = (long)stride[0];
  params.sy = (long)stride[1];
  params.sz = (long)stride[2];
  params.elem_size = (unsigned long)item_size;
  params.total = (unsigned long)(params.nx * params.ny * params.nz);
  params.offset = (long)offset;
  return zfp_metal_launch_layout(zfp_metal_ctx.unpack_ps, packed, packed_bytes, dst,
                                 dst_bytes, &params);
}

static id<MTLComputePipelineState>
zfp_metal_pick_contig_pipeline(int encode, int dim)
{
  if (encode) {
    if (dim == 1) return zfp_metal_ctx.encode1_ps;
    if (dim == 2) return zfp_metal_ctx.encode2_ps;
    return zfp_metal_ctx.encode3_ps;
  }
  else {
    if (dim == 1) return zfp_metal_ctx.decode1_ps;
    if (dim == 2) return zfp_metal_ctx.decode2_ps;
    return zfp_metal_ctx.decode3_ps;
  }
}

extern "C" int
zfp_metal_encode_contiguous_runtime(const void* src,
                                    size_t src_bytes,
                                    void* dst,
                                    size_t dst_bytes,
                                    size_t item_size,
                                    int dim)
{
  ZfpMetalLayoutParams params;
  unsigned long total = src_bytes / item_size;
  params.nx = dim == 1 ? (unsigned int)total : 1;
  params.ny = 1;
  params.nz = 1;
  params.sx = 1;
  params.sy = 1;
  params.sz = 1;
  params.elem_size = (unsigned long)item_size;
  params.total = total;
  params.offset = 0;
  return zfp_metal_launch_layout(zfp_metal_pick_contig_pipeline(1, dim), src, src_bytes, dst, dst_bytes, &params);
}

extern "C" int
zfp_metal_decode_contiguous_runtime(const void* src,
                                    size_t src_bytes,
                                    void* dst,
                                    size_t dst_bytes,
                                    size_t item_size,
                                    int dim)
{
  ZfpMetalLayoutParams params;
  unsigned long total = dst_bytes / item_size;
  params.nx = dim == 1 ? (unsigned int)total : 1;
  params.ny = 1;
  params.nz = 1;
  params.sx = 1;
  params.sy = 1;
  params.sz = 1;
  params.elem_size = (unsigned long)item_size;
  params.total = total;
  params.offset = 0;
  return zfp_metal_launch_layout(zfp_metal_pick_contig_pipeline(0, dim), src, src_bytes, dst, dst_bytes, &params);
}

#else

#include "metal_runtime.h"

extern "C" int
zfp_metal_pack_runtime(const void* src,
                       size_t src_bytes,
                       void* packed,
                       const unsigned int dims[3],
                       const ptrdiff_t stride[3],
                       long long int offset,
                       size_t item_size)
{
  (void)src;
  (void)src_bytes;
  (void)packed;
  (void)dims;
  (void)stride;
  (void)offset;
  (void)item_size;
  return 0;
}

extern "C" int
zfp_metal_unpack_runtime(const void* packed,
                         size_t packed_bytes,
                         void* dst,
                         size_t dst_bytes,
                         const unsigned int dims[3],
                         const ptrdiff_t stride[3],
                         long long int offset,
                         size_t item_size)
{
  (void)packed;
  (void)packed_bytes;
  (void)dst;
  (void)dst_bytes;
  (void)dims;
  (void)stride;
  (void)offset;
  (void)item_size;
  return 0;
}

extern "C" int
zfp_metal_encode_contiguous_runtime(const void* src,
                                    size_t src_bytes,
                                    void* dst,
                                    size_t dst_bytes,
                                    size_t item_size,
                                    int dim)
{
  (void)src;
  (void)src_bytes;
  (void)dst;
  (void)dst_bytes;
  (void)item_size;
  (void)dim;
  return 0;
}

extern "C" int
zfp_metal_decode_contiguous_runtime(const void* src,
                                    size_t src_bytes,
                                    void* dst,
                                    size_t dst_bytes,
                                    size_t item_size,
                                    int dim)
{
  (void)src;
  (void)src_bytes;
  (void)dst;
  (void)dst_bytes;
  (void)item_size;
  (void)dim;
  return 0;
}

extern "C" size_t
zfp_metal_encode1d_float_runtime(const float* src,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_float_runtime(const void* stream_words,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  float* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode2d_float_runtime(const float* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode2d_float_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  float* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode3d_float_runtime(const float* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode3d_float_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  float* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode1d_double_runtime(const double* src,
                                   unsigned int dim,
                                   int sx,
                                   unsigned int maxbits,
                                   void* stream_words,
                                   size_t stream_capacity_bytes,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_double_runtime(const void* stream_words,
                                   unsigned int dim,
                                   int sx,
                                   unsigned int maxbits,
                                   double* dst,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode2d_double_runtime(const double* src,
                                   unsigned int nx,
                                   unsigned int ny,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   unsigned int maxbits,
                                   void* stream_words,
                                   size_t stream_capacity_bytes,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode2d_double_runtime(const void* stream_words,
                                   unsigned int nx,
                                   unsigned int ny,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   unsigned int maxbits,
                                   double* dst,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode3d_double_runtime(const double* src,
                                   unsigned int nx,
                                   unsigned int ny,
                                   unsigned int nz,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   ptrdiff_t sz,
                                   unsigned int maxbits,
                                   void* stream_words,
                                   size_t stream_capacity_bytes,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode3d_double_runtime(const void* stream_words,
                                   unsigned int nx,
                                   unsigned int ny,
                                   unsigned int nz,
                                   ptrdiff_t sx,
                                   ptrdiff_t sy,
                                   ptrdiff_t sz,
                                   unsigned int maxbits,
                                   double* dst,
                                   size_t data_span_bytes,
                                   size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode1d_int32_runtime(const int* src,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_int32_runtime(const void* stream_words,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  int* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode2d_int32_runtime(const int* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode2d_int32_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  int* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode3d_int32_runtime(const int* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode3d_int32_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  int* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode1d_int64_runtime(const long* src,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_int64_runtime(const void* stream_words,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  long* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode2d_int64_runtime(const long* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode2d_int64_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  long* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_encode3d_int64_runtime(const long* src,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)src;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode3d_int64_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  unsigned int nz,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  ptrdiff_t sz,
                                  unsigned int maxbits,
                                  long* dst,
                                  size_t data_span_bytes,
                                  size_t data_offset_bytes)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)nz;
  (void)sx;
  (void)sy;
  (void)sz;
  (void)maxbits;
  (void)dst;
  (void)data_span_bytes;
  (void)data_offset_bytes;
  return 0;
}

#endif
