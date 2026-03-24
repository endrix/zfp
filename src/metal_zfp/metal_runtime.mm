#ifdef ZFP_WITH_METAL_NATIVE

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdio.h>
#include <string.h>
#include <stddef.h>

#include "metal_runtime.h"

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
  nil, nil, nil, nil,
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

  if (!zfp_metal_ensure_buffer(zfp_metal_ctx.device, &zfp_metal_ctx.params_buf, &zfp_metal_ctx.params_cap, sizeof(ZfpMetalLayoutParams))) {
    zfp_metal_log_error_once("params buffer allocation", nil);
    return 0;
  }

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
                         size_t fill_dst_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!pso)
    return 0;

  if (!zfp_metal_ensure_buffer(zfp_metal_ctx.device, &zfp_metal_ctx.params_buf,
                               &zfp_metal_ctx.params_cap, sizeof(ZfpMetalCodec1dParams)))
    return 0;
  memcpy([zfp_metal_ctx.params_buf contents], params, sizeof(ZfpMetalCodec1dParams));

  id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];

  if (fill_dst_bytes > 0) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst_buf range:NSMakeRange(0, fill_dst_bytes) value:0];
    [blit endEncoding];
  }

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:0 atIndex:0];
  [enc setBuffer:dst_buf offset:0 atIndex:1];
  [enc setBuffer:zfp_metal_ctx.params_buf offset:0 atIndex:2];

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
                                 size_t stream_capacity_bytes)
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
                                                                    length:(size_t)dim * (size_t)(sx > 0 ? sx : -sx) * sizeof(int)
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

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.encode1d_int32_ps, src_buf, dst_buf, &params, stream_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode1d_int32_runtime(const void* stream_words,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 int* dst)
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
                                                                    length:(size_t)dim * (size_t)(sx > 0 ? sx : -sx) * sizeof(int)
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

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.decode1d_int32_ps, src_buf, dst_buf, &params, 0))
    return 0;

  return stream_bytes;
}

/* Forward declarations for 2D/3D launch helpers (defined later in file). */
static int
zfp_metal_launch_codec2d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec2dParams* params,
                         size_t fill_dst_bytes);

static int
zfp_metal_launch_codec3d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec3dParams* params,
                         size_t fill_dst_bytes);

extern "C" size_t
zfp_metal_encode2d_int32_runtime(const int* src,
                                 unsigned int nx,
                                 unsigned int ny,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes)
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

  size_t src_len = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t row_span = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  if (row_span > src_len)
    src_len = row_span;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                    length:src_len * sizeof(int)
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

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.encode2d_int32_ps, src_buf, dst_buf, &params, stream_bytes))
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
                                 int* dst)
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

  size_t dst_len = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t row_span = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  if (row_span > dst_len)
    dst_len = row_span;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                    length:dst_len * sizeof(int)
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

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.decode2d_int32_ps, src_buf, dst_buf, &params, 0))
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
                                 size_t stream_capacity_bytes)
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

  size_t span = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t s2 = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  size_t s3 = (size_t)(sz > 0 ? sz : -sz) * (size_t)nz;
  if (s2 > span) span = s2;
  if (s3 > span) span = s3;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                    length:span * sizeof(int)
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

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.encode3d_int32_ps, src_buf, dst_buf, &params, stream_bytes))
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
                                 int* dst)
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

  size_t span = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t s2 = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  size_t s3 = (size_t)(sz > 0 ? sz : -sz) * (size_t)nz;
  if (s2 > span) span = s2;
  if (s3 > span) span = s3;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                    length:span * sizeof(int)
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

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.decode3d_int32_ps, src_buf, dst_buf, &params, 0))
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
                                 size_t stream_capacity_bytes)
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
                                                                     length:(size_t)dim * (size_t)(sx > 0 ? sx : -sx) * sizeof(long)
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

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.encode1d_int64_ps, src_buf, dst_buf, &params, stream_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode1d_int64_runtime(const void* stream_words,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 long* dst)
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
                                                                     length:(size_t)dim * (size_t)(sx > 0 ? sx : -sx) * sizeof(long)
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

  if (!zfp_metal_launch_codec1d(zfp_metal_ctx.decode1d_int64_ps, src_buf, dst_buf, &params, 0))
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
                                 size_t stream_capacity_bytes)
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

  size_t src_len = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t row_span = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  if (row_span > src_len)
    src_len = row_span;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                     length:src_len * sizeof(long)
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

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.encode2d_int64_ps, src_buf, dst_buf, &params, stream_bytes))
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
                                 long* dst)
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

  size_t span = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t row_span = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  if (row_span > span)
    span = row_span;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                     length:span * sizeof(long)
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

  if (!zfp_metal_launch_codec2d(zfp_metal_ctx.decode2d_int64_ps, src_buf, dst_buf, &params, 0))
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
                                 size_t stream_capacity_bytes)
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

  size_t span = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t s2 = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  size_t s3 = (size_t)(sz > 0 ? sz : -sz) * (size_t)nz;
  if (s2 > span) span = s2;
  if (s3 > span) span = s3;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                     length:span * sizeof(long)
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

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.encode3d_int64_ps, src_buf, dst_buf, &params, stream_bytes))
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
                                 long* dst)
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

  size_t span = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t s2 = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  size_t s3 = (size_t)(sz > 0 ? sz : -sz) * (size_t)nz;
  if (s2 > span) span = s2;
  if (s3 > span) span = s3;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                     length:span * sizeof(long)
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

  if (!zfp_metal_launch_codec3d(zfp_metal_ctx.decode3d_int64_ps, src_buf, dst_buf, &params, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode1d_float_runtime(const float* src,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes)
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
                                                                   length:(size_t)dim * (size_t)(sx > 0 ? sx : -sx) * sizeof(float)
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

  if (!zfp_metal_launch_codec1d(zfp_metal_select_1d_encode_pso(maxbits), src_buf, dst_buf, &params, stream_bytes))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_decode1d_float_runtime(const void* stream_words,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 float* dst)
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
                                                                   length:(size_t)dim * (size_t)(sx > 0 ? sx : -sx) * sizeof(float)
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

  if (!zfp_metal_launch_codec1d(zfp_metal_select_1d_decode_pso(maxbits), src_buf, dst_buf, &params, 0))
    return 0;

  return stream_bytes;
}

static int
zfp_metal_launch_codec2d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec2dParams* params,
                         size_t fill_dst_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!pso)
    return 0;

  if (!zfp_metal_ensure_buffer(zfp_metal_ctx.device, &zfp_metal_ctx.params_buf,
                               &zfp_metal_ctx.params_cap, sizeof(ZfpMetalCodec2dParams)))
    return 0;
  memcpy([zfp_metal_ctx.params_buf contents], params, sizeof(ZfpMetalCodec2dParams));

  id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];

  if (fill_dst_bytes > 0) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst_buf range:NSMakeRange(0, fill_dst_bytes) value:0];
    [blit endEncoding];
  }

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:0 atIndex:0];
  [enc setBuffer:dst_buf offset:0 atIndex:1];
  [enc setBuffer:zfp_metal_ctx.params_buf offset:0 atIndex:2];

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
  return [cb status] == MTLCommandBufferStatusCompleted;
}

static int
zfp_metal_launch_codec3d(id<MTLComputePipelineState> pso,
                         id<MTLBuffer> src_buf,
                         id<MTLBuffer> dst_buf,
                         const ZfpMetalCodec3dParams* params,
                         size_t fill_dst_bytes)
{
  if (!zfp_metal_init_context())
    return 0;
  if (!pso)
    return 0;

  if (!zfp_metal_ensure_buffer(zfp_metal_ctx.device, &zfp_metal_ctx.params_buf,
                               &zfp_metal_ctx.params_cap, sizeof(ZfpMetalCodec3dParams)))
    return 0;
  memcpy([zfp_metal_ctx.params_buf contents], params, sizeof(ZfpMetalCodec3dParams));

  id<MTLCommandBuffer> cb = [zfp_metal_ctx.queue commandBuffer];

  if (fill_dst_bytes > 0) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst_buf range:NSMakeRange(0, fill_dst_bytes) value:0];
    [blit endEncoding];
  }

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:0 atIndex:0];
  [enc setBuffer:dst_buf offset:0 atIndex:1];
  [enc setBuffer:zfp_metal_ctx.params_buf offset:0 atIndex:2];

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
  return [cb status] == MTLCommandBufferStatusCompleted;
}

extern "C" size_t
zfp_metal_encode2d_float_runtime(const float* src,
                                 unsigned int nx,
                                 unsigned int ny,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes)
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

  size_t src_len = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t row_span = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  if (row_span > src_len)
    src_len = row_span;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                   length:src_len * sizeof(float)
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

  if (!zfp_metal_launch_codec2d(zfp_metal_select_2d_encode_pso(maxbits), src_buf, dst_buf, &params, stream_bytes))
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
                                 float* dst)
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

  size_t dst_len = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t row_span = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  if (row_span > dst_len)
    dst_len = row_span;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                   length:dst_len * sizeof(float)
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

  if (!zfp_metal_launch_codec2d(zfp_metal_select_2d_decode_pso(maxbits), src_buf, dst_buf, &params, 0))
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
                                 size_t stream_capacity_bytes)
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

  size_t span = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t s2 = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  size_t s3 = (size_t)(sz > 0 ? sz : -sz) * (size_t)nz;
  if (s2 > span) span = s2;
  if (s3 > span) span = s3;

  id<MTLBuffer> src_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:(void*)src
                                                                   length:span * sizeof(float)
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

  if (!zfp_metal_launch_codec3d(zfp_metal_select_3d_encode_pso(maxbits), src_buf, dst_buf, &params, stream_bytes))
    return 0;

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
                                 float* dst)
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

  size_t span = (size_t)(sx > 0 ? sx : -sx) * (size_t)nx;
  size_t s2 = (size_t)(sy > 0 ? sy : -sy) * (size_t)ny;
  size_t s3 = (size_t)(sz > 0 ? sz : -sz) * (size_t)nz;
  if (s2 > span) span = s2;
  if (s3 > span) span = s3;

  id<MTLBuffer> dst_buf = [zfp_metal_ctx.device newBufferWithBytesNoCopy:dst
                                                                   length:span * sizeof(float)
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

  if (!zfp_metal_launch_codec3d(zfp_metal_select_3d_decode_pso(maxbits), src_buf, dst_buf, &params, 0))
    return 0;

  return stream_bytes;
}

extern "C" size_t
zfp_metal_encode1d_double_runtime(const double* src,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes)
{
  /* Metal does not support double-precision; fall through to CPU. */
  (void)src; (void)dim; (void)sx; (void)maxbits;
  (void)stream_words; (void)stream_capacity_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_double_runtime(const void* stream_words,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  double* dst)
{
  (void)stream_words; (void)dim; (void)sx; (void)maxbits; (void)dst;
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
                                  size_t stream_capacity_bytes)
{
  (void)src; (void)nx; (void)ny; (void)sx; (void)sy; (void)maxbits;
  (void)stream_words; (void)stream_capacity_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode2d_double_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  double* dst)
{
  (void)stream_words; (void)nx; (void)ny; (void)sx; (void)sy;
  (void)maxbits; (void)dst;
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
                                  size_t stream_capacity_bytes)
{
  (void)src; (void)nx; (void)ny; (void)nz; (void)sx; (void)sy; (void)sz;
  (void)maxbits; (void)stream_words; (void)stream_capacity_bytes;
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
                                  double* dst)
{
  (void)stream_words; (void)nx; (void)ny; (void)nz;
  (void)sx; (void)sy; (void)sz; (void)maxbits; (void)dst;
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
  if (!zfp_metal_ctx.params_buf)
    return 0;

  memcpy([zfp_metal_ctx.params_buf contents], params, sizeof(ZfpMetalLayoutParams));

  cb = [zfp_metal_ctx.queue commandBuffer];
  enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:0 atIndex:0];
  [enc setBuffer:dst_buf offset:0 atIndex:1];
  [enc setBuffer:zfp_metal_ctx.params_buf offset:0 atIndex:2];

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
                                 size_t stream_capacity_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_float_runtime(const void* stream_words,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 float* dst)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
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
                                 size_t stream_capacity_bytes)
{
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

extern "C" size_t
zfp_metal_decode2d_float_runtime(const void* stream_words,
                                 unsigned int nx,
                                 unsigned int ny,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 unsigned int maxbits,
                                 float* dst)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
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
                                 size_t stream_capacity_bytes)
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
                                 float* dst)
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
  return 0;
}

extern "C" size_t
zfp_metal_encode1d_double_runtime(const double* src,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  void* stream_words,
                                  size_t stream_capacity_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_double_runtime(const void* stream_words,
                                  unsigned int dim,
                                  int sx,
                                  unsigned int maxbits,
                                  double* dst)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
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
                                  size_t stream_capacity_bytes)
{
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

extern "C" size_t
zfp_metal_decode2d_double_runtime(const void* stream_words,
                                  unsigned int nx,
                                  unsigned int ny,
                                  ptrdiff_t sx,
                                  ptrdiff_t sy,
                                  unsigned int maxbits,
                                  double* dst)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
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
                                  size_t stream_capacity_bytes)
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
                                  double* dst)
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
  return 0;
}

extern "C" size_t
zfp_metal_encode1d_int32_runtime(const int* src,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_int32_runtime(const void* stream_words,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 int* dst)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
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
                                 size_t stream_capacity_bytes)
{
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

extern "C" size_t
zfp_metal_decode2d_int32_runtime(const void* stream_words,
                                 unsigned int nx,
                                 unsigned int ny,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 unsigned int maxbits,
                                 int* dst)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
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
                                 size_t stream_capacity_bytes)
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
                                 int* dst)
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
  return 0;
}

extern "C" size_t
zfp_metal_encode1d_int64_runtime(const long* src,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 void* stream_words,
                                 size_t stream_capacity_bytes)
{
  (void)src;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)stream_words;
  (void)stream_capacity_bytes;
  return 0;
}

extern "C" size_t
zfp_metal_decode1d_int64_runtime(const void* stream_words,
                                 unsigned int dim,
                                 int sx,
                                 unsigned int maxbits,
                                 long* dst)
{
  (void)stream_words;
  (void)dim;
  (void)sx;
  (void)maxbits;
  (void)dst;
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
                                 size_t stream_capacity_bytes)
{
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

extern "C" size_t
zfp_metal_decode2d_int64_runtime(const void* stream_words,
                                 unsigned int nx,
                                 unsigned int ny,
                                 ptrdiff_t sx,
                                 ptrdiff_t sy,
                                 unsigned int maxbits,
                                 long* dst)
{
  (void)stream_words;
  (void)nx;
  (void)ny;
  (void)sx;
  (void)sy;
  (void)maxbits;
  (void)dst;
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
                                 size_t stream_capacity_bytes)
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
                                 long* dst)
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
  return 0;
}

#endif
