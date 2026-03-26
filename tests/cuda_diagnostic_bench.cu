/*
 * cuda_diagnostic_bench.cu — Standalone CUDA diagnostic benchmark for ZFP
 *
 * Produces the same pipeline-stage breakdown table as Metal ZFP_METAL_PROFILE=3.
 * Measures 3D float decode at ~10M elements (274,625 blocks) by default.
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_XX cuda_diagnostic_bench.cu -o cuda_diag_bench
 *
 * where sm_XX matches your GPU (e.g. sm_80 for A100, sm_89 for RTX 4090).
 *
 * Usage:
 *   ./cuda_diag_bench              # defaults: 217^3 elements, rate 8
 *   ./cuda_diag_bench 256          # 256^3 elements, rate 8
 *   ./cuda_diag_bench 217 16       # 217^3 elements, rate 16
 *   ./cuda_diag_bench 217 8 10     # 217^3 elements, rate 8, 10 repeats
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cuda_runtime.h>

/* ====================================================================
 * ZFP constants and type traits (self-contained, no zfp.h dependency)
 * ==================================================================== */

typedef unsigned long long Word;  /* 64-bit bitstream word */
typedef unsigned long long uint64;
#ifndef __uint_defined
typedef unsigned int uint;
#define __uint_defined
#endif
typedef unsigned int uint32;

#define NBMASK 0xaaaaaaaaaaaaaaaaull
#define LDEXP(x, e) ldexp(x, e)

/* Permutation table for 3D (64-element blocks) */
#define index_3d(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

static __device__ __constant__ unsigned char c_perm_3d[64] = {
  index_3d(0,0,0),
  index_3d(1,0,0), index_3d(0,1,0), index_3d(0,0,1),
  index_3d(0,1,1), index_3d(1,0,1), index_3d(1,1,0),
  index_3d(2,0,0), index_3d(0,2,0), index_3d(0,0,2),
  index_3d(1,1,1),
  index_3d(2,1,0), index_3d(2,0,1), index_3d(0,2,1),
  index_3d(1,2,0), index_3d(1,0,2), index_3d(0,1,2),
  index_3d(3,0,0), index_3d(0,3,0), index_3d(0,0,3),
  index_3d(2,1,1), index_3d(1,2,1), index_3d(1,1,2),
  index_3d(0,2,2), index_3d(2,0,2), index_3d(2,2,0),
  index_3d(3,1,0), index_3d(3,0,1), index_3d(0,3,1),
  index_3d(1,3,0), index_3d(1,0,3), index_3d(0,1,3),
  index_3d(1,2,2), index_3d(2,1,2), index_3d(2,2,1),
  index_3d(3,1,1), index_3d(1,3,1), index_3d(1,1,3),
  index_3d(3,2,0), index_3d(3,0,2), index_3d(0,3,2),
  index_3d(2,3,0), index_3d(2,0,3), index_3d(0,2,3),
  index_3d(2,2,2),
  index_3d(3,2,1), index_3d(3,1,2), index_3d(1,3,2),
  index_3d(2,3,1), index_3d(2,1,3), index_3d(1,2,3),
  index_3d(0,3,3), index_3d(3,0,3), index_3d(3,3,0),
  index_3d(3,2,2), index_3d(2,3,2), index_3d(2,2,3),
  index_3d(1,3,3), index_3d(3,1,3), index_3d(3,3,1),
  index_3d(2,3,3), index_3d(3,2,3), index_3d(3,3,2),
  index_3d(3,3,3),
};
#undef index_3d

/* Type traits for float */
static inline __host__ __device__ int zfp_ebias_float()  { return 127; }
static inline __host__ __device__ int zfp_ebits_float()  { return 8; }
static inline __host__ __device__ int zfp_prec_float()   { return 32; }
static inline __host__ __device__ int zfp_min_exp_float() { return -1074; }

/* ====================================================================
 * Forward transforms (encode side — needed to generate compressed data)
 * ==================================================================== */

/* Forward lifting transform of 4-vector */
template <class Int, uint s>
__device__ static void fwd_lift(Int* p)
{
  Int x = *p; p += s;
  Int y = *p; p += s;
  Int z = *p; p += s;
  Int w = *p; p += s;

  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

/* 3D forward transform */
template<typename Int>
__device__ void fwd_xform3(Int *p)
{
  for (uint z = 0; z < 4; z++)
    for (uint y = 0; y < 4; y++)
      fwd_lift<Int,1>(p + 4 * y + 16 * z);
  for (uint x = 0; x < 4; x++)
    for (uint z = 0; z < 4; z++)
      fwd_lift<Int,4>(p + 16 * z + 1 * x);
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      fwd_lift<Int,16>(p + 1 * x + 4 * y);
}

/* int2uint (negabinary) */
inline __device__ unsigned int int2uint(int x)
{
  return (x + (unsigned int)0xaaaaaaaau) ^ (unsigned int)0xaaaaaaaau;
}

/* uint2int (negabinary inverse) */
inline __device__ int uint2int(unsigned int x)
{
  return (int)((x ^ 0xaaaaaaaau) - 0xaaaaaaaau);
}

/* Exponent computation */
__device__ static int compute_exponent(float x)
{
  int e = -zfp_ebias_float();
  if (x > 0) {
    frexp(x, &e);
    e = max(e, 1 - zfp_ebias_float());
  }
  return e;
}

/* Maximum exponent of a block */
__device__ static int max_exponent_block(const float* p)
{
  float mx = 0;
  for (int i = 0; i < 64; i++) {
    float f = fabsf(p[i]);
    mx = fmaxf(mx, f);
  }
  return compute_exponent(mx);
}

/* Precision */
__device__ static int zfp_precision(int maxexp, int maxprec, int minexp)
{
  return min(maxprec, max(0, maxexp - minexp + 8));
}

/* ====================================================================
 * BlockWriter — bitstream encoder (always atomicAdd, same as CUDA ZFP)
 * ==================================================================== */

struct BlockWriter
{
  uint m_word_index;
  uint m_start_bit;
  uint m_current_bit;
  int m_maxbits;
  Word *m_stream;

  __device__ BlockWriter(Word *stream, int maxbits, uint block_idx)
    : m_current_bit(0), m_maxbits(maxbits), m_stream(stream)
  {
    m_word_index = (uint)(((size_t)block_idx * maxbits) / (sizeof(Word) * 8));
    m_start_bit  = (uint)(((size_t)block_idx * maxbits) % (sizeof(Word) * 8));
  }

  __device__ unsigned long long int
  write_bits(unsigned long long int bits, uint n_bits)
  {
    const uint wbits = sizeof(Word) * 8;
    uint seg_start = (m_start_bit + m_current_bit) % wbits;
    uint write_index = m_word_index + (uint)((m_start_bit + m_current_bit) / wbits);
    uint seg_end = seg_start + n_bits - 1;
    uint shift = seg_start;

    Word left = (bits >> n_bits) << n_bits;
    Word b = bits - left;
    Word add = b << shift;
    atomicAdd(&m_stream[write_index], add);

    bool straddle = seg_start < sizeof(Word) * 8 && seg_end >= sizeof(Word) * 8;
    if (straddle) {
      Word rem = b >> (sizeof(Word) * 8 - shift);
      atomicAdd(&m_stream[write_index + 1], rem);
    }
    m_current_bit += n_bits;
    return bits >> (Word)n_bits;
  }

  __device__ uint write_bit(unsigned int bit)
  {
    const uint wbits = sizeof(Word) * 8;
    uint seg_start = (m_start_bit + m_current_bit) % wbits;
    uint write_index = m_word_index + (uint)((m_start_bit + m_current_bit) / wbits);
    uint shift = seg_start;

    Word add = (Word)bit << shift;
    atomicAdd(&m_stream[write_index], add);
    m_current_bit += 1;
    return bit;
  }
};

/* ====================================================================
 * Encode pipeline (encode_block + zfp_encode_block for float 3D)
 * ==================================================================== */

__device__ void encode_block_ints(BlockWriter &stream, int maxbits, int maxprec, int *iblock)
{
  const int BlockSize = 64;

  /* Forward decorrelating transform */
  fwd_xform3(iblock);

  /* Reorder and convert to unsigned */
  unsigned int ublock[BlockSize];
  for (int i = 0; i < BlockSize; i++)
    ublock[i] = int2uint(iblock[c_perm_3d[i]]);

  /* Encode bit planes */
  uint intprec = (uint)(CHAR_BIT * sizeof(unsigned int));
  uint kmin = intprec > (uint)maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;

  for (uint k = intprec, n = 0; bits && k-- > kmin;) {
    uint64 x = 0;
    for (uint i = 0; i < BlockSize; i++)
      x += (uint64)((ublock[i] >> k) & 1u) << i;

    uint m = min(n, bits);
    bits -= m;
    x = stream.write_bits(x, m);

    for (; n < BlockSize && bits && (bits--, stream.write_bit(!!x)); x >>= 1, n++)
      for (; n < BlockSize - 1 && bits && (bits--, !stream.write_bit(x & 1u)); x >>= 1, n++)
        ;
  }
}

__device__ void zfp_encode_block_float3d(float *fblock, int maxbits, uint block_idx, Word *stream)
{
  BlockWriter writer(stream, maxbits, block_idx);
  int emax = max_exponent_block(fblock);
  int maxprec = zfp_precision(emax, zfp_prec_float(), zfp_min_exp_float());
  uint e = maxprec ? (uint)(emax + zfp_ebias_float()) : 0;
  if (e) {
    uint ebits = zfp_ebits_float() + 1;
    writer.write_bits(2 * e + 1, ebits);
    int iblock[64];
    float s = LDEXP(1.0, zfp_prec_float() - 2 - emax);
    for (int i = 0; i < 64; i++)
      iblock[i] = (int)(s * fblock[i]);
    encode_block_ints(writer, maxbits - ebits, maxprec, iblock);
  }
}

/* ====================================================================
 * Full encode kernel (3D float)
 * ==================================================================== */

__global__ void kernel_encode3d_float(
  uint maxbits, const float* scalars, Word* stream,
  uint3 dims, uint3 padded_dims, uint tot_blocks)
{
  typedef unsigned long long int ull;
  const ull blockId = blockIdx.x + blockIdx.y * (ull)gridDim.x +
                      (ull)gridDim.x * gridDim.y * blockIdx.z;
  const uint block_idx = (uint)(blockId * blockDim.x + threadIdx.x);
  if (block_idx >= tot_blocks) return;

  uint3 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;
  block_dims.z = padded_dims.z >> 2;

  uint3 block;
  block.x = (block_idx % block_dims.x) * 4;
  block.y = ((block_idx / block_dims.x) % block_dims.y) * 4;
  block.z = (block_idx / (block_dims.x * block_dims.y)) * 4;

  long long offset = (long long)block.x + (long long)block.y * dims.x +
                     (long long)block.z * dims.x * dims.y;

  float fblock[64];
  /* Full-block gather (contiguous stride 1, sx=1, sy=dims.x, sz=dims.x*dims.y) */
  int sx = 1, sy = dims.x, sz = dims.x * dims.y;
  bool partial = (block.x + 4 > dims.x) || (block.y + 4 > dims.y) || (block.z + 4 > dims.z);
  if (partial) {
    uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    uint nz = block.z + 4 > dims.z ? dims.z - block.z : 4;
    memset(fblock, 0, sizeof(fblock));
    const float *p = scalars + offset;
    for (uint z = 0; z < nz; z++)
      for (uint y = 0; y < ny; y++)
        for (uint x = 0; x < nx; x++)
          fblock[16 * z + 4 * y + x] = p[x * sx + y * sy + z * sz];
  } else {
    const float *p = scalars + offset;
    for (uint z = 0; z < 4; z++)
      for (uint y = 0; y < 4; y++)
        for (uint x = 0; x < 4; x++)
          fblock[16 * z + 4 * y + x] = p[x * sx + y * sy + z * sz];
  }
  zfp_encode_block_float3d(fblock, maxbits, block_idx, stream);
}

/* ====================================================================
 * Inverse transforms (decode side)
 * ==================================================================== */

/* Inverse lifting transform of 4-vector */
template <class Int, uint s>
__device__ static void inv_lift(Int* p)
{
  Int x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  y += w >> 1; w -= y >> 1;
  y += w; w -= y - w;
  z += x; x -= z - x;
  y += z; z -= y - z;
  w += x; x -= w - x;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

/* 3D inverse transform */
template<typename Int>
__device__ void inv_xform3(Int *p)
{
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      inv_lift<Int, 16>(p + 1 * x + 4 * y);
  for (uint x = 0; x < 4; x++)
    for (uint z = 0; z < 4; z++)
      inv_lift<Int, 4>(p + 16 * z + 1 * x);
  for (uint z = 0; z < 4; z++)
    for (uint y = 0; y < 4; y++)
      inv_lift<Int, 1>(p + 4 * y + 16 * z);
}

/* Dequantize */
__device__ static float dequantize_float(int x, int e)
{
  return LDEXP((float)x, e - (int)(CHAR_BIT * sizeof(float)) + 2);
}

/* ====================================================================
 * BlockReader — bitstream decoder (exact replica of CUDA ZFP)
 * ==================================================================== */

struct BlockReader
{
  int m_maxbits;
  int m_current_bit;
  Word *m_words;
  Word m_buffer;
  bool m_valid_block;

  __device__ BlockReader(Word *b, int maxbits, int block_idx, int num_blocks)
    : m_maxbits(maxbits), m_valid_block(true)
  {
    if (block_idx >= num_blocks) m_valid_block = false;
    size_t word_index = ((size_t)block_idx * maxbits) / (sizeof(Word) * 8);
    m_words = b + word_index;
    m_buffer = *m_words;
    m_current_bit = ((size_t)block_idx * maxbits) % (sizeof(Word) * 8);
    m_buffer >>= m_current_bit;
  }

  __device__ uint read_bit()
  {
    uint bit = m_buffer & 1;
    ++m_current_bit;
    m_buffer >>= 1;
    if (m_current_bit >= (int)(sizeof(Word) * 8)) {
      m_current_bit = 0;
      ++m_words;
      m_buffer = *m_words;
    }
    return bit;
  }

  __device__ uint64 read_bits(uint n_bits)
  {
    uint64 bits;
    int rem_bits = sizeof(Word) * 8 - m_current_bit;
    int first_read = min(rem_bits, (int)n_bits);
    Word mask = ((Word)1 << first_read) - 1;
    bits = m_buffer & mask;
    m_buffer >>= n_bits;
    m_current_bit += first_read;
    int next_read = 0;
    if ((int)n_bits >= rem_bits) {
      ++m_words;
      m_buffer = *m_words;
      m_current_bit = 0;
      next_read = n_bits - first_read;
    }
    mask = ((Word)1 << next_read) - 1;
    bits += (m_buffer & mask) << first_read;
    m_buffer >>= next_read;
    m_current_bit += next_read;
    return bits;
  }
};

/* ====================================================================
 * Decode pipeline stages — individual diagnostic kernels
 * ==================================================================== */

/* ---- Diagnostic 1: memcopy ----
 * Pure memory bandwidth: read 64 uint per block, write 64 float.
 * Same as Metal's zfp_diag_memcopy_3d. */
__global__ void diag_memcopy_3d(
  const unsigned int* src, float* dst, int total_blocks)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_blocks) return;
  unsigned long long base = (unsigned long long)gid * 64ull;
  for (int i = 0; i < 64; i++) {
    unsigned int v = src[base + i];
    float f;
    memcpy(&f, &v, sizeof(float));
    dst[base + i] = f;
  }
}

/* ---- Diagnostic 2: bitread_only ----
 * Read bitstream exactly as production decode, but discard results.
 * Write single checksum per block. Measures pure serial bit-reading cost. */
__global__ void diag_bitread_only_3d(
  Word* stream, unsigned int* dst, int maxbits, int total_blocks)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_blocks) return;

  BlockReader reader(stream, maxbits, gid, total_blocks);

  uint s_cont = reader.read_bit();
  if (!s_cont) {
    dst[gid] = 0;
    return;
  }
  uint e = (uint)reader.read_bits(8u);
  uint bits = maxbits - 9u;
  uint checksum = e;

  /* Decode bit planes — same loop as production, but accumulate to checksum */
  uint n = 0, m = 0;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    uint64 x = reader.read_bits(m);
    checksum ^= (uint)(x & 0xFFFFFFFFu);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (reader.read_bit()) {
        for (; n < 63u && bits; n++) {
          bits--;
          if (reader.read_bit()) break;
        }
        x += (uint64)1 << n;
      } else {
        m = 64u;
        break;
      }
    }
    checksum ^= (uint)(x >> 32);
  }
  dst[gid] = checksum;
}

/* ---- Diagnostic 3: bitplane_only (64-bit scatter) ----
 * Bitstream decode + 64-bit scatter to ub[64] registers (CUDA's native approach).
 * No inverse transform, no dequantize. */
__global__ void diag_bitplane_only_3d(
  Word* stream, unsigned int* dst, int maxbits, int total_blocks)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_blocks) return;

  BlockReader reader(stream, maxbits, gid, total_blocks);
  unsigned int ub[64];
  memset(ub, 0, sizeof(ub));

  uint s_cont = reader.read_bit();
  if (s_cont) {
    uint e = (uint)reader.read_bits(8u);
    (void)e;
    uint bits = maxbits - 9u;

    uint n = 0, m = 0;
    for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
      m = min(n, bits);
      bits -= m;
      uint64 x = reader.read_bits(m);
      for (; bits && n < 64u; n++, m = n) {
        bits--;
        if (reader.read_bit()) {
          for (; n < 63u && bits; n++) {
            bits--;
            if (reader.read_bit()) break;
          }
          x += (uint64)1 << n;
        } else {
          m = 64u;
          break;
        }
      }
      /* 64-bit scatter — CUDA's native approach */
      for (uint i = 0; i < 64u; i++, x >>= 1)
        ub[i] += (unsigned int)(x & 1u) << k;
    }
  }

  /* Write ub to output (flat contiguous) */
  unsigned long long base = (unsigned long long)gid * 64ull;
  for (int i = 0; i < 64; i++)
    dst[base + i] = ub[i];
}

/* ---- Diagnostic 4: bitplane_split32 (split-32 scatter) ----
 * Same bitstream decode, but uses split-32 scatter (Metal's optimized approach).
 * Two loops of 32 iterations using uint32 halves instead of one 64-bit loop.
 * Tests whether split-32 helps or hurts on NVIDIA's native 64-bit ALUs. */
__global__ void diag_bitplane_split32_3d(
  Word* stream, unsigned int* dst, int maxbits, int total_blocks)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_blocks) return;

  BlockReader reader(stream, maxbits, gid, total_blocks);
  unsigned int ub[64];
  memset(ub, 0, sizeof(ub));

  uint s_cont = reader.read_bit();
  if (s_cont) {
    uint e = (uint)reader.read_bits(8u);
    (void)e;
    uint bits = maxbits - 9u;

    uint n = 0, m = 0;
    for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
      m = min(n, bits);
      bits -= m;
      uint64 x = reader.read_bits(m);
      for (; bits && n < 64u; n++, m = n) {
        bits--;
        if (reader.read_bit()) {
          for (; n < 63u && bits; n++) {
            bits--;
            if (reader.read_bit()) break;
          }
          x += (uint64)1 << n;
        } else {
          m = 64u;
          break;
        }
      }
      /* Split-32 scatter — Metal's approach */
      unsigned int lo = (unsigned int)(x & 0xFFFFFFFFu);
      unsigned int hi = (unsigned int)(x >> 32);
      for (uint i = 0; i < 32u; i++, lo >>= 1)
        ub[i] += (lo & 1u) << k;
      for (uint i = 0; i < 32u; i++, hi >>= 1)
        ub[32 + i] += (hi & 1u) << k;
    }
  }

  unsigned long long base = (unsigned long long)gid * 64ull;
  for (int i = 0; i < 64; i++)
    dst[base + i] = ub[i];
}

/* ---- Diagnostic 5: transform_only ----
 * Permute + uint2int + inv_lift + dequantize on synthetic data.
 * No bitstream reading. Measures pure ALU/transform cost. */
__global__ void diag_transform_only_3d(
  const unsigned int* src, float* dst, int total_blocks)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= total_blocks) return;

  unsigned long long base = (unsigned long long)gid * 64ull;

  /* Read ub values from memory (simulating bitplane output) */
  unsigned int ub[64];
  for (int i = 0; i < 64; i++)
    ub[i] = src[base + i];

  /* Permute + uint2int */
  int iblock[64];
  for (int i = 0; i < 64; i++)
    iblock[c_perm_3d[i]] = uint2int(ub[i]);

  /* Inverse lifting transform */
  inv_xform3(iblock);

  /* Dequantize with a reasonable exponent */
  float inv_w = dequantize_float(1, 10);
  for (int i = 0; i < 64; i++)
    dst[base + i] = inv_w * (float)iblock[i];
}

/* ---- Diagnostic 6: full_decode (production pipeline) ----
 * Complete decode: bitstream read + scatter + permute + inv_lift + dequantize.
 * This is the actual CUDA ZFP decode kernel, exactly as in decode3.cuh. */
__global__ void diag_full_decode_3d(
  Word* stream, float* dst,
  uint3 dims, uint3 padded_dims, int maxbits, int total_blocks)
{
  typedef unsigned long long int ull;
  const ull blockId = blockIdx.x + blockIdx.y * (ull)gridDim.x +
                      (ull)gridDim.x * gridDim.y * blockIdx.z;
  const int block_idx = (int)(blockId * blockDim.x + threadIdx.x);
  if (block_idx >= total_blocks) return;

  BlockReader reader(stream, maxbits, block_idx, total_blocks);

  float result[64];
  memset(result, 0, sizeof(result));

  /* Full decode pipeline (same as cuZFP::zfp_decode) */
  uint s_cont = reader.read_bit();
  if (s_cont) {
    uint ebits = zfp_ebits_float() + 1;
    int emax = (int)reader.read_bits(ebits - 1) - zfp_ebias_float();
    int mbits = maxbits - ebits;

    unsigned int ublock[64];
    memset(ublock, 0, sizeof(ublock));

    /* decode_ints: bitplane decode with 64-bit scatter */
    uint intprec = zfp_prec_float();
    uint bits = mbits;
    uint k2, m2, n2;
    for (k2 = intprec, m2 = n2 = 0; bits && (m2 = 0, k2-- > 0);) {
      m2 = min(n2, bits);
      bits -= m2;
      uint64 x = reader.read_bits(m2);
      for (; bits && n2 < 64u; n2++, m2 = n2) {
        bits--;
        if (reader.read_bit()) {
          for (; bits && n2 < 63u; n2++) {
            bits--;
            if (reader.read_bit()) break;
          }
          x += (uint64)1 << n2;
        } else {
          m2 = 64u;
          break;
        }
      }
      #pragma unroll 64
      for (uint i = 0; i < 64u; i++, x >>= 1)
        ublock[i] += (unsigned int)(x & 1u) << k2;
    }

    /* Permute + uint2int */
    int iblock[64];
    #pragma unroll 64
    for (int i = 0; i < 64; i++)
      iblock[c_perm_3d[i]] = uint2int(ublock[i]);

    /* Inverse transform */
    inv_xform3(iblock);

    /* Dequantize */
    float inv_w = dequantize_float(1, emax);
    #pragma unroll 64
    for (int i = 0; i < 64; i++)
      result[i] = inv_w * (float)iblock[i];
  }

  /* Scatter to output (contiguous for benchmarking, stride 1) */
  uint3 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;
  block_dims.z = padded_dims.z >> 2;

  uint3 block;
  block.x = (block_idx % block_dims.x) * 4;
  block.y = ((block_idx / block_dims.x) % block_dims.y) * 4;
  block.z = (block_idx / (block_dims.x * block_dims.y)) * 4;

  long long offset = (long long)block.x + (long long)block.y * dims.x +
                     (long long)block.z * dims.x * dims.y;

  bool partial = (block.x + 4 > dims.x) || (block.y + 4 > dims.y) || (block.z + 4 > dims.z);
  if (partial) {
    uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    uint nz = block.z + 4 > dims.z ? dims.z - block.z : 4;
    for (uint z = 0; z < nz; z++)
      for (uint y = 0; y < ny; y++)
        for (uint x = 0; x < nx; x++)
          dst[offset + x + y * dims.x + z * dims.x * dims.y] = result[16 * z + 4 * y + x];
  } else {
    int sx = 1, sy = dims.x, sz = dims.x * dims.y;
    float *p = dst + offset;
    const float *q = result;
    for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
      for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
        for (uint x = 0; x < 4; x++, p += sx)
          *p = *q++;
  }
}

/* ====================================================================
 * Helper: grid/block calculation
 * ==================================================================== */

static dim3 calc_grid(int total_threads, int block_size)
{
  int grids = (total_threads + block_size - 1) / block_size;
  /* Simple 1D grid — CUDA supports up to 2^31-1 blocks in x */
  if (grids <= 65535) return dim3(grids, 1, 1);
  int sq = (int)ceilf(sqrtf((float)grids));
  return dim3(sq, (grids + sq - 1) / sq, 1);
}

/* ====================================================================
 * Check CUDA errors
 * ==================================================================== */

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

/* ====================================================================
 * Main — generate data, compress, run diagnostics
 * ==================================================================== */

int main(int argc, char **argv)
{
  /* Parse arguments */
  int dim_n = 217;    /* default: 217^3 ≈ 10.2M elements */
  int rate = 8;       /* bits per value */
  int repeats = 5;

  if (argc > 1) dim_n = atoi(argv[1]);
  if (argc > 2) rate = atoi(argv[2]);
  if (argc > 3) repeats = atoi(argv[3]);

  int nx = dim_n, ny = dim_n, nz = dim_n;
  size_t num_elements = (size_t)nx * ny * nz;
  size_t data_bytes = num_elements * sizeof(float);

  /* Padded dims (multiple of 4) */
  int pnx = nx + (nx % 4 ? 4 - nx % 4 : 0);
  int pny = ny + (ny % 4 ? 4 - ny % 4 : 0);
  int pnz = nz + (nz % 4 ? 4 - nz % 4 : 0);
  int total_blocks = (pnx * pny * pnz) / 64;
  int maxbits = rate * 64;  /* bits per block = rate * block_size */

  /* Stream size in bytes */
  size_t stream_bits = (size_t)maxbits * total_blocks;
  size_t stream_words = (stream_bits + 63) / 64;
  size_t stream_bytes = stream_words * sizeof(Word);

  printf("=== CUDA ZFP Diagnostic Benchmark ===\n");
  printf("Grid:           %d x %d x %d = %zu elements (%.1f MB)\n",
         nx, ny, nz, num_elements, (double)data_bytes / (1024.0 * 1024.0));
  printf("Padded:         %d x %d x %d\n", pnx, pny, pnz);
  printf("Total blocks:   %d\n", total_blocks);
  printf("Rate:           %d bits/value (%d bits/block)\n", rate, maxbits);
  printf("Stream size:    %.1f MB\n", (double)stream_bytes / (1024.0 * 1024.0));
  printf("Repeats:        %d\n\n", repeats);

  /* Print GPU info */
  int device;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("GPU:            %s\n", prop.name);
  printf("Compute:        %d.%d\n", prop.major, prop.minor);
  printf("SMs:            %d\n", prop.multiProcessorCount);
  printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("Warp size:      %d\n", prop.warpSize);
  printf("Memory:         %.0f MB\n\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0));

  /* ---- Generate smooth test data on host ---- */
  printf("Generating smooth test data...\n");
  float *h_data = (float*)malloc(data_bytes);
  srand(42);
  for (size_t z = 0; z < (size_t)nz; z++)
    for (size_t y = 0; y < (size_t)ny; y++)
      for (size_t x = 0; x < (size_t)nx; x++) {
        /* Smooth function with small noise for realistic ZFP compression */
        float val = sinf(0.1f * x) * cosf(0.15f * y) * sinf(0.12f * z)
                  + 0.01f * ((float)rand() / RAND_MAX - 0.5f);
        h_data[z * (size_t)nx * ny + y * nx + x] = val;
      }
  printf("Done.\n");

  /* ---- Allocate device memory ---- */
  float *d_data;
  Word *d_stream;
  float *d_output;
  unsigned int *d_scratch;  /* for diagnostic kernels */

  CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
  CUDA_CHECK(cudaMalloc(&d_stream, stream_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, data_bytes));
  /* Scratch: max(total_blocks * 64 * sizeof(uint), data_bytes) */
  size_t scratch_bytes = (size_t)total_blocks * 64 * sizeof(unsigned int);
  if (scratch_bytes < data_bytes) scratch_bytes = data_bytes;
  CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));

  /* Copy data to device */
  CUDA_CHECK(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

  /* ---- Compress on device ---- */
  printf("Compressing on GPU (encode)...\n");
  CUDA_CHECK(cudaMemset(d_stream, 0, stream_bytes));

  uint3 dims = make_uint3(nx, ny, nz);
  uint3 padded_dims = make_uint3(pnx, pny, pnz);

  const int cuda_block_size = 128;
  int block_pad = 0;
  if (total_blocks % cuda_block_size != 0)
    block_pad = cuda_block_size - total_blocks % cuda_block_size;
  int launch_blocks = total_blocks + block_pad;

  dim3 grid_enc = calc_grid(launch_blocks, cuda_block_size);
  dim3 blk_enc(cuda_block_size, 1, 1);

  kernel_encode3d_float<<<grid_enc, blk_enc>>>(
    maxbits, d_data, d_stream, dims, padded_dims, total_blocks);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Encode done.\n\n");

  /* ---- Create CUDA events for timing ---- */
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  /* ---- Run diagnostic kernels ---- */
  struct DiagResult {
    const char *name;
    const char *description;
    float best_ms;
  };
  const int NUM_DIAGS = 6;
  DiagResult results[NUM_DIAGS] = {
    {"memcopy_3d",      "Pure bandwidth: read 64 uint + write 64 float", 1e9f},
    {"bitread_only_3d", "Bitstream read only (discard results)",         1e9f},
    {"bitplane_only_3d","Bitstream + 64-bit scatter (CUDA native)",      1e9f},
    {"bitplane_split32","Bitstream + split-32 scatter (Metal approach)",  1e9f},
    {"transform_only",  "Permute + uint2int + inv_lift + dequant",       1e9f},
    {"full_decode_3d",  "Complete production decode pipeline",           1e9f},
  };

  dim3 grid_diag = calc_grid(launch_blocks, cuda_block_size);
  dim3 blk_diag(cuda_block_size, 1, 1);

  /* Warmup all kernels once */
  printf("Warming up...\n");
  diag_memcopy_3d<<<grid_diag, blk_diag>>>((unsigned int*)d_data, d_output, total_blocks);
  diag_bitread_only_3d<<<grid_diag, blk_diag>>>(d_stream, d_scratch, maxbits, total_blocks);
  diag_bitplane_only_3d<<<grid_diag, blk_diag>>>(d_stream, d_scratch, maxbits, total_blocks);
  diag_bitplane_split32_3d<<<grid_diag, blk_diag>>>(d_stream, d_scratch, maxbits, total_blocks);
  diag_transform_only_3d<<<grid_diag, blk_diag>>>(d_scratch, d_output, total_blocks);
  diag_full_decode_3d<<<grid_diag, blk_diag>>>(d_stream, d_output, dims, padded_dims, maxbits, total_blocks);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Warmup done.\n\n");

  printf("Running %d repeats per diagnostic kernel...\n\n", repeats);

  for (int r = 0; r < repeats; r++) {
    float ms;

    /* 1: memcopy */
    CUDA_CHECK(cudaEventRecord(start));
    diag_memcopy_3d<<<grid_diag, blk_diag>>>((unsigned int*)d_data, d_output, total_blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < results[0].best_ms) results[0].best_ms = ms;

    /* 2: bitread_only */
    CUDA_CHECK(cudaEventRecord(start));
    diag_bitread_only_3d<<<grid_diag, blk_diag>>>(d_stream, d_scratch, maxbits, total_blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < results[1].best_ms) results[1].best_ms = ms;

    /* 3: bitplane_only (64-bit scatter) */
    CUDA_CHECK(cudaEventRecord(start));
    diag_bitplane_only_3d<<<grid_diag, blk_diag>>>(d_stream, d_scratch, maxbits, total_blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < results[2].best_ms) results[2].best_ms = ms;

    /* 4: bitplane_split32 */
    CUDA_CHECK(cudaEventRecord(start));
    diag_bitplane_split32_3d<<<grid_diag, blk_diag>>>(d_stream, d_scratch, maxbits, total_blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < results[3].best_ms) results[3].best_ms = ms;

    /* 5: transform_only */
    /* First fill scratch with plausible uint values */
    CUDA_CHECK(cudaMemcpy(d_scratch, d_data, (size_t)total_blocks * 64 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    diag_transform_only_3d<<<grid_diag, blk_diag>>>(d_scratch, d_output, total_blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < results[4].best_ms) results[4].best_ms = ms;

    /* 6: full_decode */
    CUDA_CHECK(cudaEventRecord(start));
    diag_full_decode_3d<<<grid_diag, blk_diag>>>(d_stream, d_output, dims, padded_dims, maxbits, total_blocks);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < results[5].best_ms) results[5].best_ms = ms;
  }

  /* ---- Print results table ---- */
  double data_gb = (double)data_bytes / (1024.0 * 1024.0 * 1024.0);

  printf("=======================================================================\n");
  printf("CUDA ZFP Diagnostic Results — 3D Float Decode (%d^3, rate %d)\n", dim_n, rate);
  printf("=======================================================================\n");
  printf("%-22s | %-50s | %8s | %8s | %8s\n",
         "Kernel", "Description", "Time(ms)", "GB/s", "% of Full");
  printf("-----------------------|----------------------------------------------------"
         "|----------|----------|----------\n");

  float full_ms = results[NUM_DIAGS - 1].best_ms;

  for (int i = 0; i < NUM_DIAGS; i++) {
    float ms = results[i].best_ms;
    double gbs = data_gb / (ms / 1000.0);
    double pct = 100.0 * ms / full_ms;
    printf("%-22s | %-50s | %8.2f | %8.1f | %7.0f%%\n",
           results[i].name, results[i].description, ms, gbs, pct);
  }
  printf("=======================================================================\n\n");

  /* Also print a markdown-friendly table */
  printf("### Markdown Table\n\n");
  printf("| Diagnostic Kernel | What It Measures | Time (ms) | GB/s | %% of Full Decode |\n");
  printf("|---|---|---|---|---|\n");
  for (int i = 0; i < NUM_DIAGS; i++) {
    float ms = results[i].best_ms;
    double gbs = data_gb / (ms / 1000.0);
    double pct = 100.0 * ms / full_ms;
    printf("| **`%s`** | %s | %.2f ms | %.1f GB/s | %.0f%% |\n",
           results[i].name, results[i].description, ms, gbs, pct);
  }
  printf("\n");

  /* Cleanup */
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_stream));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_scratch));
  free(h_data);

  printf("Done. Copy the markdown table above and compare with Metal ZFP_METAL_PROFILE=3 output.\n");
  return 0;
}
