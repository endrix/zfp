#include <metal_stdlib>
using namespace metal;

struct LayoutParams {
  uint nx;
  uint ny;
  uint nz;
  long sx;
  long sy;
  long sz;
  ulong elem_size;
  ulong total;
  long offset;
};

static inline ulong linear_to_strided_index(ulong i, constant LayoutParams& p)
{
  ulong x = i % (ulong)p.nx;
  ulong t = i / (ulong)p.nx;
  ulong y = t % (ulong)p.ny;
  ulong z = t / (ulong)p.ny;
  return (ulong)(((long)x * p.sx + (long)y * p.sy + (long)z * p.sz) - p.offset);
}

kernel void zfp_pack_strided(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong si = linear_to_strided_index(i, p);
  ulong src_byte = si * p.elem_size;
  ulong dst_byte = i * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[dst_byte + b] = src[src_byte + b];
}

kernel void zfp_unpack_strided(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong di = linear_to_strided_index(i, p);
  ulong src_byte = i * p.elem_size;
  ulong dst_byte = di * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[dst_byte + b] = src[src_byte + b];
}

kernel void zfp_encode_contig_1d(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong byte = i * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[byte + b] = src[byte + b];
}

kernel void zfp_decode_contig_1d(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong byte = i * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[byte + b] = src[byte + b];
}

kernel void zfp_encode_contig_2d(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong x = i % (ulong)p.nx;
  ulong y = i / (ulong)p.nx;
  ulong idx = x + y * (ulong)p.nx;
  ulong byte = idx * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[byte + b] = src[byte + b];
}

kernel void zfp_decode_contig_2d(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong x = i % (ulong)p.nx;
  ulong y = i / (ulong)p.nx;
  ulong idx = x + y * (ulong)p.nx;
  ulong byte = idx * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[byte + b] = src[byte + b];
}

kernel void zfp_encode_contig_3d(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong x = i % (ulong)p.nx;
  ulong t = i / (ulong)p.nx;
  ulong y = t % (ulong)p.ny;
  ulong z = t / (ulong)p.ny;
  ulong idx = x + (y + z * (ulong)p.ny) * (ulong)p.nx;
  ulong byte = idx * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[byte + b] = src[byte + b];
}

kernel void zfp_decode_contig_3d(
  device const uchar* src [[buffer(0)]],
  device uchar* dst [[buffer(1)]],
  constant LayoutParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  ulong i = (ulong)gid;
  if (i >= p.total)
    return;
  ulong x = i % (ulong)p.nx;
  ulong t = i / (ulong)p.nx;
  ulong y = t % (ulong)p.ny;
  ulong z = t / (ulong)p.ny;
  ulong idx = x + (y + z * (ulong)p.ny) * (ulong)p.nx;
  ulong byte = idx * p.elem_size;
  for (ulong b = 0; b < p.elem_size; ++b)
    dst[byte + b] = src[byte + b];
}

struct Codec1dParams {
  uint dim;
  int sx;
  uint maxbits;
  uint padded_dim;
  uint total_blocks;
};

struct Codec2dParams {
  uint nx;
  uint ny;
  long sx;
  long sy;
  uint maxbits;
  uint bx;
  uint by;
  uint total_blocks;
};

struct Codec3dParams {
  uint nx;
  uint ny;
  uint nz;
  long sx;
  long sy;
  long sz;
  uint maxbits;
  uint bx;
  uint by;
  uint bz;
  uint total_blocks;
};

static inline int zfp_precision(int maxexp, int maxprec, int minexp)
{
  int a = maxexp - minexp + 8;
  if (a < 0) a = 0;
  if (a > maxprec) a = maxprec;
  return a;
}

static inline int zfp_exponent_float(float x)
{
  int e = -127;
  if (x > 0.0f) {
    frexp(x, e);
    if (e < -126)
      e = -126;
  }
  return e;
}

static inline uint zfp_int2uint(int x)
{
  return (uint)((x + (int)0xaaaaaaaau) ^ (int)0xaaaaaaaau);
}

constant uchar zfp_perm2[16] = {
  0, 1, 4, 5,
  2, 8, 6, 9,
  3, 12, 10, 7,
  13, 11, 14, 15
};

constant uchar zfp_perm3[64] = {
  0,
  1, 4, 16,
  20, 17, 5,
  2, 8, 32,
  21,
  6, 18, 24, 9, 33, 36,
  3, 12, 48,
  22, 25, 37,
  40, 34, 10,
  7, 19, 28, 13, 49, 52,
  41, 38, 26,
  23, 29, 53,
  11, 35, 44, 14, 50, 56,
  42,
  27, 39, 45, 30, 54, 57,
  60, 51, 15,
  43, 46, 58,
  61, 55, 31,
  62, 59, 47,
  63
};

static inline int zfp_uint2int(uint x)
{
  return (int)((x ^ 0xaaaaaaaau) - 0xaaaaaaaau);
}

static inline void zfp_fwd_lift1(thread int* p)
{
  int x = p[0];
  int y = p[1];
  int z = p[2];
  int w = p[3];
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;
  p[0] = x;
  p[1] = y;
  p[2] = z;
  p[3] = w;
}

static inline void zfp_inv_lift1(thread int* p)
{
  int x = p[0];
  int y = p[1];
  int z = p[2];
  int w = p[3];
  y += w >> 1; w -= y >> 1;
  y += w; w -= y - w;
  z += x; x -= z - x;
  y += z; z -= y - z;
  w += x; x -= w - x;
  p[0] = x;
  p[1] = y;
  p[2] = z;
  p[3] = w;
}

static inline void zfp_fwd_lift2_row(thread int* p)
{
  zfp_fwd_lift1(p + 0);
  zfp_fwd_lift1(p + 4);
  zfp_fwd_lift1(p + 8);
  zfp_fwd_lift1(p + 12);
}

static inline void zfp_fwd_lift2_col(thread int* p)
{
  int c0[4] = { p[0], p[4], p[8], p[12] };
  int c1[4] = { p[1], p[5], p[9], p[13] };
  int c2[4] = { p[2], p[6], p[10], p[14] };
  int c3[4] = { p[3], p[7], p[11], p[15] };
  zfp_fwd_lift1(c0); zfp_fwd_lift1(c1); zfp_fwd_lift1(c2); zfp_fwd_lift1(c3);
  p[0] = c0[0]; p[4] = c0[1]; p[8] = c0[2]; p[12] = c0[3];
  p[1] = c1[0]; p[5] = c1[1]; p[9] = c1[2]; p[13] = c1[3];
  p[2] = c2[0]; p[6] = c2[1]; p[10] = c2[2]; p[14] = c2[3];
  p[3] = c3[0]; p[7] = c3[1]; p[11] = c3[2]; p[15] = c3[3];
}

static inline void zfp_inv_lift2_col(thread int* p)
{
  int c0[4] = { p[0], p[4], p[8], p[12] };
  int c1[4] = { p[1], p[5], p[9], p[13] };
  int c2[4] = { p[2], p[6], p[10], p[14] };
  int c3[4] = { p[3], p[7], p[11], p[15] };
  zfp_inv_lift1(c0); zfp_inv_lift1(c1); zfp_inv_lift1(c2); zfp_inv_lift1(c3);
  p[0] = c0[0]; p[4] = c0[1]; p[8] = c0[2]; p[12] = c0[3];
  p[1] = c1[0]; p[5] = c1[1]; p[9] = c1[2]; p[13] = c1[3];
  p[2] = c2[0]; p[6] = c2[1]; p[10] = c2[2]; p[14] = c2[3];
  p[3] = c3[0]; p[7] = c3[1]; p[11] = c3[2]; p[15] = c3[3];
}

static inline void zfp_inv_lift2_row(thread int* p)
{
  zfp_inv_lift1(p + 0);
  zfp_inv_lift1(p + 4);
  zfp_inv_lift1(p + 8);
  zfp_inv_lift1(p + 12);
}

static inline void zfp_fwd_lift3(thread int* p)
{
  for (uint z = 0; z < 4u; ++z)
    for (uint y = 0; y < 4u; ++y)
      zfp_fwd_lift1(p + 4u * y + 16u * z);

  for (uint x = 0; x < 4u; ++x)
    for (uint z = 0; z < 4u; ++z) {
      int c[4] = { p[16u * z + 1u * x], p[16u * z + 4u + 1u * x], p[16u * z + 8u + 1u * x], p[16u * z + 12u + 1u * x] };
      zfp_fwd_lift1(c);
      p[16u * z + 1u * x] = c[0];
      p[16u * z + 4u + 1u * x] = c[1];
      p[16u * z + 8u + 1u * x] = c[2];
      p[16u * z + 12u + 1u * x] = c[3];
    }

  for (uint y = 0; y < 4u; ++y)
    for (uint x = 0; x < 4u; ++x) {
      int c[4] = { p[1u * x + 4u * y], p[16u + 1u * x + 4u * y], p[32u + 1u * x + 4u * y], p[48u + 1u * x + 4u * y] };
      zfp_fwd_lift1(c);
      p[1u * x + 4u * y] = c[0];
      p[16u + 1u * x + 4u * y] = c[1];
      p[32u + 1u * x + 4u * y] = c[2];
      p[48u + 1u * x + 4u * y] = c[3];
    }
}

static inline void zfp_inv_lift3(thread int* p)
{
  for (uint y = 0; y < 4u; ++y)
    for (uint x = 0; x < 4u; ++x) {
      int c[4] = { p[1u * x + 4u * y], p[16u + 1u * x + 4u * y], p[32u + 1u * x + 4u * y], p[48u + 1u * x + 4u * y] };
      zfp_inv_lift1(c);
      p[1u * x + 4u * y] = c[0];
      p[16u + 1u * x + 4u * y] = c[1];
      p[32u + 1u * x + 4u * y] = c[2];
      p[48u + 1u * x + 4u * y] = c[3];
    }

  for (uint x = 0; x < 4u; ++x)
    for (uint z = 0; z < 4u; ++z) {
      int c[4] = { p[16u * z + 1u * x], p[16u * z + 4u + 1u * x], p[16u * z + 8u + 1u * x], p[16u * z + 12u + 1u * x] };
      zfp_inv_lift1(c);
      p[16u * z + 1u * x] = c[0];
      p[16u * z + 4u + 1u * x] = c[1];
      p[16u * z + 8u + 1u * x] = c[2];
      p[16u * z + 12u + 1u * x] = c[3];
    }

  for (uint z = 0; z < 4u; ++z)
    for (uint y = 0; y < 4u; ++y)
      zfp_inv_lift1(p + 4u * y + 16u * z);
}

/* ========================================================================== */
/* 64-bit integer helpers for int64 codec                                     */
/* ========================================================================== */

static inline ulong zfp_int2uint_long(long x)
{
  return (ulong)((x + (long)0xaaaaaaaaaaaaaaaaul) ^ (long)0xaaaaaaaaaaaaaaaaul);
}

static inline long zfp_uint2int_long(ulong x)
{
  return (long)((x ^ 0xaaaaaaaaaaaaaaaaul) - 0xaaaaaaaaaaaaaaaaul);
}

static inline void zfp_fwd_lift1_long(thread long* p)
{
  long x = p[0];
  long y = p[1];
  long z = p[2];
  long w = p[3];
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;
  p[0] = x;
  p[1] = y;
  p[2] = z;
  p[3] = w;
}

static inline void zfp_inv_lift1_long(thread long* p)
{
  long x = p[0];
  long y = p[1];
  long z = p[2];
  long w = p[3];
  y += w >> 1; w -= y >> 1;
  y += w; w -= y - w;
  z += x; x -= z - x;
  y += z; z -= y - z;
  w += x; x -= w - x;
  p[0] = x;
  p[1] = y;
  p[2] = z;
  p[3] = w;
}

static inline void zfp_fwd_lift2_row_long(thread long* p)
{
  zfp_fwd_lift1_long(p + 0);
  zfp_fwd_lift1_long(p + 4);
  zfp_fwd_lift1_long(p + 8);
  zfp_fwd_lift1_long(p + 12);
}

static inline void zfp_fwd_lift2_col_long(thread long* p)
{
  long c0[4] = { p[0], p[4], p[8], p[12] };
  long c1[4] = { p[1], p[5], p[9], p[13] };
  long c2[4] = { p[2], p[6], p[10], p[14] };
  long c3[4] = { p[3], p[7], p[11], p[15] };
  zfp_fwd_lift1_long(c0); zfp_fwd_lift1_long(c1); zfp_fwd_lift1_long(c2); zfp_fwd_lift1_long(c3);
  p[0] = c0[0]; p[4] = c0[1]; p[8] = c0[2]; p[12] = c0[3];
  p[1] = c1[0]; p[5] = c1[1]; p[9] = c1[2]; p[13] = c1[3];
  p[2] = c2[0]; p[6] = c2[1]; p[10] = c2[2]; p[14] = c2[3];
  p[3] = c3[0]; p[7] = c3[1]; p[11] = c3[2]; p[15] = c3[3];
}

static inline void zfp_inv_lift2_col_long(thread long* p)
{
  long c0[4] = { p[0], p[4], p[8], p[12] };
  long c1[4] = { p[1], p[5], p[9], p[13] };
  long c2[4] = { p[2], p[6], p[10], p[14] };
  long c3[4] = { p[3], p[7], p[11], p[15] };
  zfp_inv_lift1_long(c0); zfp_inv_lift1_long(c1); zfp_inv_lift1_long(c2); zfp_inv_lift1_long(c3);
  p[0] = c0[0]; p[4] = c0[1]; p[8] = c0[2]; p[12] = c0[3];
  p[1] = c1[0]; p[5] = c1[1]; p[9] = c1[2]; p[13] = c1[3];
  p[2] = c2[0]; p[6] = c2[1]; p[10] = c2[2]; p[14] = c2[3];
  p[3] = c3[0]; p[7] = c3[1]; p[11] = c3[2]; p[15] = c3[3];
}

static inline void zfp_inv_lift2_row_long(thread long* p)
{
  zfp_inv_lift1_long(p + 0);
  zfp_inv_lift1_long(p + 4);
  zfp_inv_lift1_long(p + 8);
  zfp_inv_lift1_long(p + 12);
}

static inline void zfp_fwd_lift3_long(thread long* p)
{
  for (uint z = 0; z < 4u; ++z)
    for (uint y = 0; y < 4u; ++y)
      zfp_fwd_lift1_long(p + 4u * y + 16u * z);

  for (uint x = 0; x < 4u; ++x)
    for (uint z = 0; z < 4u; ++z) {
      long c[4] = { p[16u * z + 1u * x], p[16u * z + 4u + 1u * x], p[16u * z + 8u + 1u * x], p[16u * z + 12u + 1u * x] };
      zfp_fwd_lift1_long(c);
      p[16u * z + 1u * x] = c[0];
      p[16u * z + 4u + 1u * x] = c[1];
      p[16u * z + 8u + 1u * x] = c[2];
      p[16u * z + 12u + 1u * x] = c[3];
    }

  for (uint y = 0; y < 4u; ++y)
    for (uint x = 0; x < 4u; ++x) {
      long c[4] = { p[1u * x + 4u * y], p[16u + 1u * x + 4u * y], p[32u + 1u * x + 4u * y], p[48u + 1u * x + 4u * y] };
      zfp_fwd_lift1_long(c);
      p[1u * x + 4u * y] = c[0];
      p[16u + 1u * x + 4u * y] = c[1];
      p[32u + 1u * x + 4u * y] = c[2];
      p[48u + 1u * x + 4u * y] = c[3];
    }
}

static inline void zfp_inv_lift3_long(thread long* p)
{
  for (uint y = 0; y < 4u; ++y)
    for (uint x = 0; x < 4u; ++x) {
      long c[4] = { p[1u * x + 4u * y], p[16u + 1u * x + 4u * y], p[32u + 1u * x + 4u * y], p[48u + 1u * x + 4u * y] };
      zfp_inv_lift1_long(c);
      p[1u * x + 4u * y] = c[0];
      p[16u + 1u * x + 4u * y] = c[1];
      p[32u + 1u * x + 4u * y] = c[2];
      p[48u + 1u * x + 4u * y] = c[3];
    }

  for (uint x = 0; x < 4u; ++x)
    for (uint z = 0; z < 4u; ++z) {
      long c[4] = { p[16u * z + 1u * x], p[16u * z + 4u + 1u * x], p[16u * z + 8u + 1u * x], p[16u * z + 12u + 1u * x] };
      zfp_inv_lift1_long(c);
      p[16u * z + 1u * x] = c[0];
      p[16u * z + 4u + 1u * x] = c[1];
      p[16u * z + 8u + 1u * x] = c[2];
      p[16u * z + 12u + 1u * x] = c[3];
    }

  for (uint z = 0; z < 4u; ++z)
    for (uint y = 0; y < 4u; ++y)
      zfp_inv_lift1_long(p + 4u * y + 16u * z);
}

struct ZfpBlockWriter1 {
  device ulong* stream64;
  device atomic_uint* stream_atomic;
  ulong buf;        /* 64-bit accumulation buffer */
  uint buf_bits;    /* how many valid bits in buf (0..64) */
  uint wi64;        /* current 64-bit word index in stream */
  ulong start_bit;  /* absolute bit offset (for atomic mode) */
  uint total_bits;  /* bits written so far */
  bool atomic_mode;
};

static inline ZfpBlockWriter1 zfp_make_writer1(device ulong* stream, uint maxbits, uint block_idx, bool atomic_mode)
{
  ZfpBlockWriter1 w;
  ulong bit0 = (ulong)block_idx * (ulong)maxbits;
  w.stream64 = stream;
  w.stream_atomic = reinterpret_cast<device atomic_uint*>(stream);
  w.atomic_mode = atomic_mode;
  w.start_bit = bit0;
  w.total_bits = 0;

  if (!atomic_mode) {
    /* Align to 64-bit word boundary */
    w.wi64 = (uint)(bit0 >> 6u);
    uint bo = (uint)(bit0 & 63u);
    w.buf = 0ul;
    w.buf_bits = bo;
  } else {
    w.wi64 = 0;
    w.buf = 0ul;
    w.buf_bits = 0u;
  }
  return w;
}

static inline void zfp_writer_flush(thread ZfpBlockWriter1& w)
{
  if (!w.atomic_mode && w.buf_bits > 0u) {
    w.stream64[w.wi64] |= w.buf;
    w.buf = 0ul;
  }
}

static inline ulong zfp_writer_write_bits(thread ZfpBlockWriter1& w, ulong bits, uint nbits)
{
  if (nbits == 0u) return bits;
  ulong keep = bits & ((nbits >= 64u) ? ~0ul : ((1ul << nbits) - 1ul));

  if (!w.atomic_mode) {
    /* Fast path: accumulate into 64-bit buffer */
    w.buf |= (keep << w.buf_bits);
    w.buf_bits += nbits;
    if (w.buf_bits >= 64u) {
      w.stream64[w.wi64] |= w.buf;
      w.wi64 += 1u;
      uint overflow = w.buf_bits - 64u;
      /* The bits that overflowed into the next word */
      w.buf = (overflow > 0u) ? (keep >> (nbits - overflow)) : 0ul;
      w.buf_bits = overflow;
    }
  } else {
    /* Atomic path: write 32-bit chunks via atomic OR */
    ulong pos = w.start_bit + (ulong)w.total_bits;
    uint remaining = nbits;
    ulong val = keep;
    while (remaining) {
      uint wi32 = (uint)(pos >> 5u);
      uint bo = (uint)(pos & 31u);
      uint chunk = min(remaining, 32u - bo);
      ulong cmask = (chunk >= 32u) ? 0xfffffffful : ((1ul << chunk) - 1ul);
      uint part = (uint)((val & cmask) << bo);
      atomic_fetch_or_explicit(&(w.stream_atomic[wi32]), part, memory_order_relaxed);
      val >>= chunk;
      remaining -= chunk;
      pos += chunk;
    }
  }

  w.total_bits += nbits;
  return bits >> nbits;
}

static inline uint zfp_writer_write_bit(thread ZfpBlockWriter1& w, uint bit)
{
  if (!w.atomic_mode) {
    w.buf |= (ulong)(bit & 1u) << w.buf_bits;
    w.buf_bits += 1u;
    if (w.buf_bits >= 64u) {
      w.stream64[w.wi64] |= w.buf;
      w.wi64 += 1u;
      w.buf = 0ul;
      w.buf_bits = 0u;
    }
    w.total_bits += 1u;
  } else {
    zfp_writer_write_bits(w, (ulong)(bit & 1u), 1u);
  }
  return bit & 1u;
}

struct ZfpBlockReader1 {
  device const ulong* words;
  ulong buffer;
  uint current_bit;
};

static inline ZfpBlockReader1 zfp_make_reader1(device const ulong* stream, uint maxbits, uint block_idx)
{
  ZfpBlockReader1 r;
  ulong bit0 = (ulong)block_idx * (ulong)maxbits;
  uint wi = (uint)(bit0 / 64ul);
  r.words = stream + wi;
  r.buffer = r.words[0];
  r.current_bit = (uint)(bit0 % 64ul);
  r.buffer >>= r.current_bit;
  return r;
}

static inline uint zfp_reader_read_bit(thread ZfpBlockReader1& r)
{
  uint b = (uint)(r.buffer & 1ul);
  r.current_bit += 1u;
  r.buffer >>= 1u;
  if (r.current_bit >= 64u) {
    r.current_bit = 0u;
    r.words += 1;
    r.buffer = r.words[0];
  }
  return b;
}

/* Peek at the next 'count' bits without consuming them.
   count must be <= 64 and <= remaining bits in buffer + next word. */
static inline ulong zfp_reader_peek_bits(thread ZfpBlockReader1& r, uint count)
{
  if (count == 0u) return 0ul;
  uint rem = 64u - r.current_bit;
  if (count <= rem) {
    ulong mask = (count >= 64u) ? ~0ul : ((1ul << count) - 1ul);
    return r.buffer & mask;
  }
  /* Need bits from next word too */
  ulong lo = r.buffer; /* rem bits available */
  ulong hi = r.words[1];
  uint hi_bits = count - rem;
  ulong hi_mask = (hi_bits >= 64u) ? ~0ul : ((1ul << hi_bits) - 1ul);
  return lo | ((hi & hi_mask) << rem);
}

/* Skip (consume) 'count' bits that were already peeked at. */
static inline void zfp_reader_skip(thread ZfpBlockReader1& r, uint count)
{
  r.current_bit += count;
  if (r.current_bit >= 64u) {
    r.words += 1;
    r.buffer = r.words[0];
    r.current_bit -= 64u;
    r.buffer >>= r.current_bit;
  } else {
    r.buffer >>= count;
  }
}

static inline ulong zfp_reader_read_bits(thread ZfpBlockReader1& r, uint nbits)
{
  uint rem = 64u - r.current_bit;
  uint first = min(rem, nbits);
  ulong mask = (first == 64u) ? ~0ul : ((1ul << first) - 1ul);
  ulong bits = r.buffer & mask;
  r.buffer >>= first;
  r.current_bit += first;
  if (nbits >= rem) {
    r.words += 1;
    r.buffer = r.words[0];
    r.current_bit = 0u;
  }
  uint next = nbits - first;
  mask = (next == 64u) ? ~0ul : ((next == 0u) ? 0ul : ((1ul << next) - 1ul));
  bits |= (r.buffer & mask) << first;
  r.buffer >>= next;
  r.current_bit += next;
  return bits;
}

static inline void zfp_encode_block_1d_float(thread float* fblock, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  int emax = zfp_exponent_float(max(max(fabs(fblock[0]), fabs(fblock[1])), max(fabs(fblock[2]), fabs(fblock[3]))));
  int maxprec = zfp_precision(emax, 32, -1074);
  uint e = maxprec ? (uint)(emax + 127) : 0u;
  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  if (!e)
    return;
  zfp_writer_write_bits(w, (ulong)(2u * e + 1u), 9u);

  float s = ldexp(1.0f, 30 - emax);
  int iblock[4];
  iblock[0] = (int)(s * fblock[0]);
  iblock[1] = (int)(s * fblock[1]);
  iblock[2] = (int)(s * fblock[2]);
  iblock[3] = (int)(s * fblock[3]);

  zfp_fwd_lift1(iblock);

  uint ublock[4];
  ublock[0] = zfp_int2uint(iblock[0]);
  ublock[1] = zfp_int2uint(iblock[1]);
  ublock[2] = zfp_int2uint(iblock[2]);
  ublock[3] = zfp_int2uint(iblock[3]);

  uint bits = maxbits - 9u;
  uint kmin = 32u > (uint)maxprec ? 32u - (uint)maxprec : 0u;
  uint n = 0u;
  for (uint k = 32u; bits && k-- > kmin;) {
    ulong x = 0ul;
    x += (ulong)((ublock[0] >> k) & 1u) << 0u;
    x += (ulong)((ublock[1] >> k) & 1u) << 1u;
    x += (ulong)((ublock[2] >> k) & 1u) << 2u;
    x += (ulong)((ublock[3] >> k) & 1u) << 3u;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 4u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(3u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 3u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

static inline void zfp_decode_block_1d_float(device const ulong* stream, uint maxbits, uint block_idx, thread float* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  int emax = (int)e - 127;
  uint bits = maxbits - 9u;

  uint ublock[4] = {0u, 0u, 0u, 0u};
  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 4u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(3u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 4u;
        break;
      }
    }
    ublock[0] += (uint)(x & 1ul) << k; x >>= 1u;
    ublock[1] += (uint)(x & 1ul) << k; x >>= 1u;
    ublock[2] += (uint)(x & 1ul) << k; x >>= 1u;
    ublock[3] += (uint)(x & 1ul) << k;
  }

  int iblock[4];
  iblock[0] = zfp_uint2int(ublock[0]);
  iblock[1] = zfp_uint2int(ublock[1]);
  iblock[2] = zfp_uint2int(ublock[2]);
  iblock[3] = zfp_uint2int(ublock[3]);
  zfp_inv_lift1(iblock);
  float inv_w = ldexp(1.0f, emax - 30);
  out[0] = inv_w * (float)iblock[0];
  out[1] = inv_w * (float)iblock[1];
  out[2] = inv_w * (float)iblock[2];
  out[3] = inv_w * (float)iblock[3];
}

static inline void zfp_encode_block_2d_float(thread float* fblock, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  float a0 = fabs(fblock[0]);
  for (uint i = 1; i < 16; ++i)
    a0 = max(a0, fabs(fblock[i]));

  int emax = zfp_exponent_float(a0);
  int maxprec = zfp_precision(emax, 32, -1074);
  uint e = maxprec ? (uint)(emax + 127) : 0u;
  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  if (!e)
    return;

  zfp_writer_write_bits(w, (ulong)(2u * e + 1u), 9u);

  float s = ldexp(1.0f, 30 - emax);
  int ib[16];
  for (uint i = 0; i < 16; ++i)
    ib[i] = (int)(s * fblock[i]);

  zfp_fwd_lift2_row(ib);
  zfp_fwd_lift2_col(ib);

  uint ub[16];
  for (uint i = 0; i < 16; ++i)
    ub[i] = zfp_int2uint(ib[zfp_perm2[i]]);

  uint bits = maxbits - 9u;
  uint kmin = 32u > (uint)maxprec ? 32u - (uint)maxprec : 0u;
  uint n = 0u;
  for (uint k = 32u; bits && k-- > kmin;) {
    /* 2D block: only 16 bits needed; gather into uint to avoid 64-bit shifts */
    uint x_lo = 0u;
    for (uint i = 0; i < 16; ++i)
      x_lo += ((ub[i] >> k) & 1u) << i;
    ulong x = (ulong)x_lo;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 16u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(15u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 15u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

static inline void zfp_decode_block_2d_float(device const ulong* stream, uint maxbits, uint block_idx, thread float* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    for (uint i = 0; i < 16; ++i)
      out[i] = 0.0f;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  int emax = (int)e - 127;
  uint bits = maxbits - 9u;

  uint ub[16];
  for (uint i = 0; i < 16; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 16u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(15u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 16u;
        break;
      }
    }
    /* 2D block: only low 16 bits set; use uint scatter to avoid 64-bit shifts */
    uint x_lo = (uint)(x & 0xFFFFul);
    for (uint i = 0; i < 16; ++i) {
      ub[i] += (x_lo & 1u) << k;
      x_lo >>= 1u;
    }
  }

  /* Permute + uint2int directly into out[] (reinterpreted as int[]),
     avoiding a separate ib[16] array to reduce register pressure. */
  thread int* ip = (thread int*)out;
  for (uint i = 0; i < 16; ++i)
    ip[zfp_perm2[i]] = zfp_uint2int(ub[i]);

  zfp_inv_lift2_col(ip);
  zfp_inv_lift2_row(ip);

  float inv_w = ldexp(1.0f, emax - 30);
  for (uint i = 0; i < 16; ++i)
    out[i] = inv_w * (float)ip[i];
}

static inline void zfp_encode_block_3d_float(thread float* fblock, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  float a0 = fabs(fblock[0]);
  for (uint i = 1; i < 64; ++i)
    a0 = max(a0, fabs(fblock[i]));

  int emax = zfp_exponent_float(a0);
  int maxprec = zfp_precision(emax, 32, -1074);
  uint e = maxprec ? (uint)(emax + 127) : 0u;
  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  if (!e)
    return;

  zfp_writer_write_bits(w, (ulong)(2u * e + 1u), 9u);

  float s = ldexp(1.0f, 30 - emax);
  int ib[64];
  for (uint i = 0; i < 64; ++i)
    ib[i] = (int)(s * fblock[i]);

  zfp_fwd_lift3(ib);

  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = zfp_int2uint(ib[zfp_perm3[i]]);

  uint bits = maxbits - 9u;
  uint kmin = 32u > (uint)maxprec ? 32u - (uint)maxprec : 0u;
  uint n = 0u;
  for (uint k = 32u; bits && k-- > kmin;) {
    /* Split-32 gather: build x from two 32-bit halves to avoid costly
       64-bit shift emulation on Apple Silicon's 32-bit ALUs */
    uint x_lo = 0u;
    uint x_hi = 0u;
    for (uint i = 0; i < 32; ++i)
      x_lo += ((ub[i] >> k) & 1u) << i;
    for (uint i = 32; i < 64; ++i)
      x_hi += ((ub[i] >> k) & 1u) << (i - 32u);
    ulong x = (ulong)x_lo | ((ulong)x_hi << 32u);
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 64u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u); /* group test: significant bits remain */
      /* Find run of zeros before next significant coefficient */
      uint z = (uint)ctz(x);
      uint inner_max = min(63u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        /* Found the coefficient within budget */
        if (z < 63u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        /* Ran out of inner budget (n reached 63 or bits exhausted) */
        x >>= run;
        n += run;
        /* The coefficient at position n is implicitly significant */
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

static inline void zfp_decode_block_3d_float(device const ulong* stream, uint maxbits, uint block_idx, thread float* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    for (uint i = 0; i < 64; ++i)
      out[i] = 0.0f;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  int emax = (int)e - 127;
  uint bits = maxbits - 9u;

  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        /* Group test passed: find next significant coefficient */
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            /* Found: consume the '1' bit */
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    /* Split-32 scatter: use 32-bit ops to avoid costly 64-bit shift
       emulation on Apple Silicon's 32-bit ALUs */
    uint x_lo = (uint)(x & 0xFFFFFFFFul);
    uint x_hi = (uint)(x >> 32u);
    for (uint i = 0; i < 32; ++i) {
      ub[i] += (x_lo & 1u) << k;
      x_lo >>= 1u;
    }
    for (uint i = 32; i < 64; ++i) {
      ub[i] += (x_hi & 1u) << k;
      x_hi >>= 1u;
    }
  }

  /* Permute + uint2int directly into out[] (reinterpreted as int[]),
     avoiding a separate ib[64] array to reduce register pressure. */
  thread int* ip = (thread int*)out;
  for (uint i = 0; i < 64; ++i)
    ip[zfp_perm3[i]] = zfp_uint2int(ub[i]);

  zfp_inv_lift3(ip);

  float inv_w = ldexp(1.0f, emax - 30);
  for (uint i = 0; i < 64; ++i)
    out[i] = inv_w * (float)ip[i];
}

/* ========================================================================== */
/* Threadgroup-memory bitstream reader (Phase B prefetch optimization)        */
/* ========================================================================== */

struct ZfpTgBlockReader1 {
  threadgroup const ulong* words;
  ulong buffer;
  uint current_bit;
};

static inline ZfpTgBlockReader1 zfp_make_tg_reader1(threadgroup const ulong* base, uint offset_bits)
{
  ZfpTgBlockReader1 r;
  uint wi = offset_bits / 64u;
  r.words = base + wi;
  r.buffer = r.words[0];
  r.current_bit = offset_bits % 64u;
  r.buffer >>= r.current_bit;
  return r;
}

static inline uint zfp_tg_reader_read_bit(thread ZfpTgBlockReader1& r)
{
  uint b = (uint)(r.buffer & 1ul);
  r.current_bit += 1u;
  r.buffer >>= 1u;
  if (r.current_bit >= 64u) {
    r.current_bit = 0u;
    r.words += 1;
    r.buffer = r.words[0];
  }
  return b;
}

static inline ulong zfp_tg_reader_peek_bits(thread ZfpTgBlockReader1& r, uint count)
{
  if (count == 0u) return 0ul;
  uint rem = 64u - r.current_bit;
  if (count <= rem) {
    ulong mask = (count >= 64u) ? ~0ul : ((1ul << count) - 1ul);
    return r.buffer & mask;
  }
  ulong lo = r.buffer;
  ulong hi = r.words[1];
  uint hi_bits = count - rem;
  ulong hi_mask = (hi_bits >= 64u) ? ~0ul : ((1ul << hi_bits) - 1ul);
  return lo | ((hi & hi_mask) << rem);
}

static inline void zfp_tg_reader_skip(thread ZfpTgBlockReader1& r, uint count)
{
  r.current_bit += count;
  if (r.current_bit >= 64u) {
    r.words += 1;
    r.buffer = r.words[0];
    r.current_bit -= 64u;
    r.buffer >>= r.current_bit;
  } else {
    r.buffer >>= count;
  }
}

static inline ulong zfp_tg_reader_read_bits(thread ZfpTgBlockReader1& r, uint nbits)
{
  uint rem = 64u - r.current_bit;
  uint first = min(rem, nbits);
  ulong mask = (first == 64u) ? ~0ul : ((1ul << first) - 1ul);
  ulong bits = r.buffer & mask;
  r.buffer >>= first;
  r.current_bit += first;
  if (nbits >= rem) {
    r.words += 1;
    r.buffer = r.words[0];
    r.current_bit = 0u;
  }
  uint next = nbits - first;
  mask = (next == 64u) ? ~0ul : ((next == 0u) ? 0ul : ((1ul << next) - 1ul));
  bits |= (r.buffer & mask) << first;
  r.buffer >>= next;
  r.current_bit += next;
  return bits;
}

/* Decode a 3D float block from threadgroup memory (Phase B prefetch). */
static inline void zfp_decode_block_3d_float_tg(threadgroup const ulong* tg_base, uint offset_bits, uint maxbits, thread float* out)
{
  ZfpTgBlockReader1 r = zfp_make_tg_reader1(tg_base, offset_bits);
  uint s_cont = zfp_tg_reader_read_bit(r);
  if (!s_cont) {
    for (uint i = 0; i < 64; ++i)
      out[i] = 0.0f;
    return;
  }

  uint e = (uint)zfp_tg_reader_read_bits(r, 8u);
  int emax = (int)e - 127;
  uint bits = maxbits - 9u;

  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_tg_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_tg_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_tg_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_tg_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_tg_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    for (uint i = 0; i < 64; ++i) {
      ub[i] += (uint)(x & 1ul) << k;
      x >>= 1u;
    }
  }

  /* Permute + uint2int directly into out[] (reinterpreted as int[]),
     avoiding a separate ib[64] array to reduce register pressure. */
  thread int* ip = (thread int*)out;
  for (uint i = 0; i < 64; ++i)
    ip[zfp_perm3[i]] = zfp_uint2int(ub[i]);

  zfp_inv_lift3(ip);

  float inv_w = ldexp(1.0f, emax - 30);
  for (uint i = 0; i < 64; ++i)
    out[i] = inv_w * (float)ip[i];
}

kernel void zfp_encode1d_float(
  device const float* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec1dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint x = block_idx * 4u;
  long offset = (long)x * (long)p.sx;
  float fblock[4];

  if (x + 4u > p.dim) {
    uint nx = p.dim - x;
    for (uint i = 0; i < 4u; ++i)
      fblock[i] = i < nx ? src[offset + (long)i * (long)p.sx] : 0.0f;
    if (nx == 0u) {
      fblock[1] = fblock[0];
      fblock[2] = fblock[1];
      fblock[3] = fblock[0];
    }
    else if (nx == 1u) {
      fblock[1] = fblock[0];
      fblock[2] = fblock[1];
      fblock[3] = fblock[0];
    }
    else if (nx == 2u) {
      fblock[2] = fblock[1];
      fblock[3] = fblock[0];
    }
    else if (nx == 3u) {
      fblock[3] = fblock[0];
    }
  }
  else {
    fblock[0] = src[offset + 0l * (long)p.sx];
    fblock[1] = src[offset + 1l * (long)p.sx];
    fblock[2] = src[offset + 2l * (long)p.sx];
    fblock[3] = src[offset + 3l * (long)p.sx];
  }

  zfp_encode_block_1d_float(fblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode1d_float(
  device const ulong* stream [[buffer(0)]],
  device float* out [[buffer(1)]],
  constant Codec1dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  float fblock[4];
  zfp_decode_block_1d_float(stream, p.maxbits, block_idx, fblock);

  uint x = block_idx * 4u;
  long offset = (long)x * (long)p.sx;
  if (x + 4u > p.dim) {
    uint nx = p.dim - x;
    for (uint i = 0; i < nx; ++i)
      out[offset + (long)i * (long)p.sx] = fblock[i];
  }
  else {
    out[offset + 0l * (long)p.sx] = fblock[0];
    out[offset + 1l * (long)p.sx] = fblock[1];
    out[offset + 2l * (long)p.sx] = fblock[2];
    out[offset + 3l * (long)p.sx] = fblock[3];
  }
}

kernel void zfp_encode2d_float(
  device const float* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec2dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint bx = block_idx % p.bx;
  uint by = block_idx / p.bx;
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy;

  float fblock[16];
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        fblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy];
  }
  else {
    for (uint y = 0; y < 4u; ++y) {
      for (uint x = 0; x < 4u; ++x) {
        uint ax = x0 + x;
        uint ay = y0 + y;
        if (ax < p.nx && ay < p.ny)
          fblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy];
        else
          fblock[4u * y + x] = 0.0f;
      }
    }

    uint nx = x0 + 4u > p.nx ? p.nx - x0 : 4u;
    uint ny = y0 + 4u > p.ny ? p.ny - y0 : 4u;
    for (uint y = 0; y < 4u; ++y)
      if (y < ny)
        for (uint x = nx; x < 4u; ++x)
          fblock[4u * y + x] = fblock[4u * y + nx - 1u];
    for (uint x = 0; x < 4u; ++x)
      for (uint y = ny; y < 4u; ++y)
        fblock[4u * y + x] = fblock[4u * (ny - 1u) + x];
  }

  zfp_encode_block_2d_float(fblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode2d_float(
  device const ulong* stream [[buffer(0)]],
  device float* dst [[buffer(1)]],
  constant Codec2dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  uint bx = block_idx % p.bx;
  uint by = block_idx / p.bx;
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy;

  float fblock[16];
  zfp_decode_block_2d_float(stream, p.maxbits, block_idx, fblock);

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        dst[base + (long)x * p.sx + (long)y * p.sy] = fblock[4u * y + x];
  }
  else {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        if (x0 + x < p.nx && y0 + y < p.ny)
          dst[base + (long)x * p.sx + (long)y * p.sy] = fblock[4u * y + x];
  }
}

kernel void zfp_encode3d_float(
  device const float* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint bx = block_idx % p.bx;
  uint by = (block_idx / p.bx) % p.by;
  uint bz = block_idx / (p.bx * p.by);
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  uint z0 = bz * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  float fblock[64];
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint idx = x + 4u * (y + 4u * z);
          fblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz];
        }
  }
  else {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint ax = x0 + x;
          uint ay = y0 + y;
          uint az = z0 + z;
          uint idx = x + 4u * (y + 4u * z);
          if (ax < p.nx && ay < p.ny && az < p.nz)
            fblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz];
          else
            fblock[idx] = 0.0f;
        }
  }

  zfp_encode_block_3d_float(fblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode3d_float(
  device const ulong* stream [[buffer(0)]],
  device float* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  uint bx = block_idx % p.bx;
  uint by = (block_idx / p.bx) % p.by;
  uint bz = block_idx / (p.bx * p.by);
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  uint z0 = bz * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  float fblock[64];
  zfp_decode_block_3d_float(stream, p.maxbits, block_idx, fblock);

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint idx = x + 4u * (y + 4u * z);
          dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = fblock[idx];
        }
  }
  else {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x)
          if (x0 + x < p.nx && y0 + y < p.ny && z0 + z < p.nz) {
            uint idx = x + 4u * (y + 4u * z);
            dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = fblock[idx];
          }
  }
}

/* ========================================================================== */
/* Threadgroup-memory ub[] decode: reduces register pressure by keeping the   */
/* 64-element accumulation buffer in threadgroup SRAM instead of registers.   */
/* Each thread still processes one block, but ub[64] lives in TG memory.     */
/* ========================================================================== */

/* Inverse 1D lifting transform operating on threadgroup memory */
static inline void zfp_inv_lift1_tg(threadgroup int* p)
{
  int x = p[0];
  int y = p[1];
  int z = p[2];
  int w = p[3];
  y += w >> 1; w -= y >> 1;
  y += w; w -= y - w;
  z += x; x -= z - x;
  y += z; z -= y - z;
  w += x; x -= w - x;
  p[0] = x;
  p[1] = y;
  p[2] = z;
  p[3] = w;
}

/* 3D inverse lifting transform operating on threadgroup memory */
static inline void zfp_inv_lift3_tg(threadgroup int* p)
{
  for (uint y = 0; y < 4u; ++y)
    for (uint x = 0; x < 4u; ++x) {
      int c[4] = { p[1u * x + 4u * y], p[16u + 1u * x + 4u * y], p[32u + 1u * x + 4u * y], p[48u + 1u * x + 4u * y] };
      zfp_inv_lift1(c);
      p[1u * x + 4u * y] = c[0];
      p[16u + 1u * x + 4u * y] = c[1];
      p[32u + 1u * x + 4u * y] = c[2];
      p[48u + 1u * x + 4u * y] = c[3];
    }

  for (uint x = 0; x < 4u; ++x)
    for (uint z = 0; z < 4u; ++z) {
      int c[4] = { p[16u * z + 1u * x], p[16u * z + 4u + 1u * x], p[16u * z + 8u + 1u * x], p[16u * z + 12u + 1u * x] };
      zfp_inv_lift1(c);
      p[16u * z + 1u * x] = c[0];
      p[16u * z + 4u + 1u * x] = c[1];
      p[16u * z + 8u + 1u * x] = c[2];
      p[16u * z + 12u + 1u * x] = c[3];
    }

  for (uint z = 0; z < 4u; ++z)
    for (uint y = 0; y < 4u; ++y)
      zfp_inv_lift1_tg(p + 4u * y + 16u * z);
}

/* Decode one 3D float block with ub[] in threadgroup memory.
   Takes a pointer to a 64-element threadgroup int/uint region.
   Writes decoded float values into the same region (reinterpreted). */
static inline void zfp_decode_block_3d_float_tgub(
  device const ulong* stream, uint maxbits, uint block_idx,
  threadgroup uint* tg_ub)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    for (uint i = 0; i < 64; ++i)
      ((threadgroup float*)tg_ub)[i] = 0.0f;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  int emax = (int)e - 127;
  uint bits = maxbits - 9u;

  /* Zero-init ub in threadgroup memory */
  for (uint i = 0; i < 64; ++i)
    tg_ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    /* Scatter bitplane to threadgroup memory */
    for (uint i = 0; i < 64; ++i) {
      tg_ub[i] += (uint)(x & 1ul) << k;
      x >>= 1u;
    }
  }

  /* Permute + uint2int into the same TG region (reinterpreted as int) */
  threadgroup int* ip = (threadgroup int*)tg_ub;
  /* Need a temp copy since permute is not in-place */
  uint ub_copy[64];
  for (uint i = 0; i < 64; ++i)
    ub_copy[i] = tg_ub[i];
  for (uint i = 0; i < 64; ++i)
    ip[zfp_perm3[i]] = zfp_uint2int(ub_copy[i]);

  /* Inverse lifting transform in threadgroup memory */
  zfp_inv_lift3_tg(ip);

  /* Dequantize: write float results into same TG region */
  threadgroup float* fp = (threadgroup float*)tg_ub;
  float inv_w = ldexp(1.0f, emax - 30);
  for (uint i = 0; i < 64; ++i)
    fp[i] = inv_w * (float)ip[i];
}

/* 3D float decode kernel using threadgroup memory for ub[].
   Each thread in the threadgroup processes one independent ZFP block,
   but its ub[64] array lives in threadgroup SRAM to free register space.
   Threadgroup size should match the number of blocks per TG (e.g., 32). */
#define TGUB_BLOCKS_PER_TG 32u
kernel void zfp_decode3d_float_tgub(
  device const ulong* stream [[buffer(0)]],
  device float* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]],
  threadgroup uint* tg_mem [[threadgroup(0)]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  /* Each thread gets its own 64-uint region in threadgroup memory */
  threadgroup uint* my_ub = tg_mem + lid * 64u;

  /* Decode the block */
  zfp_decode_block_3d_float_tgub(stream, p.maxbits, block_idx, my_ub);

  /* Write output to device memory */
  threadgroup float* my_fp = (threadgroup float*)my_ub;

  uint bx_idx = block_idx % p.bx;
  uint by_idx = (block_idx / p.bx) % p.by;
  uint bz_idx = block_idx / (p.bx * p.by);
  uint x0 = bx_idx * 4u;
  uint y0 = by_idx * 4u;
  uint z0 = bz_idx * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint zz = 0; zz < 4u; ++zz)
      for (uint yy = 0; yy < 4u; ++yy)
        for (uint xx = 0; xx < 4u; ++xx) {
          uint idx = xx + 4u * (yy + 4u * zz);
          dst[base + (long)xx * p.sx + (long)yy * p.sy + (long)zz * p.sz] = my_fp[idx];
        }
  }
  else {
    for (uint zz = 0; zz < 4u; ++zz)
      for (uint yy = 0; yy < 4u; ++yy)
        for (uint xx = 0; xx < 4u; ++xx)
          if (x0 + xx < p.nx && y0 + yy < p.ny && z0 + zz < p.nz) {
            uint idx = xx + 4u * (yy + 4u * zz);
            dst[base + (long)xx * p.sx + (long)yy * p.sy + (long)zz * p.sz] = my_fp[idx];
          }
  }
}

/* ========================================================================== */
/* Diagnostic kernels: isolate bitplane decode vs transform cost              */
/* Enabled only with ZFP_METAL_CODEC_EXPERIMENTAL.                           */
/* ========================================================================== */

/* Diagnostic 1: Bitplane decode only.
   Reads compressed stream, decodes bitplanes into ub[64] (uint), writes raw
   uint output. Skips permute, uint2int, lifting, dequant. Measures the serial
   bitplane decode cost in isolation. */
kernel void zfp_diag_bitplane_only_3d(
  device const ulong* stream [[buffer(0)]],
  device uint* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= p.total_blocks)
    return;

  ZfpBlockReader1 r = zfp_make_reader1(stream, p.maxbits, gid);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    for (uint i = 0; i < 64; ++i)
      dst[(ulong)gid * 64ul + i] = 0u;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  (void)e; /* not needed for bitplane-only */
  uint bits = p.maxbits - 9u;

  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    for (uint i = 0; i < 64; ++i) {
      ub[i] += (uint)(x & 1ul) << k;
      x >>= 1u;
    }
  }

  /* Write raw uint values (no permute, no lift, no dequant) */
  for (uint i = 0; i < 64; ++i)
    dst[(ulong)gid * 64ul + i] = ub[i];
}

/* Diagnostic 2: Transform only (permute + uint2int + inverse lift + dequant).
   Reads 64 uint values per block from input, applies the ZFP inverse transform
   pipeline, writes float output. Measures the cost of the post-decode math. */
kernel void zfp_diag_transform_only_3d(
  device const uint* src [[buffer(0)]],
  device float* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= p.total_blocks)
    return;

  ulong base = (ulong)gid * 64ul;
  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = src[base + i];

  /* Permute + uint2int into float array (reinterpreted as int) */
  float out[64];
  thread int* ip = (thread int*)out;
  for (uint i = 0; i < 64; ++i)
    ip[zfp_perm3[i]] = zfp_uint2int(ub[i]);

  /* Inverse lifting transform */
  zfp_inv_lift3(ip);

  /* Dequantize: use emax=0 for diagnostic (just tests the multiply path) */
  float inv_w = ldexp(1.0f, -30);
  for (uint i = 0; i < 64; ++i)
    out[i] = inv_w * (float)ip[i];

  /* Write output */
  for (uint i = 0; i < 64; ++i)
    dst[base + i] = out[i];
}

/* Diagnostic 3: Memory-only (read + write, no compute).
   Measures raw memory throughput as a baseline reference. */
kernel void zfp_diag_memcopy_3d(
  device const uint* src [[buffer(0)]],
  device float* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= p.total_blocks)
    return;

  ulong base = (ulong)gid * 64ul;
  for (uint i = 0; i < 64; ++i)
    dst[base + i] = as_type<float>(src[base + i]);
}

/* Diagnostic 4: Bitplane decode using device memory for ub[].
   Same bitstream reading as bitplane_only_3d, but accumulates ub[] into the
   device-memory output buffer directly (instead of thread-private registers).
   Tests whether register pressure from ub[64] is the occupancy bottleneck. */
kernel void zfp_diag_bitplane_devub_3d(
  device const ulong* stream [[buffer(0)]],
  device uint* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= p.total_blocks)
    return;

  ulong base = (ulong)gid * 64ul;

  /* Zero-init output in device memory (used as ub accumulator) */
  for (uint i = 0; i < 64; ++i)
    dst[base + i] = 0u;

  ZfpBlockReader1 r = zfp_make_reader1(stream, p.maxbits, gid);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont)
    return;

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  (void)e;
  uint bits = p.maxbits - 9u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    /* Scatter bitplane to device memory instead of registers */
    for (uint i = 0; i < 64; ++i) {
      dst[base + i] += (uint)(x & 1ul) << k;
      x >>= 1u;
    }
  }
}

/* Diagnostic 5: Bitstream reading only (no ub[] accumulation).
   Reads the exact same bitstream as bitplane_only_3d but discards results.
   Writes a single checksum value per block. Measures pure serial bit-reading
   cost without register pressure from ub[64]. */
kernel void zfp_diag_bitread_only_3d(
  device const ulong* stream [[buffer(0)]],
  device uint* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= p.total_blocks)
    return;

  ZfpBlockReader1 r = zfp_make_reader1(stream, p.maxbits, gid);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    dst[gid] = 0u;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  uint bits = p.maxbits - 9u;
  uint checksum = e;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    checksum ^= (uint)(x & 0xFFFFFFFFul);
  }

  /* Write single checksum (minimal output, minimal register pressure) */
  dst[gid] = checksum;
}

/* Diagnostic 6: Bitplane decode with ub[] in threadgroup memory.
   Same bitstream reading and scatter as bitplane_only_3d, but ub[64]
   lives in threadgroup SRAM instead of registers.
   Tests whether reduced register pressure improves occupancy & throughput. */
kernel void zfp_diag_bitplane_tgub_3d(
  device const ulong* stream [[buffer(0)]],
  device uint* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]],
  uint lid [[thread_index_in_threadgroup]],
  threadgroup uint* tg_mem [[threadgroup(0)]])
{
  if (gid >= p.total_blocks)
    return;

  /* Each thread gets its own 64-uint region in TG memory */
  threadgroup uint* ub = tg_mem + lid * 64u;

  ZfpBlockReader1 r = zfp_make_reader1(stream, p.maxbits, gid);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    for (uint i = 0; i < 64; ++i)
      dst[(ulong)gid * 64ul + i] = 0u;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  (void)e;
  uint bits = p.maxbits - 9u;

  for (uint i = 0; i < 64; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    /* Scatter bitplane to threadgroup memory */
    for (uint i = 0; i < 64; ++i) {
      ub[i] += (uint)(x & 1ul) << k;
      x >>= 1u;
    }
  }

  /* Write from TG memory to device output */
  for (uint i = 0; i < 64; ++i)
    dst[(ulong)gid * 64ul + i] = ub[i];
}

/* Diagnostic 7: Bitplane decode with split-32 scatter.
   Splits the 64-bit bitplane word x into two 32-bit halves and processes
   them with 32-bit arithmetic only. Tests whether 64-bit shift emulation
   is a significant cost on Apple Silicon's 32-bit ALUs. */
kernel void zfp_diag_bitplane_split32_3d(
  device const ulong* stream [[buffer(0)]],
  device uint* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= p.total_blocks)
    return;

  ZfpBlockReader1 r = zfp_make_reader1(stream, p.maxbits, gid);
  uint s_cont = zfp_reader_read_bit(r);
  if (!s_cont) {
    for (uint i = 0; i < 64; ++i)
      dst[(ulong)gid * 64ul + i] = 0u;
    return;
  }

  uint e = (uint)zfp_reader_read_bits(r, 8u);
  (void)e;
  uint bits = p.maxbits - 9u;

  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    /* Split-32 scatter: process lo and hi halves with 32-bit ops only */
    uint x_lo = (uint)(x & 0xFFFFFFFFul);
    uint x_hi = (uint)(x >> 32u);
    for (uint i = 0; i < 32; ++i) {
      ub[i] += (x_lo & 1u) << k;
      x_lo >>= 1u;
    }
    for (uint i = 32; i < 64; ++i) {
      ub[i] += (x_hi & 1u) << k;
      x_hi >>= 1u;
    }
  }

  for (uint i = 0; i < 64; ++i)
    dst[(ulong)gid * 64ul + i] = ub[i];
}

/* ========================================================================== */
/* Phase B: Threadgroup-prefetch 3D float decode kernel                       */
/* Each threadgroup cooperatively loads compressed data into on-chip SRAM,    */
/* then each thread decodes its block from fast threadgroup memory.           */
/* ========================================================================== */

kernel void zfp_decode3d_float_tg(
  device const ulong* stream [[buffer(0)]],
  device float* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]],
  uint lid [[thread_position_in_threadgroup]],
  uint tg_id [[threadgroup_position_in_grid]],
  uint tg_size [[threads_per_threadgroup]])
{
  /* Number of 64-bit words per block = maxbits / 64.
     For rate-8 3D float: maxbits=512 → 8 words per block.
     For rate-16: maxbits=1024 → 16 words per block. */
  uint words_per_block = p.maxbits / 64u;

  /* Threadgroup memory: each thread's block data, laid out contiguously.
     Max allocation: 128 threads * 32 words * 8 bytes = 32768 bytes = 32 KB.
     Apple Silicon threadgroup memory limit: 32 KB — fits for up to rate-32. */
  threadgroup ulong tg_stream[4096]; /* 4096 * 8 = 32768 bytes max */

  /* --- Phase 1: Cooperative coalesced load from device → threadgroup --- */
  /* Use gid - lid to compute the first block index for this threadgroup.
     This is correct even for the last non-uniform threadgroup, where
     tg_id * tg_size would give the wrong answer. */
  uint first_block = gid - lid;
  uint active_blocks = min(tg_size, p.total_blocks > first_block ? p.total_blocks - first_block : 0u);
  uint total_words = active_blocks * words_per_block;

  /* Starting word index in the device bitstream for this threadgroup */
  ulong tg_word_offset = (ulong)first_block * (ulong)words_per_block;

  /* Each thread loads multiple words in a strided pattern for coalescing */
  for (uint w = lid; w < total_words; w += tg_size) {
    tg_stream[w] = stream[tg_word_offset + w];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  /* --- Phase 2: Each thread decodes its block from threadgroup memory --- */
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  uint bx = block_idx % p.bx;
  uint by = (block_idx / p.bx) % p.by;
  uint bz = block_idx / (p.bx * p.by);
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  uint z0 = bz * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  /* Offset within threadgroup memory for this thread's block (in bits) */
  uint tg_bit_offset = lid * p.maxbits;

  float fblock[64];
  zfp_decode_block_3d_float_tg(tg_stream, tg_bit_offset, p.maxbits, fblock);

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint idx = x + 4u * (y + 4u * z);
          dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = fblock[idx];
        }
  }
  else {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x)
          if (x0 + x < p.nx && y0 + y < p.ny && z0 + z < p.nz) {
            uint idx = x + 4u * (y + 4u * z);
            dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = fblock[idx];
          }
  }
}

/* ========================================================================== */
/* Int32 codec – integer path (no exponent, no quantization)                  */
/* ========================================================================== */

/* Encode a 1D block of 4 int32 values.
   Integer path: lift → int2uint → bitplane encode.
   No exponent header, no zero-block check. Full maxbits budget. */
static inline void zfp_encode_block_1d_int32(thread int* iblock, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  zfp_fwd_lift1(iblock);

  uint ublock[4];
  ublock[0] = zfp_int2uint(iblock[0]);
  ublock[1] = zfp_int2uint(iblock[1]);
  ublock[2] = zfp_int2uint(iblock[2]);
  ublock[3] = zfp_int2uint(iblock[3]);

  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  uint bits = maxbits;
  uint n = 0u;
  for (uint k = 32u; bits && k-- > 0u;) {
    ulong x = 0ul;
    x += (ulong)((ublock[0] >> k) & 1u) << 0u;
    x += (ulong)((ublock[1] >> k) & 1u) << 1u;
    x += (ulong)((ublock[2] >> k) & 1u) << 2u;
    x += (ulong)((ublock[3] >> k) & 1u) << 3u;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 4u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(3u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 3u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

/* Decode a 1D block of 4 int32 values.
   Integer path: bitplane decode → uint2int → inv_lift. No exponent. */
static inline void zfp_decode_block_1d_int32(device const ulong* stream, uint maxbits, uint block_idx, thread int* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint bits = maxbits;

  uint ublock[4] = {0u, 0u, 0u, 0u};
  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 4u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(3u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 4u;
        break;
      }
    }
    ublock[0] += (uint)(x & 1ul) << k; x >>= 1u;
    ublock[1] += (uint)(x & 1ul) << k; x >>= 1u;
    ublock[2] += (uint)(x & 1ul) << k; x >>= 1u;
    ublock[3] += (uint)(x & 1ul) << k;
  }

  out[0] = zfp_uint2int(ublock[0]);
  out[1] = zfp_uint2int(ublock[1]);
  out[2] = zfp_uint2int(ublock[2]);
  out[3] = zfp_uint2int(ublock[3]);
  zfp_inv_lift1(out);
}

/* Encode a 2D block of 16 int32 values. */
static inline void zfp_encode_block_2d_int32(thread int* ib, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  zfp_fwd_lift2_row(ib);
  zfp_fwd_lift2_col(ib);

  uint ub[16];
  for (uint i = 0; i < 16; ++i)
    ub[i] = zfp_int2uint(ib[zfp_perm2[i]]);

  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  uint bits = maxbits;
  uint n = 0u;
  for (uint k = 32u; bits && k-- > 0u;) {
    /* 2D block: only 16 bits needed; gather into uint to avoid 64-bit shifts */
    uint x_lo = 0u;
    for (uint i = 0; i < 16; ++i)
      x_lo += ((ub[i] >> k) & 1u) << i;
    ulong x = (ulong)x_lo;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 16u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(15u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 15u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

/* Decode a 2D block of 16 int32 values. */
static inline void zfp_decode_block_2d_int32(device const ulong* stream, uint maxbits, uint block_idx, thread int* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint bits = maxbits;

  uint ub[16];
  for (uint i = 0; i < 16; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 16u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(15u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 16u;
        break;
      }
    }
    /* 2D block: only low 16 bits set; use uint scatter to avoid 64-bit shifts */
    uint x_lo = (uint)(x & 0xFFFFul);
    for (uint i = 0; i < 16; ++i) {
      ub[i] += (x_lo & 1u) << k;
      x_lo >>= 1u;
    }
  }

  for (uint i = 0; i < 16; ++i)
    out[zfp_perm2[i]] = zfp_uint2int(ub[i]);

  zfp_inv_lift2_col(out);
  zfp_inv_lift2_row(out);
}

/* Encode a 3D block of 64 int32 values. */
static inline void zfp_encode_block_3d_int32(thread int* ib, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  zfp_fwd_lift3(ib);

  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = zfp_int2uint(ib[zfp_perm3[i]]);

  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  uint bits = maxbits;
  uint n = 0u;
  for (uint k = 32u; bits && k-- > 0u;) {
    /* Split-32 gather: build x from two 32-bit halves to avoid costly
       64-bit shift emulation on Apple Silicon's 32-bit ALUs */
    uint x_lo = 0u;
    uint x_hi = 0u;
    for (uint i = 0; i < 32; ++i)
      x_lo += ((ub[i] >> k) & 1u) << i;
    for (uint i = 32; i < 64; ++i)
      x_hi += ((ub[i] >> k) & 1u) << (i - 32u);
    ulong x = (ulong)x_lo | ((ulong)x_hi << 32u);
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 64u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(63u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 63u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

/* Decode a 3D block of 64 int32 values. */
static inline void zfp_decode_block_3d_int32(device const ulong* stream, uint maxbits, uint block_idx, thread int* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint bits = maxbits;

  uint ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = 0u;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 32u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    /* Split-32 scatter: use 32-bit ops to avoid costly 64-bit shift
       emulation on Apple Silicon's 32-bit ALUs */
    uint x_lo = (uint)(x & 0xFFFFFFFFul);
    uint x_hi = (uint)(x >> 32u);
    for (uint i = 0; i < 32; ++i) {
      ub[i] += (x_lo & 1u) << k;
      x_lo >>= 1u;
    }
    for (uint i = 32; i < 64; ++i) {
      ub[i] += (x_hi & 1u) << k;
      x_hi >>= 1u;
    }
  }

  for (uint i = 0; i < 64; ++i)
    out[zfp_perm3[i]] = zfp_uint2int(ub[i]);

  zfp_inv_lift3(out);
}

/* ========================================================================== */
/* Int32 kernel entry points                                                  */
/* ========================================================================== */

kernel void zfp_encode1d_int32(
  device const int* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec1dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint x = block_idx * 4u;
  long offset = (long)x * (long)p.sx;
  int iblock[4];

  if (x + 4u > p.dim) {
    uint nx = p.dim - x;
    for (uint i = 0; i < 4u; ++i)
      iblock[i] = i < nx ? src[offset + (long)i * (long)p.sx] : 0;
    if (nx <= 1u) {
      iblock[1] = iblock[0];
      iblock[2] = iblock[1];
      iblock[3] = iblock[0];
    }
    else if (nx == 2u) {
      iblock[2] = iblock[1];
      iblock[3] = iblock[0];
    }
    else if (nx == 3u) {
      iblock[3] = iblock[0];
    }
  }
  else {
    iblock[0] = src[offset + 0l * (long)p.sx];
    iblock[1] = src[offset + 1l * (long)p.sx];
    iblock[2] = src[offset + 2l * (long)p.sx];
    iblock[3] = src[offset + 3l * (long)p.sx];
  }

  zfp_encode_block_1d_int32(iblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode1d_int32(
  device const ulong* stream [[buffer(0)]],
  device int* out [[buffer(1)]],
  constant Codec1dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  int iblock[4];
  zfp_decode_block_1d_int32(stream, p.maxbits, block_idx, iblock);

  uint x = block_idx * 4u;
  long offset = (long)x * (long)p.sx;
  if (x + 4u > p.dim) {
    uint nx = p.dim - x;
    for (uint i = 0; i < nx; ++i)
      out[offset + (long)i * (long)p.sx] = iblock[i];
  }
  else {
    out[offset + 0l * (long)p.sx] = iblock[0];
    out[offset + 1l * (long)p.sx] = iblock[1];
    out[offset + 2l * (long)p.sx] = iblock[2];
    out[offset + 3l * (long)p.sx] = iblock[3];
  }
}

kernel void zfp_encode2d_int32(
  device const int* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec2dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint bx = block_idx % p.bx;
  uint by = block_idx / p.bx;
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy;

  int iblock[16];
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        iblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy];
  }
  else {
    for (uint y = 0; y < 4u; ++y) {
      for (uint x = 0; x < 4u; ++x) {
        uint ax = x0 + x;
        uint ay = y0 + y;
        if (ax < p.nx && ay < p.ny)
          iblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy];
        else
          iblock[4u * y + x] = 0;
      }
    }

    uint nx = x0 + 4u > p.nx ? p.nx - x0 : 4u;
    uint ny = y0 + 4u > p.ny ? p.ny - y0 : 4u;
    for (uint y = 0; y < 4u; ++y)
      if (y < ny)
        for (uint x = nx; x < 4u; ++x)
          iblock[4u * y + x] = iblock[4u * y + nx - 1u];
    for (uint x = 0; x < 4u; ++x)
      for (uint y = ny; y < 4u; ++y)
        iblock[4u * y + x] = iblock[4u * (ny - 1u) + x];
  }

  zfp_encode_block_2d_int32(iblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode2d_int32(
  device const ulong* stream [[buffer(0)]],
  device int* dst [[buffer(1)]],
  constant Codec2dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  uint bx = block_idx % p.bx;
  uint by = block_idx / p.bx;
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy;

  int iblock[16];
  zfp_decode_block_2d_int32(stream, p.maxbits, block_idx, iblock);

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        dst[base + (long)x * p.sx + (long)y * p.sy] = iblock[4u * y + x];
  }
  else {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        if (x0 + x < p.nx && y0 + y < p.ny)
          dst[base + (long)x * p.sx + (long)y * p.sy] = iblock[4u * y + x];
  }
}

kernel void zfp_encode3d_int32(
  device const int* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint bx = block_idx % p.bx;
  uint by = (block_idx / p.bx) % p.by;
  uint bz = block_idx / (p.bx * p.by);
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  uint z0 = bz * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  int iblock[64];
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint idx = x + 4u * (y + 4u * z);
          iblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz];
        }
  }
  else {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint ax = x0 + x;
          uint ay = y0 + y;
          uint az = z0 + z;
          uint idx = x + 4u * (y + 4u * z);
          if (ax < p.nx && ay < p.ny && az < p.nz)
            iblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz];
          else
            iblock[idx] = 0;
        }
  }

  zfp_encode_block_3d_int32(iblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode3d_int32(
  device const ulong* stream [[buffer(0)]],
  device int* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  uint bx = block_idx % p.bx;
  uint by = (block_idx / p.bx) % p.by;
  uint bz = block_idx / (p.bx * p.by);
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  uint z0 = bz * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  int iblock[64];
  zfp_decode_block_3d_int32(stream, p.maxbits, block_idx, iblock);

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint idx = x + 4u * (y + 4u * z);
          dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = iblock[idx];
        }
  }
  else {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x)
          if (x0 + x < p.nx && y0 + y < p.ny && z0 + z < p.nz) {
            uint idx = x + 4u * (y + 4u * z);
            dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = iblock[idx];
          }
  }
}

/* ========================================================================== */
/* Int64 block encode/decode functions                                        */
/* ========================================================================== */

/* Encode a 1D block of 4 int64 values. */
static inline void zfp_encode_block_1d_int64(thread long* iblock, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  zfp_fwd_lift1_long(iblock);

  ulong ublock[4];
  ublock[0] = zfp_int2uint_long(iblock[0]);
  ublock[1] = zfp_int2uint_long(iblock[1]);
  ublock[2] = zfp_int2uint_long(iblock[2]);
  ublock[3] = zfp_int2uint_long(iblock[3]);

  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  uint bits = maxbits;
  uint n = 0u;
  for (uint k = 64u; bits && k-- > 0u;) {
    ulong x = 0ul;
    x += (ulong)((ublock[0] >> k) & 1ul) << 0u;
    x += (ulong)((ublock[1] >> k) & 1ul) << 1u;
    x += (ulong)((ublock[2] >> k) & 1ul) << 2u;
    x += (ulong)((ublock[3] >> k) & 1ul) << 3u;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 4u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(3u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 3u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

/* Decode a 1D block of 4 int64 values. */
static inline void zfp_decode_block_1d_int64(device const ulong* stream, uint maxbits, uint block_idx, thread long* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint bits = maxbits;

  ulong ublock[4] = {0ul, 0ul, 0ul, 0ul};
  uint n = 0u;
  uint m = 0u;
  for (uint k = 64u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 4u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(3u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 4u;
        break;
      }
    }
    ublock[0] += (ulong)(x & 1ul) << k; x >>= 1u;
    ublock[1] += (ulong)(x & 1ul) << k; x >>= 1u;
    ublock[2] += (ulong)(x & 1ul) << k; x >>= 1u;
    ublock[3] += (ulong)(x & 1ul) << k;
  }

  out[0] = zfp_uint2int_long(ublock[0]);
  out[1] = zfp_uint2int_long(ublock[1]);
  out[2] = zfp_uint2int_long(ublock[2]);
  out[3] = zfp_uint2int_long(ublock[3]);
  zfp_inv_lift1_long(out);
}

/* Encode a 2D block of 16 int64 values. */
static inline void zfp_encode_block_2d_int64(thread long* ib, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  zfp_fwd_lift2_row_long(ib);
  zfp_fwd_lift2_col_long(ib);

  ulong ub[16];
  for (uint i = 0; i < 16; ++i)
    ub[i] = zfp_int2uint_long(ib[zfp_perm2[i]]);

  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  uint bits = maxbits;
  uint n = 0u;
  for (uint k = 64u; bits && k-- > 0u;) {
    /* 2D block: only 16 bits needed; gather into uint to avoid 64-bit shifts */
    uint x_lo = 0u;
    for (uint i = 0; i < 16; ++i)
      x_lo += (uint)((ub[i] >> k) & 1ul) << i;
    ulong x = (ulong)x_lo;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 16u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(15u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 15u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

/* Decode a 2D block of 16 int64 values. */
static inline void zfp_decode_block_2d_int64(device const ulong* stream, uint maxbits, uint block_idx, thread long* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint bits = maxbits;

  ulong ub[16];
  for (uint i = 0; i < 16; ++i)
    ub[i] = 0ul;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 64u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 16u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(15u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 16u;
        break;
      }
    }
    /* 2D block: only low 16 bits set; use uint scatter to avoid 64-bit shifts.
       ub[] is ulong so accumulate with (ulong) cast for k > 31 */
    uint x_lo = (uint)(x & 0xFFFFul);
    for (uint i = 0; i < 16; ++i) {
      ub[i] += (ulong)(x_lo & 1u) << k;
      x_lo >>= 1u;
    }
  }

  for (uint i = 0; i < 16; ++i)
    out[zfp_perm2[i]] = zfp_uint2int_long(ub[i]);

  zfp_inv_lift2_col_long(out);
  zfp_inv_lift2_row_long(out);
}

/* Encode a 3D block of 64 int64 values. */
static inline void zfp_encode_block_3d_int64(thread long* ib, uint maxbits, uint block_idx, device ulong* stream, bool atomic_mode)
{
  zfp_fwd_lift3_long(ib);

  ulong ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = zfp_int2uint_long(ib[zfp_perm3[i]]);

  ZfpBlockWriter1 w = zfp_make_writer1(stream, maxbits, block_idx, atomic_mode);
  uint bits = maxbits;
  uint n = 0u;
  for (uint k = 64u; bits && k-- > 0u;) {
    /* Split-32 gather: build x from two 32-bit halves to avoid costly
       64-bit shift emulation on Apple Silicon's 32-bit ALUs.
       Note: ub[i]>>k is inherently 64-bit since ub is ulong, but we
       save by accumulating into 32-bit halves for the << i shift. */
    uint x_lo = 0u;
    uint x_hi = 0u;
    for (uint i = 0; i < 32; ++i)
      x_lo += (uint)((ub[i] >> k) & 1ul) << i;
    for (uint i = 32; i < 64; ++i)
      x_hi += (uint)((ub[i] >> k) & 1ul) << (i - 32u);
    ulong x = (ulong)x_lo | ((ulong)x_hi << 32u);
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    while (n < 64u && bits) {
      bits--;
      if (!x) {
        zfp_writer_write_bit(w, 0u);
        break;
      }
      zfp_writer_write_bit(w, 1u);
      uint z = (uint)ctz(x);
      uint inner_max = min(63u - n, bits);
      uint run = min(z, inner_max);
      if (run > 0u) {
        zfp_writer_write_bits(w, 0ul, run);
        bits -= run;
      }
      if (z <= inner_max) {
        if (z < 63u - n && bits) {
          bits--;
          zfp_writer_write_bit(w, 1u);
        }
        x >>= z + 1u;
        n += z + 1u;
      } else {
        x >>= run;
        n += run;
        x >>= 1u;
        n++;
      }
    }
  }
  zfp_writer_flush(w);
}

/* Decode a 3D block of 64 int64 values. */
static inline void zfp_decode_block_3d_int64(device const ulong* stream, uint maxbits, uint block_idx, thread long* out)
{
  ZfpBlockReader1 r = zfp_make_reader1(stream, maxbits, block_idx);
  uint bits = maxbits;

  ulong ub[64];
  for (uint i = 0; i < 64; ++i)
    ub[i] = 0ul;

  uint n = 0u;
  uint m = 0u;
  for (uint k = 64u; bits && (m = 0u, k-- > 0u);) {
    m = min(n, bits);
    bits -= m;
    ulong x = zfp_reader_read_bits(r, m);
    for (; bits && n < 64u; n++, m = n) {
      bits--;
      if (zfp_reader_read_bit(r)) {
        uint inner_max = min(63u - n, bits);
        if (inner_max > 0u) {
          ulong peek = zfp_reader_peek_bits(r, inner_max);
          uint z = peek ? (uint)ctz(peek) : inner_max;
          uint run = min(z, inner_max);
          if (run > 0u) {
            zfp_reader_skip(r, run);
            bits -= run;
            n += run;
          }
          if (z < inner_max) {
            zfp_reader_skip(r, 1u);
            bits--;
          }
        }
        x += 1ul << n;
      }
      else {
        m = 64u;
        break;
      }
    }
    /* Split-32 scatter: use 32-bit shifts on x to avoid costly 64-bit
       shift emulation on Apple Silicon's 32-bit ALUs.
       Note: ub[i] is ulong so the <<k accumulation is inherently 64-bit,
       but we still save by iterating x in 32-bit halves. */
    uint x_lo = (uint)(x & 0xFFFFFFFFul);
    uint x_hi = (uint)(x >> 32u);
    for (uint i = 0; i < 32; ++i) {
      ub[i] += (ulong)(x_lo & 1u) << k;
      x_lo >>= 1u;
    }
    for (uint i = 32; i < 64; ++i) {
      ub[i] += (ulong)(x_hi & 1u) << k;
      x_hi >>= 1u;
    }
  }

  for (uint i = 0; i < 64; ++i)
    out[zfp_perm3[i]] = zfp_uint2int_long(ub[i]);

  zfp_inv_lift3_long(out);
}

/* ========================================================================== */
/* Int64 kernel entry points                                                  */
/* ========================================================================== */

kernel void zfp_encode1d_int64(
  device const long* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec1dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint x = block_idx * 4u;
  long offset = (long)x * (long)p.sx;
  long iblock[4];

  if (x + 4u > p.dim) {
    uint nx = p.dim - x;
    for (uint i = 0; i < 4u; ++i)
      iblock[i] = i < nx ? src[offset + (long)i * (long)p.sx] : 0l;
    if (nx <= 1u) {
      iblock[1] = iblock[0];
      iblock[2] = iblock[1];
      iblock[3] = iblock[0];
    }
    else if (nx == 2u) {
      iblock[2] = iblock[1];
      iblock[3] = iblock[0];
    }
    else if (nx == 3u) {
      iblock[3] = iblock[0];
    }
  }
  else {
    iblock[0] = src[offset + 0l * (long)p.sx];
    iblock[1] = src[offset + 1l * (long)p.sx];
    iblock[2] = src[offset + 2l * (long)p.sx];
    iblock[3] = src[offset + 3l * (long)p.sx];
  }

  zfp_encode_block_1d_int64(iblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode1d_int64(
  device const ulong* stream [[buffer(0)]],
  device long* out [[buffer(1)]],
  constant Codec1dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  long iblock[4];
  zfp_decode_block_1d_int64(stream, p.maxbits, block_idx, iblock);

  uint x = block_idx * 4u;
  long offset = (long)x * (long)p.sx;
  if (x + 4u > p.dim) {
    uint nx = p.dim - x;
    for (uint i = 0; i < nx; ++i)
      out[offset + (long)i * (long)p.sx] = iblock[i];
  }
  else {
    out[offset + 0l * (long)p.sx] = iblock[0];
    out[offset + 1l * (long)p.sx] = iblock[1];
    out[offset + 2l * (long)p.sx] = iblock[2];
    out[offset + 3l * (long)p.sx] = iblock[3];
  }
}

kernel void zfp_encode2d_int64(
  device const long* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec2dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint bx = block_idx % p.bx;
  uint by = block_idx / p.bx;
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy;

  long iblock[16];
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        iblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy];
  }
  else {
    for (uint y = 0; y < 4u; ++y) {
      for (uint x = 0; x < 4u; ++x) {
        uint ax = x0 + x;
        uint ay = y0 + y;
        if (ax < p.nx && ay < p.ny)
          iblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy];
        else
          iblock[4u * y + x] = 0l;
      }
    }

    uint nx = x0 + 4u > p.nx ? p.nx - x0 : 4u;
    uint ny = y0 + 4u > p.ny ? p.ny - y0 : 4u;
    for (uint y = 0; y < 4u; ++y)
      if (y < ny)
        for (uint x = nx; x < 4u; ++x)
          iblock[4u * y + x] = iblock[4u * y + nx - 1u];
    for (uint x = 0; x < 4u; ++x)
      for (uint y = ny; y < 4u; ++y)
        iblock[4u * y + x] = iblock[4u * (ny - 1u) + x];
  }

  zfp_encode_block_2d_int64(iblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode2d_int64(
  device const ulong* stream [[buffer(0)]],
  device long* dst [[buffer(1)]],
  constant Codec2dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  uint bx = block_idx % p.bx;
  uint by = block_idx / p.bx;
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy;

  long iblock[16];
  zfp_decode_block_2d_int64(stream, p.maxbits, block_idx, iblock);

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        dst[base + (long)x * p.sx + (long)y * p.sy] = iblock[4u * y + x];
  }
  else {
    for (uint y = 0; y < 4u; ++y)
      for (uint x = 0; x < 4u; ++x)
        if (x0 + x < p.nx && y0 + y < p.ny)
          dst[base + (long)x * p.sx + (long)y * p.sy] = iblock[4u * y + x];
  }
}

kernel void zfp_encode3d_int64(
  device const long* src [[buffer(0)]],
  device ulong* stream [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;
  bool atomic_mode = (p.maxbits & 31u) != 0u;

  uint bx = block_idx % p.bx;
  uint by = (block_idx / p.bx) % p.by;
  uint bz = block_idx / (p.bx * p.by);
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  uint z0 = bz * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  long iblock[64];
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint idx = x + 4u * (y + 4u * z);
          iblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz];
        }
  }
  else {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint ax = x0 + x;
          uint ay = y0 + y;
          uint az = z0 + z;
          uint idx = x + 4u * (y + 4u * z);
          if (ax < p.nx && ay < p.ny && az < p.nz)
            iblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz];
          else
            iblock[idx] = 0l;
        }
  }

  zfp_encode_block_3d_int64(iblock, p.maxbits, block_idx, stream, atomic_mode);
}

kernel void zfp_decode3d_int64(
  device const ulong* stream [[buffer(0)]],
  device long* dst [[buffer(1)]],
  constant Codec3dParams& p [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  uint block_idx = gid;
  if (block_idx >= p.total_blocks)
    return;

  uint bx = block_idx % p.bx;
  uint by = (block_idx / p.bx) % p.by;
  uint bz = block_idx / (p.bx * p.by);
  uint x0 = bx * 4u;
  uint y0 = by * 4u;
  uint z0 = bz * 4u;
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz;

  long iblock[64];
  zfp_decode_block_3d_int64(stream, p.maxbits, block_idx, iblock);

  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x) {
          uint idx = x + 4u * (y + 4u * z);
          dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = iblock[idx];
        }
  }
  else {
    for (uint z = 0; z < 4u; ++z)
      for (uint y = 0; y < 4u; ++y)
        for (uint x = 0; x < 4u; ++x)
          if (x0 + x < p.nx && y0 + y < p.ny && z0 + z < p.nz) {
            uint idx = x + 4u * (y + 4u * z);
            dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = iblock[idx];
          }
  }
}

#define ZFP_DEFINE_RATE_KERNELS_1D(RATE) \
kernel void zfp_encode1d_float_r##RATE( \
  device const float* in [[buffer(0)]], \
  device ulong* out [[buffer(1)]], \
  constant Codec1dParams& p [[buffer(2)]], \
  uint gid [[thread_position_in_grid]]) \
{ \
  uint block_idx = gid; \
  if (block_idx >= p.total_blocks) \
    return; \
  uint i0 = block_idx * 4u; \
  long offset = (long)i0 * (long)p.sx; \
  float fblock[4] = {0.0f, 0.0f, 0.0f, 0.0f}; \
  for (uint i = 0; i < 4u; ++i) { \
    uint idx = i0 + i; \
    if (idx < p.dim) \
      fblock[i] = in[offset + (long)i * (long)p.sx]; \
    else if (idx > 0u) \
      fblock[i] = fblock[i - 1u]; \
  } \
  zfp_encode_block_1d_float(fblock, (RATE) * 4u, block_idx, out, ((((RATE) * 4u) & 31u) != 0u)); \
} \
kernel void zfp_decode1d_float_r##RATE( \
  device const ulong* in [[buffer(0)]], \
  device float* out [[buffer(1)]], \
  constant Codec1dParams& p [[buffer(2)]], \
  uint gid [[thread_position_in_grid]]) \
{ \
  uint block_idx = gid; \
  if (block_idx >= p.total_blocks) \
    return; \
  uint i0 = block_idx * 4u; \
  long offset = (long)i0 * (long)p.sx; \
  float fblock[4]; \
  zfp_decode_block_1d_float(in, (RATE) * 4u, block_idx, fblock); \
  if (i0 + 4u <= p.dim) { \
    out[offset + 0l * (long)p.sx] = fblock[0]; \
    out[offset + 1l * (long)p.sx] = fblock[1]; \
    out[offset + 2l * (long)p.sx] = fblock[2]; \
    out[offset + 3l * (long)p.sx] = fblock[3]; \
  } \
  else { \
    for (uint i = 0; i < 4u; ++i) { \
      uint idx = i0 + i; \
      if (idx < p.dim) \
        out[offset + (long)i * (long)p.sx] = fblock[i]; \
    } \
  } \
}

#define ZFP_DEFINE_RATE_KERNELS_2D(RATE) \
kernel void zfp_encode2d_float_r##RATE( \
  device const float* src [[buffer(0)]], \
  device ulong* stream [[buffer(1)]], \
  constant Codec2dParams& p [[buffer(2)]], \
  uint gid [[thread_position_in_grid]]) \
{ \
  uint block_idx = gid; \
  if (block_idx >= p.total_blocks) \
    return; \
  uint bx = block_idx % p.bx; \
  uint by = block_idx / p.bx; \
  uint x0 = bx * 4u; \
  uint y0 = by * 4u; \
  long base = (long)x0 * p.sx + (long)y0 * p.sy; \
  float fblock[16]; \
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) { \
    for (uint y = 0; y < 4u; ++y) \
      for (uint x = 0; x < 4u; ++x) \
        fblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy]; \
  } \
  else { \
    for (uint y = 0; y < 4u; ++y) { \
      for (uint x = 0; x < 4u; ++x) { \
        uint ax = x0 + x; \
        uint ay = y0 + y; \
        if (ax < p.nx && ay < p.ny) \
          fblock[4u * y + x] = src[base + (long)x * p.sx + (long)y * p.sy]; \
        else \
          fblock[4u * y + x] = 0.0f; \
      } \
    } \
    uint nx = x0 + 4u > p.nx ? p.nx - x0 : 4u; \
    uint ny = y0 + 4u > p.ny ? p.ny - y0 : 4u; \
    for (uint y = 0; y < 4u; ++y) \
      if (y < ny) \
        for (uint x = nx; x < 4u; ++x) \
          fblock[4u * y + x] = fblock[4u * y + nx - 1u]; \
    for (uint x = 0; x < 4u; ++x) \
      for (uint y = ny; y < 4u; ++y) \
        fblock[4u * y + x] = fblock[4u * (ny - 1u) + x]; \
  } \
  zfp_encode_block_2d_float(fblock, (RATE) * 16u, block_idx, stream, ((((RATE) * 16u) & 31u) != 0u)); \
} \
kernel void zfp_decode2d_float_r##RATE( \
  device const ulong* stream [[buffer(0)]], \
  device float* dst [[buffer(1)]], \
  constant Codec2dParams& p [[buffer(2)]], \
  uint gid [[thread_position_in_grid]]) \
{ \
  uint block_idx = gid; \
  if (block_idx >= p.total_blocks) \
    return; \
  uint bx = block_idx % p.bx; \
  uint by = block_idx / p.bx; \
  uint x0 = bx * 4u; \
  uint y0 = by * 4u; \
  long base = (long)x0 * p.sx + (long)y0 * p.sy; \
  float fblock[16]; \
  zfp_decode_block_2d_float(stream, (RATE) * 16u, block_idx, fblock); \
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny) { \
    for (uint y = 0; y < 4u; ++y) \
      for (uint x = 0; x < 4u; ++x) \
        dst[base + (long)x * p.sx + (long)y * p.sy] = fblock[4u * y + x]; \
  } \
  else { \
    for (uint y = 0; y < 4u; ++y) \
      for (uint x = 0; x < 4u; ++x) \
        if (x0 + x < p.nx && y0 + y < p.ny) \
          dst[base + (long)x * p.sx + (long)y * p.sy] = fblock[4u * y + x]; \
  } \
}

#define ZFP_DEFINE_RATE_KERNELS_3D(RATE) \
kernel void zfp_encode3d_float_r##RATE( \
  device const float* src [[buffer(0)]], \
  device ulong* stream [[buffer(1)]], \
  constant Codec3dParams& p [[buffer(2)]], \
  uint gid [[thread_position_in_grid]]) \
{ \
  uint block_idx = gid; \
  if (block_idx >= p.total_blocks) \
    return; \
  uint bx = block_idx % p.bx; \
  uint by = (block_idx / p.bx) % p.by; \
  uint bz = block_idx / (p.bx * p.by); \
  uint x0 = bx * 4u; \
  uint y0 = by * 4u; \
  uint z0 = bz * 4u; \
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz; \
  float fblock[64]; \
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) { \
    for (uint z = 0; z < 4u; ++z) \
      for (uint y = 0; y < 4u; ++y) \
        for (uint x = 0; x < 4u; ++x) { \
          uint idx = x + 4u * (y + 4u * z); \
          fblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz]; \
        } \
  } \
  else { \
    for (uint z = 0; z < 4u; ++z) \
      for (uint y = 0; y < 4u; ++y) \
        for (uint x = 0; x < 4u; ++x) { \
          uint ax = x0 + x; \
          uint ay = y0 + y; \
          uint az = z0 + z; \
          uint idx = x + 4u * (y + 4u * z); \
          if (ax < p.nx && ay < p.ny && az < p.nz) \
            fblock[idx] = src[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz]; \
          else \
            fblock[idx] = 0.0f; \
        } \
  } \
  zfp_encode_block_3d_float(fblock, (RATE) * 64u, block_idx, stream, ((((RATE) * 64u) & 31u) != 0u)); \
} \
kernel void zfp_decode3d_float_r##RATE( \
  device const ulong* stream [[buffer(0)]], \
  device float* dst [[buffer(1)]], \
  constant Codec3dParams& p [[buffer(2)]], \
  uint gid [[thread_position_in_grid]]) \
{ \
  uint block_idx = gid; \
  if (block_idx >= p.total_blocks) \
    return; \
  uint bx = block_idx % p.bx; \
  uint by = (block_idx / p.bx) % p.by; \
  uint bz = block_idx / (p.bx * p.by); \
  uint x0 = bx * 4u; \
  uint y0 = by * 4u; \
  uint z0 = bz * 4u; \
  long base = (long)x0 * p.sx + (long)y0 * p.sy + (long)z0 * p.sz; \
  float fblock[64]; \
  zfp_decode_block_3d_float(stream, (RATE) * 64u, block_idx, fblock); \
  if (x0 + 4u <= p.nx && y0 + 4u <= p.ny && z0 + 4u <= p.nz) { \
    for (uint z = 0; z < 4u; ++z) \
      for (uint y = 0; y < 4u; ++y) \
        for (uint x = 0; x < 4u; ++x) { \
          uint idx = x + 4u * (y + 4u * z); \
          dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = fblock[idx]; \
        } \
  } \
  else { \
    for (uint z = 0; z < 4u; ++z) \
      for (uint y = 0; y < 4u; ++y) \
        for (uint x = 0; x < 4u; ++x) \
          if (x0 + x < p.nx && y0 + y < p.ny && z0 + z < p.nz) { \
            uint idx = x + 4u * (y + 4u * z); \
            dst[base + (long)x * p.sx + (long)y * p.sy + (long)z * p.sz] = fblock[idx]; \
          } \
  } \
}

ZFP_DEFINE_RATE_KERNELS_1D(8)
ZFP_DEFINE_RATE_KERNELS_1D(16)
ZFP_DEFINE_RATE_KERNELS_1D(32)

ZFP_DEFINE_RATE_KERNELS_2D(8)
ZFP_DEFINE_RATE_KERNELS_2D(16)
ZFP_DEFINE_RATE_KERNELS_2D(32)

ZFP_DEFINE_RATE_KERNELS_3D(8)
ZFP_DEFINE_RATE_KERNELS_3D(16)
ZFP_DEFINE_RATE_KERNELS_3D(32)

#undef ZFP_DEFINE_RATE_KERNELS_1D
#undef ZFP_DEFINE_RATE_KERNELS_2D
#undef ZFP_DEFINE_RATE_KERNELS_3D
