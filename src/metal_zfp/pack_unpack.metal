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

struct ZfpBlockWriter1 {
  device atomic_uint* stream_atomic;
  device uint* stream_plain;
  ulong start_bit;
  uint current_bit;
  uint maxbits;
  bool atomic_mode;
  bool cache_valid;
  uint cache_wi;
  uint cache_val;
};

static inline ZfpBlockWriter1 zfp_make_writer1(device ulong* stream, uint maxbits, uint block_idx, bool atomic_mode)
{
  ZfpBlockWriter1 w;
  ulong bit0 = (ulong)block_idx * (ulong)maxbits;
  w.stream_atomic = reinterpret_cast<device atomic_uint*>(stream);
  w.stream_plain = reinterpret_cast<device uint*>(stream);
  w.start_bit = bit0;
  w.current_bit = 0;
  w.maxbits = maxbits;
  w.atomic_mode = atomic_mode;
  w.cache_valid = false;
  w.cache_wi = 0u;
  w.cache_val = 0u;
  return w;
}

static inline void zfp_writer_flush(thread ZfpBlockWriter1& w)
{
  if (!w.atomic_mode && w.cache_valid) {
    w.stream_plain[w.cache_wi] = w.stream_plain[w.cache_wi] | w.cache_val;
    w.cache_valid = false;
    w.cache_val = 0u;
  }
}

static inline ulong zfp_writer_write_bits(thread ZfpBlockWriter1& w, ulong bits, uint nbits)
{
  ulong keep = bits & ((nbits == 64u) ? ~0ul : ((1ul << nbits) - 1ul));
  uint remaining = nbits;
  ulong pos = w.start_bit + (ulong)w.current_bit;
  while (remaining) {
    uint wi = (uint)(pos >> 5u);
    uint bo = (uint)(pos & 31u);
    uint chunk = min(remaining, 32u - bo);
    ulong cmask = (chunk == 64u) ? ~0ul : ((chunk == 32u) ? 0xfffffffful : ((1ul << chunk) - 1ul));
    uint part = (uint)((keep & cmask) << bo);
    if (w.atomic_mode)
      atomic_fetch_or_explicit(&(w.stream_atomic[wi]), part, memory_order_relaxed);
    else {
      if (!w.cache_valid || w.cache_wi != wi) {
        zfp_writer_flush(w);
        w.cache_valid = true;
        w.cache_wi = wi;
        w.cache_val = 0u;
      }
      w.cache_val |= part;
    }
    keep >>= chunk;
    remaining -= chunk;
    pos += chunk;
  }
  w.current_bit += nbits;
  return bits >> nbits;
}

static inline uint zfp_writer_write_bit(thread ZfpBlockWriter1& w, uint bit)
{
  zfp_writer_write_bits(w, (ulong)(bit & 1u), 1u);
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
    for (; n < 4u && bits && (bits--, zfp_writer_write_bit(w, x ? 1u : 0u)); x >>= 1u, n++) {
      for (; n < 3u && bits && (bits--, !zfp_writer_write_bit(w, (uint)(x & 1ul))); x >>= 1u, n++) {
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
        for (; bits && n < 3u; n++) {
          bits--;
          if (zfp_reader_read_bit(r))
            break;
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
    ulong x = 0ul;
    for (uint i = 0; i < 16; ++i)
      x += (ulong)((ub[i] >> k) & 1u) << i;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    for (; n < 16u && bits && (bits--, zfp_writer_write_bit(w, x ? 1u : 0u)); x >>= 1u, n++) {
      for (; n < 15u && bits && (bits--, !zfp_writer_write_bit(w, (uint)(x & 1ul))); x >>= 1u, n++) {
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
        for (; bits && n < 15u; n++) {
          bits--;
          if (zfp_reader_read_bit(r))
            break;
        }
        x += 1ul << n;
      }
      else {
        m = 16u;
        break;
      }
    }
    for (uint i = 0; i < 16; ++i) {
      ub[i] += (uint)(x & 1ul) << k;
      x >>= 1u;
    }
  }

  int ib[16];
  for (uint i = 0; i < 16; ++i)
    ib[zfp_perm2[i]] = zfp_uint2int(ub[i]);

  zfp_inv_lift2_col(ib);
  zfp_inv_lift2_row(ib);

  float inv_w = ldexp(1.0f, emax - 30);
  for (uint i = 0; i < 16; ++i)
    out[i] = inv_w * (float)ib[i];
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
    ulong x = 0ul;
    for (uint i = 0; i < 64; ++i)
      x += (ulong)((ub[i] >> k) & 1u) << i;
    uint m = min(n, bits);
    bits -= m;
    x = zfp_writer_write_bits(w, x, m);
    for (; n < 64u && bits && (bits--, zfp_writer_write_bit(w, x ? 1u : 0u)); x >>= 1u, n++) {
      for (; n < 63u && bits && (bits--, !zfp_writer_write_bit(w, (uint)(x & 1ul))); x >>= 1u, n++) {
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
        for (; bits && n < 63u; n++) {
          bits--;
          if (zfp_reader_read_bit(r))
            break;
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

  int ib[64];
  for (uint i = 0; i < 64; ++i)
    ib[zfp_perm3[i]] = zfp_uint2int(ub[i]);

  zfp_inv_lift3(ib);

  float inv_w = ldexp(1.0f, emax - 30);
  for (uint i = 0; i < 64; ++i)
    out[i] = inv_w * (float)ib[i];
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
  zfp_encode_block_1d_float(fblock, RATE, block_idx, out, ((RATE & 31u) != 0u)); \
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
  zfp_decode_block_1d_float(in, RATE, block_idx, fblock); \
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
  zfp_encode_block_2d_float(fblock, RATE, block_idx, stream, ((RATE & 31u) != 0u)); \
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
  zfp_decode_block_2d_float(stream, RATE, block_idx, fblock); \
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
  zfp_encode_block_3d_float(fblock, RATE, block_idx, stream, ((RATE & 31u) != 0u)); \
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
  zfp_decode_block_3d_float(stream, RATE, block_idx, fblock); \
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
