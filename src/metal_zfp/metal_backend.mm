#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>

#include "../gpu_backend.h"
#include "../gpu_common.h"
#include "metal_runtime.h"

static int zfp_metal_warned = 0;

static void
zfp_metal_warn_once(const char* msg)
{
  if (!zfp_metal_warned) {
    fprintf(stderr, "zfp: %s\n", msg);
    zfp_metal_warned = 1;
  }
}

static size_t
zfp_type_size_local(zfp_type type)
{
  switch (type) {
    case zfp_type_float:  return sizeof(float);
    case zfp_type_double: return sizeof(double);
    case zfp_type_int32:  return sizeof(int32);
    case zfp_type_int64:  return sizeof(int64);
    default:              return 0;
  }
}

static size_t
zfp_elem_count_3d(const uint dims[3])
{
  size_t nx = dims[0] ? (size_t)dims[0] : 1;
  size_t ny = dims[1] ? (size_t)dims[1] : 1;
  size_t nz = dims[2] ? (size_t)dims[2] : 1;
  return nx * ny * nz;
}

static size_t
zfp_calc_stream_bytes_1d(uint dim, uint maxbits)
{
  size_t blocks = dim / 4u;
  if (dim % 4u)
    blocks++;
  size_t total_bits = blocks * (size_t)maxbits;
  size_t words = total_bits / 64u;
  if (total_bits % 64u)
    words++;
  return words * sizeof(unsigned long long);
}

static int
zfp_dim_count_3d(const uint dims[3])
{
  int d = 0;
  if (dims[0]) d++;
  if (dims[1]) d++;
  if (dims[2]) d++;
  return d ? d : 1;
}

static zfp_bool
zfp_layout_bounds_3d(const uint dims[3], const ptrdiff_t stride[3], long long int* imin, long long int* imax)
{
  long long int nx = dims[0] ? (long long int)dims[0] : 1;
  long long int ny = dims[1] ? (long long int)dims[1] : 1;
  long long int nz = dims[2] ? (long long int)dims[2] : 1;
  long long int lo = 0;
  long long int hi = 0;

  if (stride[0] < 0)
    lo += (long long int)stride[0] * (nx - 1);
  else
    hi += (long long int)stride[0] * (nx - 1);

  if (stride[1] < 0)
    lo += (long long int)stride[1] * (ny - 1);
  else
    hi += (long long int)stride[1] * (ny - 1);

  if (stride[2] < 0)
    lo += (long long int)stride[2] * (nz - 1);
  else
    hi += (long long int)stride[2] * (nz - 1);

  if (hi < lo)
    return zfp_false;

  if (imin)
    *imin = lo;
  if (imax)
    *imax = hi;
  return zfp_true;
}

static void
zfp_pack_cpu(const void* src, void* dst, const uint dims[3], const ptrdiff_t stride[3], long long int offset, size_t item_size)
{
  const unsigned char* in = (const unsigned char*)src;
  unsigned char* out = (unsigned char*)dst;
  size_t nx = dims[0] ? (size_t)dims[0] : 1;
  size_t ny = dims[1] ? (size_t)dims[1] : 1;
  size_t nz = dims[2] ? (size_t)dims[2] : 1;
  size_t x, y, z, i = 0;

  for (z = 0; z < nz; z++)
    for (y = 0; y < ny; y++)
      for (x = 0; x < nx; x++, i++) {
        long long int sidx = (long long int)x * stride[0] +
                             (long long int)y * stride[1] +
                             (long long int)z * stride[2] -
                             offset;
        memcpy(out + i * item_size, in + (size_t)sidx * item_size, item_size);
      }
}

static void
zfp_unpack_cpu(const void* src, void* dst, const uint dims[3], const ptrdiff_t stride[3], long long int offset, size_t item_size)
{
  const unsigned char* in = (const unsigned char*)src;
  unsigned char* out = (unsigned char*)dst;
  size_t nx = dims[0] ? (size_t)dims[0] : 1;
  size_t ny = dims[1] ? (size_t)dims[1] : 1;
  size_t nz = dims[2] ? (size_t)dims[2] : 1;
  size_t x, y, z, i = 0;

  for (z = 0; z < nz; z++)
    for (y = 0; y < ny; y++)
      for (x = 0; x < nx; x++, i++) {
        long long int didx = (long long int)x * stride[0] +
                             (long long int)y * stride[1] +
                             (long long int)z * stride[2] -
                             offset;
        memcpy(out + (size_t)didx * item_size, in + i * item_size, item_size);
      }
}

static size_t
zfp_fallback_encode1d_float(const float* src, uint dim, int sx, uint maxbits, void* stream_words, size_t stream_capacity_bytes)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  size_t bytes;

  if (!src || !stream_words)
    return 0;

  bs = stream_open(stream_words, stream_capacity_bytes);
  if (!bs)
    return 0;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return 0;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 4.0, zfp_type_float, 1, zfp_false);
  field = zfp_field_1d((void*)src, zfp_type_float, dim);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
  }
  zfp_field_set_stride_1d(field, sx);

  bytes = zfp_compress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return bytes;
}

static zfp_bool
zfp_fallback_decode1d_float(const void* stream_words, size_t stream_bytes, float* dst, uint dim, int sx, uint maxbits)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  zfp_bool ok;

  if (!stream_words || !dst)
    return zfp_false;

  bs = stream_open((void*)stream_words, stream_bytes);
  if (!bs)
    return zfp_false;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return zfp_false;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 4.0, zfp_type_float, 1, zfp_false);
  field = zfp_field_1d(dst, zfp_type_float, dim);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return zfp_false;
  }
  zfp_field_set_stride_1d(field, sx);

  ok = zfp_decompress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return ok;
}

static size_t
zfp_fallback_encode2d_float(const float* src,
                           uint nx,
                           uint ny,
                           ptrdiff_t sx,
                           ptrdiff_t sy,
                           uint maxbits,
                           void* stream_words,
                           size_t stream_capacity_bytes)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  size_t bytes;

  if (!src || !stream_words)
    return 0;

  bs = stream_open(stream_words, stream_capacity_bytes);
  if (!bs)
    return 0;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return 0;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 16.0, zfp_type_float, 2, zfp_false);
  field = zfp_field_2d((void*)src, zfp_type_float, nx, ny);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
  }
  zfp_field_set_stride_2d(field, sx, sy);
  bytes = zfp_compress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return bytes;
}

static zfp_bool
zfp_fallback_decode2d_float(const void* stream_words,
                           size_t stream_bytes,
                           float* dst,
                           uint nx,
                           uint ny,
                           ptrdiff_t sx,
                           ptrdiff_t sy,
                           uint maxbits)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  zfp_bool ok;

  if (!stream_words || !dst)
    return zfp_false;

  bs = stream_open((void*)stream_words, stream_bytes);
  if (!bs)
    return zfp_false;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return zfp_false;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 16.0, zfp_type_float, 2, zfp_false);
  field = zfp_field_2d(dst, zfp_type_float, nx, ny);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return zfp_false;
  }
  zfp_field_set_stride_2d(field, sx, sy);

  ok = zfp_decompress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return ok;
}

static size_t
zfp_fallback_encode3d_float(const float* src,
                           uint nx,
                           uint ny,
                           uint nz,
                           ptrdiff_t sx,
                           ptrdiff_t sy,
                           ptrdiff_t sz,
                           uint maxbits,
                           void* stream_words,
                           size_t stream_capacity_bytes)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  size_t bytes;

  if (!src || !stream_words)
    return 0;

  bs = stream_open(stream_words, stream_capacity_bytes);
  if (!bs)
    return 0;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return 0;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 64.0, zfp_type_float, 3, zfp_false);
  field = zfp_field_3d((void*)src, zfp_type_float, nx, ny, nz);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
  }
  zfp_field_set_stride_3d(field, sx, sy, sz);
  bytes = zfp_compress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return bytes;
}

static zfp_bool
zfp_fallback_decode3d_float(const void* stream_words,
                            size_t stream_bytes,
                            float* dst,
                            uint nx,
                            uint ny,
                            uint nz,
                            ptrdiff_t sx,
                            ptrdiff_t sy,
                            ptrdiff_t sz,
                            uint maxbits)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  zfp_bool ok;

  if (!stream_words || !dst)
    return zfp_false;

  bs = stream_open((void*)stream_words, stream_bytes);
  if (!bs)
    return zfp_false;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return zfp_false;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 64.0, zfp_type_float, 3, zfp_false);
  field = zfp_field_3d(dst, zfp_type_float, nx, ny, nz);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return zfp_false;
  }
  zfp_field_set_stride_3d(field, sx, sy, sz);
  ok = zfp_decompress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return ok;
}

static size_t
zfp_fallback_encode1d_double(const double* src, uint dim, int sx, uint maxbits, void* stream_words, size_t stream_capacity_bytes)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  size_t bytes;

  if (!src || !stream_words)
    return 0;

  bs = stream_open(stream_words, stream_capacity_bytes);
  if (!bs)
    return 0;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return 0;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 4.0, zfp_type_double, 1, zfp_false);
  field = zfp_field_1d((void*)src, zfp_type_double, dim);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
  }
  zfp_field_set_stride_1d(field, sx);

  bytes = zfp_compress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return bytes;
}

static zfp_bool
zfp_fallback_decode1d_double(const void* stream_words, size_t stream_bytes, double* dst, uint dim, int sx, uint maxbits)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  zfp_bool ok;

  if (!stream_words || !dst)
    return zfp_false;

  bs = stream_open((void*)stream_words, stream_bytes);
  if (!bs)
    return zfp_false;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return zfp_false;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 4.0, zfp_type_double, 1, zfp_false);
  field = zfp_field_1d(dst, zfp_type_double, dim);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return zfp_false;
  }
  zfp_field_set_stride_1d(field, sx);

  ok = zfp_decompress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return ok;
}

static size_t
zfp_fallback_encode2d_double(const double* src,
                            uint nx,
                            uint ny,
                            ptrdiff_t sx,
                            ptrdiff_t sy,
                            uint maxbits,
                            void* stream_words,
                            size_t stream_capacity_bytes)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  size_t bytes;

  if (!src || !stream_words)
    return 0;

  bs = stream_open(stream_words, stream_capacity_bytes);
  if (!bs)
    return 0;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return 0;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 16.0, zfp_type_double, 2, zfp_false);
  field = zfp_field_2d((void*)src, zfp_type_double, nx, ny);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
  }
  zfp_field_set_stride_2d(field, sx, sy);
  bytes = zfp_compress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return bytes;
}

static zfp_bool
zfp_fallback_decode2d_double(const void* stream_words,
                            size_t stream_bytes,
                            double* dst,
                            uint nx,
                            uint ny,
                            ptrdiff_t sx,
                            ptrdiff_t sy,
                            uint maxbits)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  zfp_bool ok;

  if (!stream_words || !dst)
    return zfp_false;

  bs = stream_open((void*)stream_words, stream_bytes);
  if (!bs)
    return zfp_false;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return zfp_false;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 16.0, zfp_type_double, 2, zfp_false);
  field = zfp_field_2d(dst, zfp_type_double, nx, ny);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return zfp_false;
  }
  zfp_field_set_stride_2d(field, sx, sy);

  ok = zfp_decompress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return ok;
}

static size_t
zfp_fallback_encode3d_double(const double* src,
                            uint nx,
                            uint ny,
                            uint nz,
                            ptrdiff_t sx,
                            ptrdiff_t sy,
                            ptrdiff_t sz,
                            uint maxbits,
                            void* stream_words,
                            size_t stream_capacity_bytes)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  size_t bytes;

  if (!src || !stream_words)
    return 0;

  bs = stream_open(stream_words, stream_capacity_bytes);
  if (!bs)
    return 0;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return 0;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 64.0, zfp_type_double, 3, zfp_false);
  field = zfp_field_3d((void*)src, zfp_type_double, nx, ny, nz);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
  }
  zfp_field_set_stride_3d(field, sx, sy, sz);
  bytes = zfp_compress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return bytes;
}

static zfp_bool
zfp_fallback_decode3d_double(const void* stream_words,
                            size_t stream_bytes,
                            double* dst,
                            uint nx,
                            uint ny,
                            uint nz,
                            ptrdiff_t sx,
                            ptrdiff_t sy,
                            ptrdiff_t sz,
                            uint maxbits)
{
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* bs;
  zfp_bool ok;

  if (!stream_words || !dst)
    return zfp_false;

  bs = stream_open((void*)stream_words, stream_bytes);
  if (!bs)
    return zfp_false;

  zfp = zfp_stream_open(bs);
  if (!zfp) {
    stream_close(bs);
    return zfp_false;
  }

  zfp_stream_set_rate(zfp, (double)maxbits / 64.0, zfp_type_double, 3, zfp_false);
  field = zfp_field_3d(dst, zfp_type_double, nx, ny, nz);
  if (!field) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return zfp_false;
  }
  zfp_field_set_stride_3d(field, sx, sy, sz);
  ok = zfp_decompress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);
  return ok;
}

static size_t
zfp_compress_serial_from_buffer(zfp_stream* stream, const zfp_field* field, void* packed)
{
  zfp_field temp = *field;
  zfp_exec_policy saved_policy = zfp_stream_execution(stream);
  size_t bytes;
  temp.data = packed;
  temp.sx = temp.sy = temp.sz = temp.sw = 0;
  zfp_stream_set_execution(stream, zfp_exec_serial);
  bytes = zfp_compress(stream, &temp);
  zfp_stream_set_execution(stream, saved_policy);
  return bytes;
}

static void
zfp_decompress_serial_to_buffer(zfp_stream* stream, zfp_field* field, void* packed)
{
  zfp_field temp = *field;
  zfp_exec_policy saved_policy = zfp_stream_execution(stream);
  temp.data = packed;
  temp.sx = temp.sy = temp.sz = temp.sw = 0;
  zfp_stream_set_execution(stream, zfp_exec_serial);
  zfp_decompress(stream, &temp);
  zfp_stream_set_execution(stream, saved_policy);
}

extern "C" size_t
zfp_gpu_compress(zfp_stream* stream, const zfp_field* field)
{
  uint dims[3];
  ptrdiff_t stride[3];
  long long int offset = 0;

  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;
  zfp_default_strides_3d(field, stride);

#ifndef ZFP_WITH_METAL_NATIVE
  if (!zfp_is_contiguous_3d(dims, stride, &offset))
    return 0;

  (void)offset;
  zfp_metal_warn_once("Metal backend fallback to serial path (kernels not wired yet)");
  {
    zfp_exec_policy saved_policy = zfp_stream_execution(stream);
    size_t bytes;
    zfp_stream_set_execution(stream, zfp_exec_serial);
    bytes = zfp_compress(stream, field);
    zfp_stream_set_execution(stream, saved_policy);
    return bytes;
  }
#else
  {
    long long int imin = 0, imax = 0;
    void* base;
    void* packed;
    size_t elem_size = zfp_type_size_local(field->type);
    size_t elem_count;
    size_t span_elems;
    size_t packed_bytes;
    size_t span_bytes;
    size_t bytes;
    int dim;

    if (!elem_size)
      return 0;
    if (!zfp_is_contiguous_3d(dims, stride, &offset))
      return 0;

    if (field->type == zfp_type_float && dims[0] && !dims[1] && !dims[2] && stride[0] == 1) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t stream_bytes = zfp_calc_stream_bytes_1d(dims[0], (uint)stream->maxbits);
      size_t got = zfp_metal_encode1d_float_runtime((const float*)field->data,
                                                    dims[0],
                                                    (int)stride[0],
                                                    (uint)stream->maxbits,
                                                    stream_data(stream->stream),
                                                    stream_bytes);
      if (!got)
        got = zfp_fallback_encode1d_float((const float*)field->data, dims[0], (int)stride[0], (uint)stream->maxbits, stream_data(stream->stream), stream_bytes);
      if (got) {
        stream_wseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        stream_flush(stream->stream);
        return got;
      }
#endif
    }

    if (field->type == zfp_type_float && dims[0] && dims[1] && !dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0]) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t px = (dims[0] + 3u) & ~3u;
      size_t py = (dims[1] + 3u) & ~3u;
      size_t blocks = (px * py) / 16u;
      size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
      size_t got = zfp_metal_encode2d_float_runtime((const float*)field->data,
                                                    dims[0],
                                                    dims[1],
                                                    stride[0],
                                                    stride[1],
                                                    (uint)stream->maxbits,
                                                    stream_data(stream->stream),
                                                    stream_bytes);
      if (!got)
        got = zfp_metal_host_encode2d_float(stream,
                                            (const float*)field->data,
                                            dims[0],
                                            dims[1],
                                            stride[0],
                                            stride[1],
                                            (uint)stream->maxbits,
                                            stream_data(stream->stream),
                                            stream_bytes);
      if (!got)
        got = zfp_fallback_encode2d_float((const float*)field->data,
                                          dims[0],
                                          dims[1],
                                          stride[0],
                                          stride[1],
                                          (uint)stream->maxbits,
                                          stream_data(stream->stream),
                                          stream_bytes);
      if (got) {
        stream_wseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        stream_flush(stream->stream);
        return got;
      }
#endif
    }

    if (field->type == zfp_type_float && dims[0] && dims[1] && dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0] && stride[2] >= (ptrdiff_t)(dims[0] * dims[1])) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t px = (dims[0] + 3u) & ~3u;
      size_t py = (dims[1] + 3u) & ~3u;
      size_t pz = (dims[2] + 3u) & ~3u;
      size_t blocks = (px * py * pz) / 64u;
      size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
      size_t got = zfp_metal_encode3d_float_runtime((const float*)field->data,
                                                    dims[0],
                                                    dims[1],
                                                    dims[2],
                                                    stride[0],
                                                    stride[1],
                                                    stride[2],
                                                    (uint)stream->maxbits,
                                                    stream_data(stream->stream),
                                                    stream_bytes);
      if (!got)
        got = zfp_fallback_encode3d_float((const float*)field->data,
                                          dims[0],
                                          dims[1],
                                          dims[2],
                                          stride[0],
                                          stride[1],
                                          stride[2],
                                          (uint)stream->maxbits,
                                          stream_data(stream->stream),
                                          stream_bytes);
      if (got) {
        stream_wseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        stream_flush(stream->stream);
        return got;
      }
#endif
    }

    if (field->type == zfp_type_double && dims[0] && !dims[1] && !dims[2] && stride[0] == 1) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t stream_bytes = zfp_calc_stream_bytes_1d(dims[0], (uint)stream->maxbits);
      size_t got = zfp_metal_encode1d_double_runtime((const double*)field->data,
                                                     dims[0],
                                                     (int)stride[0],
                                                     (uint)stream->maxbits,
                                                     stream_data(stream->stream),
                                                     stream_bytes);
      if (!got)
        got = zfp_fallback_encode1d_double((const double*)field->data, dims[0], (int)stride[0], (uint)stream->maxbits, stream_data(stream->stream), stream_bytes);
      if (got) {
        stream_wseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        stream_flush(stream->stream);
        return got;
      }
#endif
    }

    if (field->type == zfp_type_double && dims[0] && dims[1] && !dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0]) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t px = (dims[0] + 3u) & ~3u;
      size_t py = (dims[1] + 3u) & ~3u;
      size_t blocks = (px * py) / 16u;
      size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
      size_t got = zfp_metal_encode2d_double_runtime((const double*)field->data,
                                                     dims[0],
                                                     dims[1],
                                                     stride[0],
                                                     stride[1],
                                                     (uint)stream->maxbits,
                                                     stream_data(stream->stream),
                                                     stream_bytes);
      if (!got)
        got = zfp_fallback_encode2d_double((const double*)field->data,
                                           dims[0],
                                           dims[1],
                                           stride[0],
                                           stride[1],
                                           (uint)stream->maxbits,
                                           stream_data(stream->stream),
                                           stream_bytes);
      if (got) {
        stream_wseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        stream_flush(stream->stream);
        return got;
      }
#endif
    }

    if (field->type == zfp_type_double && dims[0] && dims[1] && dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0] && stride[2] >= (ptrdiff_t)(dims[0] * dims[1])) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t px = (dims[0] + 3u) & ~3u;
      size_t py = (dims[1] + 3u) & ~3u;
      size_t pz = (dims[2] + 3u) & ~3u;
      size_t blocks = (px * py * pz) / 64u;
      size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
      size_t got = zfp_metal_encode3d_double_runtime((const double*)field->data,
                                                     dims[0],
                                                     dims[1],
                                                     dims[2],
                                                     stride[0],
                                                     stride[1],
                                                     stride[2],
                                                     (uint)stream->maxbits,
                                                     stream_data(stream->stream),
                                                     stream_bytes);
      if (!got)
        got = zfp_fallback_encode3d_double((const double*)field->data,
                                           dims[0],
                                           dims[1],
                                           dims[2],
                                           stride[0],
                                           stride[1],
                                           stride[2],
                                           (uint)stream->maxbits,
                                           stream_data(stream->stream),
                                           stream_bytes);
      if (got) {
        stream_wseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        stream_flush(stream->stream);
        return got;
      }
#endif
    }

    if (!(stride[0] == 1 &&
          stride[1] == (ptrdiff_t)dims[0] &&
          stride[2] == (ptrdiff_t)(dims[0] * dims[1]))) {
      zfp_exec_policy saved_policy = zfp_stream_execution(stream);
      size_t bytes;
      zfp_stream_set_execution(stream, zfp_exec_serial);
      bytes = zfp_compress(stream, field);
      zfp_stream_set_execution(stream, saved_policy);
      return bytes;
    }

    if (!zfp_layout_bounds_3d(dims, stride, &imin, &imax))
      return 0;

    elem_count = zfp_elem_count_3d(dims);
    dim = zfp_dim_count_3d(dims);
    span_elems = (size_t)(imax - imin + 1);
    packed_bytes = elem_count * elem_size;
    span_bytes = span_elems * elem_size;
    offset = imin;
    base = zfp_offset_void(field->type, field->data, offset);
    if (!base)
      return 0;

    packed = malloc(packed_bytes);
    if (!packed)
      return 0;

    if (!zfp_metal_encode_contiguous_runtime(base, span_bytes, packed, packed_bytes, elem_size, dim) &&
        !zfp_metal_pack_runtime(base, span_bytes, packed, dims, stride, offset, elem_size)) {
      zfp_metal_warn_once("native Metal pack failed; using CPU packing fallback");
      zfp_pack_cpu(base, packed, dims, stride, offset, elem_size);
    }

    bytes = zfp_compress_serial_from_buffer(stream, field, packed);
    free(packed);
    return bytes;
  }
#endif
}

extern "C" void
zfp_gpu_decompress(zfp_stream* stream, zfp_field* field)
{
  uint dims[3];
  ptrdiff_t stride[3];
  long long int offset = 0;

  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;
  zfp_default_strides_3d(field, stride);

#ifndef ZFP_WITH_METAL_NATIVE
  if (!zfp_is_contiguous_3d(dims, stride, &offset))
    return;

  (void)offset;
  zfp_metal_warn_once("Metal backend fallback to serial path (kernels not wired yet)");
  {
    zfp_exec_policy saved_policy = zfp_stream_execution(stream);
    zfp_stream_set_execution(stream, zfp_exec_serial);
    zfp_decompress(stream, field);
    zfp_stream_set_execution(stream, saved_policy);
  }
#else
  {
    long long int imin = 0, imax = 0;
    void* base;
    void* packed;
    size_t elem_size = zfp_type_size_local(field->type);
    size_t elem_count;
    size_t span_elems;
    size_t packed_bytes;
    size_t span_bytes;
    int dim;

    if (!elem_size)
      return;
    if (!zfp_is_contiguous_3d(dims, stride, &offset))
      return;

    if (field->type == zfp_type_float && dims[0] && !dims[1] && !dims[2] && stride[0] == 1) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t got = zfp_metal_decode1d_float_runtime(stream_data(stream->stream),
                                                    dims[0],
                                                    (int)stride[0],
                                                    (uint)stream->maxbits,
                                                    (float*)field->data);
      if (!got) {
        size_t stream_bytes = zfp_calc_stream_bytes_1d(dims[0], (uint)stream->maxbits);
        if (zfp_fallback_decode1d_float(stream_data(stream->stream), stream_bytes, (float*)field->data, dims[0], (int)stride[0], (uint)stream->maxbits))
          got = stream_bytes;
      }
      if (got) {
        stream_rseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        return;
      }
#endif
    }

    if (field->type == zfp_type_float && dims[0] && dims[1] && !dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0]) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t got = zfp_metal_decode2d_float_runtime(stream_data(stream->stream),
                                                    dims[0],
                                                    dims[1],
                                                    stride[0],
                                                    stride[1],
                                                    (uint)stream->maxbits,
                                                    (float*)field->data);
      if (!got) {
        size_t px = (dims[0] + 3u) & ~3u;
        size_t py = (dims[1] + 3u) & ~3u;
        size_t blocks = (px * py) / 16u;
        size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
        if (zfp_metal_host_decode2d_float(stream,
                                          stream_data(stream->stream),
                                          stream_bytes,
                                          (float*)field->data,
                                          dims[0],
                                          dims[1],
                                          stride[0],
                                          stride[1],
                                          (uint)stream->maxbits))
          got = stream_bytes;
      }
      if (!got) {
        size_t px = (dims[0] + 3u) & ~3u;
        size_t py = (dims[1] + 3u) & ~3u;
        size_t blocks = (px * py) / 16u;
        size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
        if (zfp_fallback_decode2d_float(stream_data(stream->stream),
                                        stream_bytes,
                                        (float*)field->data,
                                        dims[0],
                                        dims[1],
                                        stride[0],
                                        stride[1],
                                        (uint)stream->maxbits))
          got = stream_bytes;
      }
      if (got) {
        stream_rseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        return;
      }
#endif
    }

    if (field->type == zfp_type_float && dims[0] && dims[1] && dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0] && stride[2] >= (ptrdiff_t)(dims[0] * dims[1])) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t got = zfp_metal_decode3d_float_runtime(stream_data(stream->stream),
                                                    dims[0],
                                                    dims[1],
                                                    dims[2],
                                                    stride[0],
                                                    stride[1],
                                                    stride[2],
                                                    (uint)stream->maxbits,
                                                    (float*)field->data);
      if (!got) {
        size_t px = (dims[0] + 3u) & ~3u;
        size_t py = (dims[1] + 3u) & ~3u;
        size_t pz = (dims[2] + 3u) & ~3u;
        size_t blocks = (px * py * pz) / 64u;
        size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
        if (zfp_fallback_decode3d_float(stream_data(stream->stream),
                                        stream_bytes,
                                        (float*)field->data,
                                        dims[0],
                                        dims[1],
                                        dims[2],
                                        stride[0],
                                        stride[1],
                                        stride[2],
                                        (uint)stream->maxbits))
          got = stream_bytes;
      }
      if (got) {
        stream_rseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        return;
      }
#endif
    }

    if (field->type == zfp_type_double && dims[0] && !dims[1] && !dims[2] && stride[0] == 1) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t got = zfp_metal_decode1d_double_runtime(stream_data(stream->stream),
                                                     dims[0],
                                                     (int)stride[0],
                                                     (uint)stream->maxbits,
                                                     (double*)field->data);
      if (!got) {
        size_t stream_bytes = zfp_calc_stream_bytes_1d(dims[0], (uint)stream->maxbits);
        if (zfp_fallback_decode1d_double(stream_data(stream->stream), stream_bytes, (double*)field->data, dims[0], (int)stride[0], (uint)stream->maxbits))
          got = stream_bytes;
      }
      if (got) {
        stream_rseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        return;
      }
#endif
    }

    if (field->type == zfp_type_double && dims[0] && dims[1] && !dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0]) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t got = zfp_metal_decode2d_double_runtime(stream_data(stream->stream),
                                                     dims[0],
                                                     dims[1],
                                                     stride[0],
                                                     stride[1],
                                                     (uint)stream->maxbits,
                                                     (double*)field->data);
      if (!got) {
        size_t px = (dims[0] + 3u) & ~3u;
        size_t py = (dims[1] + 3u) & ~3u;
        size_t blocks = (px * py) / 16u;
        size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
        if (zfp_fallback_decode2d_double(stream_data(stream->stream),
                                         stream_bytes,
                                         (double*)field->data,
                                         dims[0],
                                         dims[1],
                                         stride[0],
                                         stride[1],
                                         (uint)stream->maxbits))
          got = stream_bytes;
      }
      if (got) {
        stream_rseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        return;
      }
#endif
    }

    if (field->type == zfp_type_double && dims[0] && dims[1] && dims[2] &&
        stride[0] == 1 && stride[1] >= (ptrdiff_t)dims[0] && stride[2] >= (ptrdiff_t)(dims[0] * dims[1])) {
#ifdef ZFP_WITH_METAL_CODEC_EXPERIMENTAL
      size_t got = zfp_metal_decode3d_double_runtime(stream_data(stream->stream),
                                                     dims[0],
                                                     dims[1],
                                                     dims[2],
                                                     stride[0],
                                                     stride[1],
                                                     stride[2],
                                                     (uint)stream->maxbits,
                                                     (double*)field->data);
      if (!got) {
        size_t px = (dims[0] + 3u) & ~3u;
        size_t py = (dims[1] + 3u) & ~3u;
        size_t pz = (dims[2] + 3u) & ~3u;
        size_t blocks = (px * py * pz) / 64u;
        size_t stream_bytes = (blocks * (size_t)stream->maxbits + 7u) / 8u;
        if (zfp_fallback_decode3d_double(stream_data(stream->stream),
                                         stream_bytes,
                                         (double*)field->data,
                                         dims[0],
                                         dims[1],
                                         dims[2],
                                         stride[0],
                                         stride[1],
                                         stride[2],
                                         (uint)stream->maxbits))
          got = stream_bytes;
      }
      if (got) {
        stream_rseek(stream->stream, (bitstream_offset)(got * CHAR_BIT));
        return;
      }
#endif
    }

    if (!(stride[0] == 1 &&
          stride[1] == (ptrdiff_t)dims[0] &&
          stride[2] == (ptrdiff_t)(dims[0] * dims[1]))) {
      zfp_exec_policy saved_policy = zfp_stream_execution(stream);
      zfp_stream_set_execution(stream, zfp_exec_serial);
      zfp_decompress(stream, field);
      zfp_stream_set_execution(stream, saved_policy);
      return;
    }

    if (!zfp_layout_bounds_3d(dims, stride, &imin, &imax))
      return;

    elem_count = zfp_elem_count_3d(dims);
    dim = zfp_dim_count_3d(dims);
    span_elems = (size_t)(imax - imin + 1);
    packed_bytes = elem_count * elem_size;
    span_bytes = span_elems * elem_size;
    offset = imin;
    base = zfp_offset_void(field->type, field->data, offset);
    if (!base)
      return;

    packed = malloc(packed_bytes);
    if (!packed)
      return;

    zfp_decompress_serial_to_buffer(stream, field, packed);

    if (!zfp_metal_decode_contiguous_runtime(packed, packed_bytes, base, span_bytes, elem_size, dim) &&
        !zfp_metal_unpack_runtime(packed, packed_bytes, base, span_bytes, dims, stride, offset, elem_size)) {
      zfp_metal_warn_once("native Metal unpack failed; using CPU unpack fallback");
      zfp_unpack_cpu(packed, base, dims, stride, offset, elem_size);
    }

    free(packed);
  }
#endif
}
