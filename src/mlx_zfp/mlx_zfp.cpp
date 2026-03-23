#include "mlx_zfp.h"

#include <stdexcept>

#include "mlx/ops.h"
#include "zfp.h"

namespace zfp_mlx {

using mlx::core::array;

static zfp_type
to_zfp_type(const mlx::core::Dtype& dtype)
{
  if (dtype == mlx::core::float32)
    return zfp_type_float;
  if (dtype == mlx::core::float64)
    return zfp_type_double;
  if (dtype == mlx::core::int32)
    return zfp_type_int32;
  if (dtype == mlx::core::int64)
    return zfp_type_int64;

  throw std::invalid_argument("zfp_mlx: supported dtypes are float32, float64, int32, int64");
}

static void
set_field_size(zfp_field* field, const mlx::core::Shape& shape)
{
  if (shape.size() == 1)
    zfp_field_set_size_1d(field, (size_t)shape[0]);
  else if (shape.size() == 2)
    zfp_field_set_size_2d(field, (size_t)shape[1], (size_t)shape[0]);
  else if (shape.size() == 3)
    zfp_field_set_size_3d(field, (size_t)shape[2], (size_t)shape[1], (size_t)shape[0]);
  else
    throw std::invalid_argument("zfp_mlx: only 1D/2D/3D arrays are supported");
}

static void
set_field_type_and_size(zfp_field* field, const mlx::core::Shape& shape, const mlx::core::Dtype& dtype)
{
  zfp_field_set_type(field, to_zfp_type(dtype));
  set_field_size(field, shape);
}

static mlx::core::Dtype
validate_supported_dtype(const mlx::core::Dtype& dtype)
{
  (void)to_zfp_type(dtype);
  return dtype;
}

array compress(const array& input, double rate)
{
  if (rate <= 0)
    throw std::invalid_argument("zfp_mlx.compress: rate must be > 0");

  if (input.ndim() < 1 || input.ndim() > 3)
    throw std::invalid_argument("zfp_mlx.compress: only 1D/2D/3D arrays are supported");

  const mlx::core::Dtype dtype = validate_supported_dtype(input.dtype());
  const zfp_type ztype = to_zfp_type(dtype);
  const uint dims = (uint)input.ndim();

  array contiguous = mlx::core::contiguous(input);
  contiguous.eval();

  zfp_field* field = zfp_field_alloc();
  zfp_stream* zfp = zfp_stream_open(0);
  bitstream* bs = 0;

  if (!field || !zfp) {
    if (field)
      zfp_field_free(field);
    if (zfp)
      zfp_stream_close(zfp);
    throw std::runtime_error("zfp_mlx.compress: failed to allocate zfp structures");
  }

  try {
    zfp_field_set_pointer(field, contiguous.data<void>());
    set_field_type_and_size(field, contiguous.shape(), dtype);

    if (!zfp_stream_set_rate(zfp, rate, ztype, dims, 0))
      throw std::runtime_error("zfp_mlx.compress: failed to set fixed-rate mode");

    const size_t payload_max = zfp_stream_maximum_size(zfp, field);
    mlx::core::Shape out_shape(1);
    out_shape[0] = (mlx::core::ShapeElem)(payload_max + sizeof(uint64));
    array out = mlx::core::zeros(out_shape, mlx::core::uint8);
    out.eval();

    uint8* out_ptr = out.data<uint8>();
    bs = stream_open(out_ptr + sizeof(uint64), payload_max);
    if (!bs)
      throw std::runtime_error("zfp_mlx.compress: failed to open bitstream");

    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);

    const size_t payload_size = zfp_compress(zfp, field);
    if (!payload_size)
      throw std::runtime_error("zfp_mlx.compress: compression failed");

    *((uint64*)out_ptr) = (uint64)payload_size;

    stream_close(bs);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    return out;
  }
  catch (...) {
    if (bs)
      stream_close(bs);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    throw;
  }
}

array decompress(
    const array& compressed,
    const mlx::core::Shape& shape,
    const mlx::core::Dtype& dtype,
    double rate)
{
  if (rate <= 0)
    throw std::invalid_argument("zfp_mlx.decompress: rate must be > 0");

  if (shape.empty() || shape.size() > 3)
    throw std::invalid_argument("zfp_mlx.decompress: shape must be 1D/2D/3D");

  if (compressed.ndim() != 1)
    throw std::invalid_argument("zfp_mlx.decompress: compressed input must be a 1D uint8 array");

  if (compressed.dtype() != mlx::core::uint8)
    throw std::invalid_argument("zfp_mlx.decompress: compressed input dtype must be uint8");

  const zfp_type ztype = to_zfp_type(validate_supported_dtype(dtype));
  const uint dims = (uint)shape.size();

  array output = mlx::core::zeros(shape, dtype);
  array compressed_eval = compressed;
  output.eval();
  compressed_eval.eval();

  const uint8* in_ptr = compressed_eval.data<uint8>();
  const size_t total_bytes = compressed_eval.nbytes();
  if (total_bytes < sizeof(uint64))
    throw std::invalid_argument("zfp_mlx.decompress: compressed input missing size header");

  const uint64 payload_size = *((const uint64*)in_ptr);
  if (payload_size > total_bytes - sizeof(uint64))
    throw std::invalid_argument("zfp_mlx.decompress: invalid payload size header");

  zfp_field* field = zfp_field_alloc();
  zfp_stream* zfp = zfp_stream_open(0);
  bitstream* bs = 0;

  if (!field || !zfp) {
    if (field)
      zfp_field_free(field);
    if (zfp)
      zfp_stream_close(zfp);
    throw std::runtime_error("zfp_mlx.decompress: failed to allocate zfp structures");
  }

  try {
    zfp_field_set_pointer(field, output.data<void>());
    set_field_type_and_size(field, shape, dtype);

    if (!zfp_stream_set_rate(zfp, rate, ztype, dims, 0))
      throw std::runtime_error("zfp_mlx.decompress: failed to set fixed-rate mode");

    bs = stream_open((void*)(in_ptr + sizeof(uint64)), (size_t)payload_size);
    if (!bs)
      throw std::runtime_error("zfp_mlx.decompress: failed to open bitstream");

    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);

    const size_t decoded_size = zfp_decompress(zfp, field);
    if (!decoded_size)
      throw std::runtime_error("zfp_mlx.decompress: decompression failed");

    stream_close(bs);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    return output;
  }
  catch (...) {
    if (bs)
      stream_close(bs);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    throw;
  }
}

array decompress_like(const array& compressed, const array& like, double rate)
{
  return decompress(compressed, like.shape(), like.dtype(), rate);
}

} // namespace zfp_mlx
