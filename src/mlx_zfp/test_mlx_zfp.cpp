#include "mlx_zfp.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "mlx/ops.h"

static void expect(bool cond, const char* msg)
{
  if (!cond)
    throw std::runtime_error(msg);
}

int main()
{
  const double rate = 16.0;

  {
    mlx::core::Shape shape(1);
    shape[0] = 32;
    mlx::core::array x = mlx::core::arange(32);
    x = mlx::core::astype(x, mlx::core::float32);
    x = mlx::core::reshape(x, shape);

    mlx::core::array c = zfp_mlx::compress(x, rate);
    mlx::core::array y = zfp_mlx::decompress(c, shape, mlx::core::float32, rate);
    mlx::core::array y_like = zfp_mlx::decompress_like(c, x, rate);

    expect(c.ndim() == 1, "compressed output must be 1D");
    expect(c.dtype() == mlx::core::uint8, "compressed output dtype must be uint8");
    expect(c.nbytes() >= sizeof(uint64_t), "compressed output must include size header");
    expect(y.shape() == shape, "decompressed shape mismatch");
    expect(y.dtype() == mlx::core::float32, "decompressed dtype mismatch");
    expect(y_like.shape() == shape, "decompress_like shape mismatch");
    expect(y_like.dtype() == mlx::core::float32, "decompress_like dtype mismatch");
  }

  {
    mlx::core::Shape shape(2);
    shape[0] = 4;
    shape[1] = 8;
    mlx::core::array x = mlx::core::arange(32);
    x = mlx::core::astype(x, mlx::core::int32);
    x = mlx::core::reshape(x, shape);

    mlx::core::array c = zfp_mlx::compress(x, rate);
    mlx::core::array y = zfp_mlx::decompress(c, shape, mlx::core::int32, rate);
    expect(y.shape() == shape, "int32 decompressed shape mismatch");
    expect(y.dtype() == mlx::core::int32, "int32 decompressed dtype mismatch");
  }

  {
    mlx::core::Shape shape(3);
    shape[0] = 2;
    shape[1] = 4;
    shape[2] = 4;
    mlx::core::array x = mlx::core::arange(32);
    x = mlx::core::astype(x, mlx::core::float32);
    x = mlx::core::reshape(x, shape);

    mlx::core::array c = zfp_mlx::compress(x, rate);
    mlx::core::array y = zfp_mlx::decompress(c, shape, mlx::core::float32, rate);

    expect(y.shape() == shape, "3d decompressed shape mismatch");
    expect(y.dtype() == mlx::core::float32, "3d decompressed dtype mismatch");
  }

  {
    bool threw = false;
    mlx::core::Shape bad_shape(1);
    bad_shape[0] = 4;
    mlx::core::array bad = mlx::core::zeros(bad_shape, mlx::core::int16);
    try {
      (void)zfp_mlx::compress(bad, rate);
    }
    catch (const std::invalid_argument&) {
      threw = true;
    }
    expect(threw, "compress should reject int16 dtype");
  }

  std::cout << "testMlxZfp passed\n";
  return EXIT_SUCCESS;
}
