#ifndef ZFP_MLX_ZFP_H
#define ZFP_MLX_ZFP_H

#include "mlx/array.h"
#include "mlx/dtype.h"

#include <vector>

namespace zfp_mlx {

mlx::core::array compress(const mlx::core::array& input, double rate);

mlx::core::array decompress(
  const mlx::core::array& compressed,
  const mlx::core::Shape& shape,
  const mlx::core::Dtype& dtype,
  double rate);

mlx::core::array decompress_like(
  const mlx::core::array& compressed,
  const mlx::core::array& like,
  double rate);

} // namespace zfp_mlx

#endif
