#include "gpu_common.h"

#include <algorithm>

extern "C" void
zfp_default_strides_3d(const zfp_field* field, ptrdiff_t stride[3])
{
  stride[0] = field->sx ? field->sx : 1;
  stride[1] = field->sy ? field->sy : (ptrdiff_t)field->nx;
  stride[2] = field->sz ? field->sz : (ptrdiff_t)(field->nx * field->ny);
}

extern "C" zfp_bool
zfp_is_contiguous_3d(const uint dims[3], const ptrdiff_t stride[3], long long int* offset)
{
  typedef long long int int64;
  int64 idims[3];
  int64 imin;
  int64 imax;
  int64 ns;
  int d;

  idims[0] = dims[0];
  idims[1] = dims[1];
  idims[2] = dims[2];

  d = 0;
  if (dims[0] != 0)
    d++;
  if (dims[1] != 0)
    d++;
  if (dims[2] != 0)
    d++;

  if (d == 3) {
    imin = std::min((int64)stride[0], (int64)0) * (idims[0] - 1) +
           std::min((int64)stride[1], (int64)0) * (idims[1] - 1) +
           std::min((int64)stride[2], (int64)0) * (idims[2] - 1);

    imax = std::max((int64)stride[0], (int64)0) * (idims[0] - 1) +
           std::max((int64)stride[1], (int64)0) * (idims[1] - 1) +
           std::max((int64)stride[2], (int64)0) * (idims[2] - 1);

    if (offset)
      *offset = imin;

    ns = idims[0] * idims[1] * idims[2];
    return (zfp_bool)(imax - imin + 1 == ns);
  }

  if (d == 2) {
    imin = std::min((int64)stride[0], (int64)0) * (idims[0] - 1) +
           std::min((int64)stride[1], (int64)0) * (idims[1] - 1);

    imax = std::max((int64)stride[0], (int64)0) * (idims[0] - 1) +
           std::max((int64)stride[1], (int64)0) * (idims[1] - 1);

    if (offset)
      *offset = imin;

    return (zfp_bool)(imax - imin + 1 == idims[0] * idims[1]);
  }

  if (offset)
    *offset = stride[0] < 0 ? stride[0] * ((int64)dims[0] - 1) : 0;

  return (zfp_bool)(std::abs((int)stride[0]) == 1);
}

extern "C" void*
zfp_offset_void(zfp_type type, void* ptr, long long int offset)
{
  if (!ptr)
    return 0;

  if (type == zfp_type_float) {
    float* data = (float*)ptr;
    return (void*)(&data[offset]);
  }
  if (type == zfp_type_double) {
    double* data = (double*)ptr;
    return (void*)(&data[offset]);
  }
  if (type == zfp_type_int32) {
    int* data = (int*)ptr;
    return (void*)(&data[offset]);
  }
  if (type == zfp_type_int64) {
    long long int* data = (long long int*)ptr;
    return (void*)(&data[offset]);
  }

  return 0;
}
