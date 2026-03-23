#ifndef ZFP_GPU_COMMON_H
#define ZFP_GPU_COMMON_H

#include <stddef.h>

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif

void zfp_default_strides_3d(const zfp_field* field, ptrdiff_t stride[3]);
zfp_bool zfp_is_contiguous_3d(const uint dims[3], const ptrdiff_t stride[3], long long int* offset);
void* zfp_offset_void(zfp_type type, void* ptr, long long int offset);

#ifdef __cplusplus
}
#endif

#endif
